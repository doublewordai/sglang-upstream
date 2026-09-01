# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Radix prefix cache over hierarchical sparse attention (hisparse).

Retains finished requests' prefixes with their MLA latent KV resident in the
pinned host (Grace) pool. The tree itself stores only logical kv indices, as
always; host-row ownership lives in the coordinator's ``logical_to_host_row``
side table, so node insertion/splitting needs no changes. Retained logical
indices keep their lightning-indexer keys valid for free (the index buffer is
device-resident at full logical size and retained indices are never reused).
Device resources (mapping rows, per-request device-buffer slots) are released
at request finish exactly as without retention; a matched prefix is
re-activated by pointing the new request's host-row table at the retained
rows — decode-time swap-in then demand-pages the latent from host as usual.

Intended for the PD-disaggregation decode server (extend-over-a-retained-
prefix never runs there: prefill happens on the prefill arm and only the
suffix KV is transferred).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.radix_cache import (
    InsertParams,
    RadixCache,
    RadixKey,
)

if TYPE_CHECKING:
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class HiSparseRadixCache(RadixCache):
    """RadixCache whose retained values' KV lives in the hisparse host pool."""

    def __init__(self, params):
        super().__init__(params)
        self.coordinator: "HiSparseCoordinator" = None

    def attach_coordinator(self, coordinator: "HiSparseCoordinator") -> None:
        self.coordinator = coordinator
        coordinator.init_retention()

    # ------------------------------------------------------------------

    def retained_prefix_len(self, prefix_indices: torch.Tensor) -> int:
        """Longest head of a matched prefix whose host rows are retained.

        A prefix inserted by a still-running request (cache_unfinished_req)
        shares logical indices with that request but has no side-table rows
        yet; adopting it would read rows that do not exist. Trim to the
        covered head instead of asserting.
        """
        if prefix_indices.numel() == 0:
            return 0
        rows = self.coordinator.logical_to_host_row[
            prefix_indices.to(device="cpu", dtype=torch.int64)
        ]
        missing = (rows < 0).nonzero()
        return int(missing[0]) if missing.numel() > 0 else prefix_indices.numel()

    def cache_finished_req(
        self, req: "Req", is_insert: bool = True, *, kv_len_to_handle: int
    ):
        coord = self.coordinator
        assert coord is not None, "attach_coordinator() must run before serving"
        idx = req.req_pool_idx

        if self.disable_finished_insert:
            is_insert = False

        if self.disable:
            coord.request_finished(req)
            kv_indices = self.req_to_token_pool.req_to_token[
                idx, req.cache_protected_len : kv_len_to_handle
            ]
            self.token_to_kv_pool_allocator.free_segment(
                kv_indices, start_pos=req.cache_protected_len
            )
            return

        # 1. Complete the host mirror; retain only what is provably mirrored.
        safe_len = coord.flush_pending_to_host(req, kv_len_to_handle)
        rows_all = coord.host_rows_snapshot(req)
        host_len = int(rows_all.numel())
        adopted = int(coord.req_adopted_len[idx])
        prot = req.cache_protected_len

        token_ids = (req.origin_input_ids + req.output_ids)[:safe_len]
        full_kv = self.req_to_token_pool.req_to_token[idx, :kv_len_to_handle]
        kv_indices = full_kv[: len(token_ids)]

        radix_key = RadixKey(
            token_ids,
            req.extra_key,
            is_bigram=self.is_eagle,
            cache_salt=req.cache_salt,
        ).page_aligned(self.page_size)
        key_len = len(radix_key)
        values = kv_indices[:key_len].to(dtype=torch.int64, copy=True)
        assert key_len <= host_len, (
            f"host mirror shorter than the insert range: {key_len=} {host_len=}"
        )

        # 2. Insert; freed_end = length already present in the tree.
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result = self.insert(
                InsertParams(key=radix_key, value=values, priority=priority)
            )
            freed_end = result.prefix_len
        else:
            freed_end = key_len

        # 3. Host-row ownership (compress_ratio == 1). For each position in
        #    [adopted, key_len): if the tree's logical index there has no
        #    side-table row yet, this request's row becomes the retained one
        #    (covers both newly inserted nodes and nodes inserted earlier by
        #    cache_unfinished_req, whose logical indices this request shares);
        #    if the table already holds a row (duplicate content), free ours.
        #    Rows past key_len are an un-inserted tail — free. Rows retained
        #    for logical indices that step 4 then frees (dedup range) are
        #    released again by the allocator's retention hook — consistent.
        if key_len > adopted:
            pos_values = full_kv[adopted:key_len].to(device="cpu", dtype=torch.int64)
            pos_rows = rows_all[adopted:key_len].to(device="cpu", dtype=torch.int64)
            vacant = coord.logical_to_host_row[pos_values] < 0
            if bool(vacant.any()):
                coord.retain_rows(pos_values[vacant], pos_rows[vacant])
            if bool((~vacant).any()):
                coord.free_unretained_rows(pos_rows[~vacant])
        if host_len > max(key_len, adopted):
            coord.free_unretained_rows(rows_all[max(key_len, adopted) :])

        # 4. Free the logical indices exactly as upstream (the allocator's
        #    retention hook releases side-table rows for tree-evicted indices;
        #    the ranges freed here carry none).
        self.token_to_kv_pool_allocator.free_segments(
            [
                (full_kv[prot:freed_end], prot),
                (full_kv[key_len:], key_len),
            ]
        )

        # 5. Release device resources; host bytes now belong to the tree.
        coord.release_for_retention(req)

        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)
