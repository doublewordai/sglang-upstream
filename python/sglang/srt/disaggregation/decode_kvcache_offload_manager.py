from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    build_kv_host_pool,
)
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.runtime_context import get_schedule
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class DecodeKVCacheOffloadManager:
    """Manage decode-side KV cache offloading lifecycle and operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        tree_cache: BasePrefixCache,
        server_args: ServerArgs,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = get_schedule().page_size
        self.request_counter = 0
        self.tree_cache = tree_cache
        env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
        if env_stride is None or env_stride <= 0:
            self.offload_stride = self.page_size
        else:
            self.offload_stride = max(
                self.page_size, (env_stride // self.page_size) * self.page_size
            )
        # Blob-style backends group pages by absolute chain position; offload
        # chunks must end on group boundaries so every group is written whole
        # by a single op (a group split across ops loses all pages after its
        # first, by the head-group rule). Group-aligned chunking below.
        self.offload_group_tokens = 0
        if server_args.hicache_storage_backend == "blob":
            try:
                cfg = json.loads(
                    server_args.hicache_storage_backend_extra_config or "{}"
                )
                blob_pages = int(cfg.get("blob_pages", 16))
            except Exception:
                blob_pages = 16
            self.offload_group_tokens = max(
                self.page_size, blob_pages * self.page_size
            )
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        if not isinstance(kv_cache, (MHATokenToKVPool, MLATokenToKVPool)):
            raise ValueError("Unsupported KV cache type for decode offload")

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        hicache_storage_backend_extra_config = {}
        if server_args.hicache_storage_backend_extra_config:
            try:
                hicache_storage_backend_extra_config = json.loads(
                    server_args.hicache_storage_backend_extra_config
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid hicache storage backend extra config JSON: {e}"
                )

        # DSA models: build the full hybrid stack (latent anchor + index-K
        # sidecar + HybridCacheController) so the decode offload ALSO writes
        # indexer pages. Without this, the offload writes latent-only objects
        # and a prefill arm's storage hit truncates at |P| (R never usable).
        self.is_dsa = (
            isinstance(kv_cache, DSATokenToKVPool)
            and bool(kv_cache.index_k_with_scale_buffer)
            and server_args.hicache_storage_backend is not None
        )
        self.indexer_host_pool = None
        if self.is_dsa:
            from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
                _get_allocator_type,
                build_anchor_sidecar_stack,
            )
            from sglang.srt.mem_cache.memory_pool_host import DSAIndexerPoolHost

            params = CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                page_size=self.page_size,
                tp_cache_group=tp_group,
            )
            full_layer_mapping = {i: i for i in range(kv_cache.layer_num)}
            (
                self.decode_host_mem_pool,
                self.cache_controller,
            ) = build_anchor_sidecar_stack(
                params=params,
                server_args=server_args,
                kv_pool=kv_cache,
                sidecar_pool_name=PoolName.INDEXER,
                full_layer_mapping=full_layer_mapping,
                load_cache_event=threading.Event(),
                storage_backend=server_args.hicache_storage_backend,
                use_mla=isinstance(kv_cache, MLATokenToKVPool),
                override_kv_cache_dim=kv_cache.kv_cache_dim,
                sidecar_host_pool_factory=lambda kv_host_pool: DSAIndexerPoolHost(
                    kv_cache,
                    kv_host_pool,
                    server_args.hicache_mem_layout,
                    allocator_type=_get_allocator_type(server_args),
                ),
                prefetch_threshold=self.page_size,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=hicache_storage_backend_extra_config,
            )
            self.indexer_host_pool = self.decode_host_mem_pool.get_pool(
                PoolName.INDEXER
            )
        else:
            self.decode_host_mem_pool = build_kv_host_pool(
                kv_pool=kv_cache,
                page_size=self.page_size,
                server_args=server_args,
                use_mla=isinstance(kv_cache, MLATokenToKVPool),
            )
            self.cache_controller = HiCacheController(
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                mem_pool_host=self.decode_host_mem_pool,
                page_size=self.page_size,
                tp_group=tp_group,
                io_backend=server_args.hicache_io_backend,
                load_cache_event=threading.Event(),
                storage_backend=server_args.hicache_storage_backend,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=hicache_storage_backend_extra_config,
            )

        self.ongoing_offload = {}
        self.ongoing_backup = {}
        self.offloaded_state = {}
        self.offload_inflight = {}
        # rid -> full page-key chain of the context so far (prefill + offloaded
        # incremental pages). Blob-style storage backends need the absolute
        # chain position to group pages correctly.
        self.offload_page_keys: dict = {}
        logger.info("Enable offload kv cache for decode side")

    def release_host_resources(self) -> None:
        self.decode_host_mem_pool.destroy()

    def _mark_offload_started(self, rid):
        self.offload_inflight[rid] = self.offload_inflight.get(rid, 0) + 1

    def _mark_offload_finished(self, rid):
        count = self.offload_inflight.get(rid, 0)
        if count <= 1:
            self.offload_inflight.pop(rid, None)
        else:
            self.offload_inflight[rid] = count - 1

    def _has_inflight_offload(self, rid):
        return self.offload_inflight.get(rid, 0) > 0

    def offload_kv_cache(self, req) -> bool:
        """Offload incremental KV cache for decode side."""

        if self.cache_controller is None or self.decode_host_mem_pool is None:
            return False

        if req.req_pool_idx == -1 or len(req.output_ids) == 0:
            return False

        token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            return False

        # Prefill side offloads page-aligned origin_input_ids, decode side offloads the incremental part
        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        prefill_offloaded_len = (
            len(req.origin_input_ids) // self.page_size * self.page_size
        )
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_hashes = self._compute_prefix_hash(
                req.origin_input_ids[:prefill_offloaded_len]
            )
            last_prefill_hash = (
                prefill_hashes[-1] if prefill_offloaded_len > 0 else None
            )
            self.offload_page_keys[req.rid] = list(prefill_hashes)
            state = OffloadedState(
                prefill_len=prefill_offloaded_len,
                inc_len=0,
                last_hash=last_prefill_hash,
            )
            self.offloaded_state[req.rid] = state
        incremental_total = len(all_tokens) - state.prefill_len
        incremental_new = incremental_total - state.inc_len
        incremental_aligned_len = (
            incremental_new // self.offload_stride * self.offload_stride
        )
        incremental_aligned_len = self._group_align_incremental(
            state, incremental_aligned_len, req
        )

        if incremental_aligned_len == 0:
            return False

        # Extract incremental tokens and indices for the newly available chunk
        start = state.prefill_len + state.inc_len
        end = start + incremental_aligned_len
        incremental_tokens = all_tokens[start:end]
        incremental_indices = token_indices[start:end]

        # Prefill-aligned GPU slots are freed at request finish in
        # _release_finished_req, NOT here. The decoding request
        # continues to attend to those slots via req_to_token; freeing
        # them mid-decode races with concurrent admission, which can
        # reuse the slots and produce cross-pollinated KV reads.

        # Asynchronously offload incremental KV cache from device to host
        self.request_counter += 1
        ack_id = self.request_counter
        device_indices = incremental_indices.long()
        # DSA: pre-allocate indexer host rows and pass the sidecar transfer so
        # the device->host offload copies index-K alongside latent KV.
        indexer_host_indices = None
        write_kwargs = {}
        if self.is_dsa:
            indexer_host_indices = self.indexer_host_pool.alloc(len(device_indices))
            if indexer_host_indices is None:
                logger.warning(
                    "Decode offload: indexer host pool full; latent-only offload for %s",
                    req.rid,
                )
            else:
                write_kwargs["extra_pools"] = [
                    PoolTransfer(
                        name=PoolName.INDEXER,
                        host_indices=indexer_host_indices,
                        device_indices=device_indices,
                    )
                ]
        host_indices = self.cache_controller.write(
            device_indices=device_indices,
            node_id=ack_id,
            **write_kwargs,
        )
        if host_indices is None:
            if indexer_host_indices is not None:
                self.indexer_host_pool.free(indexer_host_indices)
            logger.error(f"Not enough host memory for request {req.rid}")
            return False

        self._mark_offload_started(req.rid)
        self.ongoing_offload[ack_id] = (
            req,
            host_indices,
            incremental_tokens,
            time.time(),
            start,
            end,
            indexer_host_indices,
        )
        state.inc_len += incremental_aligned_len
        return True

    def _group_align_incremental(self, state, page_aligned_len, req) -> int:
        """Cap an offload chunk so storage groups are written whole.

        Blob backends write only groups whose FIRST page lies inside the op's
        page range; an op must therefore end on a group boundary (any start is
        fine). While decoding we wait for a chunk that spans at least one full
        group past the next boundary; at request finish we flush the remaining
        page-aligned tail (its final group is written partial-from-head).
        The sliver between a turn boundary and the next group boundary is not
        offloadable by design (bounded < group per turn, like the prefill
        side's node-boundary rule).
        """
        G = self.offload_group_tokens
        if G <= self.page_size or page_aligned_len == 0:
            return page_aligned_len
        start_abs = state.prefill_len + state.inc_len
        if start_abs % G == 0:
            whole = (page_aligned_len // G) * G
            if req.finished():
                return page_aligned_len  # flush tail (partial final group ok)
            return whole
        to_boundary = G - (start_abs % G)
        if page_aligned_len < to_boundary:
            return 0 if not req.finished() else page_aligned_len
        rest = page_aligned_len - to_boundary
        if req.finished():
            return page_aligned_len
        whole_groups = (rest // G) * G
        if whole_groups == 0:
            return 0  # wait for a full group past the boundary
        return to_boundary + whole_groups

    def check_offload_progress(self):
        """Check the progress of offload from device to host and backup from host to storage."""
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                len(cc.ack_write_queue),
                cc.ack_backup_queue.qsize(),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        n_write, n_backup = map(int, qsizes.tolist())
        self._check_offload_progress(n_write)
        self._check_backup_progress(n_backup)

    def _check_offload_progress(self, finish_count):
        """Check the progress of offload from device to host."""
        while finish_count > 0:
            ack = self.cache_controller.ack_write_queue.pop(0)
            ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                (
                    req,
                    host_indices,
                    incremental_tokens,
                    start_time,
                    start,
                    end,
                    indexer_host_indices,
                ) = self.ongoing_offload.pop(ack_id)

                self._mark_offload_finished(req.rid)
                prior_hash = (
                    self.offloaded_state[req.rid].last_hash
                    if req.rid in self.offloaded_state
                    else None
                )
                prior_page_keys = self.offload_page_keys.get(req.rid, [])
                last_hash = self._trigger_backup(
                    req,
                    host_indices,
                    incremental_tokens,
                    start_time,
                    prior_hash,
                    indexer_host_indices,
                    prior_page_keys,
                )
                if req.rid in self.offloaded_state:
                    self.offloaded_state[req.rid].last_hash = last_hash

                if req.finished() and not self._has_inflight_offload(req.rid):
                    state = self.offloaded_state.get(req.rid)
                    start_offset = state.prefill_len if state is not None else start
                    self._release_finished_req(req, start_offset)
            finish_count -= 1

    def _release_finished_req(self, req: Req, start_offset: int):
        # Defensive guard: ReqToTokenPool.free sets req_pool_idx to None,
        # so a previously-released request must be skipped here to avoid
        # non-idempotent side effects (e.g. tree_cache.protected_size_
        # double-decrement, host pool double-free).
        if req.req_pool_idx is None or req.req_pool_idx == -1:
            return

        kv_committed_len = req.effective_kv_committed_len()

        # Free the prefill-aligned slots. Previously this was done
        # eagerly in offload_kv_cache (mid-decode), which raced with
        # concurrent admission. Now consolidated here at request
        # finish, where the request is guaranteed to no longer attend
        # to those slots.
        state = self.offloaded_state.get(req.rid)
        if state is not None and state.prefill_len > 0:
            prefill_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : state.prefill_len
            ]
            self.token_to_kv_pool_allocator.free(prefill_indices)
        start = start_offset
        end = kv_committed_len
        # Free the incremental part of the request (DSA-aware)
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, start:end]
        self.token_to_kv_pool_allocator.free(kv_indices)

        # Free over-allocated KV cache slots (e.g. from speculative decoding v2).
        # Without spec v2, start_p == end_p so this is a no-op.
        start_p, end_p = kv_committed_len, req.kv.kv_allocated_len
        if self.page_size > 1:
            start_p = ceil_align(start_p, self.page_size)
        if start_p < end_p:
            overalloc_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_p:end_p
            ]
            self.token_to_kv_pool_allocator.free(overalloc_indices)

        self.req_to_token_pool.free(req)
        req.kv = None
        self.tree_cache.protected_size_ -= len(req.prefix_indices)
        if req.rid in self.offloaded_state:
            del self.offloaded_state[req.rid]
        self.offload_page_keys.pop(req.rid, None)

    def _check_backup_progress(self, finish_count):
        """Check the progress of backup from host to storage."""
        for _ in range(finish_count):
            storage_operation = self.cache_controller.ack_backup_queue.get()
            ack_id = storage_operation.id
            req_id, host_indices, start_time, indexer_host_indices = (
                self.ongoing_backup.pop(ack_id)
            )

            # Release host memory
            self.decode_host_mem_pool.free(host_indices)
            if indexer_host_indices is not None:
                self.indexer_host_pool.free(indexer_host_indices)

            logger.debug(
                f"Finished backup request {req_id}, free host memory, len:{len(host_indices)}, cost time:{time.time() - start_time:.2f} seconds."
            )

    def _trigger_backup(
        self,
        req,
        host_indices,
        incremental_tokens,
        start_time,
        prior_hash,
        indexer_host_indices=None,
        prior_page_keys=None,
    ):
        """Trigger async backup from host to storage."""
        page_hashes = self._compute_prefix_hash(incremental_tokens, prior_hash)
        extra_pools = None
        if self.is_dsa and indexer_host_indices is not None:
            extra_pools = [
                PoolTransfer(
                    name=PoolName.INDEXER,
                    host_indices=indexer_host_indices,
                    keys=page_hashes,
                )
            ]
        write_kwargs = {}
        if extra_pools is not None:
            write_kwargs["extra_pools"] = extra_pools
        if prior_page_keys:
            # absolute chain position so group-based backends address pages
            # from the context root (see blob backend)
            write_kwargs["prefix_keys"] = list(prior_page_keys)
        ack_id = self.cache_controller.write_storage(
            host_indices,
            incremental_tokens,
            hash_value=page_hashes,
            **write_kwargs,
        )
        self.ongoing_backup[ack_id] = (
            req.rid,
            host_indices,
            start_time,
            indexer_host_indices,
        )
        self.offload_page_keys[req.rid] = list(prior_page_keys or []) + list(page_hashes)
        return page_hashes[-1] if len(page_hashes) > 0 else prior_hash

    def _compute_prefix_hash(self, tokens, prior_hash=""):
        page_hashes = []
        last_hash = prior_hash
        for offset in range(0, len(tokens), self.page_size):
            page_tokens = tokens[offset : offset + self.page_size]
            last_hash = self.cache_controller.get_hash_str(page_tokens, last_hash)
            page_hashes.append(last_hash)
        return page_hashes

    def finalize_release_on_finish(self, req: Req):
        """Free any remaining tail KV that was not offloaded due to non-aligned length."""
        # ReqToTokenPool.free sets req_pool_idx to None on release, so
        # guard against both sentinels here.
        if req.req_pool_idx is None or req.req_pool_idx == -1:
            return
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_len = len(req.origin_input_ids) // self.page_size * self.page_size
            inc_len = 0
        else:
            prefill_len = state.prefill_len
            inc_len = state.inc_len
        # Prefill-aligned slots are freed by _release_finished_req. Make
        # sure state exists so it can find prefill_len.
        if state is None:
            self.offloaded_state[req.rid] = OffloadedState(
                prefill_len=prefill_len, inc_len=0, last_hash=None
            )
        if self._has_inflight_offload(req.rid):
            return
        start_offset = prefill_len
        self._release_finished_req(req, start_offset)
