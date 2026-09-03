"""MTP target-verify equivalence for the HiSparse shared-index prefetch.

Port note: this extends the upstream plan-then-IO prefetch structure
(test/manual/kernels/test_hisparse_prefetch.py) to the speculative path: the
anchor records ONE plan per draft position (num_newest = p+1), the last
position's call issues one multi-position IO group per skip layer on the
prefetch stream, and skip layers return the anchor's stashed per-position slot
tables. Verified against the synchronous path (the pre-patch behaviour under
speculative decoding) on a real HiSparseCoordinator stack:

  1. eager multi-step: identical page tables per (layer, position), identical
     device KV bytes (all layers), identical anchor LRU/token maps, identical
     post-accept host backups; skip layers' LRU maps stay at init in the
     prefetch stack (never read — copies are IO-only) — asserted explicitly.
  2. CUDA-graph-captured verify step replayed against the eager synchronous
     reference, bit-identical (the fork/event pattern must capture, mirroring
     the upstream graph test for the decode path).

Run: python3 -m pytest test/registered/unit/managers/test_hisparse_prefetch_spec.py -x -s
"""

import os
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu

# ---------------------------------------------------------------------------
# Config: GLM-like freq-4 pattern on 8 layers (anchors 0 and 4, skips 1-3/5-7)
# ---------------------------------------------------------------------------
SIZE = 4096  # hisparse device pool tokens
PAGE_SIZE = 64
TOP_K = 256
DEVICE_BUFFER_SIZE = 4 * TOP_K  # >= num_draft_tokens * top_k (prod MTP rule)
HOST_TO_DEVICE_RATIO = 2
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
KV_CACHE_DIM = 576
LAYER_NUM = 8
MAX_NUM_REQS = 8
MAX_CONTEXT_LEN = 4096
N_POS = 4  # speculative_num_draft_tokens (chain topk=1)
PATTERN = [i % 4 != (2 if i >= 4 else 0) for i in range(LAYER_NUM)]
# -> [F, T, T, T, F, T, T, T]  (index_topk_freq=4 shape)
ANCHORS = [i for i, s in enumerate(PATTERN) if not s]
SKIPS = [i for i, s in enumerate(PATTERN) if s]
FILL_LEN = 1088  # dbs + page: > DEVICE_BUFFER_SIZE, page-aligned (host alloc)
STEPS = 4
ACCEPT_PER_STEP = 2


def _make_req(rid, origin_input_ids=None, output_ids=None):
    """Create a minimal mock Req object with the fields HiSparseCoordinator uses."""
    if origin_input_ids is None:
        origin_input_ids = list(range(64))
    req = SimpleNamespace(
        rid=rid,
        origin_input_ids=origin_input_ids,
        output_ids=output_ids or [],
        fill_ids=origin_input_ids + (output_ids or []),
        seqlen=len(origin_input_ids) + len(output_ids or []),
        req_pool_idx=None,
        kv=SimpleNamespace(kv_allocated_len=0),
        kv_committed_len=0,
        finished_reason=None,
        hisparse_staging=False,
        staging=False,
        inflight_middle_chunks=0,
    )
    req.finished = lambda: req.finished_reason is not None
    req.set_extend_range = lambda start, end: setattr(
        req, "extend_range", Range(start, end)
    )
    return req


class _Stack:
    """A minimal but real HiSparse component stack around one coordinator."""

    def __init__(self, shared_index_layers, num_draft_tokens):
        from sglang.srt.mem_cache.pool_host.common import (
            ALLOC_MEMORY_FUNCS,
            alloc_with_pin_memory,
        )

        self._original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory

        from sglang.srt.mem_cache.allocator.hisparse import (
            HiSparseTokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool

        self.device_pool = HiSparseDSATokenToKVPool(
            size=SIZE,
            page_size=PAGE_SIZE,
            kv_lora_rank=KV_LORA_RANK,
            dtype=torch.bfloat16,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=LAYER_NUM,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=KV_CACHE_DIM,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )
        self.allocator = HiSparseTokenToKVPoolAllocator(
            size=SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=self.device_pool,
            need_sort=False,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        self.req_to_token_pool = ReqToTokenPool(
            size=MAX_NUM_REQS,
            max_context_len=MAX_CONTEXT_LEN,
            device="cuda",
            enable_memory_saver=False,
        )
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        self.coordinator = HiSparseCoordinator(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            top_k=TOP_K,
            device_buffer_size=DEVICE_BUFFER_SIZE,
            device="cuda",
            tp_group=None,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
            shared_index_layers=shared_index_layers,
            num_draft_tokens=num_draft_tokens,
        )
        assert (self.coordinator.enable_prefetch) == (shared_index_layers is not None)

    def reset(self):
        self.allocator.clear()
        self.req_to_token_pool.clear()
        self.coordinator.mem_pool_host.clear()
        c = self.coordinator
        c.req_to_device_buffer.zero_()
        c.req_device_buffer_size.zero_()
        c.req_to_host_pool.fill_(-1)
        c.req_to_host_pool_allocated_len.zero_()
        c.req_device_buffer_tokens.fill_(-1)
        c.req_device_buffer_token_locs.fill_(-1)
        c.lru_slots[:] = c._lru_init.view(1, 1, -1)
        c.ack_staging_queue.clear()
        c._has_pending_backup = False


class TestHiSparseVerifyPrefetchSpec(unittest.TestCase):
    """Sync (pre-patch) vs prefetch (patched) verify paths must be identical."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for HiSparse tests.")
        if is_npu() or is_xpu():
            raise unittest.SkipTest("HiSparse tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            raise unittest.SkipTest("CUDA/ROCm not available.")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29601")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        cls.tp_group = torch.distributed.group.WORLD
        cls.sync = _Stack(None, N_POS)  # pre-patch behaviour under spec
        cls.prefetch = _Stack(PATTERN, N_POS)  # patched behaviour

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS

        ALLOC_MEMORY_FUNCS["cuda"] = cls.sync._original_alloc
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    # -- admission / helpers (mirrors test_hisparse_unit.py) ---------------
    @staticmethod
    def _kv_pattern(layer_id, token_id):
        v = (layer_id * 10000 + token_id + 1) * 0.001
        return float(torch.tensor(v, dtype=torch.bfloat16))

    def _admit(self, stack, req, fill_len):
        device = stack.allocator.device
        kv_loc = stack.allocator.alloc_logical_only(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=fill_len,
        )
        self.assertIsNotNone(kv_loc, "KV alloc failed")
        stack.req_to_token_pool.write(
            (req.req_pool_idx, slice(0, len(kv_loc))), kv_loc
        )
        req.kv.kv_allocated_len = fill_len
        req.kv_committed_len = fill_len
        req.full_untruncated_fill_ids = array("q", range(fill_len))
        req.extend_range = None

        host_pool = stack.coordinator.mem_pool_host
        host_indices = host_pool.alloc(fill_len)
        self.assertIsNotNone(host_indices, "Host alloc failed")
        host_indices = host_indices.to(device="cuda")
        stack.coordinator.req_to_host_pool[req.req_pool_idx, :fill_len] = host_indices
        stack.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx] = fill_len
        for lid in range(LAYER_NUM):
            for i in range(fill_len):
                host_pool.kv_buffer[lid][host_indices[i]] = self._kv_pattern(lid, i)
        stack.coordinator.admit_request_direct(req)
        return kv_loc

    def _alloc_step_tokens(self, stack, req, seq_len, n):
        """Allocate the n verify tokens' KV (logical) as production does."""
        device = stack.allocator.device
        return stack.allocator.alloc_extend(
            prefix_lens=torch.tensor([seq_len], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
            seq_lens=torch.tensor([seq_len + n], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([seq_len + n], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=n,
        )

    def _verify_step(self, stack, rpi, sls, tks):
        """One target-verify forward's swap-ins: 8 layers x N_POS positions.

        Returns {(layer, p): page_table.clone()} and the interleaved table.
        """
        c = stack.coordinator
        c.num_real_reqs[0] = rpi.shape[0]
        outs = {}
        out = torch.empty((rpi.shape[0] * N_POS, TOP_K), dtype=torch.int32, device="cuda")
        outv = out.view(rpi.shape[0], N_POS, TOP_K)
        for layer in range(LAYER_NUM):
            for p in range(N_POS):
                pt = c.swap_in_selected_pages(
                    rpi,
                    sls + (p + 1),
                    tks[p],
                    layer,
                    num_newest=p + 1,
                )
                outs[(layer, p)] = pt.clone()
                outv[:, p].copy_(pt)
        return outs, out

    def _commit(self, stack, rpi, sls, accept_lens):
        c = stack.coordinator
        c.commit_verify_tokens(
            sls,
            torch.tensor(accept_lens, dtype=torch.int64, device="cuda"),
            rpi,
            sls.cpu(),
            rpi.cpu(),
        )
        torch.cuda.synchronize()

    def _make_selections(self, rng, seq_lens, overlap=0.5):
        """Per-position top-k sets with cross-position and cross-step overlap.

        Each position selects a shared core (half of TOP_K, evolving per step)
        plus position-fresh tokens, and positions p>=1 include a few draft
        tokens [seq_len, seq_len+p] (they resolve to the reserved page via the
        newest-token window, exactly as production verify selections do).
        Selections are duplicate-free, as production top-k of distinct token
        positions is: the kernel's shared-memory hash is only deterministic
        for distinct tokens.
        """
        tks = []
        core = []
        for r, seq_len in enumerate(seq_lens):
            core.append(set(rng.sample(range(seq_len), TOP_K // 2)))
        for p in range(N_POS):
            rows = []
            for r, seq_len in enumerate(seq_lens):
                drafts = set(range(seq_len, seq_len + p + 1))
                fresh_n = TOP_K - len(core[r]) - len(drafts)
                pool = [
                    t
                    for t in rng.sample(range(seq_len), min(4 * fresh_n, seq_len))
                    if t not in core[r]
                ]
                fresh = set(rng.sample(pool, fresh_n))
                sel = sorted(core[r] | drafts | fresh)
                assert len(sel) == TOP_K, f"selection size {len(sel)} != {TOP_K}"
                rows.append(torch.tensor(sel, dtype=torch.int32))
            tks.append(torch.stack(rows).contiguous().to("cuda"))
        return tks

    # -- the test ----------------------------------------------------------
    def test_verify_prefetch_matches_sync(self):
        import random

        torch.manual_seed(7)
        for stack in (self.sync, self.prefetch):
            stack.reset()
        reqs = [_make_req(f"verify-{i}", list(range(FILL_LEN))) for i in range(2)]
        for stack in (self.sync, self.prefetch):
            for req in reqs:
                idx = stack.req_to_token_pool.alloc([req])
                self.assertIsNotNone(idx)
                self._admit(stack, req, FILL_LEN)

            seq_lens = [FILL_LEN, FILL_LEN]
            rng = random.Random(11)
            lru_init = self.prefetch.coordinator.lru_slots.clone()

        for step in range(STEPS):
            # Selections per position [bs, TOP_K] (shared across layers per
            # anchor group in production; here all layers of a group share).
            tks = self._make_selections(rng, seq_lens)
            rpi = torch.tensor(
                [r.req_pool_idx for r in reqs], dtype=torch.int64, device="cuda"
            )
            sls = torch.tensor(seq_lens, dtype=torch.int64, device="cuda")

            # Allocate verify tokens + map reserved slots (production order):
            # out_cache_loc is the [bs*n] row-major locs of the verify tokens
            # (r0p0..r0p3, r1p0..).
            for stack in (self.sync, self.prefetch):
                per_req = [
                    self._alloc_step_tokens(stack, req, seq_lens[i], N_POS)
                    for i, req in enumerate(reqs)
                ]
                out_cache_loc = torch.cat(per_req, dim=0).to("cuda")
                stack.coordinator.map_verify_locs_to_buffer(
                    sls,
                    out_cache_loc,
                    rpi,
                    sls.cpu(),
                    rpi.cpu(),
                    N_POS,
                )

            outs_sync, table_sync = self._verify_step(self.sync, rpi, sls, tks)
            outs_pre, table_pre = self._verify_step(self.prefetch, rpi, sls, tks)
            torch.cuda.synchronize()

            # 1. Page tables identical per (layer, position) and interleaved.
            for key in outs_sync:
                self.assertTrue(
                    torch.equal(outs_sync[key], outs_pre[key]),
                    f"step {step}: page table differs at {key}",
                )
            self.assertTrue(torch.equal(table_sync, table_pre))

            # 2. Device KV bytes identical on every layer.
            for lid in range(LAYER_NUM):
                self.assertTrue(
                    torch.equal(
                        self.sync.device_pool.kv_buffer[lid],
                        self.prefetch.device_pool.kv_buffer[lid],
                    ),
                    f"step {step}: device KV differs at layer {lid}",
                )

            # 3. Anchor LRU/token maps identical; skip layers' maps untouched
            #    by the prefetch stack (copies are IO-only; only the planning
            #    kernel reads them, and skips never plan under prefetch).
            for lid in ANCHORS:
                self.assertTrue(
                    torch.equal(
                        self.sync.coordinator.lru_slots[lid],
                        self.prefetch.coordinator.lru_slots[lid],
                    ),
                    f"step {step}: anchor {lid} LRU diverged",
                )
                self.assertTrue(
                    torch.equal(
                        self.sync.coordinator.req_device_buffer_tokens[lid],
                        self.prefetch.coordinator.req_device_buffer_tokens[lid],
                    ),
                    f"step {step}: anchor {lid} token map diverged",
                )
            for lid in SKIPS:
                self.assertTrue(
                    torch.equal(
                        self.prefetch.coordinator.lru_slots[lid],
                        lru_init[lid],
                    ),
                    f"step {step}: skip {lid} LRU mutated under prefetch",
                )

            # 4. Post-accept host backup identical (accept 2 of 4 per req).
            accept = [ACCEPT_PER_STEP, ACCEPT_PER_STEP]
            self._commit(self.sync, rpi, sls, accept)
            self._commit(self.prefetch, rpi, sls, accept)
            for lid in range(LAYER_NUM):
                self.assertTrue(
                    torch.equal(
                        self.sync.coordinator.mem_pool_host.kv_buffer[lid],
                        self.prefetch.coordinator.mem_pool_host.kv_buffer[lid],
                    ),
                    f"step {step}: host backup differs at layer {lid}",
                )
            seq_lens = [s + ACCEPT_PER_STEP for s in seq_lens]

        # 5. Miss plans were recorded and are plausible (misses > 0 at least
        #    once — the test does exercise the copy path).
        total = 0
        for p in range(N_POS):
            total += int(
                self.prefetch.coordinator._miss_count_v[p, : len(reqs)]
                .sum()
                .item()
            )
        self.assertGreater(total, 0, "no misses recorded; test too easy")

    def test_verify_prefetch_cuda_graph_replay(self):
        """The fork/event prefetch pattern captures into a CUDA graph and the
        replay is bit-identical to the eager synchronous path (mirrors the
        upstream graph test for the decode path, at n=4 positions)."""
        import random

        torch.manual_seed(13)
        for stack in (self.sync, self.prefetch):
            stack.reset()
        reqs = [_make_req(f"graph-{i}", list(range(FILL_LEN))) for i in range(2)]
        for stack in (self.sync, self.prefetch):
            for req in reqs:
                idx = stack.req_to_token_pool.alloc([req])
                self.assertIsNotNone(idx)
                self._admit(stack, req, FILL_LEN)

        seq_lens = [FILL_LEN, FILL_LEN]
        rng = random.Random(3)
        rpi = torch.tensor(
            [r.req_pool_idx for r in reqs], dtype=torch.int64, device="cuda"
        )
        sls_buf = torch.tensor(seq_lens, dtype=torch.int64, device="cuda")
        tks_buf = [
            torch.zeros((len(reqs), TOP_K), dtype=torch.int32, device="cuda")
            for _ in range(N_POS)
        ]

        def captured_step():
            out = torch.empty(
                (rpi.shape[0] * N_POS, TOP_K), dtype=torch.int32, device="cuda"
            )
            outv = out.view(rpi.shape[0], N_POS, TOP_K)
            for layer in range(LAYER_NUM):
                for p in range(N_POS):
                    pt = self.prefetch.coordinator.swap_in_selected_pages(
                        rpi,
                        sls_buf + (p + 1),
                        tks_buf[p],
                        layer,
                        num_newest=p + 1,
                    )
                    outv[:, p].copy_(pt)
            return out

        # num_real_reqs is a persistent device tensor updated before replay
        # (CPU->CUDA copies are illegal inside capture unless pinned).
        self.prefetch.coordinator.num_real_reqs[0] = rpi.shape[0]

        # Warm up (JIT-compile all kernel instantiations) on a side stream so
        # capture never sees a compile, then reset the mutated LRU/KV state.
        warm = torch.cuda.Stream()
        tks = self._make_selections(rng, seq_lens)
        for p in range(N_POS):
            tks_buf[p].copy_(tks[p])
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            captured_step()
        torch.cuda.current_stream().wait_stream(warm)
        torch.cuda.synchronize()
        # Reset the warmup's mutations and bring BOTH stacks back to the
        # canonical pre-step state (admission + verify slot mapping, exactly
        # like the eager test's per-step prep).
        for stack in (self.sync, self.prefetch):
            stack.reset()
            for req in reqs:
                idx = stack.req_to_token_pool.alloc([req])
                self.assertIsNotNone(idx)
                self._admit(stack, req, FILL_LEN)
        new_rpi = torch.tensor(
            [r.req_pool_idx for r in reqs], dtype=torch.int64, device="cuda"
        )
        rpi.copy_(new_rpi)  # the graph reads this tensor at replay time
        sls = torch.tensor(seq_lens, dtype=torch.int64, device="cuda")
        for stack in (self.sync, self.prefetch):
            per_req = [
                self._alloc_step_tokens(stack, req, seq_lens[i], N_POS)
                for i, req in enumerate(reqs)
            ]
            out_cache_loc = torch.cat(per_req, dim=0).to("cuda")
            stack.coordinator.map_verify_locs_to_buffer(
                sls, out_cache_loc, new_rpi, sls.cpu(), new_rpi.cpu(), N_POS
            )
            # The warmup fetched rows into the prefetch stack's device pool;
            # scrub BOTH pools to a finite sentinel so unwritten rows compare equal and any
            # row a step does not rewrite is visibly the sentinel on both sides.
            for lid in range(LAYER_NUM):
                stack.device_pool.kv_buffer[lid].fill_(-777.0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = captured_step()
        torch.cuda.synchronize()

        # Replay against the eager synchronous reference, 3 fresh steps.
        for step in range(3):
            tks = self._make_selections(rng, seq_lens)
            for p in range(N_POS):
                tks_buf[p].copy_(tks[p])
            _, ref = self._verify_step(self.sync, rpi, sls_buf, tks)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(ref, out), f"graph step {step}: page table differs"
            )
            for lid in range(LAYER_NUM):
                self.assertTrue(
                    torch.equal(
                        self.sync.device_pool.kv_buffer[lid],
                        self.prefetch.device_pool.kv_buffer[lid],
                    ),
                    f"graph step {step}: device KV differs at layer {lid}",
                )




class TestResolveSpecGate(unittest.TestCase):
    """SGLANG_HISPARSE_SPEC_PREFETCH gating of resolve_shared_index_layers."""

    def _config(self):
        from sglang.srt.configs.model_config import dsa_layer_skips_topk

        cfg = SimpleNamespace(
            num_hidden_layers=8,
            index_topk_pattern=[False, True, True, True] * 2,
            index_topk_freq=1,
            cli_factor=1,
        )
        assert any(
            dsa_layer_skips_topk(cfg, i) for i in range(8)
        ), "test config must produce a sharing pattern"
        return cfg

    def _resolve(self, cfg, **kw):
        from sglang.srt.managers.hisparse_coordinator import (
            resolve_shared_index_layers,
        )

        return resolve_shared_index_layers(
            hf_text_config=cfg, pp_size=kw.get("pp_size", 1),
            is_speculative=kw.get("is_speculative", True),
        )

    def test_spec_off_by_default(self):
        import os

        os.environ.pop("SGLANG_HISPARSE_SPEC_PREFETCH", None)
        self.assertIsNone(self._resolve(self._config()))

    def test_spec_flag_enables(self):
        import os

        os.environ["SGLANG_HISPARSE_SPEC_PREFETCH"] = "1"
        try:
            self.assertTrue(any(self._resolve(self._config())))
        finally:
            os.environ.pop("SGLANG_HISPARSE_SPEC_PREFETCH", None)

    def test_decode_default_on(self):
        import os

        os.environ.pop("SGLANG_HISPARSE_SPEC_PREFETCH", None)
        self.assertTrue(any(self._resolve(self._config(), is_speculative=False)))

    def test_pp_off(self):
        import os

        os.environ["SGLANG_HISPARSE_SPEC_PREFETCH"] = "1"
        try:
            self.assertIsNone(
                self._resolve(self._config(), pp_size=4, is_speculative=False)
            )
        finally:
            os.environ.pop("SGLANG_HISPARSE_SPEC_PREFETCH", None)


if __name__ == "__main__":
    unittest.main()
