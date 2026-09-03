"""Scale test: prefill-arm handover at ~0.2M / 0.5M / 1M retained tokens.

Serving loop mimics the real hicache lifecycle: per session, alloc device rows
-> insert (auto write-through backup) -> evict device rows, so the host tier
accumulates fragmented retained rows exactly like a long-lived prefill rank.
Then measures, per scale:

  export: tree walk + protect + (direct) page-row collection + full checksums
  staging: pinned gather into contiguous per-layer buffers (GB/s)
  manifest: serialized bytes
  import: fresh-heir bulk tree insert (tree-rebuild time) + turn-2 match latency

Usage: python test_handover_scale.py [--scales 200k 500k 1m] [--reps 3]
"""

import argparse
import json
import statistics
import sys
import time
import traceback
from array import array

import numpy as np
import torch

PAGE = 64
import os
torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))  # cgroup-aware: default sees all 288 cores -> OMP thrash


def make_stack(size_tokens, layer_num, hicache_ratio, seed, model_path="dummy"):
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29519")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("gloo", world_size=1, rank=0)

    from sglang.srt import runtime_context
    from sglang.srt.server_args import ServerArgs

    sa = ServerArgs(model_path=model_path)
    sa.enable_hierarchical_cache = True
    sa.hicache_ratio = hicache_ratio
    sa.hicache_size = 0
    sa.hicache_mem_layout = "layer_first"
    sa.hicache_write_policy = "write_through"
    sa.hicache_io_backend = "kernel"
    sa.hicache_storage_backend = None
    sa.served_model_name = model_path
    runtime_context.publish(sa, role="scheduler")

    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

    dev = torch.device("cuda:0")
    torch.cuda.set_device(0)
    pool = DSATokenToKVPool(
        size=size_tokens,
        page_size=PAGE,
        kv_lora_rank=512,
        dtype=torch.float8_e4m3fn,
        qk_rope_head_dim=64,
        layer_num=layer_num,
        device=dev,
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=576,
    )
    allocator = PagedTokenToKVPoolAllocator(
        size_tokens, PAGE, torch.float8_e4m3fn, dev, pool, need_sort=True
    )
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=allocator,
        page_size=PAGE,
    )
    tree = HiRadixCache(params=params, server_args=sa)
    tree._rng = torch.Generator(device="cpu").manual_seed(seed)
    return tree, pool, allocator


def serve_and_evict(tree, pool, allocator, tokens, layer_num):
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    assert len(tokens) % PAGE == 0
    dev_idx = allocator.alloc(len(tokens))
    assert dev_idx is not None, "device pool exhausted"
    dev_idx = dev_idx.to(torch.int64)
    kv_rand = torch.randint(
        0, 255, (len(tokens), 576), dtype=torch.uint8, generator=tree._rng
    ).to("cuda")
    for l in range(layer_num):
        pool.kv_buffer[l][dev_idx, 0, :] = kv_rand.view(pool.kv_buffer[l].dtype)
    pages = (dev_idx.view(-1, PAGE)[:, 0] // PAGE).to(torch.int64)
    idx_rand = torch.randint(
        0, 255, (len(pages), pool.index_k_with_scale_buffer[0].shape[-1]),
        dtype=torch.uint8, generator=tree._rng,
    ).to("cuda")
    for l in range(layer_num):
        pool.index_k_with_scale_buffer[l][pages] = idx_rand
    key = RadixKey(array("q", list(tokens)))
    tree.insert(InsertParams(key=key, value=dev_idx, priority=0))
    # drain the write-through ack so the node unlocks before we evict
    deadline = time.monotonic() + 60
    while tree.ongoing_write_through:
        torch.cuda.synchronize()
        tree.check_hicache_events()
        assert time.monotonic() < deadline, "write-through ack never completed"
        time.sleep(0.002)
    # reclaim device rows (hicache lifecycle: host tier retains)
    tree.evict(EvictParams(num_tokens=len(tokens)))
    tree.check_hicache_events()
    torch.cuda.synchronize()


def gen_sessions(total_tokens, rng):
    """Sessions with shared group prefixes, ~2-6k tokens each."""
    sessions = []
    made = 0
    group_prefix = list(rng.integers(1000, 50000, size=32 * PAGE))
    while made < total_tokens:
        n_pages = int(rng.integers(32, 96))  # 2k-6k tokens
        share = rng.random() < 0.6
        if share:
            toks = group_prefix + list(rng.integers(1000, 50000, size=n_pages * PAGE))
        else:
            toks = list(rng.integers(1000, 50000, size=(n_pages + 32) * PAGE))
        sessions.append(toks)
        made += len(toks)
    return sessions


def run_scale(n_tokens_target, layer_num, device_tokens, hicache_ratio, reps, res_rows):
    print(f"\n=== scale {n_tokens_target} tokens (layer_num={layer_num}) ===")
    tree_a, pool_a, alloc_a = make_stack(device_tokens, layer_num, hicache_ratio, 11)
    rng = np.random.default_rng(1234)
    sessions = gen_sessions(n_tokens_target, rng)
    t0 = time.perf_counter()
    for s in sessions:
        serve_and_evict(tree_a, pool_a, alloc_a, s, layer_num)
    serve_s = time.perf_counter() - t0
    n_retained = sum(
        len(n.key)
        for stack_i in [tree_a.root_node]
        for n in _walk(stack_i)
        if n.backuped
    )
    print(
        f"served {len(sessions)} sessions in {serve_s:.1f}s; retained host tokens "
        f"~{n_retained}"
    )

    from sglang.srt.mem_cache.handover import prefill_arm
    from sglang.srt.mem_cache.handover.manifest import (
        bytes_to_manifest,
        manifest_to_bytes,
        page_checksums,
    )
    from sglang.srt.mem_cache.handover.prefill_arm import (
        _per_layer_host_buffers,
        alloc_heir_rows,
        heir_page_rows,
        tree_pools,
    )
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    # fresh heir per rep? Heir pools are big; rebuild one heir and reset between reps
    tree_b, pool_b, alloc_b = make_stack(
        device_tokens, layer_num, hicache_ratio, 12
    )
    staging_cache = {}  # pinned staging reused across reps (prod pre-warms)

    for rep in range(reps):
        # -- export (direct + checksums) --
        t0 = time.perf_counter()
        export = prefill_arm.PrefillExport.build(
            tree_a, "dummy", staged=False, with_checksums=(rep == 0)
        )
        t_export_ck = time.perf_counter() - t0
        t0 = time.perf_counter()
        export_nc = prefill_arm.PrefillExport.build(
            tree_a, "dummy", staged=False, with_checksums=False
        )
        t_export = time.perf_counter() - t0
        m = export_nc.manifest
        mbytes = len(manifest_to_bytes(m))
        export.release_protections()
        export_nc.release_protections()

        # -- staging (pinned alloc cached after rep 0) --
        t0 = time.perf_counter()
        export_st = prefill_arm.PrefillExport.build(
            tree_a, "dummy", staged=True, with_checksums=False,
            staging_cache=staging_cache,
        )
        t_stage_build = time.perf_counter() - t0  # walk + gather (+ alloc rep 0)
        t_stage_alloc = getattr(m, "_stage_alloc_s", 0.0)
        t_stage_gather = getattr(m, "_stage_gather_s", 0.0)
        staged_bytes = sum(
            m.num_pages * export_st.pool_specs[nm].item_len * export_st.pool_specs[nm].layer_num
            for nm in export_st.pool_specs
        )

        # -- import (tree rebuild), fresh heir state --
        tree_b.reset()
        tree_b.token_to_kv_pool_host.clear()
        slots = alloc_heir_rows(tree_b, m.num_tokens)
        dst_pages = heir_page_rows(slots, PAGE)
        src_pages = export_st.src_pages  # staged: contiguous
        # land bytes by memcpy (in-process stand-in; wire test separate)
        pools_a, pools_b = tree_pools(tree_a), tree_pools(tree_b)
        t0 = time.perf_counter()
        for name in pools_a:
            spec = export_st.pool_specs[name]
            item = spec.item_len
            bufs_a = export_st.staging[name]
            bufs_b = _per_layer_host_buffers(pools_b[name])
            dst_rows_t = torch.as_tensor(dst_pages, dtype=torch.long)
            for la, lb in zip(bufs_a, bufs_b):
                lb.view(-1, item).index_copy_(0, dst_rows_t, la)
        t_land = time.perf_counter() - t0

        t0 = time.perf_counter()
        stats = prefill_arm.import_manifest(tree_b, m, slots, need_hashes=False)
        t_import = time.perf_counter() - t0

        # turn-2 match latency (p50 over all sessions)
        lat = []
        for s in sessions[:200]:
            t0 = time.perf_counter()
            res = tree_b.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", list(s))))
            )
            lat.append(time.perf_counter() - t0)
            assert res.host_hit_length >= len(s) - PAGE, (
                f"hit {res.host_hit_length} < {len(s) - PAGE}"
            )
        lat_ms = sorted(lat)[len(lat) // 2] * 1e3

        # checksum verify cost (heir side, full)
        t0 = time.perf_counter()
        for name in pools_b:
            page_checksums(
                _per_layer_host_buffers(pools_b[name]),
                dst_pages,
                export_st.pool_specs[name].item_len,
            )
        t_ck_b = time.perf_counter() - t0

        row = {
            "scale_tokens": m.num_tokens,
            "pages": m.num_pages,
            "chains": len(m.chains),
            "manifest_MB": round(mbytes / 1e6, 2),
            "t_export_s": round(t_export, 3),
            "t_export_with_ck_s": round(t_export_ck, 3),
            "t_stage_walk_gather_s": round(t_stage_build, 3),
            "t_stage_alloc_s": round(t_stage_alloc, 3),
            "t_stage_gather_s": round(t_stage_gather, 3),
            "stage_gather_GBps": round(
                staged_bytes / 1e9 / max(t_stage_gather, 1e-9), 1
            ),
            "staged_GB": round(staged_bytes / 1e9, 2),
            "stage_GBps": round(staged_bytes / 1e9 / max(t_stage_build, 1e-9), 1),
            "t_land_memcpy_s": round(t_land, 3),
            "t_import_tree_s": round(t_import, 3),
            "import_tok_per_s": int(m.num_tokens / max(t_import, 1e-9)),
            "t_heir_checksum_s": round(t_ck_b, 3),
            "match_p50_ms": round(lat_ms, 3),
            "rep": rep,
        }
        res_rows.append(row)
        print(json.dumps(row))
        del export_st

    return res_rows


def _walk(root):
    stack = [root]
    while stack:
        n = stack.pop()
        if n is not root:
            yield n
        stack.extend(list(n.children.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", default=["200k", "500k", "1m"])
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    mult = {"200k": 200_000, "500k": 500_000, "1m": 1_000_000}
    layer_num = 4  # small rig; per-layer bytes scale linearly
    device_tokens = 131072
    rows = []
    for s in args.scales:
        target = mult[s]
        # host pool: 2x target headroom (ratio set so host >= target + device)
        hicache_ratio = max(2, (target * 2) // device_tokens)
        rows = run_scale(target, layer_num, device_tokens, hicache_ratio, args.reps, rows)
    with open("scale_results.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote scale_results.json")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
