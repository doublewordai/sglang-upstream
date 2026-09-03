"""Unit test: prefill-arm handover export -> (local push) -> import.

Builds two engine-like HiRadixCache stacks (DSA: MLATokenToKVPoolHost anchor +
DSAIndexerPoolHost sidecar) in one process, serves synthetic sessions on A
(real insert + real write_backup D2H), exports A, "pushes" by direct
pool-row copy (in-process stand-in for the NIXL WRITE), imports into B, and
verifies:

  * turn-2 semantics: match_prefix on B hits the full session prefix in the
    host tier (host_hit_length == len) with best_match_node.backuped
  * byte-verified rows: every landed KV + indexer page on B equals A's
  * checksums catch a corrupted page
  * manifest serialization round-trips
  * load_back revives the prefix into B's device pool byte-identically

Run on one GPU:
  python test_handover_unit.py [--staged] [--no-checksums]
"""

import argparse
import time
import sys
import traceback
from array import array

import numpy as np
import torch

PAGE = 64
import os
torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))  # cgroup-aware: default sees all 288 cores -> OMP thrash


def make_stack(size_tokens, layer_num, hicache_ratio, seed, model_path="dummy"):
    """One engine-like DSA HiRadixCache stack on cuda:0."""
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("gloo", world_size=1, rank=0)

    from sglang.srt import runtime_context
    from sglang.srt.server_args import ServerArgs

    sa = ServerArgs(model_path=model_path)
    sa.enable_hierarchical_cache = True
    sa.hicache_ratio = hicache_ratio
    sa.hicache_size = 0  # ratio-based
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


def serve_session(tree, pool, allocator, tokens, layer_num):
    """Insert one session (page-aligned) with random device KV+indexer bytes."""
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    assert len(tokens) % PAGE == 0
    dev_idx = allocator.alloc(len(tokens))
    assert dev_idx is not None
    dev_idx = dev_idx.to(torch.int64)
    kv_rand = torch.randint(
        0, 255, (len(tokens), 576), dtype=torch.uint8, generator=tree._rng
    ).to("cuda")
    for l in range(layer_num):
        pool.kv_buffer[l][dev_idx, 0, :] = kv_rand.view(pool.kv_buffer[l].dtype)
    pages = (dev_idx.view(-1, PAGE)[:, 0] // PAGE).to(torch.int64)
    page_stride = pool.index_k_with_scale_buffer[0].shape[-1]
    idx_rand = torch.randint(
        0, 255, (len(pages), page_stride), dtype=torch.uint8, generator=tree._rng
    ).to("cuda")
    for l in range(layer_num):
        pool.index_k_with_scale_buffer[l][pages] = idx_rand
    key = RadixKey(array("q", list(tokens)))
    tree.insert(InsertParams(key=key, value=dev_idx, priority=0))
    torch.cuda.synchronize()
    return dev_idx


def backup_all(tree):
    """Write-backup every non-backuped node, parents first (write_through)."""

    def pre(n):
        count = 0
        if n is not tree.root_node and not n.backuped and not n.evicted:
            got = tree.write_backup(n)
            assert got > 0, f"write_backup returned 0 for node {n.id}"
            count += 1
        for c in n.children.values():
            count += pre(c)
        return count

    n = pre(tree.root_node)
    tree.writing_check()
    torch.cuda.synchronize()
    return n


def _evicted_path(node):
    out = []
    while node.evicted:
        out.insert(0, node)
        node = node.parent
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--no-checksums", action="store_true")
    args = ap.parse_args()

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

    layer_num = 4
    size_tokens = 4096
    tree_a, pool_a, alloc_a = make_stack(size_tokens, layer_num, 2, 1)
    tree_b, pool_b, alloc_b = make_stack(size_tokens, layer_num, 2, 2)

    rng = np.random.default_rng(7)
    base = list(rng.integers(1000, 50000, size=5 * PAGE))
    sessions = [
        base + list(rng.integers(1000, 50000, size=3 * PAGE)),
        base + list(rng.integers(1000, 50000, size=2 * PAGE)),
        list(rng.integers(1000, 50000, size=4 * PAGE)),
    ]
    served = []
    for tokens in sessions:
        dev_idx = serve_session(tree_a, pool_a, alloc_a, tokens, layer_num)
        served.append((tokens, dev_idx))
    n_backed = backup_all(tree_a)
    print(f"A: served {len(sessions)} sessions, backed up {n_backed} nodes")

    export_a = prefill_arm.PrefillExport.build(
        tree_a, "dummy", staged=args.staged, with_checksums=not args.no_checksums
    )
    m = export_a.manifest
    print(
        f"export: chains={len(m.chains)} tokens={m.num_tokens} pages={m.num_pages} "
        f"staged={m.staged} cksum_pools={sorted((m.checksums or {}).keys())}"
    )
    assert m.num_pages * PAGE == m.num_tokens

    # manifest serialization round-trip
    m2 = bytes_to_manifest(manifest_to_bytes(m))
    assert m2.fingerprint == m.fingerprint
    assert m2.num_tokens == m.num_tokens and len(m2.chains) == len(m.chains)
    for c1, c2 in zip(m.chains, m2.chains):
        assert list(c1.tokens) == list(c2.tokens)
        assert np.array_equal(c1.page_rows, c2.page_rows) or m.staged
        assert c1.page_hashes == c2.page_hashes
    print("manifest round-trip OK")

    # ---- local push: copy A's source pages into B's destination pages ----
    n_tokens = m.num_tokens
    slots = alloc_heir_rows(tree_b, n_tokens)
    dst_pages = heir_page_rows(slots, PAGE)
    src_pages = export_a.src_pages
    pools_a, pools_b = tree_pools(tree_a), tree_pools(tree_b)
    landed = {}
    for name in pools_a:
        spec = export_a.pool_specs[name]
        item = spec.item_len
        bufs_a = export_a.staging[name] if export_a.staging else _per_layer_host_buffers(
            pools_a[name]
        )
        bufs_b = _per_layer_host_buffers(pools_b[name])
        src_rows_t = torch.as_tensor(src_pages, dtype=torch.long)
        dst_rows_t = torch.as_tensor(dst_pages, dtype=torch.long)
        for la, lb in zip(bufs_a, bufs_b):
            src2d = la.view(-1, item)
            lb.view(-1, item).index_copy_(0, dst_rows_t, src2d.index_select(0, src_rows_t))
        landed[name] = (dst_pages, bufs_b, item)
    print(f"local push: {n_tokens} tokens, {len(src_pages)} pages/pool")

    def cksum_fn(pool_name):
        dst_p, bufs_b, item = landed[pool_name]
        return page_checksums(bufs_b, dst_p, item)

    stats = prefill_arm.import_manifest(
        tree_b, m, slots, verify_checksums_fn=cksum_fn, need_hashes=True
    )
    print(f"import stats: {stats}")
    assert stats["checksum_mismatches"] == 0

    ok = True

    # 1. turn-2 host hit on B for every session
    for tokens in sessions:
        res = tree_b.match_prefix(MatchPrefixParams(key=RadixKey(array("q", list(tokens)))))
        if res.host_hit_length != len(tokens):
            print(f"FAIL host hit: got {res.host_hit_length} want {len(tokens)}")
            ok = False
        if not res.best_match_node.backuped:
            print("FAIL best_match_node not backuped")
            ok = False
    print("host-tier hits on B: OK" if ok else "host-tier hits on B: FAIL")

    # 2. byte comparison A vs B over every canonical page (all pools, layers)
    for name in pools_a:
        spec = export_a.pool_specs[name]
        bufs_a = export_a.staging[name] if export_a.staging else _per_layer_host_buffers(
            pools_a[name]
        )
        bufs_b = landed[name][1]
        for li, (la, lb) in enumerate(zip(bufs_a, bufs_b)):
            a_flat, b_flat = la.view(-1), lb.view(-1)
            # vectorized compare
            for j in range(0, len(src_pages), 512):
                sp = src_pages[j : j + 512]
                dp = dst_pages[j : j + 512]
                for s, d in zip(sp, dp):
                    if not torch.equal(
                        a_flat[s * spec.item_len : (s + 1) * spec.item_len],
                        b_flat[d * spec.item_len : (d + 1) * spec.item_len],
                    ):
                        print(f"FAIL byte mismatch pool={name} layer={li} page={j}")
                        ok = False
                        break
    print("byte comparison A vs B: OK" if ok else "byte comparison A vs B: FAIL")

    # 3. load_back revives a session into B's device pool, byte-identical
    tokens, dev_idx_a = served[0]
    res = tree_b.match_prefix(MatchPrefixParams(key=RadixKey(array("q", list(tokens)))))
    loaded = tree_b.load_back(res.best_match_node)
    assert loaded is not None, "load_back returned None"
    assert len(loaded) == len(tokens)
    # the H2D copy is asynchronous: submit like the scheduler does
    tree_b.ready_to_load_host_cache()
    deadline = time.monotonic() + 30
    while not tree_b.cache_controller.ack_load_queue:
        assert time.monotonic() < deadline, "load ack never appeared"
        time.sleep(0.05)
    tree_b.check_hicache_events()
    tree_b.loading_check()
    torch.cuda.synchronize()
    # diagnostics: where do A-device and B-device differ?
    for l in range(layer_num):
        a_row = pool_a.kv_buffer[l][dev_idx_a.to(torch.int64)].view(torch.uint8)
        b_row = pool_b.kv_buffer[l][loaded].view(torch.uint8)
        if not torch.equal(a_row, b_row):
            ntok_diff = (
                (a_row != b_row).any(dim=-1).sum().item()
            )
            first_tok = int((a_row != b_row).any(dim=-1).nonzero()[0, 0])
            print(
                f"  diff layer {l}: {ntok_diff}/{len(tokens)} tokens differ, "
                f"first token idx {first_tok}"
            )
            # B host rows vs B device rows for the same tokens
            hv = torch.cat(
                [n.host_value for n in _evicted_path(res.best_match_node)]
            )
            b_host = tree_b.token_to_kv_pool_host.get_pool.__self__.entry_map[  # noqa
                list(tree_b.token_to_kv_pool_host.entry_map)[0]
            ].host_pool
            print(f"  host path rows: {hv[:8].tolist()}")
    same = all(
        torch.equal(
            pool_a.kv_buffer[l][dev_idx_a].view(torch.uint8),
            pool_b.kv_buffer[l][loaded].view(torch.uint8),
        )
        for l in range(layer_num)
    )
    print(f"load_back byte-identical: {same}")
    ok = ok and same

    # 4. corruption detection: flip a byte on B's landed page, must raise
    if not args.no_checksums:
        dst_p, bufs_b, item = landed["kv"]
        d0 = int(dst_pages[0])
        bufs_b[0].view(-1)[d0 * export_a.pool_specs["kv"].item_len + 5] ^= 0xFF
        try:
            prefill_arm.import_manifest(tree_b, m, slots, verify_checksums_fn=cksum_fn)
            print("FAIL: corrupted page was NOT detected")
            ok = False
        except RuntimeError as e:
            print(f"checksum corruption detected: {str(e)[:80]}")

    # 5. host eviction of imported nodes works (protections/rows consistent)
    evicted = tree_b.evict_host(m.num_tokens)
    print(f"host eviction on B after import: freed {evicted} tokens")
    tree_b.match_prefix(MatchPrefixParams(key=RadixKey(array("q", list(sessions[0])))))

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
