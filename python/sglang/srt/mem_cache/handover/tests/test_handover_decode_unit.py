"""Unit test: decode-arm (HiSparse) handover export -> local push -> import.

Builds two HiSparseCoordinator + HiSparseRadixCache stacks on one GPU, seeds
stack A with retained prefixes (host rows + device index keys + tree), exports
A, copies pages into B's pools (memcpy stand-in for the NIXL WRITE), imports
into B, and verifies: match_prefix + retained_prefix_len on B, host latent
bytes equal, indexer-key bytes equal, side-table mapping correct.

Run on one GPU: python test_handover_decode_unit.py
"""

import sys
import time
import traceback
from array import array

import numpy as np
import torch

PAGE = 64


def make_decode_stack(seed, device_tokens=16384, ratio=4, layer_num=4):
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29527")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("gloo", world_size=1, rank=0)

    from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
    from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.hisparse_radix_cache import HiSparseRadixCache
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

    dev = torch.device("cuda:0")
    torch.cuda.set_device(0)
    pool = HiSparseDSATokenToKVPool(
        size=device_tokens,
        page_size=PAGE,
        kv_lora_rank=512,
        dtype=torch.float8_e4m3fn,
        qk_rope_head_dim=64,
        layer_num=layer_num,
        device=dev,
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=576,
        host_to_device_ratio=ratio,
    )
    allocator = HiSparseTokenToKVPoolAllocator(
        device_tokens, PAGE, torch.float8_e4m3fn, dev, pool, need_sort=True,
        host_to_device_ratio=ratio,
    )
    req_pool = ReqToTokenPool(8, 8192, dev, False)
    coord = HiSparseCoordinator(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        top_k=64,
        device_buffer_size=512,
        device=dev,
        tp_group=None,
        host_to_device_ratio=ratio,
    )
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=PAGE,
    )
    tree = HiSparseRadixCache(params)
    tree.attach_coordinator(coord)
    tree._rng = torch.Generator(device="cpu").manual_seed(seed)
    return tree, coord, pool, allocator


def serve_retained(tree, coord, pool, allocator, tokens, layer_num):
    """Simulate cache_finished_req's retention: insert with logical indices,
    back host rows with random bytes, retain them; index keys at logical pages."""
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    n = len(tokens)
    logical = allocator.logical_attn_allocator.alloc(n)
    assert logical is not None
    logical = logical.to(torch.int64)
    host = coord.mem_pool_host.alloc(n)
    assert host is not None
    host = host.to(torch.int64)
    # random host latent bytes (page rows)
    host_pages = (host.view(-1, PAGE)[:, 0] // PAGE).to(torch.int64)
    item = coord.mem_pool_host.token_stride_size * PAGE
    hp_rand = torch.randint(0, 255, (len(host_pages), item), dtype=torch.uint8, generator=tree._rng)
    for l in range(layer_num):
        coord.mem_pool_host.data_refs[l].view(torch.uint8).view(-1, item)[
            host_pages
        ] = hp_rand
    # random device index-key bytes at logical pages
    logical_pages = (logical.view(-1, PAGE)[:, 0] // PAGE).to(torch.int64)
    idx_item = pool.index_k_with_scale_buffer[0].shape[-1]
    ik_rand = torch.randint(
        0, 255, (len(logical_pages), idx_item), dtype=torch.uint8, generator=tree._rng
    ).to("cuda")
    for l in range(layer_num):
        pool.index_k_with_scale_buffer[l][logical_pages] = ik_rand
    tree.insert(InsertParams(key=RadixKey(array("q", list(tokens))), value=logical, priority=0))
    coord.retain_rows(logical, host)
    torch.cuda.synchronize()
    return logical, host


def main():
    from sglang.srt.mem_cache.handover import decode_arm
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    layer_num = 4
    tree_a, coord_a, pool_a, alloc_a = make_decode_stack(41)
    tree_b, coord_b, pool_b, alloc_b = make_decode_stack(42)

    rng = np.random.default_rng(77)
    group = [int(x) for x in rng.integers(1000, 50000, size=16 * PAGE)]
    sessions = [
        group + [int(x) for x in rng.integers(1000, 50000, size=16 * PAGE)],
        group + [int(x) for x in rng.integers(1000, 50000, size=8 * PAGE)],
        [int(x) for x in rng.integers(1000, 50000, size=12 * PAGE)],
    ]
    served = []
    for s in sessions:
        logical, host = serve_retained(tree_a, coord_a, pool_a, alloc_a, s, layer_num)
        served.append((s, logical, host))

    export = decode_arm.build_decode_export(tree_a, coord_a)
    total_tokens = export.seg_tokens(PAGE)
    print(
        f"export: chains={len(export.chains)} pages={export.num_pages} "
        f"tokens={total_tokens} host_item={export.host_item_len} "
        f"index_item={export.index_item_len}"
    )

    logical_slots, host_slots, dst_logical, dst_host = decode_arm.local_push_decode(
        coord_a, coord_b, export
    )
    stats = decode_arm.import_decode(
        tree_b, coord_b, export, logical_slots, host_slots
    )
    print(f"import stats: {stats}")

    ok_ret = True
    for s, logical_a, host_a in served:
        res = tree_b.match_prefix(MatchPrefixParams(key=RadixKey(array("q", list(s)))))
        retained = tree_b.retained_prefix_len(res.device_indices)
        want = len(s) - len(s) % PAGE
        if retained < want:
            print(f"FAIL retained: got {retained} want {want}")
            ok_ret = False
    print("retained-prefix hits on B: OK" if ok_ret else "retained-prefix hits: FAIL")

    # 2. host latent bytes: A rows vs B rows (canonical pages)
    ok_host = True
    old_logical, host_rows = decode_arm.flat_pages(export)
    item = export.host_item_len
    for l in range(layer_num):
        a2d = coord_a.mem_pool_host.data_refs[l].view(torch.uint8).view(-1, item)
        b2d = coord_b.mem_pool_host.data_refs[l].view(torch.uint8).view(-1, item)
        a_rows = torch.as_tensor(host_rows, dtype=torch.long)
        b_rows = torch.as_tensor(dst_host, dtype=torch.long)
        if not torch.equal(a2d.index_select(0, a_rows), b2d.index_select(0, b_rows)):
            print(f"FAIL host latent bytes layer {l}")
            ok_host = False
    print("host latent bytes: OK" if ok_host else "host latent bytes: FAIL")

    # 3. index-key bytes: A old logical pages vs B dst logical pages
    ok_idx = True
    idx_item = export.index_item_len
    src_t = torch.as_tensor(old_logical, dtype=torch.long, device="cuda")
    dst_t = torch.as_tensor(dst_logical, dtype=torch.long, device="cuda")
    for l in range(layer_num):
        a2d = pool_a.index_k_with_scale_buffer[l].view(-1, idx_item)
        b2d = pool_b.index_k_with_scale_buffer[l].view(-1, idx_item)
        if not torch.equal(a2d.index_select(0, src_t), b2d.index_select(0, dst_t)):
            print(f"FAIL index key bytes layer {l}")
            ok_idx = False
    print("index-key bytes: OK" if ok_idx else "index-key bytes: FAIL")

    # 4. side table: every matched prefix's retained logical has a host row
    ok_side = True
    res = tree_b.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", list(sessions[0]))))
    )
    rows = coord_b.logical_to_host_row[res.device_indices.cpu()]
    if bool((rows < 0).any()):
        print("FAIL side table has holes")
        ok_side = False
    else:
        print(f"side table: {len(rows)} logical indices all retained OK")
    ok = ok_ret and ok_host and ok_idx and ok_side

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1




if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
