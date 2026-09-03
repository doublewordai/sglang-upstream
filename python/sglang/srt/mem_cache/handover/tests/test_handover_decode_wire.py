"""Wire test: decode-arm (HiSparse) handover push over NIXL/UCCL, incl. the
VRAM->VRAM index-key path (a wire combination the cache-handover lane never
measured: their matrix had DRAM senders only).

Old side on node 1: builds a decode stack, serves retained sessions, exports;
control socket: phase 1 = sizes, phase 2 = push (host latent DRAM->DRAM,
index keys VRAM->VRAM, manifest DRAM->DRAM), wait DONE.
Heir on node 2: allocates logical indices + host rows, registers pools,
sends buffers, waits notifs, deserializes, imports, verifies retained
prefixes + bytes.

Usage (two srun steps):
  python test_handover_decode_wire.py --role old  --port 24474 [--tokens 200000]
  python test_handover_decode_wire.py --role heir --host <ip> --port 24474
"""

import argparse
import base64
import json
import socket
import sys
import time
import traceback
from array import array

import numpy as np
import torch

PAGE = 64


def make_decode_stack(seed, device_tokens=32768, ratio=8, layer_num=4):
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", os.environ.get("WH_MASTER_PORT", "24476"))
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
    req_pool = ReqToTokenPool(8, 16384, dev, False)
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
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    n = len(tokens)
    logical = allocator.logical_attn_allocator.alloc(n).to(torch.int64)
    host = coord.mem_pool_host.alloc(n).to(torch.int64)
    host_pages = (host.view(-1, PAGE)[:, 0] // PAGE).to(torch.int64)
    item = coord.mem_pool_host.token_stride_size * PAGE
    hp_rand = torch.randint(0, 255, (len(host_pages), item), dtype=torch.uint8, generator=tree._rng)
    for l in range(layer_num):
        coord.mem_pool_host.data_refs[l].view(torch.uint8).view(-1, item)[host_pages] = hp_rand
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


def gen_sessions(total_tokens, rng):
    sessions, made = [], 0
    group = [int(x) for x in rng.integers(1000, 50000, size=16 * PAGE)]
    while made < total_tokens:
        n_pages = int(rng.integers(16, 48))
        if rng.random() < 0.6:
            toks = group + [int(x) for x in rng.integers(1000, 50000, size=n_pages * PAGE - 16 * PAGE)]
        else:
            toks = [int(x) for x in rng.integers(1000, 50000, size=n_pages * PAGE)]
        sessions.append(toks)
        made += len(toks)
    return sessions




def page_ck(bufs, rows, item):
    """Per-page int64 sums over per-layer buffers (CPU or CUDA)."""
    import torch as T

    rows_t = T.as_tensor(rows, dtype=T.long, device=bufs[0].device)
    n_int = item // 8
    out = np.zeros(len(rows), dtype=np.int64)
    for b in bufs:
        iv = b.view(T.uint8).view(-1).view(T.int64) if b.is_cuda else b.view(T.uint8).view(T.int64)
        g = iv.view(-1, n_int)[rows_t]
        out += g.sum(dim=1).cpu().numpy()
    return out

def send_msg(sock, obj):
    b = json.dumps(obj).encode()
    sock.sendall(len(b).to_bytes(8, "little") + b)


def recv_msg(sock):
    n = int.from_bytes(recv_exact(sock, 8), "little")
    return json.loads(recv_exact(sock, n))


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf


def run_old(port, n_tokens, layer_num):
    from sglang.srt.mem_cache.handover import decode_arm
    from sglang.srt.mem_cache.handover.decode_arm import build_decode_export
    from sglang.srt.mem_cache.handover.decode_arm import decode_export_to_bytes
    from sglang.srt.mem_cache.handover.transfer import (
        HandoverNixlAgent,
        build_pool_descriptors,
    )

    tree, coord, pool, allocator = make_decode_stack(51, layer_num=layer_num)
    rng = np.random.default_rng(11)
    sessions = gen_sessions(n_tokens, rng)
    for s in sessions:
        serve_retained(tree, coord, pool, allocator, s, layer_num)
    export = build_decode_export(tree, coord)
    mbytes = decode_export_to_bytes(export, PAGE)
    print(
        f"[old] served {len(sessions)} sessions; export chains={len(export.chains)} "
        f"pages={export.num_pages} manifest={len(mbytes)/1e6:.2f}MB",
        flush=True,
    )

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"[old] listening on {port}", flush=True)
    conn, addr = srv.accept()
    old_logical, host_rows = decode_arm.flat_pages(export)
    ck_host = page_ck(coord.mem_pool_host.data_refs, host_rows, export.host_item_len)
    ck_index = page_ck(pool.index_k_with_scale_buffer, old_logical, export.index_item_len)
    send_msg(
        conn,
        {
            "num_pages": export.num_pages,
            "manifest_len": len(mbytes),
            "layer_num": export.layer_num,
            "host_item_len": export.host_item_len,
            "index_item_len": export.index_item_len,
            "ck_host": [int(x) for x in ck_host],
            "ck_index": [int(x) for x in ck_index],
        },
    )
    req = recv_msg(conn)

    agent = HandoverNixlAgent(backend="UCCL", gpu_id=0, cxi_device_index=0)
    host_ptrs, host_lens, _ = coord.mem_pool_host.get_contiguous_buf_infos()
    agent.register("src_host", list(zip(host_ptrs, host_lens)), "DRAM")
    # VRAM->VRAM WRITE fails with NIXL_ERR_BACKEND on this UCCL stack (posted
    # OK, errored in check_xfer_state) -> stage the index keys through pinned
    # DRAM on the old side and push DRAM->VRAM (cache-handover-validated combo)
    idx_staging = [
        torch.empty(tuple(b.shape), dtype=torch.uint8, pin_memory=True)
        for b in pool.index_k_with_scale_buffer
    ]
    idx_lens = [b.numel() for b in idx_staging]
    t_d2h = time.perf_counter()
    for b, st in zip(pool.index_k_with_scale_buffer, idx_staging):
        st.copy_(b.view(torch.uint8).view(b.shape), non_blocking=True)
    torch.cuda.synchronize()
    t_d2h = time.perf_counter() - t_d2h
    print(f"[old] index-key D2H staging: {sum(x.numel() for x in idx_staging)/1e9:.2f}GB in {t_d2h*1e3:.0f}ms", flush=True)
    idx_ptrs = [int(b.data_ptr()) for b in idx_staging]
    agent.register("src_index", list(zip(idx_ptrs, idx_lens)), "DRAM")
    msrc = torch.frombuffer(bytearray(mbytes), dtype=torch.uint8).pin_memory()
    agent.register("src_manifest", [(msrc.data_ptr(), len(mbytes))], "DRAM")
    agent.add_peer(base64.b64decode(req["agent_metadata"]))

    old_logical, host_rows = decode_arm.flat_pages(export)
    t0 = time.perf_counter()
    handles = []
    total_bytes = 0
    # host latent DRAM->DRAM
    src_addrs, dst_addrs = build_pool_descriptors(
        host_ptrs, req["host"]["ptrs"], export.host_item_len,
        host_rows, np.array(req["host"]["dst_pages"], dtype=np.int64),
    )
    total_bytes += sum(l for _, l in src_addrs)
    handles.append(agent.push_write(src_addrs, dst_addrs, "DRAM", "DRAM", 0,
                                    req["agent_name"], "handover_host"))
    # index keys VRAM->VRAM
    src_addrs, dst_addrs = build_pool_descriptors(
        idx_ptrs, req["index"]["ptrs"], export.index_item_len,
        old_logical, np.array(req["index"]["dst_pages"], dtype=np.int64),
    )
    total_bytes += sum(l for _, l in src_addrs)
    handles.append(agent.push_write(src_addrs, dst_addrs, "DRAM", "VRAM", 0,
                                    req["agent_name"], "handover_index"))
    # manifest
    mdst = req["manifest"]
    handles.append(agent.push_write(
        [(msrc.data_ptr(), len(mbytes))], [(mdst["ptr"], mdst["len"])],
        "DRAM", "DRAM", 0, req["agent_name"], "handover_manifest"))
    ok = agent.wait(handles, timeout_s=300)
    t_wire = time.perf_counter() - t0
    print(
        f"[old] push done={ok} bytes={total_bytes/1e9:.3f}GB wire={t_wire*1e3:.0f}ms "
        f"-> {total_bytes/1e9/t_wire:.2f} GB/s (host DRAM->DRAM + index VRAM->VRAM)",
        flush=True,
    )
    send_msg(conn, {"status": "ok" if ok else "timeout", "bytes": total_bytes,
                    "t_wire_s": t_wire})
    conn.close(); srv.close(); agent.close()
    print("[old] done", flush=True)


def run_heir(host, port, layer_num):
    from sglang.srt.mem_cache.handover import decode_arm
    from sglang.srt.mem_cache.handover.decode_arm import (
        alloc_heir_decode,
        bytes_to_decode_export,
        import_decode,
    )
    from sglang.srt.mem_cache.handover.transfer import HandoverNixlAgent
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    deadline = time.monotonic() + 180
    sock = None
    while sock is None:
        try:
            sock = socket.create_connection((host, port), timeout=60)
        except ConnectionRefusedError:
            if time.monotonic() > deadline:
                raise TimeoutError("old side never came up")
            time.sleep(1.0)
    sock.settimeout(300)
    info = recv_msg(sock)
    n_pages = info["num_pages"]
    print(f"[heir] old side: {n_pages} pages, {info['layer_num']} layers", flush=True)

    tree, coord, pool, allocator = make_decode_stack(52, layer_num=info["layer_num"])
    logical_slots, host_slots = alloc_heir_decode(coord, n_pages * PAGE, PAGE)
    dst_logical = (logical_slots.cpu().view(-1, PAGE)[:, 0] // PAGE).to(torch.int64).numpy()
    dst_host = (host_slots.cpu().view(-1, PAGE)[:, 0] // PAGE).to(torch.int64).numpy()

    agent = HandoverNixlAgent(backend="UCCL", gpu_id=0, cxi_device_index=0)
    host_ptrs, host_lens, _ = coord.mem_pool_host.get_contiguous_buf_infos()
    agent.register("dst_host", list(zip(host_ptrs, host_lens)), "DRAM")
    idx_ptrs = [int(b.data_ptr()) for b in pool.index_k_with_scale_buffer]
    idx_lens = [b.numel() * b.element_size() for b in pool.index_k_with_scale_buffer]
    agent.register("dst_index", list(zip(idx_ptrs, idx_lens)), "VRAM")
    mbuf = torch.empty(info["manifest_len"], dtype=torch.uint8).pin_memory()
    agent.register("dst_manifest", [(mbuf.data_ptr(), mbuf.numel())], "DRAM")

    t0 = time.perf_counter()
    send_msg(sock, {
        "agent_metadata": base64.b64encode(agent.metadata()).decode(),
        "agent_name": agent.agent.name,
        "host": {"ptrs": [int(p) for p in host_ptrs],
                 "dst_pages": [int(x) for x in dst_host]},
        "index": {"ptrs": idx_ptrs,
                  "dst_pages": [int(x) for x in dst_logical]},
        "manifest": {"ptr": int(mbuf.data_ptr()), "len": int(mbuf.numel())},
    })
    seen = set()
    deadline = time.monotonic() + 300
    while len(seen) < 3:
        for _p, msgs in agent.poll_notifications().items():
            for m in msgs:
                seen.add(m.decode())
        if len(seen) >= 3:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"notifs {seen}")
        time.sleep(0.002)
    t_notif = time.perf_counter() - t0
    rep = recv_msg(sock)
    print(f"[heir] notifs {sorted(seen)} after {t_notif*1e3:.0f}ms; "
          f"old: {rep['bytes']/1e9:.3f}GB in {rep['t_wire_s']*1e3:.0f}ms "
          f"({rep['bytes']/1e9/rep['t_wire_s']:.2f} GB/s)", flush=True)

    ok = True
    got_host = page_ck(coord.mem_pool_host.data_refs, dst_host, info["host_item_len"])
    got_index = page_ck(pool.index_k_with_scale_buffer, dst_logical, info["index_item_len"])
    bad_h = int((got_host != np.array(info["ck_host"], dtype=np.int64)).sum())
    bad_i = int((got_index != np.array(info["ck_index"], dtype=np.int64)).sum())
    print(f"[heir] checksums: host {bad_h}/{n_pages} mismatch, index {bad_i}/{n_pages} mismatch", flush=True)
    if bad_h or bad_i:
        ok = False
    export, psz = bytes_to_decode_export(mbuf.numpy().tobytes())
    stats = import_decode(tree, coord, export, logical_slots, host_slots)
    print(f"[heir] import: {stats}", flush=True)

    # verify: retained prefixes
    rng = np.random.default_rng(11)
    sessions = gen_sessions(200000, rng)
    for s in sessions:
        res = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", list(s)))))
        retained = tree.retained_prefix_len(res.device_indices)
        if retained < len(s) - PAGE:
            print(f"FAIL retained {retained} < {len(s)}")
            ok = False
    print("RESULT:", "PASS" if ok else "FAIL", flush=True)
    agent.close(); sock.close()
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["old", "heir"])
    ap.add_argument("--host")
    ap.add_argument("--port", type=int, default=24474)
    ap.add_argument("--tokens", type=int, default=200000)
    ap.add_argument("--layers", type=int, default=4)
    a = ap.parse_args()
    if a.role == "old":
        import os

        os.environ["WH_MASTER_PORT"] = "24476"
        run_old(a.port, a.tokens, a.layers)
        return 0
    import os

    os.environ["WH_MASTER_PORT"] = "24477"
    assert a.host
    return run_heir(a.host, a.port, a.layers)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
