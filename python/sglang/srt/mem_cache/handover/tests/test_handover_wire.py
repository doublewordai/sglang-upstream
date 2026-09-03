"""Wire test: prefill-arm handover push over NIXL/UCCL between two nodes.

Old side (engine A stand-in) on node 1: serves sessions, exports (direct
mode), answers a control socket: phase 1 reports size/fingerprint, phase 2
pushes page bytes + manifest into the heir's registered buffers with NIXL
WRITEs (old-initiated, the shape UCCL supports), waits handles DONE.

Heir (engine B stand-in) on node 2: allocates destination rows + a manifest
buffer, registers them with its own NIXL agent, sends its agent metadata +
buffer specs + destination page mapping, waits for the completion notifs,
then deserializes, checksum-verifies, imports, and verifies turn-2 hits.

Usage (two shells / srun steps):
  python test_handover_wire.py --role old  --port 90000 [--tokens 300000]
  python test_handover_wire.py --role heir --host <node1-ip> --port 90000
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
import os
torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))  # cgroup-aware: default sees all 288 cores -> OMP thrash


# ---------------------------------------------------------------------------
# shared stack builder (same as unit test)
# ---------------------------------------------------------------------------


def make_stack(size_tokens, layer_num, hicache_ratio, seed, model_path="dummy"):
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # lane-local ports (2446x block); the node default 295xx collides with
    # other lanes' sglang engines sharing the holder job
    os.environ.setdefault("MASTER_PORT", os.environ.get("WH_MASTER_PORT", "24466"))
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

    dev_idx = allocator.alloc(len(tokens)).to(torch.int64)
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
    tree.insert(
        InsertParams(key=RadixKey(array("q", list(tokens))), value=dev_idx, priority=0)
    )
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
    sessions, made = [], 0
    group_prefix = list(rng.integers(1000, 50000, size=32 * PAGE))
    while made < total_tokens:
        n_pages = int(rng.integers(32, 96))
        if rng.random() < 0.6:
            toks = group_prefix + list(rng.integers(1000, 50000, size=n_pages * PAGE))
        else:
            toks = list(rng.integers(1000, 50000, size=(n_pages + 32) * PAGE))
        sessions.append(toks)
        made += len(toks)
    return sessions


# ---------------------------------------------------------------------------
# control channel helpers
# ---------------------------------------------------------------------------


def send_msg(sock, obj):
    b = json.dumps(obj).encode()
    sock.sendall(len(b).to_bytes(8, "little") + b)


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf


def recv_msg(sock):
    n = int.from_bytes(recv_exact(sock, 8), "little")
    return json.loads(recv_exact(sock, n))


# ---------------------------------------------------------------------------
# old side
# ---------------------------------------------------------------------------


def run_old(port, n_tokens, layer_num):
    import os

    os.environ["WH_MASTER_PORT"] = "24466"
    from sglang.srt.mem_cache.handover import prefill_arm
    from sglang.srt.mem_cache.handover.transfer import (
        HandoverNixlAgent,
        build_pool_descriptors,
    )
    from sglang.srt.mem_cache.handover.manifest import manifest_to_bytes

    device_tokens = 65536
    hicache_ratio = max(2, (n_tokens * 2) // device_tokens)
    tree, pool, allocator = make_stack(device_tokens, layer_num, hicache_ratio, 21)
    rng = np.random.default_rng(5)
    sessions = gen_sessions(n_tokens, rng)
    for s in sessions:
        serve_and_evict(tree, pool, allocator, s, layer_num)
    print(f"[old] served {len(sessions)} sessions, target {n_tokens} tokens", flush=True)

    t0 = time.perf_counter()
    export = prefill_arm.PrefillExport.build(
        tree, "dummy", staged=False, with_checksums=True
    )
    t_export = time.perf_counter() - t0
    m = export.manifest
    mbytes = manifest_to_bytes(m)
    print(
        f"[old] export {t_export*1e3:.0f}ms chains={len(m.chains)} tokens={m.num_tokens} "
        f"pages={m.num_pages} manifest={len(mbytes)/1e6:.2f}MB",
        flush=True,
    )

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"[old] listening on {port}", flush=True)
    conn, addr = srv.accept()
    print(f"[old] heir connected from {addr}", flush=True)

    # phase 1: sizes
    send_msg(
        conn,
        {
            "num_tokens": m.num_tokens,
            "num_pages": m.num_pages,
            "manifest_len": len(mbytes),
            "fingerprint": m.fingerprint,
            "page_size": m.page_size,
        },
    )
    req = recv_msg(conn)  # phase 2: heir buffers
    assert req["fingerprint"] == m.fingerprint, "fingerprint mismatch"

    agent = HandoverNixlAgent(backend="UCCL", gpu_id=0, cxi_device_index=0)
    # register source pool buffers (direct mode)
    reg = []
    for name, spec in export.pool_specs.items():
        reg.append(
            agent.register(
                f"src_{name}",
                list(zip(spec.data_ptrs, spec.data_lens)),
                "DRAM",
            )
        )
    manifest_src = torch.frombuffer(
        bytearray(mbytes), dtype=torch.uint8
    ).clone()  # writable pinned-ish copy
    manifest_src_pin = manifest_src.pin_memory()
    agent.register("src_manifest", [(manifest_src_pin.data_ptr(), len(mbytes))], "DRAM")
    agent.add_peer(base64.b64decode(req["agent_metadata"]))

    t0 = time.perf_counter()
    handles = []
    total_bytes = 0
    n_desc = 0
    for name, spec in export.pool_specs.items():
        dst = req["pools"][name]
        assert dst["item_len"] == spec.item_len, (
            f"item_len mismatch {name}: {dst['item_len']} vs {spec.item_len}"
        )
        src_addrs, dst_addrs = build_pool_descriptors(
            spec.data_ptrs,
            dst["ptrs"],
            spec.item_len,
            export.src_pages,
            np.array(dst["dst_pages"], dtype=np.int64),
        )
        n_desc += len(src_addrs)
        total_bytes += sum(l for _, l in src_addrs)
        handles.append(
            agent.push_write(
                src_addrs,
                dst_addrs,
                "DRAM",
                "DRAM",
                dst_gpu_id=0,
                peer_name=req["agent_name"],
                notif=f"handover_{name}",
            )
        )
    # manifest push
    mdst = req["manifest"]
    handles.append(
        agent.push_write(
            [(manifest_src_pin.data_ptr(), len(mbytes))],
            [(mdst["ptr"], mdst["len"])],
            "DRAM",
            "DRAM",
            dst_gpu_id=0,
            peer_name=req["agent_name"],
            notif="handover_manifest",
        )
    )
    t_post = time.perf_counter() - t0
    ok = agent.wait(handles, timeout_s=300)
    t_wire = time.perf_counter() - t0
    print(
        f"[old] push done={ok} bytes={total_bytes/1e9:.3f}GB descs={n_desc} "
        f"post={t_post*1e3:.1f}ms wire={t_wire*1e3:.1f}ms "
        f"-> {total_bytes/1e9/t_wire:.2f} GB/s (post-to-done)",
        flush=True,
    )
    send_msg(
        conn,
        {
            "status": "ok" if ok else "timeout",
            "bytes": total_bytes,
            "t_wire_s": t_wire,
            "n_desc": n_desc,
        },
    )
    export.release_protections()
    conn.close()
    srv.close()
    agent.close()
    print("[old] done", flush=True)


# ---------------------------------------------------------------------------
# heir side
# ---------------------------------------------------------------------------


def run_heir(host, port, n_tokens, layer_num):
    import os

    os.environ["WH_MASTER_PORT"] = "24467"
    from sglang.srt.mem_cache.handover import prefill_arm
    from sglang.srt.mem_cache.handover.prefill_arm import (
        _per_layer_host_buffers,
        alloc_heir_rows,
        heir_page_rows,
        tree_pools,
    )
    from sglang.srt.mem_cache.handover.manifest import (
        bytes_to_manifest,
        page_checksums,
    )
    from sglang.srt.mem_cache.handover.transfer import HandoverNixlAgent
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    device_tokens = 65536
    # bounded retry: the old side takes ~40s (stack build + serving) before
    # its control socket listens; connection-refused is instant, so poll.
    deadline = time.monotonic() + 180
    sock = None
    while sock is None:
        try:
            sock = socket.create_connection((host, port), timeout=60)
        except ConnectionRefusedError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"old side {host}:{port} never came up")
            time.sleep(1.0)
    sock.settimeout(300)
    info = recv_msg(sock)  # phase 1
    n_tok, n_pages = info["num_tokens"], info["num_pages"]
    print(f"[heir] old side has {n_tok} tokens / {n_pages} pages", flush=True)

    hicache_ratio = max(2, (n_tok * 2) // device_tokens)
    tree, pool, allocator = make_stack(device_tokens, layer_num, hicache_ratio, 22)
    slots = alloc_heir_rows(tree, n_tok)
    dst_pages = heir_page_rows(slots, PAGE)

    agent = HandoverNixlAgent(backend="UCCL", gpu_id=0, cxi_device_index=0)
    pools = tree_pools(tree)
    pools_msg = {}
    for name, p in pools.items():
        spec_ptrs, spec_lens, items = p.get_contiguous_buf_infos()
        assert len(set(items)) == 1, f"non-uniform item lens in {name}"
        item = int(items[0])
        agent.register(f"dst_{name}", list(zip(spec_ptrs, spec_lens)), "DRAM")
        pools_msg[name] = {
            "ptrs": [int(x) for x in spec_ptrs],
            "item_len": item,
            "dst_pages": [int(x) for x in dst_pages],
        }
    manifest_buf = torch.empty(info["manifest_len"], dtype=torch.uint8).pin_memory()
    agent.register(
        "dst_manifest", [(manifest_buf.data_ptr(), manifest_buf.numel())], "DRAM"
    )

    t_alloc = time.perf_counter()
    send_msg(
        sock,
        {
            "agent_metadata": base64.b64encode(agent.metadata()).decode(),
            "agent_name": agent.agent.name,
            "fingerprint": info["fingerprint"],
            "pools": pools_msg,
            "manifest": {
                "ptr": int(manifest_buf.data_ptr()),
                "len": int(manifest_buf.numel()),
            },
        },
    )

    # wait for our notifications (WRITEs landed)
    deadline = time.monotonic() + 300
    seen = set()
    while len(seen) < len(pools) + 1:
        notifs = agent.poll_notifications()
        for _peer, msgs in notifs.items():
            for msg in msgs:
                seen.add(msg.decode())
        if time.monotonic() > deadline:
            raise TimeoutError(f"only saw notifs {seen}")
        time.sleep(0.002)
    t_notif = time.perf_counter() - t_alloc
    rep = recv_msg(sock)
    print(
        f"[heir] notifs {sorted(seen)} after {t_notif*1e3:.0f}ms; old reports "
        f"{rep['bytes']/1e9:.3f}GB in {rep['t_wire_s']*1e3:.1f}ms "
        f"({rep['bytes']/1e9/rep['t_wire_s']:.2f} GB/s)",
        flush=True,
    )

    m = bytes_to_manifest(manifest_buf.numpy().tobytes())
    assert m.fingerprint == info["fingerprint"]

    # checksum-verify landed rows
    t0 = time.perf_counter()
    for name, want in (m.checksums or {}).items():
        got = page_checksums(
            _per_layer_host_buffers(pools[name]), dst_pages, pools_msg[name]["item_len"]
        )
        bad = int((got != want).sum())
        assert bad == 0, f"checksum mismatch pool {name}: {bad}/{len(want)} pages"
    t_ck = time.perf_counter() - t0

    t0 = time.perf_counter()
    stats = prefill_arm.import_manifest(tree, m, slots, need_hashes=False)
    t_import = time.perf_counter() - t0

    # turn-2 hits
    rng = np.random.default_rng(5)
    sessions = gen_sessions(n_tokens, rng)
    hits = 0
    t0 = time.perf_counter()
    for s in sessions:
        res = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", list(s)))))
        if res.host_hit_length >= len(s) - PAGE:
            hits += 1
    t_match = time.perf_counter() - t0

    print(
        f"[heir] checksum {t_ck*1e3:.0f}ms OK; import {t_import*1e3:.0f}ms "
        f"({stats}); hits {hits}/{len(sessions)} (match {t_match*1e3:.0f}ms total)",
        flush=True,
    )
    ok = hits == len(sessions)
    print("RESULT:", "PASS" if ok else "FAIL", flush=True)
    agent.close()
    sock.close()
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["old", "heir"])
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=90000)
    ap.add_argument("--tokens", type=int, default=300000)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()
    if args.role == "old":
        run_old(args.port, args.tokens, args.layers)
        return 0
    assert args.host, "heir needs --host"
    return run_heir(args.host, args.port, args.tokens, args.layers)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
