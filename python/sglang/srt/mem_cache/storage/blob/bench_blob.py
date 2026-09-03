#!/usr/bin/env python3
"""Microbench: blob backend write/read throughput at GLM-5.3 rig geometry.

Geometry: latent 656 B/token/layer + index-K 132 B/token/layer, 78 layers,
page 64 tokens -> group object (16 pages = 1024 tokens) = 63,045,632 B on disk.
Pool memory: 1024 host pages (64k tokens) = 4.0 GB latent + 0.67 GB index.

Sweeps: io_streams in {1,4,8,16}; O_DIRECT vs buffered (scout ask: how much of
retrieval time is the filesystem-client layer). Writes use fresh keys per
timed iteration (no skip path). Buffered objects are fsync'd + fadvise'd
DONTNEED before reading so iter-1 is page-cache-cold; iter-2+ report the
warm-cache ceiling. O_DIRECT reads are cold by construction.

Usage: python bench_blob.py <benchdir> [n_groups]
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import torch

sys.path.insert(0, "/scratch/s6p/fergus.s6p/src/sglang-blob-backend-0902/python")
from sglang.srt.mem_cache.hicache_storage import (  # noqa: E402
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.blob.hicache_blob import HiCacheBlob  # noqa: E402

PAGE = 64
GROUP_PAGES = 16
N_LAYERS = 78
BPT_KV = 656
BPT_IDX = 132
POOL_PAGES = 1024
OBJ_BYTES = 63_045_632  # measured on-disk group object size (np=16)


class BenchPool:
    def __init__(self, bpt, dtype=torch.uint8):
        self.page_size = PAGE
        self.size = POOL_PAGES * PAGE
        self.page_num = POOL_PAGES + 1
        self.start_layer = 0
        self.end_layer = N_LAYERS
        self.layer_num = N_LAYERS
        self.dtype = dtype
        self.buf = torch.zeros((N_LAYERS, self.page_num, PAGE, bpt), dtype=dtype)
        self.buf.random_(0, 251)

    def get_data_page(self, index, flat=True):
        return self.buf[:, index // self.page_size, :, :].reshape(-1)

    def set_from_flat_data_page(self, index, data_page):
        self.buf[:, index // self.page_size, :, :] = data_page.reshape(
            self.buf.shape[0], PAGE, self.buf.shape[-1]
        )


def chain_keys(n_pages, seed):
    keys = []
    h = hashlib.sha256(f"bench{seed}".encode()).hexdigest()
    for i in range(n_pages):
        h = hashlib.sha256((h + f"p{i}").encode()).hexdigest()
        keys.append(h)
    return keys


def make_backend(root, streams, dio, kv_pool, idx_pool):
    cfg = HiCacheStorageConfig(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
        is_mla_model=True, enable_storage_metrics=False, is_page_first_layout=False,
        model_name="zai-org/GLM-5.3",
        extra_config={"blob_pages": GROUP_PAGES, "io_streams": streams,
                      "direct_io": dio, "write_manifest": False},
    )
    b = HiCacheBlob(cfg, file_path=root)
    b.register_mem_pool_host(kv_pool)
    b.register_mem_host_pool_v2(idx_pool, PoolName.INDEXER)
    return b


def transfers_for(keys):
    n = len(keys)
    return [
        PoolTransfer(name=PoolName.KV,
                     host_indices=torch.arange(n * PAGE, dtype=torch.int64),
                     keys=keys),
        PoolTransfer(name=PoolName.INDEXER,
                     host_indices=torch.arange(n * PAGE, dtype=torch.int64),
                     keys=keys),
    ]


def cool_page_cache(bdir):
    """fsync + POSIX_FADV_DONTNEED every object (best-effort page-cache cool)."""
    for d in os.listdir(bdir):
        sub = os.path.join(bdir, d)
        if not os.path.isdir(sub):
            continue
        for fn in os.listdir(sub):
            if not fn.endswith(".blob"):
                continue
            p = os.path.join(sub, fn)
            fd = os.open(p, os.O_RDONLY)
            try:
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


if __name__ == "__main__":
    root = sys.argv[1]
    n_groups = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    n_pages = n_groups * GROUP_PAGES

    print("== lsscsi (local disks, for the record) ==")
    try:
        out = subprocess.run(["lsscsi"], capture_output=True, text=True, timeout=20).stdout
        print(out.strip() or "(none)")
    except Exception as e:
        print("lsscsi failed:", e)

    kv_pool = BenchPool(BPT_KV)
    idx_pool = BenchPool(BPT_IDX)
    print(f"pools: kv {kv_pool.buf.numel()/1e9:.2f} GB, idx {idx_pool.buf.numel()/1e9:.2f} GB; "
          f"{n_groups} groups/iter = {n_groups * OBJ_BYTES / 1e9:.2f} GB")

    results = []
    for dio in (True, False):
        for streams in (1, 4, 8, 16):
            tag = f"{'dio' if dio else 'buf'}-s{streams}"
            bdir = os.path.join(root, tag)
            shutil.rmtree(bdir, ignore_errors=True)
            b = make_backend(bdir, streams, dio, kv_pool, idx_pool)

            # writes: fresh seed per iteration (never the skip path)
            w_ts = []
            for it in range(3):
                keys = chain_keys(n_pages, seed=1000 * it + streams + (0 if dio else 100))
                tr = transfers_for(keys)
                t0 = time.perf_counter()
                r = b.batch_write_v3(tr)
                w_ts.append(time.perf_counter() - t0)
                assert all(r[PoolName.KV]) and all(r[PoolName.INDEXER]), f"write fail {tag}"
            w_ts.sort()
            w_p50 = w_ts[1]

            # read target: the last-written seed
            keys = chain_keys(n_pages, seed=2000 + streams + (0 if dio else 100))
            r2 = b.batch_write_v3(transfers_for(keys))
            assert all(r2[PoolName.KV])
            if not dio:
                cool_page_cache(bdir)

            b._coverage.clear()
            r_ts = []
            for it in range(21):
                tr = transfers_for(keys)
                t0 = time.perf_counter()
                rr = b.batch_read_v3(tr)
                r_ts.append(time.perf_counter() - t0)
                assert all(rr[PoolName.KV]) and all(rr[PoolName.INDEXER]), (
                    f"read miss {tag} iter {it}")
            r_cold = r_ts[0]
            rest = sorted(r_ts[1:])
            r_p50 = rest[len(rest) // 2]
            r_p90 = rest[int(len(rest) * 0.9) - 1]

            row = {
                "config": tag, "direct_io": dio, "io_streams": streams,
                "n_groups": n_groups, "obj_bytes": OBJ_BYTES,
                "write_GBps_p50": round(n_groups * OBJ_BYTES / w_p50 / 1e9, 2),
                "write_s_p50": round(w_p50, 3),
                "read_cold_GBps": round(n_groups * OBJ_BYTES / r_cold / 1e9, 2),
                "read_GBps_p50": round(n_groups * OBJ_BYTES / r_p50 / 1e9, 2),
                "read_GBps_p90": round(n_groups * OBJ_BYTES / r_p90 / 1e9, 2),
                "read_iters": 21,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
            b.clear()
            shutil.rmtree(bdir, ignore_errors=True)

    print("\n== summary (decimal GB/s per whole object incl. header) ==")
    print(f"{'config':>10} {'write p50':>10} {'read cold':>10} {'read p50':>10} {'read p90':>10}")
    for r in results:
        print(f"{r['config']:>10} {r['write_GBps_p50']:>10.2f} {r['read_cold_GBps']:>10.2f} "
              f"{r['read_GBps_p50']:>10.2f} {r['read_GBps_p90']:>10.2f}")
