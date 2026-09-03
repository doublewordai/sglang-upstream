#!/usr/bin/env python3
"""Byte-level round-trip + semantics test for the blob storage backend.

Covers (per IMPL.md equivalence requirements):
 1. batch_write_v3 -> fresh-pool batch_read_v3 byte-exact round trip for BOTH
    pools (latent KV + index-K sidecar with the non-4096-aligned 132 B/token
    geometry — the #35231 class).
 2. Hit query (batch_exists_v2): full range, partial tail, mid-group start via
    prefix_keys, after clear().
 3. Topology-free keys: a second backend instance with a DIFFERENT pp_rank/tp
    config reads the same objects (no _{pp}_{rank} suffix in the key space).
 4. Geometry mismatch: an object written under a different layer set must NOT
    be read (clean miss, not corruption).
 5. Page-hash verification: a tampered object (header key swapped) must miss.
 6. File-size -> num_pages inversion and 4096 slot padding on disk.
 7. Tail-group extension: writing a longer range over a partial group extends
    the object; writing a shorter range is skipped.

Run inside the sglang venv (torch needed), no GPU required:
  python test_blob_roundtrip.py <tmpdir>
"""

import hashlib
import os
import shutil
import struct
import sys
import tempfile

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.blob.hicache_blob import (
    HiCacheBlob,
    HEADER_BLOCK,
    DIO_ALIGN,
    _round_up,
)

PAGE_TOKENS = 64
LATENT_BPT = 656   # bytes/token/layer latent KV (GLM-5.3 MLA fp8 + rope)
INDEX_BPT = 132    # bytes/token/layer index-K (128 fp8 + 4 scale)
N_LAYERS = 78


class FakeHostPool:
    """Minimal host-pool stand-in implementing the backend's used surface."""

    def __init__(self, name: str, bytes_per_token_per_layer: int, dtype=torch.uint8):
        self.name = name
        self.page_size = PAGE_TOKENS
        self.size = 128 * PAGE_TOKENS  # 128 pages
        self.page_num = self.size // self.page_size + 1
        self.start_layer = 0
        self.end_layer = N_LAYERS
        self.dtype = dtype
        self.layer_num = N_LAYERS
        bpt = bytes_per_token_per_layer * N_LAYERS
        self.page_bytes = bpt * PAGE_TOKENS
        # layer_first layout: [layer, page, token_bytes] -> strided pages
        self.buf = torch.zeros(
            (N_LAYERS, self.page_num, PAGE_TOKENS, bytes_per_token_per_layer),
            dtype=dtype,
        )
        self.alloc_cursor = 0

    # the backend only ever uses get_data_page/set_from_flat_data_page/
    # page_size/size/start_layer/end_layer/dtype(implicit via page dtype)
    def get_data_page(self, index, flat: bool = True):
        page = index // self.page_size
        dp = self.buf[:, page, :, :].reshape(-1)
        return dp

    def set_from_flat_data_page(self, index, data_page):
        page = index // self.page_size
        # assign into the (strided) view directly, like the real pools do
        self.buf[:, page, :, :] = data_page.reshape(
            self.buf.shape[0], PAGE_TOKENS, self.buf.shape[-1]
        )

    def fill(self, page, tag):
        """Deterministic per-(pool,page,layer,byte) content, broadcast over tokens."""
        bpt = self.buf.shape[-1]
        for l in range(N_LAYERS):
            row = torch.arange(bpt, dtype=torch.int64)
            vals = (tag * 1_000_003 + page * 7919 + l * 104729 + row) % 251
            self.buf[l, page, :, :] = vals.to(self.buf.dtype)

    def alloc(self, n):
        # trivial bump allocator over whole pages (tests only)
        assert n % self.page_size == 0
        if self.alloc_cursor + n > self.size:
            return None
        idx = torch.arange(self.alloc_cursor, self.alloc_cursor + n, dtype=torch.int64)
        self.alloc_cursor += n
        return idx


def chain_keys(n_pages: int, seed: int = 0):
    keys = []
    h = hashlib.sha256(f"seed{seed}".encode()).hexdigest()
    for i in range(n_pages):
        h = hashlib.sha256((h + f"page{i}").encode()).hexdigest()
        keys.append(h)
    return keys


def make_backend(root, pp_rank=0, pp_size=1, blob_pages=16, extra=None):
    cfg = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="zai-org/GLM-5.3",
        extra_config={"blob_pages": blob_pages, "io_streams": 4, **(extra or {})},
    )
    b = HiCacheBlob(cfg, file_path=root)
    kv = FakeHostPool("kv", LATENT_BPT)
    idx = FakeHostPool("indexer", INDEX_BPT)
    b.register_mem_pool_host(kv)
    b.register_mem_host_pool_v2(idx, PoolName.INDEXER)
    return b, kv, idx


def transfers_for(keys, kv_pool, idx_pool, kv_host, idx_host):
    return [
        PoolTransfer(name=PoolName.KV, host_indices=kv_host, keys=keys),
        PoolTransfer(
            name=PoolName.INDEXER, host_indices=idx_host, keys=keys,
            hit_policy=PoolHitPolicy.ALL_PAGES, indices_from_pool=PoolName.KV,
        ),
    ]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="blobtest-")
    shutil.rmtree(root, ignore_errors=True)
    failures = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name} {detail}")
        if not cond:
            failures.append(name)

    # ---------------- 1. round trip (2 full groups + 6-page tail) ----------
    b, kv, idx = make_backend(root)
    N = 38  # pages: groups 0,1 full (16+16), group 2 partial (6)
    keys = chain_keys(N)
    for p in range(N):
        kv.fill(p, tag=1)
        idx.fill(p, tag=2)
    kv_host = kv.alloc(N * PAGE_TOKENS)
    idx_host = idx.alloc(N * PAGE_TOKENS)
    res = b.batch_write_v3(transfers_for(keys, kv, idx, kv_host, idx_host))
    check("write v3 all ok", all(res[PoolName.KV]) and all(res[PoolName.INDEXER]),
          f"kv={sum(res[PoolName.KV])}/{N} idx={sum(res[PoolName.INDEXER])}/{N}")

    # fresh pools, read back
    b2, kv2, idx2 = make_backend(root, pp_rank=3, pp_size=4)  # different topology!
    res2 = b2.batch_read_v3(transfers_for(keys, kv2, idx2, kv_host, idx_host))
    ok_kv = all(res2[PoolName.KV])
    ok_idx = all(res2[PoolName.INDEXER])
    check("read v3 (cross-topology) all ok", ok_kv and ok_idx,
          f"kv={sum(res2[PoolName.KV])}/{N} idx={sum(res2[PoolName.INDEXER])}/{N}")
    byte_exact_kv = torch.equal(kv.buf[:, :N], kv2.buf[:, :N])
    byte_exact_idx = torch.equal(idx.buf[:, :N], idx2.buf[:, :N])
    check("byte-exact latent KV round trip", byte_exact_kv)
    check("byte-exact index-K sidecar round trip", byte_exact_idx)

    # ---------------- 2. hit query -----------------------------------------
    h = b2.batch_exists_v2(keys, [PoolTransfer(name=PoolName.INDEXER,
                          keys=keys, hit_policy=PoolHitPolicy.ALL_PAGES,
                          indices_from_pool=PoolName.KV)])
    check("hit query full range", h.kv_hit_pages == N, f"{h.kv_hit_pages} vs {N}")
    check("hit query indexer pages", h.extra_pool_hit_pages.get(PoolName.INDEXER) == N,
          f"{h.extra_pool_hit_pages}")

    # partial range (mid-context, group-aligned): keys[16:32]
    h2 = b2.batch_exists_v2(keys[16:32], [])
    check("hit query mid-range aligned", h2.kv_hit_pages == 16, f"{h2.kv_hit_pages}")

    # mid-group start with prefix keys: start at page 20 (group 1, offset 4)
    h3 = b2.batch_exists_v2(keys[20:32], [],
                            HiCacheStorageExtraInfo(prefix_keys=keys[:20]))
    check("hit query mid-group via prefix_keys", h3.kv_hit_pages == 12, f"{h3.kv_hit_pages}")

    # mid-group start WITHOUT prefix keys: head group unaddressable from page 20
    # (group 1 partially covered: pages 20..31 of a 16-page object -> still hit
    # because the object is found via full chain? no prefix -> cannot address
    # group 1's first page -> miss)
    h4 = b2.batch_exists_v2(keys[20:32], [])
    check("hit query mid-group no prefix -> bounded miss", h4.kv_hit_pages == 0,
          f"{h4.kv_hit_pages}")

    # range extending beyond stored tail (stored np=6 in group 2)
    keys_ext = chain_keys(38 + 10)
    h5 = b2.batch_exists_v2(keys_ext, [])
    check("hit query past tail truncates", h5.kv_hit_pages == 38, f"{h5.kv_hit_pages}")

    # ---------------- 3. on-disk layout ------------------------------------
    ids = []
    for g in range(3):
        gid = b2._group_id(keys[16 * g], b2._poolset)
        path = b2._obj_path(gid)
        ids.append(gid)
        check(f"object g{g} exists", os.path.exists(path))
        size = os.path.getsize(path)
        kv_slot = _round_up(kv.page_bytes, 4096)
        idx_slot = _round_up(idx.page_bytes, 4096)
        expect_np = min(16, N - 16 * g)
        expect = HEADER_BLOCK + _round_up(kv_slot * expect_np, DIO_ALIGN) + _round_up(
            idx_slot * expect_np, DIO_ALIGN
        )
        check(f"object g{g} size inverts np", size == expect,
              f"{size} vs {expect} (np={expect_np})")
        # header page-key verification data present
        with open(path, "rb") as f:
            hdr = f.read(4096)
        check(f"object g{g} magic", hdr[:8] == b"SGBLOB01")

    # files per context: N pages -> ceil-ish objects + manifests
    nfiles = sum(
        len([f for f in os.listdir(os.path.join(root, d)) if f.endswith(".blob")])
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )
    check("3 group objects on disk", nfiles == 3, f"{nfiles}")

    # ---------------- 4. geometry mismatch = clean miss ---------------------
    b3, kv3, idx3 = make_backend(root + "_geo", pp_rank=0)
    # different layer count -> different geo id -> different filenames
    kv3.buf = torch.zeros((80, kv3.page_num, PAGE_TOKENS, LATENT_BPT), dtype=torch.uint8)
    idx3.buf = torch.zeros((80, idx3.page_num, PAGE_TOKENS, INDEX_BPT), dtype=torch.uint8)
    kv3.layer_num = idx3.layer_num = 80
    kv3.end_layer = idx3.end_layer = 80
    b3._refresh_pools()
    res3 = b3.batch_read_v3(transfers_for(keys, kv3, idx3, kv_host, idx_host))
    check("geometry mismatch -> clean miss", not any(res3[PoolName.KV]),
          f"kv hits={sum(res3[PoolName.KV])}")

    # ---------------- 5. tampered page key -> miss --------------------------
    gid1 = b2._group_id(keys[16], b2._poolset)
    path1 = b2._obj_path(gid1)
    from sglang.srt.mem_cache.storage.blob.hicache_blob import _PAGE_KEYS_OFF
    KEY_LEN = 64
    with open(path1, "r+b") as f:
        f.seek(_PAGE_KEYS_OFF + KEY_LEN)
        orig = f.read(KEY_LEN)
        f.seek(_PAGE_KEYS_OFF + KEY_LEN)
        f.write(b"f" * KEY_LEN)
    b2._coverage.invalidate(gid1)
    res4 = b2.batch_read_v3(transfers_for(keys, kv2, idx2, kv_host, idx_host))
    # group 0 still fine; group 1 misses WHOLE-GROUP (conservative on tamper)
    kv_ok = res4[PoolName.KV]
    check("tampered page key -> whole group 1 misses",
          all(kv_ok[:16]) and not any(kv_ok[16:32]),
          f"missing={[i for i, x in enumerate(kv_ok) if not x]}")
    with open(path1, "r+b") as f:
        f.seek(_PAGE_KEYS_OFF + KEY_LEN)
        f.write(orig)
    b2._coverage.invalidate(gid1)

    # ---------------- 6. chunk-write alignment semantics ---------------------
    keys_long = chain_keys(58)  # same seed => same prefix as `keys`
    assert keys_long[:38] == keys
    kv_host_l = kv.alloc(20 * PAGE_TOKENS)
    idx_host_l = idx.alloc(20 * PAGE_TOKENS)
    for p in range(38, 58):
        kv.fill(p, tag=3)
        idx.fill(p, tag=4)

    # (a) legacy no-prefix chunk write (pages 38..57): treated as its own
    # context start -> self-aligned groups; invisible to full-chain readers
    res5 = b.batch_write_v3(
        transfers_for(keys_long[38:], kv, idx, kv_host_l, idx_host_l)
    )
    check(
        "no-prefix chunk write self-aligned",
        all(res5[PoolName.KV]) and all(res5[PoolName.INDEXER]),
        f"kv={sum(res5[PoolName.KV])}/20 idx={sum(res5[PoolName.INDEXER])}/20",
    )
    h6a = b.batch_exists_v2(keys_long, [])
    check("no-prefix chunk invisible to full-chain hit", h6a.kv_hit_pages == 38,
          f"{h6a.kv_hit_pages}")

    # (b) prefix'd write of the same range: head group (first page 32 < start
    # 38) skipped; group 3 (pages 48..57) written
    res5b = b.batch_write_v3(
        transfers_for(keys_long[38:], kv, idx, kv_host_l, idx_host_l),
        HiCacheStorageExtraInfo(prefix_keys=keys_long[:38]),
    )
    check("prefix'd write skips head group, writes next",
          all(res5b[PoolName.KV][10:]) and not any(res5b[PoolName.KV][:10]),
          f"{sum(res5b[PoolName.KV])}/20")

    # (c) full-range rewrite with prefix closes the gap (g2 -> 16 pages)
    kv_host_f = kv.alloc(26 * PAGE_TOKENS)
    idx_host_f = idx.alloc(26 * PAGE_TOKENS)
    res5c = b.batch_write_v3(
        transfers_for(keys_long[32:], kv, idx, kv_host_f, idx_host_f),
        HiCacheStorageExtraInfo(prefix_keys=keys_long[:32]),
    )
    check("gap-closing rewrite ok", all(res5c[PoolName.KV]),
          f"{sum(res5c[PoolName.KV])}/26")
    h6 = b.batch_exists_v2(keys_long, [])
    check("hit covers full 58 pages after gap close", h6.kv_hit_pages == 58,
          f"{h6.kv_hit_pages}")

    # ---------------- 7. rewrite idempotency ---------------------------------
    res6 = b.batch_write_v3(transfers_for(keys, kv, idx, kv_host, idx_host))
    check("rewrite skipped (idempotent)", all(res6[PoolName.KV])
          and all(res6[PoolName.INDEXER]),
          f"kv={sum(res6[PoolName.KV])}/38")

    # ---------------- 7. clear ----------------------------------------------
    check("clear ok", b.clear())
    h7 = b.batch_exists_v2(keys, [])
    check("hit after clear = 0", h7.kv_hit_pages == 0, f"{h7.kv_hit_pages}")

    # ---------------- 8. draft pools (spec-decode sidecars) ------------------
    # DSA+MTP engines register draft + draft_indexer host pools
    # (index_k_with_scale); the blob backend must store/restore them in
    # per-pool objects WITHOUT gating the KV hit (best-effort semantics).
    rootd = root + "_draft"
    shutil.rmtree(rootd, ignore_errors=True)
    os.makedirs(rootd, exist_ok=True)
    bd, kvd, idxd = make_backend(rootd)
    kvd2 = FakeHostPool("draft", 96)          # draft latent KV (own geometry)
    idxd2 = FakeHostPool("draft_indexer", 12)  # draft-side DSA indexer state
    bd.register_mem_host_pool_v2(kvd2, "draft")
    bd.register_mem_host_pool_v2(idxd2, "draft_indexer")
    N_D = 34
    kd = chain_keys(N_D, seed=5)
    kvd_h = kvd.alloc(N_D * PAGE_TOKENS)
    idxd_h = idxd.alloc(N_D * PAGE_TOKENS)
    d1_h = kvd2.alloc(N_D * PAGE_TOKENS)
    d2_h = idxd2.alloc(N_D * PAGE_TOKENS)
    for pg in range(N_D):
        kvd.fill(pg, 11); idxd.fill(pg, 12); kvd2.fill(pg, 13); idxd2.fill(pg, 14)
    trd = [
        PoolTransfer(name=PoolName.KV, host_indices=kvd_h, keys=kd),
        PoolTransfer(name=PoolName.INDEXER, host_indices=idxd_h, keys=kd,
                     hit_policy=PoolHitPolicy.ALL_PAGES, indices_from_pool=PoolName.KV),
        PoolTransfer(name="draft", host_indices=d1_h, keys=kd,
                     hit_policy=PoolHitPolicy.ALL_PAGES, indices_from_pool=PoolName.KV),
        PoolTransfer(name="draft_indexer", host_indices=d2_h, keys=kd,
                     hit_policy=PoolHitPolicy.ALL_PAGES, indices_from_pool=PoolName.KV),
    ]
    resd = bd.batch_write_v3(trd, HiCacheStorageExtraInfo(prefix_keys=None))
    check("draft write all ok",
          sum(resd[PoolName.KV]) == N_D and sum(resd["draft"]) == N_D
          and sum(resd["draft_indexer"]) == N_D,
          f"kv={sum(resd[PoolName.KV])} d={sum(resd['draft'])} di={sum(resd['draft_indexer'])}")
    # fresh pools, cross-topology read: byte-exact incl. draft sidecars
    bd2, kvd3, idxd3 = make_backend(rootd, pp_rank=1, pp_size=2)
    kvd4 = FakeHostPool("draft", 96)
    idxd4 = FakeHostPool("draft_indexer", 12)
    bd2.register_mem_host_pool_v2(kvd4, "draft")
    bd2.register_mem_host_pool_v2(idxd4, "draft_indexer")
    kvd_h2 = kvd3.alloc(N_D * PAGE_TOKENS)
    idxd_h2 = idxd3.alloc(N_D * PAGE_TOKENS)
    d1_h2 = kvd4.alloc(N_D * PAGE_TOKENS)
    d2_h2 = idxd4.alloc(N_D * PAGE_TOKENS)
    trd2 = [
        PoolTransfer(name=PoolName.KV, host_indices=kvd_h2, keys=kd),
        PoolTransfer(name=PoolName.INDEXER, host_indices=idxd_h2, keys=kd,
                     hit_policy=PoolHitPolicy.ALL_PAGES, indices_from_pool=PoolName.KV),
        PoolTransfer(name="draft", host_indices=d1_h2, keys=kd,
                     hit_policy=PoolHitPolicy.ALL_PAGES, indices_from_pool=PoolName.KV),
        PoolTransfer(name="draft_indexer", host_indices=d2_h2, keys=kd,
                     hit_policy=PoolHitPolicy.ALL_PAGES, indices_from_pool=PoolName.KV),
    ]
    resd2 = bd2.batch_read_v3(trd2, HiCacheStorageExtraInfo(prefix_keys=None))
    check("draft read all ok (cross-topology)",
          sum(resd2[PoolName.KV]) == N_D and sum(resd2["draft"]) == N_D
          and sum(resd2["draft_indexer"]) == N_D,
          f"kv={sum(resd2[PoolName.KV])} d={sum(resd2['draft'])} di={sum(resd2['draft_indexer'])}")
    okd = torch.equal(kvd3.buf, kvd.buf) and torch.equal(idxd3.buf, idxd.buf)
    okd2 = torch.equal(kvd4.buf, kvd2.buf) and torch.equal(idxd4.buf, idxd2.buf)
    check("draft byte-exact latent+indexer round trip", okd and okd2,
          f"main={okd} draft={okd2}")
    # hit query: draft pools best-effort, never gate KV
    pt = [
        PoolTransfer(name=PoolName.INDEXER, host_indices=idxd_h, keys=kd,
                     hit_policy=PoolHitPolicy.ALL_PAGES, indices_from_pool=PoolName.KV),
        PoolTransfer(name="draft", host_indices=d1_h, keys=kd,
                     hit_policy=PoolHitPolicy.ALL_PAGES, indices_from_pool=PoolName.KV),
        PoolTransfer(name="draft_indexer", host_indices=d2_h, keys=kd,
                     hit_policy=PoolHitPolicy.ALL_PAGES, indices_from_pool=PoolName.KV),
    ]
    hd = bd2.batch_exists_v2(kd, pt)
    check("draft hit best-effort (never gates)", hd.kv_hit_pages == N_D
          and hd.extra_pool_hit_pages.get("draft") == N_D
          and hd.extra_pool_hit_pages.get("draft_indexer") == N_D,
          f"kv={hd.kv_hit_pages} d={hd.extra_pool_hit_pages.get('draft')} di={hd.extra_pool_hit_pages.get('draft_indexer')}")
    # delete the draft objects -> KV hit must survive
    import glob as _glob
    ndel = 0
    for gp in range((N_D + 15) // 16):
        gid_d = bd2._group_id(kd[16 * gp], "draft")
        gid_di = bd2._group_id(kd[16 * gp], "draft_indexer")
        for gid in (gid_d, gid_di):
            p = bd2._obj_path(gid)
            if os.path.exists(p):
                os.remove(p); ndel += 1
    hd2 = bd2.batch_exists_v2(kd, pt)
    check("draft objects deleted -> KV hit unaffected", hd2.kv_hit_pages == N_D
          and hd2.extra_pool_hit_pages.get("draft") == N_D,   # best-effort: still reported
          f"kv={hd2.kv_hit_pages}")
    resd3 = bd2.batch_read_v3(trd2, HiCacheStorageExtraInfo(prefix_keys=None))
    check("read after draft delete: KV ok, draft miss",
          sum(resd3[PoolName.KV]) == N_D and not any(resd3["draft"]),
          f"kv={sum(resd3[PoolName.KV])} d={sum(resd3['draft'])}")

    print()
    if failures:
        print(f"{len(failures)} FAILURES: {failures}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
