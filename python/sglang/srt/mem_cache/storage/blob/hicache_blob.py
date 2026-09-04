# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Large-blob storage backend for the HiCache L3 tier (Lustre-oriented).

Design (lane blob-backend, 2026-09-02):

- ONE object file per group of ``blob_pages`` host-pool pages (default 16 pages
  = 1024 tokens at page_size 64) x the rank's full layer slab. The object
  contains one segment per registered non-draft pool (for DSA: latent KV +
  index-K sidecar), so a single read/write covers every pool of the group.
- Keys are topology-free: filename = sha1(model | first-page chain hash |
  layer set | geometry id). No ``_{pp_size}_{pp_rank}`` suffix, so any arm or
  generation with the same model + layer slab shares objects. The group's
  first-page chain hash pins the context prefix; the header stores every
  page's chain hash and readers verify them before accepting bytes.
- Per-page slots inside each segment are rounded up to 4096-byte multiples
  (the upstream NIXL indexer-pool alignment bug class: non-4096-multiple page
  strides silently corrupt DSA index-K under O_DIRECT). Slot-level padding
  makes file size -> num_pages invertible and every slot 4096-aligned.
- O_DIRECT for data segments (no page-cache churn next to the pinned host
  pools); the header is one 4096 pread. Writes go to a tmp name and are
  published with os.replace.
- An internal thread pool (``io_streams``) parallelizes object IO within one
  batch call; the controller-facing calls stay synchronous.
- Existence/hit queries stat group objects; an in-memory TTL cache remembers
  positives (same pattern as the file backend).
- Eviction sweeps buckets round-robin and enforces a byte budget and/or a
  max age, deleting oldest first.
- Draft pools (spec-decode sidecars) are best-effort and go to separate
  per-pool objects; they never gate the KV hit.

The backend also implements the combined per-op IO protocol "v3"
(``batch_read_v3``/``batch_write_v3``) used by HybridCacheController: one call
with KV + sidecar PoolTransfers, so one object IO serves every pool (the
legacy v1+v2 pair would read the same object twice and stage whole-op sidecar
pages in host memory).
"""

from __future__ import annotations

import hashlib
import json
import logging
import mmap
import os
import struct
import subprocess
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.pool_host.base import HostKVCache

logger = logging.getLogger(__name__)

# kv-unit lane: per-phase timing of the blob read path (DIO read vs host
# scatter), one INFO line per batch. Default off; no behavior change.
_BLOB_PHASE_STATS = os.environ.get("SGLANG_KVU_BLOB_PHASE_STATS", "0") == "1"

MAGIC = b"SGBLOB01"
VERSION = 1
HEADER_SIZE = 4096  # header content size (packed fields + page keys)
# O_DIRECT file offsets must be aligned to the OS page size on this
# filesystem (measured: 4096-aligned offsets fail with EINVAL, 64 KiB-aligned
# work on the aarch64 GH200 nodes; buffer lengths are unconstrained).
DIO_ALIGN = max(getattr(mmap, "PAGESIZE", 4096), 4096)
HEADER_BLOCK = DIO_ALIGN  # on-disk header block (content padded with zeros)
DIO_BLOCK = 4096  # per-page slot padding granularity (size->np inversion)
CHUNK_SIZE = 4 * 1024 * 1024  # O_DIRECT pwrite/pread chunk
MAX_POOLS = 8
MAX_POOLSET_LEN = 128
KEY_HEX_LEN = 64  # sha256 hex page key

# header: magic(8) version(I) page_tokens(I) num_pages(I) blob_pages(I)
#         layer_start(I) layer_end(I) n_pools(I) poolset_len(I)
_HEAD_FIXED = struct.Struct("<8sIIIIIIII")
# per-pool entry: name_len(B) + name(31s) + page_bytes(Q) + seg_off(Q) + seg_len(Q)
_POOL_ENTRY = struct.Struct("<B31sQQQ")
_POOL_ENTRY_SIZE = _POOL_ENTRY.size  # 56
_POOL_TABLE_OFF = _HEAD_FIXED.size + MAX_POOLSET_LEN
assert _POOL_TABLE_OFF + MAX_POOLS * _POOL_ENTRY_SIZE <= HEADER_SIZE
_PAGE_KEYS_OFF = _POOL_TABLE_OFF + MAX_POOLS * _POOL_ENTRY_SIZE
assert _PAGE_KEYS_OFF + 32 * KEY_HEX_LEN <= HEADER_SIZE


def _round_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _nstr(x) -> str:
    """Normalize a PoolName (str-Enum: str() gives 'PoolName.KV') to its value."""
    if isinstance(x, PoolName):
        return x.value
    return str(x)


class _PoolGeo:
    """Geometry of one registered pool as seen by this engine."""

    __slots__ = ("name", "page_bytes", "slot_bytes", "host_pool", "dtype")

    def __init__(self, name: str, host_pool):
        self.name = name
        self.host_pool = host_pool
        page = host_pool.get_data_page(self._probe_index(host_pool))
        self.page_bytes = page.numel() * page.element_size()
        self.dtype = page.dtype
        # #35231 fix pattern: round every slot to OS-page multiples.
        self.slot_bytes = _round_up(self.page_bytes, DIO_BLOCK)

    @staticmethod
    def _probe_index(host_pool) -> int:
        ps = int(getattr(host_pool, "page_size", 1) or 1)
        return 0

    def __repr__(self):
        return f"_PoolGeo({self.name}, page={self.page_bytes}, slot={self.slot_bytes})"


class _CoverageCache:
    """TTL cache: group id -> num_pages present (positive answers only)."""

    def __init__(self, ttl: float):
        self.ttl = ttl
        self.lock = threading.Lock()
        self.cache: "OrderedDict[str, Tuple[int, float]]" = OrderedDict()
        self.max_entries = 200_000

    def get(self, key: str) -> Optional[int]:
        with self.lock:
            ent = self.cache.get(key)
            if ent is None:
                return None
            np_, ts = ent
            if self.ttl >= 0 and time.monotonic() - ts > self.ttl:
                del self.cache[key]
                return None
            return np_

    def put(self, key: str, np_: int):
        with self.lock:
            self.cache[key] = (np_, time.monotonic())
            if len(self.cache) > self.max_entries:
                self.cache.popitem(last=False)

    def invalidate(self, key: str):
        with self.lock:
            self.cache.pop(key, None)

    def clear(self):
        with self.lock:
            self.cache.clear()


class HiCacheBlob(HiCacheStorage):
    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache-blob"
    ):
        cfg = storage_config.extra_config or {}
        self.storage_config = storage_config

        from sglang.srt.environ import envs

        self.file_path = (
            envs.SGLANG_HICACHE_BLOB_BACKEND_STORAGE_DIR.get()
            or envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get()
            or cfg.get("storage_dir")
            or file_path
        )

        self.blob_pages = int(cfg.get("blob_pages", 16))
        if self.blob_pages < 1:
            raise ValueError("blob_pages must be >= 1")
        # header content must hold blob_pages page keys; keep within the
        # on-disk 64 KiB header block so segment offsets stay DIO-aligned
        self._header_size = _PAGE_KEYS_OFF + self.blob_pages * KEY_HEX_LEN
        if self._header_size > HEADER_BLOCK:
            raise ValueError(
                f"blob_pages={self.blob_pages} exceeds header block "
                f"({_PAGE_KEYS_OFF + self.blob_pages * KEY_HEX_LEN} > {HEADER_BLOCK})"
            )
        self.io_streams = int(cfg.get("io_streams", 8))
        self.use_direct_io = bool(cfg.get("direct_io", True))
        self.stripe_count = int(cfg.get("stripe_count", 8))
        self.evict_max_bytes = int(cfg.get("evict_max_bytes", 0))
        self.evict_max_age_s = float(cfg.get("evict_max_age_s", 0))
        self.evict_interval_s = float(cfg.get("evict_interval_s", 300))
        self.write_manifest = bool(cfg.get("write_manifest", True))
        metadata_ttl = float(cfg.get("metadata_ttl", 5.0))

        model_name = storage_config.model_name or ""
        self.model_tag = "-".join(model_name.split("/"))[:64]

        self._pools: Dict[str, _PoolGeo] = {}
        self._reg_by_str: Dict[str, Any] = {}
        self._geo_lock = threading.Lock()
        self._coverage = _CoverageCache(metadata_ttl)
        self._io_pool = ThreadPoolExecutor(
            max_workers=max(1, self.io_streams), thread_name_prefix="blob-io"
        )

        self._total_written_bytes = 0
        self._total_written_objects = 0
        self._total_read_bytes = 0
        self._total_read_objects = 0

        self._make_root()
        self._evictor_thread = None
        if self.evict_max_bytes > 0 or self.evict_max_age_s > 0:
            self._evict_stop = threading.Event()
            self._evictor_thread = threading.Thread(
                target=self._evictor_loop, name="blob-evictor", daemon=True
            )
            self._evictor_thread.start()

    # ------------------------------------------------------------------
    # directory layout
    # ------------------------------------------------------------------

    def _make_root(self):
        os.makedirs(self.file_path, exist_ok=True)
        self._try_setstripe(self.file_path)
        for i in range(256):
            os.makedirs(os.path.join(self.file_path, f"{i:02x}"), exist_ok=True)

    def _try_setstripe(self, path: str):
        """Apply lfs setstripe -c N when the lfs tool exists (no-op elsewhere)."""
        try:
            rc = subprocess.call(
                ["lfs", "setstripe", "-c", str(self.stripe_count), path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            if rc != 0:
                logger.debug(
                    "[blob] lfs setstripe rc=%d for %s (ok on non-Lustre)", rc, path
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # pool registration & geometry
    # ------------------------------------------------------------------

    @property
    def _primary_pool(self) -> "HostKVCache":
        pool = getattr(self, "mem_pool_host", None)
        if pool is None:
            raise RuntimeError("blob backend used before register_mem_pool_host")
        if hasattr(pool, "anchor_entry"):  # HostPoolGroup
            pool = pool.anchor_entry.host_pool
        return pool

    def _refresh_pools(self):
        """Snapshot registered pool geometries (idempotent; cheap after first)."""
        with self._geo_lock:
            primary = self._primary_pool
            reg = {_nstr(k): v for k, v in getattr(self, "registered_pools", {}).items()}
            self._reg_by_str = reg
            pools: Dict[str, _PoolGeo] = {}
            kv_name = PoolName.KV.value
            pools[kv_name] = self._pools.get(kv_name) or _PoolGeo(kv_name, primary)
            for name, hp in reg.items():
                if name.startswith("draft"):
                    continue  # drafts go to per-pool objects, not the group object
                pools[name] = self._pools.get(name) or _PoolGeo(name, hp)
            for name in list(pools.keys()):
                if name.startswith("draft"):
                    pools.pop(name)
            self._pools = pools
            layer_start = int(getattr(primary, "start_layer", 0) or 0)
            layer_end = int(getattr(primary, "end_layer", 0) or 0)
            page_tokens = int(getattr(primary, "page_size", 1) or 1)
            poolset = "+".join(sorted(pools.keys()))
            geo_desc = json.dumps(
                {
                    "p": page_tokens,
                    "b": self.blob_pages,
                    "ls": layer_start,
                    "le": layer_end,
                    "pools": {n: g.page_bytes for n, g in sorted(pools.items())},
                },
                sort_keys=True,
            )
            geo_id = hashlib.sha1(geo_desc.encode()).hexdigest()[:16]
            self._key_prefix = f"{self.model_tag}|L{layer_start}-{layer_end}|{geo_id}"
            self._poolset = poolset
            self._page_tokens = page_tokens
            self._draft_poolsets = {n for n in reg if n.startswith("draft")}

    # ------------------------------------------------------------------
    # keys / paths
    # ------------------------------------------------------------------

    def _group_id(self, first_page_key: str, poolset: str) -> str:
        h = hashlib.sha1(
            f"{self._key_prefix}|{poolset}|{first_page_key}".encode()
        ).hexdigest()
        return h[:32]

    def _obj_path(self, group_id: str) -> str:
        return os.path.join(self.file_path, group_id[:2], group_id + ".blob")

    def _manifest_path(self, first_page_key: str) -> str:
        # One journal per CONTEXT (the chain-root page key is stable across
        # every op and both arms of the same conversation): appends one JSON
        # line per write op instead of one file per op.
        h = hashlib.sha1(
            f"{self.model_tag}|mf|{self._poolset}|{first_page_key}".encode()
        ).hexdigest()[:24]
        return os.path.join(self.file_path, h[:2], f"mf_{h}.jsonl")

    # ------------------------------------------------------------------
    # header packing
    # ------------------------------------------------------------------

    def _pack_header(
        self,
        num_pages: int,
        pool_geos: Sequence[_PoolGeo],
        poolset: str,
        page_keys: Sequence[str],
    ) -> bytearray:
        buf = bytearray(self._header_size)
        seg_off = HEADER_BLOCK
        seg_lens = []
        for geo in pool_geos:
            # segment extent padded so every segment START stays DIO-aligned
            seg_len = _round_up(geo.slot_bytes * num_pages, DIO_ALIGN)
            seg_lens.append(seg_len)
            seg_off += seg_len
        poolset_b = poolset.encode()[:MAX_POOLSET_LEN]
        _HEAD_FIXED.pack_into(
            buf,
            0,
            MAGIC,
            VERSION,
            self._page_tokens,
            num_pages,
            self.blob_pages,
            int(getattr(self._primary_pool, "start_layer", 0) or 0),
            int(getattr(self._primary_pool, "end_layer", 0) or 0),
            len(pool_geos),
            len(poolset_b),
        )
        buf[_HEAD_FIXED.size : _HEAD_FIXED.size + len(poolset_b)] = poolset_b
        for i, geo in enumerate(pool_geos):
            nm = geo.name.encode()[:31]
            _POOL_ENTRY.pack_into(
                buf,
                _POOL_TABLE_OFF + i * _POOL_ENTRY_SIZE,
                len(nm),
                nm,
                geo.page_bytes,
                HEADER_BLOCK + sum(seg_lens[:i]),
                seg_lens[i],
            )
        for i, k in enumerate(page_keys[:num_pages]):
            if len(k) != KEY_HEX_LEN:
                raise ValueError(f"bad page key length: {len(k)}")
            off = _PAGE_KEYS_OFF + i * KEY_HEX_LEN
            buf[off : off + KEY_HEX_LEN] = k.encode()
        return buf

    @staticmethod
    def _unpack_header(buf: bytes) -> dict:
        (
            magic,
            version,
            page_tokens,
            num_pages,
            blob_pages,
            layer_start,
            layer_end,
            n_pools,
            poolset_len,
        ) = _HEAD_FIXED.unpack_from(buf, 0)
        if magic != MAGIC:
            raise ValueError("bad magic")
        if version != VERSION:
            raise ValueError(f"unsupported version {version}")
        poolset = buf[_HEAD_FIXED.size : _HEAD_FIXED.size + poolset_len].decode()
        pools = []
        for i in range(n_pools):
            (
                name_len,
                name_raw,
                page_bytes,
                seg_off,
                seg_len,
            ) = _POOL_ENTRY.unpack_from(buf, _POOL_TABLE_OFF + i * _POOL_ENTRY_SIZE)
            pools.append(
                {
                    "name": name_raw[:name_len].decode(),
                    "page_bytes": page_bytes,
                    "seg_off": seg_off,
                    "seg_len": seg_len,
                }
            )
        keys = []
        for i in range(num_pages):
            off = _PAGE_KEYS_OFF + i * KEY_HEX_LEN
            keys.append(buf[off : off + KEY_HEX_LEN].decode())
        return {
            "page_tokens": page_tokens,
            "num_pages": num_pages,
            "blob_pages": blob_pages,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "poolset": poolset,
            "pools": pools,
            "keys": keys,
        }

    # ------------------------------------------------------------------
    # chain helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _full_chain(keys: List[str], extra_info) -> Tuple[List[str], int]:
        """(full page-key chain from context root, start page of this call)."""
        prefix: List[str] = []
        if extra_info is not None and extra_info.prefix_keys:
            prefix = list(extra_info.prefix_keys)
        return prefix + list(keys), len(prefix)

    # ------------------------------------------------------------------
    # existence / hit queries
    # ------------------------------------------------------------------

    def _coverage_of_group(self, group_id: str) -> int:
        """num_pages stored for a group (0 = absent). stat + size inversion."""
        cached = self._coverage.get(group_id)
        if cached is not None:
            return cached
        try:
            st = os.stat(self._obj_path(group_id))
        except FileNotFoundError:
            return 0
        np_ = self._np_from_size(st.st_size)
        if np_ == 0 and st.st_size > 0:
            logger.warning(
                "[blob] object %s has unexpected size %d", group_id, st.st_size
            )
        self._coverage.put(group_id, np_)
        return np_

    def _np_from_size(self, size: int) -> int:
        """Invert file size -> num_pages under the padded segment layout."""
        pools = list(self._pools.values())
        if not pools:
            return 0
        for np_ in range(1, self.blob_pages + 1):
            total = HEADER_BLOCK + sum(
                _round_up(g.slot_bytes * np_, DIO_ALIGN) for g in pools
            )
            if total == size:
                return np_
        return 0

    def _hit_pages(
        self, keys: List[str], extra_info
    ) -> int:
        """Longest prefix of `keys` (pages) fully present in storage."""
        if not keys:
            return 0
        self._refresh_pools()
        full_chain, start = self._full_chain(keys, extra_info)
        n_abs = len(full_chain)
        g0 = start // self.blob_pages
        hit_abs = start
        g = g0
        trace = []
        while g * self.blob_pages < n_abs:
            p0 = g * self.blob_pages
            if p0 > hit_abs:
                trace.append(f"gap@{p0}")
                break  # gap before this group
            gid = self._group_id(full_chain[p0], self._poolset)
            np_ = self._coverage_of_group(gid)
            trace.append(f"g{g}:{gid[:8]}:np{np_}")
            if np_ == 0:
                break
            hit_abs = min(p0 + np_, n_abs)
            if np_ < self.blob_pages:
                break  # partial group: prefix ends here
            g += 1
        hit = max(0, min(hit_abs, n_abs) - start)
        logger.info(
            "[blob] hit_pages start=%d n_abs=%d nkeys=%d hit=%d trace=%s",
            start, n_abs, n_abs - start, hit, ",".join(trace),
        )
        return hit

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        kv_hit = self._hit_pages(keys, extra_info)
        hit_count: Dict[str, int] = {PoolName.KV: kv_hit} if kv_hit else {}
        final = kv_hit
        for t in pool_transfers or []:
            if final == 0:
                break
            if t.hit_policy != PoolHitPolicy.ALL_PAGES:
                raise NotImplementedError(
                    f"[blob] hit policy {t.hit_policy} not supported (pool {t.name})"
                )
            name = _nstr(t.name)
            if name.startswith("draft"):
                hit_count[name] = final  # best-effort pools never gate the hit
                continue
            if name not in self._pools:
                final = 0  # pool unknown to this engine -> cannot be present
                break
            hit_count[name] = final  # group objects contain every pool
        return PoolTransferResult(final, hit_count)

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        return self._hit_pages(keys, extra_info)

    def exists(self, key: str) -> bool:
        """Legacy single-key probe: treat the key as a possible group start."""
        try:
            self._refresh_pools()
        except RuntimeError:
            return False
        return self._coverage_of_group(self._group_id(key, self._poolset)) > 0

    # ------------------------------------------------------------------
    # IO primitives
    # ------------------------------------------------------------------

    @staticmethod
    def _aligned_buffer(size: int) -> mmap.mmap:
        return mmap.mmap(-1, size)  # page-aligned anonymous memory

    def _open_read(self, path: str) -> int:
        flags = os.O_RDONLY
        if self.use_direct_io:
            flags |= getattr(os, "O_DIRECT", 0)
        return os.open(path, flags)

    def _open_write(self, path: str) -> int:
        flags = os.O_WRONLY | os.O_CREAT
        if self.use_direct_io:
            flags |= getattr(os, "O_DIRECT", 0)
        return os.open(path, flags, 0o644)

    def _pwrite_all(self, fd: int, buf, total: int):
        off = 0
        mv = memoryview(buf)
        while off < total:
            n = min(CHUNK_SIZE, total - off)
            w = os.pwrite(fd, mv[off : off + n], off)
            if w != n:
                raise IOError(f"short pwrite {w} != {n} at {off}")
            off += w

    def _pread_all(self, fd: int, buf, total: int, offset: int = 0):
        """Read `total` bytes at `offset` into (a view of) `buf` (preadv-based)."""
        mv = memoryview(buf)
        off = 0
        while off < total:
            n = min(CHUNK_SIZE, total - off)
            view = mv[off : off + n]
            done = 0
            while done < n:
                got = os.preadv(fd, [view[done:]], offset + off + done)
                if got == 0:
                    raise IOError(f"short pread at {offset + off + done}/{total}")
                done += got
                if got < n and done < n:
                    # DIO short read mid-chunk: next offset would be unaligned;
                    # treat as truncation (the caller reports a group miss)
                    raise IOError(
                        f"short DIO pread {done}/{n} at {offset + off}"
                    )
            off += n

    # ------------------------------------------------------------------
    # read path
    # ------------------------------------------------------------------

    def _read_group(
        self,
        group_id: str,
        poolset: str,
        expected_keys: List[Optional[str]],
        pool_targets: Dict[str, List[Tuple[int, int, str, int]]],
        results: Dict[str, List[bool]],
    ) -> None:
        """Read one group object; scatter pages into host pools.

        pool_targets: pool name -> [(page_in_group, host_page_token_idx,
                       transfer_name, transfer_page_idx)]
        Fills results[transfer_name][transfer_page_idx] = True on success.
        """
        path = self._obj_path(group_id)
        _t_hdr = 0.0
        _t_read = 0.0
        _t_sc = 0.0
        _n_bytes = 0
        _n_pages = 0
        _t0 = time.perf_counter() if _BLOB_PHASE_STATS else 0.0
        try:
            fd = self._open_read(path)
        except FileNotFoundError:
            return None
        except OSError:
            return None
        try:
            hbuf = self._aligned_buffer(HEADER_BLOCK)
            self._pread_all(fd, hbuf, HEADER_BLOCK)
            header = self._unpack_header(hbuf)
            # hbuf (and its views) are freed by refcounting

            if header["poolset"] != poolset or header["page_tokens"] != self._page_tokens:
                logger.debug(
                    "[blob] poolset/page mismatch for %s: stored %s/%d, want %s/%d",
                    group_id,
                    header["poolset"],
                    header["page_tokens"],
                    poolset,
                    self._page_tokens,
                )
                return
            hmap = {p["name"]: p for p in header["pools"]}
            for name in pool_targets:
                geo = self._pools.get(name)
                h = hmap.get(name)
                if geo is None or h is None or h["page_bytes"] != geo.page_bytes:
                    return

            np_ = header["num_pages"]
            page_keys = header["keys"]

            # verify per-page chain hashes before accepting any bytes
            for name, targets in pool_targets.items():
                for (p, _host_idx, _tn, _ti) in targets:
                    if p >= np_:
                        continue
                    exp = expected_keys[p] if p < len(expected_keys) else None
                    if exp is not None and page_keys[p] != exp:
                        logger.warning(
                            "[blob] page-key mismatch in %s at page %d; miss",
                            group_id,
                            p,
                        )
                        self._coverage.invalidate(group_id)
                        return

            seg_cache: Dict[str, Any] = {}
            if _BLOB_PHASE_STATS:
                _t_hdr = time.perf_counter() - _t0
            try:
                for name, targets in pool_targets.items():
                    geo = self._pools[name]
                    h = hmap[name]
                    pool = geo.host_pool
                    _ps = int(getattr(pool, "page_size", 1) or 1)
                    _bulk = getattr(pool, "set_from_flat_data_pages_bulk", None)
                    _k = 0
                    _nt = len(targets)
                    while _k < _nt:
                        (p, host_idx, tn, ti) = targets[_k]
                        if p >= np_:
                            _k += 1
                            continue
                        seg = seg_cache.get(name)
                        if seg is None:
                            m = self._aligned_buffer(h["seg_len"])
                            _tr = time.perf_counter() if _BLOB_PHASE_STATS else 0.0
                            self._pread_all(fd, m, h["seg_len"], h["seg_off"])
                            if _BLOB_PHASE_STATS:
                                _t_read += time.perf_counter() - _tr
                                _n_bytes += h["seg_len"]
                            seg = memoryview(m)
                            seg_cache[name] = seg
                        # kv-unit: run-batched scatter — consecutive segment
                        # pages mapped to consecutive host pages go as ONE
                        # bulk store (the per-page path costs layer_num
                        # copy_ calls per page; ~49% of restore wall).
                        _run = 1
                        if _bulk is not None:
                            while (
                                _k + _run < _nt
                                and targets[_k + _run][0] == p + _run
                                and targets[_k + _run][0] < np_
                                and targets[_k + _run][1] == host_idx + _run * _ps
                            ):
                                _run += 1
                        _ts = time.perf_counter() if _BLOB_PHASE_STATS else 0.0
                        if _run >= 4 and _bulk is not None:
                            _item = geo.dtype.itemsize
                            _stride = geo.slot_bytes // _item
                            _seg_t = torch.frombuffer(seg, dtype=geo.dtype)
                            _run_t = torch.as_strided(
                                _seg_t,
                                (_run, pool.layer_num, pool.item_bytes),
                                (_stride, pool.item_bytes, 1),
                                storage_offset=(p * geo.slot_bytes) // _item,
                            )
                            _bulk(host_idx, _run_t, _run)
                            for _r in range(_run):
                                _tr2 = targets[_k + _r]
                                results[_tr2[2]][_tr2[3]] = True
                            if _BLOB_PHASE_STATS:
                                _t_sc += time.perf_counter() - _ts
                                _n_pages += _run
                            _k += _run
                            continue
                        src = seg[
                            p * geo.slot_bytes : p * geo.slot_bytes + geo.page_bytes
                        ]
                        page_t = torch.frombuffer(src, dtype=geo.dtype)
                        pool.set_from_flat_data_page(host_idx, page_t)
                        if _BLOB_PHASE_STATS:
                            _t_sc += time.perf_counter() - _ts
                            _n_pages += 1
                        results[tn][ti] = True
                        _k += 1
                self._total_read_objects += 1
                self._total_read_bytes += HEADER_BLOCK + sum(
                    h["seg_len"] for h in header["pools"]
                )
            finally:
                # explicit close() raises while torch views exist; drop refs
                # and let refcounting unmap the anonymous mmaps
                seg_cache.clear()
            if _BLOB_PHASE_STATS:
                return {
                    "hdr": _t_hdr,
                    "read": _t_read,
                    "sc": _t_sc,
                    "bytes": _n_bytes,
                    "pages": _n_pages,
                }
            return None
        except Exception as e:  # store failure = miss, never a raise
            logger.warning("[blob] read group %s failed: %s", group_id, e)
            return None
        finally:
            os.close(fd)

    def _read_transfers(
        self, transfers: List[PoolTransfer], extra_info
    ) -> Dict[Any, List[bool]]:
        self._refresh_pools()
        results: Dict[Any, List[bool]] = {
            t.name: [False] * len(t.keys or []) for t in transfers
        }
        if not transfers:
            return results

        main = transfers[0]
        keys = list(main.keys or [])
        if not keys:
            return results
        full_chain, start = self._full_chain(keys, extra_info)

        # Build the plan: group -> poolset -> pool -> targets
        group_plan: Dict[int, Dict[str, Dict[str, List[Tuple[int, int, str, int]]]]] = {}
        for t_idx, t in enumerate(transfers):
            name = _nstr(t.name)
            tkeys = list(t.keys or [])
            if not tkeys:
                continue
            if name.startswith("draft"):
                poolset = name
                if name not in self._reg_by_str:
                    continue
                if name not in self._pools:
                    with self._geo_lock:
                        self._pools[name] = _PoolGeo(name, self._reg_by_str[name])
            else:
                poolset = self._poolset
                if name not in self._pools:
                    logger.warning("[blob] read for unknown pool %s skipped", name)
                    continue
                tkeys = keys  # main-path pools share the KV key list
            geo = self._pools[name]
            hp = geo.host_pool
            ps = int(getattr(hp, "page_size", 1) or 1)
            host_indices = t.host_indices
            if host_indices is None:
                continue
            for i in range(len(tkeys)):
                abs_page = start + i
                g = abs_page // self.blob_pages
                p_in_g = abs_page % self.blob_pages
                host_idx = int(host_indices[i * ps].item())
                group_plan.setdefault(g, {}).setdefault(poolset, {}).setdefault(
                    name, []
                ).append((p_in_g, host_idx, t.name, i))

        futures = []
        for g, by_poolset in group_plan.items():
            p0 = g * self.blob_pages
            if p0 >= len(full_chain):
                continue  # group not addressable (no first-page key)
            for poolset, pool_targets in by_poolset.items():
                gid = self._group_id(full_chain[p0], poolset)
                if poolset == self._poolset:
                    expected = [
                        full_chain[p0 + p] if p0 + p < len(full_chain) else None
                        for p in range(self.blob_pages)
                    ]
                else:
                    expected = [None] * self.blob_pages
                futures.append(
                    self._io_pool.submit(
                        self._read_group, gid, poolset, expected, pool_targets, results
                    )
                )
        _pt0 = time.perf_counter() if _BLOB_PHASE_STATS else 0.0
        _agg = {"hdr": 0.0, "read": 0.0, "sc": 0.0, "bytes": 0, "pages": 0, "groups": 0}
        for fut in futures:
            try:
                _r = fut.result()
                if _r is not None:
                    _agg["hdr"] += _r["hdr"]
                    _agg["read"] += _r["read"]
                    _agg["sc"] += _r["sc"]
                    _agg["bytes"] += _r["bytes"]
                    _agg["pages"] += _r["pages"]
                    _agg["groups"] += 1
            except Exception as e:  # pragma: no cover
                logger.warning("[blob] group read future failed: %s", e)
        if _BLOB_PHASE_STATS and _agg["groups"]:
            _tt = time.perf_counter() - _pt0
            logger.info(
                "[kvu-blob-phase] groups=%d pages=%d bytes=%.3e t_total_s=%.3f "
                "t_header_s=%.3f t_read_s=%.3f t_scatter_s=%.3f t_other_s=%.3f "
                "read_gbps=%.2f scatter_gbps=%.2f",
                _agg["groups"],
                _agg["pages"],
                float(_agg["bytes"]),
                _tt,
                _agg["hdr"],
                _agg["read"],
                _agg["sc"],
                _tt - _agg["hdr"] - _agg["read"] - _agg["sc"],
                (_agg["bytes"] / 1e9 / _agg["read"]) if _agg["read"] > 0 else 0.0,
                (_agg["bytes"] / 1e9 / _agg["sc"]) if _agg["sc"] > 0 else 0.0,
            )
        return results

    # ------------------------------------------------------------------
    # write path
    # ------------------------------------------------------------------

    def _write_group(
        self,
        group_id: str,
        num_pages: int,
        poolset: str,
        page_keys: List[str],
        pool_pages: Dict[str, List[Tuple[int, int]]],
        skip_if_np_ge: Optional[int],
    ) -> Tuple[str, int]:
        """Write one group object. pool_pages: name -> [(page_in_group, host_page_idx)].

        Every pool must cover pages [0, num_pages). Returns ("ok"|"skip"|"fail", np).
        """
        path = self._obj_path(group_id)
        if skip_if_np_ge is not None and self._coverage_of_group(group_id) >= skip_if_np_ge:
            return "skip", skip_if_np_ge

        names = sorted(pool_pages.keys())
        geos = [self._pools[n] for n in names]
        seg_base = {}
        off = HEADER_BLOCK
        for geo in geos:
            seg_base[geo.name] = off
            off += _round_up(geo.slot_bytes * num_pages, DIO_ALIGN)
        total = off
        buf = self._aligned_buffer(total)
        mv = memoryview(buf)
        tmp = path + f".tmp.{os.getpid()}.{threading.get_ident()}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            header = self._pack_header(num_pages, geos, poolset, page_keys)
            mv[: self._header_size] = bytes(header)
            mv[self._header_size : HEADER_BLOCK] = bytes(
                HEADER_BLOCK - self._header_size
            )
            for name, targets in pool_pages.items():
                geo = self._pools[name]
                pool = geo.host_pool
                base = seg_base[name]
                for (p, host_idx) in targets:
                    page = pool.get_data_page(host_idx)
                    dst = mv[
                        base + p * geo.slot_bytes : base
                        + p * geo.slot_bytes
                        + geo.page_bytes
                    ]
                    page_bytes = page.contiguous().view(torch.uint8).numpy()
                    dst[:] = page_bytes
            fd = self._open_write(tmp)
            try:
                self._pwrite_all(fd, buf, total)
            finally:
                os.close(fd)
            os.replace(tmp, path)
            self._coverage.put(group_id, num_pages)
            self._total_written_objects += 1
            self._total_written_bytes += total
            return "ok", num_pages
        except Exception as e:  # store failure = fail status, never a raise
            logger.warning("[blob] write %s failed: %s", group_id, e)
            try:
                os.remove(tmp)
            except OSError:
                pass
            return "fail", 0

    def _write_transfers(
        self, transfers: List[PoolTransfer], extra_info
    ) -> Dict[Any, List[bool]]:
        self._refresh_pools()
        results: Dict[Any, List[bool]] = {
            t.name: [False] * len(t.keys or []) for t in transfers
        }
        if not transfers:
            return results
        name_obj = {_nstr(t.name): t.name for t in transfers}

        main = transfers[0]
        keys = list(main.keys or [])
        if not keys:
            return results
        full_chain, start = self._full_chain(keys, extra_info)
        end = start + len(keys)

        # plan: group -> poolset -> pool -> [(p_in_g, host_idx)]
        main_plan: Dict[int, Dict[str, List[Tuple[int, int]]]] = {}
        draft_plan: Dict[int, Dict[str, Dict[str, List[Tuple[int, int]]]]] = {}

        for t in transfers:
            name = _nstr(t.name)
            tkeys = list(t.keys or [])
            host_indices = t.host_indices
            if host_indices is None:
                continue
            if not name.startswith("draft"):
                if name not in self._pools:
                    logger.warning("[blob] write for unknown pool %s skipped", name)
                    continue
                tkeys = keys
                geo = self._pools[name]
            else:
                if name not in self._reg_by_str:
                    continue
                if name not in self._pools:
                    with self._geo_lock:
                        self._pools[name] = _PoolGeo(name, self._reg_by_str[name])
                geo = self._pools[name]
            hp = geo.host_pool
            ps = int(getattr(hp, "page_size", 1) or 1)
            for i in range(len(tkeys)):
                abs_page = start + i
                g = abs_page // self.blob_pages
                p_in_g = abs_page % self.blob_pages
                host_idx = int(host_indices[i * ps].item())
                if name.startswith("draft"):
                    draft_plan.setdefault(g, {}).setdefault(name, []).append(
                        (p_in_g, host_idx)
                    )
                else:
                    main_plan.setdefault(g, {}).setdefault(name, []).append(
                        (p_in_g, host_idx)
                    )

        # main objects: only groups whose FIRST page is within this write range
        futures = []
        kv_name = PoolName.KV.value
        # drafts may have been added to self._pools while planning this very
        # call (per-pool objects); the group object only requires the pools
        # of the main poolset.
        required_pools = [n for n in self._pools if not n.startswith("draft")]
        for g, pool_pages in main_plan.items():
            p0 = g * self.blob_pages
            if p0 < start or p0 >= end:
                continue  # head group (starts before this op) is not writable here
            np_avail = min(self.blob_pages, end - p0)
            missing = [
                n
                for n in required_pools
                if n not in pool_pages or len(pool_pages[n]) < np_avail
            ]
            if missing or len(pool_pages) != len(required_pools):
                logger.info(
                    "[blob] group %d incomplete (missing %s of %s); skipped",
                    g,
                    missing,
                    list(self._pools.keys()),
                )
                continue
            page_keys = full_chain[p0 : p0 + np_avail]
            gid = self._group_id(full_chain[p0], self._poolset)
            futures.append(
                (
                    g,
                    np_avail,
                    self._io_pool.submit(
                        self._write_group,
                        gid,
                        np_avail,
                        self._poolset,
                        page_keys,
                        pool_pages,
                        np_avail,
                    ),
                )
            )
        for g, np_avail, fut in futures:
            try:
                status, _ = fut.result()
            except Exception as e:  # pragma: no cover
                logger.warning("[blob] group write future failed: %s", e)
                status = "fail"
            if status not in ("ok", "skip"):
                continue
            p0 = g * self.blob_pages
            for name in self._pools:
                key = name_obj.get(name)
                if key is None or key not in results:
                    continue
                for i in range(len(keys)):
                    abs_page = start + i
                    if p0 <= abs_page < p0 + np_avail:
                        results[key][i] = True

        # draft per-pool objects (best-effort)
        for g, by_name in draft_plan.items():
            p0 = g * self.blob_pages
            if p0 < start or p0 >= end:
                continue
            np_avail = min(self.blob_pages, end - p0)
            for name, targets in by_name.items():
                if len(targets) < np_avail:
                    continue
                page_keys = full_chain[p0 : p0 + np_avail]
                gid = self._group_id(full_chain[p0], name)
                status, _ = self._write_group(
                    gid, np_avail, name, page_keys, {name: targets}, np_avail
                )
                if status not in ("ok", "skip"):
                    continue
                key = name_obj.get(name)
                if key is None or key not in results:
                    continue
                for (p_in_g, _host_idx) in targets:
                    abs_page = p0 + p_in_g
                    idx = abs_page - start
                    if 0 <= idx < len(results[key]):
                        results[key][idx] = True

        if self.write_manifest:
            try:
                self._write_manifest(keys, full_chain, start, main_plan)
            except Exception as e:
                logger.debug("[blob] manifest write failed: %s", e)
        return results

    def _write_manifest(self, keys, full_chain, start, main_plan):
        groups = {}
        for g in main_plan:
            p0 = g * self.blob_pages
            if p0 < start or p0 >= start + len(keys):
                continue
            np_avail = min(self.blob_pages, start + len(keys) - p0)
            groups[self._group_id(full_chain[p0], self._poolset)] = np_avail
        doc = {
            "ts": time.time(),
            "model": self.model_tag,
            "poolset": self._poolset,
            "layer_set": f"L{getattr(self._primary_pool, 'start_layer', 0)}"
            f"-{getattr(self._primary_pool, 'end_layer', 0)}",
            "first_key": full_chain[0] if full_chain else None,
            "pages": [start, start + len(keys)],
            "groups": groups,
        }
        line = (json.dumps(doc) + "\n").encode()
        fd = os.open(
            self._manifest_path(full_chain[0]),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            os.write(fd, line)
        finally:
            os.close(fd)

    # ------------------------------------------------------------------
    # public API: v3 (combined per-op IO), v2, v1
    # ------------------------------------------------------------------

    def batch_read_v3(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[Any, List[bool]]:
        """Combined read: KV + sidecar transfers served by single object IOs."""
        return self._read_transfers(transfers, extra_info)

    def batch_write_v3(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[Any, List[bool]]:
        """Combined write: KV + sidecar transfers into single group objects."""
        return self._write_transfers(transfers, extra_info)

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[Any, List[bool]]:
        return self._read_transfers(transfers, extra_info)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[Any, List[bool]]:
        return self._write_transfers(transfers, extra_info)

    # --- v1 (primary pool only; used by plain controllers e.g. decode offload)

    def _kv_transfer_v1(self, keys: List[str], host_indices: torch.Tensor) -> PoolTransfer:
        return PoolTransfer(name=PoolName.KV, host_indices=host_indices, keys=list(keys))

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        res = self._read_transfers([self._kv_transfer_v1(keys, host_indices)], extra_info)
        return res.get(PoolName.KV, [False] * len(keys))

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        res = self._write_transfers([self._kv_transfer_v1(keys, host_indices)], extra_info)
        return res.get(PoolName.KV, [False] * len(keys))

    # --- deprecated single-key API (interface compliance; the engine uses
    #     zero-copy v1 / v2 / v3 paths for this backend)

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        """Fill `target_location` (a flat dummy page) from the group at `key`.

        Best-effort legacy path: only works when `key` is a group-start key.
        """
        if target_location is None:
            return None
        try:
            self._refresh_pools()
        except RuntimeError:
            return None
        gid = self._group_id(key, self._poolset)
        path = self._obj_path(gid)
        try:
            fd = self._open_read(path)
        except OSError:
            return None
        try:
            hbuf = self._aligned_buffer(HEADER_BLOCK)
            self._pread_all(fd, hbuf, HEADER_BLOCK)
            header = self._unpack_header(hbuf)
            if (
                header["poolset"] != self._poolset
                or header["page_tokens"] != self._page_tokens
                or header["keys"][0] != key
                or header["num_pages"] < 1
            ):
                return None
            h = header["pools"][0]
            geo = self._pools[str(PoolName.KV)]
            seg = self._aligned_buffer(h["seg_len"])
            self._pread_all(fd, seg, h["seg_len"], h["seg_off"])
            src = memoryview(seg)[0 : geo.page_bytes]
            page_t = torch.frombuffer(src, dtype=geo.dtype).clone()
            target_location.view(torch.uint8).copy_(
                page_t.view(torch.uint8).reshape(-1)
            )
            return target_location
        except (IOError, OSError, ValueError, IndexError) as e:
            logger.warning("[blob] legacy get %s failed: %s", key, e)
            return None
        finally:
            os.close(fd)

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        return False  # single-key writes need full group context

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        return [self.get(k, t) for k, t in zip(keys, target_locations or [None] * len(keys))]

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None) -> bool:
        return False

    # ------------------------------------------------------------------
    # eviction / clear / stats
    # ------------------------------------------------------------------

    def _evictor_loop(self):
        bucket = 0
        while not self._evict_stop.wait(self.evict_interval_s):
            try:
                bucket = (bucket + 1) % 256
                self._evict_bucket(os.path.join(self.file_path, f"{bucket:02x}"))
            except Exception as e:
                logger.debug("[blob] evictor pass failed: %s", e)

    def _evict_bucket(self, path: str):
        now = time.time()
        entries = []
        total = 0
        with os.scandir(path) as it:
            for e in it:
                if not e.name.endswith(".blob"):
                    continue  # skips .tmp files too
                try:
                    st = e.stat()
                except OSError:
                    continue
                entries.append((st.st_mtime, st.st_size, e.path))
                total += st.st_size
        if not entries:
            return
        removed = 0
        if self.evict_max_age_s > 0:
            for mtime, size, p in list(entries):
                if now - mtime > self.evict_max_age_s:
                    try:
                        os.remove(p)
                        entries.remove((mtime, size, p))
                        total -= size
                        removed += 1
                        self._coverage.invalidate(os.path.basename(p)[:32])
                    except OSError:
                        pass
        if self.evict_max_bytes > 0 and total > self.evict_max_bytes:
            entries.sort()
            for mtime, size, p in entries:
                if total <= self.evict_max_bytes:
                    break
                try:
                    os.remove(p)
                    total -= size
                    removed += 1
                    self._coverage.invalidate(os.path.basename(p)[:32])
                except OSError:
                    pass
        if removed:
            logger.info("[blob] evicted %d objects from %s", removed, path)

    def clear(self) -> bool:
        try:
            for b in os.listdir(self.file_path):
                d = os.path.join(self.file_path, b)
                if not os.path.isdir(d):
                    continue
                for fn in os.listdir(d):
                    if fn.endswith(".blob") or fn.startswith("mf_"):
                        try:
                            os.remove(os.path.join(d, fn))
                        except OSError:
                            pass
            self._coverage.clear()
            logger.info("[blob] cleared storage at %s", self.file_path)
            return True
        except Exception as e:
            logger.error("[blob] clear failed: %s", e)
            return False

    def get_stats(self):
        return {
            "blob_written_objects": self._total_written_objects,
            "blob_written_bytes": self._total_written_bytes,
            "blob_read_objects": self._total_read_objects,
            "blob_read_bytes": self._total_read_bytes,
        }

    def close(self):
        if getattr(self, "_evict_stop", None) is not None:
            self._evict_stop.set()
        if getattr(self, "_io_pool", None) is not None:
            self._io_pool.shutdown(wait=False)
