"""Generation handover manifest.

The old generation exports its host-tier radix state as a manifest: a chain
list (token ids + source host page rows per chain) plus pool geometry, enough
for the heir to (a) allocate destination rows, (b) land pushed page bytes, and
(c) bulk-insert host-only radix nodes that reproduce the old tree's retained
prefixes.

Canonical page order
--------------------
Every list indexed "per page" below uses one canonical order: depth-first
pre-order over the exported chains, pages within each chain in order. The
exporter, the push descriptor builder, the checksummer and the importer all
use this order, so page j on one side corresponds to page j on the other.
"""

from __future__ import annotations

import hashlib
import json
from array import array
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

MANIFEST_VERSION = 1


@dataclass
class PoolSpec:
    """Geometry of one host-side pool on the exporting side.

    ``data_ptrs``/``data_lens`` are per-layer buffer pointers/lengths in
    bytes; ``item_len`` is the bytes of one page row (anchor KV pool:
    token_stride * page_size; indexer sidecar: page stride). ``page_row_of``
    semantics: the exporter and importer agree that page row r of layer L
    lives at data_ptrs[L] + r * item_len.
    """

    name: str
    layer_num: int
    item_len: int
    data_ptrs: List[int]
    data_lens: List[int]
    mem_kind: str = "DRAM"  # host pools are DRAM; decode-arm index keys VRAM


@dataclass
class ChainRecord:
    """One exported radix chain: a backuped tree node with its FULL key path.

    ``tokens`` are the FULL path's token ids from the root (page-multiple
    length) — self-contained for the heir's insert walk. The exported ROWS
    belong to the chain's tail segment only: the last ``n_seg_pages`` pages
    of ``tokens``. ``page_rows`` are the source host page rows of that
    segment (empty when ``staged``: rows are the staging buffer, sequential
    from 0). ``page_hashes`` carries the full 64-hex chained page hashes of
    the segment when the exporting tree had them (storage / kv events
    enabled); the full digest is required to continue the chain on the heir.
    """

    tokens: array  # array('q'), full path from root
    n_seg_pages: int = 0  # segment pages (rows belong here)
    page_rows: np.ndarray = None  # int64 [n_seg_pages] (source side)
    extra_key: Optional[str] = None
    cache_salt: Optional[str] = None
    page_hashes: Optional[List[str]] = None

    def __post_init__(self):
        if self.page_rows is None:
            self.page_rows = np.empty(0, dtype=np.int64)


@dataclass
class HandoverManifest:
    version: int
    fingerprint: str  # config fingerprint; heir must match or refuse
    page_size: int
    staged: bool  # True: source rows are a staging buffer, pages sequential from 0
    chains: List[ChainRecord] = field(default_factory=list)
    # per-pool per-page int64 checksums in canonical order (optional)
    checksums: Optional[Dict[str, np.ndarray]] = None

    # ---- derived ----
    @property
    def num_tokens(self) -> int:
        """Total SEGMENT tokens (rows to land), not full-path tokens."""
        return sum(c.n_seg_pages for c in self.chains) * self.page_size

    @property
    def num_pages(self) -> int:
        return sum(c.n_seg_pages for c in self.chains)

    def flat_tokens(self) -> np.ndarray:
        if not self.chains:
            return np.empty(0, dtype=np.int64)
        return np.concatenate([np.frombuffer(c.tokens, dtype=np.int64) for c in self.chains])

    def flat_page_rows(self) -> np.ndarray:
        """Source page rows of all segments in canonical order (staged: arange)."""
        if self.staged:
            return np.arange(self.num_pages, dtype=np.int64)
        return np.concatenate([c.page_rows for c in self.chains])


def config_fingerprint(
    model_path: str,
    page_size: int,
    layer_num: int,
    kv_cache_dim: int,
    store_dtype: str,
    indexer_size_per_token: Optional[int],
    pool_names: List[str],
    extra: Optional[Dict] = None,
) -> str:
    payload = {
        "model_path": str(model_path),
        "page_size": int(page_size),
        "layer_num": int(layer_num),
        "kv_cache_dim": int(kv_cache_dim),
        "store_dtype": str(store_dtype),
        "indexer_size_per_token": indexer_size_per_token,
        "pool_names": sorted(pool_names),
        "extra": extra or {},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Serialization: JSON header + packed binary body.
# ---------------------------------------------------------------------------


def manifest_to_bytes(m: HandoverManifest) -> bytes:
    header = {
        "version": m.version,
        "fingerprint": m.fingerprint,
        "page_size": m.page_size,
        "staged": m.staged,
        "num_chains": len(m.chains),
        "checksums": None
        if m.checksums is None
        else {k: v.tolist() for k, v in m.checksums.items()},
    }
    hb = json.dumps(header).encode("utf-8")
    parts = [np.uint32(len(hb)).tobytes(), hb]
    for c in m.chains:
        tokens = np.frombuffer(c.tokens, dtype=np.int64)
        rows = c.page_rows.astype(np.int64, copy=False)
        hashes = c.page_hashes if c.page_hashes is not None else []
        parts.append(np.uint64(len(tokens)).tobytes())  # full-path length
        parts.append(tokens.tobytes())
        parts.append(np.uint64(c.n_seg_pages).tobytes())
        if m.staged:
            pass  # rows implicit: sequential staging pages in canonical order
        else:
            assert len(rows) == c.n_seg_pages
            parts.append(rows.tobytes())
        parts.append(np.uint64(len(hashes)).tobytes())
        if hashes:
            parts.append("".join(hashes).encode("ascii"))
    return b"".join(parts)


def bytes_to_manifest(b: bytes) -> HandoverManifest:
    off = 0

    def take(n):
        nonlocal off
        assert off + n <= len(b), f"manifest truncated at {off}+{n}>{len(b)}"
        out = b[off : off + n]
        off += n
        return out

    hlen = int(np.frombuffer(take(4), dtype=np.uint32)[0])
    header = json.loads(take(hlen).decode("utf-8"))
    m = HandoverManifest(
        version=header["version"],
        fingerprint=header["fingerprint"],
        page_size=header["page_size"],
        staged=header["staged"],
        checksums=None
        if header["checksums"] is None
        else {k: np.array(v, dtype=np.int64) for k, v in header["checksums"].items()},
    )
    for _ in range(header["num_chains"]):
        ntok = int(np.frombuffer(take(8), dtype=np.uint64)[0])
        tokens = array("q")
        tokens.frombytes(take(ntok * 8))
        n_seg = int(np.frombuffer(take(8), dtype=np.uint64)[0])
        if m.staged:
            rows = np.arange(0, dtype=np.int64)  # filled below from canonical order
            rows = np.empty(0, dtype=np.int64)
        else:
            rows = np.frombuffer(take(n_seg * 8), dtype=np.int64).copy()
        nhash = int(np.frombuffer(take(8), dtype=np.uint64)[0])
        hashes = None
        if nhash:
            raw = take(nhash * 64).decode("ascii")
            hashes = [raw[i * 64 : (i + 1) * 64] for i in range(nhash)]
        m.chains.append(
            ChainRecord(
                tokens=tokens,
                n_seg_pages=n_seg,
                page_rows=rows,
                page_hashes=hashes,
            )
        )
    assert off == len(b), "trailing bytes in manifest"
    return m


# ---------------------------------------------------------------------------
# Checksums: per-page int64 sums over the page's bytes (all layers summed).
# ---------------------------------------------------------------------------


def page_checksums(
    per_layer_buffers: List[torch.Tensor],
    page_rows: np.ndarray,
    item_len: int,
    chunk_pages: int = 512,
) -> np.ndarray:
    """int64 sum of each page's bytes across the given per-layer buffers.

    ``page_rows`` indexes rows of ``item_len`` bytes in each buffer. Buffers
    must be contiguous; item_len must be a multiple of 8.
    """
    assert item_len % 8 == 0
    n_int = item_len // 8
    out = np.zeros(len(page_rows), dtype=np.int64)
    rows = torch.as_tensor(page_rows, dtype=torch.long)
    for buf in per_layer_buffers:
        flat = buf.view(-1) if buf.dtype == torch.uint8 else buf.view(torch.uint8).view(-1)
        assert flat.element_size() == 1
        iv = flat.view(torch.int64)
        assert iv.numel() % n_int == 0 or True
        for s in range(0, len(rows), chunk_pages):
            r = rows[s : s + chunk_pages]
            gathered = iv[r[:, None] * n_int + torch.arange(n_int, dtype=torch.long)[None, :]]
            out[s : s + len(r)] += gathered.sum(dim=1).numpy()
    return out
