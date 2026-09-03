"""Decode-arm (HiSparse) generation handover: export / push / import.

Decode-arm state (HiSparseRadixCache + HiSparseCoordinator):
  * the radix tree stores LOGICAL kv indices in node.value;
  * host-row ownership lives in ``coordinator.logical_to_host_row``
    (logical index -> host token slot, -1 = unretained);
  * latent KV bytes live in ``coordinator.mem_pool_host`` (MLATokenToKVPoolHost,
    layer_first) at those host slots;
  * lightning-indexer keys are DEVICE-resident at logical indices
    (``mem_pool_device.index_k_with_scale_buffer[layer][logical_page]``,
    index buffer sized to the full logical space; retained indices never reused).

Export walks the tree and emits, per backuped node, the retained HEAD run of
its logical indices (matching ``retained_prefix_len``'s trimming): full-path
key + old logical pages + host row pages. Import allocates fresh heir logical
pages + host rows, lands the pushed bytes, inserts the tree (plain values =
heir logical indices), and fills ``logical_to_host_row`` via
``retain_rows``.

Pools pushed: host latent (DRAM, per-layer page rows) + indexer keys
(VRAM -> VRAM, per-layer logical pages). Same push machinery as the prefill
arm (run-grouped page descriptors, old-initiated WRITE).
"""

from __future__ import annotations

import logging
import threading
import time
from array import array
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class DecodeChain:
    """One retained tree node: full key path + retained head segment."""

    tokens: array  # array('q'), FULL path from root
    n_seg_pages: int  # retained head pages of the node's own segment
    old_logical_pages: np.ndarray  # int64 [n_seg_pages]
    host_row_pages: np.ndarray  # int64 [n_seg_pages] (source host pool)
    extra_key: Optional[str] = None
    cache_salt: Optional[str] = None


@dataclass
class DecodeExport:
    chains: List[DecodeChain] = field(default_factory=list)
    host_item_len: int = 0  # bytes per host page row (latent KV)
    index_item_len: int = 0  # bytes per indexer-key page row
    layer_num: int = 0

    @property
    def num_pages(self) -> int:
        return sum(c.n_seg_pages for c in self.chains)

    def seg_tokens(self, page_size: int) -> int:
        return sum(c.n_seg_pages for c in self.chains) * page_size


def _walk_retained(tree, coordinator, page_size: int) -> List[DecodeChain]:
    """DFS pre-order; per node, the retained head run of its value indices."""
    out: List[DecodeChain] = []

    def dfs(node, path_tokens):
        for c in node.children.values():
            if len(c.key) == 0:
                continue
            seg = c.key.raw_token_ids()
            full = path_tokens + seg if len(path_tokens) else seg
            if c.value is not None and len(c.value) > 0:
                hv = c.value.to(device="cpu", dtype=torch.int64)
                rows = coordinator.logical_to_host_row[hv]
                # retained head run (first missing page ends it)
                n_ret = 0
                for i in range(0, len(hv), page_size):
                    if (rows[i : i + page_size] >= 0).all():
                        n_ret += page_size
                    else:
                        break
                if n_ret >= page_size:
                    n_pages = n_ret // page_size
                    old_logical = (
                        hv.view(-1, page_size)[:, 0] // page_size
                    ).numpy()[:n_pages]
                    host_rows = rows[: n_pages * page_size]
                    host_pages = (
                        host_rows.view(-1, page_size)[:, 0] // page_size
                    ).numpy()
                    out.append(
                        DecodeChain(
                            tokens=full,
                            n_seg_pages=n_pages,
                            old_logical_pages=old_logical,
                            host_row_pages=host_pages,
                            extra_key=c.key.extra_key,
                            cache_salt=c.key.cache_salt,
                        )
                    )
            dfs(c, full)

    dfs(tree.root_node, array("q"))
    return out


def build_decode_export(tree, coordinator) -> DecodeExport:
    page_size = tree.page_size
    chains = _walk_retained(tree, coordinator, page_size)
    _, _, host_items = coordinator.mem_pool_host.get_contiguous_buf_infos()
    host_item = int(host_items[0])
    idx_pool = coordinator.mem_pool_device
    index_item = idx_pool.index_k_with_scale_buffer[0].shape[-1]
    return DecodeExport(
        chains=chains,
        host_item_len=host_item,
        index_item_len=int(index_item),
        layer_num=idx_pool.layer_num,
    )


def decode_pool_specs(coordinator) -> Dict[str, dict]:
    """Source-side buffer specs for the push: host latent (DRAM) + index keys (VRAM)."""
    host_ptrs, host_lens, host_items = coordinator.mem_pool_host.get_contiguous_buf_infos()
    dev = coordinator.mem_pool_device
    idx_ptrs = [int(b.data_ptr()) for b in dev.index_k_with_scale_buffer]
    idx_lens = [b.numel() for b in dev.index_k_with_scale_buffer]
    return {
        "host_latent": {
            "ptrs": [int(p) for p in host_ptrs],
            "lens": [int(l) for l in host_lens],
            "item_len": int(host_items[0]),
            "mem_kind": "DRAM",
        },
        "index_keys": {
            "ptrs": idx_ptrs,
            "lens": idx_lens,
            "item_len": int(dev.index_k_with_scale_buffer[0].shape[-1]),
            "mem_kind": "VRAM",
        },
    }


def flat_pages(export: DecodeExport) -> Tuple[np.ndarray, np.ndarray]:
    """Canonical order: (old logical pages, host row pages) across all chains."""
    if not export.chains:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return (
        np.concatenate([c.old_logical_pages for c in export.chains]),
        np.concatenate([c.host_row_pages for c in export.chains]),
    )


# ---------------------------------------------------------------------------
# Heir-side allocation + import
# ---------------------------------------------------------------------------


def alloc_heir_decode(coordinator, n_tokens: int, page_size: int):
    """Allocate heir logical indices and host rows for ``n_tokens``.

    Returns (logical_slots, host_slots) in canonical order (whole pages)."""
    assert n_tokens % page_size == 0
    logical = coordinator.token_to_kv_pool_allocator.logical_attn_allocator.alloc(
        n_tokens
    )
    if logical is None:
        raise RuntimeError("heir logical pool too small for handover")
    host = coordinator.mem_pool_host.alloc(n_tokens)
    if host is None:
        coordinator.token_to_kv_pool_allocator.logical_attn_allocator.free(logical)
        raise RuntimeError("heir host pool too small for handover")
    return logical.to(torch.int64), host.to(torch.int64)


def import_decode(
    tree,
    coordinator,
    export: DecodeExport,
    logical_slots: torch.Tensor,
    host_slots: torch.Tensor,
) -> Dict:
    """Insert chains as plain radix values (heir logical indices) and retain rows."""
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    page_size = tree.page_size
    stats = {"chains": 0, "tokens": 0}
    offset = 0
    for chain in export.chains:
        seg_tokens = chain.n_seg_pages * page_size
        logical = logical_slots[offset : offset + seg_tokens].clone()
        host = host_slots[offset : offset + seg_tokens].clone()
        offset += seg_tokens
        key = RadixKey(
            array("q", chain.tokens),
            extra_key=chain.extra_key,
            cache_salt=chain.cache_salt,
        ).page_aligned(page_size)
        assert len(key) >= seg_tokens
        # Plain insert needs the FULL path's logical indices (the parent path's
        # values already sit in the heir tree from earlier chains; reconstruct
        # them by walking, then append this chain's segment).
        path_vals, matched = _path_values(tree, key)
        want_matched = len(key) - seg_tokens
        assert matched >= want_matched, (
            f"heir tree path shorter ({matched}) than chain's parent path "
            f"({want_matched}); import order broken"
        )
        if want_matched == 0:
            value = logical
        else:
            value = torch.cat([path_vals[:want_matched].to(logical.device), logical])
        tree.insert(InsertParams(key=key, value=value, priority=0))
        coordinator.retain_rows(logical, host)
        stats["chains"] += 1
        stats["tokens"] += seg_tokens
    assert offset == len(logical_slots) == len(host_slots)
    return stats


def _path_values(tree, key):
    """Concatenated node.value along the longest existing path matching
    ``key``; returns (values, matched_tokens)."""
    page_size = tree.page_size
    node = tree.root_node
    child_key = key.child_key(page_size)
    vals = []
    matched = 0
    while len(key) > 0 and child_key in node.children:
        node = node.children[child_key]
        plen = node.key.match(key, page_size=page_size)
        if node.value is not None and len(node.value) > 0:
            vals.append(node.value[:plen].to(torch.int64))
        matched += plen
        if plen < len(node.key):
            break
        key = key[plen:]
        if len(key):
            child_key = key.child_key(page_size)
    out = torch.cat(vals) if vals else torch.empty(0, dtype=torch.int64, device="cuda" if torch.cuda.is_available() else "cpu")
    return out, matched


# ---------------------------------------------------------------------------
# Local (in-process) landing for unit tests: memcpy stand-in for the wire.
# ---------------------------------------------------------------------------


def local_push_decode(coordinator_src, coordinator_dst, export: DecodeExport):
    """Copy source pages into the heir's pools (canonical order). Returns
    (logical_slots, host_slots, dst_logical_pages, dst_host_pages)."""
    page_size = 64  # coordinator page size
    old_logical, host_rows = flat_pages(export)
    n_pages = len(old_logical)
    if n_pages == 0:
        return None, None, None
    logical_slots, host_slots = alloc_heir_decode(
        coordinator_dst, n_pages * page_size, page_size
    )
    dst_logical = (
        logical_slots.cpu().view(-1, page_size)[:, 0] // page_size
    ).to(torch.int64).numpy()
    dst_host = (
        host_slots.cpu().view(-1, page_size)[:, 0] // page_size
    ).to(torch.int64).numpy()

    # host latent: src host row pages -> dst host row pages
    item = export.host_item_len
    src_rows_t = torch.as_tensor(host_rows, dtype=torch.long)
    dst_rows_t = torch.as_tensor(dst_host, dtype=torch.long)
    for bsrc, bdst in zip(
        coordinator_src.mem_pool_host.data_refs,
        coordinator_dst.mem_pool_host.data_refs,
    ):
        src2d = bsrc.view(torch.uint8).view(-1, item)
        dst2d = bdst.view(torch.uint8).view(-1, item)
        dst2d.index_copy_(0, dst_rows_t, src2d.index_select(0, src_rows_t))

    # index keys: src old logical pages -> dst logical pages (device->device)
    dev_src = coordinator_src.mem_pool_device
    dev_dst = coordinator_dst.mem_pool_device
    idx_item = export.index_item_len
    src_idx_t = torch.as_tensor(old_logical, dtype=torch.long, device=dev_src.device)
    dst_idx_t = torch.as_tensor(dst_logical, dtype=torch.long, device=dev_dst.device)
    for bsrc, bdst in zip(
        dev_src.index_k_with_scale_buffer, dev_dst.index_k_with_scale_buffer
    ):
        bdst.view(-1, idx_item).index_copy_(
            0, dst_idx_t, bsrc.view(-1, idx_item).index_select(0, src_idx_t)
        )
    torch.cuda.synchronize()
    return logical_slots, host_slots, dst_logical, dst_host


# ---------------------------------------------------------------------------
# Heir-side allocation + import
# ---------------------------------------------------------------------------


def alloc_heir_decode(coordinator, n_tokens: int, page_size: int):
    """Allocate heir logical indices and host rows for ``n_tokens``.

    Returns (logical_slots, host_slots) in canonical order (whole pages)."""
    assert n_tokens % page_size == 0
    logical = coordinator.token_to_kv_pool_allocator.logical_attn_allocator.alloc(
        n_tokens
    )
    if logical is None:
        raise RuntimeError("heir logical pool too small for handover")
    host = coordinator.mem_pool_host.alloc(n_tokens)
    if host is None:
        coordinator.token_to_kv_pool_allocator.logical_attn_allocator.free(logical)
        raise RuntimeError("heir host pool too small for handover")
    return logical.to(torch.int64), host.to(torch.int64)


def import_decode(
    tree,
    coordinator,
    export: DecodeExport,
    logical_slots: torch.Tensor,
    host_slots: torch.Tensor,
) -> Dict:
    """Insert chains as plain radix values (heir logical indices) and retain rows."""
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    page_size = tree.page_size
    stats = {"chains": 0, "tokens": 0}
    offset = 0
    for chain in export.chains:
        seg_tokens = chain.n_seg_pages * page_size
        logical = logical_slots[offset : offset + seg_tokens].clone()
        host = host_slots[offset : offset + seg_tokens].clone()
        offset += seg_tokens
        key = RadixKey(
            array("q", chain.tokens),
            extra_key=chain.extra_key,
            cache_salt=chain.cache_salt,
        ).page_aligned(page_size)
        assert len(key) >= seg_tokens
        # Plain insert needs the FULL path's logical indices (the parent path's
        # values already sit in the heir tree from earlier chains; reconstruct
        # them by walking, then append this chain's segment).
        path_vals, matched = _path_values(tree, key)
        want_matched = len(key) - seg_tokens
        assert matched >= want_matched, (
            f"heir tree path shorter ({matched}) than chain's parent path "
            f"({want_matched}); import order broken"
        )
        if want_matched == 0:
            value = logical
        else:
            value = torch.cat([path_vals[:want_matched].to(logical.device), logical])
        tree.insert(InsertParams(key=key, value=value, priority=0))
        coordinator.retain_rows(logical, host)
        stats["chains"] += 1
        stats["tokens"] += seg_tokens
    assert offset == len(logical_slots) == len(host_slots)
    return stats


def _path_values(tree, key):
    """Concatenated node.value along the longest existing path matching
    ``key``; returns (values, matched_tokens)."""
    page_size = tree.page_size
    node = tree.root_node
    child_key = key.child_key(page_size)
    vals = []
    matched = 0
    while len(key) > 0 and child_key in node.children:
        node = node.children[child_key]
        plen = node.key.match(key, page_size=page_size)
        if node.value is not None and len(node.value) > 0:
            vals.append(node.value[:plen].to(torch.int64))
        matched += plen
        if plen < len(node.key):
            break
        key = key[plen:]
        if len(key):
            child_key = key.child_key(page_size)
    out = torch.cat(vals) if vals else torch.empty(0, dtype=torch.int64, device="cuda" if torch.cuda.is_available() else "cpu")
    return out, matched


# ---------------------------------------------------------------------------
# Local (in-process) landing for unit tests: memcpy stand-in for the wire.
# ---------------------------------------------------------------------------


def local_push_decode(coordinator_src, coordinator_dst, export: DecodeExport):
    """Copy source pages into the heir's pools (canonical order). Returns
    (logical_slots, host_slots, dst_logical_pages, dst_host_pages)."""
    page_size = 64  # coordinator page size
    old_logical, host_rows = flat_pages(export)
    n_pages = len(old_logical)
    if n_pages == 0:
        return None, None, None
    logical_slots, host_slots = alloc_heir_decode(
        coordinator_dst, n_pages * page_size, page_size
    )
    dst_logical = (
        logical_slots.cpu().view(-1, page_size)[:, 0] // page_size
    ).to(torch.int64).numpy()
    dst_host = (
        host_slots.cpu().view(-1, page_size)[:, 0] // page_size
    ).to(torch.int64).numpy()

    # host latent: src host row pages -> dst host row pages
    src_host_ptrs = [
        int(b.data_ptr()) for b in coordinator_src.mem_pool_host.data_refs
    ]
    dst_host_ptrs = [
        int(b.data_ptr()) for b in coordinator_dst.mem_pool_host.data_refs
    ]
    item = export.host_item_len
    src_rows_t = torch.as_tensor(host_rows, dtype=torch.long)
    dst_rows_t = torch.as_tensor(dst_host, dtype=torch.long)
    for sp, dp in zip(src_host_ptrs, dst_host_ptrs):
        src2d = _ptr_view(sp, item * (coordinator_src.mem_pool_host.size // page_size), item)
        dst2d = _ptr_view(dp, item * (coordinator_dst.mem_pool_host.size // page_size), item)
        dst2d.index_copy_(0, dst_rows_t, src2d.index_select(0, src_rows_t))

    # index keys: src old logical pages -> dst logical pages (device->device)
    dev_src = coordinator_src.mem_pool_device
    dev_dst = coordinator_dst.mem_pool_device
    idx_item = export.index_item_len
    src_idx_t = torch.as_tensor(old_logical, dtype=torch.long, device=dev_src.device)
    dst_idx_t = torch.as_tensor(dst_logical, dtype=torch.long, device=dev_dst.device)
    for bsrc, bdst in zip(
        dev_src.index_k_with_scale_buffer, dev_dst.index_k_with_scale_buffer
    ):
        bdst.view(-1, idx_item).index_copy_(
            0, dst_idx_t, bsrc.view(-1, idx_item).index_select(0, src_idx_t)
        )
    torch.cuda.synchronize()
    return logical_slots, host_slots, dst_logical, dst_host


def _ptr_view(ptr: int, nbytes: int, item: int) -> torch.Tensor:
    return torch.frombuffer(
        bytearray(0), dtype=torch.uint8
    ) if False else _PtrView(ptr, nbytes).view(item)


class _PtrView:
    """Minimal ctypes-backed uint8 buffer view over a raw pointer."""

    def __init__(self, ptr: int, nbytes: int):
        import ctypes

        self._buf = (ctypes.c_char * nbytes).from_address(ptr)
        self._t = torch.frombuffer(
            bytearray(0), dtype=torch.uint8
        ) if False else torch.tensor([], dtype=torch.uint8)

    def view(self, item: int) -> torch.Tensor:
        import ctypes

        # torch view over the ctypes buffer
        t = torch.frombuffer(self._buf, dtype=torch.uint8)
        return t.view(-1, item)


# ---------------------------------------------------------------------------
# Serialization (mirror of the prefill-arm manifest format)
# ---------------------------------------------------------------------------


def decode_export_to_bytes(export: DecodeExport, page_size: int) -> bytes:
    import json as _json

    header = {
        "n_chains": len(export.chains),
        "host_item_len": export.host_item_len,
        "index_item_len": export.index_item_len,
        "layer_num": export.layer_num,
        "page_size": page_size,
    }
    hb = _json.dumps(header).encode()
    parts = [np.uint32(len(hb)).tobytes(), hb]
    for c in export.chains:
        tokens = np.frombuffer(c.tokens, dtype=np.int64)
        parts.append(np.uint64(len(tokens)).tobytes())
        parts.append(tokens.tobytes())
        parts.append(np.uint64(c.n_seg_pages).tobytes())
        parts.append(c.old_logical_pages.astype(np.int64).tobytes())
        parts.append(c.host_row_pages.astype(np.int64).tobytes())
    return b"".join(parts)


def bytes_to_decode_export(b: bytes) -> Tuple[DecodeExport, int]:
    import json as _json
    from array import array as _array

    off = 0

    def take(n):
        nonlocal off
        assert off + n <= len(b), "decode manifest truncated"
        out = b[off : off + n]
        off += n
        return out

    hlen = int(np.frombuffer(take(4), dtype=np.uint32)[0])
    header = _json.loads(take(hlen).decode())
    export = DecodeExport(
        host_item_len=int(header["host_item_len"]),
        index_item_len=int(header["index_item_len"]),
        layer_num=int(header["layer_num"]),
    )
    for _ in range(header["n_chains"]):
        ntok = int(np.frombuffer(take(8), dtype=np.uint64)[0])
        tokens = _array("q")
        tokens.frombytes(take(ntok * 8))
        n_seg = int(np.frombuffer(take(8), dtype=np.uint64)[0])
        old_logical = np.frombuffer(take(n_seg * 8), dtype=np.int64).copy()
        host_rows = np.frombuffer(take(n_seg * 8), dtype=np.int64).copy()
        export.chains.append(
            DecodeChain(
                tokens=tokens,
                n_seg_pages=n_seg,
                old_logical_pages=old_logical,
                host_row_pages=host_rows,
            )
        )
    assert off == len(b)
    return export, int(header["page_size"])
