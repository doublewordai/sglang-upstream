"""Prefill-arm generation handover: export / push / import.

Prefill arm state (HiRadixCache + HostPoolGroup[MLATokenToKVPoolHost +
DSAIndexerPoolHost], write_through): radix nodes whose ``host_value`` is set
are backed by host pool rows; one set of host slot indices addresses both the
KV anchor pool and the indexer sidecar (page row = slot // page_size).

Export walks the tree depth-first and emits one ChainRecord per backuped
*node* (page-aligned key segment + its source page rows), in pre-order so a
chain's parent path is always present before the chain itself. Import
bulk-inserts host-only nodes (value=None, host_value=heir rows) via the same
algorithm as HiRadixCache._insert_helper_host.

Two source modes:
  * ``direct`` – push reads the exporter's pool rows in place. Exported nodes
    are protected with ``host_ref_counter`` for the push window (host
    eviction skips protected nodes). Page runs may be short if the exporter's
    host pool fragmented.
  * ``staged`` – the exporter first gathers all exported pages into fresh
    contiguous pinned per-layer staging buffers and pushes from there:
    race-free against concurrent tree mutation, maximal run lengths.
    Protections are released right after the gather.

Only ``layer_first`` host pools are supported (prod layout); page_first
staging/descriptor addressing is not implemented.
"""

from __future__ import annotations

import logging
import threading
import time
from array import array
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.mem_cache.handover.manifest import (
    ChainRecord,
    HandoverManifest,
    PoolSpec,
    config_fingerprint,
    page_checksums,
)
from sglang.srt.mem_cache.utils import get_hash_str

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pool helpers (shared by export and import)
# ---------------------------------------------------------------------------


def _host_pools(host_group) -> Dict[str, object]:
    """name -> host pool. Accepts a HostPoolGroup or a bare pool."""
    entries = getattr(host_group, "entries", None)
    if entries is not None:
        return {str(e.name): e.host_pool for e in entries}
    return {"KV": host_group}


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def _per_layer_host_buffers(pool) -> List[torch.Tensor]:
    """Contiguous per-layer uint8 buffers of one host pool (layer_first)."""
    layout = getattr(pool, "layout", "layer_first")
    if layout != "layer_first":
        raise NotImplementedError(
            f"handover supports layer_first host pools, got {layout!r}"
        )
    # MLATokenToKVPoolHost/DSAIndexerPoolHost layer_first keep per-layer
    # contiguous views in data_refs / index_k_data_refs.
    if hasattr(pool, "data_refs"):
        bufs = list(pool.data_refs)
    elif hasattr(pool, "index_k_data_refs"):
        bufs = list(pool.index_k_data_refs)
    elif hasattr(pool, "get_hybrid_pool_buffer"):
        bufs = pool.get_hybrid_pool_buffer()
    elif isinstance(pool.kv_buffer, (list, tuple)):
        bufs = list(pool.kv_buffer)
    else:
        bufs = [pool.kv_buffer]
    if isinstance(bufs[0], list):  # nested (some stacks return per-layer lists)
        bufs = [b for sub in bufs for b in sub]
    return [_as_u8(b) for b in bufs]


def _buf_infos(pool) -> Tuple[List[int], List[int], int]:
    ptrs, lens, items = pool.get_contiguous_buf_infos()
    assert len(set(items)) == 1, "pool with non-uniform item lens"
    return [int(p) for p in ptrs], [int(l) for l in lens], int(items[0])


def _pool_item_len(pool) -> int:
    return _buf_infos(pool)[2]


def tree_pools(tree) -> Dict[str, object]:
    return _host_pools(tree.token_to_kv_pool_host)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def iter_backuped_chains(root):
    """All backuped nodes with their FULL key path from root, DFS pre-order.

    Each entry is (full_path_tokens: array('q'), node). Full-path keys make
    every chain self-contained on the heir: the insert walk from root
    consumes the parent path and appends the node's own segment (exactly the
    ``_insert_helper_host(last_host_node, suffix_key, ...)`` shape of the
    storage-prefetch path). Split-safe regardless of insertion order.
    """
    out = []

    def dfs(node, path_tokens):
        for c in node.children.values():
            if len(c.key) == 0:
                continue
            seg = c.key.raw_token_ids()
            full = path_tokens + seg if len(path_tokens) else seg
            if c.backuped:
                out.append((full, c))
            dfs(c, full)

    dfs(root, array("q"))
    return out


class PrefillExport:
    """Everything the old generation needs to serve one handover push."""

    def __init__(self, manifest, pool_specs, src_pages, nodes, staging):
        self.manifest = manifest
        self.pool_specs = pool_specs  # name -> PoolSpec (ptrs point at source)
        self.src_pages = src_pages  # canonical-order np.int64 page rows (source)
        self.nodes = nodes  # exported nodes (protections held while direct)
        self.staging = staging  # per-pool staging tensors (staged mode) or {}

    @classmethod
    def build(
        cls,
        tree,
        model_path: str,
        staged: bool = False,
        with_checksums: bool = True,
    ) -> "PrefillExport":
        # Flush any completed write-through acks so host_value rows are final.
        writing_check = getattr(tree, "writing_check", None)
        if writing_check is not None:
            try:
                writing_check()
            except Exception:
                logger.exception("writing_check during handover export; continuing")

        page_size = tree.page_size
        pools = tree_pools(tree)
        chains_and_nodes = iter_backuped_chains(tree.root_node)
        nodes = [n for _, n in chains_and_nodes]
        chains: List[ChainRecord] = []
        for full_tokens, n in chains_and_nodes:
            if n.key.is_bigram:
                raise NotImplementedError("bigram (eagle) keys not supported yet")
            hv = n.host_value
            assert hv is not None and len(hv) % page_size == 0, (
                "node host_value not page-aligned"
            )
            seg_pages = len(hv) // page_size
            assert len(full_tokens) >= len(hv)
            page_rows = (hv.view(-1, page_size)[:, 0] // page_size).to(torch.int64).numpy()
            hashes: Optional[List[str]] = None
            if n.hash_value is not None and len(n.hash_value) > 0:
                hashes = list(n.hash_value)
                assert len(hashes) == seg_pages
            chains.append(
                ChainRecord(
                    tokens=full_tokens,
                    n_seg_pages=seg_pages,
                    page_rows=page_rows,
                    extra_key=n.key.extra_key,
                    cache_salt=n.key.cache_salt,
                    page_hashes=hashes,
                )
            )
            n.protect_host()  # holds through push (direct) or gather (staged)

        manifest = HandoverManifest(
            version=1,
            fingerprint=cls.fingerprint(tree, model_path),
            page_size=page_size,
            staged=staged,
            chains=chains,
            checksums=None,
        )

        staging: Dict[str, List[torch.Tensor]] = {}
        if staged:
            staging = cls._stage(tree, manifest, pools)
            src_pages = manifest.flat_page_rows()  # arange over staging rows
        else:
            src_pages = manifest.flat_page_rows()

        pool_specs: Dict[str, PoolSpec] = {}
        for name, pool in pools.items():
            if staged:
                bufs = staging[name]
                ptrs = [int(b.data_ptr()) for b in bufs]
                lens = [b.numel() for b in bufs]
                item_len = _pool_item_len(pool)
            else:
                ptrs, lens, item_len = _buf_infos(pool)
            pool_specs[name] = PoolSpec(
                name=name,
                layer_num=len(ptrs),
                item_len=item_len,
                data_ptrs=ptrs,
                data_lens=lens,
                mem_kind="DRAM",
            )

        if with_checksums and manifest.num_pages > 0:
            manifest.checksums = {}
            for name, pool in pools.items():
                bufs = staging[name] if staged else _per_layer_host_buffers(pool)
                manifest.checksums[name] = page_checksums(
                    bufs, src_pages, pool_specs[name].item_len
                )

        export = cls(manifest, pool_specs, src_pages, nodes, staging)
        if staged:
            # staging holds a private copy; release pool protections now
            export.release_protections()
        return export

    # -- staging -----------------------------------------------------------

    @staticmethod
    def _stage(tree, manifest, pools) -> Dict[str, List[torch.Tensor]]:
        """Gather exported pages into per-pool contiguous pinned staging buffers.

        Staging page j holds canonical page j; the push uses src pages
        0..num_pages-1 with each pool's own item_len.
        """
        rows = manifest.flat_page_rows()  # source rows (pre-stage)
        n = len(rows)
        out: Dict[str, List[torch.Tensor]] = {}
        rows_t = torch.as_tensor(rows, dtype=torch.long)
        for name, pool in pools.items():
            bufs = _per_layer_host_buffers(pool)
            item_len = _pool_item_len(pool)
            staging = [
                torch.empty((n, item_len), dtype=torch.uint8, pin_memory=True)
                for _ in bufs
            ]
            errors = []

            def copy_layer(i: int) -> None:
                try:
                    staging[i].copy_(_gather_rows(bufs[i], rows_t, item_len))
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            threads = [
                threading.Thread(target=copy_layer, args=(i,)) for i in range(len(bufs))
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if errors:
                raise errors[0]
            out[name] = staging
        return out

    # -- fingerprint ---------------------------------------------------------

    @staticmethod
    def fingerprint(tree, model_path: str) -> str:
        pools = tree_pools(tree)
        kv_pool = pools.get("KV", next(iter(pools.values())))
        dev = getattr(tree, "kv_cache", None)
        store_dtype = (
            str(getattr(dev, "store_dtype", "unknown"))
            if dev is not None
            else str(getattr(kv_pool, "dtype", "unknown"))
        )
        indexer_spt = None
        idx_pool = pools.get("INDEXER")
        if idx_pool is not None:
            indexer_spt = int(getattr(idx_pool, "indexer_size_per_token", -1))
        return config_fingerprint(
            model_path=str(model_path),
            page_size=tree.page_size,
            layer_num=int(getattr(kv_pool, "layer_num", -1)),
            kv_cache_dim=int(getattr(kv_pool, "kv_cache_dim", -1)),
            store_dtype=store_dtype,
            indexer_size_per_token=indexer_spt,
            pool_names=sorted(pools.keys()),
            extra={"tree": type(tree).__name__},
        )

    # -- lifecycle -----------------------------------------------------------

    def release_protections(self) -> None:
        for n in self.nodes:
            try:
                n.release_host()
            except RuntimeError:
                logger.warning("release_host on node %s with zero counter", n.id)
        self.nodes = []


def _gather_rows(buf: torch.Tensor, rows: torch.Tensor, item_len: int) -> torch.Tensor:
    """Gather ``rows`` (page-row ids) of ``item_len`` bytes from a flat uint8 buffer."""
    if len(rows) == 0:
        return torch.empty(0, item_len, dtype=torch.uint8)
    flat = buf.view(-1)
    idx = rows[:, None] * item_len + torch.arange(item_len, dtype=torch.long)[None, :]
    return flat[idx]


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def alloc_heir_rows(tree, num_tokens: int) -> torch.Tensor:
    """Allocate ``num_tokens`` host slots on the heir's anchor pool."""
    page_size = tree.page_size
    assert num_tokens % page_size == 0
    slots = tree.token_to_kv_pool_host.alloc(num_tokens)
    if slots is None:
        raise RuntimeError(
            f"heir host pool too small for handover: need {num_tokens} tokens"
        )
    return slots


def heir_page_rows(slots: torch.Tensor, page_size: int) -> np.ndarray:
    return (slots.view(-1, page_size)[:, 0] // page_size).to(torch.int64).numpy()


def import_manifest(
    tree,
    manifest: HandoverManifest,
    slots: torch.Tensor,
    verify_checksums_fn=None,
    need_hashes: Optional[bool] = None,
) -> Dict:
    """Bulk-insert the manifest's chains as host-only radix nodes.

    ``slots`` is the heir's allocated host slot tensor (canonical order).
    ``verify_checksums_fn(pool_name) -> np.ndarray`` computes per-page
    checksums over the heir's landed rows; compared against the manifest's
    checksums (all pages) when both exist, mismatch raises.
    ``need_hashes``: whether the heir's tree wants hash values (storage/kv
    events). Defaults to the tree's own flags. When the manifest carries
    hashes they are reused; otherwise recomputed from the token path.
    """
    from sglang.srt.mem_cache.radix_cache import RadixKey

    page_size = tree.page_size
    if need_hashes is None:
        need_hashes = bool(
            getattr(tree, "enable_storage", False)
            or getattr(tree, "enable_kv_cache_events", False)
        )
    stats = {"chains": 0, "tokens": 0, "pages": 0, "checksum_mismatches": 0}

    if verify_checksums_fn is not None and manifest.checksums:
        for name, want in manifest.checksums.items():
            got = verify_checksums_fn(name)
            if got is None:
                continue
            bad = int((got != want).sum())
            stats["checksum_mismatches"] += bad
            if bad:
                first = int(np.argmax(got != want))
                raise RuntimeError(
                    f"handover checksum mismatch in pool {name}: {bad} pages "
                    f"differ (first at canonical page {first})"
                )

    offset = 0
    for chain in manifest.chains:
        seg_tokens = chain.n_seg_pages * page_size
        rows = slots[offset : offset + seg_tokens].clone()
        offset += seg_tokens
        key = RadixKey(
            array("q", chain.tokens),
            extra_key=chain.extra_key,
            cache_salt=chain.cache_salt,
        )
        hashes = None
        if need_hashes:
            hashes = _recompute_chain_hashes(tree, chain)
            if chain.page_hashes is not None:
                # verify the manifest's segment hashes against the tail of
                # the recomputed full-path chain (integrity check)
                n_seg = len(chain.page_hashes)
                assert hashes[-n_seg:] == list(chain.page_hashes), (
                    "hash mismatch between manifest and recomputed chain"
                )
        _insert_host_chain(tree, key, rows, hashes)
        stats["chains"] += 1
        stats["tokens"] += seg_tokens
        stats["pages"] += chain.n_seg_pages
    assert offset == len(slots), (offset, len(slots))
    return stats


def _recompute_chain_hashes(tree, chain) -> Optional[List[str]]:
    """Chained per-page hashes for the chain from the parent's last hash."""
    page_size = tree.page_size
    n_pages = len(chain.tokens) // page_size
    if n_pages == 0:
        return None
    prior = _parent_last_hash(tree, chain)
    hashes = get_hash_str(chain.tokens, prior, page_size=page_size)
    if not isinstance(hashes, list):
        hashes = [hashes]
    return hashes


def _parent_last_hash(tree, chain) -> Optional[str]:
    """Last page hash along the longest existing heir-tree path matching the key."""
    from sglang.srt.mem_cache.radix_cache import RadixKey

    page_size = tree.page_size
    rk = RadixKey(
        array("q", chain.tokens),
        extra_key=chain.extra_key,
        cache_salt=chain.cache_salt,
    )
    node = tree.root_node
    last = None
    child_key = rk.child_key(page_size)
    while len(rk) > 0 and child_key in node.children:
        child = node.children[child_key]
        prefix_len = child.key.match(rk, page_size=page_size)
        if prefix_len == 0:
            break
        if child.hash_value:
            last = child.hash_value[-1]
        if prefix_len < len(child.key):
            break
        rk = rk[prefix_len:]
        node = child
        if len(rk):
            child_key = rk.child_key(page_size)
    return last


def _insert_host_chain(tree, key, host_value, hash_value) -> int:
    """Insert one host-only chain whose ``key`` is the FULL path from root.

    Walks the heir tree from the root consuming the parent path (splitting
    partially-matched nodes exactly like the storage-prefetch path), then
    appends the remaining segment (== ``host_value``'s rows) as a host-only
    node under the final node. ``hash_value`` (if given) is the full-path
    chained page-hash list; the walk slices it down to the segment.
    """
    from sglang.srt.mem_cache.radix_cache import TreeNode

    page_size = tree.page_size
    node = tree.root_node
    node.last_access_time = time.monotonic()
    assert len(host_value) % page_size == 0
    if len(key) == 0:
        return 0

    child_key = key.child_key(page_size)
    matched_length = 0
    while len(key) > 0 and child_key in node.children.keys():
        node = node.children[child_key]
        node.last_access_time = time.monotonic()
        prefix_len = node.key.match(key, page_size=page_size)
        key = key[prefix_len:]
        if hash_value is not None:
            hash_value = hash_value[prefix_len // page_size :]
        matched_length += prefix_len

        if prefix_len < len(node.key):
            node = tree._split_node(node.key, node, prefix_len)

        if len(key):
            child_key = key.child_key(page_size)

    if len(key) == 0:
        # full path already present (import of a strict-prefix chain after a
        # longer one); nothing new to append
        return matched_length
    assert len(key) == len(host_value), (
        f"handover insert walk mismatch: {len(key)} remaining tokens but "
        f"{len(host_value)} segment rows (parent path incomplete?)"
    )
    new_node = TreeNode(priority=node.priority)
    new_node.parent = node
    new_node.key = key
    new_node.value = None
    new_node.host_value = host_value.clone()
    new_node.hash_value = hash_value
    node.children[child_key] = new_node
    _host_leaf_status(tree, new_node)
    _leaf_status(tree, node)
    _host_leaf_status(tree, node)
    _store_event(tree, new_node)
    return matched_length


def _host_leaf_status(tree, node) -> None:
    fn = getattr(tree, "_update_host_leaf_status", None)
    if fn is not None:
        fn(node)


def _leaf_status(tree, node) -> None:
    fn = getattr(tree, "_update_leaf_status", None)
    if fn is not None:
        fn(node)


def _store_event(tree, node) -> None:
    fn = getattr(tree, "_record_store_event", None)
    if fn is not None and getattr(tree, "enable_kv_cache_events", False):
        from sglang.srt.disaggregation.kv_events import StorageMedium

        fn(node, medium=StorageMedium.CPU)
