"""Session host-page migration between prefill DP ranks.

When the prefix-affinity dispatcher must bounce a warm session to a different
DP rank (load balance, imbalance guard, LRU eviction of the dispatcher index),
the holder rank exports the session's host-tier pages (Full KV + DSA indexer
sidecar) and pushes them to the target rank over NIXL/UCCL WRITE (the UCCL
backend has no working READ; see lane cache-handover). The target imports them
as host-only radix nodes, so the bounced request load_backs the pages and
prefills only its new tokens instead of re-prefilling the whole context.

Components:
- ``SessionMigrationAgent``: one per scheduler process. A background TCP server
  (thread) speaks a small pickle protocol; tree/allocator work is marshalled
  onto the scheduler event-loop thread via ``poll()`` (tree structures are
  single-threaded); NIXL transfers and checksums run on the agent thread.
- ``export_session``: walk the unified radix tree from the best host match to
  the root, concatenating (key, host_value) segments -> manifest.
- Target-side import: allocate host rows (page runs in the anchor pool, shared
  index space with the INDEXER sidecar), receive rows by WRITE, then
  ``UnifiedTreeCore.insert_host`` from the root.

Control flow (driven by the DP controller's decision hook):
  controller -> target agent : ADOPT {tokens, holder_addr}
  target -> holder agent     : EXPORT_META {tokens}       (how many pages?)
  target -> holder agent     : PUSH {dst_rows, target_meta, target_name}
  holder                     : export (protect nodes) + NIXL WRITE + checksums
  target                     : checksum rows, insert_host, free unused rows
  controller <- target agent : {ok, verified, imported_tokens, timings}
"""

from __future__ import annotations

import base64
import json
import logging
import os
import pickle
import socket
import threading
import time
import uuid
from array import array
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.utils import get_hash_str

logger = logging.getLogger(__name__)

MIGRATION_PORT_BASE_ENV = "SGLANG_RANK_MIGRATION_PORT_BASE"
MIGRATION_STATS_FILE_ENV = "SGLANG_RANK_MIGRATION_STATS_FILE"
MIGRATION_TIMEOUT_ENV = "SGLANG_RANK_MIGRATION_TIMEOUT_S"
MIGRATION_MAX_PAGES_PER_XFER_ENV = "SGLANG_RANK_MIGRATION_MAX_PAGES_PER_XFER"

_DEFAULT_TIMEOUT_S = 120.0
_DEFAULT_MAX_PAGES_PER_XFER = 256


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


# --------------------------------------------------------------------------
# TCP helpers (length-prefixed pickle; internal protocol, same code both sides)
# --------------------------------------------------------------------------


def send_msg(sock: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(len(data).to_bytes(8, "little") + data)


def recv_msg(sock: socket.socket, timeout: Optional[float] = None) -> Any:
    if timeout is not None:
        sock.settimeout(timeout)
    hdr = b""
    while len(hdr) < 8:
        chunk = sock.recv(8 - len(hdr))
        if not chunk:
            raise ConnectionError("peer closed while reading header")
        hdr += chunk
    n = int.from_bytes(hdr, "little")
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("peer closed while reading body")
        data += chunk
    if timeout is not None:
        sock.settimeout(None)
    return pickle.loads(data)


def tcp_rpc(addr: Tuple[str, int], msg: Any, timeout: float) -> Any:
    """One request/response round to a migration agent."""
    sock = socket.create_connection(addr, timeout=timeout)
    try:
        sock.settimeout(None)  # drop create_connection's timeout
        send_msg(sock, msg)
        return recv_msg(sock, timeout)
    finally:
        sock.close()


# --------------------------------------------------------------------------
# Scheduler-thread executor: tree ops must run on the scheduler event loop.
# --------------------------------------------------------------------------


class SchedulerExecutor:
    def __init__(self) -> None:
        self._q: List[Tuple[Callable[[], Any], threading.Event, list]] = []
        self._lock = threading.Lock()

    def poll(self) -> None:
        """Run pending callables; called from the scheduler event loop."""
        with self._lock:
            batch = self._q
            self._q = []
        for fn, done, out in batch:
            try:
                out.append(fn())
                out.append(None)
            except BaseException as e:  # noqa: BLE001
                out.append(None)
                out.append(e)
            finally:
                done.set()

    def run(self, fn: Callable[[], Any], timeout: Optional[float]) -> Any:
        done = threading.Event()
        out: list = []
        with self._lock:
            self._q.append((fn, done, out))
        if not done.wait(timeout):
            raise TimeoutError("scheduler executor timed out")
        result, err = out[0], out[1]
        if err is not None:
            raise err
        return result


# --------------------------------------------------------------------------
# Host-pool geometry (works for the DSA stack: KV anchor + INDEXER sidecar).
# Both hicache layouts are supported:
#   layer_first: L per-layer regions; a page row of layer l is a contiguous
#     item-bytes chunk; the flat dlist index is l * rows_per_layer + page.
#   page_first: ONE region per pool; a page row is one contiguous chunk that
#     spans ALL layers; the flat dlist index is just the page id.
# --------------------------------------------------------------------------


class PoolGeom:
    """Per-pool NIXL geometry: base pointers + page-row descriptor math.

    A page row (64 host slots) is `span` contiguous chunks:
      layer_first: one chunk per layer, base[l] + page * item
      page_first:  one chunk spanning all layers, base[0] + page * item
    """

    def __init__(self, name: str, bases, item: int, rows_per_layer: int, span: int, cksum_mats, n_descs: int):
        self.name = name
        self.bases = [int(b) for b in bases]   # region base ptrs (len == span for layer_first)
        self.item = int(item)                  # bytes per page-row chunk
        self.rows_per_layer = int(rows_per_layer)
        self.span = int(span)                  # chunks per page row
        self.cksum_mats = cksum_mats           # [(tensor (n_rows, row_bytes) uint8)]
        self.n_descs = int(n_descs)

    def page_descs(self, pages: np.ndarray) -> np.ndarray:
        """(span*len(pages), 3) uint64 [addr, len, 0] descs for these pages."""
        return _spec_page_descs(self.spec(), pages)

    def spec(self) -> dict:
        return {
            "bases": self.bases,
            "item": self.item,
            "rows_per_layer": self.rows_per_layer,
            "span": self.span,
            "n_descs": self.n_descs,
        }

    def checksum(self, pages: np.ndarray) -> int:
        total = 0
        rows = torch.from_numpy(pages.astype(np.int64))
        for mat in self.cksum_mats:
            gathered = mat[rows].contiguous()
            if gathered.dtype != torch.uint8:
                gathered = gathered.view(torch.uint8)
            total += int(torch.sum(gathered.view(torch.int64)).item())
        return total

    def page_bytes(self) -> int:
        return self.item * self.span

    def fingerprint(self) -> dict:
        return {
            "n_descs": self.n_descs,
            "span": self.span,
            "item_bytes": self.item,
        }


def _spec_page_descs(spec: dict, pages: np.ndarray) -> np.ndarray:
    """Build (span*len(pages), 3) DRAM descs from a geometry spec + page ids."""
    p = np.asarray(pages, dtype=np.uint64)
    item = np.uint64(spec["item"])
    if spec["span"] == 1:
        addrs = p * item + np.uint64(spec["bases"][0])
        lens = np.full(len(p), spec["item"], dtype=np.uint64)
        return np.column_stack([addrs, lens, np.zeros(len(p), dtype=np.uint64)])
    chunks = []
    for base in spec["bases"]:
        addrs = p * item + np.uint64(base)
        chunks.append(
            np.column_stack(
                [addrs, np.full(len(p), spec["item"], dtype=np.uint64), np.zeros(len(p), dtype=np.uint64)]
            )
        )
    return np.vstack(chunks)


def _pool_geoms(tree_cache: Any) -> Dict[str, PoolGeom]:
    group = tree_cache.host_pool_group
    if group is None:
        raise RuntimeError("session migration needs a hierarchical cache")
    geoms: Dict[str, PoolGeom] = {}

    kv = group.get_pool(PoolName.KV)
    page_size = kv.page_size
    if kv.layout == "layer_first":
        ptrs, lens, items = kv.get_contiguous_buf_infos()
        item = int(items[0])
        n = int(lens[0]) // item
        cksum_mats = [t.reshape(-1, page_size * t.shape[-1]) for t in kv.data_refs]
        geoms["kv"] = PoolGeom("kv", ptrs, item, n, len(ptrs), cksum_mats, n * len(ptrs))
    elif kv.layout == "page_first":
        buf = kv.kv_buffer  # (size, layer_num, 1, kv_cache_dim), contiguous
        if not buf.is_contiguous():
            raise RuntimeError("page_first KV buffer not contiguous")
        row_elems = page_size * (buf.shape[1] * buf.shape[2] * buf.shape[3])
        n = buf.shape[0] // page_size
        flat = buf.reshape(n, row_elems)
        geoms["kv"] = PoolGeom(
            "kv", [buf.data_ptr()], int(row_elems * buf.element_size()), n, 1, [flat], n
        )
    else:
        raise RuntimeError(f"unsupported KV hicache layout {kv.layout}")

    indexer = group.get_pool(PoolName.INDEXER)
    if indexer.layout == "layer_first":
        refs = list(indexer.index_k_data_refs)  # (page_num, stride) per layer
        item = int(indexer.indexer_page_stride_size * indexer.dtype.itemsize)
        n = int(refs[0].numel() * refs[0].element_size()) // item
        cksum_mats = [t.reshape(t.shape[0], -1) for t in refs]
        geoms["indexer"] = PoolGeom(
            "indexer", [t.data_ptr() for t in refs], item, n, len(refs), cksum_mats, n * len(refs)
        )
    elif indexer.layout == "page_first":
        buf = indexer.index_k_with_scale_buffer  # (page_num, layer_num, 1, stride)
        if not buf.is_contiguous():
            raise RuntimeError("page_first indexer buffer not contiguous")
        row_elems = buf.shape[1] * buf.shape[2] * buf.shape[3]
        n = int(buf.shape[0])
        flat = buf.reshape(n, row_elems)
        geoms["indexer"] = PoolGeom(
            "indexer", [buf.data_ptr()], int(row_elems * buf.dtype.itemsize), n, 1, [flat], n
        )
    else:
        raise RuntimeError(f"unsupported indexer layout {indexer.layout}")
    return geoms


def _fingerprint(geoms: Dict[str, PoolGeom], page_size: int) -> dict:
    return {
        "page_size": page_size,
        "pools": {name: g.fingerprint() for name, g in geoms.items()},
    }


# --------------------------------------------------------------------------
# Export / import on the unified radix tree (scheduler thread only)
# --------------------------------------------------------------------------


def segs_available(node, root) -> bool:
    while node is not root:
        if node.component_data[0].host_value is not None:
            return True
        node = node.parent
    return False


def export_session(
    tree_cache: Any,
    token_ids: List[int],
    extra_key: Optional[str] = None,
    cache_salt: Optional[str] = None,
) -> Optional[dict]:
    """Collect the session's host-resident prefix. Runs on the scheduler thread.

    Returns None when the tree holds no host pages for these tokens. Nodes are
    host-locked against eviction; the caller must ``release_export`` afterwards.
    """
    page_size = tree_cache.page_size
    key = RadixKey(
        token_ids=token_ids, extra_key=extra_key, cache_salt=cache_salt
    ).page_aligned(page_size)
    if len(key) == 0:
        return None
    mr = tree_cache.match_prefix(MatchPrefixParams(key=key))
    core = tree_cache.tree_core
    node = mr.best_match_node
    node = core.node_by_id(node) if isinstance(node, int) else node
    root = core.root_node
    if node is root or not segs_available(node, root):
        try:
            avail = tree_cache.cache_controller.mem_pool_host.available_size()
        except Exception:  # noqa: BLE001
            avail = -1
        logger.info(
            "[session-migration] export miss: best_node=%s key_len=%s backuped=%s "
            "host_hit=%s device_hit=%s pool_avail=%s req_tokens=%d",
            getattr(node, "id", None),
            len(node.key) if node is not root else 0,
            node.backuped if node is not root else None,
            mr.host_hit_length,
            len(mr.device_indices),
            avail,
            len(token_ids),
        )
    if node is root:
        return None
    # Walk up from the best match collecting the contiguous backuped chain.
    # A node without host_value ends the chain: everything below it cannot be
    # exported (its pages are missing), so drop the deeper segments.
    segs: List[Any] = []
    n = node
    while n is not root:
        if n.component_data[0].host_value is None:  # ComponentType.FULL == 0
            segs = []
        else:
            segs.append(n)
        n = n.parent
    if not segs:
        return None
    segs.reverse()
    # Drain pending write-through acks so host rows actually hold the data.
    for _ in range(200):
        if all(s.write_through_pending_id is None for s in segs):
            break
        tree_cache.check_hicache_events()
        time.sleep(0.01)
    tokens_out: List[int] = []
    host_vals = []
    for s in segs:
        seg_len = len(s.key)
        tokens_out.extend(s.key.token_ids[:seg_len])
        host_vals.append(s.component_data[0].host_value[:seg_len])
    host = torch.cat(host_vals)
    assert len(host) == len(tokens_out), (
        f"host_value len {len(host)} != key len {len(tokens_out)}"
    )
    # Page-run invariant: each 64-slot page is a consecutive run starting at a
    # row boundary (the allocators guarantee this; verify before RDMA).
    hv = host.detach().clone()
    pages = hv.reshape(-1, page_size)
    starts = pages[:, 0]
    assert bool(((starts % page_size) == 0).all()), "host pages not row-aligned"
    diffs = pages[:, 1:] - pages[:, :-1]
    assert bool((diffs == 1).all()), "host page slots not consecutive"
    rows = (starts // page_size).numpy().astype(np.int64)
    # Protect from host eviction while the push is in flight.
    for s in segs:
        core.inc_host_lock_ref(s.id)
    return {
        "tokens": tokens_out,
        "rows": rows,
        "n_pages": len(rows),
        "protected_ids": [s.id for s in segs],
        "extra_key": extra_key,
        "cache_salt": cache_salt,
    }


def release_export(tree_cache: Any, manifest: dict) -> None:
    core = tree_cache.tree_core
    for nid in manifest["protected_ids"]:
        core.dec_host_lock_ref(nid)


def import_session(
    tree_cache: Any,
    tokens: List[int],
    dst_slots: torch.Tensor,
    extra_key: Optional[str] = None,
    cache_salt: Optional[str] = None,
) -> Tuple[int, int]:
    """Insert host-only nodes for ``tokens`` with host_value=dst_slots.

    Returns (matched, imported): the prefix the target already held (its own
    rows stay) and the suffix newly imported (using dst_slots[matched:]).
    """
    page_size = tree_cache.page_size
    key = RadixKey(
        token_ids=array("q", tokens), extra_key=extra_key, cache_salt=cache_salt
    ).page_aligned(page_size)
    n_tokens = len(key)
    assert len(dst_slots) == n_tokens, (
        f"slots {len(dst_slots)} != tokens {n_tokens}"
    )
    hash_chain = get_hash_str(list(key.token_ids[:n_tokens]), None, page_size=page_size)
    if not isinstance(hash_chain, list):
        hash_chain = [hash_chain]
    core = tree_cache.tree_core
    res = core.insert_host(
        core.root_node.id, key, dst_slots, hash_chain[: n_tokens // page_size]
    )
    if getattr(res, "host_insert_dropped", False):
        return n_tokens, 0
    return res.prefix_len, n_tokens - res.prefix_len


def alloc_pages(tree_cache: Any, n_pages: int, page_size: int) -> Optional[torch.Tensor]:
    """Allocate n_pages*page_size host slots; None when the pool is full."""
    group = tree_cache.cache_controller.mem_pool_host
    need = n_pages * page_size
    slots = group.alloc(need)
    if slots is None:
        # Evict host-only leaves and retry once (same shape as write()).
        tree_cache.evict_host(need, 0)  # ComponentType.FULL
        slots = group.alloc(need)
    return slots


def free_slots(tree_cache: Any, slots: Optional[torch.Tensor]) -> None:
    if slots is not None and len(slots) > 0:
        tree_cache.cache_controller.mem_pool_host.free(slots)


def slots_to_rows(slots: torch.Tensor, page_size: int) -> np.ndarray:
    pages = slots.reshape(-1, page_size)
    starts = pages[:, 0]
    assert bool(((starts % page_size) == 0).all()), "allocated pages not row-aligned"
    diffs = pages[:, 1:] - pages[:, :-1]
    assert bool((diffs == 1).all()), "allocated page slots not consecutive"
    return (starts // page_size).numpy().astype(np.int64)


# --------------------------------------------------------------------------
# NIXL plane: lazy agent + pool registration + prepped dlists
# --------------------------------------------------------------------------


class NixlPlane:
    def __init__(self, tree_cache: Any, cxi_device_index: Optional[int]) -> None:
        self.tree_cache = tree_cache
        self.cxi_device_index = cxi_device_index
        self.agent = None
        self.geoms: Optional[Dict[str, PoolGeom]] = None
        self._peers: set = set()
        self._peer_canonical: Dict[str, str] = {}
        self._lock = threading.RLock()

    def ensure(self) -> None:
        with self._lock:
            if self.agent is not None:
                return
            from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t

            if self.cxi_device_index is not None:
                os.environ["UCCL_CXI_DEVICE_INDEX"] = str(self.cxi_device_index)
            backend = os.environ.get("SGLANG_DISAGGREGATION_NIXL_BACKEND", "UCCL")
            params = json.loads(
                os.environ.get(
                    "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS", '{"num_cpus": ""}'
                )
            )
            cfg = nixl_agent_config(
                backends=[],
                num_threads=0,
                sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
            )
            self.agent = nixl_agent(str(uuid.uuid4()), cfg)
            self.agent.create_backend(backend, params)
            if backend not in self.agent.get_plugin_list():
                raise RuntimeError(
                    f"NIXL backend {backend} unavailable; plugins="
                    f"{self.agent.get_plugin_list()}"
                )
            self.geoms = _pool_geoms(self.tree_cache)
            for geom in self.geoms.values():
                if geom.span == 1:
                    self.agent.register_memory(
                        [(geom.bases[0], geom.item * geom.n_descs, 0, "")], "DRAM"
                    )
                else:
                    for b in geom.bases:
                        self.agent.register_memory(
                            [(b, geom.item * geom.rows_per_layer, 0, "")], "DRAM"
                        )
            logger.info(
                "[session-migration] NIXL agent up (%s), pools=%s",
                backend,
                {p: g.fingerprint() for p, g in self.geoms.items()},
            )

    @property
    def name(self) -> str:
        self.ensure()
        return self.agent.name

    def metadata(self) -> bytes:
        self.ensure()
        return self.agent.get_agent_metadata()

    def add_peer(self, peer_name: str, peer_meta_b64: str) -> str:
        """Register a remote agent; returns its canonical NIXL name."""
        self.ensure()
        with self._lock:
            if peer_name in self._peers:
                return self._peer_canonical.get(peer_name, peer_name)
            canonical = self.agent.add_remote_agent(base64.b64decode(peer_meta_b64))
            self._peer_canonical[peer_name] = canonical
            self._peers.add(peer_name)
            return canonical

    def drain_notifs(self, seconds: float = 1.0) -> int:
        """Drain transfer-completion notifs (a WRITE's remote completion)."""
        self.ensure()
        seen = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            got = False
            for _peer, msgs in self.agent.get_new_notifs().items():
                seen += len(msgs)
                got = got or len(msgs) > 0
            if not got:
                break
        return seen

    def write_pages(
        self,
        peer_name: str,
        peer_meta_b64: str,
        per_pool: Dict[str, Tuple[np.ndarray, np.ndarray]],
        dst_specs: Dict[str, dict],
    ) -> float:
        """WRITE pages to peer via initialize_xfer (h2h-proven path).

        per_pool: {pool: (src_pages, dst_pages)} page-row ids on each side;
        dst_specs: {pool: peer geometry spec} (bases/item/span) so the dst
        descriptors reference the PEER's registered addresses.
        """
        self.ensure()
        canonical = self.add_peer(peer_name, peer_meta_b64)
        max_pages = _env_int(MIGRATION_MAX_PAGES_PER_XFER_ENV, _DEFAULT_MAX_PAGES_PER_XFER)
        t0 = time.perf_counter()
        for pool, (src_pages, dst_pages) in per_pool.items():
            geom = self.geoms[pool]
            spec = dst_specs[pool]
            if (spec["span"] != geom.span or spec["item"] != geom.item
                    or spec["n_descs"] != geom.n_descs):
                raise RuntimeError(
                    f"pool {pool} geometry mismatch: local {geom.fingerprint()} "
                    f"vs remote {spec}"
                )
            for off in range(0, len(src_pages), max_pages):
                s = src_pages[off : off + max_pages]
                d = dst_pages[off : off + max_pages]
                sd = self.agent.get_xfer_descs(geom.page_descs(s), "DRAM")
                dd = self.agent.get_xfer_descs(_spec_page_descs(spec, d), "DRAM")
                notif = f"mig_{pool}_{off}".encode()
                h = self.agent.initialize_xfer("WRITE", sd, dd, canonical, notif)
                if not h:
                    raise RuntimeError("initialize_xfer returned None")
                if self.agent.transfer(h) == "ERR":
                    raise RuntimeError(f"transfer ERR for {pool} pages {off}")
                self._wait_done(h)
                self.agent.release_xfer_handle(h)
        return time.perf_counter() - t0

    def _wait_done(self, handle, timeout: float = 300.0) -> None:
        t0 = time.perf_counter()
        while True:
            st = self.agent.check_xfer_state(handle)
            if st == "DONE":
                return
            if st == "ERR":
                raise RuntimeError("NIXL transfer ERR")
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError("NIXL transfer timeout")
            time.sleep(0)

    def checksum(self, pool: str, pages: np.ndarray) -> int:
        self.ensure()
        return self.geoms[pool].checksum(pages)


# --------------------------------------------------------------------------
# The per-scheduler agent
# --------------------------------------------------------------------------


class SessionMigrationAgent:
    def __init__(
        self,
        tree_cache: Any,
        dp_rank: int,
        port: int,
        cxi_device_index: Optional[int] = None,
        stats_file: Optional[str] = None,
    ) -> None:
        self.tree_cache = tree_cache
        self.dp_rank = dp_rank
        self.port = port
        self.stats_file = stats_file
        self.exec = SchedulerExecutor()
        self.plane = NixlPlane(tree_cache, cxi_device_index)
        self._srv: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._fp: Optional[dict] = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("0.0.0.0", self.port))
        self._srv.listen(8)
        self._srv.settimeout(1.0)
        self._thread = threading.Thread(
            target=self._serve_loop,
            daemon=True,
            name=f"session-migration-r{self.dp_rank}",
        )
        self._thread.start()
        logger.info(
            "[session-migration] agent up dp_rank=%d port=%d", self.dp_rank, self.port
        )

    def poll(self) -> None:
        """Drain scheduler-thread work; call once per scheduler loop pass."""
        self.exec.poll()

    def _serve_loop(self) -> None:
        while True:
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                self._handle_conn(conn)
            except Exception as e:  # noqa: BLE001
                logger.warning("[session-migration] conn error: %s", e, exc_info=True)
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _handle_conn(self, conn: socket.socket) -> None:
        msg = recv_msg(conn, timeout=300.0)
        op = msg.get("op")
        try:
            if op == "export_meta":
                resp = self._op_export_meta(msg)
            elif op == "push":
                resp = self._op_push(msg)
            elif op == "adopt":
                resp = self._op_adopt(msg)
            elif op == "ping":
                resp = {"ok": True, "rank": self.dp_rank}
            else:
                resp = {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as e:  # noqa: BLE001
            first = str(e).splitlines()[0][:300]
            logger.error("[session-migration] op %s failed: %s", op, first, exc_info=True)
            resp = {"ok": False, "error": first}
        send_msg(conn, resp)

    # -- helpers -----------------------------------------------------------

    def _fingerprint(self) -> dict:
        if self._fp is None:
            self._fp = _fingerprint(
                _pool_geoms(self.tree_cache), self.tree_cache.page_size
            )
        return self._fp

    def _log_stats(self, rec: dict) -> None:
        logger.info("[session-migration] %s", json.dumps(rec))
        if self.stats_file:
            try:
                with open(self.stats_file, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except OSError:
                pass

    # -- ops ---------------------------------------------------------------

    def _op_export_meta(self, msg: dict) -> dict:
        man = self.exec.run(
            lambda: export_session(
                self.tree_cache,
                msg["tokens"],
                msg.get("extra_key"),
                msg.get("cache_salt"),
            ),
            timeout=30.0,
        )
        if man is None:
            return {"ok": False, "reason": "no host prefix"}
        self.exec.run(lambda: release_export(self.tree_cache, man), timeout=30.0)
        return {
            "ok": True,
            "n_pages": man["n_pages"],
            "n_tokens": len(man["tokens"]),
            "fingerprint": self._fingerprint(),
            # NIXL agent identity: the target adds us so the UCCL handshake is
            # bidirectional before it asks us to push.
            "nixl_name": self.plane.name,
            "nixl_meta": base64.b64encode(self.plane.metadata()).decode(),
        }

    def _op_push(self, msg: dict) -> dict:
        """Holder side: export + WRITE pages to the requesting target."""
        t0 = time.perf_counter()
        man = self.exec.run(
            lambda: export_session(
                self.tree_cache,
                msg["tokens"],
                msg.get("extra_key"),
                msg.get("cache_salt"),
            ),
            timeout=60.0,
        )
        if man is None:
            return {"ok": False, "reason": "no host prefix"}
        try:
            t_export = time.perf_counter() - t0
            fp = self._fingerprint()
            if msg.get("fingerprint") != fp:
                return {"ok": False, "error": "pool fingerprint mismatch"}
            dst_rows = msg["dst_rows"]  # {pool: np.ndarray}
            n = man["n_pages"]
            per_pool = {}
            for pool in ("kv", "indexer"):
                d = dst_rows[pool]
                if len(d) < n:
                    return {"ok": False, "error": f"dst_rows too short for {pool}"}
                per_pool[pool] = (man["rows"], d[:n])
            xfer_s = self.plane.write_pages(
                msg["peer_name"], msg["peer_meta"], per_pool, msg["dst_spec"]
            )
            t_ck0 = time.perf_counter()
            checksums = {
                pool: self.plane.checksum(pool, man["rows"]) for pool in per_pool
            }
            t_ck = time.perf_counter() - t_ck0
            nbytes = sum(
                len(per_pool[p][0]) * self.plane.geoms[p].page_bytes()
                for p in per_pool
            )
            return {
                "ok": True,
                "meta": base64.b64encode(self.plane.metadata()).decode(),
                "name": self.plane.name,
                "tokens": man["tokens"],
                "src_rows": {p: v[0] for p, v in per_pool.items()},
                "checksums": checksums,
                "nbytes": nbytes,
                "t_export_s": round(t_export, 6),
                "t_xfer_s": round(xfer_s, 6),
                "t_checksum_s": round(t_ck, 6),
            }
        finally:
            self.exec.run(
                lambda: release_export(self.tree_cache, man), timeout=60.0
            )

    def _op_adopt(self, msg: dict) -> dict:
        """Target side: pull a session's pages from the holder, then import.

        msg: {tokens, extra_key, cache_salt, holder: (host, port), req_id}
        """
        t0 = time.perf_counter()
        holder = tuple(msg["holder"])
        timeout = float(msg.get("timeout", _DEFAULT_TIMEOUT_S))
        tokens = msg["tokens"]
        page_size = self.tree_cache.page_size
        slots: Optional[torch.Tensor] = None
        ok = False
        try:
            # 1. Ask the holder what it retains.
            meta_resp = tcp_rpc(
                holder,
                {
                    "op": "export_meta",
                    "tokens": tokens,
                    "extra_key": msg.get("extra_key"),
                    "cache_salt": msg.get("cache_salt"),
                },
                timeout,
            )
            if not meta_resp.get("ok"):
                return {"ok": False, "reason": meta_resp.get("reason", "export_meta failed")}
            n_pages = meta_resp["n_pages"]
            if n_pages == 0:
                return {"ok": False, "reason": "holder holds nothing"}
            if meta_resp.get("fingerprint") != self._fingerprint():
                return {"ok": False, "error": "pool fingerprint mismatch"}
            # Bidirectional NIXL handshake: add the holder before asking it to
            # push (UCCL rejects prepXferDlist until both sides exchanged).
            self.plane.add_peer(meta_resp["nixl_name"], meta_resp["nixl_meta"])

            # 2. Allocate target rows (page runs in the anchor pool).
            slots = self.exec.run(
                lambda: alloc_pages(self.tree_cache, n_pages, page_size), timeout=60.0
            )
            if slots is None:
                return {"ok": False, "reason": "host pool full"}
            dst_rows = slots_to_rows(slots, page_size)
            dst_by_pool = {"kv": dst_rows, "indexer": dst_rows}

            # 3. Ask the holder to push into our rows.
            dst_specs = {p: g.spec() for p, g in self.plane.geoms.items()}
            push_resp = tcp_rpc(
                holder,
                {
                    "op": "push",
                    "tokens": tokens,
                    "extra_key": msg.get("extra_key"),
                    "cache_salt": msg.get("cache_salt"),
                    "peer_name": self.plane.name,
                    "peer_meta": base64.b64encode(self.plane.metadata()).decode(),
                    "dst_rows": dst_by_pool,
                    "dst_spec": dst_specs,
                    "fingerprint": self._fingerprint(),
                },
                timeout,
            )
            if not push_resp.get("ok"):
                return {"ok": False, "error": push_resp.get("error", "push failed")}
            self.plane.add_peer(push_resp["name"], push_resp["meta"])
            self.plane.drain_notifs(seconds=5.0)

            # 4. Byte-verify: checksum our rows and compare with the holder's.
            t_ck0 = time.perf_counter()
            verified = True
            mismatch = None
            for pool, src_rows in push_resp["src_rows"].items():
                ours = self.plane.checksum(pool, dst_rows[: len(src_rows)])
                if ours != push_resp["checksums"][pool]:
                    verified = False
                    mismatch = pool
                    break
            t_verify = time.perf_counter() - t_ck0
            if not verified:
                return {
                    "ok": False,
                    "error": f"checksum mismatch on {mismatch}",
                }

            # 5. Import as host-only radix nodes.
            man_tokens = push_resp["tokens"]
            matched, imported = self.exec.run(
                lambda: import_session(
                    self.tree_cache,
                    man_tokens,
                    slots[: len(man_tokens)],
                    msg.get("extra_key"),
                    msg.get("cache_salt"),
                ),
                timeout=timeout,
            )
            # The matched prefix keeps the target's own rows; free what we
            # pre-allocated for it (insert_host used slots[matched:]).
            if matched > 0:
                self.exec.run(
                    lambda: free_slots(self.tree_cache, slots[:matched].clone()),
                    timeout=30.0,
                )
            n_leftover = n_pages - (matched + imported) // page_size
            if n_leftover > 0:
                self.exec.run(
                    lambda: free_slots(
                        self.tree_cache, slots[(matched + imported) :].clone()
                    ),
                    timeout=30.0,
                )
            ok = True
            total = time.perf_counter() - t0
            xfer_s = push_resp.get("t_xfer_s")
            nbytes = push_resp.get("nbytes", 0)
            rec = {
                "event": "session_migration",
                "req_id": msg.get("req_id"),
                "dp_rank": self.dp_rank,
                "tokens_requested": len(tokens),
                "tokens_imported": imported,
                "tokens_matched_existing": matched,
                "pages": n_pages,
                "bytes": nbytes,
                "t_total_s": round(total, 6),
                "t_xfer_s": xfer_s,
                "t_export_s": push_resp.get("t_export_s"),
                "t_verify_s": round(t_verify, 6),
                "t_insert_s": None,
                "gbps": round((nbytes / 1e9) / xfer_s, 3) if xfer_s else None,
                "verified": verified,
            }
            self._log_stats(rec)
            return {
                "ok": True,
                "tokens_imported": imported,
                "tokens_matched_existing": matched,
                "t_total_s": round(total, 6),
                "t_xfer_s": xfer_s,
                "gbps": rec["gbps"],
                "verified": verified,
            }
        finally:
            if not ok and slots is not None:
                try:
                    self.exec.run(
                        lambda: free_slots(self.tree_cache, slots), timeout=30.0
                    )
                except Exception:  # noqa: BLE001
                    logger.warning("[session-migration] failed to free slots", exc_info=True)
