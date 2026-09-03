"""NIXL/UCCL push for the generation handover.

The wire shape is the same as the PD prefill->decode KV transfer: the
*sender* (old generation) drives WRITEs into buffers the receiver (heir)
registered with its own NIXL agent; the receiver's page rows are known to
the sender through the control handshake. Descriptors are page-granular,
run-grouped and capped below 2 GB (CXI fi_write 2^32-1 byte cap).

Only WRITE is used: the UCCL NIXL backend rejects READ (cache-handover
lane: createXferReq NIXL_ERR_NOT_FOUND for every read combo, and one
malformed read "completed" at wire speed with wrong bytes).
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# CXI single-descriptor cap is 2^32-1 bytes; keep a safety margin.
MAX_DESC_BYTES = 2 * 1024**3 - 64 * 1024**2


def _nixl_device_id(mem_kind: str, gpu_id: int) -> int:
    return gpu_id if mem_kind == "VRAM" else 0


class HandoverNixlAgent:
    """Thin wrapper over a nixl agent pinned to one UCCL/CXI device.

    ``cxi_device_index`` sets UCCL_CXI_DEVICE_INDEX before backend creation
    (per-task CUDA_VISIBLE_DEVICES remapping otherwise puts every agent on
    "GPU 0" and agents share a NIC — cache-handover lane measured 67.7 vs
    90.3 GB/s for 4 agents).
    """

    def __init__(
        self,
        backend: str = "UCCL",
        gpu_id: int = 0,
        cxi_device_index: Optional[int] = None,
        num_threads: int = 0,
    ):
        try:
            from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t
        except ImportError as e:
            import sys

            print("NIXL IMPORT FAILED:", e, file=sys.stderr)
            print("sys.executable:", sys.executable, file=sys.stderr)
            print("PYTHONPATH:", os.environ.get("PYTHONPATH"), file=sys.stderr)
            print(
                "nixl-ish sys.path entries:",
                [p for p in sys.path if "nixl" in p],
                file=sys.stderr,
            )
            raise

        if cxi_device_index is not None:
            os.environ["UCCL_CXI_DEVICE_INDEX"] = str(cxi_device_index)
        agent_config = nixl_agent_config(
            backends=[],
            num_threads=num_threads,
            sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
        )
        self.agent = nixl_agent(str(uuid.uuid4()), agent_config)
        backend_params = {"num_cpus": ""} if backend == "UCCL" else {}
        self.agent.create_backend(backend, backend_params)
        if backend not in self.agent.get_plugin_list():
            raise ValueError(f"NIXL backend {backend!r} not available")
        self.backend = backend
        self.gpu_id = gpu_id
        self._registered: Dict[str, object] = {}

    # -- peer management ---------------------------------------------------

    def metadata(self) -> bytes:
        return self.agent.get_agent_metadata()

    def add_peer(self, metadata: bytes) -> None:
        self.agent.add_remote_agent(metadata)

    # -- memory registration ------------------------------------------------

    def register(self, key: str, addrs: List[Tuple[int, int]], mem_kind: str):
        """Register (ptr, len) pairs; keeps descs under ``key`` for reuse."""
        reqs = [
            (int(p), int(l), _nixl_device_id(mem_kind, self.gpu_id), "")
            for p, l in addrs
        ]
        if not reqs:
            raise ValueError("register() called with no addresses")
        descs = self.agent.register_memory(reqs, mem_kind)
        if not descs:
            raise RuntimeError(f"NIXL register_memory failed for {key}")
        self._registered[key] = descs
        return descs

    def has(self, key: str) -> bool:
        return key in self._registered

    # -- transfer ------------------------------------------------------------

    def push_write(
        self,
        src_addrs: List[Tuple[int, int]],
        dst_addrs: List[Tuple[int, int]],
        src_mem_kind: str,
        dst_mem_kind: str,
        dst_gpu_id: int,
        peer_name: str,
        notif: str,
        dst_agent_metadata: Optional[bytes] = None,
    ):
        """Post one WRITE of matched (src,dst) byte ranges.

        ``src_addrs``/``dst_addrs`` are (ptr, nbytes) pairs. Returns the
        transfer handle.
        """
        src_reqs = np.array(
            [
                (p, l, _nixl_device_id(src_mem_kind, self.gpu_id))
                for p, l in src_addrs
            ],
            dtype=np.uint64,
        )
        dst_reqs = np.array(
            [(p, l, _nixl_device_id(dst_mem_kind, dst_gpu_id)) for p, l in dst_addrs],
            dtype=np.uint64,
        )
        src_descs = self.agent.get_xfer_descs(src_reqs, src_mem_kind)
        dst_descs = self.agent.get_xfer_descs(dst_reqs, dst_mem_kind)
        handle = self.agent.initialize_xfer(
            "WRITE", src_descs, dst_descs, peer_name, notif.encode("ascii")
        )
        if not handle:
            raise RuntimeError("initialize_xfer failed")
        state = self.agent.transfer(handle)
        if state == "ERR":
            raise RuntimeError("transfer() posted ERR")
        return handle

    def wait(self, handles: List, timeout_s: float) -> bool:
        """Poll check_xfer_state until all DONE. False on timeout."""
        deadline = time.monotonic() + timeout_s
        pending = list(handles)
        while pending:
            nxt = []
            for h in pending:
                state = self.agent.check_xfer_state(h)
                if state == "ERR":
                    raise RuntimeError("NIXL transfer ERR while waiting")
                if state != "DONE":
                    nxt.append(h)
            if not nxt:
                return True
            if time.monotonic() > deadline:
                return False
            pending = nxt
            time.sleep(0.001)
        return True

    def poll_notifications(self) -> Dict[str, List[bytes]]:
        return self.agent.get_new_notifs()

    def close(self):
        try:
            self.agent.remove_agent()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Descriptor building
# ---------------------------------------------------------------------------


def run_group_pages(
    src_pages: np.ndarray, dst_pages: np.ndarray
) -> List[Tuple[int, int, int]]:
    """Group page-index pairs into (src_start, dst_start, count) runs where
    both sides advance by one. Returns runs in canonical order."""
    if len(src_pages) == 0:
        return []
    assert len(src_pages) == len(dst_pages)
    runs = []
    s0, d0 = int(src_pages[0]), int(dst_pages[0])
    count = 1
    for i in range(1, len(src_pages)):
        s, d = int(src_pages[i]), int(dst_pages[i])
        if s == s0 + count and d == d0 + count:
            count += 1
        else:
            runs.append((s0, d0, count))
            s0, d0, count = s, d, 1
    runs.append((s0, d0, count))
    return runs


def build_pool_descriptors(
    src_base_ptrs: List[int],
    dst_base_ptrs: List[int],
    item_len: int,
    src_pages: np.ndarray,
    dst_pages: np.ndarray,
    max_desc_bytes: int = MAX_DESC_BYTES,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """All-layers-at-once descriptors: every layer pushes the same pages."""
    assert len(src_base_ptrs) == len(dst_base_ptrs)
    max_pages = max(1, max_desc_bytes // item_len)
    src_addrs: List[Tuple[int, int]] = []
    dst_addrs: List[Tuple[int, int]] = []
    for s0, d0, count in run_group_pages(src_pages, dst_pages):
        for off in range(0, count, max_pages):
            n = min(max_pages, count - off)
            nbytes = n * item_len
            for lp, rp in zip(src_base_ptrs, dst_base_ptrs):
                src_addrs.append((lp + (s0 + off) * item_len, nbytes))
                dst_addrs.append((rp + (d0 + off) * item_len, nbytes))
    return src_addrs, dst_addrs
