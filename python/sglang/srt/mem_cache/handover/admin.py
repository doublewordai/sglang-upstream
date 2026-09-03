"""Engine-side handover orchestration (admin endpoints' implementation).

Flow (heir-initiated, old-initiated WRITE on the wire):

  heir POST /handover/import {src_host, src_port}
    -> scheduler: heir_import()
       1. POST http://src/handover/export {"phase":"info"}  -> sizes+fingerprint
       2. allocate heir host rows; create NIXL agent; register pools+manifest
       3. POST http://src/handover/export {"phase":"push", payload...}
       4. old side pushes (NIXL WRITEs), waits DONE, replies
       5. heir waits its notifs, checksum-verifies, imports, returns stats

  old POST /handover/export {"phase":"info"|"push"}
    -> scheduler: handover_export() (export cached between phases;
       protections held while cached)
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.request
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from sglang.srt.mem_cache.handover.manifest import (
    bytes_to_manifest,
    manifest_to_bytes,
    page_checksums,
)
from sglang.srt.mem_cache.handover.prefill_arm import (
    PrefillExport,
    _per_layer_host_buffers,
    alloc_heir_rows,
    heir_page_rows,
    tree_pools,
)
from sglang.srt.mem_cache.handover.transfer import (
    HandoverNixlAgent,
    build_pool_descriptors,
)

logger = logging.getLogger(__name__)


def _http_json(url: str, payload: dict, timeout: float, admin_key: str = None) -> dict:
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if admin_key:
        headers["Authorization"] = f"Bearer {admin_key}"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class _ExportState:
    """Per-generation cached export on the old side (protections held)."""

    def __init__(self):
        self.export: Optional[PrefillExport] = None
        self.fingerprint: Optional[str] = None
        self.model_path: Optional[str] = None
        self.staged: bool = False
        self.agent: Optional[HandoverNixlAgent] = None
        self.reg_keys: list = []

    def close(self):
        if self.export is not None:
            self.export.release_protections()
            self.export = None
        if self.agent is not None:
            self.agent.close()
            self.agent = None


_export_state = _ExportState()


def handover_export_info(
    tree_cache, model_path: str, staged: bool = False, with_checksums: bool = True
) -> Tuple[bool, str, Optional[dict]]:
    """Phase 'info': build (or reuse) the export; return sizes."""
    global _export_state
    try:
        if _export_state.export is not None and (
            _export_state.model_path == model_path
        ):
            export = _export_state.export
        else:
            _export_state.close()
            export = PrefillExport.build(
                tree_cache, model_path, staged=staged, with_checksums=with_checksums
            )
            _export_state.export = export
            _export_state.model_path = model_path
            _export_state.staged = staged
            _export_state.fingerprint = export.manifest.fingerprint
        m = export.manifest
        return (
            True,
            "ok",
            {
                "num_tokens": m.num_tokens,
                "num_pages": m.num_pages,
                "num_chains": len(m.chains),
                "page_size": m.page_size,
                "fingerprint": m.fingerprint,
                "manifest_len": len(manifest_to_bytes(m)),
            },
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("handover export info failed")
        _export_state.close()
        return False, f"export failed: {e}", None


def handover_export_push(payload_json: str, timeout_s: float = 300.0) -> Tuple[bool, str, Optional[dict]]:
    """Phase 'push': push the cached export into the heir's registered buffers."""
    global _export_state
    export = _export_state.export
    if export is None:
        return False, "no export built (call phase=info first)", None
    try:
        req = json.loads(payload_json)
        m = export.manifest
        if req["fingerprint"] != m.fingerprint:
            return False, "fingerprint mismatch", None
        if _export_state.agent is None:
            _export_state.agent = HandoverNixlAgent(
                backend="UCCL", gpu_id=0, cxi_device_index=0
            )
        agent = _export_state.agent
        for name, spec in export.pool_specs.items():
            key = f"src_{name}"
            if key not in agent._registered:
                agent.register(key, list(zip(spec.data_ptrs, spec.data_lens)), "DRAM")
        agent.add_peer(base64.b64decode(req["agent_metadata"]))

        mbytes = manifest_to_bytes(m)
        if not hasattr(_export_state, "_manifest_src") or _export_state._manifest_src is None:
            _export_state._manifest_src = torch.frombuffer(
                bytearray(mbytes), dtype=torch.uint8
            ).pin_memory()
        else:
            buf = _export_state._manifest_src
            if buf.numel() < len(mbytes):
                _export_state._manifest_src = torch.frombuffer(
                    bytearray(mbytes), dtype=torch.uint8
                ).pin_memory()
            else:
                _export_state._manifest_src[: len(mbytes)] = torch.frombuffer(
                    bytearray(mbytes), dtype=torch.uint8
                )
        if "src_manifest" not in agent._registered:
            agent.register(
                "src_manifest",
                [(_export_state._manifest_src.data_ptr(), _export_state._manifest_src.numel())],
                "DRAM",
            )

        t0 = time.perf_counter()
        handles = []
        total_bytes = 0
        n_desc = 0
        for name, spec in export.pool_specs.items():
            dst = req["pools"][name]
            if int(dst["item_len"]) != spec.item_len:
                return False, f"item_len mismatch for {name}", None
            src_addrs, dst_addrs = build_pool_descriptors(
                spec.data_ptrs,
                [int(p) for p in dst["ptrs"]],
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
        mdst = req["manifest"]
        handles.append(
            agent.push_write(
                [(_export_state._manifest_src.data_ptr(), len(mbytes))],
                [(int(mdst["ptr"]), int(mdst["len"]))],
                "DRAM",
                "DRAM",
                dst_gpu_id=0,
                peer_name=req["agent_name"],
                notif="handover_manifest",
            )
        )
        ok = agent.wait(handles, timeout_s=timeout_s)
        t_wire = time.perf_counter() - t0
        if not ok:
            return False, "push timed out", None
        return (
            True,
            "ok",
            {
                "bytes": total_bytes,
                "t_wire_s": t_wire,
                "n_desc": n_desc,
                "gbps": total_bytes / 1e9 / t_wire if t_wire > 0 else 0.0,
            },
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("handover export push failed")
        return False, f"push failed: {e}", None


def handover_export_release() -> Tuple[bool, str]:
    """Release the cached export (after the heir is done or on failure)."""
    global _export_state
    _export_state.close()
    return True, "released"


def heir_import(
    tree_cache,
    model_path: str,
    src_host: str,
    src_port: int,
    http_port: int,
    timeout_s: float = 600.0,
    verify: bool = True,
    admin_key: Optional[str] = None,
) -> Tuple[bool, str, Optional[dict]]:
    """Heir side: pull the old generation's host-tier cache into this engine.

    ``http_port`` is the old generation's HTTP server port (the export
    endpoint). ``src_port`` is unused (kept for signature symmetry; the NIXL
    handshake rides the HTTP payload).
    """
    from sglang.srt.mem_cache.handover import prefill_arm

    base = f"http://{src_host}:{http_port}/handover/export"
    t_all0 = time.perf_counter()
    try:
        ok_call = _http_json(
            base, {"phase": "info", "model_path": model_path},
            timeout=timeout_s, admin_key=admin_key,
        )
        ok, msg, info = ok_call.get("success", False), ok_call.get("message", ""), ok_call.get("data")
        if not ok:
            return False, f"old side export info failed: {msg}", None

        # fingerprint check against OUR stack
        fp_self = PrefillExport.fingerprint(tree_cache, model_path)
        if info["fingerprint"] != fp_self:
            return (
                False,
                f"config fingerprint mismatch: old {info['fingerprint']} vs heir {fp_self}",
                None,
            )

        n_tok = info["num_tokens"]
        if n_tok == 0:
            return True, "old side cache empty; nothing to import", {
                "tokens": 0, "pages": 0, "chains": 0
            }

        slots = alloc_heir_rows(tree_cache, n_tok)
        try:
            page_size = tree_cache.page_size
            dst_pages = heir_page_rows(slots, page_size)
            pools = tree_pools(tree_cache)
            agent = HandoverNixlAgent(
                backend="UCCL", gpu_id=0, cxi_device_index=0
            )
            pools_msg = {}
            for name, p in pools.items():
                spec_ptrs, spec_lens, items = p.get_contiguous_buf_infos()
                assert len(set(items)) == 1
                item = int(items[0])
                agent.register(
                    f"dst_{name}", list(zip(spec_ptrs, spec_lens)), "DRAM"
                )
                pools_msg[name] = {
                    "ptrs": [int(x) for x in spec_ptrs],
                    "item_len": item,
                    "dst_pages": [int(x) for x in dst_pages],
                }
            manifest_len = int(info.get("manifest_len", 0) or 0) or 64 * 1024 * 1024
            manifest_buf = torch.empty(manifest_len, dtype=torch.uint8).pin_memory()
            agent.register(
                "dst_manifest",
                [(manifest_buf.data_ptr(), manifest_buf.numel())],
                "DRAM",
            )

            t0 = time.perf_counter()
            phase2 = {
                "fingerprint": info["fingerprint"],
                "agent_metadata": base64.b64encode(agent.metadata()).decode(),
                "agent_name": agent.agent.name,
                "pools": pools_msg,
                "manifest": {
                    "ptr": int(manifest_buf.data_ptr()),
                    "len": int(manifest_buf.numel()),
                },
            }
            push_rep = _http_json(
                base,
                {
                    "phase": "push",
                    "payload_json": json.dumps(phase2),
                    "timeout_s": timeout_s,
                },
                timeout=timeout_s,
                admin_key=admin_key,
            )
            if not push_rep.get("success"):
                return False, f"old side push failed: {push_rep.get('message')}", None

            # wait for our notifs (all pools + manifest)
            deadline = time.monotonic() + timeout_s
            seen = set()
            while len(seen) < len(pools) + 1:
                for _peer, msgs in agent.poll_notifications().items():
                    for msg in msgs:
                        seen.add(msg.decode())
                if len(seen) >= len(pools) + 1:
                    break
                if time.monotonic() > deadline:
                    return False, f"timed out waiting notifs (saw {seen})", None
                time.sleep(0.002)
            t_notif = time.perf_counter() - t0

            m = bytes_to_manifest(manifest_buf.numpy().tobytes())
            assert m.fingerprint == info["fingerprint"]

            t_ck = 0.0
            if verify and m.checksums:
                tc0 = time.perf_counter()
                for name, want in m.checksums.items():
                    got = page_checksums(
                        _per_layer_host_buffers(pools[name]),
                        dst_pages,
                        pools_msg[name]["item_len"],
                    )
                    bad = int((got != want).sum())
                    if bad:
                        return False, (
                            f"checksum mismatch in pool {name}: {bad}/{len(want)} pages"
                        ), None
                t_ck = time.perf_counter() - tc0

            ti0 = time.perf_counter()
            stats = prefill_arm.import_manifest(tree_cache, m, slots, need_hashes=False)
            t_import = time.perf_counter() - ti0
            agent.close()

            return (
                True,
                "ok",
                {
                    **stats,
                    "wire_s": push_rep["data"].get("t_wire_s"),
                    "wire_gbps": push_rep["data"].get("gbps"),
                    "bytes": push_rep["data"].get("bytes"),
                    "t_notif_s": t_notif,
                    "t_checksum_s": t_ck,
                    "t_import_s": t_import,
                    "t_total_s": time.perf_counter() - t_all0,
                },
            )
        except Exception:
            # free the allocated rows on failure (tree untouched)
            try:
                tree_cache.token_to_kv_pool_host.free(slots)
            except Exception:
                logger.exception("failed to free heir rows after import failure")
            raise
    except Exception as e:  # noqa: BLE001
        logger.exception("heir import failed")
        return False, f"import failed: {e}", None
