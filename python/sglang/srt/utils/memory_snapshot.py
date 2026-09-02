"""Env-gated CUDA memory snapshot / peak tracking for allocation tracing.

Enabled by setting SGLANG_MEM_SNAPSHOT_DIR=<dir>. Inert otherwise (no history
recording, no dumps, hooks are cheap no-ops).

Env knobs:
  SGLANG_MEM_SNAPSHOT_DIR          directory for pickle snapshots + peak.jsonl
  SGLANG_MEM_SNAPSHOT_MAX_ENTRIES  history ring size (default 200000)
  SGLANG_MEM_SNAPSHOT_LAYER        decoder layer index for mid-forward dumps (default 40)
  SGLANG_MEM_SNAPSHOT_MIDFORWARD   comma list mode:count of mid-forward snapshot triggers
                                   (default "DECODE:1,TARGET_VERIFY:1,EXTEND:2")
  SGLANG_MEM_SNAPSHOT_EVERY        track peaks on every Nth batch of each mode (default 10)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_DIR = os.getenv("SGLANG_MEM_SNAPSHOT_DIR", "")
_ENABLED = bool(_DIR)
_MAX_ENTRIES = int(os.getenv("SGLANG_MEM_SNAPSHOT_MAX_ENTRIES", "200000"))
_LAYER = int(os.getenv("SGLANG_MEM_SNAPSHOT_LAYER", "40"))
_MIDFORWARD: Dict[str, int] = {}
for _kv in os.getenv(
    "SGLANG_MEM_SNAPSHOT_MIDFORWARD", "DECODE:1,TARGET_VERIFY:1,EXTEND:2"
).split(","):
    if _kv.strip():
        _k, _, _v = _kv.partition(":")
        _MIDFORWARD[_k.strip()] = int(_v or 1)
_EVERY = int(os.getenv("SGLANG_MEM_SNAPSHOT_EVERY", "10"))

_history_on = False
_midforward_left: Dict[str, int] = dict(_MIDFORWARD)
_mode_counts: Dict[str, int] = defaultdict(int)
_installed = set()
_pending_midforward = False
_last_info: Dict[str, Any] = {}

GB = 1024**3


def _ensure_history():
    global _history_on
    if _ENABLED and not _history_on:
        torch.cuda.memory._record_memory_history(max_entries=_MAX_ENTRIES)
        _history_on = True


def memsnap_enabled() -> bool:
    return _ENABLED


def _dump_snapshot(tag: str):
    if not _ENABLED:
        return
    try:
        if torch.cuda.is_current_stream_capturing():
            logger.info(f"[mem-snap] skip {tag}: stream capturing")
            return
        _ensure_history()
        os.makedirs(_DIR, exist_ok=True)
        snap = torch.cuda.memory.snapshot()
        path = os.path.join(_DIR, f"snap_{tag}_{time.time():.3f}.pkl")
        with open(path, "wb") as f:
            pickle.dump(snap, f)
        segs = snap.get("segments", [])
        total = sum(s.get("total_size", 0) for s in segs)
        graph_pool = sum(
            s.get("total_size", 0) for s in segs if s.get("segment_pool_id") is not None
        )
        alloc = torch.cuda.memory_allocated()
        logger.info(
            f"[mem-snap] dumped {tag}: reserved={total / GB:.3f} GiB "
            f"(graph pools {graph_pool / GB:.3f}), allocated={alloc / GB:.3f} GiB -> {path}"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[mem-snap] dump {tag} failed: {e}")


def memsnap_phase(tag: str):
    """Dump a full snapshot at a boot phase boundary (after_weights etc.)."""
    if _ENABLED:
        _dump_snapshot(tag)


def install_memsnap_hooks(model_runner, model):
    """Register per-model-call peak tracking + optional mid-forward snapshot.

    Called once per ModelRunner after its model exists. The runner's forward()
    calls memsnap_forward_begin() with the ForwardBatch; the module hooks here
    do reset/measure around the nn.Module call (covers eager and graph capture;
    replay does not allocate).
    """
    global _installed
    if not _ENABLED or model is None or id(model) in _installed:
        return
    _installed.add(id(model))
    _ensure_history()

    import torch.nn as nn

    # find the decoder layer ModuleList (model.model.layers for GLM/DS-style)
    layers = None
    cand = getattr(model, "model", model)
    layers = getattr(cand, "layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) <= _LAYER:
        logger.info(f"[mem-snap] no decoder layer list with >{_LAYER} layers found; mid-forward dump disabled")
        layers = None

    def _pre(module, args):
        global _pending_midforward
        if not _last_info.get("track", False):
            return
        torch.cuda.reset_peak_memory_stats()
        _pending_midforward = _should_midforward()

    def _post(module, args, output):
        global _pending_midforward
        if not _last_info.get("track", False):
            _pending_midforward = False
            return
        try:
            info = dict(_last_info)
            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            row = {
                "t": time.time(),
                "mode": info.get("mode", "?"),
                "bs": info.get("bs", 0),
                "ntok": info.get("ntok", 0),
                "alloc_before": info.get("alloc_before", 0),
                "peak": peak,
                "transient": peak - info.get("alloc_before", 0),
            }
            os.makedirs(_DIR, exist_ok=True)
            with open(os.path.join(_DIR, "peaks.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")
            logger.info(
                f"[mem-snap] peak mode={row['mode']} bs={row['bs']} ntok={row['ntok']} "
                f"alloc_before={row['alloc_before'] / GB:.3f} peak={peak / GB:.3f} "
                f"transient={(peak - row['alloc_before']) / GB:.3f} GiB"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[mem-snap] post-hook failed: {e}")

    def _layer_pre(module, args):
        global _pending_midforward
        if _pending_midforward and not torch.cuda.is_current_stream_capturing():
            _pending_midforward = False
            info = dict(_last_info)
            _dump_snapshot(
                f"midforward_{info.get('mode', '?')}_bs{info.get('bs', 0)}_tok{info.get('ntok', 0)}"
            )

    model.register_forward_pre_hook(_pre)
    model.register_forward_hook(_post)
    if layers is not None:
        layers[_LAYER].register_forward_pre_hook(_layer_pre)
    logger.info(
        f"[mem-snap] hooks installed (layer={_LAYER if layers is not None else None}, "
        f"midforward={_MIDFORWARD}, every={_EVERY})"
    )


def _should_midforward() -> bool:
    mode = str(_last_info.get("mode", ""))
    if _midforward_left.get(mode, 0) > 0:
        _midforward_left[mode] -= 1
        return True
    return False


def memsnap_forward_begin(forward_batch: "ForwardBatch"):
    """Called at the top of ModelRunner.forward() with the live ForwardBatch."""
    if not _ENABLED:
        return
    try:
        mode = str(getattr(forward_batch.forward_mode, "name", forward_batch.forward_mode)).split(".")[-1]
        bs = int(getattr(forward_batch, "batch_size", 0) or 0)
        ntok = int(getattr(forward_batch, "input_ids", None).numel()) if getattr(forward_batch, "input_ids", None) is not None else 0
        _mode_counts[mode] += 1
        # track first 3 batches of every mode, then every _EVERY-th
        if _mode_counts[mode] <= 3 or _mode_counts[mode] % _EVERY == 0:
            _last_info.update(
                mode=mode,
                bs=bs,
                ntok=ntok,
                alloc_before=torch.cuda.memory_allocated(),
                track=True,
            )
        else:
            _last_info.update(track=False)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[mem-snap] forward_begin failed: {e}")


def should_track_current() -> bool:
    return bool(_last_info.get("track", False))
