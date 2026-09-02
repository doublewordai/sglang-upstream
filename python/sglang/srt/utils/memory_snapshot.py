"""Env-gated CUDA memory snapshot / peak tracking for allocation tracing.

Enabled by setting SGLANG_MEM_SNAPSHOT_DIR=<dir>. Inert otherwise.

How it works: ModelRunner.forward is wrapped per-instance (the model itself is
invoked via model.forward(...) which bypasses nn.Module hooks, but the decoder
layers are called via Module.__call__ so a layer pre-hook works for mid-forward
dumps). A stack handles the draft/target nesting of speculative decoding; peak
stats are reset only at the outermost forward.

Env knobs:
  SGLANG_MEM_SNAPSHOT_DIR          directory for pickle snapshots + peaks.jsonl
  SGLANG_MEM_SNAPSHOT_MAX_ENTRIES  history ring size (default 200000)
  SGLANG_MEM_SNAPSHOT_LAYER        decoder layer index for mid-forward dumps (default 40)
  SGLANG_MEM_SNAPSHOT_MIDFORWARD   comma list mode:count of mid-forward snapshot triggers
  SGLANG_MEM_SNAPSHOT_EVERY        track peaks on every Nth batch of each mode (default 10)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict

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
_layer_hook_installed = False
_pending_midforward = False
_stack: list = []  # outermost-first list of info dicts

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
        snap = torch.cuda.memory._snapshot()
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


def _find_layers(model):
    import torch.nn as nn

    cand = getattr(model, "model", model)
    layers = getattr(cand, "layers", None)
    if isinstance(layers, nn.ModuleList) and len(layers) > _LAYER:
        return layers
    return None


def install_memsnap_hooks(model_runner, model):
    """Wrap ModelRunner.forward + install the layer mid-forward dump hook."""
    global _layer_hook_installed
    if not _ENABLED or getattr(model_runner, "_memsnap_wrapped", False):
        return
    _ensure_history()

    layers = _find_layers(model)

    def _layer_pre(module, args):
        global _pending_midforward
        if _pending_midforward and not torch.cuda.is_current_stream_capturing():
            _pending_midforward = False
            info = _stack[-1] if _stack else {}
            _dump_snapshot(
                f"midforward_{info.get('mode', '?')}_bs{info.get('bs', 0)}_tok{info.get('ntok', 0)}"
            )

    if layers is not None and not _layer_hook_installed:
        layers[_LAYER].register_forward_pre_hook(_layer_pre)
        _layer_hook_installed = True

    orig_forward = model_runner.forward

    def wrapped(forward_batch, *args, **kwargs):
        _begin(forward_batch)
        try:
            return orig_forward(forward_batch, *args, **kwargs)
        finally:
            _end()

    model_runner.forward = wrapped
    model_runner._memsnap_wrapped = True
    logger.info(
        f"[mem-snap] forward wrapped (layer={_LAYER if layers is not None else None}, "
        f"midforward={_MIDFORWARD}, every={_EVERY})"
    )


def _begin(forward_batch: "ForwardBatch"):
    if not _ENABLED:
        return
    try:
        mode = (
            str(getattr(forward_batch.forward_mode, "name", forward_batch.forward_mode))
            .split(".")[-1]
        )
        bs = int(getattr(forward_batch, "batch_size", 0) or 0)
        ii = getattr(forward_batch, "input_ids", None)
        ntok = int(ii.numel()) if ii is not None else 0
        _mode_counts[mode] += 1
        track = _mode_counts[mode] <= 3 or _mode_counts[mode] % _EVERY == 0
        info: Dict[str, Any] = {
            "mode": mode,
            "bs": bs,
            "ntok": ntok,
            "track": track,
        }
        outermost = not _stack
        _stack.append(info)
        if track and outermost:
            info["alloc_before"] = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            global _pending_midforward
            if _midforward_left.get(mode, 0) > 0:
                _midforward_left[mode] -= 1
                _pending_midforward = True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[mem-snap] begin failed: {e}")


def _end():
    if not _ENABLED or not _stack:
        return
    info = _stack.pop()
    if not info.get("track", False) or _stack:
        return  # inner (draft) call or untracked: no row
    try:
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        alloc_before = info.get("alloc_before", 0)
        row = {
            "t": time.time(),
            "mode": info.get("mode", "?"),
            "bs": info.get("bs", 0),
            "ntok": info.get("ntok", 0),
            "alloc_before": alloc_before,
            "peak": peak,
            "transient": peak - alloc_before,
        }
        os.makedirs(_DIR, exist_ok=True)
        with open(os.path.join(_DIR, "peaks.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        logger.info(
            f"[mem-snap] peak mode={row['mode']} bs={row['bs']} ntok={row['ntok']} "
            f"alloc_before={alloc_before / GB:.3f} peak={row['peak'] / GB:.3f} "
            f"transient={row['transient'] / GB:.3f} GiB"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[mem-snap] end failed: {e}")


def memsnap_forward_begin(forward_batch: "ForwardBatch"):
    """Kept for compatibility; wrapping happens in install_memsnap_hooks."""
    return
