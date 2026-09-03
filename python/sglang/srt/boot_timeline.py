"""Fine-grained boot timeline behind SGLANG_BOOT_TIMELINE=1 (fast-boot lane).

Emits [BOOT_TIMELINE] lines with process-relative seconds plus cumulative
filesystem read counters from /proc/self/io, so weight loading, JIT caches,
host-pool allocation and CUDA graph capture can be attributed per phase.
Default OFF; mark()/wrap_weights_iterator() are no-ops then (zero behavior
change, no iterator wrapping).
"""
from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("SGLANG_BOOT_TIMELINE", "0") == "1"
_T0 = time.perf_counter()

def _io():
    try:
        rb = rc = -1
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    rb = int(line.split()[1])
                elif line.startswith("rchar:"):
                    rc = int(line.split()[1])
        return rb, rc
    except OSError:
        return -1, -1

def mark(name: str, **kw) -> None:
    if not _ENABLED:
        return
    t = time.perf_counter() - _T0
    rb, rc = _io()
    extra = " ".join(f"{k}={v}" for k, v in kw.items())
    logger.warning(
        "[BOOT_TIMELINE] pid=%d t=+%.3fs read_mb=%d rchar_mb=%d %s %s",
        os.getpid(), t, rb // 1000000, rc // 1000000, name, extra,
    )

def wrap_weights_iterator(weights):
    """Count yielded tensors / non-contiguous views / bytes (timeline only)."""
    if not _ENABLED:
        return weights
    stats = {"n": 0, "noncontig": 0, "mb": 0}

    def gen():
        for name, t in weights:
            stats["n"] += 1
            stats["mb"] += t.numel() * t.element_size() // 1000000
            if not t.is_contiguous():
                stats["noncontig"] += 1
            yield name, t
        mark("weights_yield_done", tensors=stats["n"],
             noncontig=stats["noncontig"], yielded_mb=stats["mb"])

    return gen()


def maybe_copy_weight_view_before_h2d(loaded_weight):
    """fast-boot: stage safetensors mmap-backed weights into anon storage before H2D.

    Behind SGLANG_WEIGHT_LOAD_STAGE_VIEWS (default off). Pageable H2D straight
    from file-backed pages re-reads them 2-6x from disk on GH200 (measured: a
    decode rank's 67 GB slice read 162 GB via direct copy_ vs 78 GB staged,
    47.9s -> 26.5s cold). safetensors tensors view the mmap but report
    storage_offset==0 with storage==tensor bytes, so the upstream
    SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D view test never fires for per-expert
    checkpoints - hence this unconditional CPU clone (byte-identical; transient
    per-tensor anon copy). Non-CPU sources pass through. Lives in this leaf
    module to avoid the linear -> weight_utils -> model_config -> quantization
    -> linear import cycle.
    """
    import torch
    from sglang.srt.environ import envs

    if not envs.SGLANG_WEIGHT_LOAD_STAGE_VIEWS.get():
        return loaded_weight
    if loaded_weight.device.type != "cpu":
        return loaded_weight
    if loaded_weight.numel() == 0:
        return loaded_weight
    return loaded_weight.clone(memory_format=torch.contiguous_format)
