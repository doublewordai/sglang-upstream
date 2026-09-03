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
