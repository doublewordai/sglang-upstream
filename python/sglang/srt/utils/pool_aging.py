"""Pool-aging telemetry: periodic structured logging of pool usage + process RSS.

Tells load-driven exhaustion (device/host usage -> 1.0 while RSS is flat) from
memory aging (RSS/VmHWM growth while pool usage is flat) across a generation.

Enabled with SGLANG_POOL_AGING_LOG_SECS=<seconds> (0 = off, the default).
Emits one line per interval per scheduler rank:

  [pool-aging] {"ts": ..., "step": N, "rss_kb": ..., "vmhwm_kb": ...,
                "gpu_free_gb": ..., "gpu_total_gb": ...,
                "device_tokens": ..., "device_capacity": ...,
                "host_tokens": ..., "host_capacity": ...,
                "evicted_bytes": ..., "available_host_gb": ...}
"""
from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

_INTERVAL = float(os.getenv("SGLANG_POOL_AGING_LOG_SECS", "0") or 0)
_last = 0.0
_step = 0


def _read_status_kb(field: str) -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def _pool_numbers(scheduler):
    """(device_tokens, device_capacity, host_tokens, host_capacity)."""
    device_tokens = device_capacity = host_tokens = host_capacity = -1
    try:
        coord = getattr(scheduler, "hisparse_coordinator", None)
        if coord is not None and hasattr(coord, "get_token_stats"):
            st = coord.get_token_stats()
            return st.device_tokens, -1, st.host_tokens, -1
    except Exception:  # noqa: BLE001
        pass
    try:
        alloc = getattr(scheduler, "token_to_kv_pool_allocator", None)
        if alloc is not None:
            cap = getattr(alloc, "size", None)
            avail = alloc.available_size() if hasattr(alloc, "available_size") else None
            if cap is not None and avail is not None:
                device_tokens, device_capacity = cap - avail, cap
    except Exception:  # noqa: BLE001
        pass
    try:
        tree = getattr(scheduler, "tree_cache", None)
        host = getattr(tree, "host_pool", None)
        if host is None:
            hicache = getattr(tree, "hicache", None)
            host = getattr(hicache, "host_pool", None) if hicache else None
        if host is not None and hasattr(host, "available_size"):
            cap = getattr(host, "size", None)
            if cap is not None:
                host_tokens, host_capacity = cap - host.available_size(), cap
    except Exception:  # noqa: BLE001
        pass
    return device_tokens, device_capacity, host_tokens, host_capacity


def maybe_log_pool_aging(scheduler) -> None:
    """Call once per scheduler loop iteration; rate-limited internally."""
    global _last, _step
    if _INTERVAL <= 0:
        return
    _step += 1
    now = time.time()
    if now - _last < _INTERVAL:
        return
    _last = now
    try:
        row = {
            "ts": round(now, 3),
            "step": _step,
            "rss_kb": _read_status_kb("VmRSS"),
            "vmhwm_kb": _read_status_kb("VmHWM"),
        }
        try:
            import torch

            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                row["gpu_free_gb"] = round(free_b / 1e9, 3)
                row["gpu_total_gb"] = round(total_b / 1e9, 3)
        except Exception:  # noqa: BLE001
            pass
        dt, dc, ht, hc = _pool_numbers(scheduler)
        row.update(
            device_tokens=dt,
            device_capacity=dc,
            host_tokens=ht,
            host_capacity=hc,
        )
        try:
            row["available_host_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / 1e9, 3)
        except Exception:  # noqa: BLE001
            pass
        logger.info("[pool-aging] " + json.dumps(row))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[pool-aging] failed: {e}")
