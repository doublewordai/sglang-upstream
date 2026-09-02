"""Flag-gated per-request cold-path tracing (SGLANG_COLD_TRACE=1).

Emits one INFO log line per lifecycle event so a cold turn can be decomposed
end-to-end from engine logs alone:
    COLDTRACE <event> t=<unix_s> <k>=<v> ...
Events are identified by rid + bootstrap_room so the prefill and decode arms'
logs can be joined. Import-light on purpose (no sglang imports).
"""

import logging
import os
import time

_ENABLED = os.environ.get("SGLANG_COLD_TRACE", "0") == "1"

logger = logging.getLogger(__name__)


def cold_trace_enabled() -> bool:
    return _ENABLED


def cold_trace(event: str, **kw) -> None:
    if not _ENABLED:
        return
    parts = [f"COLDTRACE {event}", f"t={time.time():.6f}"]
    for k, v in kw.items():
        parts.append(f"{k}={v}")
    logger.info(" ".join(parts))
