"""Per-step speculative decoding telemetry.

Enabled by ``SGLANG_SPEC_TELEMETRY_DIR``. Each decode verify round produces
two JSONL records joined offline by step index ``i``:

- worker side (``side="w"``): wall-clock timestamp, batch size, active draft
  steps/tokens, seq_lens sum, and (when ``SGLANG_SPEC_CAPTURE_CONFIDENCE`` is
  on) per-request draft-token confidences.
- processor side (``side="p"``): per-request ``num_correct_drafts`` once the
  verify result reaches the scheduler's result processor (which may be a few
  steps later under overlap scheduling).

Confidence lag. The worker reads confidences zero-sync (one-round-stale
staged delivery), so a ``"conf"`` field carrying ``"conf_lag": 1`` does NOT
describe the record it sits on: it describes the most recent PRIOR worker-side
decode record with ``k > 0`` from the same process (prefill and ``k == 0``
rounds neither produce nor deliver confidences, so the lag is one step in the
subsequence of ``k > 0`` worker records, not necessarily ``i - 1``). Its rows
align with THAT record's request list (length = that record's ``bs``) and its
column count with that record's ``k``. Fitters must shift ``conf`` back by one
``k > 0`` worker record before joining per-request; records without
``conf_lag`` carry same-round confidences (legacy traces).

Under CUDA graphs the only honest step cost is the wall-clock gap between
consecutive decode rounds, so analysis derives step time from successive
worker-side timestamps (filtering rounds separated by prefill or idle gaps).
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional

from sglang.srt.environ import envs

_FLUSH_EVERY = 200


class SpecTelemetry:
    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"spec_telemetry-{os.getpid()}.jsonl")
        # Line-buffered: the server is torn down with signals, so records
        # must reach the OS as they are written or partial runs leave
        # nothing on disk.
        self._file = open(path, "a", buffering=1)
        self._step_ct = 0
        self._unflushed_ct = 0

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record) + "\n")
        self._unflushed_ct += 1
        if self._unflushed_ct >= _FLUSH_EVERY:
            self._file.flush()
            self._unflushed_ct = 0

    def on_decode_step(
        self,
        num_reqs: int,
        num_steps: int,
        num_draft_tokens: int,
        seq_lens_sum: int,
        confidences: Optional[List[List[float]]] = None,
        worker_round_idx: Optional[int] = None,
        conf_lag: Optional[int] = None,
    ) -> int:
        """Log the worker-side half of a decode round. Returns the step index
        used to join with the processor-side half.

        ``worker_round_idx`` counts ALL worker rounds including prefill, so
        the offline fitter can reject decode-round pairs with a prefill in
        between (whose wall gap includes the prefill, not just the step).

        ``conf_lag`` marks how many ``k > 0`` worker records back the
        ``confidences`` belong to (the zero-sync staged read delivers the
        previous round's values; see the module docstring). ``None``/0 means
        same-round confidences.
        """
        step_idx = self._step_ct
        self._step_ct += 1
        record = {
            "side": "w",
            "i": step_idx,
            "t": time.perf_counter(),
            "bs": num_reqs,
            "k": num_steps,
            "ndt": num_draft_tokens,
            "sls": seq_lens_sum,
        }
        if worker_round_idx is not None:
            record["ri"] = worker_round_idx
        if confidences is not None:
            record["conf"] = confidences
            if conf_lag:
                record["conf_lag"] = conf_lag
        self._write(record)
        return step_idx

    def on_verify_result(self, step_idx: int, num_correct_drafts: List[int]) -> None:
        """Log the processor-side half: per-request correct drafts (no bonus)."""
        self._write({"side": "p", "i": step_idx, "accept": num_correct_drafts})


_spec_telemetry: Optional[SpecTelemetry] = None
_spec_telemetry_checked = False


def get_spec_telemetry() -> Optional[SpecTelemetry]:
    global _spec_telemetry, _spec_telemetry_checked
    if not _spec_telemetry_checked:
        _spec_telemetry_checked = True
        out_dir = envs.SGLANG_SPEC_TELEMETRY_DIR.get()
        if out_dir:
            _spec_telemetry = SpecTelemetry(out_dir)
    return _spec_telemetry


def capture_draft_confidence() -> bool:
    """Whether draft-token confidences should be computed and captured.

    Forces the real softmax probability path where the topk=1 chain fast
    path would otherwise discard probabilities (``topk_p = ones``), so it
    carries a measurable extraction cost — keep it off for cost-surface
    calibration runs, on for trace collection and gated policies.
    """
    return envs.SGLANG_SPEC_CAPTURE_CONFIDENCE.get()
