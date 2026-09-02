"""Flag-gated capture of the target's final-layer hidden states for draft training.

Enabled by setting SGLANG_DRAFT_CAPTURE_DIR (output directory). Optional:
  SGLANG_DRAFT_CAPTURE_MODES  comma list of forward modes to capture
                              (default "extend"; choices: extend, decode)
  SGLANG_DRAFT_CAPTURE_TAG    label baked into the shard filename (e.g. prefill/decode)

What is captured: per request, the target's POST-FINAL-NORM hidden states —
the exact tensor fed to the lm_head, which is also the tensor the EAGLE/NextN
draft consumes through hnorm (spec_info.hidden_states). Records are keyed by a
stable 64-bit hash of the request id and the absolute position of the first
token, so multi-chunk prefills reassemble by sorting on (rid_hash, start_pos).

Shard file format (append-only, little-endian), one file per process:
  shard-<tag>-<world_rank>.bin
records:
  u32 magic 0x44524331 ("DRC1")   u32 version 1
  u64 rid_hash                    i64 start_pos
  i32 n_tok                       i32 hidden_dim        i32 dtype_code (0 = fp16)
  i32[n_tok] input token ids      fp16[n_tok * hidden_dim] hidden states

Safety properties (capture must never corrupt or stall the engine):
  - only plain ForwardMode.EXTEND / ForwardMode.DECODE batches are captured;
    DP-padding-fabricated batches (forward_batch._original_forward_mode set),
    idle/spec/mixed modes and CUDA-graph capture passes are skipped;
  - per-batch consistency checks (sum(extend_seq_lens) == rows, rids present)
    skip the batch on mismatch and count the skip in shard-*.stats.json;
  - writes happen on a background thread from a bounded queue; overflow drops
    the batch and counts it; any exception in the hook is swallowed and counted.

CUDA-graph note: under graph REPLAY the python forward does not execute, so
decode capture requires --disable-cuda-graph. Prefill (chunked extend) runs
eager in the production configuration, so extend-mode capture works with
graphs enabled on the decode arm.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch

MAGIC = 0x44524331
VERSION = 1
DTYPE_FP16 = 0
STATS_EVERY = 500
_HEADER = struct.Struct("<IIQqiii")
INT32_BYTES = 4
FP16_BYTES = 2


def _stable_rid_hash(rid: str) -> int:
    h = hashlib.sha1(rid.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


@dataclass
class _Stats:
    records: int = 0
    tokens: int = 0
    bytes: int = 0
    dropped_batches: int = 0
    dropped_tokens: int = 0
    skipped_batches: int = 0
    skipped_reasons: dict = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def note_skip(self, reason: str):
        self.skipped_batches += 1
        self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1


class _Writer(threading.Thread):
    def __init__(self, path: str, stats_path: str, q: "queue.Queue"):
        super().__init__(daemon=True, name="draft-capture-writer")
        self.path = path
        self.stats_path = stats_path
        self.q = q
        self.f = open(path, "ab", buffering=1024 * 1024)
        self.stats = _Stats()
        self._stop = threading.Event()

    def run(self):
        while True:
            try:
                item = self.q.get(timeout=1.0)
            except queue.Empty:
                self._flush()
                if self._stop.is_set() and self.q.empty():
                    break
                continue
            header, tokens, hidden = item
            try:
                self.f.write(header)
                self.f.write(tokens)
                self.f.write(hidden)
                self.stats.records += 1
                self.stats.tokens += len(tokens) // INT32_BYTES
                self.stats.bytes += len(header) + len(tokens) + len(hidden)
                if self.stats.records % STATS_EVERY == 0:
                    self._flush()
            except Exception as e:  # noqa: BLE001
                self.stats.note_skip(f"write_error:{type(e).__name__}")
            finally:
                self.q.task_done()
        self._flush()

    def _flush(self):
        try:
            self.f.flush()
            os.fsync(self.f.fileno())
        except Exception:  # noqa: BLE001
            pass
        tmp = self.stats_path + ".tmp"
        try:
            with open(tmp, "w") as sf:
                json.dump(self.stats.__dict__, sf)
            os.replace(tmp, self.stats_path)
        except Exception:  # noqa: BLE001
            pass

    def stop(self):
        self._stop.set()


class DraftCapture:
    """Process-wide singleton; created lazily on first use."""

    def __init__(self):
        self.dir = os.environ.get("SGLANG_DRAFT_CAPTURE_DIR", "")
        modes = os.environ.get("SGLANG_DRAFT_CAPTURE_MODES", "extend")
        self.modes = {m.strip().lower() for m in modes.split(",") if m.strip()}
        self.tag = os.environ.get("SGLANG_DRAFT_CAPTURE_TAG", "x")
        self.enabled = bool(self.dir) and bool(self.modes)
        self.writer: Optional[_Writer] = None
        self.q: Optional[queue.Queue] = None
        self.hidden_dim = 0
        if self.enabled:
            try:
                rank = torch.distributed.get_rank()
            except Exception:  # noqa: BLE001
                rank = 0
            os.makedirs(self.dir, exist_ok=True)
            base = os.path.join(self.dir, f"shard-{self.tag}-{rank}")
            self.q = queue.Queue(maxsize=8)
            self.writer = _Writer(base + ".bin", base + ".stats.json", self.q)
            self.writer.start()

    # ------------------------------------------------------------------
    def maybe_capture(self, forward_batch, hidden_states: torch.Tensor):
        if not self.enabled:
            return
        try:
            self._capture(forward_batch, hidden_states)
        except Exception as e:  # noqa: BLE001
            if self.writer is not None:
                self.writer.stats.note_skip(f"error:{type(e).__name__}")
                if self.writer.stats.skipped_batches <= 3:
                    import logging

                    logging.getLogger(__name__).exception(
                        "draft_capture error (engine continues)"
                    )

    def _capture(self, forward_batch, hidden_states: torch.Tensor):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        if hidden_states is None or hidden_states.dim() != 2 or hidden_states.shape[0] == 0:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        if getattr(forward_batch, "_original_forward_mode", None) is not None:
            return  # DP-padding-fabricated batch (idle/padded rows), not real tokens
        fm = forward_batch.forward_mode
        is_extend = fm == ForwardMode.EXTEND
        is_decode = fm == ForwardMode.DECODE
        if is_extend and "extend" not in self.modes:
            return
        if is_decode and "decode" not in self.modes:
            return
        if not (is_extend or is_decode):
            self.writer.stats.note_skip(f"mode:{fm.name}")
            return

        input_ids = forward_batch.input_ids
        if input_ids is None or input_ids.shape[0] != hidden_states.shape[0]:
            self.writer.stats.note_skip("shape_mismatch")
            return
        if input_ids.shape[0] > 65536:
            self.writer.stats.note_skip("too_large")
            return
        rids = forward_batch.rids
        bs = len(rids) if rids else 0
        if bs == 0:
            self.writer.stats.note_skip("no_rids")
            return

        hidden_dim = hidden_states.shape[1]
        if self.hidden_dim == 0:
            self.hidden_dim = hidden_dim
        elif self.hidden_dim != hidden_dim:
            self.writer.stats.note_skip("hidden_dim_changed")
            return

        if is_extend:
            extend_lens = self._extend_lens(forward_batch, bs)
            if extend_lens is None:
                return
            if sum(extend_lens) != input_ids.shape[0]:
                self.writer.stats.note_skip("extend_len_mismatch")
                return
        else:
            if input_ids.shape[0] != bs:
                self.writer.stats.note_skip("decode_len_mismatch")
                return
            extend_lens = None

        positions = forward_batch.positions
        if positions is None or positions.shape[0] != input_ids.shape[0]:
            self.writer.stats.note_skip("no_positions")
            return

        # GPU-side conversions, then one sync D2H per tensor.
        tokens_cpu = (
            input_ids.to(torch.int32).cpu().numpy().tobytes()
        )
        hidden_cpu = (
            hidden_states.to(torch.float16).contiguous().cpu().numpy().tobytes()
        )
        positions_cpu = positions.cpu().tolist()

        off = 0
        for r in range(bs):
            n = extend_lens[r] if is_extend else 1
            if n <= 0:
                continue
            start = int(positions_cpu[off])
            rid_hash = _stable_rid_hash(rids[r])
            header = _HEADER.pack(
                MAGIC, VERSION, rid_hash, start, n, hidden_dim, DTYPE_FP16
            )
            tok = tokens_cpu[off * INT32_BYTES : (off + n) * INT32_BYTES]
            hid = hidden_cpu[
                off * hidden_dim * FP16_BYTES : (off + n) * hidden_dim * FP16_BYTES
            ]
            try:
                self.q.put_nowait((header, tok, hid))
            except queue.Full:
                self.writer.stats.dropped_batches += 1
                self.writer.stats.dropped_tokens += n
            off += n

    def _extend_lens(self, forward_batch, bs) -> Optional[List[int]]:
        el = forward_batch.extend_seq_lens_cpu
        if el is None:
            el = forward_batch.extend_seq_lens
            el = el.tolist() if el is not None else None
        if el is None or len(el) != bs:
            self.writer.stats.note_skip("no_extend_lens")
            return None
        return el

    def close(self):
        if self.writer is not None:
            self.writer.stop()


_singleton: Optional[DraftCapture] = None
_lock = threading.Lock()


def get_draft_capture() -> DraftCapture:
    global _singleton
    if _singleton is None:
        with _lock:
            if _singleton is None:
                _singleton = DraftCapture()
    return _singleton


def maybe_capture_draft_input(forward_batch, hidden_states: torch.Tensor):
    """Hook entry: call with the target's post-final-norm hidden states."""
    cap = get_draft_capture()
    if cap.enabled:
        cap.maybe_capture(forward_batch, hidden_states)
