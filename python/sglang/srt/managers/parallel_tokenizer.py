"""Exact parallel chunked tokenization for long prompts (SGLANG_PARALLEL_TOKENIZE=1).

Agentic prompts re-send the whole transcript every turn; the HF fast tokenizer
encodes them single-threaded at ~0.8 Mtok/s, which is 0.4-1.3 s of pure CPU on
the serving event loop for 300k-1M-token prompts. This module splits the
rendered prompt at provably-safe token boundaries and encodes the chunks
concurrently via the tokenizer's own batch API (rayon thread pool), then
concatenates. Output is token-identical to the single-shot encode.

Exactness argument (GLM-5.3, verified against its tokenizer.json):
  * pre_tokenizer = Sequence[Split(regex=Llama3-style, Isolated), ByteLevel].
    Every position p with text[p-1] == '\n' and text[p] non-whitespace is a
    pre-token boundary: the only pre-token patterns that can contain '\n'
    (`\\s*[\\r\\n]+`, ` ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*`, `\\s+(?!\\S)`, `\\s+`)
    always end at the last '\\n'/'\\r' of a whitespace run, and a '\\n'
    immediately followed by non-whitespace is exactly that last position.
  * Byte-level BPE encodes each pre-token independently, so concatenating the
    chunk encodings equals the single-shot encoding when every split point is
    a pre-token boundary.
  * Added/special tokens are matched before pre-tokenization and are atomic;
    none of GLM-5.3's added tokens contain '\\n' (asserted at init), so no
    split point can fall inside one.
  * The post-processor is ByteLevel (adds no tokens), so add_special_tokens
    does not change the id sequence; chunks are encoded with
    add_special_tokens=False and the result is identical either way (asserted
    once at init on a probe string).
  * Belt and braces: every split point is re-verified per request by
    re-encoding a local window straddling the boundary both ways (split vs
    single-shot, batched); any mismatch falls back to a full single-shot
    encode, so the returned ids can never diverge from the stock tokenizer.

Env flags (all default off):
  SGLANG_PARALLEL_TOKENIZE=1            enable
  SGLANG_PARALLEL_TOKENIZE_MIN_CHARS    minimum prompt length to parallelize (default 262144)
  SGLANG_PARALLEL_TOKENIZE_CHUNK_CHARS  target chunk size in chars (default 131072)
  SGLANG_PARALLEL_TOKENIZE_MAX_CHUNKS   max chunks (default 16)
  SGLANG_PARALLEL_TOKENIZE_VERIFY=0     disable the per-request window verification
"""

from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("SGLANG_PARALLEL_TOKENIZE", "0") == "1"
_MIN_CHARS = int(os.environ.get("SGLANG_PARALLEL_TOKENIZE_MIN_CHARS", 262144))
_CHUNK_CHARS = int(os.environ.get("SGLANG_PARALLEL_TOKENIZE_CHUNK_CHARS", 131072))
_MAX_CHUNKS = int(os.environ.get("SGLANG_PARALLEL_TOKENIZE_MAX_CHUNKS", "16"))
_VERIFY = os.environ.get("SGLANG_PARALLEL_TOKENIZE_VERIFY", "1") == "1"
_VERIFY_WINDOW = 128


def parallel_tokenize_enabled() -> bool:
    return _ENABLED


class ParallelTokenizer:
    """Wraps an HF fast tokenizer with an exact chunked-parallel encode."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._checked = False
        self._usable = False

    def _check_usable(self) -> bool:
        """One-time structural check of the tokenizer config."""
        if self._checked:
            return self._usable
        self._checked = True
        try:
            import tokenizers  # HF fast tokenizers

            if not isinstance(self.tokenizer, tokenizers.Tokenizer):
                return False
            for t in self.tokenizer.get_added_tokens_decoder().values():
                if "\n" in t.content or "\r" in t.content:
                    # a split after '\n' could fall inside an added token
                    return False
            # post-processor must not inject tokens (ByteLevel doesn't)
            probe = "hello world"
            a = self.tokenizer.encode(probe, add_special_tokens=True).ids
            b = self.tokenizer.encode(probe, add_special_tokens=False).ids
            if a != b:
                return False
            self._usable = True
        except Exception:
            self._usable = False
        return self._usable

    def _split_points(self, text: str) -> List[int]:
        """Split positions: after a '\\n' immediately followed by non-whitespace."""
        pts: List[int] = []
        n = len(text)
        start = 0
        while n - start > _CHUNK_CHARS:
            hi = start + _CHUNK_CHARS  # split point must be <= hi (< n by loop guard)
            idx = text.rfind("\n", start + 1, hi)
            while idx > start:
                p = idx + 1
                if not text[p].isspace():
                    break
                idx = text.rfind("\n", start + 1, idx)
            if idx <= start:
                break  # no safe split point in this window
            pts.append(idx + 1)
            start = idx + 1
        # downsample to at most _MAX_CHUNKS chunks, keeping spread
        if len(pts) > _MAX_CHUNKS - 1:
            m = _MAX_CHUNKS - 1
            pts = [pts[round(i * (len(pts) - 1) / m)] for i in range(m)]
        return pts

    def encode(self, text: str, **kwargs) -> List[int]:
        """Drop-in for tokenizer.encode(): token-identical, parallel for long text."""
        if not _ENABLED or len(text) < _MIN_CHARS or not self._check_usable():
            return self.tokenizer.encode(text, **kwargs)

        pts = self._split_points(text)
        if len(pts) < 2:
            return self.tokenizer.encode(text, **kwargs)

        chunks: List[str] = []
        prev = 0
        for p in pts:
            chunks.append(text[prev:p])
            prev = p
        chunks.append(text[prev:])

        batch = list(chunks)
        if _VERIFY:
            n = len(text)
            for p in pts:
                lo = max(0, p - _VERIFY_WINDOW)
                hi = min(n, p + _VERIFY_WINDOW)
                batch.append(text[lo:hi])
                batch.append(text[lo:p])
                batch.append(text[p:hi])

        encodings = self.tokenizer.encode_batch(batch, add_special_tokens=False)

        if _VERIFY:
            k = len(chunks)
            for j, p in enumerate(pts):
                whole = encodings[k + 3 * j].ids
                split_ids = encodings[k + 3 * j + 1].ids + encodings[k + 3 * j + 2].ids
                if whole != split_ids:
                    logger.info(
                        "parallel_tokenizer: boundary verification failed at char %d "
                        "(len=%d, %d chunks); falling back to single-shot encode",
                        p,
                        len(text),
                        k,
                    )
                    return self.tokenizer.encode(text, **kwargs)

        ids: List[int] = []
        for e in encodings[:k]:
            ids.extend(e.ids)
        return ids
