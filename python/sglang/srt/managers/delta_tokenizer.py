"""Delta tokenization cache for the tokenizer manager.

Speeds up warm-turn prompt tokenization for agent-style traffic where each
request resends the full conversation and the rendered prompt grows by
appending (measured: 99.9% of real pi-agent turn pairs are pure appends).

Exactness argument (verified by scripts/oracle_delta.py over all 3,282 real
turn boundaries, 100% identical ids):
  1. Added (special) tokens are extracted atomically before BPE in the
     `tokenizers` pipeline; a special-token occurrence is therefore a hard
     split point in ANY text containing it.
  2. The cache stores (rendered_prompt, input_ids, token_end_offsets) of the
     previous request of a session. A new request's rendered prompt shares a
     character prefix [0, L) with the cached one.
  3. The cut c is chosen as the END of the last special-token occurrence that
     lies fully within [0, L). Both the cached and the new tokenization have
     a token boundary at c (the atomic special token ends there), and BPE
     pre-tokens never span that boundary, so
         ids_new == ids_old[:j] + encode(text_new[c:])
     with j = number of cached tokens ending at or before c.

If no special-token boundary exists in the shared prefix (or the cache misses),
we fall back to a full encode — the output is then identical to the stock path
by construction.

The per-session key is a hash of the system prompt and the first
non-system message content. A key "collision" is harmless: correctness comes
from the common-prefix computation, the key only selects the candidate entry;
worst case the cut is early and more of the prompt is re-encoded.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from collections import OrderedDict
from typing import List, Optional, Tuple

from sglang.srt.managers.parallel_tokenizer import (
    parallel_encode_exact,
    parallel_tokenize_enabled,
)


class DeltaTokenizerCache:
    VERIFY_WINDOW = 8

    def __init__(self, tokenizer, max_sessions: int = 64):
        self.tokenizer = tokenizer
        specials = [t for t in tokenizer.get_added_vocab() if t]
        # longest first so rfind's leftmost-longest semantics per token are
        # irrelevant; we take the rightmost END among all specials anyway
        self.specials = sorted(specials, key=len, reverse=True)
        self.max_sessions = max_sessions
        self.cache: "OrderedDict[str, Tuple[str, List[int], List[int]]]" = OrderedDict()
        self.stats = {"hit": 0, "miss": 0, "fallback": 0, "bytes_saved": 0}

    @staticmethod
    def _common_prefix_len(a: str, b: str) -> int:
        # C-speed: binary search on slice equality
        lo, hi = 0, min(len(a), len(b))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    @classmethod
    def session_key(cls, messages) -> Optional[str]:
        """Stable per-conversation fingerprint: system + first non-system msg."""
        try:
            sysmsg = next(
                (m for m in messages if isinstance(m, dict) and m.get("role") == "system"),
                None,
            )
            first = next(
                (m for m in messages if isinstance(m, dict) and m.get("role") != "system"),
                None,
            )
        except TypeError:
            return None
        if first is None:
            return None
        h = hashlib.md5()
        content = sysmsg.get("content") if sysmsg else ""
        h.update((content if isinstance(content, str) else json.dumps(content)).encode())
        h.update(b"\x00")
        content = first.get("content")
        h.update((content if isinstance(content, str) else json.dumps(content))[:4096].encode())
        return h.hexdigest()

    def _last_special_end(self, text: str, limit: int) -> int:
        c = -1
        for s in self.specials:
            p = text.rfind(s, 0, limit)
            if p >= 0:
                e = p + len(s)
                if e > c:
                    c = e
        return c

    def encode(self, text: str, key: Optional[str], **encode_kwargs) -> List[int]:
        """Return input_ids for `text`, reusing the cached prefix when possible.

        `encode_kwargs` must be the same as the stock call site uses (e.g.
        add_special_tokens=False) so that full-encode fallbacks match exactly.
        """
        entry = self.cache.get(key) if key else None
        if entry is not None:
            old_text, old_ids, old_ends = entry
            limit = self._common_prefix_len(old_text, text)
            c = self._last_special_end(text, limit)
            if c > 0:
                j = bisect.bisect_right(old_ends, c)
                # Boundary-window re-verification: the last VERIFY_WINDOW tokens
                # of the cached prefix must re-encode identically when cut at c.
                # Catches any (unexpected) merge crossing the cut; on mismatch we
                # fall back to a full encode, so ids are never wrong.
                w = min(self.VERIFY_WINDOW, j)
                s0 = old_ends[j - w - 1] if j > w else 0
                if self.tokenizer.encode(
                    old_text[s0:c], **encode_kwargs
                ) == old_ids[j - w : j]:
                    suffix = self.tokenizer(
                        text[c:], return_offsets_mapping=True, **encode_kwargs
                    )
                    ids = old_ids[:j] + suffix["input_ids"]
                    ends = old_ends[:j] + [c + e for _, e in suffix["offset_mapping"]]
                    self._store(key, text, ids, ends)
                    self.stats["hit"] += 1
                    self.stats["bytes_saved"] += c
                    return ids
            self.stats["fallback"] += 1
        # miss (or no usable boundary): full encode, remember for next time.
        # Composes with the parallel chunked tokenizer (SGLANG_PARALLEL_TOKENIZE):
        # the parallel path is token-identical, so the cached entry is identical.
        ids, ends = parallel_encode_exact(self.tokenizer, text, **encode_kwargs)
        if ends is None:
            enc = self.tokenizer(text, return_offsets_mapping=True, **encode_kwargs)
            ids = enc["input_ids"]
            ends = [e for _, e in enc["offset_mapping"]]
        ids = enc["input_ids"]
        ends = [e for _, e in enc["offset_mapping"]]
        if key:
            self._store(key, text, ids, ends)
        self.stats["miss"] += 1
        return ids

    def _store(self, key, text, ids, ends):
        self.cache[key] = (text, ids, ends)
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_sessions:
            self.cache.popitem(last=False)
