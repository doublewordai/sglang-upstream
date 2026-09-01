"""Approximate prefix-to-DP-rank index for cache-aware DP dispatch.

Each request's input_ids is split into fixed-size blocks; block i is keyed by
hash(key_{i-1}, block_i), so a key identifies the whole prefix up to and
including that block. Per DP rank we keep an LRU set of keys the rank has been
sent. The ranks with the longest run of leading keys are the ones most likely
to still hold the prefix in their radix cache, so a multi-turn session keeps
landing on the rank that retains its context.
"""

from collections import OrderedDict
from typing import List, Sequence, Tuple


class PrefixAffinityIndex:
    def __init__(
        self, *, dp_size: int, block_tokens: int, max_blocks_per_rank: int
    ) -> None:
        assert block_tokens > 0 and max_blocks_per_rank > 0
        self.block_tokens = block_tokens
        self.max_blocks_per_rank = max_blocks_per_rank
        self._keys: List["OrderedDict[int, None]"] = [
            OrderedDict() for _ in range(dp_size)
        ]

    def prefix_keys(self, input_ids: Sequence[int]) -> List[int]:
        """Chained block hashes of input_ids; the tail partial block is ignored."""
        b = self.block_tokens
        keys: List[int] = []
        prev = 0
        for start in range(0, len(input_ids) - b + 1, b):
            prev = hash((prev, tuple(input_ids[start : start + b])))
            keys.append(prev)
        return keys

    def longest_match(self, keys: Sequence[int]) -> Tuple[List[int], int]:
        """Ranks sharing the longest leading run of keys, and that run's length
        in tokens. ([], 0) when no rank has seen the first block."""
        candidates = list(range(len(self._keys)))
        depth = 0
        for key in keys:
            alive = [r for r in candidates if key in self._keys[r]]
            if not alive:
                break
            candidates = alive
            depth += 1
        if depth == 0:
            return [], 0
        return candidates, depth * self.block_tokens

    def record(self, *, rank: int, keys: Sequence[int]) -> None:
        lru = self._keys[rank]
        for key in keys:
            if key in lru:
                lru.move_to_end(key)
            else:
                lru[key] = None
        while len(lru) > self.max_blocks_per_rank:
            lru.popitem(last=False)
