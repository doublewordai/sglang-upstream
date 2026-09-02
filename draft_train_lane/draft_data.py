"""Training data for the draft: windows of (tokens, teacher-forced target hiddens).

Real mode: built from a capture dir (draft_capture_reader format). Sequences are
reassembled per rid; rids are clustered into sessions by the hash of their first
4096 tokens (all real sessions share the COMMON/IMPL prefix but diverge within
the lane brief, well inside 4096 tokens); whole sessions are held out for val.
ABSOLUTE positions are preserved (rope needs them; capture stores them).

Synthetic mode: Markov tokens + deterministic hidden states (low-rank token basis
+ noise) so both the CE and the feature loss can genuinely decrease — this is
the "prove the loop" data.
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft_capture_reader import load_dir, reassemble  # noqa: E402

VOCAB = 154880
HIDDEN = 6144


class SyntheticData:
    def __init__(self, seed=0, n_seqs=64, seq_len=8192):
        g = torch.Generator().manual_seed(seed)
        self.n_seqs, self.seq_len = n_seqs, seq_len
        self.nexts = torch.randint(0, VOCAB, (VOCAB, 8), generator=g)
        basis = torch.randn(VOCAB, 64, generator=g) * 0.5
        proj = torch.randn(64, HIDDEN, generator=g) / 8.0
        emb = (basis @ proj).numpy().astype(np.float16)  # [V, H]
        self.emb = emb
        self.noise = 0.05 * float(emb[:2000].astype(np.float32).std())  # fp32 slice: full fp16 std overflows
        self.seqs = []
        for i in range(n_seqs):
            t = int(torch.randint(0, VOCAB, (1,), generator=g))
            toks = [t]
            for _ in range(seq_len - 1):
                t = int(self.nexts[toks[-1], torch.randint(0, 8, (1,), generator=g)])
                toks.append(t)
            self.seqs.append(np.array(toks, dtype=np.int64))

    def hidden(self, tokens: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(abs(int(tokens[0])) % (2**31))
        h = self.emb[tokens].astype(np.float32)
        h = h + rng.randn(*h.shape).astype(np.float32) * self.noise
        return h.astype(np.float16)

    def windows(self, n, window, rng, split="train"):
        lo = 0 if split == "train" else self.n_seqs // 2
        hi = self.n_seqs // 2 if split == "train" else self.n_seqs
        out = []
        for _ in range(n):
            si = rng.randint(lo, hi)
            s = rng.randint(1, self.seq_len - window - 1)
            out.append((si, s))
        return out

    def get(self, si, s, window):
        toks = self.seqs[si][s : s + window]
        prev = self.hidden(self.seqs[si][s - 1 : s + window - 1])
        return toks.astype(np.int64), prev, s  # abs positions s..s+window-1


class RealData:
    def __init__(self, capdir, window=1024, holdout_sessions=6, seed=0):
        self.window = window
        seqs, shards = load_dir(capdir)
        print(f"[data] shards={len(shards)} requests={len(seqs)}")
        sess = {}
        streams = {}
        for rid, recs in seqs.items():
            toks, hids, starts, contig = reassemble(recs)
            if len(toks) < window + 2 or not contig:
                continue
            streams[rid] = (toks, hids.astype(np.float16), int(starts[0]))
            h = hashlib.sha1(toks[: min(4096, len(toks))].tobytes()).digest()[:8]
            sess.setdefault(h, []).append(rid)
        sessions = sorted(sess.values(), key=lambda r: min(streams[r][2] for r in r))
        print(f"[data] {len(streams)} usable rids in {len(sessions)} sessions")
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(sessions))
        val_s = set(perm[:holdout_sessions].tolist())
        self.val_rids, self.train_rids = set(), set()
        for i, s in enumerate(sessions):
            (self.val_rids if i in val_s else self.train_rids).update(s)
        self.streams = streams
        self.train_idx = self._index(self.train_rids)
        self.val_idx = self._index(self.val_rids)
        ntr = sum(len(self.streams[r][0]) for r in self.train_rids)
        nva = sum(len(self.streams[r][0]) for r in self.val_rids)
        print(
            f"[data] train {len(self.train_rids)} rids / {ntr} tok / {len(self.train_idx)} win; "
            f"val {len(self.val_rids)} rids / {nva} tok / {len(self.val_idx)} win"
        )

    def _index(self, rids):
        idx = []
        for rid in rids:
            toks, _, _ = self.streams[rid]
            n = len(toks)
            for s in range(1, n - self.window, max(1, self.window // 2)):
                idx.append((rid, s))
        return idx

    def get(self, rid, s, window):
        toks, hids, start = self.streams[rid]
        return (
            toks[s : s + window].astype(np.int64),
            hids[s - 1 : s + window - 1],
            start + s,
        )


def make_batch(getter, items, window, device):
    """items: list of args for getter. Returns tokens [B,n], prev [B,n,H], pos [B,n]."""
    B = len(items)
    tokens = torch.zeros(B, window, dtype=torch.long)
    prev = torch.zeros(B, window, HIDDEN, dtype=torch.float16)
    pos = torch.zeros(B, window, dtype=torch.long)
    for i, it in enumerate(items):
        t, p, a = getter(*it, window)
        tokens[i] = torch.from_numpy(np.asarray(t[-window:]))
        prev[i] = torch.from_numpy(np.asarray(p[-window:]))
        pos[i] = int(a) + torch.arange(window)
    return tokens.to(device), prev.to(device), pos.to(device)
