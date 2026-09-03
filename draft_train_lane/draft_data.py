"""Training data for the draft: windows of (tokens, teacher-forced target hiddens).

Real mode: built from a capture dir (draft_capture_reader format). Sequences are
reassembled per rid; rids are clustered into sessions by the hash of their first
4096 tokens (all real sessions share the COMMON/IMPL prefix but diverge within
the lane brief, well inside 4096 tokens); whole sessions are held out for val.
ABSOLUTE positions are preserved (rope needs them; capture stores them).
Sessions whose rid streams tile the position axis contiguously (replay traffic:
every token is prefilled exactly once) become ONE stream, so windows may span
turn boundaries; non-contiguous sessions (e.g. real lane traffic whose outputs
are decode-computed and never prefilled) fall back to per-rid streams. When
`attn_sink` > 0, `get` prepends the session's first `attn_sink` tokens
(positions 1..A; position 0 is never a draft input: the engine's draft-extend
pairs input_ids[1:] with target hiddens) as attention-only context — the
LongSpec anchor-offset block — and returns `ctx_len` so the losses skip
supervising it.

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
        pos = np.arange(s, s + window, dtype=np.int64)  # abs positions s..s+window-1
        return toks.astype(np.int64), prev, pos, 0


class RealData:
    def __init__(self, capdir, window=1024, holdout_sessions=6, seed=0,
                 attn_sink=0):
        self.window = window
        self.attn_sink = int(attn_sink)
        seqs, shards = load_dir(capdir)
        print(f"[data] shards={len(shards)} requests={len(seqs)}")
        sess = {}
        streams = {}
        for rid, recs in seqs.items():
            toks, hids, starts, contig = reassemble(recs)
            if len(toks) < 2 or not contig:
                continue
            streams[rid] = (toks, hids.astype(np.float16), int(starts[0]))
            h = hashlib.sha1(toks[: min(4096, len(toks))].tobytes()).digest()[:8]
            sess.setdefault(h, []).append(rid)
        # Session streams: join a session's rid streams when they tile the
        # position axis contiguously (replay traffic: every token is prefilled
        # exactly once) -> one stream, windows may span turn boundaries, and the
        # session-start sink block is available. Non-contiguous sessions (real
        # lane traffic: outputs are decode-computed, never prefilled) keep
        # per-rid streams (windows stay within a rid; no sink for those).
        self.streams = {}
        n_joined = 0
        for h, rids in sess.items():
            rids = sorted(rids, key=lambda r: streams[r][2])
            contiguous = True
            pos = streams[rids[0]][2]
            for r in rids:
                t, _, st = streams[r]
                if st != pos:
                    contiguous = False
                    break
                pos = st + len(t)
            if contiguous and streams[rids[0]][2] == 0:
                # A stream that tiles from session position 0 (replay traffic:
                # the first request's prompt starts at 0): one stream, windows
                # may span turn boundaries, and the session-start sink block
                # (positions 1..A with prev-hiddens h_0..h_{A-1}) is available.
                n_joined += 1
                self.streams[h] = (
                    np.concatenate([streams[r][0] for r in rids]),
                    np.concatenate([streams[r][1] for r in rids]),
                    streams[rids[0]][2],
                    True,
                )
            else:
                # Non-contiguous, or starting past position 0 (e.g. lane
                # traffic captured mid-session): per-rid streams, no sink.
                for r in rids:
                    t, hd, st = streams[r]
                    self.streams[r] = (t, hd, st, False)
        sessions = sorted(self.streams, key=lambda k: self.streams[k][2])
        if self.attn_sink:
            # Sink arms train only on contiguous (joined) sessions: every batch
            # then has the uniform [sink A | window w] shape (no ragged ctx, no
            # padding ambiguity in the loss). Non-joined streams (real lane
            # traffic with decode-computed output gaps) are excluded — noted
            # limitation; replay captures are fully contiguous.
            before = len(sessions)
            sessions = [k for k in sessions if self.streams[k][3]]
            print(
                f"[data] attn_sink={self.attn_sink}: restricted to "
                f"{len(sessions)}/{before} contiguous sessions"
            )
        print(
            f"[data] {len(self.streams)} streams ({n_joined} contiguous multi-rid "
            f"sessions) in {len(sess)} session groups"
        )
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(sessions))
        val_s = set(perm[:holdout_sessions].tolist())
        self.val_keys, self.train_keys = set(), set()
        for i, k in enumerate(sessions):
            (self.val_keys if i in val_s else self.train_keys).add(k)
        self.train_idx = self._index(self.train_keys)
        self.val_idx = self._index(self.val_keys)
        ntr = sum(len(self.streams[k][0]) for k in self.train_keys)
        nva = sum(len(self.streams[k][0]) for k in self.val_keys)
        print(
            f"[data] train {len(self.train_keys)} streams / {ntr} tok / {len(self.train_idx)} win; "
            f"val {len(self.val_keys)} streams / {nva} tok / {len(self.val_idx)} win"
        )

    def _index(self, keys):
        idx = []
        for k in keys:
            toks, _, start, joined = self.streams[k]
            n = len(toks)
            # with a sink, windows must start past the sink block (s > A); the
            # early-session region is covered by the window itself there
            lo = (self.attn_sink + 1) if (self.attn_sink and joined) else 1
            for s in range(lo, n - self.window, max(1, self.window // 2)):
                idx.append((k, s))
        return idx

    def get(self, key, s, window):
        """Returns (tokens, prev_hiddens, positions, ctx_len).

        tokens/prev_hiddens: arrays of length ctx_len + window; positions:
        session-absolute int64; ctx_len: leading sink tokens (attention-only,
        not supervised) or 0. The window occupies session positions
        [s, s+window); the sink block is the session's tokens at positions
        [1, 1+A) with their own prev-hiddens. Sink exists only for contiguous
        (joined) sessions; other windows return ctx_len=0 (window-only
        conditioning — the per-batch loss mask adapts via make_batch's min).
        """
        toks, hids, start, joined = self.streams[key]
        if self.attn_sink and joined and s > self.attn_sink:
            A = self.attn_sink
            st = np.concatenate([toks[1 : 1 + A], toks[s : s + window]]).astype(np.int64)
            sp = np.concatenate(
                [hids[0:A], hids[s - 1 : s + window - 1]]
            ).astype(np.float16)
            spos = np.concatenate(
                [
                    np.arange(1, 1 + A, dtype=np.int64),
                    np.arange(s, s + window, dtype=np.int64),
                ]
            )
            return st, sp, spos, A
        return (
            toks[s : s + window].astype(np.int64),
            hids[s - 1 : s + window - 1],
            np.arange(s, s + window, dtype=np.int64),
            0,
        )


def make_batch(getter, items, window, device):
    """items: list of args for getter. Returns tokens [B,n], prev [B,n,H],
    pos [B,n], ctx_len. All rows in a batch have the same length (uniform
    [sink A | window w] when a sink is configured — RealData restricts sink
    arms to contiguous sessions — or plain [window] otherwise), so n is exact
    and no padding is needed. ctx_len is the uniform leading-context count
    the losses must skip supervising."""
    out = [getter(*it, window) for it in items]
    B = len(out)
    n = out[0][0].shape[0]
    assert all(t.shape[0] == n for t, _, _, _ in out), "ragged batch"
    ctx_len = out[0][3]
    assert all(c == ctx_len for _, _, _, c in out), "mixed ctx_len in batch"
    tokens = torch.zeros(B, n, dtype=torch.long)
    prev = torch.zeros(B, n, HIDDEN, dtype=torch.float16)
    pos = torch.zeros(B, n, dtype=torch.long)
    for i, (t, p, a, c) in enumerate(out):
        tokens[i] = torch.from_numpy(np.asarray(t))
        prev[i] = torch.from_numpy(np.asarray(p, dtype=np.float16))
        pos[i] = torch.from_numpy(np.asarray(a, dtype=np.int64))
    return tokens.to(device), prev.to(device), pos.to(device), ctx_len
