#!/usr/bin/env python3
"""Reader/validator for draft-capture shard files (format: see sglang/srt/draft_capture.py).

Subcommands:
  stats <dir>                    per-shard counts + per-rid reassembly coverage
  validate <dir> --sent F.jsonl  dummy-rig test: match captured records against the
                                 requests the client sent (content-matched, no rid
                                 knowledge needed) and check decode records against
                                 the server-reported output ids.

Works with numpy only (no torch) so it can run anywhere.
"""
from __future__ import annotations

import glob
import json
import struct
import sys
from collections import defaultdict

import numpy as np

MAGIC = 0x44524331
HEADER = struct.Struct("<IIQqiii")
HEADER_SIZE = HEADER.size  # 36


def iter_records(path):
    """Yield (rid_hash, start_pos, n_tok, hidden_dim, tokens i32[n], hidden f16[n,d])."""
    with open(path, "rb") as f:
        while True:
            hdr = f.read(HEADER_SIZE)
            if len(hdr) == 0:
                break
            if len(hdr) != HEADER_SIZE:
                raise ValueError(f"{path}: truncated header at {f.tell() - len(hdr)}")
            magic, version, rid, start, n, hdim, dt = HEADER.unpack(hdr)
            if magic != MAGIC:
                raise ValueError(f"{path}: bad magic {magic:#x} at {f.tell() - HEADER_SIZE}")
            if version != 1:
                raise ValueError(f"{path}: bad version {version}")
            if dt != 0:
                raise ValueError(f"{path}: unknown dtype code {dt}")
            tok = np.frombuffer(f.read(4 * n), dtype="<i4")
            hid = np.frombuffer(f.read(2 * n * hdim), dtype="<f2")
            if tok.shape[0] != n or hid.shape[0] != n * hdim:
                raise ValueError(f"{path}: truncated payload")
            yield rid, start, n, hdim, tok, hid.reshape(n, hdim)


def load_dir(capdir):
    """Return {rid_hash: [(start_pos, tokens, hidden), ...]} plus shard stats."""
    seqs = defaultdict(list)
    shards = []
    for path in sorted(glob.glob(f"{capdir}/shard-*.bin")):
        nrec = ntok = 0
        for rid, start, n, hdim, tok, hid in iter_records(path):
            seqs[rid].append((start, tok, hid))
            nrec += 1
            ntok += n
        shards.append((path.split("/")[-1], nrec, ntok))
    for rid in seqs:
        seqs[rid].sort(key=lambda t: t[0])
    return seqs, shards


def reassemble(recs):
    """Concatenate a rid's records -> (tokens, hidden, starts, contiguous)."""
    toks = np.concatenate([t for _, t, _ in recs]) if recs else np.zeros(0, "<i4")
    hids = np.concatenate([h for _, _, h in recs], 0) if recs else np.zeros((0, 0), "<f2")
    starts = np.array([s for s, _, _ in recs], dtype=np.int64)
    contig = all(
        recs[i + 1][0] == recs[i][0] + len(recs[i][1]) for i in range(len(recs) - 1)
    )
    return toks, hids, starts, contig


def hidden_health(hids):
    rms = np.sqrt(np.mean(np.asarray(hids, dtype=np.float32) ** 2, axis=1))
    finite = bool(np.isfinite(hids).all())
    return {
        "finite": finite,
        "rms_p50": float(np.median(rms)),
        "rms_min": float(rms.min()),
        "rms_max": float(rms.max()),
        "zero_rows": int((rms == 0).sum()),
    }


def cmd_stats(capdir):
    seqs, shards = load_dir(capdir)
    print(f"shards: {len(shards)}")
    for name, nrec, ntok in shards:
        print(f"  {name}: {nrec} records, {ntok} tokens")
    total_tok = sum(ntok for _, _, ntok in shards)
    n_contig = sum(1 for recs in seqs.values() if reassemble(recs)[3])
    print(f"requests: {len(seqs)}, total tokens: {total_tok}, contiguous: {n_contig}")
    lens = sorted(sum(len(t) for _, t, _ in recs) for recs in seqs.values())
    if lens:
        print(f"per-request tokens: min {lens[0]} p50 {lens[len(lens)//2]} max {lens[-1]}")
    for path in sorted(glob.glob(f"{capdir}/shard-*.stats.json")):
        print(f"  stats {path.split('/')[-1]}: {json.load(open(path))}")


def find_match(seqs, needle, min_start=0):
    """Find the rid whose assembled token stream contains `needle` (returns rid, start_pos)."""
    hits = []
    for rid, recs in seqs.items():
        toks, _, starts, _ = reassemble(recs)
        # find needle as a contiguous run
        n = len(needle)
        for i in range(0, len(toks) - n + 1):
            if np.array_equal(toks[i : i + n], needle):
                # absolute position of toks[i]
                pos = _abs_pos(recs, i)
                if pos >= min_start:
                    hits.append((rid, pos))
                break
    return hits


def _abs_pos(recs, idx):
    off = 0
    for start, tok, _ in recs:
        if off <= idx < off + len(tok):
            return start + (idx - off)
        off += len(tok)
    return -1


def cmd_validate(capdir, sent_path):
    seqs, shards = load_dir(capdir)
    sent = [json.loads(l) for l in open(sent_path)]
    failures = []
    for s in sent:
        name = s["name"]
        prompt = np.array(s["input_ids"], dtype=np.int64)
        out = np.array(s.get("output_ids", []), dtype=np.int64)
        L = len(prompt)
        # 1) match the rid by the request's match key (unique tokens that this
        #    request itself prefilled — the growth region for prefix-reuse reqs)
        key = np.array(s.get("match_ids") or s["input_ids"][-min(len(s["input_ids"]), 2048):], dtype=np.int64)
        hits = find_match(seqs, key)
        if not hits:
            failures.append(f"{name}: no captured sequence contains its prompt tail")
            continue
        rid, tail_pos = hits[0]
        toks, hids, starts, contig = reassemble(seqs[rid])
        first_start = int(starts[0])
        # assembled stream must be prompt[first_start:] (+ output_ids when decode capture is on)
        want_decode = bool(s.get("expect_decode", True)) and len(out) > 0
        expected = prompt[first_start:]
        if want_decode:
            expected = np.concatenate([expected, out])
        if not np.array_equal(toks, expected):
            k = min(len(toks), len(expected))
            bad = int((toks[:k] != expected[:k]).sum()) if k else 0
            failures.append(
                f"{name}: assembled stream ({len(toks)}) != prompt[{first_start}:]+outputs ({len(expected)}), {bad} differing tokens"
            )
            continue
        print(
            f"{name}: rid {rid:#x} start {first_start} prefill {L - first_start} tok, "
            f"decode {len(out)} tok, records {len(seqs[rid])}, contig={contig}"
        )
        # 2) decode records: positions L..L+N-1, tokens == output ids
        dec_toks, dec_pos = [], []
        for start, tok, _ in seqs[rid]:
            for j in range(len(tok)):
                if start + j >= L:
                    dec_toks.append(int(tok[j]))
                    dec_pos.append(int(start + j))
        if s.get("expect_decode", True):
            if not dec_toks:
                failures.append(f"{name}: expected decode records, got none")
            else:
                if dec_pos != list(range(L, L + len(dec_pos))):
                    failures.append(f"{name}: decode positions not contiguous: {dec_pos[:8]}")
                if len(out) > 0 and dec_toks != [int(x) for x in out]:
                    failures.append(
                        f"{name}: decode tokens {dec_toks[:8]} != output ids {list(out[:8])}"
                    )
                else:
                    print(f"  decode: {len(dec_toks)} records, tokens==output_ids OK")
        hh = hidden_health(hids)
        print(f"  hidden: {hh}")
        if not hh["finite"]:
            failures.append(f"{name}: non-finite hidden values")
        if hh["zero_rows"] > 0:
            failures.append(f"{name}: {hh['zero_rows']} zero hidden rows")
    if failures:
        print("FAILURES:")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("VALIDATE-OK")
    return 0


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    cmd, capdir = sys.argv[1], sys.argv[2]
    if cmd == "stats":
        cmd_stats(capdir)
        return 0
    if cmd == "validate":
        sent = sys.argv[sys.argv.index("--sent") + 1]
        return cmd_validate(capdir, sent)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
