#!/usr/bin/env python3
"""Segment classifier for the pi-agent corpus: tool-JSON / code / prose shares.

Token-level classification matters for the two-drafter gate (Gate A in
variants.md): if one segment class is both large (>15% of tokens) and
differentially hard for the draft, a specialized drafter pays off.

Run (CPU):  python3 segment_shares.py [--max-chars N]
Writes: segment_shares.json next to the corpus.
"""
import argparse
import json
import os
import re

TOK = "/projects/s6p/hf/hub/models--zai-org--GLM-5.3/snapshots/e0b07fd2751b42d5efa199cc02c2b271deadc516/tokenizer.json"
CORPUS = "/scratch/s6p/fergus.s6p/grace-1m/lanes/workload/out/corpus_pi.txt"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "segment_shares.json")

# Marker-based segmentation (priority: tool-JSON > code fence > prose).
TOOL_JSON = re.compile(
    r'\{\s*"(?:name|command|path|cmd|file_path|pattern|query|url)"\s*:[^{}]'
    r'(?:(?!\n\n).)*?\}',  # conservative: single-level, ends before blank line
    re.S,
)
FENCE = re.compile(r"```.*?(?:```|$)", re.S)


def segments(text):
    """Yield (kind, chunk) covering the text, in order."""
    pos = 0
    spans = []
    for m in TOOL_JSON.finditer(text):
        spans.append((m.start(), m.end(), "tool_json"))
    for m in FENCE.finditer(text):
        spans.append((m.start(), m.end(), "code"))
    spans.sort()
    last = 0
    for s, e, k in spans:
        if s < last:
            continue  # overlapping (JSON inside a fence counts as code)
        if s > pos:
            yield "prose", text[pos:s]
        yield k, text[s:e]
        pos = e
    if pos < len(text):
        yield "prose", text[pos:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    text = open(CORPUS, encoding="utf-8", errors="replace").read()
    if args.max_chars:
        text = text[: args.max_chars]
    tok = Tokenizer.from_file(TOK)

    counts = {}
    chars = {}
    nseg = {}
    for kind, chunk in segments(text):
        n = len(tok.encode(chunk).ids)
        counts[kind] = counts.get(kind, 0) + n
        chars[kind] = chars.get(kind, 0) + len(chunk)
        nseg[kind] = nseg.get(kind, 0) + 1
    total = sum(counts.values())
    out = {
        "corpus_chars": len(text),
        "tokens": total,
        "shares": {k: {"tokens": v, "share": v / total, "chars": chars[k], "segments": nseg[k]}
                   for k, v in counts.items()},
    }
    print(json.dumps(out, indent=2))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
