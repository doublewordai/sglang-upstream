#!/usr/bin/env python3
"""Inspect + build segment-type labeling for the corpus (two-drafter analysis).

Segment taxonomy (from the workload lane's profile): thinking, tool-call JSON,
tool results / prose. This script measures what in-band markers exist in
corpus_pi.txt and produces a per-line segment map we can use to label training
windows once capture data exists.
"""
import json
import re
import sys

CORPUS = "/scratch/s6p/fergus.s6p/grace-1m/lanes/workload/out/corpus_pi.txt"


def main():
    data = open(CORPUS, encoding="utf-8", errors="replace").read()
    print(f"corpus chars: {len(data)}")

    for pat, name in [
        (r"<think>", "think_open"),
        (r"</think>", "think_close"),
        (r"ichtung|思考", "zh_think"),
        (r"\btool_call\b", "tool_call_word"),
        (r"^\s*\{[\"']name[\"']", "json_name_start"),
        (r"```", "fence"),
        (r"\bdef \w+\(", "python_def"),
        (r"\bfunction\b", "function_word"),
    ]:
        n = len(re.findall(pat, data, re.M))
        print(f"{name:18s} {n}")

    # show a window around the first 'tool_call' occurrence
    i = data.find("tool_call")
    if i >= 0:
        print("\n--- around first tool_call ---")
        print(data[i - 200 : i + 300])

    # character-level coarse segmentation with a simple priority scanner
    segs = []
    pos = 0
    pat_think_open = re.compile(r"<think>|<\|begin▁of▁sentence\|>")
    # Fallback heuristic segments (documented approximations):
    rules = [
        ("json", re.compile(r"\{[^{}]{0,400}?\"(?:name|command|path|cmd)\"\s*:.*?\}", re.S)),
        ("code", re.compile(r"```.*?```", re.S)),
    ]
    print("\ncoarse heuristic counts (2MB sample):")
    sample = data[:2_000_000]
    for name, pat in rules:
        print(name, len(pat.findall(sample)))


if __name__ == "__main__":
    main()
