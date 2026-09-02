#!/usr/bin/env python3
"""Send deterministic requests to the dummy rig and record what was sent.

Writes sent.jsonl rows: {name, input_ids, output_ids, expect_decode}.
Content is seeded random ids in [1000, 150000); each request's stream is
unique, so captured records can be matched by content alone (rids are internal).
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request

VOCAB = 154880
LOW, HIGH = 1000, 150000


def post(port, body, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def output_ids_of(resp, debug_name=None):
    meta = resp.get("meta_info", {})
    if debug_name:
        print(f"  meta_info[{debug_name}] keys={sorted(meta.keys())}")
        for k in ("output_ids", "output_token_logprobs", "finish_reason"):
            if k in meta:
                v = meta[k]
                s = str(v)
                print(f"    {k} = {s[:120]}")
    otl = meta.get("output_token_logprobs")
    if otl:
        # entries are [logprob, token_id, text]
        return [int(x[1]) for x in otl]
    if "output_ids" in meta and meta["output_ids"] and min(meta["output_ids"]) >= 0:
        return list(meta["output_ids"])
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=57000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-decode", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(0)
    rows = []

    def gen(name, n, max_new, prefix=None, expect_decode=True, match_ids=None, debug=False):
        if prefix is None:
            ids = [rng.randint(LOW, HIGH) for _ in range(n)]
        else:
            ids = list(prefix) + [rng.randint(LOW, HIGH) for _ in range(n)]
        t0 = time.time()
        resp = post(
            args.port,
            {
                "input_ids": ids,
                "sampling_params": {
                    "max_new_tokens": max_new,
                    "temperature": 0,
                    "ignore_eos": True,
                },
                "return_logprob": True,
            },
        )
        out = output_ids_of(resp, debug_name=name if debug else None)
        dt = time.time() - t0
        print(
            f"{name}: prompt {len(ids)} -> output {len(out)} ids in {dt:.1f}s "
            f"({resp.get('meta_info', {}).get('finish_reason', {})})"
        )
        rows.append(
            {
                "name": name,
                "input_ids": ids,
                "output_ids": out,
                "expect_decode": bool(expect_decode and args.expect_decode),
                "match_ids": match_ids if match_ids is not None else ids[-min(len(ids), 2048):],
            }
        )
        return ids

    # A: long prompt, multi-chunk prefill (chunked-prefill-size 2048)
    a = gen("A", 5000, 12, debug=True)
    # B: reuse A's prefix; only the ~300-token growth should be captured
    gen("B", 300, 4, prefix=a)
    # B's match key = its growth tokens (the cached prefix lives in A's records)
    rows[-1]["match_ids"] = rows[-1]["input_ids"][-200:]
    # C: short, no reuse
    gen("C", 100, 4)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
