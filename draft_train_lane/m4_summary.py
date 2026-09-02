#!/usr/bin/env python3
"""M4 A/B summary: table over all accept-*.json in the m4/ dir."""
import glob
import json
import sys

rows = []
for p in sorted(glob.glob("/scratch/s6p/fergus.s6p/grace-1m/lanes/draft-train/m4/accept-*.json")):
    try:
        rows.append(json.load(open(p)))
    except Exception as e:
        print(f"skip {p}: {e}", file=sys.stderr)

if not rows:
    print("no accept-*.json found")
    sys.exit(0)

print(f"{'arm':10s} {'n':>5s} {'mean':>7s} {'p50':>7s} {'max':>7s}")
for r in rows:
    print(
        f"{r['tag']:10s} {r['n']:5d} {r['accept_len_mean']:7.3f} "
        f"{r['accept_len_p50']:7.3f} {r['accept_len_max']:7.3f}"
    )

# depth curves per draft
for draft in ("old", "new"):
    sub = [r for r in rows if r["tag"].startswith(draft + "-")]
    if len(sub) >= 2:
        pts = "  ".join(f"d{r['tag'].split('-')[1]}={r['accept_len_mean']:.3f}" for r in sub)
        print(f"\n{draft}: {pts}")
old = {r["tag"].split("-")[1]: r["accept_len_mean"] for r in rows if r["tag"].startswith("old-")}
new = {r["tag"].split("-")[1]: r["accept_len_mean"] for r in rows if r["tag"].startswith("new-")}
for d in sorted(set(old) & set(new)):
    delta = new[d] - old[d]
    print(f"depth {d}: new-old = {delta:+.3f} ({100*delta/max(old[d],1e-9):+.1f}%)")
