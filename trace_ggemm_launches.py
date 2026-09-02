"""Per-launch (grid, dur) for the grouped MoE GEMMs + act/quant in the trace.
Grid encodes padded M; pairs with dur give achieved TF/s per layer.
"""
import bisect, gzip, json, statistics, sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)
events = data["traceEvents"] if isinstance(data, dict) else data
steps = [e for e in events if e.get("cat") == "user_annotation" and e.get("name", "").startswith("step[")]
steps.sort(key=lambda e: e["ts"])
bounds = [(e["ts"], e["ts"] + e["dur"]) for e in steps]

kernels = [e for e in events if e.get("cat") == "kernel" and "dur" in e]

def in_step(ts):
    i = bisect.bisect_right(bounds, (ts, float("inf"))) - 1
    return i if (i >= 0 and ts < bounds[i][1]) else -1

# grouped gemm templates: N,K are in the name; block_m/n from template tail
GU = ("4096u, 6144u, 64u", 4096, 6144)   # gate_up: N=4096 K=6144
DN = ("6144u, 2048u, 64u", 6144, 2048)   # down:   N=6144 K=2048

recs = []
for k in kernels:
    n = k["name"]
    if "sm90_fp8_gemm_1d2d_impl" not in n:
        continue
    for tag, N, K in (GU, DN):
        if tag in n:
            i = in_step(k["ts"])
            if i < 0:
                continue
            g = k.get("args", {}).get("grid", [])
            # parse block dims from name tail: ..., 128u, 192u, 128u, 128u... -> bm, bn, bk?
            # print raw grid; compute later
            recs.append((tag, i, tuple(g), k["dur"], n))
            break

by = defaultdict(list)
for tag, i, g, d, n in recs:
    by[tag].append((i, g, d, n))

for tag, N, K in (GU, DN):
    print(f"=== {tag} N={N} K={K}: {len(by[tag])} launches ===")
    for i, g, d, n in by[tag][:40]:
        # grid x = m_blocks*n_blocks (flattened?) or z=group? print all
        print(f"  step {i:3d} grid={g} dur={d:8.1f}us  name_tail={n[n.find(tag)-30:n.find(tag)+90]}")
