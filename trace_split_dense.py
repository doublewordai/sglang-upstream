"""Split dense deep_gemm / quant / act kernels by (name, grid) to attribute
shared-expert vs attention GEMMs in the prefill 8k trace.
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
sel = []
for k in kernels:
    i = bisect.bisect_right(bounds, (k["ts"], float("inf"))) - 1
    if i < 0 or k["ts"] >= bounds[i][1]:
        continue
    n = k["name"]
    if ("sm90_fp8_gemm_1d2d_impl" in n and ", 1u," in n) or "per_token_group_quant" in n or "act_and_mul" in n or "nvjet" in n:
        sel.append(k)

per = defaultdict(lambda: defaultdict(list))
for k in sel:
    g = tuple(k.get("args", {}).get("grid", []))
    # dense deep_gemm: name template block sizes; grid gives m-blocks x n-blocks
    key = (k["name"][:72], g)
    per[key][k["ts"]].append(k["dur"])

print(f"{'us/step':>9} {'n/step':>6} {'p50us':>7} {'p90us':>7}  grid  name")
rows = []
for (name, g), byt in per.items():
    tots = [sum(v) for v in byt.values()]
    counts = [len(v) for v in byt.values()]
    alld = sorted(d for v in byt.values() for d in v)
    rows.append((statistics.median(tots), statistics.median(counts), alld[len(alld)//2], alld[int(len(alld)*.9)], g, name))
rows.sort(reverse=True)
for tot, cnt, p50, p90, g, name in rows:
    if tot < 20:
        continue
    print(f"{tot:9.1f} {cnt:6.1f} {p50:7.1f} {p90:7.1f}  {str(g):>18}  {name}")
