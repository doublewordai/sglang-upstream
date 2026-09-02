"""MoE kernel inventory from a torch-profiler chrome trace (prefill 8k chunk).

python3 trace_moe_inventory.py <trace.json.gz> [--steps N]
Per kernel name (within step[...] spans): count/step, median total us/step,
p50/p90 per-launch us, grid. Sorted by total us/step.
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

kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
per_name = defaultdict(lambda: defaultdict(list))  # name -> step_idx -> [durs]
for k in kernels:
    i = bisect.bisect_right(bounds, (k["ts"], float("inf"))) - 1
    if i < 0 or k["ts"] >= bounds[i][1]:
        continue
    per_name[k["name"]][i].append(k["dur"])

nsteps = len(steps)
print(f"# {path}: {nsteps} steps, {len(kernels)} kernels")
rows = []
for name, bystep in per_name.items():
    counts = [len(v) for v in bystep.values()]
    tots = [sum(v) for v in bystep.values()]
    alld = sorted(d for v in bystep.values() for d in v)
    med_count = statistics.median(counts)
    med_tot = statistics.median(tots)
    p50 = alld[len(alld) // 2] if alld else 0
    p90 = alld[int(len(alld) * 0.9)] if alld else 0
    rows.append((med_tot, name, med_count, p50, p90))
rows.sort(reverse=True)
print(f"{'us/step':>10} {'n/step':>7} {'p50us':>8} {'p90us':>8}  kernel")
for tot, name, cnt, p50, p90 in rows:
    if tot < 5:
        continue
    short = name if len(name) <= 110 else name[:107] + "..."
    print(f"{tot:10.1f} {cnt:7.1f} {p50:8.1f} {p90:8.1f}  {short}")
