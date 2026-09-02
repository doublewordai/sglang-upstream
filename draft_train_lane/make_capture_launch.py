#!/usr/bin/env python3
"""Patch l3-launch-v17.sh -> l3-launch-capture.sh (draft-train capture rig)."""
P = "/scratch/s6p/fergus.s6p/grace-1m/lanes/draft-train/l3-launch-capture.sh"
s = open(P).read()
orig = s

# 1. ports -> lane range 57000-57999 (draft-train = LANES.txt line 17)
s = s.replace(
    "PREFILL_PORT=31400\nDECODE_PORT=31300\nLB_PORT=31500\nBOOTSTRAP_PORT=8998",
    "PREFILL_PORT=57000\nDECODE_PORT=57100\nLB_PORT=57200\nBOOTSTRAP_PORT=57300",
)
s = s.replace('"$PREFILL_MASTER:28400"', '"$PREFILL_MASTER:57400"')
s = s.replace('"$DECODE_MASTER:28100"', '"$DECODE_MASTER:57500"')
s = s.replace('"$MASTER:28200"', '"$MASTER:57600"')

# 2. sglang tree -> draft-train worktree
s = s.replace(
    'SGLANG_TREE="${SGLANG_TREE:-$SCRATCH/src/sglang-hostpage-0902}"',
    'SGLANG_TREE="${SGLANG_TREE:-$SCRATCH/src/sglang-draft-train-0902}"',
)

# 3. capture env passed into the big srun step
old_export = (
    'MODEL="$MODEL",SERVED_NAME="$SERVED_NAME",OVERRIDE_ARGS="$OVERRIDE_ARGS",DECODE_A2A="$DECODE_A2A",CTX="$CTX",'
    'MTT_DECODE="${MTT_DECODE:-150000}",MRR_DECODE="${MRR_DECODE:-512}",HISPARSE_RATIO="$HISPARSE_RATIO",'
    'HISPARSE_BUF="$HISPARSE_BUF",U_ROOT="$U_ROOT",GRAPH_FLAGS="$GRAPH_FLAGS",SGLANG_TREE="$SGLANG_TREE",'
    'SGLANG_MAP_HOST_POOL_PRIVATE="$SGLANG_MAP_HOST_POOL_PRIVATE" \\'
)
new_export = old_export.replace(
    'SGLANG_MAP_HOST_POOL_PRIVATE="$SGLANG_MAP_HOST_POOL_PRIVATE" \\',
    'SGLANG_MAP_HOST_POOL_PRIVATE="$SGLANG_MAP_HOST_POOL_PRIVATE",'
    'SGLANG_DRAFT_CAPTURE_DIR="$SGLANG_DRAFT_CAPTURE_DIR",'
    'SGLANG_DRAFT_CAPTURE_MODES="${SGLANG_DRAFT_CAPTURE_MODES:-extend}",'
    'SGLANG_DRAFT_CAPTURE_TAG="${SGLANG_DRAFT_CAPTURE_TAG:-prefill}" \\',
)
assert old_export in s, "export anchor not found"
s = s.replace(old_export, new_export)

# default capture dir next to U_ROOT
s = s.replace(
    "U_ROOT=$SCRATCH/runs/glm-isambard/U-uccl-send-abort",
    "U_ROOT=$SCRATCH/runs/glm-isambard/U-uccl-send-abort\n"
    "# draft-train capture output (extend-only; ~12.3 KB/token, fp16 hidden 6144)\n"
    'export SGLANG_DRAFT_CAPTURE_DIR="${SGLANG_DRAFT_CAPTURE_DIR:-'
    '$SCRATCH/grace-1m/lanes/draft-train/capture/real-DATE}"\n'
    'mkdir -p "$SGLANG_DRAFT_CAPTURE_DIR"',
)
# DATE placeholder resolved at runtime (avoid $(date) inside the python patch)
s = s.replace(
    "capture/real-DATE",
    'capture/real-$(date -u +%Y%m%d-%H%M%S)"',
).replace(
    '"${SGLANG_DRAFT_CAPTURE_DIR:-$SCRATCH/grace-1m/lanes/draft-train/capture/real-$(date -u +%Y%m%d-%H%M%S)"',
    '${SGLANG_DRAFT_CAPTURE_DIR:-$SCRATCH/grace-1m/lanes/draft-train/capture/real-$(date -u +%Y%m%d-%H%M%S)}',
)

# 4. drop --enable-metrics (DuplicateTimeseries crash until the metrics lane merges)
s = s.replace(" --enable-metrics\n", "\n")

# 5. fix the never-booted NONDISAGG branch (fallback 4-node rig)
s = s.replace(
    "--tp-size 16 --dp-size 16 --enable-dp-attention --enable-dp-lm-head --ep-size 16 \\\n"
    "                --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm \\",
    "--tp-size 16 --dp-size 16 --enable-dp-attention --enable-dp-lm-head --ep-size 16 \\\n"
    "                --moe-a2a-backend deepep --deepep-mode normal --moe-runner-backend deep_gemm \\",
)
s = s.replace(
    "--moe-dense-tp-size 1 --load-balance-method follow_bootstrap_room --mem-fraction-static 0.88",
    "--moe-dense-tp-size 1 --load-balance-method prefix_affinity --mem-fraction-static 0.88",
)

# header note
s = s.replace(
    "# The Doubleword GLM PD-disagg serving system on 8 GH200 nodes (prod configuration).",
    "# draft-train CAPTURE rig: l3-launch-v17.sh + (a) SGLANG_TREE=sglang-draft-train-0902 (capture\n"
    "# hook), (b) SGLANG_DRAFT_CAPTURE_* env (extend-mode hidden-state capture on the prefill arm,\n"
    "# which runs eager extends), (c) lane port range 57000-57999, (d) no --enable-metrics,\n"
    "# (e) NONDISAGG branch flags fixed (prefix_affinity + deepep normal) as a 4-node fallback.",
)

assert s != orig
open(P, "w").write(s)
print("patched OK")
