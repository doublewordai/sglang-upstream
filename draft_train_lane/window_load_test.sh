#!/usr/bin/env bash
# draft-train relaunch: draft-attention-window engine test on the dummy78 rig.
# Arm A: spec on, no window (control). Arm B: spec on + --speculative-draft-window-size
# 2048 --speculative-draft-attn-sink 64. Greedy outputs must be IDENTICAL (verify is
# untouched; only the drafter candidate keys change). Accept lines may differ.
# Runs eager prefill + decode CUDA graphs on (exercises the paged-path mask under
# graph capture). One GPU (CUDA_VISIBLE_DEVICES set by caller).
set -uo pipefail
S=/scratch/s6p/fergus.s6p
LANE=$S/grace-1m/lanes/draft-train
MODEL=$S/grace-1m/dummy78
DRAFT_DIR=${DRAFT_DIR:-$LANE/export_roundtrip}
PORT=${PORT:-57010}
OUT=${OUT:-$LANE/logs/window-load-$(date -u +%H%M%S)}
W=${W:-2048}
A=${A:-64}

if ss -ltn | grep -q ":$PORT "; then echo "PORT $PORT IN USE"; exit 1; fi
mkdir -p "$OUT"

export T_WITH_EP=1
source $S/runs/glm-isambard/U-uccl-send-abort/scripts/env-U.sh >/dev/null 2>&1 || true
export PYTHONPATH=$S/src/sglang-draft-train-0902/python:${PYTHONPATH:-}
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 SGLANG_JIT_DEEPGEMM_PRECOMPILE=false
export PYTHONDONTWRITEBYTECODE=1

boot_and_probe () {
  local TAG=$1; shift
  local EXTRA="$@"
  setsid python -m sglang.launch_server --model-path "$MODEL" --load-format dummy \
    --trust-remote-code --json-model-override-args "{\"qk_rope_head_dim\": 64}" \
    --host 127.0.0.1 --port $PORT --tp-size 1 --random-seed 42 \
    --disable-prefill-cuda-graph \
    --speculative-algorithm EAGLE --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --speculative-draft-model-path "$DRAFT_DIR" $EXTRA \
    --chunked-prefill-size 4096 --context-length 8192 --max-running-requests 4 \
    --max-total-tokens 16384 --mem-fraction-static 0.80 \
    > "$OUT/server-$TAG.log" 2>&1 &
  local SRV=$!
  for i in $(seq 1 96); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || { echo "$TAG SERVER-DIED"; tail -40 "$OUT/server-$TAG.log"; return 1; }
    sleep 5
  done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "$TAG HEALTH-TIMEOUT"; tail -40 "$OUT/server-$TAG.log"; return 1; }
  echo "$TAG SERVER-READY"
  # long-ish prompt to push past the window, then greedy
  python - "$PORT" "$OUT/probe-$TAG.json" << "PY"
import json, sys, urllib.request
port, outp = sys.argv[1], sys.argv[2]
prompt = ("List of prime numbers: 2 3 5 7 11 13 17 19 23 29 31 37 41 43 47. " * 120)
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/generate",
    data=json.dumps({"text": prompt, "sampling_params": {"max_new_tokens": 64, "temperature": 0, "ignore_eos": True}}).encode(),
    headers={"Content-Type": "application/json"},
)
r = json.load(urllib.request.urlopen(req, timeout=600))
with open(outp, "w") as f:
    json.dump({"text": r["text"][-512:]}, f)
print("probe ok", outp)
PY
  sleep 2
  grep -i "accept" "$OUT/server-$TAG.log" | tail -3 > "$OUT/accept-$TAG.txt" || true
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  for i in $(seq 1 20); do ss -ltn | grep -q ":$PORT " || break; sleep 3; done
}

boot_and_probe nowin || exit 1
boot_and_probe win --speculative-draft-window-size $W --speculative-draft-attn-sink $A || exit 1

echo "=== nowin text tail ==="; cat "$OUT/probe-nowin.json"
echo "=== win text tail ==="; cat "$OUT/probe-win.json"
if python - "$OUT" << "PY"
import json, sys
out = sys.argv[1]
a = json.load(open(f"{out}/probe-nowin.json"))["text"]
b = json.load(open(f"{out}/probe-win.json"))["text"]
sys.exit(0 if a == b else 1)
PY
then echo "EXACTNESS-OK: greedy outputs identical (window changes proposals only)"; else echo "EXACTNESS-FAIL: outputs differ"; fi
echo "=== accept lines nowin ==="; cat "$OUT/accept-nowin.txt"
echo "=== accept lines win ==="; cat "$OUT/accept-win.txt"
echo "WINDOW-LOAD-TEST-DONE OUT=$OUT"
