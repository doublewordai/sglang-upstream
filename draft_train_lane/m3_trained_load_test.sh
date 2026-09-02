#!/usr/bin/env bash
# draft-train M3: load test of the exported draft dir on the 1-GPU dummy78 rig.
# Target = dummy78 (78 layers, 8 experts, dummy weights); draft = <DRAFT_DIR>
# (roundtrip export of the REAL layer-78 NextN, 256 experts). Proves the export
# loads through GlmMoeDsaForCausalLMNextN and the EAGLE worker runs with it.
# Run inside an srun step on nid010151 with one GPU (CUDA_VISIBLE_DEVICES set).
set -uo pipefail
S=/scratch/s6p/fergus.s6p
LANE=$S/grace-1m/lanes/draft-train
MODEL=${MODEL:-$S/grace-1m/dummy78}
DRAFT_DIR=${DRAFT_DIR:-$LANE/export_trained}
PORT=${PORT:-57000}
OUT=${OUT:-$LANE/logs/m3-load-$(date -u +%H%M%S)}

if ss -ltn | grep -q ":$PORT "; then echo "PORT $PORT IN USE"; exit 1; fi
mkdir -p "$OUT"

export T_WITH_EP=1
source $S/runs/glm-isambard/U-uccl-send-abort/scripts/env-U.sh >/dev/null 2>&1 || true
export PYTHONPATH=$S/src/sglang-draft-train-0902/python:${PYTHONPATH:-}
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 SGLANG_JIT_DEEPGEMM_PRECOMPILE=false
export PYTHONDONTWRITEBYTECODE=1

setsid python -m sglang.launch_server --model-path "$MODEL" --load-format dummy \
  --trust-remote-code --json-model-override-args '{"qk_rope_head_dim": 64}' \
  --host 127.0.0.1 --port $PORT --tp-size 1 --random-seed 42 \
  --disable-cuda-graph \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-draft-model-path "$DRAFT_DIR" \
  --chunked-prefill-size 4096 --context-length 8192 --max-running-requests 4 \
  --max-total-tokens 16384 --mem-fraction-static 0.80 \
  > "$OUT/server.log" 2>&1 &
SRV=$!

for i in $(seq 1 96); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo SERVER-DIED; tail -40 "$OUT/server.log"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo HEALTH-TIMEOUT; tail -40 "$OUT/server.log"; exit 1; }
echo SERVER-READY

curl -s "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 48, "temperature": 0, "ignore_eos": true}}' \
  | head -c 600; echo
sleep 2
grep -m3 -i "accept" "$OUT/server.log" | tail -3 || true
echo "M3-LOAD-TEST-OUTPUT ABOVE (OUT=$OUT)"
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo DONE
