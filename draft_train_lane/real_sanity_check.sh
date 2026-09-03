#!/usr/bin/env bash
# REAL-WEIGHTS HARNESS SANITY CHECK (supervisor gate: original draft head must
# score >= 0.7 top-1 on real GLM-5.3 hiddens before any training/scaling claims).
# Boots the REAL GLM-5.3 (tp2, GPUs 1-2 of nid010151, eager, port 57000) with
# the capture hook, sends real corpus prompts, tears down, then evals the
# ORIGINAL extracted draft weights on the captured windows.
set -uo pipefail
S=/scratch/s6p/fergus.s6p
LANE=$S/grace-1m/lanes/draft-train
MODEL=/projects/s6p/hf/hub/models--zai-org--GLM-5.3/snapshots/e0b07fd2751b42d5efa199cc02c2b271deadc516
CAP=$LANE/capture/real-sanity
PORT=57000
OUT=$LANE/logs/real-sanity-$(date -u +%H%M%S)
mkdir -p "$OUT" "$CAP"
rm -f "$CAP"/*.bin "$CAP"/*.json 2>/dev/null

export T_WITH_EP=1
source $S/runs/glm-isambard/U-uccl-send-abort/scripts/env-U.sh >/dev/null 2>&1
export PYTHONPATH=$S/src/sglang-draft-train-0902/python:${PYTHONPATH:-}
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 SGLANG_JIT_DEEPGEMM_PRECOMPILE=false
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_DRAFT_CAPTURE_DIR=$CAP SGLANG_DRAFT_CAPTURE_TAG=real

setsid python -m sglang.launch_server --model-path "$MODEL" \
  --trust-remote-code --served-model-name glm-5.3-fp8 \
  --host 127.0.0.1 --port $PORT --tp-size 4 --random-seed 42 \
  --disable-cuda-graph \
  --chunked-prefill-size 8192 --context-length 32768 --max-running-requests 4 \
  --mem-fraction-static 0.85 \
  > "$OUT/server.log" 2>&1 &
SRV=$!
echo "server pid=$SRV; waiting (weight load ~10-20 min)"
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo SERVER-DIED; tail -40 "$OUT/server.log"; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo HEALTH-TIMEOUT; tail -40 "$OUT/server.log"; exit 1; }
echo SERVER-READY

# send real corpus prompts (sequential, distinct content)
python - "$LANE" "$PORT" <<'PYEOF'
import json, sys, urllib.request

lane, port = sys.argv[1], sys.argv[2]
text = open(f"{lane}/../workload/out/corpus_pi.txt", encoding="utf-8", errors="replace").read()
# 8 prompts of ~3-5k chars from different parts of the corpus
for k in range(8):
    chunk = text[k * 120000 : k * 120000 + 4000]
    body = json.dumps({
        "text": chunk,
        "sampling_params": {"max_new_tokens": 32, "temperature": 0, "ignore_eos": True},
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    print(f"prompt {k}: ok, gen {len(d.get('output_ids', []))} tokens", flush=True)
PYEOF
echo PROMPTS-DONE
sleep 20

# teardown (flushes capture shards)
kill $SRV 2>/dev/null; sleep 45
pkill -f "launch_server.*$PORT" 2>/dev/null; sleep 10

# stats + THE GATE: original draft weights on the real capture
~/sglang-venv/bin/python $LANE/draft_capture_reader.py stats "$CAP" | tee "$OUT/stats.txt"
CUDA_VISIBLE_DEVICES=3 ~/sglang-venv/bin/python $LANE/eval_draft.py \
  --data "$CAP" --max-windows 32 --window 1024 --chain --per-segment \
  --weights $LANE/draft_weights 2>&1 | tee "$OUT/eval.txt"
echo "SANITY-DONE (gate: top1 >= 0.7 on original weights; OUT=$OUT)"
