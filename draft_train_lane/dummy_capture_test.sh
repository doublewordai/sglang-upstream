#!/usr/bin/env bash
# draft-train M1: 1-GPU dummy rig (dummy-glm53, 6 layers, TP1) + capture-hook test.
# Run INSIDE an srun step on nid010151 with one GPU in CUDA_VISIBLE_DEVICES, e.g.:
#   ssh isambard 'srun --overlap --jobid=6256423 -N1 -n1 -w nid010151 --gres=gpu:4 \
#     --cpus-per-task=16 --input=none bash -lc "CUDA_VISIBLE_DEVICES=1 bash \
#     /scratch/s6p/fergus.s6p/grace-1m/lanes/draft-train/dummy_capture_test.sh"'
# Env knobs: MODES (default extend,decode), GRAPHS (default --disable-cuda-graph),
#            PORT (default 57000)
set -uo pipefail
S=/scratch/s6p/fergus.s6p
LANE=$S/grace-1m/lanes/draft-train
MODEL=$S/grace-1m/dummy-glm53
MODES=${MODES:-extend,decode}
GRAPHS=${GRAPHS-"--disable-cuda-graph"}   # set GRAPHS= (empty) to enable graphs
PREFILL_CG=${PREFILL_CG:-0}               # 1 -> --disable-prefill-cuda-graph (real-rig shape)
PORT=${PORT:-57000}
CAPDIR=${CAPDIR:-$LANE/capture/dummy-$(date -u +%Y%m%d-%H%M%S)}

if ss -ltn | grep -q ":$PORT "; then echo "PORT $PORT IN USE"; exit 1; fi

export T_WITH_EP=1
source $S/runs/glm-isambard/U-uccl-send-abort/scripts/env-U.sh >/dev/null 2>&1 || true
export PYTHONPATH=$S/src/sglang-draft-train-0902/python:${PYTHONPATH:-}
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_DRAFT_CAPTURE_DIR=$CAPDIR
export SGLANG_DRAFT_CAPTURE_MODES=$MODES
export SGLANG_DRAFT_CAPTURE_TAG=dummy
rm -rf "$CAPDIR"; mkdir -p "$CAPDIR"
EXTRA=(); [[ "$PREFILL_CG" == "1" ]] && EXTRA+=(--disable-prefill-cuda-graph)
echo "CAPDIR=$CAPDIR MODES=$MODES GRAPHS=$GRAPHS PREFILL_CG=$PREFILL_CG"

setsid python -m sglang.launch_server --model-path "$MODEL" --load-format dummy \
  --trust-remote-code --json-model-override-args '{"qk_rope_head_dim": 64}' \
  --host 127.0.0.1 --port $PORT --tp-size 1 --random-seed 42 $GRAPHS \
  --chunked-prefill-size 2048 --context-length 8192 --max-running-requests 8 \
  --max-total-tokens 16384 --mem-fraction-static 0.55 "${EXTRA[@]}" \
  > "$CAPDIR/server.log" 2>&1 &
SRV=$!

for i in $(seq 1 72); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo SERVER-DIED; tail -40 "$CAPDIR/server.log"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo HEALTH-TIMEOUT; tail -40 "$CAPDIR/server.log"; exit 1; }
echo SERVER-READY

EXPECT_DEC=0; [[ "$MODES" == *decode* ]] && EXPECT_DEC=1
python "$LANE/dummy_capture_client.py" --port $PORT --out "$CAPDIR/sent.jsonl" --expect-decode $EXPECT_DEC 2>&1 | tee "$CAPDIR/client.log"
sleep 3   # let the writer thread flush

# validate BEFORE teardown (the server's self-SIGKILL cleanup can take the
# step down with it; capture files are already flushed by then)
python3 "$LANE/draft_capture_reader.py" stats "$CAPDIR" 2>&1 | tee "$CAPDIR/stats.log"
python3 "$LANE/draft_capture_reader.py" validate "$CAPDIR" --sent "$CAPDIR/sent.jsonl" 2>&1 | tee "$CAPDIR/validate.log"
RC=${PIPESTATUS[0]}

echo "stopping server..."
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; echo "server exit: $?"
sleep 2
echo "DONE RC=$RC CAPDIR=$CAPDIR"
