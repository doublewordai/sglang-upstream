#!/usr/bin/env bash
# draft-train: the full real-data pipeline, sequenced. Each phase is idempotent
# and logged; run ON the persistent node inside `nohup`:
#
#   HOLDER=<8-node jobid> nohup bash run_realdata_pipeline.sh > logs/pipeline.out 2>&1 &
#
# Phases: boot-capture -> replay(full) -> validate -> train(real) -> export ->
#         m4 A/B (old vs new, depth 3..6). Set SKIP_CAPTURE=1 to start from an
#         existing capture dir, SKIP_AB=1 to stop before the A/B.
set -uo pipefail
S=/scratch/s6p/fergus.s6p
LANE=$S/grace-1m/lanes/draft-train
HOLDER=${HOLDER:?need HOLDER (8-node system)}
CAP=$LANE/capture/real
LOGS=$LANE/logs
REPLAY=$S/pd-serve/bench/replay_client_ss_v2.py
SESS=$S/grace-1m/lanes/workload/out/sessions_pi_measured.jsonl
CORPUS=$S/grace-1m/lanes/workload/out/corpus_pi.txt
TOK=/projects/s6p/hf/hub/models--zai-org--GLM-5.3/snapshots/e0b07fd2751b42d5efa199cc02c2b271deadc516/tokenizer.json
DUR=${DURATION:-}          # empty = full session timeline
cd "$LANE"
phase() { echo; echo "===== [$(date -u +%H:%M:%S)] PHASE: $1 ====="; }

# ---------- 1. boot capture system ----------
if [ "${SKIP_CAPTURE:-0}" != 1 ]; then
  phase "boot-capture"
  rm -rf "$CAP"; mkdir -p "$CAP"
  SGLANG_DRAFT_CAPTURE_DIR=$CAP SGLANG_DRAFT_CAPTURE_TAG=prefill \
    HOLDER=$HOLDER nohup bash $LANE/l3-launch-capture.sh > $LOGS/pipe-boot.out 2>&1 &
  BOOT=$!
  for i in $(seq 1 300); do
    grep -q "PD disagg system ready" $LOGS/pipe-boot.out && break
    grep -q "boot failed" $LOGS/pipe-boot.out && { echo BOOT-FAILED; exit 1; }
    kill -0 $BOOT 2>/dev/null || { echo BOOT-DIED; exit 1; }
    sleep 10
  done
  grep -q "PD disagg system ready" $LOGS/pipe-boot.out || { echo BOOT-TIMEOUT; exit 1; }
  echo SYSTEM-READY
  ELOG=$(grep -m1 "^log=" $LOGS/pipe-boot.out | cut -d= -f2)
  DECODE_MASTER=$(scontrol show hostnames "$(squeue -j "$HOLDER" -h -o %N)" | sed -n 5p)

  # ---------- 2. replay ----------
  phase "replay (DUR=${DUR:-full})"
  DURFLAG=""; [ -n "$DUR" ] && DURFLAG="--duration $DUR"
  srun --overlap --jobid="$HOLDER" -N1 -n1 -w "$DECODE_MASTER" --gres=gpu:0 \
    --cpus-per-task=8 --input=none bash -c "
      export T_WITH_EP=1; source $S/runs/glm-isambard/U-uccl-send-abort/scripts/env-U.sh >/dev/null 2>&1
      cd $LANE && PYTHONDONTWRITEBYTECODE=1 ~/sglang-venv/bin/python '$REPLAY' '$SESS' \
        --base-url http://127.0.0.1:57200 --model glm-5.3-fp8 \
        --concurrency 16 --gap-scale 1.0 --timeout 1800 $DURFLAG \
        --tokenizer '$TOK' --corpus '$CORPUS' --out '$LANE/capture_replay_requests.jsonl'
    " > $LOGS/pipe-replay.out 2>&1 || echo "REPLAY-EXIT-NONZERO (check pipe-replay.out)"

  # ---------- 3. teardown + validate ----------
  phase "teardown + validate capture"
  SRV=$(grep -m1 "srun launched (pid=" $LOGS/pipe-boot.out | sed "s/.*pid=\([0-9]*\).*/\1/")
  LB=$(grep -m1 "LB router launched (pid=" $LOGS/pipe-boot.out | sed "s/.*pid=\([0-9]*\).*/\1/")
  kill $SRV $LB $BOOT 2>/dev/null; sleep 60
  ~/sglang-venv/bin/python $LANE/draft_capture_reader.py stats "$CAP" | tee $LOGS/pipe-capstats.txt
  # decode-validate needs --sent + decode records; capture is extend-only by design

  # ---------- 3.5 pre-training baseline eval (decisive semantic check) ----------
  phase "baseline eval (semantic check)"
  srun --overlap --jobid="$HOLDER" -N1 -n1 -w "$DECODE_MASTER" --gres=gpu:4 \
    --cpus-per-task=16 --input=none bash -c "
      export T_WITH_EP=1; source $S/runs/glm-isambard/U-uccl-send-abort/scripts/env-U.sh >/dev/null 2>&1
      export PYTHONDONTWRITEBYTECODE=1; cd $LANE
      CUDA_VISIBLE_DEVICES=0 ~/sglang-venv/bin/python $LANE/eval_draft.py \
        --data '$CAP' --weights $LANE/draft_weights --max-windows 64 --window 2048 --chain
    " > $LOGS/pipe-baseeval.out 2>&1 || echo "BASEEVAL-FAILED (see pipe-baseeval.out)"
  tail -20 $LOGS/pipe-baseeval.out
fi

# ---------- 4. train on real data (GPU step: needs the holder) ----------
phase "train (real data)"
NGPU=${NGPU:-3}; PORT=${TPORT:-57100}; STEPS=${STEPS:-400}
TRAIN_NODE=${TRAIN_NODE:-$(scontrol show hostnames "$(squeue -j "$HOLDER" -h -o %N)" | sed -n 5p)}
echo "train node: $TRAIN_NODE"
srun --overlap --jobid="$HOLDER" -N1 -n1 -w "$TRAIN_NODE" --gres=gpu:4 \
  --cpus-per-task=48 --input=none bash -c "
    export T_WITH_EP=1; source $S/runs/glm-isambard/U-uccl-send-abort/scripts/env-U.sh >/dev/null 2>&1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
    cd $LANE
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3} ~/sglang-venv/bin/torchrun \
      --nproc-per-node=$NGPU --master-port $PORT \
      $LANE/train_draft.py --data '$CAP' --steps $STEPS \
      --window 2048 --micro-bs 4 --strategy fsdp \
      ${CHAIN_WEIGHT:+--chain-weight $CHAIN_WEIGHT} \
      --out $LANE/runs/real
  " > $LOGS/pipe-train.out 2>&1 \
  || { echo TRAIN-FAILED; tail -30 $LOGS/pipe-train.out; exit 1; }
tail -3 $LOGS/pipe-train.out

# ---------- 5. export (GPU step for fast fp8 dequant/requant) ----------
phase "export"
srun --overlap --jobid="$HOLDER" -N1 -n1 -w "$TRAIN_NODE" --gres=gpu:4 \
  --cpus-per-task=16 --input=none bash -c "
    export T_WITH_EP=1; source $S/runs/glm-isambard/U-uccl-send-abort/scripts/env-U.sh >/dev/null 2>&1
    export PYTHONDONTWRITEBYTECODE=1; cd $LANE
    CUDA_VISIBLE_DEVICES=0 ~/sglang-venv/bin/python $LANE/export_draft.py \
      --weights-dir $LANE/draft_weights --ft $LANE/runs/real/draft_finetuned.pt \
      --orig-ckpt /projects/s6p/hf/hub/models--zai-org--GLM-5.3/snapshots/e0b07fd2751b42d5efa199cc02c2b271deadc516 \
      --out $LANE/export_trained_real
  " > $LOGS/pipe-export.out 2>&1 || { echo EXPORT-FAILED; tail -20 $LOGS/pipe-export.out; exit 1; }
tail -2 $LOGS/pipe-export.out

# ---------- 6. M4 A/B ----------
if [ "${SKIP_AB:-0}" != 1 ]; then
  phase "M4 A/B"
  # run_m4_ab uses $LANE/export_trained as the new draft; point it at the real one
  ln -sfn $LANE/export_trained_real $LANE/export_trained
  HOLDER=$HOLDER ARMS="old:3 old:4 old:5 old:6 new:3 new:4 new:5 new:6" \
    bash $LANE/run_m4_ab.sh > $LOGS/pipe-ab.out 2>&1
  ~/sglang-venv/bin/python $LANE/m4_summary.py | tee $LOGS/pipe-ab-summary.txt
fi
phase "PIPELINE-DONE"
