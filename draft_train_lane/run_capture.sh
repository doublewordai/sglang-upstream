#!/usr/bin/env bash
# draft-train: the 8-node CAPTURE job alone, ready to launch the moment a
# system is assigned. Runs on the persistent node:
#
#   HOLDER=<8-node jobid> [DURATION=] nohup bash run_capture.sh > logs/capture-job.out 2>&1 &
#
# Boots the capture-rig PD system (l3-launch-capture.sh: my worktree +
# SGLANG_DRAFT_CAPTURE_DIR env, ports 57000-57999, prod v17 topology), replays
# the measured pi session traffic with real corpus content, then tears down
# and prints capture stats. Output: capture/real/ (token+hidden windows).
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
DUR=${DURATION:-}   # empty = full session timeline (~4.5 h of traffic)
# CAP as argv (bulletproof vs env-chain weirdness): bash run_capture.sh <capdir>
CAP=${1:-${CAP:-$LANE/capture/real}}
cd "$LANE"

# single-instance lock: concurrent invocations clobber each other (shared
# boot log + capture dir). A second instance exits immediately.
exec 9>"$LANE/.capture-job.lock"
if ! flock -n 9; then
    echo "another capture job is running (lock held); exiting"
    exit 1
fi

echo "=== capture job: HOLDER=$HOLDER DURATION=${DUR:-full} start $(date -u +%FT%TZ) ==="
rm -rf "$CAP"; mkdir -p "$CAP"

# 1. boot the capture rig
SGLANG_DRAFT_CAPTURE_DIR=$CAP SGLANG_DRAFT_CAPTURE_TAG=prefill \
  HOLDER=$HOLDER SMOKE=${SMOKE:-0} SMOKE_NODE=${SMOKE_NODE:-} MODEL=${MODEL:-} NODES_OVERRIDE=${NODES_OVERRIDE:-} \
  nohup bash $LANE/l3-launch-capture.sh > $LOGS/capjob-boot.out 2>&1 &
BOOT=$!
for i in $(seq 1 300); do
  grep -q "PD disagg system ready" $LOGS/capjob-boot.out && break
  grep -q "boot failed" $LOGS/capjob-boot.out && { echo BOOT-FAILED; tail -30 $LOGS/capjob-boot.out; exit 1; }
  kill -0 $BOOT 2>/dev/null || { echo BOOT-DIED; tail -30 $LOGS/capjob-boot.out; exit 1; }
  sleep 10
done
grep -q "PD disagg system ready" $LOGS/capjob-boot.out || { echo BOOT-TIMEOUT; exit 1; }
echo SYSTEM-READY
DECODE_MASTER=${DECODE_MASTER:-$(scontrol show hostnames "$(squeue -j "$HOLDER" -h -o %N)" | sed -n 5p)}

# 2. replay
echo "=== replay start $(date -u +%T) ==="
DURFLAG=""; [ -n "$DUR" ] && DURFLAG="--duration $DUR"
srun --overlap --jobid="$HOLDER" -N1 -n1 -w "$DECODE_MASTER" --gres=gpu:0 \
  --cpus-per-task=8 --input=none bash -c "
    export T_WITH_EP=1; source $S/runs/glm-isambard/U-uccl-send-abort/scripts/env-U.sh >/dev/null 2>&1
    cd $LANE && PYTHONDONTWRITEBYTECODE=1 ~/sglang-venv/bin/python '$REPLAY' '$SESS' \
      --base-url http://127.0.0.1:57200 --model glm-5.3-fp8 \
      --concurrency 16 --gap-scale 1.0 --timeout 1800 $DURFLAG $REPLAY_EXTRA \
      --tokenizer '$TOK' --corpus '$CORPUS' --out '$CAP/requests.jsonl'
  " > $LOGS/capjob-replay.out 2>&1
echo "replay exit=$? at $(date -u +%T)"
tail -3 $LOGS/capjob-replay.out

# 3. flush + teardown (shards flush on rotate/teardown; give it a beat)
sleep 30
SRV=$(grep -m1 "srun launched (pid=" $LOGS/capjob-boot.out | sed "s/.*pid=\([0-9]*\).*/\1/")
LB=$(grep -m1 "LB router launched (pid=" $LOGS/capjob-boot.out | sed "s/.*pid=\([0-9]*\).*/\1/")
kill $SRV $LB $BOOT 2>/dev/null; sleep 90

# 4. stats
echo "=== capture stats ==="
~/sglang-venv/bin/python $LANE/draft_capture_reader.py stats "$CAP" | tee $LOGS/capjob-stats.txt
echo "=== capture job done $(date -u +%FT%TZ); next: eval_draft.py then run_realdata_pipeline.sh with SKIP_CAPTURE=1 ==="
