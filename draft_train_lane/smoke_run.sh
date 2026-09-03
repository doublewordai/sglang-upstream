#!/usr/bin/env bash
# Clean smoke relaunch of the capture job (no long inline ssh commands):
# 1-GPU dummy78 dry-run through run_capture.sh + SMOKE launcher mode.
# Usage (on the persistent node):  nohup bash smoke_run.sh > logs/smoke-capture.out 2>&1 &
set -u
S=/scratch/s6p/fergus.s6p
LANE=$S/grace-1m/lanes/draft-train
cd "$LANE"
rm -rf "$LANE/capture/smoke"
export SMOKE=1
export SMOKE_NODE=nid010159
export MODEL=$S/grace-1m/dummy78
export CAP=$LANE/capture/smoke
export DECODE_MASTER=nid010159
export DURATION=90
export REPLAY_EXTRA="--max-steps 2 --max-output 64"
export HOLDER=6255852
echo "smoke_run: CAP=$CAP SMOKE_NODE=$SMOKE_NODE MODEL=$MODEL HOLDER=$HOLDER"
exec bash "$LANE/run_capture.sh" "$CAP"
