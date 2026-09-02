#!/usr/bin/env bash
# draft-train M4: accept-length A/B, old vs new draft x depth 3..6, corpus replay.
# Runs ON the persistent node (nohup). One boot + 15-min replay per arm.
#
#   HOLDER=<jobid> [ARMS="old:3 old:4 old:5 old:6 new:3 new:4 new:5 new:6"] \
#   [DURATION=900] nohup bash run_m4_ab.sh > logs/m4-ab.out 2>&1 &
#
# old  = checkpoint NextN (no --speculative-draft-model-path)
# new  = $LANE/export_trained (the fine-tuned draft)
set -uo pipefail
S=/scratch/s6p/fergus.s6p
LANE=$S/grace-1m/lanes/draft-train
HOLDER=${HOLDER:?need HOLDER}
ARMS=${ARMS:-"old:3 old:4 old:5 old:6 new:3 new:4 new:5 new:6"}
DURATION=${DURATION:-900}
REPLAY=$S/pd-serve/bench/replay_client_ss_v2.py
SESS=$S/grace-1m/lanes/workload/out/sessions_pi_measured.jsonl
CORPUS=$S/grace-1m/lanes/workload/out/corpus_pi.txt
TOK=/projects/s6p/hf/hub/models--zai-org--GLM-5.3/snapshots/e0b07fd2751b42d5efa199cc02c2b271deadc516/tokenizer.json
OUT=$LANE/m4
mkdir -p "$OUT"

NODES=$(scontrol show hostnames "$(squeue -j "$HOLDER" -h -o %N)" | head -8 | paste -sd,)
DECODE_MASTER=$(scontrol show hostnames "$(squeue -j "$HOLDER" -h -o %N)" | sed -n 5p)
echo "nodes=$NODES decode_master=$DECODE_MASTER arms=$ARMS duration=$DURATION"

run_arm() {
  local draft=$1 depth=$2
  local tag="$draft-$depth"
  local flags="--speculative-algorithm EAGLE --speculative-num-steps $depth --speculative-eagle-topk 1 --speculative-num-draft-tokens $((depth+1))"
  if [ "$draft" = "new" ]; then
    flags="$flags --speculative-draft-model-path $LANE/export_trained"
  fi
  echo "=== ARM $tag: flags=$flags ==="
  local bootlog=$OUT/boot-$tag.out
  SPEC_FLAGS="$flags" HOLDER="$HOLDER" HISPARSE_BUF=8192 MK_TMAX=256 \
    nohup bash $LANE/l3-launch-capture.sh > "$bootlog" 2>&1 &
  local bootpid=$!

  # wait for the system-ready banner (weight load + JIT: up to 50 min)
  local ready=0
  for i in $(seq 1 300); do
    grep -q "PD disagg system ready" "$bootlog" && { ready=1; break; }
    grep -q "boot failed\|NOT ready" "$bootlog" && break
    kill -0 $bootpid 2>/dev/null || break
    sleep 10
  done
  if [ "$ready" != 1 ]; then
    echo "ARM $tag BOOT-FAILED"; tail -30 "$bootlog"
    kill $bootpid 2>/dev/null
    return 1
  fi
  local engine_log
  engine_log=$(grep -m1 "^log=" "$bootlog" | cut -d= -f2)
  echo "ARM $tag READY engine_log=$engine_log"

  # replay (15 min steady + drain) on the decode master
  srun --overlap --jobid="$HOLDER" -N1 -n1 -w "$DECODE_MASTER" --gres=gpu:0 \
    --cpus-per-task=8 --input=none bash -c "
      export T_WITH_EP=1; source $S/runs/glm-isambard/U-uccl-send-abort/scripts/env-U.sh >/dev/null 2>&1
      python3 '$REPLAY' '$SESS' --base-url http://127.0.0.1:57200 --model glm-5.3-fp8 \
        --concurrency 16 --gap-scale 1.0 --duration $DURATION --timeout 1800 \
        --tokenizer '$TOK' --corpus '$CORPUS' --out '$OUT/requests-$tag.jsonl'
    " > "$OUT/replay-$tag.out" 2>&1
  echo "ARM $tag replay exit=$?"

  # accept length from the decode-arm log lines (skip the first 120 s ramp)
  if [ -n "$engine_log" ] && [ -f "$engine_log" ]; then
    python3 - "$engine_log" "$tag" <<'PYEOF'
import json, re, sys
log, tag = sys.argv[1], sys.argv[2]
rows = []
t0 = None
for line in open(log, errors="replace"):
    m = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*accept len: ([\d.]+)", line)
    if m:
        t = m.group(1)
        if t0 is None:
            t0 = t
        rows.append((t, float(m.group(2))))
import datetime
def p(ts):
    return datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
if rows:
    start = p(rows[0][0])
    steady = [v for ts, v in rows if (p(ts) - start).total_seconds() >= 120]
    vals = steady or [v for _, v in rows]
    vals.sort()
    out = {
        "tag": tag, "n": len(vals),
        "accept_len_mean": sum(vals) / len(vals),
        "accept_len_p50": vals[len(vals)//2],
        "accept_len_max": vals[-1],
    }
    print(json.dumps(out))
    with open(f"/scratch/s6p/fergus.s6p/grace-1m/lanes/draft-train/m4/accept-{tag}.json", "w") as f:
        json.dump(out, f)
else:
    print("NO ACCEPT LINES")
PYEOF
  fi

  # teardown: kill the boot srun + LB srun (pids echoed by the boot script)
  local srv lb
  srv=$(grep -m1 "srun launched (pid=" "$bootlog" | sed "s/.*pid=\([0-9]*\).*/\1/")
  lb=$(grep -m1 "LB router launched (pid=" "$bootlog" | sed "s/.*pid=\([0-9]*\).*/\1/")
  kill $srv $lb $bootpid 2>/dev/null
  sleep 30
  echo "ARM $tag DONE"
}

for arm in $ARMS; do
  draft=${arm%%:*}; depth=${arm##*:}
  run_arm "$draft" "$depth"
done
echo "M4-AB-ALL-DONE"
