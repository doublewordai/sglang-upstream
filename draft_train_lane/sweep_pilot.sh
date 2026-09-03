#!/usr/bin/env bash
# draft-train pilot sweep: lr x window x steps(epochs) on the dummy capture.
# Single GPU, strategy=none, trainable-only checkpoints (~1 GB each, deleted
# after eval). Appends one JSON row per config to sweep_results.jsonl.
# Run detached:  nohup bash sweep_pilot.sh > logs/sweep.out 2>&1 &
set -uo pipefail
S=/scratch/s6p/fergus.s6p
LANE=$S/grace-1m/lanes/draft-train
DATA=$LANE/capture/dummy-20260902-195434
OUT=$LANE/runs/sweep
PY=~/sglang-venv/bin/python
cd "$LANE"
mkdir -p "$OUT"

LRS="1e-5 5e-5 2e-4"
WINS="256 512 1024"
STEPS="25 100 400"

for lr in $LRS; do
for w in $WINS; do
for st in $STEPS; do
  tag="lr${lr}-w${w}-s${st}"
  run=$OUT/$tag
  rm -rf "$run"; mkdir -p "$run"
  echo "=== $tag ==="
  CUDA_VISIBLE_DEVICES=1 ~/sglang-venv/bin/torchrun --nproc-per-node=1 --master-port 57101 \
    $LANE/train_draft.py --data "$DATA" --steps "$st" \
    --window "$w" --micro-bs 2 --holdout 0 --strategy none --lr "$lr" \
    --save-trainable-only --out "$run" > "$run/train.log" 2>&1
  if [ $? -ne 0 ]; then echo "$tag TRAIN-FAIL" >> sweep_results.jsonl.incomplete; continue; fi
  CUDA_VISIBLE_DEVICES=1 timeout 900 $PY eval_draft.py --data "$DATA" \
    --max-windows 64 --window "$w" --chain --ft-ckpt "$run/draft_finetuned.pt" \
    > "$run/eval.json" 2>"$run/eval.err" || { echo "$tag EVAL-FAIL" >> sweep_results.jsonl.incomplete; continue; }
  PYTHONDONTWRITEBYTECODE=1 $PY - "$tag" "$run/eval.json" >> sweep_results.jsonl <<'PYEOF'
import json, sys
tag, path = sys.argv[1], sys.argv[2]
txt = open(path).read()
j = json.loads(txt[txt.index("{"):])
d = j.get("chain_top1_by_depth") or []
# estimated accept length for a depth-4 draft (topk-1):
# E = 1 + p1 + p1p2 + p1p2p3
e4 = None
if len(d) >= 4:
    e4 = 1.0
    acc = 1.0
    for p in d[1:4]:
        acc *= p
        e4 += acc
row = {"tag": tag, "ce": round(j["ce"], 4), "top1": round(j["top1"], 4),
       "top4": round(j["top4"], 4), "feature_mse": round(j["feature_mse"], 4),
       "chain_top1_by_depth": [round(x, 4) for x in d],
       "est_accept_len_d4": round(e4, 4) if e4 else None,
       "label_positions": j.get("label_positions")}
print(json.dumps(row))
PYEOF
  rm -rf "$run"
done
done
done
echo "SWEEP-DONE $(date -u +%T)"
