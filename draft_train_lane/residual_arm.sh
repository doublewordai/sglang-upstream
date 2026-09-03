#!/usr/bin/env bash
# Residual-objective arm on the sweep's best config (lr2e-4 w256 s400).
set -uo pipefail
S=/scratch/s6p/fergus.s6p
LANE=$S/grace-1m/lanes/draft-train
DATA=$LANE/capture/dummy-20260902-195434
cd "$LANE"
run_one() {
  local tag=$1; shift
  local run=$LANE/runs/obj-$tag
  rm -rf "$run"; mkdir -p "$run"
  echo "=== $tag: $* ==="
  CUDA_VISIBLE_DEVICES=1 ~/sglang-venv/bin/torchrun --nproc-per-node=1 --master-port 57101 \
    $LANE/train_draft.py --data "$DATA" --steps 400 --window 256 --micro-bs 2 \
    --holdout 0 --strategy none --lr 2e-4 --save-trainable-only --out "$run" "$@" > "$run/train.log" 2>&1 || { echo "$tag TRAIN-FAIL"; return 1; }
  CUDA_VISIBLE_DEVICES=1 timeout 900 ~/sglang-venv/bin/python eval_draft.py --data "$DATA" \
    --max-windows 64 --window 256 --chain --ft-ckpt "$run/draft_finetuned.pt" \
    > "$run/eval.json" 2>"$run/eval.err" || { echo "$tag EVAL-FAIL"; tail -3 "$run/eval.err"; return 1; }
  PYTHONDONTWRITEBYTECODE=1 ~/sglang-venv/bin/python - "$tag" "$run/eval.json" >> obj_results.jsonl <<'PYEOF'
import json, sys
tag, path = sys.argv[1], sys.argv[2]
txt = open(path).read()
j = json.loads(txt[txt.index("{"):])
d = j.get("chain_top1_by_depth") or []
e4 = None
if len(d) >= 4:
    e4, acc = 1.0, 1.0
    for p in d[1:4]:
        acc *= p
        e4 += acc
print(json.dumps({"tag": tag, "ce": round(j["ce"], 4), "top1": round(j["top1"], 4),
                  "feature_mse": round(j["feature_mse"], 4),
                  "chain_top1_by_depth": [round(x, 4) for x in d],
                  "est_accept_len_d4": round(e4, 4) if e4 else None}))
PYEOF
  rm -rf "$run"
}
run_one chain        --chain-weight 0.5 --chain-len 6 --chains-per-window 2
run_one chain-res05  --chain-weight 0.5 --chain-residual 0.5 --chain-len 6 --chains-per-window 2
run_one chain-res10  --chain-weight 0.5 --chain-residual 1.0 --chain-len 6 --chains-per-window 2
echo OBJ-ARMS-DONE
