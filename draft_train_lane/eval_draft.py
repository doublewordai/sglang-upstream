#!/usr/bin/env python3
"""Pre-training baseline eval of the ORIGINAL (extracted, untrained) draft
weights on captured windows. Run on a GPU node:

  CUDA_VISIBLE_DEVICES=0 python eval_draft.py --data <capdir> --weights <draft_weights_dir> \
      [--max-windows 64] [--window 2048]

This is the decisive semantic check of capture + training-model equivalence:
if the original NextN layer scores a plausible top-1 on real traffic (EAGLE
drafts typically 40-70% top-1), the captured hiddens are the right tensors in
the right layout. A near-zero score means capture/model mismatch to debug
BEFORE training. Also reports top-4 and per-depth chain top-1 (chain_rollout)
if --chain is set.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft_data import RealData  # noqa: E402
from draft_model import DraftNextN, chain_loss, draft_loss  # noqa: E402


def build_model(weights_dir: str) -> DraftNextN:
    """Same construction as train_draft.build_model."""
    from safetensors.torch import load_file

    import train_draft  # reuse build_model verbatim
    return train_draft.build_model(weights_dir, "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="capture dir")
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "draft_weights"))
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--chain", action="store_true", help="also report per-depth chain top-1")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = build_model(args.weights).cuda().eval()
    # holdout_sessions=0: baseline eval over all data (train/val split is the
    # trainer's concern; the semantic check wants maximum coverage)
    data = RealData(args.data, window=args.window, holdout_sessions=0, seed=0)
    idx = data.train_idx[: args.max_windows]
    n = len(idx)
    print(f"windows: {n} (of {len(data.train_idx)}), window={args.window}")

    ce_sum = mse_sum = 0.0
    top1 = top4 = tok = 0.0
    chain_top1 = None
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for i, (rid, s) in enumerate(idx):
            t, p, a = data.get(rid, s, args.window)
            tokens = torch.from_numpy(t.astype(np.int64)).unsqueeze(0).cuda()
            prev = torch.from_numpy(np.asarray(p, dtype=np.float16)).unsqueeze(0).cuda()
            pos = (int(a) + torch.arange(args.window)).unsqueeze(0).cuda()
            _, m = draft_loss(model, tokens, prev, pos, 1.0, return_metrics=True)
            w = tokens.shape[1] - 1  # label positions in this window
            ce_sum += m["ce"] * w
            mse_sum += m["mse"] * w
            top1 += m["top1"] * w
            top4 += m["top4"] * w
            tok += w
            if args.chain and i < 8:
                _, cm = chain_loss(model, tokens[:1], prev[:1], pos[:1],
                                   return_metrics=True, chain_len=6, n_chains=4)
                if chain_top1 is None:
                    chain_top1 = [0.0] * 6
                for j in range(6):
                    chain_top1[j] += cm["top1_by_depth"][j] / 8
    out = {
        "windows": n,
        "label_positions": int(tok),
        "ce": ce_sum / max(tok, 1),
        "feature_mse": mse_sum / max(tok, 1),
        "top1": top1 / max(tok, 1),
        "top4": top4 / max(tok, 1),
    }
    if args.chain:
        out["chain_top1_by_depth"] = chain_top1
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
