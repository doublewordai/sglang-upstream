#!/usr/bin/env python3
"""Eval of draft weights on captured windows. Run on a GPU node:

  CUDA_VISIBLE_DEVICES=0 python eval_draft.py --data <capdir> --weights <draft_weights_dir> \
      [--max-windows 64] [--window 2048] [--chain] [--per-segment] [--by-depth]

The decisive semantic check of capture + training-model equivalence: if the
original NextN layer scores a plausible top-1 on real traffic (EAGLE drafts
typically 40-70% top-1), the captured hiddens are the right tensors in the
right layout. Also reports top-4, per-depth chain top-1 (--chain), per-segment
top-1 (--per-segment), and — the lane's headline instrument — metrics bucketed
by CONTEXT DEPTH (--by-depth: the window's session position 0-50k / 50-100k /
100-165k / 165k+), which measures whether drafting accuracy falls with context
length (the OWL length-generalization question).

--attn-window/--attn-sink evaluate an arm with the draft attention window
(matches --speculative-draft-window-size/--speculative-draft-attn-sink at
serving); default 0/0 = the control (full/dense window, no sink).
"""
import argparse
import json
import os
import sys

TOK_DEFAULT = "/projects/s6p/hf/hub/models--zai-org--GLM-5.3/snapshots/e0b07fd2751b42d5efa199cc02c2b271deadc516/tokenizer.json"

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft_data import RealData  # noqa: E402
from draft_model import DraftNextN, chain_loss, draft_loss  # noqa: E402

DEPTH_BUCKETS = [(0, 50_000), (50_000, 100_000), (100_000, 165_000), (165_000, 1 << 40)]


def bucket_of(start_pos: int) -> int:
    for i, (lo, hi) in enumerate(DEPTH_BUCKETS):
        if lo <= start_pos < hi:
            return i
    return len(DEPTH_BUCKETS) - 1


def build_model(weights_dir: str, attn_window=0, attn_sink=0) -> DraftNextN:
    """Same construction as train_draft.build_model."""
    from safetensors.torch import load_file

    import train_draft  # reuse build_model verbatim
    return train_draft.build_model(weights_dir, "cpu", attn_window=attn_window,
                                   attn_sink=attn_sink)


def est_accept_d4(chain_top1):
    """Expected accepted length at D=4 from per-depth chain top-1:
    1 + p1 + p1*p2 + p1*p2*p3 (sequential verification)."""
    if chain_top1 is None:
        return None
    p = [1.0] + list(chain_top1)  # depth 0 = seed prediction
    return 1.0 + p[1] + p[1] * p[2] + p[1] * p[2] * p[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="capture dir")
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "draft_weights"))
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--chain", action="store_true", help="also report per-depth chain top-1")
    ap.add_argument("--chain-len", type=int, default=6)
    ap.add_argument("--per-segment", action="store_true",
                    help="top-1 by segment class (prose/code/tool-JSON) via marker segmentation")
    ap.add_argument("--by-depth", action="store_true",
                    help="top-1/top-4/chain metrics bucketed by context depth "
                         "(the accept-vs-context-length instrument)")
    ap.add_argument("--attn-window", type=int, default=0)
    ap.add_argument("--attn-sink", type=int, default=0)
    ap.add_argument("--tokenizer", default=TOK_DEFAULT)
    ap.add_argument("--ft-ckpt", help="overlay a draft_finetuned.pt onto the model")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = build_model(args.weights, args.attn_window, args.attn_sink).cuda().eval()
    if args.ft_ckpt:
        import train_draft
        sd = torch.load(args.ft_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # only actual trainable parameters must be present in the overlay;
        # frozen buffers (experts, kv_b, gate bias), embed and lm_head keep
        # their original values and may be absent
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        real_missing = [k for k in missing if k in trainable]
        assert not real_missing and not unexpected, (real_missing, unexpected)
        print(f"overlaid {args.ft_ckpt}")
    # holdout_sessions=0: eval over all data (train/val split is the trainer's
    # concern; the semantic check wants maximum coverage)
    data = RealData(args.data, window=args.window, holdout_sessions=0, seed=0,
                    attn_sink=args.attn_sink)
    idx = data.train_idx[: args.max_windows]
    n = len(idx)
    print(f"windows: {n} (of {len(data.train_idx)}), window={args.window}, "
          f"attn_window={args.attn_window}, attn_sink={args.attn_sink}")

    ce_sum = mse_sum = 0.0
    top1 = top4 = tok = 0.0
    chain_top1 = None
    by_depth = [
        {"top1": 0.0, "tok": 0.0, "chain": [0.0] * args.chain_len, "chains": 0}
        for _ in DEPTH_BUCKETS
    ]
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for i, (key, s) in enumerate(idx):
            t, p, a, ctx_len = data.get(key, s, args.window)
            start_pos = int(a[ctx_len])  # window start in session coords
            tokens = torch.from_numpy(t.astype(np.int64)).unsqueeze(0).cuda()
            prev = torch.from_numpy(np.asarray(p, dtype=np.float16)).unsqueeze(0).cuda()
            pos = torch.from_numpy(a.astype(np.int64)).unsqueeze(0).cuda()
            _, m = draft_loss(model, tokens, prev, pos, 1.0, return_metrics=True,
                              ctx_len=ctx_len)
            w = tokens.shape[1] - 1 - ctx_len  # supervised positions
            ce_sum += m["ce"] * w
            mse_sum += m["mse"] * w
            top1 += m["top1"] * w
            top4 += m["top4"] * w
            tok += w
            bd = by_depth[bucket_of(start_pos)]
            bd["top1"] += m["top1"] * w
            bd["tok"] += w
            if args.chain and i < 16:
                _, cm = chain_loss(model, tokens[:1], prev[:1], pos[:1],
                                   return_metrics=True, chain_len=args.chain_len,
                                   n_chains=4, ctx_len=ctx_len)
                if chain_top1 is None:
                    chain_top1 = [0.0] * args.chain_len
                for j in range(args.chain_len):
                    chain_top1[j] += cm["top1_by_depth"][j] / 16
                    bd["chain"][j] += cm["top1_by_depth"][j] * cm["chains"] / 4
                bd["chains"] += cm["chains"] / 4
    out = {
        "windows": n,
        "label_positions": int(tok),
        "ce": ce_sum / max(tok, 1),
        "feature_mse": mse_sum / max(tok, 1),
        "top1": top1 / max(tok, 1),
        "top4": top4 / max(tok, 1),
        "attn_window": args.attn_window,
        "attn_sink": args.attn_sink,
    }
    if args.chain:
        out["chain_top1_by_depth"] = chain_top1
        out["est_accept_d4"] = est_accept_d4(chain_top1)
    if args.by_depth:
        rows = []
        for (lo, hi), bd in zip(DEPTH_BUCKETS, by_depth):
            if bd["tok"] == 0:
                continue
            row = {
                "depth": f"{lo//1000}k-{hi//1000}k" if hi < (1 << 40) else f">{lo//1000}k",
                "windows_tok": int(bd["tok"]),
                "top1": bd["top1"] / bd["tok"],
            }
            if bd["chains"]:
                ctd = [c / bd["chains"] for c in bd["chain"]]
                row["chain_top1_by_depth"] = [round(x, 4) for x in ctd]
                row["est_accept_d4"] = round(est_accept_d4(ctd), 4)
            rows.append(row)
        out["by_depth"] = rows

    seg_stats = {}
    if args.per_segment:
        # per-position top-1 by segment class of the PREDICTED token
        from tokenizers import Tokenizer

        from segment_shares import FENCE, TOOL_JSON

        tk = Tokenizer.from_file(args.tokenizer)
        hits, total_by = {}, {}
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for key, s in idx:
                t, p, a, ctx_len = data.get(key, s, args.window)
                tokens = torch.from_numpy(t.astype(np.int64)).unsqueeze(0).cuda()
                prev = torch.from_numpy(np.asarray(p, dtype=np.float16)).unsqueeze(0).cuda()
                pos = torch.from_numpy(a.astype(np.int64)).unsqueeze(0).cuda()
                _, logits = model(tokens, prev, pos, compute_logits=True)
                # window positions are [ctx_len, n): their predictions are
                # logits[ctx_len:-1] vs labels t[ctx_len+1:]
                pred = logits[0, ctx_len:-1].argmax(-1).cpu()
                labels = tokens[0, ctx_len + 1 :].cpu()
                correct = (pred == labels)
                tw = t[ctx_len:]
                text = tk.decode(tw.tolist(), skip_special_tokens=False)
                enc = tk.encode(text)
                ids = enc.ids
                # two-pointer alignment: classify only positions where the
                # re-encoded ids match the original (handles small drift)
                ia = ib = 0
                agree = []
                while ia < len(tw) and ib < len(ids):
                    if int(tw[ia]) == ids[ib]:
                        agree.append((ia, ib))
                        ia += 1
                        ib += 1
                    elif ib + 1 < len(ids) and int(tw[ia]) == ids[ib + 1]:
                        ib += 1
                    else:
                        ia += 1
                if len(agree) < 0.8 * len(tw):
                    continue  # too much drift: skip window
                # class per char from non-overlapping priority spans
                ccls = bytearray(b"p") * len(text)  # p=prose
                spans = sorted(
                    [(m.start(), m.end(), "j") for m in TOOL_JSON.finditer(text)]
                    + [(m.start(), m.end(), "c") for m in FENCE.finditer(text)]
                )
                last = 0
                for st, en, k in spans:
                    if st < last:
                        continue
                    for ci in range(st, en):
                        ccls[ci] = ord(k)
                    last = en
                # token class from its offset midpoint (re-encoded index)
                tcls = []
                for st, en in enc.offsets:
                    mid = (st + en) // 2 if en > st else st
                    tcls.append(chr(ccls[mid]) if mid < len(text) else "p")
                for ia, ib in agree[1:]:  # class of the predicted token
                    k = tcls[ib]
                    hits[k] = hits.get(k, 0) + int(correct[ia - 1])
                    total_by[k] = total_by.get(k, 0) + 1
        seg_stats = {
            "per_segment_top1": {
                k: hits[k] / total_by[k] for k in sorted(total_by) if total_by[k]
            },
            "per_segment_tokens": dict(total_by),
        }
        out.update(seg_stats)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
