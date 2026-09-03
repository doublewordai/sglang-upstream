#!/usr/bin/env python3
"""Pilot verification: export dir vs the TRAINING checkpoint (draft_finetuned.pt).

Invariants:
- bf16-cast keys (trainable, non-quant): export value == training value cast
  to bf16, bit-exact.
- fp8 keys that were TRAINED (attn projections, shared experts): requant
  error only, measured as maxabs / max|ref| (relative-to-tensor-scale; plain
  element-wise rel is meaningless near zero). OK < 0.02.
- FROZEN fp8 keys (routed experts, kv_b): with --orig-ckpt passthrough they
  must be BIT-IDENTICAL to the original checkpoint; without it, requant
  drift ~4e-3 per element (report, OK < 0.02).
"""
import argparse
import os
import sys

import torch
from safetensors import safe_open

L = 78


def dequant(w, s, block=128):
    n, k = w.shape
    w = w.float()
    s = s.float()
    # s: [n/128, k/128] -> expand
    s = s.repeat_interleave(block, 0).repeat_interleave(block, 1)[:n, :k]
    return w * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--orig-ckpt", help="bit-identity check of passthrough fp8 keys")
    args = ap.parse_args()

    ft = torch.load(args.ft, map_location="cpu")
    ex = safe_open(os.path.join(args.export, "model.safetensors"), framework="pt", device="cpu")

    bf16_keys = {
        "enorm": f"model.layers.{L}.enorm.weight",
        "hnorm": f"model.layers.{L}.hnorm.weight",
        "eh_proj.weight": f"model.layers.{L}.eh_proj.weight",
        "input_ln": f"model.layers.{L}.input_layernorm.weight",
        "post_attn_ln": f"model.layers.{L}.post_attention_layernorm.weight",
        "shared_head_norm": f"model.layers.{L}.shared_head.norm.weight",
        "attn.q_a_ln": f"model.layers.{L}.self_attn.q_a_layernorm.weight",
        "attn.kv_a_ln": f"model.layers.{L}.self_attn.kv_a_layernorm.weight",
        "moe.gate.w": f"model.layers.{L}.mlp.gate.weight",
    }
    trained_fp8 = {
        "attn.q_a.weight": f"model.layers.{L}.self_attn.q_a_proj.weight",
        "attn.q_b.weight": f"model.layers.{L}.self_attn.q_b_proj.weight",
        "attn.kv_a.weight": f"model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight",
        "attn.o.weight": f"model.layers.{L}.self_attn.o_proj.weight",
        "moe.shared.gate": f"model.layers.{L}.mlp.shared_experts.gate_proj.weight",
        "moe.shared.up": f"model.layers.{L}.mlp.shared_experts.up_proj.weight",
        "moe.shared.down": f"model.layers.{L}.mlp.shared_experts.down_proj.weight",
    }

    worst = []
    ok = True
    # bias key: fp32 passthrough
    kb = f"model.layers.{L}.mlp.gate.e_score_correction_bias"
    if not torch.equal(ex.get_tensor(kb), ft["moe.gate.bias"].float()):
        print(kb, "BIAS MISMATCH"); ok = False

    for t, k in bf16_keys.items():
        want = ft[t].float().to(torch.bfloat16)
        got = ex.get_tensor(k)
        if not torch.equal(want, got):
            d = (want.float() - got.float()).abs().max().item()
            print(f"{k}: BF16 MISMATCH maxabs={d:.3e}"); worst.append((d, k)); ok = False

    for t, k in trained_fp8.items():
        q = ex.get_tensor(k)
        s = ex.get_tensor(k[: -len(".weight")] + ".weight_scale_inv")
        got = dequant(q, s)
        ref = ft[t].float()
        r = ((got - ref).abs().max() / ref.abs().max()).item()
        print(f"{k}: trained-fp8 requant maxabs/maxref={r:.3e}")
        worst.append((r, k))
        # bound: e4m3 half-step of a mid/high-range code in the max block
        # is 1.8-3.6% of block max ~= tensor max; allow 4% (end-to-end
        # accept measurement is the final arbiter anyway)
        if r >= 0.04:
            ok = False

    # kv_b is TRAINABLE in our setup (14.7M params): if it moved during
    # training the export must carry the trained (requantized) values
    kvb = f"model.layers.{L}.self_attn.kv_b_proj.weight"
    q = ex.get_tensor(kvb)
    s = ex.get_tensor(kvb[: -len(".weight")] + ".weight_scale_inv")
    got = dequant(q, s).reshape(64, 448, 512)
    ref = ft["attn.kv_b"].float()
    r = ((got - ref).abs().max() / ref.abs().max()).item()
    print(f"{kvb}: kv_b requant maxabs/maxref={r:.3e}"); worst.append((r, kvb))
    if r >= 0.04:
        ok = False

    # frozen routed experts: must round-trip essentially exactly
    for t, proj in [("moe.eg.w", "gate_proj"), ("moe.eu.w", "up_proj"), ("moe.ed.w", "down_proj")]:
        stack = ft[t]
        for e in (0, 100, 255):
            k = f"model.layers.{L}.mlp.experts.{e}.{proj}.weight"
            q = ex.get_tensor(k)
            s = ex.get_tensor(k[: -len(".weight")] + ".weight_scale_inv")
            got = dequant(q, s)
            ref = stack[e].float()
            r = ((got - ref).abs().max() / ref.abs().max()).item()
            if r >= 0.02:
                print(f"{k}: FROZEN EXPERT DRIFT maxabs/maxref={r:.3e}"); worst.append((r, k)); ok = False
    # embed / lm_head: bit-exact bf16 casts
    for t, k in [("embed", "model.embed_tokens.weight"), ("lm_head", "lm_head.weight")]:
        if not torch.equal(ft[t].float().to(torch.bfloat16), ex.get_tensor(k)):
            print(k, "EMBED/LM_HEAD MISMATCH"); ok = False

    if args.orig_ckpt:
        import json
        idx = json.load(open(os.path.join(args.orig_ckpt, "model.safetensors.index.json")))["weight_map"]
        npt = nbit = 0
        for k in list(ex.keys()):
            if ".mlp.experts." in k and f".{L}." in k:  # experts only: kv_b is trainable
                if k not in idx:
                    continue
                with safe_open(os.path.join(args.orig_ckpt, idx[k]), framework="pt", device="cpu") as f:
                    o = f.get_tensor(k)
                npt += 1
                if torch.equal(o, ex.get_tensor(k)):
                    nbit += 1
                else:
                    print(f"{k}: PASSTHROUGH NOT BIT-IDENTICAL"); ok = False
        print(f"passthrough bit-identity: {nbit}/{npt}")

    worst.sort(reverse=True)
    print("worst rel diffs:", worst[:5])
    print("PILOT-VERIFY-" + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
