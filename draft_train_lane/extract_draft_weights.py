#!/usr/bin/env python3
"""Extract the GLM-5.3 NextN (layer 78) + embed/lm_head into training-ready bf16 safetensors.

Run on Isambard (CPU work, inside a srun step with --gres=gpu:0):
  python3 extract_draft_weights.py --ckpt /projects/s6p/hf/hub/models--zai-org--GLM-5.3/snapshots/e0b07fd... \
      --out /scratch/s6p/fergus.s6p/grace-1m/lanes/draft-train/draft_weights

fp8 block-quantized tensors (weight + weight_scale_inv, 128x128 blocks) are dequantized
to bf16 (w_fp8.float() * scale, per sglang block_quant_dequant). meta.json records every
original key with its dtype/format so the exporter can re-quantize the same keys.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors import safe_open

CKPT = "/projects/s6p/hf/hub/models--zai-org--GLM-5.3/snapshots/e0b07fd2751b42d5efa199cc02c2b271deadc516"
L = 78


def scale_key(key: str) -> str:
    return key[: -len(".weight")] + ".weight_scale_inv"


def load_ckpt_tensor(ckpt, key):
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
    if key not in idx["weight_map"]:
        raise KeyError(key)
    path = os.path.join(ckpt, idx["weight_map"][key])
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def dequant(w, s, block=128):
    """w: fp8 [N, K]; s: [N/128, K/128] -> bf16"""
    n, k = w.shape
    srep = (
        s.repeat_interleave(block, dim=-2).repeat_interleave(block, dim=-1)[..., :n, :k]
    )
    return (w.to(torch.float32) * srep).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta = {"layer": L, "orig": {}}

    def note(name, dtype, quant):
        meta["orig"][name] = {"dtype": dtype, "quant": quant}

    out = {}

    idx = json.load(open(os.path.join(args.ckpt, "model.safetensors.index.json")))["weight_map"]

    def put(name, key, quant=True):
        w = load_ckpt_tensor(args.ckpt, key)
        if quant and scale_key(key) in idx:
            s = load_ckpt_tensor(args.ckpt, scale_key(key))
            out[name] = dequant(w, s)
            note(name, "fp8_block", True)
        else:
            assert "float8" not in str(w.dtype), f"{key} is fp8 but has no scale?!"
            out[name] = w.to(torch.bfloat16)
            note(name, str(w.dtype).split(".")[-1], False)
        print(f"{name}: {tuple(out[name].shape)}")

    P = f"model.layers.{L}."
    put("enorm", P + "enorm.weight", quant=False)
    put("hnorm", P + "hnorm.weight", quant=False)
    put("eh_proj", P + "eh_proj.weight", quant=False)
    put("input_ln", P + "input_layernorm.weight", quant=False)
    put("post_attn_ln", P + "post_attention_layernorm.weight", quant=False)
    put("shared_head_norm", P + "shared_head.norm.weight", quant=False)

    A = P + "self_attn."
    put("q_a", A + "q_a_proj.weight")
    put("q_a_ln", A + "q_a_layernorm.weight", quant=False)
    put("q_b", A + "q_b_proj.weight")
    put("kv_a", A + "kv_a_proj_with_mqa.weight")
    put("kv_a_ln", A + "kv_a_layernorm.weight", quant=False)
    kv_b = load_ckpt_tensor(args.ckpt, A + "kv_b_proj.weight")
    if scale_key(A + "kv_b_proj.weight") in idx:
        kv_b = dequant(kv_b, load_ckpt_tensor(args.ckpt, scale_key(A + "kv_b_proj.weight")))
    else:
        kv_b = kv_b.to(torch.bfloat16)
    out["kv_b"] = kv_b.view(64, 192 + 256, 512)
    note("kv_b", "fp8_block", True)
    print("kv_b:", tuple(out["kv_b"].shape))
    put("o_proj", A + "o_proj.weight")

    M = P + "mlp."
    gate = load_ckpt_tensor(args.ckpt, M + "gate.weight").to(torch.bfloat16)
    out["gate_w"] = gate
    note("gate_w", "bf16", False)
    out["gate_bias"] = load_ckpt_tensor(
        args.ckpt, M + "gate.e_score_correction_bias"
    ).to(torch.float32)
    note("gate_bias", "fp32", False)
    print("gate_w:", tuple(gate.shape), "gate_bias:", tuple(out["gate_bias"].shape))

    put("s_gate", M + "shared_experts.gate_proj.weight")
    put("s_up", M + "shared_experts.up_proj.weight")
    put("s_down", M + "shared_experts.down_proj.weight")

    # indexer weights: pass-through (frozen, not used in training; exported unchanged)
    for name, key in [
        ("idx_k_norm_w", A + "indexer.k_norm.weight"),
        ("idx_k_norm_b", A + "indexer.k_norm.bias"),
        ("idx_weights_proj", A + "indexer.weights_proj.weight"),
        ("idx_wk", A + "indexer.wk.weight"),
        ("idx_wq_b", A + "indexer.wq_b.weight"),
    ]:
        put(name, key)

    # routed experts, stacked [E, ...]
    for proj, out_name in [("gate_proj", "e_gate"), ("up_proj", "e_up"), ("down_proj", "e_down")]:
        ws, ss = [], []
        for e in range(256):
            k = f"{M}experts.{e}.{proj}.weight"
            ws.append(load_ckpt_tensor(args.ckpt, k))
            ss.append(load_ckpt_tensor(args.ckpt, scale_key(k)))
        stacked = torch.empty(
            (256,) + ws[0].shape, dtype=torch.bfloat16
        )
        for e, (w, s) in enumerate(zip(ws, ss)):
            stacked[e] = dequant(w, s)
        out[out_name] = stacked
        note(out_name, "fp8_block", True)
        print(out_name + ":", tuple(stacked.shape), "gb:", stacked.numel() * 2 / 1e9)

    out["embed"] = load_ckpt_tensor(args.ckpt, "model.embed_tokens.weight").to(torch.bfloat16)
    note("embed", "bf16", False)
    out["lm_head"] = load_ckpt_tensor(args.ckpt, "lm_head.weight").to(torch.bfloat16)
    note("lm_head", "bf16", False)
    print("embed:", tuple(out["embed"].shape), "lm_head:", tuple(out["lm_head"].shape))

    # save in shards: experts separately (large)
    from safetensors.torch import save_file

    small = {k: v for k, v in out.items() if not k.startswith("e_")}
    save_file(small, os.path.join(args.out, "draft.safetensors"))
    save_file(
        {k: v for k, v in out.items() if k.startswith("e_")},
        os.path.join(args.out, "experts.safetensors"),
    )
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    total = sum(v.numel() * v.element_size() for v in out.values())
    print(f"saved {total/1e9:.2f} GB to {args.out}")


if __name__ == "__main__":
    main()
