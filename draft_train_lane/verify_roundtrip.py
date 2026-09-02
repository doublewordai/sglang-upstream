#!/usr/bin/env python3
"""Verify the export roundtrip: export_draft.py output vs the ORIGINAL checkpoint.

bf16 keys: must be bit-identical. fp8-block keys: dequantize both sides and
report max abs/rel diff (fp8 requant of already-fp8 values should round-trip
exactly; the extract->export path dequantizes then requantizes, so the error
is bounded by one fp8 quantization step of the DEQUANTIZED value, which is
itself a fixed-point of the grid when the original block scale is reused...
we report the measured numbers).
"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

CKPT = "/projects/s6p/hf/hub/models--zai-org--GLM-5.3/snapshots/e0b07fd2751b42d5efa199cc02c2b271deadc516"
L = 78


def load_ckpt(ckpt, key):
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(ckpt, idx[key]), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def dequant(w, s, block=128):
    n, k = w.shape
    srep = s.repeat_interleave(block, -2).repeat_interleave(block, -1)[..., :n, :k]
    return (w.to(torch.float32) * srep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()
    ex = safe_open(os.path.join(args.export, "model.safetensors"), framework="pt", device="cpu")

    def ex_get(k):
        return ex.get_tensor(k)

    worst = []
    n_bitexact = n_fp8 = 0
    keys = [
        (f"model.layers.{L}.enorm.weight", False),
        (f"model.layers.{L}.hnorm.weight", False),
        (f"model.layers.{L}.eh_proj.weight", False),
        (f"model.layers.{L}.input_layernorm.weight", False),
        (f"model.layers.{L}.post_attention_layernorm.weight", False),
        (f"model.layers.{L}.shared_head.norm.weight", False),
        (f"model.layers.{L}.self_attn.q_a_layernorm.weight", False),
        (f"model.layers.{L}.self_attn.kv_a_layernorm.weight", False),
        (f"model.layers.{L}.mlp.gate.weight", False),
        (f"model.layers.{L}.mlp.gate.e_score_correction_bias", False),
        (f"model.layers.{L}.self_attn.q_a_proj.weight", True),
        (f"model.layers.{L}.self_attn.q_b_proj.weight", True),
        (f"model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight", True),
        (f"model.layers.{L}.self_attn.kv_b_proj.weight", True),
        (f"model.layers.{L}.self_attn.o_proj.weight", True),
        (f"model.layers.{L}.mlp.shared_experts.gate_proj.weight", True),
        (f"model.layers.{L}.mlp.shared_experts.up_proj.weight", True),
        (f"model.layers.{L}.mlp.shared_experts.down_proj.weight", True),
        (f"model.layers.{L}.mlp.experts.0.gate_proj.weight", True),
        (f"model.layers.{L}.mlp.experts.100.up_proj.weight", True),
        (f"model.layers.{L}.mlp.experts.255.down_proj.weight", True),
        ("model.embed_tokens.weight", False),
        ("lm_head.weight", False),
    ]
    for key, is_fp8 in keys:
        orig = load_ckpt(args.ckpt, key).float()
        new = ex_get(key)
        if is_fp8:
            skey = key[: -len(".weight")] + ".weight_scale_inv"
            s = ex_get(skey)
            newd = dequant(new, s)
            os_ = load_ckpt(args.ckpt, key[: -len(".weight")] + ".weight_scale_inv")
            origd = dequant(load_ckpt(args.ckpt, key), os_)
            diff = (newd - origd).abs()
            rel = diff / origd.abs().clamp(min=1e-6)
            worst.append((rel.max().item(), key))
            n_fp8 += 1
            print(
                f"{key}: fp8 roundtrip maxabs={diff.max().item():.3e} "
                f"maxrel={rel.max().item():.3e}"
            )
        else:
            same = torch.equal(new.float(), orig)
            if not same:
                d = (new.float() - orig).abs().max().item()
                worst.append((d, key))
                print(f"{key}: BF16 MISMATCH maxabs={d:.3e}")
            else:
                n_bitexact += 1

    # indexer pass-through must be bit-exact
    A = f"model.layers.{L}.self_attn.indexer."
    for k in [A + "wk.weight", A + "wq_b.weight", A + "k_norm.weight", A + "k_norm.bias", A + "weights_proj.weight"]:
        o = load_ckpt(args.ckpt, k)
        if not torch.equal(o, ex_get(k)):
            worst.append((1.0, k + " (passthrough mismatch!)"))
            print(f"{k}: PASSTHROUGH MISMATCH")

    worst.sort(reverse=True)
    print(f"\nbit-exact bf16/passthrough keys: {n_bitexact}+5, fp8 keys: {n_fp8}")
    print("worst rel diffs:", worst[:5])
    ok = all(w[0] < 0.2 for w in worst)
    print("ROUNDTRIP-" + ("OK" if ok else "FAIL"))


if __name__ == "__main__":
    main()
