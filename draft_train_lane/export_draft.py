#!/usr/bin/env python3
"""Export a fine-tuned draft back into the GLM-5.3 NextN checkpoint format.

Reads the training state dict (draft_finetuned.pt, training tensor names), the
extracted-weights meta.json (which tensors were fp8 block-quantized), and the
ORIGINAL checkpoint (for byte-identical pass-through of the frozen indexer
tensors), then writes an sglang-loadable draft directory:

  <out>/model.safetensors  — all model.layers.78.* keys + embed_tokens + lm_head,
                             fp8-block requantized exactly where the original was
  <out>/config.json        — verbatim copy of the source model config
  <out>/tokenizer*, chat_template.jinja — copies

Serve with --speculative-draft-model-path <out> (GlmMoeDsaForCausalLMNextN remaps
model.layers.78.<spec> -> model.<spec> and the rest -> model.decoder.<rest>).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import load_file

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from safetensors.torch import save_file

CKPT = "/projects/s6p/hf/hub/models--zai-org--GLM-5.3/snapshots/e0b07fd2751b42d5efa199cc02c2b271deadc516"
L = 78
BLOCK = 128
FP8_MAX = 448.0


def requant_fp8_block(w: torch.Tensor):
    """bf16 [N, K] -> (fp8 e4m3, scale_inv [N/128, K/128]). Matches block_quant_dequant."""
    n, k = w.shape
    wn, wk = (n + BLOCK - 1) // BLOCK * BLOCK, (k + BLOCK - 1) // BLOCK * BLOCK
    wp = torch.zeros(wn, wk, dtype=torch.float32)
    wp[:n, :k] = w.float()
    bp = wp.view(wn // BLOCK, BLOCK, wk // BLOCK, BLOCK)
    amax = bp.abs().amax(dim=(1, 3), keepdim=False)  # [nb, kb]
    scale = (amax / FP8_MAX).clamp(min=1e-12)
    q = (bp / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    q = q.view(wn, wk)[:n, :k].contiguous()
    # scale_inv semantics: dequant = q * scale_inv  (block_quant_dequant)
    return q, scale.contiguous()


def load_ckpt_tensor(ckpt, key):
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    path = os.path.join(ckpt, idx[key])
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--weights-dir", required=True, help="extractor output (meta.json)")
    ap.add_argument("--ft", required=True, help="draft_finetuned.pt from training")
    ap.add_argument("--out", required=True)
    ap.add_argument("--orig-ckpt", help="original HF checkpoint dir: frozen expert + kv_b fp8 tensors are copied byte-identical (zero requant drift)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = json.load(open(os.path.join(args.weights_dir, "meta.json")))
    ft = torch.load(args.ft, map_location="cpu")
    # training tensor name -> original checkpoint key
    name_map = {
        "enorm": f"model.layers.{L}.enorm.weight",
        "hnorm": f"model.layers.{L}.hnorm.weight",
        "eh_proj.weight": f"model.layers.{L}.eh_proj.weight",
        "input_ln": f"model.layers.{L}.input_layernorm.weight",
        "post_attn_ln": f"model.layers.{L}.post_attention_layernorm.weight",
        "shared_head_norm": f"model.layers.{L}.shared_head.norm.weight",
        "attn.q_a.weight": f"model.layers.{L}.self_attn.q_a_proj.weight",
        "attn.q_a_ln": f"model.layers.{L}.self_attn.q_a_layernorm.weight",
        "attn.q_b.weight": f"model.layers.{L}.self_attn.q_b_proj.weight",
        "attn.kv_a.weight": f"model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight",
        "attn.kv_a_ln": f"model.layers.{L}.self_attn.kv_a_layernorm.weight",
        "attn.o.weight": f"model.layers.{L}.self_attn.o_proj.weight",
        "moe.gate.w": f"model.layers.{L}.mlp.gate.weight",
        "moe.gate.bias": f"model.layers.{L}.mlp.gate.e_score_correction_bias",
        "moe.shared.gate": f"model.layers.{L}.mlp.shared_experts.gate_proj.weight",
        "moe.shared.up": f"model.layers.{L}.mlp.shared_experts.up_proj.weight",
        "moe.shared.down": f"model.layers.{L}.mlp.shared_experts.down_proj.weight",
    }
    quant_keys = {
        "attn.q_a.weight", "attn.q_b.weight", "attn.kv_a.weight", "attn.o.weight",
        "moe.shared.gate", "moe.shared.up", "moe.shared.down",
    }

    out = {}

    def put_ft(tname, ckpt_key, requant=False, dtype=None):
        w = ft[tname].float()
        if requant:
            q, s = requant_fp8_block(w)
            out[ckpt_key] = q
            out[ckpt_key[: -len(".weight")] + ".weight_scale_inv"] = s
        else:
            if dtype is not None:
                w = w.to(dtype)
            out[ckpt_key] = w.to(
                torch.bfloat16 if dtype is None else dtype
            ).contiguous()
        print(ckpt_key, tuple(out[ckpt_key].shape), "fp8" if requant else "bf16")

    for tname, ckpt_key in name_map.items():
        assert tname in ft, f"missing {tname} in finetuned state dict"
        if tname == "moe.gate.bias":
            put_ft(tname, ckpt_key, dtype=torch.float32)
        else:
            put_ft(tname, ckpt_key, requant=(tname in quant_keys))

    # kv_b: training tensor attn.kv_b [64, 448, 512] -> flat kv_b_proj [64*448, 512]
    kv_b = ft["attn.kv_b"].float().reshape(64 * (192 + 256), 512)
    q, s = requant_fp8_block(kv_b)
    out[f"model.layers.{L}.self_attn.kv_b_proj.weight"] = q
    out[f"model.layers.{L}.self_attn.kv_b_proj.weight_scale_inv"] = s

    # routed experts: e_gate/e_up [E, 2048, 6144], e_down [E, 6144, 2048]
    for tname, proj in [("moe.eg.w", "gate_proj"), ("moe.eu.w", "up_proj"), ("moe.ed.w", "down_proj")]:
        stack = ft[tname].float()
        for e in range(stack.shape[0]):
            q, s = requant_fp8_block(stack[e])
            k = f"model.layers.{L}.mlp.experts.{e}.{proj}.weight"
            out[k] = q
            out[k[: -len(".weight")] + ".weight_scale_inv"] = s
        print(f"experts {proj}: {stack.shape[0]} requantized")

    # frozen pass-through tensors, byte-identical from the original checkpoint:
    # indexer (wk, wq_b fp8 + scales; k_norm, weights_proj bf16)
    A = f"model.layers.{L}.self_attn.indexer."
    for key in [
        A + "wk.weight", A + "wk.weight_scale_inv",
        A + "wq_b.weight", A + "wq_b.weight_scale_inv",
        A + "k_norm.weight", A + "k_norm.bias",
        A + "weights_proj.weight",
    ]:
        out[key] = load_ckpt_tensor(args.ckpt, key)
        print("passthrough", key)

    # embed + lm_head (frozen, identical to target; included so the draft loader
    # finds a complete weight set in its own directory)
    out["model.embed_tokens.weight"] = ft["embed"].to(torch.bfloat16)
    out["lm_head.weight"] = ft["lm_head"].to(torch.bfloat16)

    # frozen fp8 passthrough from the ORIGINAL checkpoint: experts and kv_b
    # never train, so their fp8 codes + scales can be copied byte-identical
    # (the bf16-dequant -> requant path costs ~0.4% per-element drift).
    if args.orig_ckpt:
        idx = json.load(open(os.path.join(args.orig_ckpt, "model.safetensors.index.json")))["weight_map"]
        # kv_b is TRAINABLE (14.7M params): only passthrough if training did
        # not move it (vs the extraction); else keep the trained requantized
        # values so the export reflects what was trained.
        small = load_file(os.path.join(args.weights_dir, "draft.safetensors"))
        kv_drift = (
            (ft["attn.kv_b"].float() - small["kv_b"].float()).abs().max().item()
            / small["kv_b"].float().abs().max().item()
        )
        kv_moved = kv_drift > 1e-3
        if kv_moved:
            print(f"kv_b moved during training (drift {kv_drift:.3e}) -> keeping trained requantized kv_b")
        kvb_keys = set() if kv_moved else {
            f"model.layers.{L}.self_attn.kv_b_proj.weight",
            f"model.layers.{L}.self_attn.kv_b_proj.weight_scale_inv",
        }
        n_passthrough = 0
        for k in list(out.keys()):
            if (".mlp.experts." in k and f".{L}." in k) or k in kvb_keys:
                if k not in idx:
                    continue
                with safe_open(os.path.join(args.orig_ckpt, idx[k]), framework="pt", device="cpu") as f:
                    out[k] = f.get_tensor(k)
                n_passthrough += 1
        print(f"passthrough from orig ckpt: {n_passthrough} fp8 tensors (experts + kv_b if unmoved)")

    save_file(out, os.path.join(args.out, "model.safetensors"))

    # metadata files
    for f in [
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "chat_template.jinja",
    ]:
        src = os.path.join(args.ckpt, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    json.dump(
        {"source": "draft-train export", "layer": L},
        open(os.path.join(args.out, "export_meta.json"), "w"),
    )
    total = sum(v.numel() * v.element_size() for v in out.values())
    print(f"exported {len(out)} tensors, {total/1e9:.2f} GB -> {args.out}")


if __name__ == "__main__":
    main()
