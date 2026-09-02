#!/usr/bin/env python3
"""Roundtrip helper: pack the EXTRACTED (untrained) weights as a training state dict.

Used to test the export path end-to-end before any training exists:
  extract_draft_weights.py -> make_roundtrip_pt.py -> export_draft.py -> load.
"""
import os
import sys

import torch
from safetensors.torch import load_file

OUT = "/scratch/s6p/fergus.s6p/grace-1m/lanes/draft-train"


def main():
    d = os.path.join(OUT, "draft_weights")
    small = load_file(os.path.join(d, "draft.safetensors"))
    experts = load_file(os.path.join(d, "experts.safetensors"))
    sd = {
        "embed": small["embed"],
        "lm_head": small["lm_head"],
        "enorm": small["enorm"],
        "hnorm": small["hnorm"],
        "eh_proj.weight": small["eh_proj"],
        "input_ln": small["input_ln"],
        "post_attn_ln": small["post_attn_ln"],
        "shared_head_norm": small["shared_head_norm"],
        "attn.q_a.weight": small["q_a"],
        "attn.q_a_ln": small["q_a_ln"],
        "attn.q_b.weight": small["q_b"],
        "attn.kv_a.weight": small["kv_a"],
        "attn.kv_a_ln": small["kv_a_ln"],
        "attn.kv_b": small["kv_b"],
        "attn.o.weight": small["o_proj"],
        "moe.gate.w": small["gate_w"],
        "moe.gate.bias": small["gate_bias"],
        "moe.shared.gate": small["s_gate"],
        "moe.shared.up": small["s_up"],
        "moe.shared.down": small["s_down"],
        "moe.eg.w": experts["e_gate"],
        "moe.eu.w": experts["e_up"],
        "moe.ed.w": experts["e_down"],
    }
    torch.save(sd, os.path.join(OUT, "roundtrip.pt"))
    print(f"saved {len(sd)} tensors -> roundtrip.pt")


if __name__ == "__main__":
    main()
