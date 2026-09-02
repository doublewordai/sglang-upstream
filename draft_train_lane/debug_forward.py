#!/usr/bin/env python3
"""One synthetic batch through the training model, printing intermediate stats."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft_data import SyntheticData, make_batch  # noqa: E402
from draft_model import (  # noqa: E402
    DraftNextN, rms_norm, rope_interleave,
)

W = "/scratch/s6p/fergus.s6p/grace-1m/lanes/draft-train/draft_weights"


def stat(name, t):
    t = t.detach().float()
    print(
        f"{name:24s} shape={tuple(t.shape)} finite={torch.isfinite(t).all().item()} "
        f"absmax={t.abs().max().item():.3e} std={t.std().item():.3e}"
    )


def main():
    from safetensors.torch import load_file
    from train_draft import build_model

    device = "cuda"
    m = build_model(W, device)
    m.eval()
    data = SyntheticData(seed=0)
    rng = __import__("numpy").random.RandomState(7)
    items = data.windows(2, 512, rng, "train")
    tokens, prev, pos = make_batch(data.get, items, 512, device)
    stat("tokens", tokens.float())
    stat("prev_hidden", prev)

    with torch.no_grad():
        x = m.embed[tokens]
        stat("embed", x)
        e = rms_norm(x, m.enorm)
        stat("enorm(emb)", e)
        h = rms_norm(prev, m.hnorm)
        stat("hnorm(prev)", h)
        eh = m.eh_proj(torch.cat((e, h), dim=-1))
        stat("eh_proj", eh)
        res = eh
        hh = rms_norm(eh, m.input_ln)
        stat("input_ln", hh)
        # attention internals
        a = m.attn
        B, n, _ = hh.shape
        q = a.q_b(rms_norm(a.q_a(hh), a.q_a_ln)).view(B, n, 64, 256)
        stat("q", q)
        kv = a.kv_a(hh)
        stat("kv", kv)
        k_nope = rms_norm(kv[..., :512], a.kv_a_ln)
        stat("k_nope", k_nope)
        q_pe = rope_interleave(q[..., 192:], pos[:, :, None, None])
        k_pe = rope_interleave(kv[..., 512:], pos[:, :, None])
        stat("q_pe", q_pe)
        stat("k_pe", k_pe)
        w_kc = a.kv_b[:, :192, :].float()
        w_vc = a.kv_b[:, 192:, :].float()
        stat("w_kc", w_kc)
        stat("w_vc", w_vc)
        k_head = torch.einsum("bnd,hed->bnhe", k_nope.float(), w_kc)
        stat("k_head", k_head)
        v_head = torch.einsum("bnd,hed->bnhe", k_nope.float(), w_vc)
        stat("v_head", v_head)
        q_nope = q[..., :192]
        scores = torch.einsum("bnhe,bmhe->bhnm", q_nope, k_head)
        scores = scores + torch.einsum("bnhd,bmd->bhnm", q_pe, k_pe)
        scores = scores * (256**-0.5)
        stat("scores", scores)
        causal = torch.ones(n, n, device=device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = scores.softmax(-1)
        stat("attn", attn)
        out = (attn @ v_head.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).reshape(B, n, 64 * 256)
        stat("attn_out", out)
        o = a.o(out)
        stat("o_proj", o)
        h2 = res + o
        stat("post_attn", h2)
        res2 = h2
        h3 = rms_norm(h2, m.post_attn_ln)
        moe = m.moe(h3)
        stat("moe", moe)
        g = rms_norm(h3 + moe + res2 - h3, m.shared_head_norm)  # placeholder
        out_f = h3 + moe
        g = rms_norm(res2 + moe, m.shared_head_norm)
        stat("g", g)
        logits = m.logits_chunked if False else None
        lg = []
        flat = g.reshape(-1, 6144)
        for i in range(0, flat.shape[0], 256):
            lg.append(torch.nn.functional.linear(flat[i : i + 256], m.lm_head))
        lg = torch.cat(lg, 0)
        stat("logits", lg)


if __name__ == "__main__":
    main()
