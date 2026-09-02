"""Draft (NextN layer 78) training model in plain PyTorch.

Faithful to the engine's math for GlmMoeDsaForCausalLMNextN (DeepseekV3ForCausalLMNextN
in sglang-integ-0902), with two deliberate deviations recorded in the lane worklog:
  1. DENSE attention over the training window (<= 2048 tokens). The engine's DSA
     indexer selects top-2048 keys per query; for window positions t < 2048 the
     selection is all-available-keys, so dense == sparse exactly for the positions
     we train on. The indexer weights are NOT part of this model (frozen at the
     checkpoint values, re-exported unchanged).
  2. Teacher-forced parallel training (standard EAGLE/DeepSeek-MTP): the hnorm
     input at every window position is the TARGET's captured hidden state, not the
     draft's own output; the feature loss (MSE to target hiddens) is what keeps the
     inference-time chain (which feeds the draft's own hidden) in-distribution.

MoE routing (verified against sglang-integ-0902 topk.py/remap_topk_for_per_rank_shared_slots,
CUDA path): top-8 by sigmoid(logits)+bias, weights = sigmoid(logits) of selected,
renormalized, whole routed sum scaled by routed_scaling_factor=2.5 AFTER the experts,
shared expert added at net weight 1.0.

Attention: MLA with q_lora 2048, kv_lora 512, 64 heads, qk_nope 192 + qk_rope 64
(interleaved GPT-J rope, theta 8e6), v_head 256, softmax scale 256^-0.5
(RadixAttention scaling in deepseek_v2.py:1770). Batched: each window attends only
within itself (block-diagonal causal).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 6144
Q_LORA = 2048
KV_LORA = 512
N_HEADS = 64
QK_NOPE = 192
QK_ROPE = 64
QK_HEAD = QK_NOPE + QK_ROPE  # 256
V_HEAD = 256
N_EXPERTS = 256
TOP_K = 8
MOE_INTER = 2048
EPS = 1e-5
ROUTED_SCALING = 2.5
ROPE_THETA = 8e6
ATTN_SCALE = QK_HEAD**-0.5


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return y.to(x.dtype) * w.to(x.dtype)


def rope_interleave(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """GPT-J interleaved RoPE. x: [..., H, D]; positions broadcastable [...]."""
    D = x.shape[-1]
    inv = ROPE_THETA ** (
        -torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D
    )
    ang = positions.float() * inv  # positions pre-shaped to broadcast: [..., 1[, 1]]
    cos = ang.cos().to(x.dtype)
    sin = ang.sin().to(x.dtype)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class MLAAttention(nn.Module):
    """Dense batched training-form MLA. Input [B, n, H]; per-window causal."""

    def __init__(self):
        super().__init__()
        self.q_a = nn.Linear(HIDDEN, Q_LORA, bias=False)
        self.q_a_ln = nn.Parameter(torch.ones(Q_LORA))
        self.q_b = nn.Linear(Q_LORA, N_HEADS * QK_HEAD, bias=False)
        self.kv_a = nn.Linear(HIDDEN, KV_LORA + QK_ROPE, bias=False)
        self.kv_a_ln = nn.Parameter(torch.ones(KV_LORA))
        self.kv_b = nn.Parameter(torch.empty(N_HEADS, QK_NOPE + V_HEAD, KV_LORA))
        self.o = nn.Linear(N_HEADS * V_HEAD, HIDDEN, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """x: [B, n, H]; positions: [B, n] absolute."""
        B, n, _ = x.shape
        q = self.q_b(rms_norm(self.q_a(x), self.q_a_ln))  # [B, n, H*256]
        q = q.view(B, n, N_HEADS, QK_HEAD)
        q_nope, q_pe = q[..., :QK_NOPE], q[..., QK_NOPE:]
        kv = self.kv_a(x)
        k_nope = rms_norm(kv[..., :KV_LORA], self.kv_a_ln)
        k_pe = kv[..., KV_LORA:]
        q_pe = rope_interleave(q_pe, positions[:, :, None, None])  # [B, n, H, 64]
        k_pe = rope_interleave(k_pe, positions[:, :, None])  # [B, n, 64]

        w_kc = self.kv_b[:, :QK_NOPE, :].float()  # [H, 192, 512]
        w_vc = self.kv_b[:, QK_NOPE:, :].float()  # [H, 256, 512]
        # [B, n, H, 192] / [B, n, H, 256]
        k_head = torch.einsum("bnd,hed->bnhe", k_nope.float(), w_kc).to(x.dtype)
        v_head = torch.einsum("bnd,hed->bnhe", k_nope.float(), w_vc).to(x.dtype)

        scores = torch.einsum("bnhe,bmhe->bhnm", q_nope, k_head)
        scores = scores + torch.einsum("bnhd,bmd->bhnm", q_pe, k_pe)
        scores = scores * ATTN_SCALE
        causal = torch.ones(n, n, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = scores.softmax(-1) @ v_head.permute(0, 2, 1, 3)  # [B, H, n, 256]
        out = attn.permute(0, 2, 1, 3).reshape(B, n, N_HEADS * V_HEAD)
        return self.o(out)


class Gate(nn.Module):
    """Router: gate weight + noaux_tc bias (frozen bias)."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.empty(N_EXPERTS, HIDDEN))
        self.bias = nn.Parameter(torch.zeros(N_EXPERTS), requires_grad=False)

    def forward(self, xf):
        logits = F.linear(xf, self.w).float()
        scores = logits.sigmoid()
        top_ids = (scores + self.bias.float()).topk(TOP_K, dim=-1).indices
        w = scores.gather(1, top_ids)
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
        return top_ids, w


class ExpertStack(nn.Module):
    """One stacked projection for all experts, held as a FROZEN REPLICATED
    buffer (not an FSDP-managed parameter): nested-FSDP param access from the
    parent forward bypasses the child unit's all-gather, and 5.8M training
    tokens cannot usefully fine-tune 9.7B expert params anyway (0.6 tok/param).
    Gradients still flow THROUGH these GEMMs to the upstream trainable params.
    Full-expert training stays possible: promote to nn.Parameter and give the
    MoE its own single FSDP unit (see worklog)."""

    def __init__(self, out_dim: int, in_dim: int):
        super().__init__()
        self.register_buffer("w", torch.empty(N_EXPERTS, out_dim, in_dim))


class SharedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.empty(MOE_INTER, HIDDEN))
        self.up = nn.Parameter(torch.empty(MOE_INTER, HIDDEN))
        self.down = nn.Parameter(torch.empty(HIDDEN, MOE_INTER))

    def forward(self, xf):
        h = F.silu(F.linear(xf, self.gate)) * F.linear(xf, self.up)
        return F.linear(h, self.down)


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = Gate()
        self.eg = ExpertStack(MOE_INTER, HIDDEN)
        self.eu = ExpertStack(MOE_INTER, HIDDEN)
        self.ed = ExpertStack(HIDDEN, MOE_INTER)
        self.shared = SharedExperts()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n, _ = x.shape
        xf = x.reshape(B * n, HIDDEN)
        top_ids, w = self.gate(xf)

        y = torch.zeros_like(xf)
        for e in torch.unique(top_ids.reshape(-1)).tolist():
            tmask = top_ids == e  # [N, 8]
            tok = tmask.any(-1)
            xe = xf[tok]
            h = F.silu(F.linear(xe, self.eg.w[e])) * F.linear(xe, self.eu.w[e])
            he = F.linear(h, self.ed.w[e])
            wm = (w[tok] * tmask[tok].float()).sum(-1, keepdim=True).to(x.dtype)
            y.index_add_(0, tok.nonzero().squeeze(-1), he * wm)
        y = y * ROUTED_SCALING
        return (y + self.shared(xf)).view(B, n, HIDDEN)


class DraftNextN(nn.Module):
    """One window batch, teacher-forced target hiddens. Everything runs inside
    forward() so FSDP wrapping is transparent to the loss code."""

    def __init__(self, vocab: int, embed: torch.Tensor, lm_head: torch.Tensor):
        super().__init__()
        self.embed = nn.Parameter(embed, requires_grad=False)
        self.lm_head = nn.Parameter(lm_head, requires_grad=False)
        self.enorm = nn.Parameter(torch.ones(HIDDEN))
        self.hnorm = nn.Parameter(torch.ones(HIDDEN))
        self.eh_proj = nn.Linear(2 * HIDDEN, HIDDEN, bias=False)
        self.input_ln = nn.Parameter(torch.ones(HIDDEN))
        self.attn = MLAAttention()
        self.post_attn_ln = nn.Parameter(torch.ones(HIDDEN))
        self.moe = MoE()
        self.shared_head_norm = nn.Parameter(torch.ones(HIDDEN))

    def forward(
        self,
        tokens: torch.Tensor,  # [B, n]
        prev_hidden: torch.Tensor,  # [B, n, H] target hiddens at t-1 (teacher forced)
        positions: torch.Tensor,  # [B, n] absolute
        compute_logits: bool = False,
        logit_chunk: int = 256,
    ):
        x = self.embed[tokens]  # [B, n, H]
        eh = torch.cat(
            (rms_norm(x, self.enorm), rms_norm(prev_hidden, self.hnorm)), dim=-1
        )
        eh = self.eh_proj(eh)
        residual = eh
        h = rms_norm(eh, self.input_ln)
        h = self.attn(h, positions)
        h = residual + h
        residual = h
        h = rms_norm(h, self.post_attn_ln)
        h = self.moe(h)
        out = h + residual
        g = rms_norm(out, self.shared_head_norm)
        if not compute_logits:
            return g, None
        logits = []
        flat = g.reshape(-1, HIDDEN)
        for i in range(0, flat.shape[0], logit_chunk):
            logits.append(F.linear(flat[i : i + logit_chunk], self.lm_head))
        return g, torch.cat(logits, 0).view(*tokens.shape, -1)


def draft_loss(
    model,  # DraftNextN or an FSDP wrapper around it
    tokens: torch.Tensor,  # [B, n]
    prev_hidden: torch.Tensor,  # [B, n, H]
    positions: torch.Tensor,  # [B, n]
    feature_weight: float = 1.0,
    return_metrics: bool = False,
):
    """CE next-token + EAGLE feature loss.

    tokens[b,t] = x_{s+t}; prev_hidden[b,t] = target hidden at s+t-1.
    Label for g_t is x_{t+1} = tokens[b,t+1]; feature target for g_t is the
    target hidden at s+t = prev_hidden[b,t+1] (post-norm on both sides).
    """
    B, n = tokens.shape
    g, logits = model(tokens, prev_hidden, positions, compute_logits=True)

    labels = tokens[:, 1:]
    ce = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels.reshape(-1)
    )
    feat = prev_hidden[:, 1:]
    mse = F.mse_loss(g[:, :-1].float(), feat.float())

    loss = ce + feature_weight * mse
    if return_metrics:
        with torch.no_grad():
            lg = logits[:, :-1]
            pred = lg.argmax(-1)
            top1 = (pred == labels).float().mean().item()
            top4 = (
                (lg.topk(4, -1).indices == labels[:, :, None]).any(-1).float().mean().item()
            )
        return loss, {"ce": ce.item(), "mse": mse.item(), "top1": top1, "top4": top4}
    return loss


def chain_loss(
    model,
    tokens: torch.Tensor,  # [B, n]
    prev_hidden: torch.Tensor,  # [B, n, H] target hiddens at t-1
    positions: torch.Tensor,  # [B, n]
    feature_weight: float = 1.0,
    chain_len: int = 8,
    n_chains: int = 2,
    detach_feedback: bool = False,
    return_metrics: bool = False,
    generator: torch.Generator | None = None,
):
    """EAGLE-3.1-style chain fine-tune (depth-stability variant).

    Seeds a chain with the TARGET hidden at one window position (exactly the
    inference re-seed after each verify), then rolls the draft's OWN hidden
    forward for chain_len steps: step j's input is
    eh_proj([enorm(Emb(x_{s+j})); hnorm(g_{j-1})]) with g_{-1} = target hidden.
    The layer is re-run over the growing chain prefix (equivalent to inference's
    accumulated draft KV: each position's input is fixed once computed).

    Loss per step: CE(lm_head(g_j), x_{s+j+1}) + feature_weight *
    MSE(g_j, h_target_{s+j}). Metrics report top-1 BY DEPTH, which is the
    diagnostic for the deep-draft drift the 3.1 architecture addresses.
    """
    B, n = tokens.shape
    device = tokens.device
    losses = []
    m = {"ce": [], "mse": [], "top1_by_depth": [0] * chain_len, "chains": 0}
    for b in range(B):
        for c in range(n_chains):
            lo = 1
            hi = n - chain_len - 1
            if hi <= lo:
                continue
            if generator is not None:
                s = int(torch.randint(lo, hi, (1,), generator=generator, device=device))
            else:
                s = int(torch.randint(lo, hi, (1,), device=device))
            gs = []
            for j in range(chain_len):
                L = j + 1
                prev = [prev_hidden[b, s - 1]] + [
                    g.detach() if detach_feedback else g for g in gs
                ]
                prev_chunk = torch.stack(prev).unsqueeze(0)  # [1, L, H]
                tok_chunk = tokens[b, s : s + L].unsqueeze(0)  # [1, L]
                pos_chunk = positions[b, s : s + L].unsqueeze(0)  # [1, L]
                g_all, lg_all = model(
                    tok_chunk, prev_chunk, pos_chunk, compute_logits=True
                )
                g_j = g_all[0, -1]  # [H]
                gs.append(g_j)
                # next-token CE from the last position's logits (computed inside
                # the model call: FSDP params are only valid there)
                lg = lg_all[0, -1]
                label = tokens[b, s + j + 1]
                ce = torch.nn.functional.cross_entropy(
                    lg.float().unsqueeze(0), label.unsqueeze(0)
                )
                # feature loss vs the target hidden at s+j
                feat = prev_hidden[b, s + j]
                mse = torch.nn.functional.mse_loss(g_j.float(), feat.float())
                losses.append(ce + feature_weight * mse)
                m["ce"].append(ce.item())
                m["mse"].append(mse.item())
                with torch.no_grad():
                    if int(lg.argmax()) == int(label):
                        m["top1_by_depth"][j] += 1
                m["chains"] += 1
    if not losses:
        z = torch.zeros(1, device=device, requires_grad=True)
        return (z, m) if return_metrics else z
    loss = torch.stack(losses).mean()
    if return_metrics:
        m["ce"] = sum(m["ce"]) / len(m["ce"])
        m["mse"] = sum(m["mse"]) / len(m["mse"])
        if m["chains"]:
            m["top1_by_depth"] = [x / m["chains"] for x in m["top1_by_depth"]]
        return loss, m
    return loss
