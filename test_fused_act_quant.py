"""Equivalence test: fused SiLU*mul + fp8 group-quant (legacy_exact_act=True)
vs the production pair (act_and_mul + row-major group quant + tma_align).

Bit-exactness is expected: the fused kernel's act expression matches
act_and_mul_kernel under --use_fast_math (same flags on both modules), the
quant arithmetic is the same kernel's, and the scale layout is the same
TMA-aligned col-major view that tma_align_input_scale produces.

Also quantifies the pre-existing fused expression (legacy_exact_act=False)
delta for the record, and verifies the downstream grouped-GEMM output is
bit-identical (real layer-5 down weights, real m_indices layout).

Usage: python test_fused_act_quant.py [--out test_fused_act_quant.json]
"""
import argparse, json
import torch

from bench_moe_layer import load_layer, route, DEV, FP8, H, I, E_LOCAL, ALIGN

torch.manual_seed(1234)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--out", default="test_fused_act_quant.json")
    args = ap.parse_args()

    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8)
    from sglang.kernels.ops.moe.ep_moe_kernels import tma_align_input_scale
    from sglang.kernels.ops.quantization.per_token_group_quant import (
        per_token_group_quant)
    from sglang.srt.layers.deep_gemm_wrapper import grouped_gemm_nt_f8f8bf16_contig

    N = 2 * I  # 4096 gate|up
    w = load_layer(args.layer)
    w2 = torch.stack([w[f"experts.{e}.down_proj.weight"] for e in range(E_LOCAL)]).to(DEV)
    w2_s = torch.stack([w[f"experts.{e}.down_proj.weight_scale_inv"]
                        for e in range(E_LOCAL)]).to(DEV)
    gw = w["gate.weight"].to(DEV).float(); gb = w["gate.e_score_correction_bias"].to(DEV)
    del w
    torch.cuda.empty_cache()

    # a realistic m_indices (router skew, 128-aligned counts)
    T = 8192 * 4
    h = torch.randn(T, H, device=DEV, dtype=torch.bfloat16) * 0.5
    ti, _ = route(h, gw, gb, "router")
    local = ti < E_LOCAL
    cnt = torch.zeros(E_LOCAL, dtype=torch.int64, device=DEV)
    cnt.scatter_add_(0, torch.where(local, ti, 0).reshape(-1), local.reshape(-1).long())
    cnt = cnt.cpu()
    aligned = (cnt + ALIGN - 1) // ALIGN * ALIGN
    m_indices = torch.repeat_interleave(
        torch.arange(E_LOCAL, dtype=torch.int32), aligned).to(DEV)
    M = int(aligned.sum())
    del h, ti, local
    torch.cuda.empty_cache()

    res = {"cases": [], "bit_exact": True}
    cases = []
    for name, M_case in (("M8192_shared", 8192), ("M20096_single", 20096), (f"M{M}_prod", M)):
        g = torch.randn(M_case, N, device=DEV, dtype=torch.bfloat16)
        cases.append((name, g))
        cases.append((name + "_scaled0.02", g * 0.02))
        cases.append((name + "_scaled30", g * 30.0))
        cases.append((name + "_zeros", torch.zeros(M_case, N, device=DEV, dtype=torch.bfloat16)))

    for name, gateup in cases:
        # production pair
        act = torch.empty(gateup.shape[0], N // 2, device=DEV, dtype=torch.bfloat16)
        silu_and_mul(gateup, act)
        q_ref, s_row = sglang_per_token_group_quant_fp8(
            act, 128, column_major_scales=False, scale_tma_aligned=False)
        s_ref = tma_align_input_scale(s_row)

        # fused (legacy-exact)
        q_new, s_new = per_token_group_quant(
            gateup, group_size=128, scale_ue8m0=False, fuse_silu_and_mul=True,
            column_major_scales=True, legacy_exact_act=True)

        # fused (pre-existing expression, for the record)
        q_old, s_old = per_token_group_quant(
            gateup, group_size=128, scale_ue8m0=False, fuse_silu_and_mul=True,
            column_major_scales=True)

        def cmp(qa, sa, qb, sb):
            q_eq = torch.equal(qa.view(torch.uint8), qb.view(torch.uint8))
            s_eq = torch.equal(sa.contiguous(), sb.contiguous().view_as(sa))
            return q_eq, s_eq

        q_eq, s_eq = cmp(q_ref, s_ref, q_new, s_new)
        d_old = (q_old.float() - q_ref.float()).abs()
        rec = dict(case=name, q_bitexact=bool(q_eq), s_bitexact=bool(s_eq))
        if not (q_eq and s_eq):
            res["bit_exact"] = False
            n_diff = (q_old.view(torch.uint8) != q_ref.view(torch.uint8)).sum().item()
            rec["n_q_diff"] = int((q_new.view(torch.uint8) != q_ref.view(torch.uint8)).sum())
        # pre-existing fused expression delta (fp8 byte level)
        rec["prefused_q_diff_frac"] = float(
            (q_old.view(torch.uint8) != q_ref.view(torch.uint8)).float().mean())
        rec["prefused_scale_maxreldiff"] = float(
            ((s_old.contiguous() - s_ref.contiguous()).abs()
             / s_ref.contiguous().abs().clamp(min=1e-12)).max())
        res["cases"].append(rec)
        print(f"{name:>24s}: legacy-exact q_bitexact={q_eq} s_bitexact={s_eq} | "
              f"pre-existing fused: q_diff_frac={rec['prefused_q_diff_frac']:.3e} "
              f"scale_maxrel={rec['prefused_scale_maxreldiff']:.3e}", flush=True)

    # downstream grouped GEMM bit-identity at the prod shape
    gateup = cases[-4][1]  # the M_prod base case
    act = torch.empty(gateup.shape[0], N // 2, device=DEV, dtype=torch.bfloat16)
    silu_and_mul(gateup, act)
    q_ref, s_row = sglang_per_token_group_quant_fp8(
        act, 128, column_major_scales=False, scale_tma_aligned=False)
    s_ref = tma_align_input_scale(s_row)
    q_new, s_new = per_token_group_quant(
        gateup, group_size=128, scale_ue8m0=False, fuse_silu_and_mul=True,
        column_major_scales=True, legacy_exact_act=True)
    out_ref = torch.empty(gateup.shape[0], H, device=DEV, dtype=torch.bfloat16)
    out_new = torch.empty_like(out_ref)
    grouped_gemm_nt_f8f8bf16_contig((q_ref, s_ref), (w2, w2_s), out_ref, m_indices)
    grouped_gemm_nt_f8f8bf16_contig((q_new, s_new), (w2, w2_s), out_new, m_indices)
    gemm_eq = torch.equal(out_ref, out_new)
    res["downstream_gemm_bitexact"] = bool(gemm_eq)
    print(f"downstream grouped down-GEMM bit-exact: {gemm_eq}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)
    assert res["bit_exact"] and gemm_eq, "NOT bit-exact - see report"


if __name__ == "__main__":
    main()
