"""M3 microbench: fused SiLU*mul + fp8 group-quant (legacy_exact_act) vs the
production trio (act_and_mul + row-major group quant + tma_align), at the
routed (M_pad from real router counts) and shared-expert (M=8192) shapes.
Interleaved rounds; CUDA events.
Also re-times the full routed sequence (scatter..gather) with the fused act+quant
replacing the trio, and the shared-expert chain (quant+gemm+act+quant+gemm).
"""
import json, statistics
import torch

from bench_moe_layer import load_layer, route, DEV, FP8, H, I, E_LOCAL, ALIGN

CEIL_BW = 3.665


def time_one(fn, loop=2):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(loop):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / loop


def main():
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8, w8a8_block_fp8_matmul_deepgemm)
    from sglang.kernels.ops.moe.ep_moe_kernels import (
        tma_align_input_scale, ep_scatter, ep_gather)
    from sglang.kernels.ops.quantization.per_token_group_quant import (
        per_token_group_quant)
    from sglang.srt.layers.deep_gemm_wrapper import grouped_gemm_nt_f8f8bf16_contig

    w = load_layer(5)
    w13 = torch.stack([torch.cat([w[f"experts.{e}.gate_proj.weight"],
                                  w[f"experts.{e}.up_proj.weight"]], 0)
                       for e in range(E_LOCAL)]).to(DEV)
    w13_s = torch.stack([torch.cat([w[f"experts.{e}.gate_proj.weight_scale_inv"],
                                    w[f"experts.{e}.up_proj.weight_scale_inv"]], 0)
                         for e in range(E_LOCAL)]).to(DEV)
    w2 = torch.stack([w[f"experts.{e}.down_proj.weight"] for e in range(E_LOCAL)]).to(DEV)
    w2_s = torch.stack([w[f"experts.{e}.down_proj.weight_scale_inv"]
                        for e in range(E_LOCAL)]).to(DEV)
    gw = w["gate.weight"].to(DEV).float(); gb = w["gate.e_score_correction_bias"].to(DEV)
    del w
    torch.cuda.empty_cache()

    res = {"kernels": {}}
    rounds = 15

    def rec(name, p50, note=""):
        res["kernels"][name] = dict(p50_us=round(p50, 1), note=note)
        print(f"{name:>34s}: p50 {p50:8.1f} us  {note}", flush=True)

    # ---------- shared-expert shape (M=8192) ----------
    for M in (8192, 67712):
        gateup = torch.randn(M, 2 * I, device=DEV, dtype=torch.bfloat16)
        act = torch.empty(M, I, device=DEV, dtype=torch.bfloat16)

        def prod_trio():
            silu_and_mul(gateup, act)
            q, s = sglang_per_token_group_quant_fp8(
                act, 128, column_major_scales=False, scale_tma_aligned=False)
            return tma_align_input_scale(s)

        def fused():
            return per_token_group_quant(
                gateup, group_size=128, scale_ue8m0=False, fuse_silu_and_mul=True,
                column_major_scales=True, legacy_exact_act=True)

        for _ in range(6):
            prod_trio(); fused()
        torch.cuda.synchronize()
        t_prod, t_fused = [], []
        for _ in range(rounds):
            t_prod.append(time_one(prod_trio))
            t_fused.append(time_one(fused))
        rec(f"prod_trio_M{M}", statistics.median(t_prod),
            "act_and_mul + row-major quant + tma_align")
        rec(f"fused_M{M}", statistics.median(t_fused),
            "fused silu*mul+quant (legacy_exact, tma-aligned scales)")
        by = M * 2 * I * 2 + M * I * 2 + M * I + M * I / 128 * 4 * 2
        res["kernels"][f"fused_M{M}"]["tbps"] = round(
            (M * 2 * I * 2 + M * I + M * (I // 128) * 4)
            / (statistics.median(t_fused) * 1e-6) / 1e12, 2)
        del gateup, act
        torch.cuda.empty_cache()

    # ---------- full routed sequence before/after ----------
    T = 8192 * 4
    h = torch.randn(T, H, device=DEV, dtype=torch.bfloat16) * 0.5
    ti, _ = route(h, gw, gb, "router")
    local = ti < E_LOCAL
    cnt = torch.zeros(E_LOCAL, dtype=torch.int64, device=DEV)
    cnt.scatter_add_(0, torch.where(local, ti, 0).reshape(-1), local.reshape(-1).long())
    cnt = cnt.cpu()
    aligned = (cnt + ALIGN - 1) // ALIGN * ALIGN
    M_pad = int(aligned.sum()); real = int(cnt.sum())
    recv_mask = local.any(dim=1)
    recv_tokens = int(recv_mask.sum())
    recv_idx = torch.nonzero(recv_mask, as_tuple=True)[0]
    recv_x_bf16 = h[recv_idx]
    recv_topk = torch.where(local[recv_idx], ti[recv_idx],
                            torch.full_like(ti[recv_idx], -1)).to(torch.int64)
    tw, _ = route(h, gw, gb, "router")
    recv_topk_w = (tw[recv_idx] / tw[recv_idx].sum(-1, keepdim=True)).float()
    recv_q, recv_s = sglang_per_token_group_quant_fp8(
        recv_x_bf16, 128, column_major_scales=False, scale_tma_aligned=False)
    del h, ti, local, recv_mask, recv_idx, recv_x_bf16, tw
    torch.cuda.empty_cache()

    input_tensor = torch.empty(M_pad, H, device=DEV, dtype=FP8)
    input_scale = torch.empty(M_pad, H // 128, device=DEV, dtype=torch.float32)
    m_indices = torch.repeat_interleave(
        torch.arange(E_LOCAL, dtype=torch.int32), aligned).to(DEV)
    output_index = torch.empty_like(recv_topk)
    gateup_out = torch.empty(M_pad, 2 * I, device=DEV, dtype=torch.bfloat16)
    down_in = torch.empty(M_pad, I, device=DEV, dtype=torch.bfloat16)
    down_out = torch.empty(M_pad, H, device=DEV, dtype=torch.bfloat16)
    gather_out = torch.empty(recv_tokens, H, device=DEV, dtype=torch.bfloat16)
    cnt_gpu = aligned.to(torch.int32).to(DEV)
    start_loc = torch.empty_like(cnt_gpu)

    def seq(prod_act):
        ep_scatter(recv_q, recv_s, recv_topk, cnt_gpu, cnt_gpu, start_loc,
                   input_tensor, input_scale, m_indices, output_index)
        a = tma_align_input_scale(input_scale)
        grouped_gemm_nt_f8f8bf16_contig((input_tensor, a), (w13, w13_s),
                                        gateup_out, m_indices)
        if prod_act:
            silu_and_mul(gateup_out, down_in)
            dq, ds = sglang_per_token_group_quant_fp8(
                down_in, 128, column_major_scales=False, scale_tma_aligned=False)
            dsa = tma_align_input_scale(ds)
        else:
            dq, dsa = per_token_group_quant(
                gateup_out, group_size=128, scale_ue8m0=False,
                fuse_silu_and_mul=True, column_major_scales=True,
                legacy_exact_act=True)
        grouped_gemm_nt_f8f8bf16_contig((dq, dsa), (w2, w2_s), down_out, m_indices)
        ep_gather(down_out, recv_topk, recv_topk_w, output_index, gather_out)

    for _ in range(4):
        seq(True); seq(False)
    torch.cuda.synchronize()
    t_before, t_after = [], []
    for _ in range(rounds):
        t_before.append(time_one(lambda: seq(True), loop=1))
        t_after.append(time_one(lambda: seq(False), loop=1))
    rec("RANKED_SEQUENCE_before", statistics.median(t_before),
        f"prod act+quant+align, M_pad={M_pad} real={real}")
    rec("RANKED_SEQUENCE_after", statistics.median(t_after), "fused act+quant")
    res["M_pad"] = M_pad; res["real"] = real

    with open("bench_m3.json", "w") as f:
        json.dump(res, f, indent=2)
    print("wrote bench_m3.json")


if __name__ == "__main__":
    main()
