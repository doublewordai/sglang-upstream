"""M2: DeepGEMM grouped-GEMM variant sweep at real prefill shapes (1 GPU).

Variants (gate_up N=4096,K=6144 and down N=6144,K=2048, G=64, real layer-5 weights):
  contig      - production call (m_grouped_fp8_gemm_nt_contiguous, all rows valid
                incl. 128-alignment padding; scheduled rows = sum ceil(c_e/128)*128)
  contig_rt   - same but compiled_dims='' (runtime N/K) if accepted
  contig_nopad- counts floored to 128 multiples (zero padding, same M ballpark)
  masked      - m_grouped_fp8_gemm_nt_masked on [G, m_cap, K] layout, masked_m=raw c_e
  dense_equiv - dense deepgemm at M=sum(c_e) with expert-0 weights = no-grouping bound
Interleaved rounds (each variant once per round, loop=2 calls per timing) to
control clock drift; report median-of-rounds. TF/s "real" = useful FLOPs (sum c_e)
divided by time; "sched" counts scheduled (padded) rows.

Chunk-size sweep (uniform routing): dp-tokens in {2048,4096,8192,16384} x 4 ranks
-> avg rows/expert {256,512,1024,2048}, contig only.

Usage: python bench_gemm_variants.py [--modes router,uniform] [--chunk-sweep] [--out m2.json]
"""
import argparse, json, math, statistics
import torch

from bench_moe_layer import load_layer, route, DEV, FP8, H, I, E_LOCAL, TOPK, ALIGN

CEIL_FP8 = 1305.0


def time_one(fn, loop=2):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(loop):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--modes", default="router,uniform")
    ap.add_argument("--dp-tokens", type=int, default=8192)
    ap.add_argument("--dp-ranks", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--chunk-sweep", action="store_true")
    ap.add_argument("--contig-only", action="store_true")
    ap.add_argument("--out", default="m2.json")
    args = ap.parse_args()

    from sglang.srt.layers.deep_gemm_wrapper import (
        grouped_gemm_nt_f8f8bf16_contig, grouped_gemm_nt_f8f8bf16_masked)
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8, w8a8_block_fp8_matmul_deepgemm)
    from sglang.kernels.ops.moe.ep_moe_kernels import tma_align_input_scale
    import deep_gemm

    w = load_layer(args.layer)
    w13 = torch.stack([torch.cat([w[f"experts.{e}.gate_proj.weight"],
                                  w[f"experts.{e}.up_proj.weight"]], 0)
                       for e in range(E_LOCAL)]).to(DEV)
    w13_s = torch.stack([torch.cat([w[f"experts.{e}.gate_proj.weight_scale_inv"],
                                    w[f"experts.{e}.up_proj.weight_scale_inv"]], 0)
                         for e in range(E_LOCAL)]).to(DEV)
    w2 = torch.stack([w[f"experts.{e}.down_proj.weight"] for e in range(E_LOCAL)]).to(DEV)
    w2_s = torch.stack([w[f"experts.{e}.down_proj.weight_scale_inv"]
                        for e in range(E_LOCAL)]).to(DEV)
    gw = w["gate.weight"].to(DEV).float()
    gb = w["gate.e_score_correction_bias"].to(DEV)
    del w
    torch.cuda.empty_cache()
    results = []

    def sweep(dp_tokens, dp_ranks, mode, contig_only=False):
        T = dp_tokens * dp_ranks
        h = torch.randn(T, H, device=DEV, dtype=torch.bfloat16) * 0.5
        ti, _ = route(h, gw, gb, mode)
        local = ti < E_LOCAL
        cnt = torch.zeros(E_LOCAL, dtype=torch.int64, device=DEV)
        cnt.scatter_add_(0, torch.where(local, ti, 0).reshape(-1), local.reshape(-1).long())
        cnt = cnt.cpu()
        del h, ti, local
        torch.cuda.empty_cache()

        real = int(cnt.sum())
        aligned = (cnt + ALIGN - 1) // ALIGN * ALIGN
        M_pad = int(aligned.sum())
        nopad = (cnt // ALIGN) * ALIGN
        M_nopad = int(nopad.sum())
        m_cap = int((int(aligned.max()) + 255) // 256 * 256)
        skew = dict(mean=round(float(cnt.float().mean()), 1),
                    std=round(float(cnt.float().std()), 1),
                    mx=int(cnt.max()), nz=int((cnt > 0).sum()))
        print(f"=== {mode} tok={dp_tokens}x{dp_ranks}: real={real} M_pad={M_pad} "
              f"pad={ (M_pad-real)/M_pad:.3f} nopad={M_nopad} m_cap={m_cap} cnt={skew}",
              flush=True)

        for gemm_name, N, K, wt, ws in (("gate_up", 2 * I, H, w13, w13_s),
                                        ("down", H, I, w2, w2_s)):
            x = torch.randn(M_pad, K, device=DEV, dtype=torch.bfloat16)
            q, s_row = sglang_per_token_group_quant_fp8(
                x, 128, column_major_scales=False, scale_tma_aligned=False)
            s_al = tma_align_input_scale(s_row)
            mi = torch.repeat_interleave(
                torch.arange(E_LOCAL, dtype=torch.int32), aligned).to(DEV)
            out = torch.empty(M_pad, N, device=DEV, dtype=torch.bfloat16)
            mi2 = torch.repeat_interleave(
                torch.arange(E_LOCAL, dtype=torch.int32), nopad).to(DEV)
            q2 = q[:M_nopad]
            s_al2 = tma_align_input_scale(s_row[:M_nopad])
            out2 = torch.empty(M_nopad, N, device=DEV, dtype=torch.bfloat16)

            variants = {
                "contig": (lambda: grouped_gemm_nt_f8f8bf16_contig(
                    (q, s_al), (wt, ws), out, mi), M_pad),
                "contig_nopad": (lambda: grouped_gemm_nt_f8f8bf16_contig(
                    (q2, s_al2), (wt, ws), out2, mi2), M_nopad),
            }
            extra = {}
            if not contig_only:
                extra["am"] = torch.empty(E_LOCAL, m_cap, K, device=DEV, dtype=FP8)
                extra["asm"] = torch.zeros(E_LOCAL, m_cap, K // 128, device=DEV,
                                           dtype=torch.float32)
                st = 0
                for e in range(E_LOCAL):
                    c = int(aligned[e])
                    if c:
                        extra["am"][e, :c] = q[st:st + c]
                        extra["asm"][e, :c] = s_row[st:st + c]
                    st += c
                extra["om"] = torch.empty(E_LOCAL, m_cap, N, device=DEV,
                                          dtype=torch.bfloat16)
                masked_m = cnt.to(torch.int32).to(DEV)
                expected_m = max(1, math.ceil(real / E_LOCAL))
                variants["masked"] = (
                    lambda: grouped_gemm_nt_f8f8bf16_masked(
                        (extra["am"], extra["asm"]), (wt, ws), extra["om"],
                        masked_m, expected_m), M_pad)
                dq = torch.randn(real, K, device=DEV, dtype=torch.bfloat16)
                dq, ds = sglang_per_token_group_quant_fp8(
                    dq, 128, column_major_scales=True, scale_tma_aligned=True)
                extra["dq"], extra["ds"] = dq, ds
                variants["dense_equiv"] = (
                    lambda: w8a8_block_fp8_matmul_deepgemm(
                        extra["dq"], wt[0], extra["ds"], ws[0], [128, 128],
                        torch.bfloat16), real)
                try:
                    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                        (q, s_al), (wt, ws), out, mi, compiled_dims='')
                    variants["contig_rt"] = (lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                        (q, s_al), (wt, ws), out, mi, compiled_dims=''), M_pad)
                except Exception as ex:
                    print(f"  contig_rt unavailable: {type(ex).__name__}: {ex}")

            for name, (fn, M) in variants.items():
                try:
                    fn()
                except Exception as ex:
                    print(f"  {gemm_name}/{name}: FAILED {type(ex).__name__}: {ex}", flush=True)
            torch.cuda.synchronize()

            times = {name: [] for name in variants}
            for r in range(args.rounds):
                for name, (fn, M) in variants.items():
                    try:
                        times[name].append(time_one(fn))
                    except Exception:
                        pass
            for name, ts in times.items():
                if not ts:
                    continue
                M_sched = variants[name][1]
                fl_real = 2.0 * real * N * K
                p50 = statistics.median(ts)
                rec = dict(shape=gemm_name, variant=name, mode=mode,
                           dp_tokens=dp_tokens, dp_ranks=dp_ranks,
                           real_rows=real, sched_rows=M_sched, m_cap=m_cap,
                           pad_frac=round((M_pad - real) / M_pad, 4),
                           p50_us=round(p50, 1),
                           p90_us=round(sorted(ts)[int(len(ts) * .9)], 1),
                           tflops_sched=round(2.0 * M_sched * N * K / (p50 * 1e-6) / 1e12, 1),
                           tflops_real=round(fl_real / (p50 * 1e-6) / 1e12, 1),
                           pct_ceiling_real=round(
                               fl_real / (p50 * 1e-6) / 1e12 / CEIL_FP8 * 100, 1))
                results.append(rec)
                print(f"  {gemm_name:>8s}/{name:>13s} M={M_sched:>6d} real={real:>6d} "
                      f"pad={rec['pad_frac']:.3f} p50={p50:8.1f}us "
                      f"{rec['tflops_real']:7.1f} TF/s real ({rec['pct_ceiling_real']:4.1f}%)",
                      flush=True)
            variants.clear()
            extra.clear()
            del x, q, s_row, s_al, mi, out, mi2, out2, q2, s_al2
            torch.cuda.empty_cache()

    for mode in args.modes.split(","):
        sweep(args.dp_tokens, args.dp_ranks, mode)
    if args.chunk_sweep:
        for tok in (2048, 4096, 8192, 16384):
            sweep(tok, args.dp_ranks, "uniform", contig_only=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
