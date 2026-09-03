"""CUTLASS (sgl_kernel es_fp8_blockwise_scaled_grouped_mm, SM90 ptr-free variant)
vs DeepGEMM m_grouped contiguous, at the real prefill shapes.

The CUTLASS path takes per-group problem sizes and exact expert offsets (no
128-row buffer alignment required); we also try 128-aligned offsets. Numerics
are sanity-checked against a per-expert dequant reference (max rel diff).

Shapes: router-skew prod (M_pad 67712 / real 63351) and single-request
(M_pad 20096 / real 15864, the 21%-pad case), gate_up (N=4096,K=6144) and
down (N=6144,K=2048).
"""
import json, statistics
import torch

from bench_moe_layer import load_layer, route, DEV, FP8, H, I, E_LOCAL, ALIGN

CEIL = 1305.0


def time_one(fn, loop=2):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(loop):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / loop


def main():
    import sgl_kernel
    from sglang.srt.layers.deep_gemm_wrapper import grouped_gemm_nt_f8f8bf16_contig
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8)
    from sglang.kernels.ops.moe.ep_moe_kernels import tma_align_input_scale

    w = load_layer(5)
    wts = {}
    wts["gate_up"] = (torch.stack([torch.cat([w[f"experts.{e}.gate_proj.weight"],
                                              w[f"experts.{e}.up_proj.weight"]], 0)
                                   for e in range(E_LOCAL)]).to(DEV),
                      torch.stack([torch.cat([w[f"experts.{e}.gate_proj.weight_scale_inv"],
                                              w[f"experts.{e}.up_proj.weight_scale_inv"]], 0)
                                   for e in range(E_LOCAL)]).to(DEV),
                      2 * I, H)
    wts["down"] = (torch.stack([w[f"experts.{e}.down_proj.weight"] for e in range(E_LOCAL)]).to(DEV),
                   torch.stack([w[f"experts.{e}.down_proj.weight_scale_inv"]
                                for e in range(E_LOCAL)]).to(DEV),
                   H, I)
    gw = w["gate.weight"].to(DEV).float(); gb = w["gate.e_score_correction_bias"].to(DEV)
    del w
    torch.cuda.empty_cache()

    out = []
    for mode, dp_tokens, dp_ranks in (("router", 8192, 4), ("router", 8192, 1)):
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
        nz = int((cnt > 0).sum())
        print(f"=== {mode} tok={dp_tokens}x{dp_ranks}: real={real} M_pad={M_pad} "
              f"pad={(M_pad-real)/M_pad:.3f} nz_experts={nz}", flush=True)

        for gemm_name in ("gate_up", "down"):
            W, WS, N, K = wts[gemm_name]
            # contiguous DeepGEMM layout
            x = torch.randn(M_pad, K, device=DEV, dtype=torch.bfloat16)
            q, s_row = sglang_per_token_group_quant_fp8(
                x, 128, column_major_scales=False, scale_tma_aligned=False)
            s_al = tma_align_input_scale(s_row)
            mi = torch.repeat_interleave(
                torch.arange(E_LOCAL, dtype=torch.int32), aligned).to(DEV)
            out_c = torch.empty(M_pad, N, device=DEV, dtype=torch.bfloat16)

            # CUTLASS exact-offset layout (slice valid prefix of each expert)
            a_es = torch.empty(real, K, device=DEV, dtype=FP8)
            s_es = torch.empty(real, K // 128, device=DEV, dtype=torch.float32)
            st = 0
            starts, sizes = [], []
            for e in range(E_LOCAL):
                c, a = int(cnt[e]), int(aligned[e])
                if c:
                    a_es[st:st + c] = q[st:st + c]      # segments share prefix
                    s_es[st:st + c] = s_row[st:st + c]
                starts.append(st); sizes.append(c)
                st += a
            out_es = torch.empty(real, N, device=DEV, dtype=torch.bfloat16)
            problem_sizes = torch.tensor(
                [[sizes[e], N, K] for e in range(E_LOCAL)],
                device=DEV, dtype=torch.int32)
            expert_offsets = torch.tensor(
                [starts[e] for e in range(E_LOCAL)], device=DEV, dtype=torch.int32)
            ab_strides = torch.full((E_LOCAL,), K, device=DEV, dtype=torch.int64)
            c_strides = torch.full((E_LOCAL,), N, device=DEV, dtype=torch.int64)
            workspace = torch.empty(90000, device=DEV, dtype=torch.uint8)

            def es_call():
                sgl_kernel.es_fp8_blockwise_scaled_grouped_mm(
                    out_es, a_es, W.transpose(1, 2), s_es,
                    WS.transpose(1, 2), ab_strides, ab_strides, c_strides,
                    problem_sizes, expert_offsets, workspace)

            def dg_call():
                grouped_gemm_nt_f8f8bf16_contig((q, s_al), (W, WS), out_c, mi)

            # numerics check vs per-expert reference (first 128 rows of 3 experts)
            try:
                es_call(); dg_call()
                torch.cuda.synchronize()
                ref_ok, ref_rel = True, 0.0
                for e in (0, E_LOCAL // 2, E_LOCAL - 1):
                    c = sizes[e]
                    if c == 0:
                        continue
                    st_e = starts[e]
                    ref = (a_es[st_e:st_e + min(c, 64)].float()
                           * s_es[st_e:st_e + min(c, 64)].repeat_interleave(128, 1).float()
                           ) @ (
                        W[e].float()
                        * WS[e].repeat_interleave(128, 0).repeat_interleave(128, 1).float()
                    ).t()
                    got = out_es[st_e:st_e + min(c, 64)].float()
                    rel = ((got - ref).norm() / ref.norm().clamp(min=1e-6)).item()
                    ref_rel = max(ref_rel, rel)
                    ref_ok = ref_ok and rel < 0.05
            except Exception as ex:
                print(f"  {gemm_name}: es call FAILED {type(ex).__name__}: {ex}", flush=True)
                ref_ok, ref_rel = False, -1.0

            for _ in range(5):
                es_call(); dg_call()
            torch.cuda.synchronize()
            t_es, t_dg = [], []
            for _ in range(15):
                t_es.append(time_one(es_call))
                t_dg.append(time_one(dg_call))
            p_es, p_dg = statistics.median(t_es), statistics.median(t_dg)
            fl_real = 2.0 * real * N * K
            rec = dict(mode=mode, dp_tokens=dp_tokens, gemm=gemm_name, real=real,
                       M_pad=M_pad, pad_frac=round((M_pad - real) / M_pad, 3),
                       cutlass_us=round(p_es, 1), deepgemm_us=round(p_dg, 1),
                       cutlass_tf_real=round(fl_real / (p_es * 1e-6) / 1e12, 1),
                       deepgemm_tf_real=round(fl_real / (p_dg * 1e-6) / 1e12, 1),
                       cutlass_pct=round(fl_real / (p_es * 1e-6) / 1e12 / CEIL * 100, 1),
                       deepgemm_pct=round(fl_real / (p_dg * 1e-6) / 1e12 / CEIL * 100, 1),
                       numerics_rel=round(ref_rel, 5), numerics_ok=ref_ok)
            out.append(rec)
            print(f"  {gemm_name:>8s}: cutlass {p_es:8.1f}us ({rec['cutlass_tf_real']:7.1f} TF/s real, "
                  f"{rec['cutlass_pct']:4.1f}%) vs deepgemm {p_dg:8.1f}us "
                  f"({rec['deepgemm_tf_real']:7.1f}, {rec['deepgemm_pct']:4.1f}%) "
                  f"numerics_rel={ref_rel:.4f}", flush=True)
            del x, q, s_row, s_al, mi, out_c, a_es, s_es, out_es
            torch.cuda.empty_cache()

    with open("bench_cutlass.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote bench_cutlass.json")


if __name__ == "__main__":
    main()
