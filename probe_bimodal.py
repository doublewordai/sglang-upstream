"""Probe gate_up grouped-GEMM bimodality: per-iteration durations, back-to-back
vs sleep-separated, and contig vs dense alternated. Also clock rate sampling.
"""
import sys, time
import torch
from bench_moe_layer import load_layer, route, DEV, FP8, H, I, E_LOCAL, ALIGN

mode = sys.argv[1] if len(sys.argv) > 1 else "router"
N, K = 2 * I, H

from sglang.srt.layers.deep_gemm_wrapper import grouped_gemm_nt_f8f8bf16_contig
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8, w8a8_block_fp8_matmul_deepgemm)
from sglang.kernels.ops.moe.ep_moe_kernels import tma_align_input_scale

w = load_layer(5)
w13 = torch.stack([torch.cat([w[f"experts.{e}.gate_proj.weight"],
                              w[f"experts.{e}.up_proj.weight"]], 0)
                   for e in range(E_LOCAL)]).to(DEV)
w13_s = torch.stack([torch.cat([w[f"experts.{e}.gate_proj.weight_scale_inv"],
                                w[f"experts.{e}.up_proj.weight_scale_inv"]], 0)
                     for e in range(E_LOCAL)]).to(DEV)
gw = w["gate.weight"].to(DEV).float(); gb = w["gate.e_score_correction_bias"].to(DEV)
del w; torch.cuda.empty_cache()

T = 8192 * 4
h = torch.randn(T, H, device=DEV, dtype=torch.bfloat16) * 0.5
ti, _ = route(h, gw, gb, mode)
local = ti < E_LOCAL
cnt = torch.zeros(E_LOCAL, dtype=torch.int64, device=DEV)
cnt.scatter_add_(0, torch.where(local, ti, 0).reshape(-1), local.reshape(-1).long())
cnt = cnt.cpu(); del h, ti, local
aligned = (cnt + ALIGN - 1) // ALIGN * ALIGN
M_pad = int(aligned.sum()); real = int(cnt.sum())
print(f"mode={mode} M_pad={M_pad} real={real}")

x = torch.randn(M_pad, K, device=DEV, dtype=torch.bfloat16)
q, s_row = sglang_per_token_group_quant_fp8(x, 128, column_major_scales=False,
                                            scale_tma_aligned=False)
s_al = tma_align_input_scale(s_row)
mi = torch.repeat_interleave(torch.arange(E_LOCAL, dtype=torch.int32), aligned).to(DEV)
out = torch.empty(M_pad, N, device=DEV, dtype=torch.bfloat16)
dq = torch.randn(real, K, device=DEV, dtype=torch.bfloat16)
dq, ds = sglang_per_token_group_quant_fp8(dq, 128, column_major_scales=True,
                                          scale_tma_aligned=True)

def timed_call(fn):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record(); fn(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0

# warmup
for _ in range(5):
    grouped_gemm_nt_f8f8bf16_contig((q, s_al), (w13, w13_s), out, mi)
    w8a8_block_fp8_matmul_deepgemm(dq, w13[0], ds, w13_s[0], [128, 128], torch.bfloat16)
torch.cuda.synchronize()

print("\n-- 30 back-to-back contig calls, per-iter us:")
durs = [timed_call(lambda: grouped_gemm_nt_f8f8bf16_contig((q, s_al), (w13, w13_s), out, mi))
        for _ in range(30)]
print(" ".join(f"{d:.0f}" for d in durs))

print("\n-- 20 sleep-separated (10ms) contig calls:")
durs2 = []
for _ in range(20):
    time.sleep(0.01)
    durs2.append(timed_call(lambda: grouped_gemm_nt_f8f8bf16_contig((q, s_al), (w13, w13_s), out, mi)))
print(" ".join(f"{d:.0f}" for d in durs2))

print("\n-- 20 alternating contig / dense (same M=real):")
alt = []
for _ in range(20):
    alt.append(("c", timed_call(lambda: grouped_gemm_nt_f8f8bf16_contig((q, s_al), (w13, w13_s), out, mi))))
    alt.append(("d", timed_call(lambda: w8a8_block_fp8_matmul_deepgemm(dq, w13[0], ds, w13_s[0], [128, 128], torch.bfloat16))))
print(" ".join(f"{k}:{d:.0f}" for k, d in alt))

print("\n-- clocks.sm sampled during load (separate thread via nvidia-smi not avail; "
      "torch.cuda.clock_rate after each 5 calls):")
durs3 = []
for i in range(10):
    durs3.append(timed_call(lambda: grouped_gemm_nt_f8f8bf16_contig((q, s_al), (w13, w13_s), out, mi)))
    try:
        print(f"iter {i}: {durs3[-1]:.0f}us clock={torch.cuda.clock_rate(0)}")
    except Exception as e:
        print(f"iter {i}: {durs3[-1]:.0f}us clock_rate n/a ({e})")
        break
