"""Calibrate the FP8 ceiling (cuBLASLt via torch._scaled_mm 8192^3 + down-shape
dense) in sustained vs burst (10 ms sleep-separated) clock regimes."""
import time
import torch
from bench_moe_layer import DEV

torch.manual_seed(0)

def bench(fn, n=20, gap=0.0):
    evs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n)]
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    for s, e in evs:
        if gap:
            time.sleep(gap)
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    us = sorted(s.elapsed_time(e) * 1000 for s, e in evs)
    return us[len(us) // 2], us[-1]

for name, M, N, K in (("cublaslt_8192cube", 8192, 8192, 8192),
                      ("dense_down_shape", 65536, 6144, 2048),
                      ("dense_gateup_shape", 65536, 4096, 6144)):
    a = torch.randn(M, K, device=DEV).div(16).to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device=DEV).div(16).to(torch.float8_e4m3fn).t()  # [K,N] col-major
    sa = torch.ones(1, device=DEV); sb = torch.ones(1, device=DEV)
    def fn():
        torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    p50s, p90s = bench(fn, gap=0.0)
    p50b, p90b = bench(fn, gap=0.01)
    fl = 2.0 * M * N * K
    print(f"{name:>20s} M={M} N={N} K={K}: sustained p50 {p50s:8.1f}us "
          f"({fl/p50s/1e6:7.1f} TF/s, p90 {p90s:7.1f}) | burst p50 {p50b:8.1f}us "
          f"({fl/p50b/1e6:7.1f} TF/s, p90 {p90b:7.1f})", flush=True)
    del a, b
    torch.cuda.empty_cache()

try:
    print("clock now:", torch.cuda.clock_rate(0))
except Exception as e:
    print("clock n/a", e)
