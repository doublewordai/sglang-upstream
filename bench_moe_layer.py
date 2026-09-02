"""Standalone prefill-MoE layer driver (1 GPU, EP4 rank view), real GLM-5.3 weights.

Reconstructs ONE EP rank's received view for one 8192-token chunk per DP rank
(production: 4 DP ranks busy -> 32768 global tokens, top-8 over 256 experts,
64 local experts -> ~65536 rows/rank/layer, 128-row expert alignment padding),
then runs the EXACT production kernel sequence of the deepep-normal prefill
path (per sglang-prefill-moe-0902 tree, moe_runner/deep_gemm.py
_run_contiguous_gemm + token_dispatcher/deepep.py dispatch_a):

  dispatch-input quant (K=6144, row-major scales)
  [a2a comm not simulated - 1 GPU]
  ep_scatter (contiguous layout + m_indices, 128-aligned)
  tma_align_input_scale
  grouped_gemm_nt_f8f8bf16_contig  gate_up  (N=4096, K=6144, G=64)
  act_and_mul (legacy silu_and_mul)
  per_token_group_quant_fp8 (K=2048, row-major) + tma_align
  grouped_gemm_nt_f8f8bf16_contig  down    (N=6144, K=2048, G=64)
  ep_gather (topk-weighted unpermute)
  shared expert: quant(K=6144, col+tma) + dense deepgemm gate_up + act_and_mul
                 + quant(K=2048, col+tma) + dense deepgemm down  (8192 local toks)
  router gate GEMM (fp32) + noaux_tc topk (torch reference; real one is fused triton)

Timing: per-kernel CUDA events, 5 warmup + 25 timed iters, p50/p90 us.
Also reports per-kernel FLOPs/bytes and TF/s vs measured ceilings
(1305 TF/s FP8 cuBLASLt, 3.665 TB/s HBM copy on this GH200).

Usage: python bench_moe_layer.py --mode router|uniform --dp-tokens 8192
       --out results_m1.json
"""
import argparse, json, math, os, time
import torch

torch.manual_seed(1234)
DEV = "cuda"
SNAP = "/projects/s6p/hf/hub/models--zai-org--GLM-5.3/snapshots/e0b07fd2751b42d5efa199cc02c2b271deadc516"
FP8 = torch.float8_e4m3fn
H = 6144           # hidden
I = 2048           # moe_intermediate
E_GLOBAL = 256
E_LOCAL = 64       # EP4
TOPK = 8
ALIGN = 128        # deepep expert_alignment (JIT deepgemm)

CEIL_FP8 = 1305.0   # TF/s, cuBLASLt 8192^3, measured 2026-09-02
CEIL_BW = 3.665     # TB/s HBM copy


def load_layer(layer: int, n_local: int = E_LOCAL):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    pfx = f"model.layers.{layer}.mlp."
    names = [pfx + "gate.weight", pfx + "gate.e_score_correction_bias"]
    for e in range(n_local):
        for p in ("gate_proj", "up_proj", "down_proj"):
            names.append(pfx + f"experts.{e}.{p}.weight")
            names.append(pfx + f"experts.{e}.{p}.weight_scale_inv")
    for p in ("gate_proj", "up_proj", "down_proj"):
        names.append(pfx + f"shared_experts.{p}.weight")
        names.append(pfx + f"shared_experts.{p}.weight_scale_inv")
    shards = sorted({wm[n] for n in names})
    out = {}
    for shard in shards:
        with safe_open(os.path.join(SNAP, shard), framework="pt", device="cpu") as f:
            for n in names:
                if wm[n] == shard:
                    out[n[len(pfx):]] = f.get_tensor(n)
    return out


def route(h_global, gate_w, gate_bias, mode):
    """noaux_tc topk (GLM-5.3): sigmoid scores + bias, top-8, renorm."""
    logits = torch.nn.functional.linear(h_global.float(), gate_w.float())  # [T,256]
    if mode == "uniform":
        logits = torch.randn_like(logits)
    scores = torch.sigmoid(logits) + gate_bias.float().unsqueeze(0)
    topk_w, topk_i = torch.topk(scores, TOPK, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    return topk_i, topk_w


def timed(fn, warmup=5, iters=25, loop=1):
    """CUDA-event timing; `loop` back-to-back calls per event pair (amortizes
    launch/alloc gaps for small kernels); returns (p50_us, p90_us) per call."""
    evs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    for _ in range(warmup * loop):
        fn()
    torch.cuda.synchronize()
    for s, e in evs:
        s.record()
        for _ in range(loop):
            fn()
        e.record()
    torch.cuda.synchronize()
    us = sorted(s.elapsed_time(e) * 1000.0 / loop for s, e in evs)
    return us[len(us) // 2], us[int(len(us) * 0.9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--mode", default="router", choices=["router", "uniform"])
    ap.add_argument("--dp-tokens", type=int, default=8192, help="tokens per DP rank")
    ap.add_argument("--dp-ranks", type=int, default=4)
    ap.add_argument("--skew-temper", type=float, default=0.0,
                    help=">0: lognormal temper of gate rows to force skew")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--out", default="results_m1.json")
    args = ap.parse_args()

    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8, w8a8_block_fp8_matmul_deepgemm)
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.ep_moe_kernels import (
        ep_scatter, ep_gather, tma_align_input_scale)
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.deep_gemm_wrapper import grouped_gemm_nt_f8f8bf16_contig
    assert deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM, "JIT deepgemm must be on"
    assert not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0

    t0 = time.time()
    w = load_layer(args.layer)
    print(f"loaded layer {args.layer} weights in {time.time()-t0:.1f}s", flush=True)

    # --- routed expert weights: w13 [64, 2I, H] fp8 (gate rows then up rows), w2 [64, H, I]
    w13 = torch.stack([torch.cat([w[f"experts.{e}.gate_proj.weight"],
                                  w[f"experts.{e}.up_proj.weight"]], 0)
                       for e in range(E_LOCAL)]).to(DEV)
    w13_s = torch.stack([torch.cat([w[f"experts.{e}.gate_proj.weight_scale_inv"],
                                    w[f"experts.{e}.up_proj.weight_scale_inv"]], 0)
                         for e in range(E_LOCAL)]).to(DEV)
    w2 = torch.stack([w[f"experts.{e}.down_proj.weight"] for e in range(E_LOCAL)]).to(DEV)
    w2_s = torch.stack([w[f"experts.{e}.down_proj.weight_scale_inv"]
                        for e in range(E_LOCAL)]).to(DEV)
    # shared expert (dense): gate first then up, matching MergedColumnParallel
    sh_w13 = torch.cat([w["shared_experts.gate_proj.weight"],
                        w["shared_experts.up_proj.weight"]], 0).to(DEV)
    sh_w13_s = torch.cat([w["shared_experts.gate_proj.weight_scale_inv"],
                          w["shared_experts.up_proj.weight_scale_inv"]], 0).to(DEV)
    sh_w2 = w["shared_experts.down_proj.weight"].to(DEV)
    sh_w2_s = w["shared_experts.down_proj.weight_scale_inv"].to(DEV)
    gate_w = w["gate.weight"].to(DEV).float()
    gate_bias = w["gate.e_score_correction_bias"].to(DEV)
    del w
    torch.cuda.empty_cache()

    T_global = args.dp_tokens * args.dp_ranks
    h_local = torch.randn(args.dp_tokens, H, device=DEV, dtype=torch.bfloat16) * 0.5
    h_global = torch.randn(T_global, H, device=DEV, dtype=torch.bfloat16) * 0.5
    if args.skew_temper > 0:
        g = torch.Generator(device="cpu").manual_seed(7)
        temper = torch.logspace(-1, 1, E_GLOBAL).pow(args.skew_temper)
        temper = (temper / temper.mean()).to(DEV, torch.float32)
        gate_w = gate_w * temper.unsqueeze(1)

    # --- routing: topk over 256 for all global tokens; local view = experts 0..63
    topk_i, topk_w = route(h_global, gate_w, gate_bias, args.mode)
    local_sel = (topk_i < E_LOCAL)              # [T, 8] bool
    n_pairs = int(local_sel.sum())
    recv_mask = local_sel.any(dim=1)            # tokens with >=1 local expert
    recv_tokens = int(recv_mask.sum())
    # per-expert valid counts
    cnt = torch.zeros(E_LOCAL, dtype=torch.int64, device=DEV)
    cnt.scatter_add_(0, torch.where(local_sel, topk_i, 0).reshape(-1),
                     local_sel.reshape(-1).long())
    cnt_cpu = cnt.cpu()
    aligned = ((cnt_cpu + ALIGN - 1) // ALIGN * ALIGN)
    M_pad = int(aligned.sum())
    real_rows = int(cnt_cpu.sum())
    skew = dict(mean=float(cnt_cpu.float().mean()), std=float(cnt_cpu.float().std()),
                mx=int(cnt_cpu.max()), mn=int(cnt_cpu.min()),
                pad_rows=M_pad - real_rows, pad_frac=(M_pad - real_rows) / M_pad)
    print(f"routing[{args.mode}]: recv_tokens={recv_tokens}/{T_global} pairs={n_pairs} "
          f"(exp {T_global*TOPK*E_LOCAL/E_GLOBAL:.0f}) M_pad={M_pad} pad={skew['pad_frac']:.3f} "
          f"cnt mean={skew['mean']:.0f} std={skew['std']:.0f} max={skew['mx']} min={skew['mn']}",
          flush=True)

    # --- simulate the dispatch output (post-a2a) on this rank:
    recv_idx = torch.nonzero(recv_mask, as_tuple=True)[0]
    recv_x_bf16 = h_global[recv_idx]                       # [R, H] bf16
    # recv_topk: local expert id or -1 (deepep normal contract)
    recv_topk = torch.where(local_sel[recv_idx], topk_i[recv_idx],
                            torch.full_like(topk_i[recv_idx], -1)).to(torch.int64)
    recv_topk_w = topk_w[recv_idx].to(torch.float32)

    # --- tensors for the runner (allocated once; production allocates per step)
    input_tensor = torch.empty(M_pad, H, device=DEV, dtype=FP8)
    input_scale = torch.empty(M_pad, H // 128, device=DEV, dtype=torch.float32)
    m_indices = torch.empty(M_pad, device=DEV, dtype=torch.int32)
    output_index = torch.empty_like(recv_topk)
    gateup_out = torch.empty(M_pad, 2 * I, device=DEV, dtype=torch.bfloat16)
    down_in = torch.empty(M_pad, I, device=DEV, dtype=torch.bfloat16)
    down_out = torch.empty(M_pad, H, device=DEV, dtype=torch.bfloat16)
    gather_out = torch.empty(recv_tokens, H, device=DEV, dtype=torch.bfloat16)
    cnt_gpu = aligned.to(torch.int32).to(DEV)
    start_loc = torch.empty_like(cnt_gpu)

    res = dict(layer=args.layer, mode=args.mode, dp_tokens=args.dp_tokens,
               dp_ranks=args.dp_ranks, skew=skew, recv_tokens=recv_tokens,
               pairs=real_rows, M_pad=M_pad, kernels={})

    def rec(name, p50, p90, flops=0.0, bytes_=0.0, note=""):
        tf = flops / (p50 * 1e-6) / 1e12 if flops else 0.0
        tb = bytes_ / (p50 * 1e-6) / 1e12 if bytes_ else 0.0
        res["kernels"][name] = dict(p50_us=p50, p90_us=p90, flops=flops,
                                    bytes=bytes_, tflops=tf, tbps=tb, note=note)
        print(f"{name:>28s}: p50 {p50:9.1f} us  p90 {p90:9.1f}  "
              + (f" {tf:7.1f} TF/s ({tf/CEIL_FP8*100:4.1f}% of {CEIL_FP8:.0f})" if flops else "")
              + (f"  {tb:6.2f} TB/s ({tb/CEIL_BW*100:4.1f}% of {CEIL_BW})" if bytes_ else ""),
              flush=True)

    MB = 1024 * 1024

    # ---- 1. dispatch-input quant (local dp_tokens rows, K=6144, row-major scale)
    p50, p90 = timed(lambda: sglang_per_token_group_quant_fp8(
        h_local, 128, column_major_scales=False, scale_tma_aligned=False),
        iters=args.iters, loop=10)
    rec("quant_dispatch_in", p50, p90,
        bytes_=args.dp_tokens * H * 2 + args.dp_tokens * H + args.dp_tokens * (H // 128) * 4)

    # quantize recv tokens once (input to ep_scatter; also the "sent" payload)
    recv_x_q, recv_x_s = sglang_per_token_group_quant_fp8(
        recv_x_bf16, 128, column_major_scales=False, scale_tma_aligned=False)

    # ---- 2. ep_scatter (ep_scatter_1 recomputes start_loc each call)
    def do_scatter():
        ep_scatter(recv_x_q, recv_x_s, recv_topk, cnt_gpu, cnt_gpu, start_loc,
                   input_tensor, input_scale, m_indices, output_index)
    p50, p90 = timed(do_scatter, iters=args.iters, loop=5)
    rec("ep_scatter", p50, p90,
        bytes_=recv_tokens * H + real_rows * H
               + (recv_tokens + real_rows) * (H // 128) * 4
               + M_pad * 4 + recv_tokens * TOPK * 8)

    # ---- 3. tma_align #1
    p50, p90 = timed(lambda: tma_align_input_scale(input_scale), iters=args.iters, loop=10)
    rec("tma_align_1", p50, p90,
        bytes_=2 * M_pad * (H // 128) * 4)
    in_scale_a = tma_align_input_scale(input_scale)

    # ---- 4. gate_up grouped GEMM
    p50, p90 = timed(lambda: grouped_gemm_nt_f8f8bf16_contig(
        (input_tensor, in_scale_a), (w13, w13_s), gateup_out, m_indices),
        iters=args.iters)
    fl = 2.0 * M_pad * (2 * I) * H
    by = M_pad * H + M_pad * (H // 128) * 4 + E_LOCAL * (2 * I) * H + M_pad * (2 * I) * 2
    rec("gemm_gateup_grouped", p50, p90, flops=fl, bytes_=by)

    # ---- 5. act_and_mul (legacy)
    p50, p90 = timed(lambda: silu_and_mul(gateup_out, down_in), iters=args.iters, loop=5)
    rec("act_and_mul", p50, p90, bytes_=M_pad * (2 * I) * 2 + M_pad * I * 2)

    # ---- 6. down-input quant + tma_align
    p50, p90 = timed(lambda: sglang_per_token_group_quant_fp8(
        down_in, 128, column_major_scales=False, scale_tma_aligned=False),
        iters=args.iters, loop=10)
    rec("quant_down_in", p50, p90, bytes_=M_pad * I * 2 + M_pad * I + M_pad * (I // 128) * 4)
    din_q, din_s = sglang_per_token_group_quant_fp8(
        down_in, 128, column_major_scales=False, scale_tma_aligned=False)
    p50, p90 = timed(lambda: tma_align_input_scale(din_s), iters=args.iters, loop=10)
    rec("tma_align_2", p50, p90, bytes_=2 * M_pad * (I // 128) * 4)
    din_s_a = tma_align_input_scale(din_s)

    # ---- 7. down grouped GEMM
    p50, p90 = timed(lambda: grouped_gemm_nt_f8f8bf16_contig(
        (din_q, din_s_a), (w2, w2_s), down_out, m_indices), iters=args.iters)
    fl = 2.0 * M_pad * H * I
    by = M_pad * I + M_pad * (I // 128) * 4 + E_LOCAL * H * I + M_pad * H * 2
    rec("gemm_down_grouped", p50, p90, flops=fl, bytes_=by)

    # ---- 8. ep_gather
    p50, p90 = timed(lambda: ep_gather(down_out, recv_topk, recv_topk_w,
                                          output_index, gather_out), iters=args.iters, loop=5)
    rec("ep_gather", p50, p90,
        bytes_=real_rows * H * 2 + recv_tokens * H * 2 + recv_tokens * TOPK * 8)

    # ---- 9. shared expert (dense path, local dp_tokens rows)
    sh_gu = torch.empty(args.dp_tokens, 2 * I, device=DEV, dtype=torch.bfloat16)
    sh_dn = torch.empty(args.dp_tokens, H, device=DEV, dtype=torch.bfloat16)
    sh_act = torch.empty(args.dp_tokens, I, device=DEV, dtype=torch.bfloat16)
    p50, p90 = timed(lambda: sglang_per_token_group_quant_fp8(
        h_local, 128, column_major_scales=True, scale_tma_aligned=True), iters=args.iters)
    rec("sh_quant_in", p50, p90, bytes_=args.dp_tokens * H * 2 + args.dp_tokens * H)
    q, s = sglang_per_token_group_quant_fp8(h_local, 128, column_major_scales=True,
                                            scale_tma_aligned=True)
    p50, p90 = timed(lambda: w8a8_block_fp8_matmul_deepgemm(
        q, sh_w13, s, sh_w13_s, [128, 128], torch.bfloat16), iters=args.iters)
    fl = 2.0 * args.dp_tokens * (2 * I) * H
    by = args.dp_tokens * H + args.dp_tokens * (H // 128) * 4 + (2 * I) * H + args.dp_tokens * (2 * I) * 2
    rec("sh_gemm_gateup", p50, p90, flops=fl, bytes_=by)
    torch.cuda.synchronize()
    sh_gu = w8a8_block_fp8_matmul_deepgemm(q, sh_w13, s, sh_w13_s, [128, 128],
                                           torch.bfloat16)
    p50, p90 = timed(lambda: silu_and_mul(sh_gu, sh_act), iters=args.iters, loop=10)
    rec("sh_act_and_mul", p50, p90, bytes_=args.dp_tokens * (2 * I) * 2 + args.dp_tokens * I * 2)
    p50, p90 = timed(lambda: sglang_per_token_group_quant_fp8(
        sh_act, 128, column_major_scales=True, scale_tma_aligned=True), iters=args.iters)
    rec("sh_quant_down_in", p50, p90, bytes_=args.dp_tokens * I * 2 + args.dp_tokens * I)
    q2, s2 = sglang_per_token_group_quant_fp8(sh_act, 128, column_major_scales=True,
                                              scale_tma_aligned=True)
    p50, p90 = timed(lambda: w8a8_block_fp8_matmul_deepgemm(
        q2, sh_w2, s2, sh_w2_s, [128, 128], torch.bfloat16), iters=args.iters)
    fl = 2.0 * args.dp_tokens * H * I
    by = args.dp_tokens * I + args.dp_tokens * (I // 128) * 4 + H * I + args.dp_tokens * H * 2
    rec("sh_gemm_down", p50, p90, flops=fl, bytes_=by)

    # ---- 10. router gate GEMM (bf16 GEMM as in prod; logits math fp32)
    gate_w_bf16 = gate_w.to(torch.bfloat16)
    p50, p90 = timed(lambda: torch.nn.functional.linear(h_local, gate_w_bf16),
                     iters=args.iters, loop=10)
    rec("router_gate_gemm", p50, p90, flops=2.0 * args.dp_tokens * H * E_GLOBAL,
        bytes_=args.dp_tokens * H * 2 + E_GLOBAL * H * 4 + args.dp_tokens * E_GLOBAL * 4)

    # ---- 11. whole routed sequence (no per-kernel sync)
    def seq():
        ep_scatter(recv_x_q, recv_x_s, recv_topk, cnt_gpu, cnt_gpu, start_loc,
                   input_tensor, input_scale, m_indices, output_index)
        a = tma_align_input_scale(input_scale)
        grouped_gemm_nt_f8f8bf16_contig((input_tensor, a), (w13, w13_s), gateup_out, m_indices)
        silu_and_mul(gateup_out, down_in)
        dq, ds = sglang_per_token_group_quant_fp8(down_in, 128,
                                                  column_major_scales=False,
                                                  scale_tma_aligned=False)
        dsa = tma_align_input_scale(ds)
        grouped_gemm_nt_f8f8bf16_contig((dq, dsa), (w2, w2_s), down_out, m_indices)
        ep_gather(down_out, recv_topk, recv_topk_w, output_index, gather_out)
    p50, p90 = timed(seq, iters=args.iters)
    fl = 2.0 * M_pad * (2 * I) * H + 2.0 * M_pad * H * I
    by = (recv_tokens * H + real_rows * H + M_pad * H + E_LOCAL * (2 * I) * H
          + M_pad * (2 * I) * 2 + M_pad * I * 2 + M_pad * I + E_LOCAL * H * I
          + M_pad * H * 2 + real_rows * H * 2 + recv_tokens * H * 2)
    rec("RANKED_SEQUENCE", p50, p90, flops=fl, bytes_=by)

    res["gpu"] = torch.cuda.get_device_name(0)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
