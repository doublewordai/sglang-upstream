/*
 * dgf_norm_quant.cuh — fused (add-)RMSNorm + per-token-group-128 fp8 quant
 * (lane decode-glue-fusion; TRT-LLM FusedRMSNormQuant shape, GLM-5.3 dims).
 *
 * One CTA per token row; the row is staged in shared memory as fp32 between the
 * norm phase and the quant phase. Variants (template flags):
 *   kAdd    : residual = residual + x, norm over the sum (FusedAddRMSNorm semantics)
 *   kMoeIn  : residual = residual + shared + alpha * x   (MoE combine epilogue:
 *             replaces [sh_out.add_(routed, alpha); next-layer fused_add_rmsnorm;
 *             quant]) — emit both h and the next GEMM's fp8 directly
 *   neither : plain RMSNorm of x (q_a_layernorm site; no residual write)
 *   kDualScale: also emit the row-major [T, K/128] scales (megakernel input layout)
 *
 * The quant stage replicates the production per_token_group_quant arithmetic
 * exactly given identical h bytes: bf16-domain group amax, amax = max(amax,
 * 1e-10) in fp32, stored scale = amax/448, q = satfinite(h_f32 * 448/amax).
 * The norm stage is a standard fp32 block reduction — it can differ from the
 * production flashinfer CuTe norm kernel by 1 bf16 ulp on ~1e-5 of elements
 * (reduction-tree order); the engine-level exactness gate is greedy-token
 * equality within the run-to-run envelope.
 *
 * Scales: col-major TMA-aligned fp32 buffer [K/128, T_pad4] (the production
 * deep_gemm quant layout: logical [T, K/128], strides (1, T_pad4)); when
 * kDualScale, also a row-major [T, K/128] fp32 buffer.
 */
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

using namespace device;

template <int kK, int kBlockSize, bool kAdd, bool kMoeIn, bool kDualScale, bool kSkipH, bool kUsePDL>
__global__ __launch_bounds__(kBlockSize, 1) void dgf_norm_quant_kernel(
    bf16_t* __restrict__ out_h,        // [T, kK] normed bf16 h
    bf16_t* __restrict__ residual,     // [T, kK] in/out residual (kAdd/kMoeIn)
    bf16_t const* __restrict__ x,      // [T, kK] input (routed for kMoeIn)
    bf16_t const* __restrict__ shared, // [T, kK] (kMoeIn only)
    bf16_t const* __restrict__ weight, // [kK] rmsnorm weight
    fp8_e4m3_t* __restrict__ out_q,    // [T, kK]
    float* __restrict__ out_s_col,     // [kK/128, t_pad4] col-major TMA-aligned
    float* __restrict__ out_s_row,     // [T, kK/128] row-major (kDualScale)
    float const alpha,                 // routed scaling (kMoeIn)
    int const num_tokens,
    int const t_pad4,
    int64_t const stride_x,             // row stride of x (strided views ok)
    float const eps) {
  constexpr int kGroups = kK / 128;
  constexpr int kWarps = kBlockSize / 32;
  constexpr int kGroupsPerWarp = (kGroups + kWarps - 1) / kWarps;
  constexpr int kElems = kGroupsPerWarp * 4;  // per thread
  static_assert(kK % 128 == 0);
  static_assert(kGroupsPerWarp * 4 * 32 * kWarps == kK, "thread mapping must tile K exactly");

  int const tid = threadIdx.x;
  int const token = blockIdx.x;
  if (token >= num_tokens) return;

  PDLWaitPrimary<kUsePDL>();

  bf16_t const* x_row = x + (int64_t)token * stride_x;
  bf16_t* res_row = residual + (int64_t)token * kK;
  bf16_t* h_row = out_h + (int64_t)token * kK;
  bf16_t const* sh_row = shared + (int64_t)token * kK;
  fp8_e4m3_t* q_row = out_q + (int64_t)token * kK;

  int const warp = tid >> 5;
  int const lane = tid & 31;

  // ---- one pass: load into registers, partial sum_sq ----
  float v[kElems];
  float sum_sq = 0.f;
  {
    float const alpha_f = alpha;
#pragma unroll
    for (int gg = 0; gg < kGroupsPerWarp; ++gg) {
      int const g = warp * kGroupsPerWarp + gg;
      int const base = g * 128 + lane * 4;
      AlignedVector<bf16_t, 4> xv, rv, sv, wv;
      xv.load(x_row + base);
      wv.load(weight + base);
      if constexpr (kMoeIn) {
        rv.load(res_row + base);
        sv.load(sh_row + base);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          // production chain: moe_out = bf16(sh + alpha*x); res' = bf16(res + moe_out)
          float v1 = cast<float>(sv[j]) + alpha_f * cast<float>(xv[j]);
          bf16_t v1r = cast<bf16_t>(v1);
          float vv = cast<float>(rv[j]) + cast<float>(v1r);
          v[gg * 4 + j] = vv;
          sum_sq += vv * vv;
        }
      } else if constexpr (kAdd) {
        rv.load(res_row + base);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float vv = cast<float>(rv[j]) + cast<float>(xv[j]);
          v[gg * 4 + j] = vv;
          sum_sq += vv * vv;
        }
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float vv = cast<float>(xv[j]);
          v[gg * 4 + j] = vv;
          sum_sq += vv * vv;
        }
      }
      (void)wv;  // weight re-loaded below (kept in L1)
    }
  }

  // block reduce sum_sq
  __shared__ float sm_red[kWarps];
  float wsum = warp::reduce_sum(sum_sq);
  if (lane == 0) sm_red[warp] = wsum;
  __syncthreads();
  float total = 0.f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) total += sm_red[w];
  float const rsqrt_scale = rsqrtf(total / (float)kK + eps);

  // ---- per-group: h + residual write + amax + quant (registers) ----
#pragma unroll
  for (int gg = 0; gg < kGroupsPerWarp; ++gg) {
    int const g = warp * kGroupsPerWarp + gg;
    int const base = g * 128 + lane * 4;
    AlignedVector<bf16_t, 4> wv, hv, rv;
    wv.load(weight + base);
    bf16_t h4[4];
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float h = v[gg * 4 + j] * rsqrt_scale * cast<float>(wv[j]);
      h4[j] = cast<bf16_t>(h);
      amax = fmaxf(amax, fabsf(cast<float>(h4[j])));
    }
    float gmax = warp::reduce_max(amax);
    gmax = fmaxf(gmax, 1e-10f);
    float const scale_inv = gmax * (1.f / 448.f);
    float const qscale = 448.f / gmax;
    if constexpr (kAdd || kMoeIn) {
#pragma unroll
      for (int j = 0; j < 4; ++j) rv[j] = cast<bf16_t>(v[gg * 4 + j]);
      rv.store(res_row + base);
    }
    if constexpr (!kSkipH) {
#pragma unroll
      for (int j = 0; j < 4; ++j) hv[j] = h4[j];
      hv.store(h_row + base);
    }
    {
      fp8_e4m3_t q4[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float qv = cast<float>(h4[j]) * qscale;
        q4[j] = cast<fp8_e4m3_t>(fminf(qv, 448.f));
      }
      AlignedVector<fp8_e4m3_t, 4> qvec;
#pragma unroll
      for (int j = 0; j < 4; ++j) qvec[j] = q4[j];
      qvec.store(q_row + base);
    }
    if (lane == 0) {
      out_s_col[(int64_t)g * t_pad4 + token] = scale_inv;
      if constexpr (kDualScale) {
        out_s_row[(int64_t)token * kGroups + g] = scale_inv;
      }
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int kK, bool kAdd, bool kMoeIn, bool kDualScale, bool kSkipH, bool kUsePDL>
struct DGFNormQuantKernel {
  static void run(
      const tvm::ffi::TensorView out_h,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView shared,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView out_q,
      const tvm::ffi::TensorView out_s_col,
      const tvm::ffi::TensorView out_s_row,
      double alpha,
      int64_t num_tokens,
      int64_t t_pad4,
      double eps) {
    using namespace host;
    auto T = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    if constexpr (kSkipH) {
      RuntimeCheck(out_h.numel() >= 1, "skip_h: out_h is a dummy");
    } else {
      TensorMatcher({T, kK}).with_dtype<bf16_t>().with_device(device).verify(out_h);
    }
    auto Sr = SymbolicSize{"stride_res"};
    TensorMatcher({T, kK}).with_strides({Sr, 1}).with_dtype<bf16_t>().with_device(device).verify(residual);
    auto Sx = SymbolicSize{"stride_x"};
    TensorMatcher({T, kK}).with_strides({Sx, 1}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({T, kK}).with_dtype<bf16_t>().with_device(device).verify(shared);
    TensorMatcher({kK}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({T, kK}).with_dtype<fp8_e4m3_t>().with_device(device).verify(out_q);
    TensorMatcher({kK / 128, t_pad4}).with_dtype<fp32_t>().with_device(device).verify(out_s_col);
    if constexpr (kDualScale) {
      TensorMatcher({T, kK / 128}).with_dtype<fp32_t>().with_device(device).verify(out_s_row);
    } else {
      RuntimeCheck(out_s_row.numel() >= 1, "dummy out_s_row");
    }
    RuntimeCheck(num_tokens == (int64_t)T.unwrap(), "num_tokens mismatch");

    constexpr int kBlockSize = 512;
    constexpr auto kernel =
        dgf_norm_quant_kernel<kK, kBlockSize, kAdd, kMoeIn, kDualScale, kSkipH, kUsePDL>;
    LaunchKernel((unsigned)num_tokens, kBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(
            kernel,
            static_cast<bf16_t*>(out_h.data_ptr()),
            static_cast<bf16_t*>(residual.data_ptr()),
            static_cast<bf16_t const*>(x.data_ptr()),
            static_cast<bf16_t const*>(shared.data_ptr()),
            static_cast<bf16_t const*>(weight.data_ptr()),
            static_cast<fp8_e4m3_t*>(out_q.data_ptr()),
            static_cast<float*>(out_s_col.data_ptr()),
            static_cast<float*>(out_s_row.data_ptr()),
            (float)alpha,
            (int)num_tokens,
            (int)t_pad4,
            (int64_t)Sx.unwrap(),
            (float)eps);
  }
};

}  // namespace sglang
