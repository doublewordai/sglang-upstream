/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// SM90 native-fp8 sparse MLA DECODE attention kernel (split-KV + combine).
//
// Production semantics (see lane SPEC.md): q [b,1,64,576] bf16; KV pool rows
// 656 B = [fp8 e4m3 nope latent 512 | 4 x fp32 group scales | bf16 rope 64];
// indices [b,2048] i32 where NEGATIVE entries are masked (all others are
// scored); cache_seqlens only sizes the schedule. V = dequantized nope latent.
//
// Numerics design (vs the production bf16-dequant FlashMLA kernel):
//   * K/V consumed as RAW stored fp8 -- no requantization at all.
//   * QK: q_nope quantized to fp8 per (request, head) row (scale s_q applied
//     in the epilogue); the 4 per-128-group KV scales are applied EXACTLY on
//     the per-group WGMMA accumulators (fp32) before summing; the rope part
//     stays bf16 on both sides (exact; it carries most of the logit energy).
//   * PV: p_tilde_g = p * s_v[j,g] * 2^E_g quantized e4m3 per output group so
//     the stored V bytes are consumed raw; 2^E_g is a per-(partition, group)
//     power of two from the partition's max s_v (folded back out at the
//     partial store, so partials are scale-free).
//   * Split-KV partials (m, l, O) in fp32 + a separate combine kernel.
//
// v1a: synchronous (single-buffered K/Vt, __syncthreads between phases);
// correctness first, pipelining comes later.
//
// Thread layout: 3 warpgroups. WG0/WG1 consumers (each owns a 256-dim half of
// V = scale groups {0,1} / {2,3}); BOTH compute the full QK + online softmax
// for every block (identical, deterministic; QK FLOPs are trivial at decode)
// which avoids all cross-WG P exchanges. WG2 producer loads K/rope/scales via
// cp.async, transposes V into Vt, and quantizes q per-row into smem.

#pragma once

#include "config.h"
#include "helpers.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// using namespace cute must be at global scope BEFORE including dense_fp8
// headers (they use bare Tensor, make_tensor etc. from cute namespace).
using namespace cute;

#include "dense_fp8_transpose_v.h"
#include "dense_fp8_utils.h"

namespace sm90 {
namespace decode {

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 1) sparse_mla_fp8_decode_kernel(
    __grid_constant__ const SparseMlaFp8DecodeParams params) {
  Kernel::devfunc(params);
}

struct SparseMlaFp8DecodeKernel {
  static constexpr int B_H = 64;      // heads (M of QK)
  static constexpr int B_TOPK = 64;   // KV rows per block
  static constexpr int D_NOPE = 512;
  static constexpr int D_ROPE = 64;
  static constexpr int D_V = 512;
  static constexpr int NUM_GROUPS = D_NOPE / 128;  // 4 fp8 scale groups
  static constexpr int NUM_THREADS = 128 * 3;      // 2 consumer WGs + 1 producer WG
  static constexpr float MAX_INIT_VAL = -1e30f;
  static constexpr int ROW_BYTES = 656;
  static constexpr int SCALE_OFF = 512;
  static constexpr int ROPE_OFF = 528;
  // per-row 16B chunks: 32 fp8 + 1 scales + 8 rope = 41
  static constexpr int CHUNKS_PER_ROW = ROW_BYTES / 16;

  using fp8_t = cutlass::float_e4m3_t;
  using bf16_t = cutlass::bfloat16_t;

  // ------------------------------------------------------------------
  // Smem layouts
  // ------------------------------------------------------------------
  template <int N>
  using SmemLayoutQTiles = decltype(tile_to_shape(
      GMMA::Layout_K_SW64_Atom<fp8_t>{}, Shape<Int<B_H>, Int<64 * N>>{}, Step<_1, _2>{}));
  using SmemLayoutQ = SmemLayoutQTiles<D_NOPE / 64>;  // (64 h, 512 k) fp8

  using SmemLayoutQRope = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<bf16_t>{}, Shape<Int<B_H>, Int<D_ROPE>>{}, Step<_1, _2>{}));

  template <int N>
  using SmemLayoutKTiles = decltype(tile_to_shape(
      GMMA::Layout_K_SW64_Atom<fp8_t>{}, Shape<Int<B_TOPK>, Int<64 * N>>{}, Step<_1, _2>{}));
  using SmemLayoutK = SmemLayoutKTiles<D_NOPE / 64>;  // (64 j, 512 k) fp8

  using SmemLayoutKRope = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<bf16_t>{}, Shape<Int<B_TOPK>, Int<D_ROPE>>{}, Step<_1, _2>{}));

  // Vt group layout: (128 d, 64 j) K-major in j, contiguous 8 KB prefix of
  // the (256, 64) half layout written by the transpose utility.
  using SmemLayoutVtGroup = decltype(tile_to_shape(
      GMMA::Layout_K_SW64_Atom<fp8_t>{}, Shape<Int<128>, Int<B_TOPK>>{}, Step<_1, _2>{}));

  // MMA atoms
  using TiledMMA_QK = decltype(make_tiled_mma(
      GMMA::MMA_64x64x32_F32E4M3E4M3_SS_TN{}, Layout<Shape<_1, _1, _1>>{}));
  using TiledMMA_QK_Rope = decltype(make_tiled_mma(
      SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}, Layout<Shape<_1, _1, _1>>{}));
  using TiledMMA_PV = decltype(make_tiled_mma(
      GMMA::MMA_64x128x32_F32E4M3E4M3_RS_TN{}, Layout<Shape<_1, _1, _1>>{}));

  // V transpose utility (src layout == our SmemLayoutK)
  using SmemTransposeV = SmemTransposeFp8_64x64<B_TOPK, D_V>;
  using SmemLayoutTransposeV_t = typename SmemTransposeV::SmemLayoutTransposeV;
  // producer writes Vt halves through the STSM-composed transpose layout
  using SmemTransposeV_half_t = SmemTransposeFp8_64x64<B_TOPK, D_V / 2>;
  using SmemLayoutTransposeVt_half_t = typename SmemTransposeV_half_t::SmemLayoutTransposeVt;
  static constexpr int HALF_VT_ELEMS = cosize_v<typename SmemTransposeV_half_t::SmemLayoutVt>;  // (256,64) fp8 = 16 KB

  struct SharedMemoryPlan {
    array_aligned<fp8_t, cosize_v<SmemLayoutQ>> q;             // 32 KB
    array_aligned<bf16_t, cosize_v<SmemLayoutQRope>> q_rope;   // 8 KB
    array_aligned<fp8_t, cosize_v<SmemLayoutK>> k;             // 32 KB
    array_aligned<bf16_t, cosize_v<SmemLayoutKRope>> k_rope;   // 8 KB
    array_aligned<float, NUM_GROUPS * B_TOPK> k_scales;        // [g][j] 1 KB
    array_aligned<fp8_t, 2 * HALF_VT_ELEMS> vt;                // 2 x (256,64) = 32 KB
    array_aligned<bool, B_TOPK> valid;                         // index >= 0?
    array_aligned<int, B_TOPK> row_idx;                        // clamped row per block slot
    array_aligned<float, NUM_GROUPS> s2e;                      // 2^E_g
    array_aligned<float, NUM_GROUPS> inv2e;                    // 2^-E_g
    array_aligned<float, 128> qmax_scratch;                    // q per-row scale s_q[h]
    array_aligned<float, NUM_GROUPS> group_max_scratch;
  };

  // ====================================================================
  static void run(const SparseMlaFp8DecodeParams& params) {
    KU_ASSERT(params.num_heads == B_H);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.topk % B_TOPK == 0);
    KU_ASSERT(params.num_splits >= 1 && params.num_splits <= 1024);

    auto kernel = &sparse_mla_fp8_decode_kernel<SparseMlaFp8DecodeKernel>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
    dim3 grid(params.num_reqs, params.num_splits);
    kernel<<<grid, NUM_THREADS, smem_size, params.stream>>>(params);
    KU_CHECK_KERNEL_LAUNCH();
  }

  // ====================================================================
  static __device__ __forceinline__ void devfunc(const SparseMlaFp8DecodeParams& params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900))
    const int req = blockIdx.x;
    const int part = blockIdx.y;
    const int P = params.num_splits;
    const int nblocks_max = params.topk / B_TOPK;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    // number of index blocks with possible content
    int nblocks = nblocks_max;
    if (params.tail_sentinel) {
      int sl = max(0, __ldg(params.seqlens + req));
      nblocks = min(nblocks_max, (sl + B_TOPK - 1) / B_TOPK);
    }

    // ----------------------------------------------------------------
    // Producer WG: E pre-pass, q load + per-row fp8 quantization
    // ----------------------------------------------------------------
    if (warpgroup_idx == 2) {
      // ---- E pre-pass: per-group max of s_v over the partition's rows ----
      if (idx_in_warpgroup == 0) {
        CUTE_UNROLL
        for (int g = 0; g < NUM_GROUPS; ++g) plan.group_max_scratch[g] = 0.f;
      }
      asm volatile("bar.sync 7, 128;\n" ::: "memory");
      for (int blk = part; blk < nblocks; blk += P) {
        const int* gIdx = params.indices + (int64_t)req * params.topk + blk * B_TOPK;
        for (int i = idx_in_warpgroup; i < B_TOPK; i += 128) {
          int t = __ldg(gIdx + i);
          if (t >= 0) {
            const float* sc = reinterpret_cast<const float*>(
                params.kv + (int64_t)t * ROW_BYTES + SCALE_OFF);
            float4 s4 = __ldg(reinterpret_cast<const float4*>(sc));
            atomicMax(reinterpret_cast<int*>(&plan.group_max_scratch[0]), __float_as_int(s4.x));
            atomicMax(reinterpret_cast<int*>(&plan.group_max_scratch[1]), __float_as_int(s4.y));
            atomicMax(reinterpret_cast<int*>(&plan.group_max_scratch[2]), __float_as_int(s4.z));
            atomicMax(reinterpret_cast<int*>(&plan.group_max_scratch[3]), __float_as_int(s4.w));
          }
        }
      }
      asm volatile("bar.sync 7, 128;\n" ::: "memory");
      if (idx_in_warpgroup == 0) {
        CUTE_UNROLL
        for (int g = 0; g < NUM_GROUPS; ++g) {
          float m = plan.group_max_scratch[g];
          // 2^E: s_max * 2^E <= 224 (half of e4m3 max 448: headroom for p <= 1)
          int e = 0;
          if (m > 0.f) {
            e = (int)floorf(log2f(224.f / m));
            e = max(0, min(30, e));
          }
          plan.s2e[g] = exp2f((float)e);
          plan.inv2e[g] = 1.0f / plan.s2e[g];
        }
      }

      // ---- q: per-row max over the nope 512 (bf16) ----
      // thread (h = iw % 64, half = iw / 64): nope 8-elem chunks k8 in [half*32, half*32+32)
      {
        const int h = idx_in_warpgroup % 64;
        const int half = idx_in_warpgroup / 64;
        const uint4* gq = reinterpret_cast<const uint4*>(
            params.q + ((int64_t)req * B_H + h) * (D_NOPE + D_ROPE) * sizeof(bf16_t));
        float mymax = 0.f;
        CUTE_UNROLL
        for (int c = 0; c < 32; ++c) {
          int k8 = half * 32 + c;
          uint4 v = __ldg(gq + k8);
          const __nv_bfloat16* bv = reinterpret_cast<const __nv_bfloat16*>(&v);
          CUTE_UNROLL
          for (int e = 0; e < 8; ++e) mymax = fmaxf(mymax, fabsf(__bfloat162float(bv[e])));
        }
        plan.qmax_scratch[idx_in_warpgroup] = mymax;
      }
      asm volatile("bar.sync 7, 128;\n" ::: "memory");
      if (idx_in_warpgroup < 64) {
        float m = fmaxf(plan.qmax_scratch[idx_in_warpgroup],
                        plan.qmax_scratch[idx_in_warpgroup + 64]);
        plan.qmax_scratch[idx_in_warpgroup] = fmaxf(m / 448.f, 1e-30f);  // now s_q[h]
      }
      asm volatile("bar.sync 7, 128;\n" ::: "memory");

      // ---- q quantize + store fp8 nope / copy bf16 rope ----
      {
        const int h = idx_in_warpgroup % 64;
        const int half = idx_in_warpgroup / 64;
        const uint4* gq = reinterpret_cast<const uint4*>(
            params.q + ((int64_t)req * B_H + h) * (D_NOPE + D_ROPE) * sizeof(bf16_t));
        const float inv_s = 1.0f / plan.qmax_scratch[h];
        Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});
        CUTE_UNROLL
        for (int c = 0; c < 32; ++c) {
          int k8 = half * 32 + c;
          uint4 v = __ldg(gq + k8);
          const __nv_bfloat16* bv = reinterpret_cast<const __nv_bfloat16*>(&v);
          uint64_t packed = 0;
          CUTE_UNROLL
          for (int e = 0; e < 8; ++e) {
            float x = __bfloat162float(bv[e]) * inv_s;
            uint8_t b = (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
            packed |= (uint64_t)b << (8 * e);
          }
          *reinterpret_cast<uint64_t*>(&sQ(h, k8 * 8)) = packed;
        }
        Tensor sQR = make_tensor(make_smem_ptr(plan.q_rope.data()), SmemLayoutQRope{});
        CUTE_UNROLL
        for (int c = 0; c < 4; ++c) {
          int k8 = half * 4 + c;
          uint4 v = __ldg(reinterpret_cast<const uint4*>(gq + 64) + k8);
          *reinterpret_cast<uint4*>(&sQR(h, k8 * 8)) = v;
        }
      }
      // q smem is consumed by WGMMA (async proxy): fence before the CTA barrier
      fence_view_async_shared();
    }

    __syncthreads();  // publish q / s_q / s2e / inv2e

    // ----------------------------------------------------------------
    // Main block loop
    // ----------------------------------------------------------------
    if (warpgroup_idx == 0 || warpgroup_idx == 1) {
      cutlass::arch::warpgroup_reg_alloc<232>();

      const float sm_scale = params.sm_scale_log2e;

      ThrMMA thr_mma_qk = TiledMMA_QK{}.get_slice(idx_in_warpgroup);
      ThrMMA thr_mma_pv = TiledMMA_PV{}.get_slice(idx_in_warpgroup);

      // coordinate tensors for fragment indexing (constant-folded when unrolled)
      Tensor cIdentityQK = make_identity_tensor(Shape<Int<B_H>, Int<B_TOPK>>{});
      Tensor cQK = thr_mma_qk.partition_C(cIdentityQK);
      Tensor cIdentityPV = make_identity_tensor(Shape<Int<B_H>, Int<128>>{});
      Tensor cPV = thr_mma_pv.partition_C(cIdentityPV);

      // accumulators
      Tensor rAcc = partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_H>, Int<B_TOPK>>{});  // 32 f32
      Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_H>, Int<B_TOPK>>{});    // 32 f32 (scores/p)
      Tensor rPt = make_tensor<float>(rP.layout());                                       // 32 f32 scratch
      // fp8 P register layout (A operand of the PV RS GMMA); must be declared
      // inside a __device__ function (decltype of a __device__ auto function)
      using rP_fp8_layout_t = decltype(flash::convert_layout_acc_Aregs<TiledMMA_PV>(
          partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_H>, Int<B_TOPK>>{}).layout()));
      Tensor rP_fp8_g0 = make_tensor<fp8_t>(rP_fp8_layout_t{});
      Tensor rP_fp8_g1 = make_tensor<fp8_t>(rP_fp8_layout_t{});
      Tensor rO_g0 = partition_fragment_C(TiledMMA_PV{}, Shape<Int<B_H>, Int<128>>{});  // 64 f32
      Tensor rO_g1 = partition_fragment_C(TiledMMA_PV{}, Shape<Int<B_H>, Int<128>>{});  // 64 f32
      cute::fill(rO_g0, 0.0f);
      cute::fill(rO_g1, 0.0f);

      float rM[2] = {MAX_INIT_VAL, MAX_INIT_VAL};
      float rL[2] = {0.f, 0.f};
      float rSq[2] = {0.f, 0.f};  // s_q for the two QK rows this thread owns
      {
        int r0 = get_AorC_row_idx(0, idx_in_warpgroup);
        int r1 = get_AorC_row_idx(1, idx_in_warpgroup);
        rSq[0] = plan.qmax_scratch[r0];
        rSq[1] = plan.qmax_scratch[r1];
      }

      const int my_g0 = warpgroup_idx * 2;  // first V group of this WG
      const float s2e0 = plan.s2e[my_g0];
      const float s2e1 = plan.s2e[my_g0 + 1];
      const float inv20 = plan.inv2e[my_g0];
      const float inv21 = plan.inv2e[my_g0 + 1];

      for (int blk = part; blk < nblocks; blk += P) {
        __syncthreads();  // wait for producer's K/Vt for this block

        // ---------------- QK ----------------
        {
          Tensor sKR = make_tensor(make_smem_ptr(plan.k_rope.data()), SmemLayoutKRope{});
          CUTE_UNROLL
          for (int g = 0; g < NUM_GROUPS; ++g) {
            Tensor sQ_t0 = make_tensor(make_smem_ptr(plan.q.data() + (2 * g) * B_H * 64), SmemLayoutQTiles<1>{});
            Tensor sK_t0 = make_tensor(make_smem_ptr(plan.k.data() + (2 * g) * B_TOPK * 64), SmemLayoutKTiles<1>{});
            Tensor sQ_t1 = make_tensor(make_smem_ptr(plan.q.data() + (2 * g + 1) * B_H * 64), SmemLayoutQTiles<1>{});
            Tensor sK_t1 = make_tensor(make_smem_ptr(plan.k.data() + (2 * g + 1) * B_TOPK * 64), SmemLayoutKTiles<1>{});
            gemm_ss(g == 0, TiledMMA_QK{}, sQ_t0, sK_t0, rAcc, idx_in_warpgroup);
            gemm_ss(false, TiledMMA_QK{}, sQ_t1, sK_t1, rAcc, idx_in_warpgroup);
            // descale-add: rP += rAcc * s_k[j, g]  (scales in [g][j] layout)
            CUTE_UNROLL
            for (int i = 0; i < size(rAcc); ++i) {
              int j = get<1>(cQK(i));
              rP(i) += rAcc(i) * plan.k_scales[g * B_TOPK + j];
            }
          }
          // rope: exact bf16 on both sides; rAcc = rope contribution only
          Tensor sQR = make_tensor(make_smem_ptr(plan.q_rope.data()), SmemLayoutQRope{});
          gemm_ss(true, TiledMMA_QK_Rope{}, sQR, sKR, rAcc, idx_in_warpgroup);
        }

        // ---------------- scores + masking + online softmax ----------------
        {
          CUTE_UNROLL
          for (int i = 0; i < size(rP); ++i) {
            int rsel = (i % 4) / 2;
            int j = get<1>(cQK(i));
            float lg = rP(i) * rSq[rsel] + rAcc(i);
            rP(i) = plan.valid[j] ? lg : -INFINITY;
          }

          float new_max[2];
          CUTE_UNROLL
          for (int r = 0; r < 2; ++r) {
            float cur_max = -INFINITY;
            CUTE_UNROLL
            for (int i = r * 2; i < size(rP); i += 4) {
              cur_max = max(cur_max, max(rP(i), rP(i + 1)));
            }
            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
            cur_max *= sm_scale;
            new_max[r] = max(rM[r], cur_max);
            float scale_o = exp2f(rM[r] - new_max[r]);
            CUTE_UNROLL
            for (int i = 0; i < size(rO_g0); ++i) {
              rO_g0(i) *= scale_o;
              rO_g1(i) *= scale_o;
            }
            float cur_sum = 0.f;
            CUTE_UNROLL
            for (int i = r * 2; i < size(rP); i += 4) {
              float p0 = exp2f(rP(i) * sm_scale - new_max[r]);
              float p1 = exp2f(rP(i + 1) * sm_scale - new_max[r]);
              rP(i) = p0;
              rP(i + 1) = p1;
              cur_sum += p0 + p1;
            }
            rL[r] = rL[r] * scale_o + cur_sum;
          }
          rM[0] = new_max[0];
          rM[1] = new_max[1];
        }

        // ---------------- P quantization (both groups) ----------------
        {
          CUTE_UNROLL
          for (int i = 0; i < size(rP); ++i) {
            int j = get<1>(cQK(i));
            rPt(i) = rP(i) * plan.k_scales[my_g0 * B_TOPK + j] * s2e0;
          }
          flash::permute_Cregs_fp8(rPt);
          {
            Tensor rPt_acc = make_tensor(rPt.data(), flash::convert_layout_acc_Aregs<TiledMMA_PV>(rPt.layout()));
            flash::convert_type_out(rPt_acc, rP_fp8_g0);
          }
          CUTE_UNROLL
          for (int i = 0; i < size(rP); ++i) {
            int j = get<1>(cQK(i));
            rPt(i) = rP(i) * plan.k_scales[(my_g0 + 1) * B_TOPK + j] * s2e1;
          }
          flash::permute_Cregs_fp8(rPt);
          {
            Tensor rPt_acc = make_tensor(rPt.data(), flash::convert_layout_acc_Aregs<TiledMMA_PV>(rPt.layout()));
            flash::convert_type_out(rPt_acc, rP_fp8_g1);
          }
        }

        // ---------------- PV (two group GEMMs into rO halves) ----------------
        {
          // WG w owns V dims [w*256, w*256+256) = Vt half w, groups 0,1 within it
          Tensor sVt0 = make_tensor(
              make_smem_ptr(plan.vt.data() + warpgroup_idx * HALF_VT_ELEMS), SmemLayoutVtGroup{});
          Tensor sVt1 = make_tensor(
              make_smem_ptr(plan.vt.data() + warpgroup_idx * HALF_VT_ELEMS + cosize_v<SmemLayoutVtGroup>),
              SmemLayoutVtGroup{});
          gemm_rs(false, TiledMMA_PV{}, rP_fp8_g0, sVt0, rO_g0, idx_in_warpgroup);
          gemm_rs(false, TiledMMA_PV{}, rP_fp8_g1, sVt1, rO_g1, idx_in_warpgroup);
        }

        __syncthreads();  // release the K/Vt buffers to the producer
      }

      // ---------------- partial store ----------------
      {
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);

        const int64_t po_base = ((int64_t)req * P + part) * B_H * D_V;
        const int64_t pm_base = ((int64_t)req * P + part) * B_H * 2;
        CUTE_UNROLL
        for (int i = 0; i < size(rO_g0); ++i) {
          auto cc = cPV(i);
          params.partial_o[po_base + (int64_t)get<0>(cc) * D_V + my_g0 * 128 + get<1>(cc)] =
              rO_g0(i) * inv20;
        }
        CUTE_UNROLL
        for (int i = 0; i < size(rO_g1); ++i) {
          auto cc = cPV(i);
          params.partial_o[po_base + (int64_t)get<0>(cc) * D_V + (my_g0 + 1) * 128 + get<1>(cc)] =
              rO_g1(i) * inv21;
        }
        if (warpgroup_idx == 0 && idx_in_warpgroup % 4 == 0) {
          int r0 = get_AorC_row_idx(0, idx_in_warpgroup);
          int r1 = get_AorC_row_idx(1, idx_in_warpgroup);
          params.partial_ml[pm_base + r0 * 2 + 0] = rM[0];
          params.partial_ml[pm_base + r0 * 2 + 1] = rL[0];
          params.partial_ml[pm_base + r1 * 2 + 0] = rM[1];
          params.partial_ml[pm_base + r1 * 2 + 1] = rL[1];
        }
      }
    } else {
      // ================================================================
      // Producer WG: per-block K/Vt load
      // ================================================================
      cutlass::arch::warpgroup_reg_dealloc<40>();

      const int64_t kv_idx_base = (int64_t)req * params.topk;
      for (int blk = part; blk < nblocks; blk += P) {
        // ---- indices + validity (first 64 threads) ----
        const int* gIdx = params.indices + kv_idx_base + blk * B_TOPK;
        if (idx_in_warpgroup < B_TOPK) {
          int t = __ldg(gIdx + idx_in_warpgroup);
          plan.valid[idx_in_warpgroup] = (t >= 0);
          plan.row_idx[idx_in_warpgroup] = (t >= 0) ? t : 0;
        }
        asm volatile("bar.sync 7, 128;\n" ::: "memory");

        // ---- cp.async loads: flat chunk mapping ----
        // chunk ch in [0, 64*41): row = ch / 41, c = ch % 41
        //   c < 32  : fp8 chunk c     -> sK(row, c*16)      src + c*16
        //   c == 32 : scales (direct __ldg + scattered stores, see below)
        //   c > 32  : rope chunk c-33 -> sKRope(row, (c-33)*16)  src + 528
        Tensor sK = make_tensor(make_smem_ptr(plan.k.data()), SmemLayoutK{});
        Tensor sKR = make_tensor(make_smem_ptr(plan.k_rope.data()), SmemLayoutKRope{});
        int64_t cache_policy = createpolicy_evict_last();
        constexpr int TOTAL_CHUNKS = B_TOPK * CHUNKS_PER_ROW;
        for (int ch = idx_in_warpgroup; ch < TOTAL_CHUNKS; ch += 128) {
          int row = ch / CHUNKS_PER_ROW;
          int c = ch % CHUNKS_PER_ROW;
          int t = plan.row_idx[row];
          const uint8_t* src = params.kv + (int64_t)t * ROW_BYTES;
          if (c < 32) {
            cp_async_cacheglobal_l2_prefetch_256B(src + c * 16, &sK(row, c * 16), true, cache_policy);
          } else if (c == 32) {
            // 4 fp32 scales -> k_scales[g][j] (plain smem stores; read by LDS)
            float4 s4 = __ldg(reinterpret_cast<const float4*>(src + SCALE_OFF));
            plan.k_scales[0 * B_TOPK + row] = s4.x;
            plan.k_scales[1 * B_TOPK + row] = s4.y;
            plan.k_scales[2 * B_TOPK + row] = s4.z;
            plan.k_scales[3 * B_TOPK + row] = s4.w;
          } else {
            int rc = c - 33;
            // 16-B chunk = 8 bf16 ELEMENTS: dest element offset is rc*8 (rc*16 would be bytes)
            cp_async_cacheglobal_l2_prefetch_256B(src + ROPE_OFF + rc * 16, &sKR(row, rc * 8), true, cache_policy);
          }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        // cp.async data must be visible to WGMMA (async proxy) readers
        fence_view_async_shared();
        asm volatile("bar.sync 7, 128;\n" ::: "memory");

        // ---- Vt transpose: two (256, 64) halves ----
        {
          Tensor sV_src = as_position_independent_swizzle_tensor(
              make_tensor(make_smem_ptr(plan.k.data()), SmemLayoutTransposeV_t{}));
          Tensor sVt_dst0 = as_position_independent_swizzle_tensor(
              make_tensor(make_smem_ptr(plan.vt.data()), SmemLayoutTransposeVt_half_t{}));
          Tensor sVt_dst1 = as_position_independent_swizzle_tensor(make_tensor(
              make_smem_ptr(plan.vt.data() + HALF_VT_ELEMS),
              SmemLayoutTransposeVt_half_t{}));
          SmemTransposeV smem_transpose_v;
          // half 0 (d 0..255, groups 0-1): src k-tiles 0..3 -> vt half 0
          CUTE_UNROLL
          for (int j = 0; j < 4; j += 2) {
            smem_transpose_v.transpose_pair(
                flatten(sV_src(_, 0, j)), flatten(sVt_dst0(_, 0, j)),
                flatten(sV_src(_, 0, j + 1)), flatten(sVt_dst0(_, 0, j + 1)));
          }
          // half 1 (d 256..511, groups 2-3): src k-tiles 4..7 -> vt half 1
          CUTE_UNROLL
          for (int j = 4; j < 8; j += 2) {
            smem_transpose_v.transpose_pair(
                flatten(sV_src(_, 0, j)), flatten(sVt_dst1(_, 0, j - 4)),
                flatten(sV_src(_, 0, j + 1)), flatten(sVt_dst1(_, 0, j - 3)));
          }
        }
        // STSM (generic proxy) writes must be visible to WGMMA readers
        fence_view_async_shared();
        __syncthreads();  // K/Vt/scales/valid ready for consumers

        // consumers compute; they __syncthreads at the end of the block
        __syncthreads();
      }
    }
#else
    if (cute::thread0()) {
      CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
    }
#endif
  }
};

// ======================================================================
// Combine kernel: partial (m, l, O) -> bf16 output
// ======================================================================
__global__ void __launch_bounds__(128) sparse_mla_fp8_combine_kernel(
    __grid_constant__ const SparseMlaFp8CombineParams params) {
  // one block per (req, head); 128 threads; d_v = 512 (4 f32 per thread)
  const int req = blockIdx.x / params.num_heads;
  const int h = blockIdx.x % params.num_heads;
  const int P = params.num_splits;

  __shared__ float s_w[1024];  // weights per partition (P <= 1024)
  __shared__ float s_ltot[1];

  if (threadIdx.x == 0) {
    float m_star = -1e30f;
    for (int p = 0; p < P; ++p) {
      const float* ml = params.partial_ml + (((int64_t)req * P + p) * params.num_heads + h) * 2;
      m_star = fmaxf(m_star, __ldg(ml));
    }
    float l_tot = 0.f;
    for (int p = 0; p < P; ++p) {
      const float* ml = params.partial_ml + (((int64_t)req * P + p) * params.num_heads + h) * 2;
      float w = exp2f(__ldg(ml) - m_star);
      s_w[p] = w;
      l_tot += __ldg(ml + 1) * w;
    }
    s_ltot[0] = l_tot;
  }
  __syncthreads();

  const float inv_l = s_ltot[0] > 0.f ? 1.0f / s_ltot[0] : 0.f;
  const int64_t o_base =
      ((int64_t)req * P) * params.num_heads * params.d_v + (int64_t)h * params.d_v;
  const int64_t out_base = ((int64_t)req * params.num_heads + h) * params.d_v;

  const int d0 = threadIdx.x * 4;
  float4 acc = {0.f, 0.f, 0.f, 0.f};
  for (int p = 0; p < P; ++p) {
    float4 o4 = __ldg(reinterpret_cast<const float4*>(
        params.partial_o + o_base + (int64_t)p * params.num_heads * params.d_v + d0));
    float w = s_w[p];
    acc.x += o4.x * w;
    acc.y += o4.y * w;
    acc.z += o4.z * w;
    acc.w += o4.w * w;
  }
  cutlass::bfloat16_t* outp = params.out + out_base + d0;
  outp[0] = (cutlass::bfloat16_t)(acc.x * inv_l);
  outp[1] = (cutlass::bfloat16_t)(acc.y * inv_l);
  outp[2] = (cutlass::bfloat16_t)(acc.z * inv_l);
  outp[3] = (cutlass::bfloat16_t)(acc.w * inv_l);
}

static void run_combine(const SparseMlaFp8CombineParams& params) {
  KU_ASSERT(params.d_v == 512);
  KU_ASSERT(params.num_splits <= 1024);
  sparse_mla_fp8_combine_kernel<<<params.num_reqs * params.num_heads, 128, 0, params.stream>>>(params);
  KU_CHECK_KERNEL_LAUNCH();
}

}  // namespace decode
}  // namespace sm90
