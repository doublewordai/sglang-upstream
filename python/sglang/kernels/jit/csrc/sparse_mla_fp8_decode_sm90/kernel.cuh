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
//   * PV (v1b numerics): p_tilde_g = p * s_v[j,g] kept in BF16 (8-bit
//     mantissa, huge range: no 2^E normalization needed) and V consumed as
//     the raw fp8 VALUES converted to bf16 (lossless) through an MN-major-B
//     bf16 WGMMA -- no Vt transpose at all (bf16 WGMMA accepts MN-major B,
//     so the K-tile layout is reused directly). Error vs the fp8-P design:
//     ~0.2% of RMS (prod level) instead of ~2.2% (measured, see lane docs).
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

  enum NamedBarriers : uint32_t {
    kv_ready = 1,  // 384 arrivals: producer published K/rope/scales/V/valid
    p_ready = 2,   // 256 arrivals: WG0 published p + scale_row
    blk_done = 3,  // 384 arrivals: both WGs finished PV for this block
    blk_end = 4,   // 384 arrivals: fused-mode gather/release around the barrier
  };

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

  // V group layout (PV B operand, MN-major): (N=128 d, K=64 j), d contiguous.
  // The producer dequantizes (raw fp8 * s_v) to TRUE bf16 values here.
  using SmemLayoutVGroupMN = decltype(tile_to_shape(
      GMMA::Layout_MN_SW128_Atom<bf16_t>{}, Shape<_128, Int<B_TOPK>>{}));
  static constexpr int V_GROUP_ELEMS = cosize_v<SmemLayoutVGroupMN>;  // 8192 bf16 = 16 KB

  // P tile (PV A operand, K-major in j): (64 h, 64 j) bf16. WG0 writes the
  // post-softmax p ONCE per block; both consumer WGs read it for their PV
  // SS GEMMs (s_v lives in V, so p is group-independent).
  using SmemLayoutP = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<bf16_t>{}, Shape<Int<B_H>, Int<B_TOPK>>{}, Step<_1, _2>{}));

  // MMA atoms
  using TiledMMA_QK = decltype(make_tiled_mma(
      GMMA::MMA_64x64x32_F32E4M3E4M3_SS_TN{}, Layout<Shape<_1, _1, _1>>{}));
  using TiledMMA_QK_Rope = decltype(make_tiled_mma(
      SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}, Layout<Shape<_1, _1, _1>>{}));
  using TiledMMA_PV = decltype(make_tiled_mma(
      SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::MN>{}, Layout<Shape<_1, _1, _1>>{}));

  struct SharedMemoryPlan {
    array_aligned<fp8_t, cosize_v<SmemLayoutQ>> q;             // 32 KB
    array_aligned<bf16_t, cosize_v<SmemLayoutQRope>> q_rope;   // 8 KB
    array_aligned<fp8_t, cosize_v<SmemLayoutK>> k;             // 32 KB
    array_aligned<bf16_t, cosize_v<SmemLayoutKRope>> k_rope;   // 8 KB
    array_aligned<float, NUM_GROUPS * B_TOPK> k_scales;        // [g][j] 1 KB
    array_aligned<bf16_t, NUM_GROUPS * V_GROUP_ELEMS> v;       // 4 x (128d,64j) bf16 = 64 KB
    array_aligned<bf16_t, cosize_v<SmemLayoutP>> p;            // (64 h, 64 j) bf16 = 8 KB
    array_aligned<float, B_H> scale_row;                       // per-head rO rescale factor
    array_aligned<bool, B_TOPK> valid;                         // index >= 0?
    array_aligned<int, B_TOPK> row_idx;                        // clamped row per block slot
    array_aligned<float, 128> qmax_scratch;                    // q per-row scale s_q[h]
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
    // Producer WG: q load (pre-quantized cp.async path or in-kernel quant)
    // ----------------------------------------------------------------
    if (warpgroup_idx == 2) {
      if (params.use_qprep) {
        // cp.async ready-made fp8 q + bf16 rope; direct s_q loads
        Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});
        Tensor sQR = make_tensor(make_smem_ptr(plan.q_rope.data()), SmemLayoutQRope{});
        const uint8_t* q8 = params.q_fp8_out + (int64_t)req * B_H * D_NOPE;
        const uint8_t* qr = reinterpret_cast<const uint8_t*>(params.q_rope_out) +
                            (int64_t)req * B_H * D_ROPE * sizeof(bf16_t);
        int64_t qpol = createpolicy_evict_first();
        CUTE_UNROLL
        for (int ch = idx_in_warpgroup; ch < B_H * 32; ch += 128) {
          int row = ch / 32;
          int c = ch % 32;
          cp_async_cacheglobal_l2_prefetch_256B(q8 + row * 512 + c * 16, &sQ(row, c * 16), true, qpol);
        }
        CUTE_UNROLL
        for (int ch = idx_in_warpgroup; ch < B_H * 8; ch += 128) {
          int row = ch / 8;
          int c = ch % 8;
          cp_async_cacheglobal_l2_prefetch_256B(qr + row * 128 + c * 16, &sQR(row, c * 8), true, qpol);
        }
        if (idx_in_warpgroup < 64) {
          plan.qmax_scratch[idx_in_warpgroup] =
              __ldg(params.q_scale_out + (int64_t)req * B_H + idx_in_warpgroup);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        fence_view_async_shared();
      } else {
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
    }

    __syncthreads();  // publish q / s_q

    // ----------------------------------------------------------------
    // Main block loop. Named barriers:
    //   kv_ready (id 1, 384): producer published K/rope/scales/V/valid
    //   p_ready  (id 2, 256): WG0 published p + scale_row
    //   blk_done (id 3, 384): both WGs finished PV; producer may overwrite
    // WG0: QK + online softmax -> p (smem) -> PV groups 0,1
    // WG1: rescale by WG0's factors -> PV groups 2,3 (no QK at all)
    // ----------------------------------------------------------------
    if (warpgroup_idx == 0 || warpgroup_idx == 1) {
      cutlass::arch::warpgroup_reg_alloc<232>();

      const float sm_scale = params.sm_scale_log2e;
      const int my_g0 = warpgroup_idx * 2;  // first V group of this WG
      Tensor rO_g0 = partition_fragment_C(TiledMMA_PV{}, Shape<Int<B_H>, Int<128>>{});  // 64 f32
      Tensor rO_g1 = partition_fragment_C(TiledMMA_PV{}, Shape<Int<B_H>, Int<128>>{});  // 64 f32
      cute::fill(rO_g0, 0.0f);
      cute::fill(rO_g1, 0.0f);

      ThrMMA thr_mma_pv = TiledMMA_PV{}.get_slice(idx_in_warpgroup);
      Tensor cIdentityPV = make_identity_tensor(Shape<Int<B_H>, Int<128>>{});
      Tensor cPV = thr_mma_pv.partition_C(cIdentityPV);
      Tensor sP = make_tensor(make_smem_ptr(plan.p.data()), SmemLayoutP{});

      if (warpgroup_idx == 0) {
        // ================= WG0: QK + softmax + p + PV g0/g1 =================
        ThrMMA thr_mma_qk = TiledMMA_QK{}.get_slice(idx_in_warpgroup);
        Tensor cIdentityQK = make_identity_tensor(Shape<Int<B_H>, Int<B_TOPK>>{});
        Tensor cQK = thr_mma_qk.partition_C(cIdentityQK);

        Tensor rAcc = partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_H>, Int<B_TOPK>>{});  // 32 f32
        Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_H>, Int<B_TOPK>>{});    // 32 f32
        float rM[2] = {MAX_INIT_VAL, MAX_INIT_VAL};
        float rL[2] = {0.f, 0.f};
        float rSq[2] = {0.f, 0.f};
        {
          int r0 = get_AorC_row_idx(0, idx_in_warpgroup);
          int r1 = get_AorC_row_idx(1, idx_in_warpgroup);
          rSq[0] = plan.qmax_scratch[r0];
          rSq[1] = plan.qmax_scratch[r1];
        }

        for (int blk = part; blk < nblocks; blk += P) {
          NamedBarrier::arrive_and_wait(384, NamedBarriers::kv_ready);

          // ---- QK: fp8 groups (clear per group, drain, exact descale) + rope ----
          {
            Tensor sKR = make_tensor(make_smem_ptr(plan.k_rope.data()), SmemLayoutKRope{});
            cute::fill(rP, 0.0f);
            CUTE_UNROLL
            for (int g = 0; g < NUM_GROUPS; ++g) {
              Tensor sQ_t0 = make_tensor(make_smem_ptr(plan.q.data() + (2 * g) * B_H * 64), SmemLayoutQTiles<1>{});
              Tensor sK_t0 = make_tensor(make_smem_ptr(plan.k.data() + (2 * g) * B_TOPK * 64), SmemLayoutKTiles<1>{});
              Tensor sQ_t1 = make_tensor(make_smem_ptr(plan.q.data() + (2 * g + 1) * B_H * 64), SmemLayoutQTiles<1>{});
              Tensor sK_t1 = make_tensor(make_smem_ptr(plan.k.data() + (2 * g + 1) * B_TOPK * 64), SmemLayoutKTiles<1>{});
              gemm_ss(true, TiledMMA_QK{}, sQ_t0, sK_t0, rAcc, idx_in_warpgroup);
              gemm_ss(false, TiledMMA_QK{}, sQ_t1, sK_t1, rAcc, idx_in_warpgroup);
              warpgroup_commit_batch();
              warpgroup_wait<0>();
              CUTE_UNROLL
              for (int i = 0; i < size(rAcc); ++i) {
                int j = get<1>(cQK(i));
                rP(i) += rAcc(i) * plan.k_scales[g * B_TOPK + j];
              }
            }
            Tensor sQR = make_tensor(make_smem_ptr(plan.q_rope.data()), SmemLayoutQRope{});
            gemm_ss(true, TiledMMA_QK_Rope{}, sQR, sKR, rAcc, idx_in_warpgroup);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
          }

          // ---- logits + mask + row maxima ----
          float new_max[2];
          {
            CUTE_UNROLL
            for (int i = 0; i < size(rP); ++i) {
              int rsel = (i % 4) / 2;
              int j = get<1>(cQK(i));
              float lg = rP(i) * rSq[rsel] + rAcc(i);
              rP(i) = plan.valid[j] ? lg : -INFINITY;
            }
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
            }
          }
          // ---- per-row rO rescale + publish factors for WG1 ----
          {
            const int r0_head = get_AorC_row_idx(0, idx_in_warpgroup);
            CUTE_UNROLL
            for (int i = 0; i < size(rO_g0); ++i) {
              int rsel = (get<0>(cPV(i)) == r0_head) ? 0 : 1;
              float s = exp2f(rM[rsel] - new_max[rsel]);
              rO_g0(i) *= s;
              rO_g1(i) *= s;
            }
            if (idx_in_warpgroup % 4 == 0) {
              plan.scale_row[r0_head] = exp2f(rM[0] - new_max[0]);
              plan.scale_row[get_AorC_row_idx(1, idx_in_warpgroup)] = exp2f(rM[1] - new_max[1]);
            }
          }
          // ---- p = exp2(l*scale - m) -> smem (bf16); l update ----
          {
            CUTE_UNROLL
            for (int r = 0; r < 2; ++r) {
              float scale_l = exp2f(rM[r] - new_max[r]);
              float cur_sum = 0.f;
              CUTE_UNROLL
              for (int i = r * 2; i < size(rP); i += 4) {
                float p0 = exp2f(rP(i) * sm_scale - new_max[r]);
                float p1 = exp2f(rP(i + 1) * sm_scale - new_max[r]);
                rP(i) = p0;
                rP(i + 1) = p1;
                cur_sum += p0 + p1;
              }
              rL[r] = rL[r] * scale_l + cur_sum;
            }
            CUTE_UNROLL
            for (int i = 0; i < size(rP); i += 2) {
              auto cc0 = cQK(i);
              // elements i, i+1 are adjacent j in the same row: pack to u32
              __nv_bfloat162 pk = __floats2bfloat162_rn(rP(i), rP(i + 1));
              *reinterpret_cast<uint32_t*>(&sP(get<0>(cc0), get<1>(cc0))) =
                  *reinterpret_cast<uint32_t*>(&pk);
            }
            rM[0] = new_max[0];
            rM[1] = new_max[1];
          }
          // p is read by WGMMA (async proxy) in BOTH WGs: fence before the barrier
          fence_view_async_shared();
          NamedBarrier::arrive_and_wait(256, NamedBarriers::p_ready);

          // ---- PV groups 0,1 (SS: A = shared p tile, B = V groups) ----
          {
            Tensor sVg0 = make_tensor(make_smem_ptr(plan.v.data() + 0 * V_GROUP_ELEMS), SmemLayoutVGroupMN{});
            Tensor sVg1 = make_tensor(make_smem_ptr(plan.v.data() + 1 * V_GROUP_ELEMS), SmemLayoutVGroupMN{});
            gemm_ss(false, TiledMMA_PV{}, sP, sVg0, rO_g0, idx_in_warpgroup);
            gemm_ss(false, TiledMMA_PV{}, sP, sVg1, rO_g1, idx_in_warpgroup);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
          }
          NamedBarrier::arrive_and_wait(384, NamedBarriers::blk_done);
        }

        // ---- partial store (o by BOTH WGs happens below; m/l by WG0) ----
        {
          rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
          rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
          rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
          rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);
          const int64_t pm_base = ((int64_t)req * P + part) * B_H * 2;
          if (idx_in_warpgroup % 4 == 0) {
            int r0 = get_AorC_row_idx(0, idx_in_warpgroup);
            int r1 = get_AorC_row_idx(1, idx_in_warpgroup);
            params.partial_ml[pm_base + r0 * 2 + 0] = rM[0];
            params.partial_ml[pm_base + r0 * 2 + 1] = rL[0];
            params.partial_ml[pm_base + r1 * 2 + 0] = rM[1];
            params.partial_ml[pm_base + r1 * 2 + 1] = rL[1];
          }
        }
      } else {
        // ================= WG1: rescale by published factors + PV g2/g3 =================
        for (int blk = part; blk < nblocks; blk += P) {
          NamedBarrier::arrive_and_wait(384, NamedBarriers::kv_ready);
          NamedBarrier::arrive_and_wait(256, NamedBarriers::p_ready);
          // rescale rO by WG0's per-row factors
          CUTE_UNROLL
          for (int i = 0; i < size(rO_g0); ++i) {
            float s = plan.scale_row[get<0>(cPV(i))];
            rO_g0(i) *= s;
            rO_g1(i) *= s;
          }
          {
            Tensor sVg2 = make_tensor(make_smem_ptr(plan.v.data() + 2 * V_GROUP_ELEMS), SmemLayoutVGroupMN{});
            Tensor sVg3 = make_tensor(make_smem_ptr(plan.v.data() + 3 * V_GROUP_ELEMS), SmemLayoutVGroupMN{});
            gemm_ss(false, TiledMMA_PV{}, sP, sVg2, rO_g0, idx_in_warpgroup);
            gemm_ss(false, TiledMMA_PV{}, sP, sVg3, rO_g1, idx_in_warpgroup);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
          }
          NamedBarrier::arrive_and_wait(384, NamedBarriers::blk_done);
        }
      }

      // ---- partial o store (both WGs, disjoint V-dim halves; adjacent-d
      // pairs packed to u64) ----
      {
        const int64_t po_base = ((int64_t)req * P + part) * B_H * D_V;
        CUTE_UNROLL
        for (int i = 0; i < size(rO_g0); i += 2) {
          auto cc = cPV(i);
          float2 f2 = make_float2(rO_g0(i), rO_g0(i + 1));
          *reinterpret_cast<uint64_t*>(
              &params.partial_o[po_base + (int64_t)get<0>(cc) * D_V + my_g0 * 128 + get<1>(cc)]) =
              *reinterpret_cast<uint64_t*>(&f2);
        }
        CUTE_UNROLL
        for (int i = 0; i < size(rO_g1); i += 2) {
          auto cc = cPV(i);
          float2 f2 = make_float2(rO_g1(i), rO_g1(i + 1));
          *reinterpret_cast<uint64_t*>(
              &params.partial_o[po_base + (int64_t)get<0>(cc) * D_V + (my_g0 + 1) * 128 + get<1>(cc)]) =
              *reinterpret_cast<uint64_t*>(&f2);
        }
      }
    } else {
      // ================================================================
      // Producer WG: per-block K/rope/scales load + V dequant
      // ================================================================
      cutlass::arch::warpgroup_reg_dealloc<40>();

      const int64_t kv_idx_base = (int64_t)req * params.topk;
      for (int blk = part; blk < nblocks; blk += P) {
        const int* gIdx = params.indices + kv_idx_base + blk * B_TOPK;
        if (idx_in_warpgroup < B_TOPK) {
          int t = __ldg(gIdx + idx_in_warpgroup);
          plan.valid[idx_in_warpgroup] = (t >= 0);
          plan.row_idx[idx_in_warpgroup] = (t >= 0) ? t : 0;
        }
        asm volatile("bar.sync 7, 128;\n" ::: "memory");

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
            float4 s4 = __ldg(reinterpret_cast<const float4*>(src + SCALE_OFF));
            plan.k_scales[0 * B_TOPK + row] = s4.x;
            plan.k_scales[1 * B_TOPK + row] = s4.y;
            plan.k_scales[2 * B_TOPK + row] = s4.z;
            plan.k_scales[3 * B_TOPK + row] = s4.w;
          } else {
            int rc = c - 33;
            // 16-B chunk = 8 bf16 ELEMENTS: dest element offset is rc*8
            cp_async_cacheglobal_l2_prefetch_256B(src + ROPE_OFF + rc * 16, &sKR(row, rc * 8), true, cache_policy);
          }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        fence_view_async_shared();
        asm volatile("bar.sync 7, 128;\n" ::: "memory");

        // ---- V dequant: v_true = fp8 * s_v[j,g] -> bf16 (4 group tiles) ----
        // vectorized: one 8-B fp8 chunk -> one 16-B bf16 chunk (16x fewer smem
        // ops than the scalar version; the scalar path was 55% of all L1
        // wavefronts). Chunk = 8 consecutive d within one (g, j) row.
        {
          Tensor sKfull = make_tensor(make_smem_ptr(plan.k.data()), SmemLayoutK{});
          CUTE_UNROLL
          for (int g = 0; g < NUM_GROUPS; ++g) {
            Tensor sVg = make_tensor(
                make_smem_ptr(plan.v.data() + g * V_GROUP_ELEMS), SmemLayoutVGroupMN{});
            CUTE_UNROLL
            for (int e = 0; e < 8; ++e) {
              int chunk = idx_in_warpgroup + e * 128;  // 1024 chunks of 8 d
              int j = chunk / 16;
              int d8 = (chunk % 16) * 8;
              // 8 fp8 bytes at (j, g*128 + d8): contiguous under the swizzle
              const uint2 raw = *reinterpret_cast<const uint2*>(&sKfull(j, g * 128 + d8));
              const __nv_fp8_e4m3* f8 = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
              const float s0 = plan.k_scales[g * B_TOPK + j];
              uint4 outv;
              __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&outv);
              CUTE_UNROLL
              for (int t = 0; t < 8; ++t) {
                ob[t] = __float2bfloat16((float)f8[t] * s0);
              }
              *reinterpret_cast<uint4*>(&sVg(d8, j)) = outv;
            }
          }
        }
        // generic-proxy writes must be visible to WGMMA readers
        fence_view_async_shared();
        NamedBarrier::arrive_and_wait(384, NamedBarriers::kv_ready);
        NamedBarrier::arrive_and_wait(384, NamedBarriers::blk_done);
      }
    }

    // ----------------------------------------------------------------
    // Fused mode: single-launch combine behind a grid-wide atomic
    // barrier. Co-residency is required (host caps b*P <= 132 CTAs at
    // 1 CTA/SM, matching the contest-kernels-bench finding).
    // ----------------------------------------------------------------
    if (params.fused) {
      NamedBarrier::arrive_and_wait(384, NamedBarriers::blk_end);
      if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(params.counter, 1);
        const int total = gridDim.x * gridDim.y;
        volatile int* c = reinterpret_cast<volatile int*>(params.counter);
        while (*c < total) __nanosleep(64);
        __threadfence();
      }
      NamedBarrier::arrive_and_wait(384, NamedBarriers::blk_end);
      // parallel combine: flat work items (unit, d4), unit = req*64 + head
      {
        const int T = gridDim.x * gridDim.y;
        const int g = blockIdx.x * gridDim.y + blockIdx.y;
        const int P = params.num_splits;
        const int total_items = params.num_reqs * 64 * 128;
        for (int w = g * NUM_THREADS + threadIdx.x; w < total_items; w += (int64_t)T * NUM_THREADS) {
          const int unit = w >> 7;          // /128
          const int d4 = w & 127;           // %128
          const int req_u = unit >> 6;      // /64
          const int h_u = unit & 63;
          const float* ml0 = params.partial_ml + (((int64_t)req_u * P) * 64 + h_u) * 2;
          // per-unit weights (P <= 32 in fused mode)
          float m_star = -1e30f;
          for (int p = 0; p < P; ++p) m_star = fmaxf(m_star, __ldg(ml0 + (int64_t)p * 128));
          float wgt[32];
          float l_tot = 0.f;
          for (int p = 0; p < P; ++p) {
            wgt[p] = exp2f(__ldg(ml0 + (int64_t)p * 128) - m_star);
            l_tot += __ldg(ml0 + (int64_t)p * 128 + 1) * wgt[p];
          }
          const float inv_l = l_tot > 0.f ? 1.0f / l_tot : 0.f;
          const float* o0 = params.partial_o + ((int64_t)req_u * P) * 64 * 512 + (int64_t)h_u * 512 + d4 * 4;
          float4 acc = {0.f, 0.f, 0.f, 0.f};
          for (int p = 0; p < P; ++p) {
            float4 o4 = __ldg(reinterpret_cast<const float4*>(o0 + (int64_t)p * 64 * 512));
            acc.x += o4.x * wgt[p];
            acc.y += o4.y * wgt[p];
            acc.z += o4.z * wgt[p];
            acc.w += o4.w * wgt[p];
          }
          cutlass::bfloat16_t* outp =
              params.out_fused + ((int64_t)req_u * 64 + h_u) * 512 + d4 * 4;
          outp[0] = (cutlass::bfloat16_t)(acc.x * inv_l);
          outp[1] = (cutlass::bfloat16_t)(acc.y * inv_l);
          outp[2] = (cutlass::bfloat16_t)(acc.z * inv_l);
          outp[3] = (cutlass::bfloat16_t)(acc.w * inv_l);
        }
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
// q pre-quantization kernel: q [b, 64, 576] bf16 -> q_fp8 [b, 64, 512] +
// q_rope [b, 64, 64] bf16 + s_q [b, 64] f32. Runs once per step; the split
// CTAs then cp.async the ready-made fp8 q instead of re-quantizing it.
// ======================================================================
__global__ void __launch_bounds__(32) sparse_mla_fp8_qprep_kernel(
    __grid_constant__ const SparseMlaFp8DecodeParams params) {
  // one warp per (req, head) row: 16 nope elems (2 x uint4) + 2 rope elems per
  // lane. Also zeroes the fused-mode barrier counter (block 0, lane 0).
  constexpr int B_H = 64, D_NOPE = 512, D_ROPE = 64;
  using bf16_t = cutlass::bfloat16_t;
  const int req = blockIdx.x / B_H;
  const int h = blockIdx.x % B_H;
  const int lane = threadIdx.x;
  const uint4* gq = reinterpret_cast<const uint4*>(
      params.q + ((int64_t)req * B_H + h) * (D_NOPE + D_ROPE) * sizeof(bf16_t));
  if (blockIdx.x == 0 && lane == 0 && params.counter != nullptr) {
    *params.counter = 0;
  }
  // pass 1: row max (each lane: 2 uint4 = 16 bf16)
  float mymax = 0.f;
  uint4 v0 = __ldg(gq + lane);
  uint4 v1 = __ldg(gq + 32 + lane);
  {
    const __nv_bfloat16* b0 = reinterpret_cast<const __nv_bfloat16*>(&v0);
    const __nv_bfloat16* b1 = reinterpret_cast<const __nv_bfloat16*>(&v1);
    CUTE_UNROLL
    for (int e = 0; e < 8; ++e) {
      mymax = fmaxf(mymax, fabsf(__bfloat162float(b0[e])));
      mymax = fmaxf(mymax, fabsf(__bfloat162float(b1[e])));
    }
  }
  CUTE_UNROLL
  for (int off = 16; off > 0; off >>= 1) mymax = fmaxf(mymax, __shfl_xor_sync(0xffffffff, mymax, off));
  const float s = fmaxf(mymax / 448.f, 1e-30f);
  if (lane == 0) params.q_scale_out[(int64_t)req * B_H + h] = s;
  const float inv_s = 1.0f / s;
  // pass 2: quantize (registers already hold the data) + rope copy
  uint8_t* q8 = params.q_fp8_out + ((int64_t)req * B_H + h) * D_NOPE;
  {
    uint64_t p0 = 0, p1 = 0;
    const __nv_bfloat16* b0 = reinterpret_cast<const __nv_bfloat16*>(&v0);
    const __nv_bfloat16* b1 = reinterpret_cast<const __nv_bfloat16*>(&v1);
    CUTE_UNROLL
    for (int e = 0; e < 8; ++e) {
      p0 |= (uint64_t)(uint8_t)__nv_cvt_float_to_fp8(__bfloat162float(b0[e]) * inv_s, __NV_SATFINITE, __NV_E4M3) << (8 * e);
      p1 |= (uint64_t)(uint8_t)__nv_cvt_float_to_fp8(__bfloat162float(b1[e]) * inv_s, __NV_SATFINITE, __NV_E4M3) << (8 * e);
    }
    // lane's chunks: fp8 bytes [lane*8, lane*8+8) and [256+lane*8, +8)
    *reinterpret_cast<uint64_t*>(q8 + lane * 8) = p0;
    *reinterpret_cast<uint64_t*>(q8 + 256 + lane * 8) = p1;
  }
  bf16_t* qr = params.q_rope_out + ((int64_t)req * B_H + h) * D_ROPE;
  {
    // rope: 64 bf16 = 8 uint4; lanes 0..7 copy one each
    if (lane < 8) {
      uint4 v = __ldg(reinterpret_cast<const uint4*>(gq + 64) + lane);
      *reinterpret_cast<uint4*>(qr + lane * 8) = v;
    }
  }
}

// ======================================================================
// Combine kernel: partial (m, l, O) -> bf16 output
// ======================================================================
__global__ void __launch_bounds__(32) sparse_mla_fp8_combine_kernel(
    __grid_constant__ const SparseMlaFp8CombineParams params) {
  // one block per (req, head, d-quarter); 32 threads x 4 f32 = 128 dims each
  const int req = blockIdx.x / (params.num_heads * 4);
  const int rem = blockIdx.x % (params.num_heads * 4);
  const int h = rem / 4;
  const int dq = rem % 4;  // d-quarter 0..3
  const int P = params.num_splits;

  __shared__ float s_w[1024];  // weights per partition (P <= 1024)
  __shared__ float s_ltot[1];
  __shared__ float s_mstar[1];

  // parallel m/l pass: each thread loads a strided subset, block-reduces
  {
    float my_m = -1e30f;
    for (int p = threadIdx.x; p < P; p += 32) {
      const float* ml = params.partial_ml + (((int64_t)req * P + p) * params.num_heads + h) * 2;
      my_m = fmaxf(my_m, __ldg(ml));
    }
    // single-warp block max reduce
    for (int off = 16; off > 0; off >>= 1) my_m = fmaxf(my_m, __shfl_xor_sync(0xffffffff, my_m, off));
    if (threadIdx.x == 0) s_mstar[0] = my_m;
    __syncthreads();
    const float m_star = s_mstar[0];
    float my_l = 0.f;
    for (int p = threadIdx.x; p < P; p += 32) {
      const float* ml = params.partial_ml + (((int64_t)req * P + p) * params.num_heads + h) * 2;
      float w = exp2f(__ldg(ml) - m_star);
      s_w[p] = w;
      my_l += __ldg(ml + 1) * w;
    }
    for (int off = 16; off > 0; off >>= 1) my_l += __shfl_xor_sync(0xffffffff, my_l, off);
    if (threadIdx.x == 0) s_ltot[0] = my_l;
    __syncthreads();
  }

  const float inv_l = s_ltot[0] > 0.f ? 1.0f / s_ltot[0] : 0.f;
  const int64_t o_base =
      ((int64_t)req * P) * params.num_heads * params.d_v + (int64_t)h * params.d_v;
  const int64_t out_base = ((int64_t)req * params.num_heads + h) * params.d_v;

  const int d0 = dq * 128 + threadIdx.x * 4;
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

static void run_qprep(const SparseMlaFp8DecodeParams& params) {
  // one warp per (req, head) row; also zeroes the fused counter
  sparse_mla_fp8_qprep_kernel<<<params.num_reqs * 64, 32, 0, params.stream>>>(params);
  KU_CHECK_KERNEL_LAUNCH();
}

static void run_combine(const SparseMlaFp8CombineParams& params) {
  KU_ASSERT(params.d_v == 512);
  KU_ASSERT(params.num_splits <= 1024);
  sparse_mla_fp8_combine_kernel<<<params.num_reqs * params.num_heads * 4, 32, 0, params.stream>>>(params);
  KU_CHECK_KERNEL_LAUNCH();
}

}  // namespace decode
}  // namespace sm90
