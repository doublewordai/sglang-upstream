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

// SM90 (Hopper) fp8 blockwise-scaled GEMM, CUTLASS ex-67 recipe:
//   A = activation [M, K] fp8 e4m3 row-major (K-major)
//   B = weight    [N, K] fp8 e4m3 row-major (K-major)
//   SFA = per-token-group scales [M, K/128] fp32 col-major
//         (stride (1, M); M must be a multiple of 4 — same buffer the
//          production per-token-group quant produces)
//   SFB = weight block scales [K/128, ceil(N/128)] fp32 row-major
//         (= weight_scale_inv.t().contiguous())
//   D = out [M, N] bf16 row-major
// Scale granularity: SFVec (1, 128, 128) — identical numerics recipe to the
// production DeepGEMM path (per-token-group A, 128x128 B, fp32 accum).
//
// Variants (template kVariant): tile shape / schedule / operand orientation.
// The swapAB orientation puts the weight on the M axis and the tokens on the
// N axis (mirrors the sm120 swap path): better CTA shapes for M <= 16.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

namespace sglang {

using namespace host;

#define CUTLASS_CHECK(status)                                                        \
  {                                                                                  \
    cutlass::Status error = status;                                                  \
    RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

namespace sm90_blockwise {

using ElementBlockScale = float;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

// Assembles the GemmKernel for one (TileShape, ClusterShape, Schedule) combo
// given the operand/scale type configuration, and runs it.
template <
    typename ElementD,
    typename LayoutATag,
    typename LayoutBTag,
    typename LayoutCTag,
    typename LayoutDTag,
    typename LayoutSFA,
    typename LayoutSFB,
    typename TileShape,
    typename ClusterShape,
    typename KernelSchedule,
    typename EpilogueSchedule,
    typename EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto>
struct GemmRunner {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  constexpr static int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  constexpr static int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  using ElementC = void;
  constexpr static int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  constexpr static int AlignmentC = AlignmentD;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      EpilogueTile,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      EpilogueSchedule>::CollectiveOp;

  using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutATag, LayoutSFA>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutBTag, LayoutSFB>,
      AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      StageCount,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // problem shape (M, N, K) with strides packed for (M, K)/(N, K)/(M, N);
  // mainloop pointer order: (A, B, SFA, SFB).
  static cutlass::Status
  run(int m,
      int n,
      int k,
      void const* ptr_a,
      void const* ptr_b,
      void const* ptr_sfa,
      LayoutSFA layout_sfa,
      void const* ptr_sfb,
      LayoutSFB layout_sfb,
      void* ptr_d,
      tvm::ffi::TensorView ref_tensor,
      cudaStream_t stream) {
    Gemm gemm_op;
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideD;

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));

    typename GemmKernel::MainloopArguments mainloop_args{
        static_cast<ElementA const*>(ptr_a), stride_a, static_cast<ElementB const*>(ptr_b),
        stride_b, static_cast<ElementBlockScale const*>(ptr_sfa), layout_sfa,
        static_cast<ElementBlockScale const*>(ptr_sfb), layout_sfb};

    typename GemmKernel::EpilogueArguments epilogue_args{
        {}, static_cast<ElementD*>(ptr_d), stride_c, static_cast<ElementD*>(ptr_d), stride_c};
    epilogue_args.thread.alpha = 1.0f;

    typename Gemm::Arguments args = {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        mainloop_args,
        epilogue_args,
    };

    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    size_t workspace_size = gemm_op.get_workspace_size(args);
    auto workspace_tensor = alloc_workspace_tensor(workspace_size, ref_tensor.device());
    void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    return gemm_op.run(stream);
  }
};

}  // namespace sm90_blockwise

// Non-swapped orientation: tokens on M, weight on N.
// SFA = per-token scales [M, K/128] col-major; SFB = weight block scales
// [K/128, N/128] row-major.
template <typename OutType, typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule,
          typename EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto>
void launch_sm90_fp8_blockwise_scaled_mm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    cudaStream_t stream) {
  using namespace sm90_blockwise;

  // (SFVecM=1, SFVecN=128, SFVecK=128), both scales MN-major.
  using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  int m = static_cast<int>(a.size(0));
  int k = static_cast<int>(a.size(1));
  int n = static_cast<int>(b.size(0));

  // Per-token scale buffer pitch (column-major [M, K/128] over a pad4(M)-row
  // allocation): the SFA K-group stride and TMA extent use the pitch, not the
  // true token count, so M can be unpadded.
  int m_sfa = static_cast<int>(scales_a.stride(1));

  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m_sfa, n, k, 1));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m_sfa, n, k, 1));

  using Runner = GemmRunner<
      OutType,
      cutlass::layout::RowMajor,     // A (N,K)-tag: activation [M, K] K-major
      cutlass::layout::ColumnMajor,  // B (N,K)-tag: weight [N, K] K-major
      cutlass::layout::RowMajor,
      cutlass::layout::RowMajor,
      LayoutSFA,
      LayoutSFB,
      TileShape,
      ClusterShape,
      KernelSchedule,
      EpilogueSchedule,
      EpilogueTile>;

  CUTLASS_CHECK(Runner::run(
      m, n, k, a.data_ptr(), b.data_ptr(), scales_a.data_ptr(), layout_SFA, scales_b.data_ptr(),
      layout_SFB, out.data_ptr(), a, stream));
}

// Swapped orientation: A' = weight [N, K] (M' = N), B' = activation (N' = M),
// D' = out^T column-major. SFA' = weight block scales ([K/128, N/128]
// row-major, MN-major view), SFB' = per-token-group scales (col-major).
template <typename OutType, typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule,
          typename EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto>
void launch_sm90_fp8_blockwise_scaled_mm_swapab(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    cudaStream_t stream) {
  using namespace sm90_blockwise;

  // Operands swapped: SFA' carries the 128x128 weight-block granularity
  // (SFVecM'=128), SFB' the per-token granularity (SFVecN'=1); both MN-major.
  using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<128, 1, 128>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  int m = static_cast<int>(a.size(0));  // tokens  -> swapped N'
  int k = static_cast<int>(a.size(1));
  int n = static_cast<int>(b.size(0));  // weight rows -> swapped M'

  // Token-scale buffer pitch: the per-token scale tensor is column-major
  // [tokens, K/128] over a pad4(tokens)-row allocation, so the K-group stride
  // (and TMA extent) of SFB' is the row pitch, not the true token count.
  // This lets the problem N' stay at the true (unpadded) token count.
  int m_sfb = static_cast<int>(scales_a.stride(1));

  // Swapped problem shape (M', N', K) = (n, m, k).
  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(n, m, k, 1));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(n, m_sfb, k, 1));

  using Runner = GemmRunner<
      OutType,
      cutlass::layout::RowMajor,     // A' (M',K)-tag: weight [N, K] K-major
      cutlass::layout::ColumnMajor,  // B' (N',K)-tag: activation [M, K] K-major
      cutlass::layout::ColumnMajor,  // D' = out^T
      cutlass::layout::ColumnMajor,
      LayoutSFA,
      LayoutSFB,
      TileShape,
      ClusterShape,
      KernelSchedule,
      EpilogueSchedule,
      EpilogueTile>;

  CUTLASS_CHECK(Runner::run(
      n, m, k, b.data_ptr(), a.data_ptr(), scales_b.data_ptr(), layout_SFA, scales_a.data_ptr(),
      layout_SFB, out.data_ptr(), a, stream));
}

template <int kVariant>
struct Fp8BlockwiseSm90Kernel {
  static void
  run(const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView mat_a,
      const tvm::ffi::TensorView mat_b,
      const tvm::ffi::TensorView scales_a,
      const tvm::ffi::TensorView scales_b) {
    RuntimeCheck(mat_a.device().device_type == kDLCUDA, "mat_a must be a CUDA tensor");
    RuntimeCheck(mat_b.device().device_type == kDLCUDA, "mat_b must be a CUDA tensor");

    RuntimeCheck(mat_a.dim() == 2, "mat_a must be a 2D tensor");
    RuntimeCheck(mat_b.dim() == 2, "mat_b must be a 2D tensor");
    RuntimeCheck(mat_a.stride(1) == 1, "mat_a must be a row-major tensor");
    RuntimeCheck(mat_b.stride(1) == 1, "mat_b must be a row-major tensor [N, K]");
    RuntimeCheck(mat_a.size(1) == mat_b.size(1), "mat_a and mat_b K dims must match");
    RuntimeCheck(host::is_type<fp8_e4m3_t>(mat_a.dtype()), "mat_a must be Float8_e4m3fn");
    RuntimeCheck(host::is_type<fp8_e4m3_t>(mat_b.dtype()), "mat_b must be Float8_e4m3fn");

    const int m = static_cast<int>(mat_a.size(0));
    const int k = static_cast<int>(mat_a.size(1));
    const int n = static_cast<int>(mat_b.size(0));
    const int g = k / 128;
    const int nb = (n + 127) / 128;

    RuntimeCheck(k % 128 == 0, "K must be a multiple of 128");
    // M may be any size >= 1: the per-token scale buffer is column-major with
    // a 4-aligned row pitch (the production pad4 TMA-aligned quant buffer),
    // and the swap launcher builds the token-scale layout from that pitch, so
    // the problem M can be the true (unpadded) token count.
    RuntimeCheck(m >= 1, "M must be at least 1");
    RuntimeCheck(m <= 32, "this kernel targets decode shapes (M <= 32)");

    RuntimeCheck(scales_a.dim() == 2, "scales_a must be 2D [M, K/128]");
    RuntimeCheck(static_cast<int>(scales_a.size(0)) == m, "scales_a rows must match M");
    RuntimeCheck(static_cast<int>(scales_a.size(1)) == g, "scales_a cols must be K/128");
    RuntimeCheck(scales_a.stride(0) == 1, "scales_a must be column-major (stride (1, M))");
    RuntimeCheck(static_cast<int>(scales_a.stride(1)) % 4 == 0,
                 "scales_a row pitch must be 4-aligned");
    RuntimeCheck(static_cast<int>(scales_a.stride(1)) >= m,
                 "scales_a row pitch must cover M rows");
    RuntimeCheck(host::is_type<float>(scales_a.dtype()), "scales_a must be Float32");

    RuntimeCheck(scales_b.dim() == 2, "scales_b must be 2D [K/128, N/128]");
    RuntimeCheck(static_cast<int>(scales_b.size(0)) == g, "scales_b rows must be K/128");
    RuntimeCheck(static_cast<int>(scales_b.size(1)) == nb, "scales_b cols must be ceil(N/128)");
    RuntimeCheck(scales_b.stride(1) == 1 || nb == 1, "scales_b must be row-major");
    RuntimeCheck(host::is_type<float>(scales_b.dtype()), "scales_b must be Float32");

    RuntimeCheck(out.dim() == 2, "out must be 2D [M, N]");
    RuntimeCheck(static_cast<int>(out.size(0)) == m && static_cast<int>(out.size(1)) == n,
                 "out must be [M, N]");
    RuntimeCheck(out.stride(1) == 1, "out must be row-major");
    RuntimeCheck(host::is_type<bf16_t>(out.dtype()), "out must be BFloat16");

    const cudaStream_t stream = LaunchKernel::resolve_device(mat_a.device());

    using Cluster = Shape<_1, _1, _1>;
    using EpiWS = cutlass::epilogue::TmaWarpSpecialized;
    using EpiCoop = cutlass::epilogue::TmaWarpSpecializedCooperative;
    if constexpr (kVariant == 0) {
      // pingpong 128x128x128
      launch_sm90_fp8_blockwise_scaled_mm<cutlass::bfloat16_t, Shape<_128, _128, _128>, Cluster,
                                          cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8Blockwise,
                                          EpiWS>(
          out, mat_a, mat_b, scales_a, scales_b, stream);
    } else if constexpr (kVariant == 1) {
      // cooperative 128x128x128 (needs the cooperative epilogue schedule tag,
      // else EpilogueTileAuto picks (64,32) -> "MMA_TILE_M must divide EPI_TILE_M")
      launch_sm90_fp8_blockwise_scaled_mm<cutlass::bfloat16_t, Shape<_128, _128, _128>, Cluster,
                                          cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise,
                                          EpiCoop>(
          out, mat_a, mat_b, scales_a, scales_b, stream);
    } else if constexpr (kVariant == 2) {
      // cooperative 256x128x128 (NumSplitsM=2)
      launch_sm90_fp8_blockwise_scaled_mm<cutlass::bfloat16_t, Shape<_256, _128, _128>, Cluster,
                                          cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise,
                                          EpiCoop>(
          out, mat_a, mat_b, scales_a, scales_b, stream);
    } else if constexpr (kVariant == 3) {
      // cooperative 128x256x128
      launch_sm90_fp8_blockwise_scaled_mm<cutlass::bfloat16_t, Shape<_128, _256, _128>, Cluster,
                                          cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise,
                                          EpiCoop>(
          out, mat_a, mat_b, scales_a, scales_b, stream);
    } else if constexpr (kVariant == 4) {
      // swap: pingpong 128x16x128 (tokens on N')
      launch_sm90_fp8_blockwise_scaled_mm_swapab<
          cutlass::bfloat16_t, Shape<_128, _16, _128>, Cluster,
          cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8Blockwise,
          EpiWS>(
          out, mat_a, mat_b, scales_a, scales_b, stream);
    } else if constexpr (kVariant == 5) {
      // swap: cooperative 128x16x128
      launch_sm90_fp8_blockwise_scaled_mm_swapab<
          cutlass::bfloat16_t, Shape<_128, _16, _128>, Cluster,
          cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise,
          EpiCoop>(
          out, mat_a, mat_b, scales_a, scales_b, stream);
    } else if constexpr (kVariant == 6) {
      // swap: pingpong 128x32x128
      launch_sm90_fp8_blockwise_scaled_mm_swapab<
          cutlass::bfloat16_t, Shape<_128, _32, _128>, Cluster,
          cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8Blockwise,
          EpiWS>(
          out, mat_a, mat_b, scales_a, scales_b, stream);
    } else if constexpr (kVariant == 7) {
      // swap: cooperative 128x32x128
      launch_sm90_fp8_blockwise_scaled_mm_swapab<
          cutlass::bfloat16_t, Shape<_128, _32, _128>, Cluster,
          cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise,
          EpiCoop>(
          out, mat_a, mat_b, scales_a, scales_b, stream);
    } else {
      Panic("unknown variant");
    }
  }
};

#endif  // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

}  // namespace sglang
