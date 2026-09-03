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

// SM90 (Hopper) mixed-dtype GEMM: fp8 e4m3 weights x bf16 activations with
// in-kernel dequant (CUTLASS sm90 mixed-input RS mainloop, ex-55/machete
// lineage). One kernel replaces per-token-group-quant + DeepGEMM fp8_gemm_nt.
//
// Orientation (manual swap, no kernel-layer SwapAB): the weight+scale tuple
// rides the A slot (register side after dequant, MMA M axis), activations ride
// the B slot (smem descriptor, MMA N axis), and D' = out^T is written through
// a column-major view. Problem shape passed to the kernel is (N, M, K).
//
//   mat_a      : [M, K] bf16 row-major activation (M <= 32, M % 4 == 0 not
//                required here; M % 8 == 0 not required -- padding handled by
//                the M-tile; any M >= 1)
//   mat_b      : [N, K] fp8 e4m3 row-major weight (as stored in production)
//   scales_b   : [ceil(K/128), N] fp32 row-major, expanded per-N-row block
//                scales: S[g, n] = weight_scale_inv[n / 128, g]
//                (one-off expansion of the production [N/128, K/128] grid;
//                N % 4 == 0 required for TMA scale alignment)
//   out        : [M, N] bf16 row-major
//   group size : 128 along K (runtime argument), K % 128 == 0.
//
// Numerics vs the production fp8 path: activations keep bf16 precision (the
// fp8 activation-quant rounding is REMOVED); weights are dequantised as
// bf16(float(fp8) * scale) in-register before a bf16 HMMA with fp32
// accumulation — one fewer rounding than production on the activation side
// and the same weight values.

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

namespace sm90_w8a16 {

using ElementBlockScale = float;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

template <
    typename TileShape,
    typename ClusterShape,
    typename KernelSchedule,
    typename EpilogueSchedule,
    typename EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto>
struct GemmRunner {
  // A' = weight, fp8 e4m3 with fp32 group scales (register side after dequant)
  using ElementA = cute::tuple<cutlass::float_e4m3_t, float>;
  using LayoutATag = cutlass::layout::RowMajor;  // weight [N, K] row-major = K-major
  constexpr static int AlignmentA = 128 / cutlass::sizeof_bits<cutlass::float_e4m3_t>::value;

  // B' = activation, bf16 (smem descriptor side)
  using ElementB = cutlass::bfloat16_t;
  using LayoutBTag = cutlass::layout::ColumnMajor;  // activation [M, K] K-major
  constexpr static int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = cutlass::bfloat16_t;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::ColumnMajor;  // D' = out^T
  using LayoutDTag = cutlass::layout::ColumnMajor;
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
      LayoutATag,
      AlignmentA,
      ElementB,
      LayoutBTag,
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

  // Problem shape (M', N', K) = (n, m, k) with the manual swap; D' = out^T.
  static cutlass::Status
  run(int m,
      int n,
      int k,
      void const* ptr_act,      // activations [m, k] bf16
      void const* ptr_weight,   // weight [n, k] fp8
      void const* ptr_scales,   // [k/128, n] fp32 row-major
      int group_size,
      void* ptr_d,              // out [m, n] bf16 (written as out^T)
      tvm::ffi::TensorView ref_tensor,
      cudaStream_t stream) {
    Gemm gemm_op;
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideD;
    using StrideScale = typename CollectiveMainloop::NonVoidStrideScale;

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(n, k, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(m, k, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(n, m, 1));
    // Scale layout (M'=n, k-groups, L), n contiguous.
    StrideScale stride_s = cutlass::make_cute_packed_stride(StrideScale{}, cute::make_shape(n, k / group_size, 1));

    typename GemmKernel::MainloopArguments mainloop_args{
        static_cast<cutlass::float_e4m3_t const*>(ptr_weight),
        stride_a,
        static_cast<cutlass::bfloat16_t const*>(ptr_act),
        stride_b,
        static_cast<ElementBlockScale const*>(ptr_scales),
        stride_s,
        group_size,
        nullptr,
        4};

    typename GemmKernel::EpilogueArguments epilogue_args{
        {}, static_cast<ElementD*>(ptr_d), stride_c, static_cast<ElementD*>(ptr_d), stride_c};
    epilogue_args.thread.alpha = 1.0f;

    typename Gemm::Arguments args = {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {n, m, k, 1},
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

}  // namespace sm90_w8a16

template <int kVariant>
struct W8A16GemmSm90Kernel {
  static void
  run(const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView mat_a,
      const tvm::ffi::TensorView mat_b,
      const tvm::ffi::TensorView scales_b) {
    RuntimeCheck(mat_a.device().device_type == kDLCUDA, "mat_a must be a CUDA tensor");
    RuntimeCheck(mat_b.device().device_type == kDLCUDA, "mat_b must be a CUDA tensor");

    RuntimeCheck(mat_a.dim() == 2, "mat_a must be a 2D tensor");
    RuntimeCheck(mat_b.dim() == 2, "mat_b must be a 2D tensor");
    RuntimeCheck(mat_a.stride(1) == 1, "mat_a must be a row-major tensor [M, K]");
    RuntimeCheck(mat_b.stride(1) == 1, "mat_b must be a row-major tensor [N, K]");
    RuntimeCheck(mat_a.size(1) == mat_b.size(1), "mat_a and mat_b K dims must match");
    RuntimeCheck(host::is_type<bf16_t>(mat_a.dtype()), "mat_a must be BFloat16");
    RuntimeCheck(host::is_type<fp8_e4m3_t>(mat_b.dtype()), "mat_b must be Float8_e4m3fn");

    const int m = static_cast<int>(mat_a.size(0));
    const int k = static_cast<int>(mat_a.size(1));
    const int n = static_cast<int>(mat_b.size(0));
    const int g = k / 128;

    RuntimeCheck(k % 128 == 0, "K must be a multiple of 128");
    RuntimeCheck(n % 4 == 0, "N must be a multiple of 4 (scale TMA alignment)");
    RuntimeCheck(m >= 1, "M must be at least 1");
    RuntimeCheck(m <= 32, "this kernel targets decode shapes (M <= 32)");

    RuntimeCheck(scales_b.dim() == 2, "scales_b must be 2D [K/128, N]");
    RuntimeCheck(static_cast<int>(scales_b.size(0)) == g, "scales_b rows must be K/128");
    RuntimeCheck(static_cast<int>(scales_b.size(1)) == n, "scales_b cols must be N");
    RuntimeCheck(scales_b.stride(1) == 1 || n == 1, "scales_b must be row-major");
    RuntimeCheck(host::is_type<float>(scales_b.dtype()), "scales_b must be Float32");

    RuntimeCheck(out.dim() == 2, "out must be 2D [M, N]");
    RuntimeCheck(static_cast<int>(out.size(0)) == m && static_cast<int>(out.size(1)) == n,
                 "out must be [M, N]");
    RuntimeCheck(out.stride(1) == 1, "out must be row-major");
    RuntimeCheck(host::is_type<bf16_t>(out.dtype()), "out must be BFloat16");

    const cudaStream_t stream = LaunchKernel::resolve_device(mat_a.device());

    using Cluster = Shape<_1, _1, _1>;
    if constexpr (kVariant == 0) {
      // basic, tile (N'=128, M'=16, K=64)
      sm90_w8a16::GemmRunner<Shape<_128, _16, _64>, Cluster,
                             cutlass::gemm::KernelTmaWarpSpecialized,
                             cutlass::epilogue::TmaWarpSpecialized>::
          run(m, n, k, mat_a.data_ptr(), mat_b.data_ptr(), scales_b.data_ptr(), 128,
              out.data_ptr(), mat_a, stream);
    } else if constexpr (kVariant == 1) {
      // basic, tile (128, 16, 128)
      sm90_w8a16::GemmRunner<Shape<_128, _16, _128>, Cluster,
                             cutlass::gemm::KernelTmaWarpSpecialized,
                             cutlass::epilogue::TmaWarpSpecialized>::
          run(m, n, k, mat_a.data_ptr(), mat_b.data_ptr(), scales_b.data_ptr(), 128,
              out.data_ptr(), mat_a, stream);
    } else if constexpr (kVariant == 2) {
      // basic, tile (64, 16, 64)
      sm90_w8a16::GemmRunner<Shape<_64, _16, _64>, Cluster,
                             cutlass::gemm::KernelTmaWarpSpecialized,
                             cutlass::epilogue::TmaWarpSpecialized>::
          run(m, n, k, mat_a.data_ptr(), mat_b.data_ptr(), scales_b.data_ptr(), 128,
              out.data_ptr(), mat_a, stream);
    } else if constexpr (kVariant == 3) {
      // cooperative, tile (128, 16, 64)
      sm90_w8a16::GemmRunner<Shape<_128, _16, _64>, Cluster,
                             cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                             cutlass::epilogue::TmaWarpSpecializedCooperative>::
          run(m, n, k, mat_a.data_ptr(), mat_b.data_ptr(), scales_b.data_ptr(), 128,
              out.data_ptr(), mat_a, stream);
    } else {
      Panic("unknown variant");
    }
  }
};

#endif  // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

}  // namespace sglang
