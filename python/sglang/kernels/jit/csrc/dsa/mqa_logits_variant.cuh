#pragma once
// Tunable block configuration for DeepGEMM's sm90_fp8_mqa_logits (lane mqa-tune).
//
// DeepGEMM hardcodes the block configuration of the prefill indexer logits
// kernel in its compiled API (csrc/apis/attention.hpp: block_qh = 128 i.e.
// BLOCK_Q = 128 / num_heads, block_kv = 256; csrc/jit_kernels/impls/
// sm90_fp8_mqa_logits.hpp: num_q_stages = 3, num_kv_stages = 3,
// num_math_threads = 512) with no runtime override.  Measured on GH200 at the
// GLM-5.3 production shapes (32 index heads x 128 dims, causal), that
// configuration sustains ~735 TF/s while the same kernel template with
// BLOCK_KV = 192, 384 math threads and a 3x5 stage pipeline sustains
// ~780 TF/s (+5-7%), bit-exact.
//
// This header includes the *unmodified* kernel template from the deep_gemm
// package and exposes it with the block configuration as template parameters,
// plus a host launcher replicating DeepGEMM's launcher (TMA descriptors, smem
// sizing, persistent-grid launch).  The block configuration does not change
// the arithmetic of any output element: every (q row, kv column) logit is
// produced by exactly one WGMMA k-loop accumulation and one epilogue
// head-reduction, both with fixed order, independent of BLOCK_KV, the stage
// counts or the number of math warpgroups.  Equivalence is validated
// bit-exactly (torch.equal) against deep_gemm.fp8_mqa_logits by
// mqa_patch_equiv.py.
//
// Supported: num_heads = 32, head_dim = 128, fp8 q/kv + fp32 per-row kv scale,
// fp32 per-(q, head) weights, non-compressed logits (max_seqlen_k = 0,
// clean_logits = False semantics: positions outside [ks, ke) receive
// unreduced values, exactly like the production call).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <deep_gemm/impls/sm90_fp8_mqa_logits.cuh>

namespace sglang {

template <uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumMathThreads, uint32_t kNumSMs>
struct MQALogitsVariantKernel {
  static constexpr uint32_t kNumHeads = 32;
  static constexpr uint32_t kHeadDim = 128;
  static constexpr uint32_t kNumTMAThreads = 128;
  static constexpr int kMaxSmem = 232448;  // SM90ArchSpec::smem_capacity

  static_assert(BLOCK_KV == kNumMathThreads / 2, "BLOCK_KV must equal math_threads/2");
  static_assert(kNumMathThreads % 128 == 0, "math threads must be a multiple of 128");
  static_assert(BLOCK_Q * kNumHeads <= 256, "WGMMA N limit");

  static void run(tvm::ffi::TensorView q, tvm::ffi::TensorView kv,
                  tvm::ffi::TensorView kv_scales, tvm::ffi::TensorView weights,
                  tvm::ffi::TensorView cu_seq_len_k_start,
                  tvm::ffi::TensorView cu_seq_len_k_end,
                  tvm::ffi::TensorView logits) {
    using namespace host;

    auto N = SymbolicSize{"num_q"};
    auto L = SymbolicSize{"num_kv"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, kNumHeads, kHeadDim})  // q
        .with_dtype<fp8_e4m3_t>()
        .with_device(device_)
        .verify(q);
    TensorMatcher({L, kHeadDim})  // kv (gathered, contiguous)
        .with_dtype<fp8_e4m3_t>()
        .with_device(device_)
        .verify(kv);
    TensorMatcher({L})  // kv per-row scales
        .with_dtype<float>()
        .with_device(device_)
        .verify(kv_scales);
    TensorMatcher({N, kNumHeads})  // weights
        .with_dtype<float>()
        .with_device(device_)
        .verify(weights);
    TensorMatcher({N})  // ks / ke
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(cu_seq_len_k_start);
    TensorMatcher({N})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(cu_seq_len_k_end);
    TensorMatcher({{-1, -1}})  // logits: padded [align(N, BLOCK_Q), stride]
        .with_dtype<float>()
        .with_device(device_)
        .verify(logits);

    const auto seq_len = static_cast<int>(N.unwrap());
    const auto seq_len_kv = static_cast<int>(L.unwrap());
    const auto stride_logits = static_cast<int>(logits.stride(0));
    CHECK_HOST(logits.size(0) >= (seq_len + BLOCK_Q - 1) / BLOCK_Q * BLOCK_Q)
        << "logits buffer needs align(seq_len, BLOCK_Q) rows";
    CHECK_HOST(stride_logits % 256 == 0) << "logits row stride must be 1024B aligned";
    CHECK_HOST(stride_logits >= seq_len_kv + (int)BLOCK_KV)
        << "logits row stride must cover seq_len_kv + BLOCK_KV";

    // Shared memory (mirrors deep_gemm's sm90_fp8_mqa_logits launcher)
    const int smem_size = (int)(
        kNumQStages * (BLOCK_Q * kNumHeads * kHeadDim) +
        kNumKVStages * (BLOCK_KV * kHeadDim) +
        kNumQStages * (BLOCK_Q * kNumHeads * 4) +
        kNumKVStages * (BLOCK_KV * 4) +
        (kNumQStages * 2 + kNumKVStages * 2 + (kNumMathThreads / 128) * 2) * 8 + 4);
    CHECK_HOST(smem_size <= kMaxSmem) << "smem overflow: " << smem_size;

    static constexpr auto kernel = deep_gemm::sm90_fp8_mqa_logits<
        kNumHeads, kHeadDim, false, BLOCK_Q, BLOCK_KV,
        kNumQStages, kNumKVStages, kNumSMs, kNumTMAThreads, kNumMathThreads, float>;
    static const bool smem_attr_ok = [] {
      return ::cudaFuncSetAttribute(
                 reinterpret_cast<const void*>(kernel),
                 cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxSmem) == cudaSuccess;
    }();
    CHECK_HOST(smem_attr_ok) << "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed";

    // TMA descriptors (mirror deep_gemm's make_tma_2d_desc calls for this kernel)
    const auto tensor_map_q = make_tma_2d_desc(
        q, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kHeadDim,
        (int64_t)seq_len * kNumHeads, kHeadDim, BLOCK_Q * kNumHeads, kHeadDim, 128);
    const auto tensor_map_kv = make_tma_2d_desc(
        kv, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kHeadDim, seq_len_kv, kHeadDim,
        BLOCK_KV, kHeadDim, 128);
    const int64_t aligned_l = (seq_len_kv + 3) / 4 * 4;  // 16B TMA alignment
    const auto tensor_map_kv_scales = make_tma_2d_desc(
        kv_scales, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, aligned_l, 1, BLOCK_KV, 1, 0, 0);
    const auto tensor_map_weights = make_tma_2d_desc(
        weights, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, kNumHeads, seq_len, kNumHeads,
        BLOCK_Q, kNumHeads, 0);

    LaunchKernel(kNumSMs, kNumTMAThreads + kNumMathThreads, device_.unwrap(),
                 (std::size_t)smem_size)(
        kernel, (uint32_t)seq_len, (uint32_t)seq_len_kv, (uint32_t)0,
        (uint32_t)stride_logits,
        reinterpret_cast<uint32_t*>(cu_seq_len_k_start.data_ptr()),
        reinterpret_cast<uint32_t*>(cu_seq_len_k_end.data_ptr()),
        static_cast<float*>(logits.data_ptr()),
        tensor_map_q, tensor_map_kv, tensor_map_kv_scales, tensor_map_weights);
  }

 private:
  // 2D TMA descriptor, mirroring deep_gemm's make_tma_2d_desc (runtime_utils.hpp)
  // for the fixed cases used here (1- or 4-byte elements, swizzle 0 or 128B).
  // NOTE: `gmem_outer_stride` is in ELEMENTS (multiplied by elem_size inside),
  // matching deep_gemm's convention.
  static CUtensorMap make_tma_2d_desc(const tvm::ffi::TensorView& t,
                                      CUtensorMapDataType dtype, int elem_size,
                                      int64_t gmem_inner_dim, int64_t gmem_outer_dim,
                                      int64_t smem_inner_dim, int64_t smem_outer_dim,
                                      int64_t gmem_outer_stride, int swizzle_bytes) {
    if (swizzle_bytes != 0) smem_inner_dim = swizzle_bytes / elem_size;
    CUtensorMapSwizzle swizzle;
    switch (swizzle_bytes) {
      case 0:   swizzle = CU_TENSOR_MAP_SWIZZLE_NONE; break;
      case 128: swizzle = CU_TENSOR_MAP_SWIZZLE_128B; break;
      default:
        CHECK_HOST(false) << "unsupported swizzle " << swizzle_bytes;
        swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
        break;
    }
    CUtensorMap m;
    const cuuint64_t gdim[2] = {(cuuint64_t)gmem_inner_dim, (cuuint64_t)gmem_outer_dim};
    const cuuint32_t sdim[2] = {(cuuint32_t)smem_inner_dim, (cuuint32_t)smem_outer_dim};
    const cuuint64_t gstr[1] = {(cuuint64_t)gmem_outer_stride * (cuuint64_t)elem_size};
    const cuuint32_t estr[2] = {1, 1};
    const auto err = cuTensorMapEncodeTiled(
        &m, dtype, 2, t.data_ptr(), gdim, gstr, sdim, estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CHECK_HOST(err == CUDA_SUCCESS) << "cuTensorMapEncodeTiled failed: " << (int)err;
    return m;
  }
};

}  // namespace sglang
