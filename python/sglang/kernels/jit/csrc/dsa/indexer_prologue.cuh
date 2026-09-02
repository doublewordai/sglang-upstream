// Lane indexer-prologue (branch lane/indexer-prologue).
//
// DSA indexer prologue kernels with the Hadamard rotation KEPT INSIDE, for
// GLM-5.3 (V3.2 backbone: q/k laid out [rope | nope], interleaved rope pairs,
// k = LayerNorm(128) fp32 affine, single 128-dim k head, 32 q heads).
//
// These are forks of deepseek_v4/main_norm_rope.cuh's
// FusedQIndexerRopeHadamardQuantKernel (kRopeFirst=true) and
// deepseek_v32/indexer_k.cuh's FusedKIndexerNormRope[Store]Kernel with the
// production (un-fused) arithmetic preserved exactly:
//
//   Q: rope(fp32) -> round bf16 -> 128-pt Hadamard(fp32) -> *128**-0.5 ->
//      round bf16 -> per-head fp8 quant with POWER-OF-2 scale
//      (act_quant(scale_fmt="ue8m0"): 2^ceil(log2(max(amax,1e-4)/448)),
//      replicated with libdevice log2f/ceilf/exp2f, the same functions
//      Triton's tl.log2/tl.ceil/tl.exp2 lower to)
//      head gate: ((w * n_heads**-0.5) * q_scale) * softmax_scale
//      (production association; the V3.2 kernel folded c1*c2).
//
//   K: LayerNorm(fp32, gamma/beta fp32) -> round bf16 -> rope(fp32) ->
//      round bf16 -> Hadamard(fp32) -> *128**-0.5 -> round bf16 ->
//      per-token fp8 quant with fp32 scale max(1e-4, amax)/448 and
//      inv-scale multiply (fused_store_index_cache.cuh arithmetic) ->
//      paged index-k store (132 B/token: 128 fp8 + 4 fp32 scale).
//
// The intermediate bf16 roundings mirror the production tensors (rope kernel
// writes bf16 in place, hadamard_transform stores bf16, act_quant /
// fused_store read bf16), so given identical GEMM outputs the fused tails are
// bit-exact against the production chain.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>

namespace sglang {

using deepseek_v4::fp8::pack_fp8;

constexpr uint32_t kIndexerPrologueQBlockSize = 256;
constexpr uint32_t kIndexerPrologueQNumWarps = kIndexerPrologueQBlockSize / device::kWarpThreads;
constexpr uint32_t kIndexerPrologueKBlockSize = 128;
constexpr uint32_t kIndexerPrologueKNumWarps = kIndexerPrologueKBlockSize / device::kWarpThreads;

#define PROLOGUE_Q_KERNEL __global__ __launch_bounds__(kIndexerPrologueQBlockSize, 16)
#define PROLOGUE_K_KERNEL __global__ __launch_bounds__(kIndexerPrologueKBlockSize, 16)

template <int64_t kRopeDim>
SGL_DEVICE device::AlignedVector<float, 4>
load_rope_first_cos_sin(const float* __restrict__ cos_sin_cache, int32_t lane_id) {
  constexpr int64_t kHalfRopeDim = kRopeDim / 2;
  const int32_t pair0 = lane_id * 2;
  const int32_t pair1 = pair0 + 1;
  device::AlignedVector<float, 4> freq;
  freq[0] = cos_sin_cache[pair0];
  freq[1] = cos_sin_cache[kHalfRopeDim + pair0];
  freq[2] = cos_sin_cache[pair1];
  freq[3] = cos_sin_cache[kHalfRopeDim + pair1];
  return freq;
}

// 128-point natural-order Walsh-Hadamard butterfly on one warp holding
// 4 elems/lane: 2 local stages (strides 1, 2) + 5 shfl_xor stages (strides
// 4..64). Same stage sequence as fast-hadamard-transform's dim-128 kernel
// (3 in-thread + 4 shuffle stages; identical pairing and fp32 add/sub ops),
// so with identical fp32 inputs the result is bit-identical.
template <bool kRoundBf16, typename DType>
SGL_DEVICE void hadamard_128_inplace(float data[4], uint32_t lane_id, float scale) {
  {
    const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a1;
    data[1] = a0 - a1;
    data[2] = a2 + a3;
    data[3] = a2 - a3;
  }
  {
    const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a2;
    data[1] = a1 + a3;
    data[2] = a0 - a2;
    data[3] = a1 - a3;
  }
#pragma unroll
  for (uint32_t mask = 1; mask < device::kWarpThreads; mask <<= 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float other = __shfl_xor_sync(0xFFFFFFFFu, data[i], mask, device::kWarpThreads);
      data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    data[i] *= scale;
    if constexpr (kRoundBf16) {
      // fast-hadamard-transform stores (x * scale) rounded to bf16.
      data[i] = cast<float>(cast<DType>(data[i]));
    }
  }
}

// ============================================================================
// Q prologue: warp-per-(token, head). rope -> hadamard -> pow2-scale fp8 quant
// + head-gate fold. See file header for the exact arithmetic contract.
// ============================================================================

struct FusedQIndexerPrologueParams {
  const void* __restrict__ q_input;  // (B, H, 128) DType
  void* __restrict__ q_fp8;          // (B, H, 128) fp8_e4m3
  // weights_out[b, h] = ((weight[b,h] * head_gate_scale) * q_scale) * softmax_scale
  const void* __restrict__ weight;   // (B, H) DType
  float* __restrict__ weights_out;   // (B, H, 1) fp32 flat
  float head_gate_scale;             // n_heads ** -0.5
  float softmax_scale;               // head_dim ** -0.5
  float hadamard_scale;              // 128 ** -0.5 (fast-hadamard `scale` arg)
  const float* __restrict__ rope_cache;  // (max_pos, 64) fp32 [cos32 | sin32]
  const void* __restrict__ positions;    // (B,) PosT
  int64_t weight_stride_batch;
  uint32_t batch_size;
  uint32_t num_heads;
};

template <typename DType, typename PosT, bool kUsePDL>
PROLOGUE_Q_KERNEL void fused_q_indexer_prologue(const __grid_constant__ FusedQIndexerPrologueParams params) {
  using namespace device;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;  // = 16
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);

  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;  // 4 fp8 / lane

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kIndexerPrologueQNumWarps + warp_id;
  const bool is_rope_lane = lane_id < kRopeSize;  // leading-dims rope (V3.2 layout)

  const uint32_t total_works = params.batch_size * params.num_heads;
  if (work_id >= total_works) return;

  const uint32_t batch_id = work_id / params.num_heads;
  const auto input_ptr = static_cast<const DType*>(params.q_input) + work_id * kHeadDim;
  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[batch_id]);
  const auto rope_cache = params.rope_cache + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float4 data, freq;
  const uint32_t head_id = work_id - batch_id * params.num_heads;
  const auto weight_val =
      cast<float>(static_cast<const DType*>(params.weight)[batch_id * params.weight_stride_batch + head_id]);

  // part 1: load + rope on the leading 64 dims (interleaved (x[2j], x[2j+1]) pairs).
  {
    Storage input_vec;
    input_vec.load(input_ptr, lane_id);
    if (is_rope_lane) {
      freq = load_rope_first_cos_sin<kRopeDim>(rope_cache, lane_id);
    }
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = cast<float>(input_vec[i]);
    }
  }
  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto fxr = freq[0];
    const auto fxi = freq[1];
    const auto fyr = freq[2];
    const auto fyi = freq[3];
    data[0] = x_real * fxr - x_imag * fxi;
    data[1] = x_real * fxi + x_imag * fxr;
    data[2] = y_real * fyr - y_imag * fyi;
    data[3] = y_real * fyi + y_imag * fyr;
  }
  // Production rope kernel writes bf16 in place.
#pragma unroll
  for (int i = 0; i < kVecSize; ++i)
    data[i] = cast<float>(cast<DType>(data[i]));

  PDLTriggerSecondary<kUsePDL>();

  // part 2: 128-point Hadamard + scale, rounded to bf16 (production tensor dtype).
  hadamard_128_inplace<true, DType>(data, lane_id, params.hadamard_scale);

  // part 3: per-head fp8 quant with the pow2 (ue8m0) scale of
  // act_quant(scale_fmt="ue8m0"): scale = 2^ceil(log2(max(amax,1e-4)/448)).
  {
    float local_max = math::abs(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i) {
      local_max = math::max(local_max, math::abs(data[i]));
    }
    const auto abs_max = warp::reduce_max(local_max);
    const auto amax = fmaxf(abs_max, 1e-4f);
    const float kFp8MaxInv = 1.0f / 448.0f;
    const auto scale = exp2f(ceilf(log2f(amax * kFp8MaxInv)));
    const auto inv_scale = 1.0f / scale;  // exact: scale is a power of two
    OutStorage result;
    result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
    result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);
    auto out_row = static_cast<uint8_t*>(params.q_fp8) + work_id * kHeadDim;
    result.store(out_row, lane_id);
    // Production association: ((w * n_heads**-0.5) * q_scale) * softmax_scale.
    params.weights_out[work_id] = (weight_val * params.head_gate_scale) * scale * params.softmax_scale;
  }
}

template <typename DType, bool kUsePDL>
struct FusedQIndexerPrologueKernel {
  template <typename PosT>
  static constexpr auto kernel = fused_q_indexer_prologue<DType, PosT, kUsePDL>;

  static void forward(
      const tvm::ffi::TensorView q_input,
      const tvm::ffi::TensorView q_fp8,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView weights_out,
      double head_gate_scale,
      double softmax_scale,
      double hadamard_scale,
      const tvm::ffi::TensorView rope_cache,
      const tvm::ffi::TensorView positions) {
    using namespace host;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kRopeDim = 64;

    auto B = SymbolicSize{"batch_size"};
    auto H = SymbolicSize{"num_heads"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, H, kHeadDim})  //
        .with_strides({-1, kHeadDim, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(q_input);
    TensorMatcher({B, H, kHeadDim})  //
        .with_strides({-1, kHeadDim, 1})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device_)
        .verify(q_fp8);
    TensorMatcher({B, H})  //
        .with_strides({-1, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(weight);
    TensorMatcher({B, H, 1})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(weights_out);
    TensorMatcher({-1, kRopeDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(rope_cache);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto num_heads = static_cast<uint32_t>(H.unwrap());
    if (batch_size == 0) return;

    const int64_t expected_batch_stride = static_cast<int64_t>(num_heads) * kHeadDim;
    RuntimeCheck(
        q_input.stride(0) == expected_batch_stride,
        "q_input must be contiguous (B, H, kHeadDim); got stride[0]=",
        q_input.stride(0));
    RuntimeCheck(
        q_fp8.stride(0) == expected_batch_stride,
        "q_fp8 must be contiguous (B, H, kHeadDim); got stride[0]=",
        q_fp8.stride(0));

    const auto params = FusedQIndexerPrologueParams{
        .q_input = q_input.data_ptr(),
        .q_fp8 = q_fp8.data_ptr(),
        .weight = weight.data_ptr(),
        .weights_out = static_cast<float*>(weights_out.data_ptr()),
        .head_gate_scale = static_cast<float>(head_gate_scale),
        .softmax_scale = static_cast<float>(softmax_scale),
        .hadamard_scale = static_cast<float>(hadamard_scale),
        .rope_cache = static_cast<const float*>(rope_cache.data_ptr()),
        .positions = positions.data_ptr(),
        .weight_stride_batch = weight.stride(0),
        .batch_size = batch_size,
        .num_heads = num_heads,
    };
    const auto total_works = batch_size * num_heads;
    const auto num_blocks = div_ceil(total_works, kIndexerPrologueQNumWarps);
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(num_blocks, kIndexerPrologueQBlockSize, device_.unwrap())  //
            .enable_pdl(kUsePDL)(k, params);
  }
};

// ============================================================================
// K prologue: warp-per-token LayerNorm -> rope -> Hadamard -> (store: fp8
// quant + paged write | no-store: bf16 out). See file header.
// ============================================================================

struct FusedKIndexerPrologueParams {
  const void* __restrict__ k_input;         // (B, 128) DType
  void* __restrict__ k_out;                 // (B, 128) DType (no-store variant)
  const float* __restrict__ weight;         // (128,) fp32  -- LayerNorm gamma
  const float* __restrict__ bias;           // (128,) fp32  -- LayerNorm beta
  const float* __restrict__ cos_sin_cache;  // (max_pos, 64) fp32 [cos32 | sin32]
  const void* __restrict__ positions;       // (B,) PosT
  int64_t k_input_stride_batch;
  uint32_t batch_size;
  float eps;
  float hadamard_scale;
};

template <typename DType, typename PosT, bool kUsePDL>
PROLOGUE_K_KERNEL void fused_k_indexer_prologue(const __grid_constant__ FusedKIndexerPrologueParams params) {
  using namespace device;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);

  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kIndexerPrologueKNumWarps + warp_id;
  const bool is_rope_lane = lane_id < kRopeSize;

  if (work_id >= params.batch_size) return;

  const auto input_ptr = static_cast<const DType*>(params.k_input) + work_id * params.k_input_stride_batch;
  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[work_id]);
  const auto cos_sin_cache = params.cos_sin_cache + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float4 data, freq, gamma, beta;

  // part 1: LayerNorm (fp32 math; production module emits bf16).
  {
    Storage input_vec;
    input_vec.load(input_ptr, lane_id);
    gamma.load(params.weight, lane_id);
    beta.load(params.bias, lane_id);
    if (is_rope_lane) freq = load_rope_first_cos_sin<kRopeDim>(cos_sin_cache, lane_id);

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = cast<float>(input_vec[i]);
      sum += data[i];
    }
    const float mean = warp::reduce_sum(sum) / kHeadDim;

    float var = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const float centered = data[i] - mean;
      var += centered * centered;
    }
    const float inv_std = math::rsqrt(warp::reduce_sum(var) / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = (data[i] - mean) * inv_std * gamma[i] + beta[i];
    }
  }
  // Production LayerNorm module returns bf16.
#pragma unroll
  for (int i = 0; i < kVecSize; ++i)
    data[i] = cast<float>(cast<DType>(data[i]));

  // part 2: rope on the leading 64 dims (interleaved pairs).
  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto fxr = freq[0];
    const auto fxi = freq[1];
    const auto fyr = freq[2];
    const auto fyi = freq[3];
    data[0] = x_real * fxr - x_imag * fxi;
    data[1] = x_real * fxi + x_imag * fxr;
    data[2] = y_real * fyr - y_imag * fyi;
    data[3] = y_real * fyi + y_imag * fyr;
  }
  // Production rope kernel writes bf16 in place.
#pragma unroll
  for (int i = 0; i < kVecSize; ++i)
    data[i] = cast<float>(cast<DType>(data[i]));

  PDLTriggerSecondary<kUsePDL>();

  // part 3: Hadamard + scale, rounded to bf16 (production tensor dtype).
  hadamard_128_inplace<true, DType>(data, lane_id, params.hadamard_scale);

  Storage out_vec;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i)
    out_vec[i] = cast<DType>(data[i]);
  auto out_row = static_cast<DType*>(params.k_out) + work_id * kHeadDim;
  out_vec.store(out_row, lane_id);
}

template <typename DType, bool kUsePDL>
struct FusedKIndexerPrologueKernel {
  template <typename PosT>
  static constexpr auto kernel = fused_k_indexer_prologue<DType, PosT, kUsePDL>;

  static void forward(
      const tvm::ffi::TensorView k_input,
      const tvm::ffi::TensorView k_out,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      double eps,
      double hadamard_scale) {
    using namespace host;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kRopeDim = 64;

    auto B = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, kHeadDim})  //
            .with_strides({-1, 1})
            .with_dtype<DType>()
            .with_device(device_)
            .verify(k_input);
    TensorMatcher({B, kHeadDim})  //
        .with_strides({kHeadDim, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(k_out);
    TensorMatcher({kHeadDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(weight);
    TensorMatcher({kHeadDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(bias);
    TensorMatcher({-1, kRopeDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(cos_sin_cache);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;

    const auto params = FusedKIndexerPrologueParams{
        .k_input = k_input.data_ptr(),
        .k_out = k_out.data_ptr(),
        .weight = static_cast<const float*>(weight.data_ptr()),
        .bias = static_cast<const float*>(bias.data_ptr()),
        .cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr()),
        .positions = positions.data_ptr(),
        .k_input_stride_batch = k_input.stride(0),
        .batch_size = batch_size,
        .eps = static_cast<float>(eps),
        .hadamard_scale = static_cast<float>(hadamard_scale),
    };
    const auto num_blocks = div_ceil(batch_size, kIndexerPrologueKNumWarps);
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(num_blocks, kIndexerPrologueKBlockSize, device_.unwrap())  //
            .enable_pdl(kUsePDL)(k, params);
  }
};

// K prologue + fused fp8 quant + paged store. Page layout matches
// fused_store_index_cache.cuh / indexer_k.cuh: 132*page_size bytes per page
// (128*page_size fp8 keys, then 4*page_size fp32 scales).
struct FusedKIndexerPrologueStoreParams {
  const void* __restrict__ k_input;         // (B, 128) DType
  void* __restrict__ cache;                 // (num_pages, 132*page_size) uint8
  const void* __restrict__ indices;         // (B,) int64 -- out_cache_loc
  const float* __restrict__ weight;         // (128,) fp32 -- LayerNorm gamma
  const float* __restrict__ bias;           // (128,) fp32 -- LayerNorm beta
  const float* __restrict__ cos_sin_cache;  // (max_pos, 64) fp32 [cos32 | sin32]
  const void* __restrict__ positions;       // (B,) PosT
  int64_t k_input_stride_batch;
  uint32_t batch_size;
  float eps;
  float hadamard_scale;
};

template <typename DType, typename PosT, bool kUsePDL, int32_t kPageBits>
PROLOGUE_K_KERNEL void fused_k_indexer_prologue_store(const __grid_constant__ FusedKIndexerPrologueStoreParams params) {
  using namespace device;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  constexpr int64_t kPageBytes = 132ll << kPageBits;
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);

  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;  // 4 fp8 / lane

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kIndexerPrologueKNumWarps + warp_id;
  const bool is_rope_lane = lane_id < kRopeSize;

  if (work_id >= params.batch_size) return;

  const auto input_ptr = static_cast<const DType*>(params.k_input) + work_id * params.k_input_stride_batch;
  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[work_id]);
  const auto cos_sin_cache = params.cos_sin_cache + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float4 data, freq, gamma, beta;

  // part 1: LayerNorm (fp32 math; production module emits bf16).
  {
    Storage input_vec;
    input_vec.load(input_ptr, lane_id);
    gamma.load(params.weight, lane_id);
    beta.load(params.bias, lane_id);
    if (is_rope_lane) freq = load_rope_first_cos_sin<kRopeDim>(cos_sin_cache, lane_id);

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = cast<float>(input_vec[i]);
      sum += data[i];
    }
    const float mean = warp::reduce_sum(sum) / kHeadDim;

    float var = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const float centered = data[i] - mean;
      var += centered * centered;
    }
    const float inv_std = math::rsqrt(warp::reduce_sum(var) / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = (data[i] - mean) * inv_std * gamma[i] + beta[i];
    }
  }
#pragma unroll
  for (int i = 0; i < kVecSize; ++i)
    data[i] = cast<float>(cast<DType>(data[i]));

  // part 2: rope on the leading 64 dims (interleaved pairs).
  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto fxr = freq[0];
    const auto fxi = freq[1];
    const auto fyr = freq[2];
    const auto fyi = freq[3];
    data[0] = x_real * fxr - x_imag * fxi;
    data[1] = x_real * fxi + x_imag * fxr;
    data[2] = y_real * fyr - y_imag * fyi;
    data[3] = y_real * fyi + y_imag * fyr;
  }
#pragma unroll
  for (int i = 0; i < kVecSize; ++i)
    data[i] = cast<float>(cast<DType>(data[i]));

  PDLTriggerSecondary<kUsePDL>();

  // part 3: Hadamard + scale, rounded to bf16 (production tensor dtype).
  hadamard_128_inplace<true, DType>(data, lane_id, params.hadamard_scale);

  // part 4: fp8 quant + paged store with the fused_store_index_cache.cuh
  // arithmetic: fp32 scale max(1e-4, amax)/448, inv-scale multiply.
  {
    float local_max = math::abs(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i)
      local_max = math::max(local_max, math::abs(data[i]));
    const auto abs_max = warp::reduce_max(local_max);
    const auto scale = fmaxf(1e-4f, abs_max) / 448.0f;
    const auto inv_scale = 1.0f / scale;

    const auto index = static_cast<const int64_t*>(params.indices)[work_id];
    const int32_t page = static_cast<int32_t>(index >> kPageBits);
    const int32_t offset = static_cast<int32_t>(index & ((1 << kPageBits) - 1));
    const auto page_ptr = static_cast<uint8_t*>(params.cache) + page * kPageBytes;
    const auto value_ptr = page_ptr + offset * kHeadDim;
    const auto scale_ptr = page_ptr + (kHeadDim << kPageBits) + offset * 4;

    OutStorage result;
    result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
    result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);
    reinterpret_cast<OutStorage*>(value_ptr)[lane_id] = result;
    if (lane_id == 0) *reinterpret_cast<float*>(scale_ptr) = scale;
  }
}

template <typename DType, bool kUsePDL, uint32_t kPageSize>
struct FusedKIndexerPrologueStoreKernel {
  static constexpr int32_t kPageBits = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = 132ll * kPageSize;
  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");

  template <typename PosT>
  static constexpr auto kernel = fused_k_indexer_prologue_store<DType, PosT, kUsePDL, kPageBits>;

  static void forward(
      const tvm::ffi::TensorView k_input,
      const tvm::ffi::TensorView cache,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      double eps,
      double hadamard_scale) {
    using namespace host;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kRopeDim = 64;

    auto B = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, kHeadDim})  //
            .with_strides({-1, 1})
            .with_dtype<DType>()
            .with_device(device_)
            .verify(k_input);
    TensorMatcher({-1, -1})  //
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B})  //
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(indices);
    TensorMatcher({kHeadDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(weight);
    TensorMatcher({kHeadDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(bias);
    TensorMatcher({-1, kRopeDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(cos_sin_cache);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;

    const auto params = FusedKIndexerPrologueStoreParams{
        .k_input = k_input.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .weight = static_cast<const float*>(weight.data_ptr()),
        .bias = static_cast<const float*>(bias.data_ptr()),
        .cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr()),
        .positions = positions.data_ptr(),
        .k_input_stride_batch = k_input.stride(0),
        .batch_size = batch_size,
        .eps = static_cast<float>(eps),
        .hadamard_scale = static_cast<float>(hadamard_scale),
    };
    const auto num_blocks = div_ceil(batch_size, kIndexerPrologueKNumWarps);
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(num_blocks, kIndexerPrologueKBlockSize, device_.unwrap())  //
            .enable_pdl(kUsePDL)(k, params);
  }
};

}  // namespace sglang
