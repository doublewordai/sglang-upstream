/**
 * \file topk_keys.cuh
 * \brief Shared key-arithmetic + scan helpers for the ka-topk-select kernels
 * (topk_ballot.cuh, topk_prefill_fused.cuh). All bit-identical to the
 * production fg kernel's key arithmetic.
 */
#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace sglang {

namespace {

constexpr uint32_t kRadix = 256;      // 8-bit histogram radix
constexpr uint32_t kMaxTopK = 2048;
constexpr uint32_t kSampleMax = 8192;
constexpr uint32_t kSampleMin = 1024;

SGL_DEVICE uint16_t sortable_f16(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
}

SGL_DEVICE uint8_t coarse16(float x) { return static_cast<uint8_t>(sortable_f16(x) >> 8); }

SGL_DEVICE uint32_t sortable_u32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Inverse of sortable_f16: the float value of a 16-bit sortable key.
SGL_DEVICE float key16_to_value(uint16_t k) {
  uint16_t bits = (k & 0x8000u) ? static_cast<uint16_t>(k & 0x7FFFu) : static_cast<uint16_t>(~k);
  return __half2float(__ushort_as_half(bits));
}

template <typename T>
SGL_DEVICE float to_float(T x);
template <>
SGL_DEVICE float to_float<float>(float x) { return x; }
template <>
SGL_DEVICE float to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
SGL_DEVICE void load4(const T* p, float out[4]);
template <>
SGL_DEVICE void load4<float>(const float* p, float out[4]) {
  float4 v = *reinterpret_cast<const float4*>(p);
  out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
}
template <>
SGL_DEVICE void load4<__nv_bfloat16>(const __nv_bfloat16* p, float out[4]) {
  uint2 u = *reinterpret_cast<const uint2*>(p);
  __nv_bfloat162 a = *reinterpret_cast<__nv_bfloat162*>(&u.x);
  __nv_bfloat162 b = *reinterpret_cast<__nv_bfloat162*>(&u.y);
  out[0] = __bfloat162float(a.x); out[1] = __bfloat162float(a.y);
  out[2] = __bfloat162float(b.x); out[3] = __bfloat162float(b.y);
}

/// Suffix sums over 257 smem ints (256 bins + [256] = 0 sentinel), Hillis-Steele.
SGL_DEVICE void suffix_scan_257(int* buf_a, int* buf_b) {
  const int tid = static_cast<int>(threadIdx.x);
  const bool active = tid < static_cast<int>(kRadix);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int j = 1 << i;
    int* src = (i & 1) ? buf_b : buf_a;
    int* dst = (i & 1) ? buf_a : buf_b;
    int value = 0;
    if (active) {
      value = src[tid];
      if (tid + j <= static_cast<int>(kRadix)) value += src[tid + j];
    }
    __syncthreads();
    if (active) dst[tid] = value;
    __syncthreads();
  }
}

} // namespace
} // namespace sglang
