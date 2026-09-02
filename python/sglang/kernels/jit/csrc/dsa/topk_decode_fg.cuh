/**
 * \file topk_decode_fg.cuh
 * \brief Full-grid decode-time top-k (k <= 2048) for DSA indexers (GLM / DSv3.2).
 *
 * Replaces sgl_kernel `fast_topk_v2` (= `topk_kernel`, one 1024-thread block per
 * row) for decode / spec-verify shapes: few rows (1..64), long rows (L up to
 * ~1M+). The production kernel is parallelism-starved -- b blocks on a 132-SM
 * GH200 read each row twice from a single SM (~30 GB/s per SM), measuring
 * ~353 us at b=4, L=1M (~90x the 16 MB one-pass byte floor).
 *
 * This implementation spreads each row across the whole grid with a two-phase
 * histogram + gather, reading every row exactly twice:
 *
 *   K1 hist    grid (chunks, B) x 256: per-block smem 256-bin coarse histogram
 *              of the fp16-coarse key (identical key to production), flushed
 *              with global atomics. Row read #1.
 *   K2 plan    grid (B,) x 256: per row suffix-scan the histogram, pick the
 *              threshold coarse bin t (all bin > t selected, count n_gt; the
 *              remaining r = topk - n_gt come from bin == t), zero counters /
 *              hist2, and re-zero hist for the next call.
 *   K3 gather  grid (chunks, B) x 256: row read #2. Per block, bin > t and
 *              bin == t positions (plus each candidate's first refinement
 *              sub-bin) are buffered in shared memory; one global atomicAdd
 *              per counter reserves contiguous slot ranges, then everything is
 *              written without further atomics. Candidates beyond the per-row
 *              cap are dropped (production-equivalent inexactness class, 16x
 *              wider cap); hist2 counts only stored candidates. Rows with
 *              length <= topk take production's naive path instead.
 *   K4 plan2   grid (B,) x 256: suffix-scan hist2 -> round-0 threshold t2,
 *              counts n_gt2 / r2; init the K4a slot counters.
 *   K4a sel0   grid (slices, B) x 256: round-0 selection over the candidate
 *              list in parallel: sub-bin > t2 -> output (block-reserved
 *              range), sub-bin == t2 -> residual list cand_b. Pure streaming
 *              (sub-bins were stored by K3, no value re-reads).
 *   K4b refine grid (B,) x 256: production's exact radix refinement rounds
 *              1..3 (8-bit passes over the remaining key bits) over the small
 *              residual list, selecting the last r2 slots. 32-bit keys make
 *              the selection exact; boundary ties (equal fp32 values) are
 *              resolved arbitrarily, exactly like production.
 *
 * Semantics match `topk_kernel` with row_starts == nullptr:
 *   - scores [B, stride] fp32 (or bf16), unit inner stride, only [0, length)
 *     is read (mask by length; the -inf sentinel may appear in-window and
 *     sorts below everything, +inf above everything, NaN like production's
 *     fp16 key),
 *   - output [B, topk] int32 raw positions in [0, length), arbitrary order;
 *     rows with length <= topk emit i for i < length else -1 (negative
 *     lengths emit all -1),
 *   - inexactness class: if a coarse threshold bin holds more candidates than
 *     the per-row cap, an arbitrary capped subset is refined (production caps
 *     at 4096 in smem; the default cap here is 65536, 16x wider).
 *
 * No host syncs; all launch shapes derive from (stride, B, cap) so the
 * sequence is CUDA-graph safe. Kernels are PDL-enabled (Hopper+): each kernel
 * waits on the previous grid before consuming its output.
 */
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <iterator>

namespace sglang {

namespace {

constexpr uint32_t kRadix = 256;
constexpr uint32_t kBlock = 256;      // all kernels use 256-thread blocks
constexpr uint32_t kChunk = 2048;     // elements per K1/K3 block (8/thread)
constexpr uint32_t kSlice = 2048;     // candidates per K4a block (8/thread)
constexpr uint32_t kMaxTopK = 2048;

struct FGTopKParams {
  const void* __restrict__ scores;     // [B, stride] T
  const int32_t* __restrict__ lengths; // [B]
  int32_t* __restrict__ out;           // [B, topk]
  int32_t* __restrict__ hist;          // [B, 256]
  int32_t* __restrict__ hist2;         // [B, 256]
  int32_t* __restrict__ plan;          // [B, 4] {t, n_gt, r, n_eq}
  int32_t* __restrict__ plan2;         // [B, 4] {t2, out_base, r2, n_gt2}
  int32_t* __restrict__ counters;      // [B, 4] {gt, eq, out, c2}
  int32_t* __restrict__ cand_a;        // [B, cap] coarse-threshold positions
  int32_t* __restrict__ cand_sub;      // [B, cap] ... and their round-0 sub-bins
  int32_t* __restrict__ cand_b;        // [B, cap] round-0 residual positions
  int32_t* __restrict__ stats;         // optional [B, 4] {n_eq, eq_appended, r, inconsistent}
  int64_t stride;
  uint32_t topk;
  uint32_t cap;
  uint32_t aligned; // 16B row alignment -> vectorized loads
};

// --- key conversions: bit-identical to the production kernel ---

SGL_DEVICE uint8_t coarse_bin(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

SGL_DEVICE uint32_t sortable_u32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Full 16-bit fp16-sortable key (monotone in the float value); the high byte
// is the production coarse bin, the low byte discriminates one coarse bin at
// ~0.05% relative granularity (a monotone, lossy projection of fp32).
SGL_DEVICE uint16_t sortable_f16(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
}

template <typename T>
SGL_DEVICE float to_float(T x);
template <>
SGL_DEVICE float to_float<float>(float x) { return x; }
template <>
SGL_DEVICE float to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

// 4-element vector loads (16B for fp32, 8B for bf16); requires 4-element
// alignment of the chunk base (guaranteed: chunk starts are multiples of
// kChunk and rows are >= 8B aligned when p.aligned).
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
/// After the call buf_a[i] = sum_{j=i..256} buf[j] (initial). All threads call
/// this (threads with tid >= 256 only participate in the barriers).
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

// --- K1: coarse histogram (row read #1) ---

template <bool kUsePDL, typename T>
__global__ __launch_bounds__(kBlock) void fg_hist_kernel(const FGTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.y;
  const int32_t length = p.lengths[row];
  if (length <= static_cast<int32_t>(p.topk)) { // naive row: no read needed
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int64_t beg = static_cast<int64_t>(blockIdx.x) * kChunk;
  if (beg >= length) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int64_t end = min(beg + kChunk, static_cast<int64_t>(length));

  __shared__ int s_hist[kRadix];
  for (uint32_t i = threadIdx.x; i < kRadix; i += kBlock) s_hist[i] = 0;
  __syncthreads();

  const T* rowp = reinterpret_cast<const T*>(p.scores) + static_cast<int64_t>(row) * p.stride;
  auto process = [&](int64_t pos, float x) {
    atomicAdd(&s_hist[coarse_bin(x)], 1);
  };
  if (p.aligned) {
    const int64_t n4 = (end - beg) >> 2; // beg % 4 == 0
    for (int64_t g = threadIdx.x; g < n4; g += kBlock) {
      float x[4];
      load4<T>(rowp + beg + g * 4, x);
#pragma unroll
      for (int e = 0; e < 4; ++e) process(beg + g * 4 + e, x[e]);
    }
    for (int64_t i = beg + (n4 << 2) + threadIdx.x; i < end; i += kBlock)
      process(i, to_float(rowp[i]));
  } else {
    for (int64_t i = beg + threadIdx.x; i < end; i += kBlock) process(i, to_float(rowp[i]));
  }
  __syncthreads();
  if (threadIdx.x < kRadix) {
    const int v = s_hist[threadIdx.x];
    if (v != 0)
      atomicAdd(&p.hist[static_cast<int64_t>(row) * kRadix + threadIdx.x], v);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// --- K2: per-row plan (one block per row) ---

template <bool kUsePDL>
__global__ __launch_bounds__(kBlock) void fg_plan_kernel(const FGTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.x;
  __shared__ int s_scan[2][kRadix + 1];
  s_scan[0][kRadix] = 0;
  s_scan[1][kRadix] = 0;

  // zero hist2 + counters for K3; hist[row] is re-zeroed below for the next
  // call (it is only consumed by this kernel).
  if (threadIdx.x < kRadix)
    p.hist2[static_cast<int64_t>(row) * kRadix + threadIdx.x] = 0;
  if (threadIdx.x == 0) {
    p.counters[row * 4 + 0] = 0;
    p.counters[row * 4 + 1] = 0;
  }

  const int32_t length = p.lengths[row];
  if (length <= static_cast<int32_t>(p.topk)) {
    if (threadIdx.x == 0) { // naive row: plan {t=-1, n_gt=length, r=0, n_eq=length}
      p.plan[row * 4 + 0] = -1;
      p.plan[row * 4 + 1] = length;
      p.plan[row * 4 + 2] = 0;
      p.plan[row * 4 + 3] = length;
    }
    if (threadIdx.x < kRadix)
      p.hist[static_cast<int64_t>(row) * kRadix + threadIdx.x] = 0;
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  if (threadIdx.x < kRadix)
    s_scan[0][threadIdx.x] = p.hist[static_cast<int64_t>(row) * kRadix + threadIdx.x];
  __syncthreads();
  // done with hist[row]: re-zero for the next call
  if (threadIdx.x < kRadix)
    p.hist[static_cast<int64_t>(row) * kRadix + threadIdx.x] = 0;

  suffix_scan_257(s_scan[0], s_scan[1]); // s_scan[0][i] = #{coarse bin >= i}
  if (threadIdx.x < kRadix) {
    const int topk = static_cast<int>(p.topk);
    if (s_scan[0][threadIdx.x] > topk && s_scan[0][threadIdx.x + 1] <= topk) {
      const int t = static_cast<int>(threadIdx.x);
      const int n_gt = s_scan[0][t + 1];
      const int n_eq = s_scan[0][t] - n_gt;
      p.plan[row * 4 + 0] = t;
      p.plan[row * 4 + 1] = n_gt;
      p.plan[row * 4 + 2] = topk - n_gt;
      p.plan[row * 4 + 3] = n_eq;
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// --- K3: gather (row read #2), block-local aggregation ---

template <bool kUsePDL, typename T>
__global__ __launch_bounds__(kBlock) void fg_gather_kernel(const FGTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.y;
  const int32_t length = p.lengths[row];
  if (length <= static_cast<int32_t>(p.topk)) { // naive (also covers length <= 0)
    if (blockIdx.x == 0) {
      for (uint32_t j = threadIdx.x; j < p.topk; j += kBlock)
        p.out[static_cast<int64_t>(row) * p.topk + j] =
            (static_cast<int32_t>(j) < length) ? static_cast<int32_t>(j) : -1;
    }
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int32_t t = p.plan[row * 4 + 0];
  const int32_t r = p.plan[row * 4 + 2];
  const int64_t beg = static_cast<int64_t>(blockIdx.x) * kChunk;
  if (beg >= length) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int64_t end = min(beg + kChunk, static_cast<int64_t>(length));
  const T* rowp = reinterpret_cast<const T*>(p.scores) + static_cast<int64_t>(row) * p.stride;
  int32_t* out_row = p.out + static_cast<int64_t>(row) * p.topk;

  __shared__ int s_gt_pos[kChunk];
  __shared__ int s_eq_pos[kChunk];
  __shared__ int s_eq_sub[kChunk];
  __shared__ int s_sub[kRadix];
  __shared__ int s_gt_n, s_eq_n, s_gt_base, s_eq_base;
  if (threadIdx.x == 0) {
    s_gt_n = 0;
    s_eq_n = 0;
  }
  for (uint32_t i = threadIdx.x; i < kRadix; i += kBlock) s_sub[i] = 0;
  __syncthreads();

  auto process = [&](int64_t pos, float x) {
    const int32_t bin = coarse_bin(x);
    if (bin > t) {
      s_gt_pos[atomicAdd(&s_gt_n, 1)] = static_cast<int32_t>(pos);
    } else if (bin == t && r > 0) {
      const int slot = atomicAdd(&s_eq_n, 1);
      s_eq_pos[slot] = static_cast<int32_t>(pos);
      // sub-key: low byte of the fp16-sortable key (finer than the coarse
      // bin, still a monotone key, so round-0 filtering on it is sound)
      const int sub = static_cast<int>(sortable_f16(x) & 0xFFu);
      s_eq_sub[slot] = sub;
      atomicAdd(&s_sub[sub], 1);
    }
  };
  if (p.aligned) {
    const int64_t n4 = (end - beg) >> 2;
    for (int64_t g = threadIdx.x; g < n4; g += kBlock) {
      float x[4];
      load4<T>(rowp + beg + g * 4, x);
#pragma unroll
      for (int e = 0; e < 4; ++e) process(beg + g * 4 + e, x[e]);
    }
    for (int64_t i = beg + (n4 << 2) + threadIdx.x; i < end; i += kBlock)
      process(i, to_float(rowp[i]));
  } else {
    for (int64_t i = beg + threadIdx.x; i < end; i += kBlock) process(i, to_float(rowp[i]));
  }
  __syncthreads();
  // one global atomic per counter per block: reserve contiguous slot ranges
  if (threadIdx.x == 0) {
    s_gt_base = atomicAdd(&p.counters[row * 4 + 0], s_gt_n);
    s_eq_base = atomicAdd(&p.counters[row * 4 + 1], s_eq_n);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < s_gt_n; i += kBlock)
    out_row[s_gt_base + i] = s_gt_pos[i];

  if (r > 0 && s_eq_base < static_cast<int32_t>(p.cap)) {
    const int n_store = min(s_eq_n, static_cast<int32_t>(p.cap) - s_eq_base);
    int32_t* cand = p.cand_a + static_cast<int64_t>(row) * p.cap + s_eq_base;
    int32_t* cand_sub = p.cand_sub + static_cast<int64_t>(row) * p.cap + s_eq_base;
    for (int i = threadIdx.x; i < n_store; i += kBlock) {
      cand[i] = s_eq_pos[i];
      cand_sub[i] = s_eq_sub[i];
    }
    if (n_store == s_eq_n) {
      // whole local set stored: flush the local sub-histogram as counted
      for (uint32_t b = threadIdx.x; b < kRadix; b += kBlock)
        if (s_sub[b] != 0)
          atomicAdd(&p.hist2[static_cast<int64_t>(row) * kRadix + b], s_sub[b]);
    } else {
      // cap boundary: hist2 must count only the stored prefix
      for (int i = threadIdx.x; i < n_store; i += kBlock)
        atomicAdd(&p.hist2[static_cast<int64_t>(row) * kRadix + s_eq_sub[i]], 1);
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// --- K4 plan2: round-0 threshold from hist2 ---

template <bool kUsePDL>
__global__ __launch_bounds__(kBlock) void fg_plan2_kernel(const FGTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.x;
  const int32_t length = p.lengths[row];
  const int32_t r0 = p.plan[row * 4 + 2];
  const int32_t n_gt = p.plan[row * 4 + 1];
  if (threadIdx.x == 0 && p.stats != nullptr) {
    p.stats[row * 4 + 0] = p.plan[row * 4 + 3];
    p.stats[row * 4 + 1] = p.counters[row * 4 + 1];
    p.stats[row * 4 + 2] = r0;
    p.stats[row * 4 + 3] = 0;
  }
  if (length <= static_cast<int32_t>(p.topk) || r0 <= 0) {
    if (threadIdx.x == 0) {
      p.plan2[row * 4 + 0] = -1;
      p.plan2[row * 4 + 1] = n_gt;
      p.plan2[row * 4 + 2] = 0;
      p.plan2[row * 4 + 3] = 0;
    }
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  __shared__ int s_scan[2][kRadix + 1];
  s_scan[0][kRadix] = 0;
  s_scan[1][kRadix] = 0;
  if (threadIdx.x < kRadix)
    s_scan[0][threadIdx.x] = p.hist2[static_cast<int64_t>(row) * kRadix + threadIdx.x];
  __syncthreads();
  suffix_scan_257(s_scan[0], s_scan[1]); // s_scan[0][i] = #{sub-bin >= i}
  if (threadIdx.x < kRadix) {
    if (s_scan[0][threadIdx.x] > r0 && s_scan[0][threadIdx.x + 1] <= r0) {
      const int t2 = static_cast<int>(threadIdx.x);
      const int n_gt2 = s_scan[0][t2 + 1];
      p.plan2[row * 4 + 0] = t2;
      p.plan2[row * 4 + 1] = n_gt + n_gt2;
      p.plan2[row * 4 + 2] = r0 - n_gt2;
      p.plan2[row * 4 + 3] = n_gt2;
    }
  }
  // K4a slot counters
  if (threadIdx.x == 0) {
    p.counters[row * 4 + 2] = n_gt; // out slot base for round-0 selections
    p.counters[row * 4 + 3] = 0;    // residual list appends
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// --- K4a: round-0 selection over the candidate list (multi-block, streaming) ---

template <bool kUsePDL>
__global__ __launch_bounds__(kBlock) void fg_select0_kernel(const FGTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.y;
  const int32_t r0 = p.plan[row * 4 + 2];
  if (r0 <= 0) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int32_t length = p.lengths[row];
  if (length <= static_cast<int32_t>(p.topk)) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int32_t t2 = p.plan2[row * 4 + 0];
  const int32_t r2 = p.plan2[row * 4 + 2];
  if (t2 < 0) { // inconsistent (proved impossible when r0 > 0)
    if (threadIdx.x == 0 && p.stats != nullptr) p.stats[row * 4 + 3] = 1;
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int32_t n_stored = min(p.counters[row * 4 + 1], static_cast<int32_t>(p.cap));
  const int64_t beg = static_cast<int64_t>(blockIdx.x) * kSlice;
  if (beg >= n_stored) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int end = static_cast<int>(min(beg + kSlice, static_cast<int64_t>(n_stored)));

  __shared__ int s_gt_pos[kSlice];
  __shared__ int s_eq_pos[kSlice];
  __shared__ int s_gt_n, s_eq_n, s_gt_base, s_eq_base;
  if (threadIdx.x == 0) {
    s_gt_n = 0;
    s_eq_n = 0;
  }
  __syncthreads();

  const int32_t* cand = p.cand_a + static_cast<int64_t>(row) * p.cap;
  const int32_t* cand_sub = p.cand_sub + static_cast<int64_t>(row) * p.cap;
  for (int i = static_cast<int>(beg) + threadIdx.x; i < end; i += kBlock) {
    const int32_t pos = cand[i];
    const int32_t bin = cand_sub[i];
    if (bin > t2) {
      s_gt_pos[atomicAdd(&s_gt_n, 1)] = pos;
    } else if (bin == t2 && r2 > 0) {
      s_eq_pos[atomicAdd(&s_eq_n, 1)] = pos;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    s_gt_base = atomicAdd(&p.counters[row * 4 + 2], s_gt_n);
    s_eq_base = atomicAdd(&p.counters[row * 4 + 3], s_eq_n);
  }
  __syncthreads();
  int32_t* out_row = p.out + static_cast<int64_t>(row) * p.topk;
  for (int i = threadIdx.x; i < s_gt_n; i += kBlock)
    out_row[s_gt_base + i] = s_gt_pos[i];
  for (int i = threadIdx.x; i < s_eq_n; i += kBlock)
    p.cand_b[static_cast<int64_t>(row) * p.cap + s_eq_base + i] = s_eq_pos[i];
  device::PDLTriggerSecondary<kUsePDL>();
}

// --- K4b: exact fp32 radix refinement over the small residual list ---
// The residual shares only its fp16-key low byte, so the selection is refined
// from scratch on the full fp32 sortable key (4 x 8-bit rounds, exactly the
// production refinement) over a small list (typically tens of entries).

template <bool kUsePDL, typename T>
__global__ __launch_bounds__(kBlock) void fg_refine_kernel(const FGTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.x;
  const int32_t length = p.lengths[row];
  const int32_t r0 = p.plan[row * 4 + 2];
  const int32_t r2 = p.plan2[row * 4 + 2];
  const int32_t n2 = p.counters[row * 4 + 3];
  if (length <= static_cast<int32_t>(p.topk) || r0 <= 0 || r2 <= 0 || n2 <= 0) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const T* rowp = reinterpret_cast<const T*>(p.scores) + static_cast<int64_t>(row) * p.stride;
  int32_t* out_row = p.out + static_cast<int64_t>(row) * p.topk;
  const int32_t* src = p.cand_b + static_cast<int64_t>(row) * p.cap;
  int32_t* dst = p.cand_a + static_cast<int64_t>(row) * p.cap; // reuse (K4a done with it)

  __shared__ int s_hist[2][kRadix + 1];
  __shared__ int s_ctr, s_next_cnt, s_last_remain, s_t;
  int topk_rem = r2;

  // round-0 histogram (fp32 key bits 24..31), built inline over the residual
  for (uint32_t i = threadIdx.x; i <= kRadix; i += kBlock) s_hist[0][i] = 0;
  if (threadIdx.x == 0) s_hist[1][kRadix] = 0; // scan sentinel
  __syncthreads();
  for (int i = threadIdx.x; i < n2; i += kBlock) {
    const float x = to_float(rowp[src[i]]);
    atomicAdd(&s_hist[0][(sortable_u32(x) >> 24) & 0xFFu], 1);
  }
  if (threadIdx.x == 0) s_ctr = p.plan2[row * 4 + 1]; // n_gt + n_gt2
  __syncthreads();
  int src_n = n2;

#pragma unroll 1
  for (int round = 0; round < 4; ++round) {
    const int shift = 24 - 8 * round;
    if (threadIdx.x == 0) {
      s_t = -1;
      s_next_cnt = 0;
    }
    __syncthreads();
    suffix_scan_257(s_hist[round & 1], s_hist[1 - (round & 1)]);
    if (threadIdx.x < kRadix) {
      const int* suf = s_hist[round & 1];
      if (suf[threadIdx.x] > topk_rem && suf[threadIdx.x + 1] <= topk_rem) {
        s_t = static_cast<int>(threadIdx.x);
        s_last_remain = topk_rem - suf[threadIdx.x + 1];
      }
    }
    __syncthreads();
    const int t = s_t;
    if (t < 0) { // inconsistent (proved impossible); bail without writing
      if (threadIdx.x == 0 && p.stats != nullptr) p.stats[row * 4 + 3] = 1;
      device::PDLTriggerSecondary<kUsePDL>();
      return;
    }
    topk_rem -= s_hist[round & 1][t + 1];
    const bool last = (round == 3) || (topk_rem == 0);
    if (!last) {
      for (uint32_t i = threadIdx.x; i <= kRadix; i += kBlock)
        s_hist[1 - (round & 1)][i] = 0;
      __syncthreads();
    }
    for (int i = threadIdx.x; i < src_n; i += kBlock) {
      const int32_t pos = src[i];
      const float x = to_float(rowp[pos]);
      const int32_t bin = static_cast<int32_t>((sortable_u32(x) >> shift) & 0xFFu);
      if (bin > t) {
        const int slot = atomicAdd(&s_ctr, 1);
        out_row[slot] = pos;
      } else if (bin == t) {
        if (round == 3) { // final: take any topk_rem of the ties (production semantics)
          const int old = atomicAdd(&s_last_remain, -1);
          if (old > 0) out_row[p.topk - old] = pos;
        } else if (topk_rem > 0) {
          const int slot = atomicAdd(&s_next_cnt, 1);
          dst[slot] = pos;
          atomicAdd(&s_hist[1 - (round & 1)][(sortable_u32(x) >> (shift - 8)) & 0xFFu], 1);
        }
      }
    }
    __syncthreads();
    if (last) break;
    src_n = s_next_cnt;
    const int32_t* tmp = src;
    src = dst;
    dst = const_cast<int32_t*>(tmp);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

} // namespace

template <bool kUsePDL>
struct TopKDecodeFG {
  static constexpr uint32_t kMaxRows = 65535; // grid.y limit for K1/K3/K4a

  template <typename T>
  static void run_t(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView workspace,
      const uint32_t cap,
      const tvm::ffi::Optional<tvm::ffi::TensorView> stats) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"score_stride"};
    auto W = SymbolicSize{"workspace_ints"};
    auto K = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, -1}).with_strides({S, 1}).with_dtype<T>().with_device(device).verify(scores);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device).verify(out);
    TensorMatcher({W}).with_dtype<int32_t>().with_device(device).verify(workspace);

    int32_t* stats_ptr = nullptr;
    if (stats.has_value()) {
      TensorMatcher({B, 4}).with_dtype<int32_t>().with_device(device).verify(stats.value());
      stats_ptr = static_cast<int32_t*>(stats.value().data_ptr());
    }

    const auto batch = static_cast<uint32_t>(B.unwrap());
    const auto stride = S.unwrap();
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");
    RuntimeCheck(batch > 0 && batch <= kMaxRows, "batch too large for grid.y");
    RuntimeCheck(cap > topk, "cap must exceed topk");

    // [hist | hist2 | plan | plan2 | counters | cand_a | cand_sub | cand_b]
    const int64_t n_hist = static_cast<int64_t>(batch) * kRadix;
    const int64_t n_small = static_cast<int64_t>(batch) * (2 * kRadix + 4 + 4 + 4);
    const int64_t need = n_small + 3 * static_cast<int64_t>(batch) * cap;
    RuntimeCheck(static_cast<int64_t>(W.unwrap()) >= need, "workspace too small");

    auto* ws = static_cast<int32_t*>(workspace.data_ptr());
    const auto* scores_ptr = static_cast<const T*>(scores.data_ptr());
    const bool aligned =
        (reinterpret_cast<uintptr_t>(scores_ptr) % 16 == 0) && (stride % 4 == 0);

    FGTopKParams p{
        .scores = scores_ptr,
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .out = static_cast<int32_t*>(out.data_ptr()),
        .hist = ws,
        .hist2 = ws + n_hist,
        .plan = ws + 2 * n_hist,
        .plan2 = ws + 2 * n_hist + 4 * batch,
        .counters = ws + 2 * n_hist + 8 * batch,
        .cand_a = ws + n_small,
        .cand_sub = ws + n_small + static_cast<int64_t>(batch) * cap,
        .cand_b = ws + n_small + 2 * static_cast<int64_t>(batch) * cap,
        .stats = stats_ptr,
        .stride = stride,
        .topk = topk,
        .cap = cap,
        .aligned = aligned ? 1u : 0u,
    };

    const uint32_t chunks = static_cast<uint32_t>((stride + kChunk - 1) / kChunk);
    const uint32_t slices = (cap + kSlice - 1) / kSlice;
    const dim3 grid_row_chunks(chunks, batch);
    const dim3 grid_slices(slices, batch);
    LaunchKernel(grid_row_chunks, kBlock, device.unwrap()).enable_pdl(kUsePDL)(fg_hist_kernel<kUsePDL, T>, p);
    LaunchKernel(batch, kBlock, device.unwrap()).enable_pdl(kUsePDL)(fg_plan_kernel<kUsePDL>, p);
    LaunchKernel(grid_row_chunks, kBlock, device.unwrap()).enable_pdl(kUsePDL)(fg_gather_kernel<kUsePDL, T>, p);
    LaunchKernel(batch, kBlock, device.unwrap()).enable_pdl(kUsePDL)(fg_plan2_kernel<kUsePDL>, p);
    LaunchKernel(grid_slices, kBlock, device.unwrap()).enable_pdl(kUsePDL)(fg_select0_kernel<kUsePDL>, p);
    LaunchKernel(batch, kBlock, device.unwrap()).enable_pdl(kUsePDL)(fg_refine_kernel<kUsePDL, T>, p);
  }

  static void run(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView workspace,
      const uint32_t cap,
      const tvm::ffi::Optional<tvm::ffi::TensorView> stats) {
    const auto dt = scores.dtype();
    if (dt.code == DLDataTypeCode::kDLFloat && dt.bits == 32 && dt.lanes == 1) {
      run_t<fp32_t>(scores, lengths, out, workspace, cap, stats);
    } else if (dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16 && dt.lanes == 1) {
      run_t<bf16_t>(scores, lengths, out, workspace, cap, stats);
    } else {
      host::RuntimeCheck(false, "scores must be fp32 or bf16");
    }
  }
};

} // namespace sglang
