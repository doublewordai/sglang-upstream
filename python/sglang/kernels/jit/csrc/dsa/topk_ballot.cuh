/**
 * \file topk_ballot.cuh
 * \brief Warp-ballot one-read exact top-k (k <= 2048) for DSA indexers.
 *
 * Design 2 of the decode-attention program (lane ka-topk-issue). Fixes the four
 * measured costs that made the persistent byte-floor design
 * (topk_decode_floor.cuh) lose to the fg chain (0.30-0.75x):
 *
 *   1. Issue-bound capture (per-element smem-atomic histogram + append):
 *      the capture pass below has NO per-element smem atomic and NO histogram.
 *      Per element: one FSETP (predicate !(x < v_s), NaN-inclusive) + 1/4
 *      VOTE.ballot per element slot; warp-popc running counts; (key, pos)
 *      pairs staged per warp without atomics, flushed with ONE global
 *      atomicAdd per CTA per chunk.
 *   2. Serial owner-CTA sample (5-8 us): a separate tiny sample kernel with a
 *      TWO-LEVEL threshold (coarse bin + sub-bin = the full 16-bit fp16 key)
 *      -- slices the crowded fp16-coarse bins that make production's 1-pass
 *      prefill select fall back on every row (prefill-energy's finding).
 *   3. Owner-CTA select (10-19 us): the select reuses the fg chain's
 *      multi-CTA machinery (plan2 / select0 slices grid / refine) on the
 *      captured candidate list.
 *   4. Grid barriers (9.5 us): none. A PDL launch chain exactly like fg's;
 *      no co-residency constraint.
 *
 * Chain (all 256-thread blocks, PDL-enabled):
 *   S1 sample    (B): strided sample -> two-level 16-bit key threshold K_s;
 *                   conservative value v_s = value(key K_s - 1), so the value
 *                   predicate x >= v_s captures a SUPERSET of {key >= K_s}.
 *                   Rows with length <= cap skip the sample (v_s = -inf:
 *                   capture-all, exact, never misses). Zeroes the counters.
 *   S2 capture   (chunks, B): THE read. pred = !(x < v_s); warp ballots
 *                   compact matches into a per-warp smem region (no atomics);
 *                   one global atomicAdd per CTA reserves a contiguous range
 *                   of the per-row candidate list (pos + full 16-bit key).
 *                   counters[0] accumulates the EXACT match count even past
 *                   the cap (ballots count, stores truncate). Naive rows
 *                   (length <= topk) write their output here.
 *   V  plan      (B): fast path iff topk <= n <= cap -- the captured list
 *                   then holds EVERY element >= v_s, a superset of the true
 *                   top-k, so the select over it is exact regardless of the
 *                   sample's quality (the sample only gates speed). Builds
 *                   the fg warm-path plan from the candidate keys (coarse
 *                   suffix scan -> tc, n_gt; sub-hist of the boundary bin ->
 *                   hist2). Miss rows set fast[row] = 0 and take the fg
 *                   2-pass fallback below (exact, fg's cap class).
 *   F1 hist      (chunks, B): fg's K1 gated per row on !fast.
 *   F2 plan      (B): fg's K2, gated.
 *   F3 gather    (chunks, B): fg's K3, gated (low-byte sub-key semantics).
 *   P2 plan2     (B): fg's plan2 for both paths.
 *   S3 select0   (slices, B): fg's select0; fast rows take the warm branch
 *                   (full 16-bit key), fallback rows the 2-pass branch.
 *   S4 refine    (B): fg's exact 4-round fp32 radix refinement over the small
 *                   residual list; re-zeroes hist2 (self-cleaning).
 *   TF transform (ceil(topk/kBlock), B): prefill entry only: dst =
 *                   page_table[seq(row)][pos], production semantics (incl.
 *                   the naive path's first-`length`-entries quirk).
 *
 * Semantics = fg's (decode entry): scores [B, stride] fp32/bf16 unit inner
 * stride, only [row_starts[row], +length) read (row_starts = 0 at decode);
 * output [B, topk] int32 window-local positions (raw at decode), arbitrary
 * order, -1 padding for length <= topk rows (negative lengths all -1),
 * boundary ties arbitrary (fg's documented rule). Inexactness class = fg's:
 * only the fallback path can cap-truncate, and only when one coarse bin holds
 * more than `cap` candidates; the fast path is always exact.
 *
 * The capture predicate !(x < v_s) is NaN-inclusive (NaN sorts above
 * everything under the fp16 key, matching fg); -inf sentinels sort below.
 *
 * Self-cleaning across calls / CUDA-graph replays: S1 zeroes counters[0..1];
 * V rewrites fast[]; F2 re-zeroes hist after consuming it (fallback rows);
 * fast rows never touch hist (stays zero); S4 re-zeroes hist2; plan / plan2 /
 * counters[2..3] are fully rewritten each call by V/P2.
 */
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace {

constexpr uint32_t kRadix = 256;
constexpr uint32_t kBlock = 512;      // all kernels use 512-thread blocks
constexpr uint32_t kChunk = 4096;     // elements per fallback hist/gather sub-chunk
constexpr uint32_t kCapChunk = 4096;  // elements per capture CTA (v3 structure; the
                                      // overflow flush stays as a safety valve)
constexpr uint32_t kWarps = kBlock / 32;
constexpr uint32_t kWarpElems = kCapChunk / kWarps + 16;  // per-warp staging region
constexpr uint32_t kSlice = 2048;     // candidates per select0 block
constexpr uint32_t kMaxTopK = 2048;
constexpr uint32_t kSampleMax = 8192;
constexpr uint32_t kSampleMin = 1024;

struct BallotTopKParams {
  const void* __restrict__ scores;        // [B, stride] T
  const int32_t* __restrict__ lengths;    // [B]
  const int32_t* __restrict__ row_starts; // [B] or nullptr (decode)
  int32_t* __restrict__ out;              // [B, topk]
  float* __restrict__ v_s;                // [B] capture threshold (S1 out)
  int32_t* __restrict__ hist;             // [B, 256] (fallback F1/F2)
  int32_t* __restrict__ hist2;            // [B, 256] (round-0 sub-bin hist)
  int32_t* __restrict__ plan;             // [B, 4] {t/tc, n_gt, r, n_eq/n}
  int32_t* __restrict__ plan2;            // [B, 4] {t2, out_base, r2, n_gt2}
  int32_t* __restrict__ counters;         // [B, 4] {n_total, n_stored, out, c2}
  int32_t* __restrict__ fast;             // [B] 1 = fast path (fg's ws_flags)
  int32_t* __restrict__ seq;              // [B] page-table row (prefill)
  const int32_t* __restrict__ page_table; // [BS, pt_stride] (prefill)
  const int32_t* __restrict__ cu_seqlens; // [BS + 1] (prefill)
  int32_t* __restrict__ cand_pos;         // [B, cap]
  int32_t* __restrict__ cand_key;         // [B, cap] full key (fast) / low byte (fallback)
  int32_t* __restrict__ cand_b;           // [B, cap] round-0 residual
  int32_t* __restrict__ stats;            // optional [B, 4] {n, n_stored, r, flags}
  int64_t stride;
  int64_t pt_stride;
  uint32_t topk;
  uint32_t cap;
  uint32_t target;      // capture-count target the sample aims at
  uint32_t prefill_bs;
  uint32_t cap_chunk;   // elements per capture CTA (decode: 4096; prefill: whole row)
};

// --- key conversions: bit-identical to the production kernel (fg) ---

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

SGL_DEVICE uint16_t sortable_f16(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
}

// Inverse of sortable_f16: the float value of a 16-bit sortable key.
SGL_DEVICE float key_to_value(uint16_t k) {
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

// ---------------------------------------------------------------------------
// S1: sample -> two-level (16-bit key) threshold, conservative value
// ---------------------------------------------------------------------------

template <bool kUsePDL, typename T>
__global__ __launch_bounds__(kBlock) void ballot_sample_kernel(const BallotTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.x;
  if (threadIdx.x == 0) { // counters + capture-hist for S2 (self-cleaning across calls)
    p.counters[row * 4 + 0] = 0;
    p.counters[row * 4 + 1] = 0;
  }
  for (uint32_t i = threadIdx.x; i < kRadix; i += kBlock)
    p.hist[static_cast<int64_t>(row) * kRadix + i] = 0;
  const int32_t length = p.lengths[row];
  if (length <= static_cast<int32_t>(p.topk) ||
      static_cast<uint32_t>(length) <= p.cap) {
    // naive row (no capture) or capture-all (n = length <= cap: exact, never
    // misses) -- either way no sample is needed.
    if (threadIdx.x == 0) p.v_s[row] = __int_as_float(0xff800000); // -inf
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  // windowed sample (W windows of E consecutive elements): alias-resistant,
  // unlike a single stride (a period-p input pattern aliases a stride-m sample)
  const uint32_t n_s = min(kSampleMax, max(kSampleMin, static_cast<uint32_t>(length) / 8)) & ~63u;
  const uint32_t W = 64;
  const uint32_t E = n_s / W;
  __shared__ uint16_t s_keys[kSampleMax];
  __shared__ int s_hist[kWarps][kRadix];  // warp-privatized (no cross-warp atomic contention)
  __shared__ int s_scan[2][kRadix + 1];
  __shared__ int s_sub[kRadix];
  __shared__ int s_tc, s_tsub, s_above;
  s_scan[0][kRadix] = 0;
  s_scan[1][kRadix] = 0;
  for (uint32_t i = threadIdx.x; i < kWarps * kRadix; i += kBlock)
    (&s_hist[0][0])[i] = 0;
  for (uint32_t i = threadIdx.x; i < kRadix; i += kBlock)
    s_sub[i] = 0;
  if (threadIdx.x == 0) s_tc = 0;
  __syncthreads();

  const T* rowp = reinterpret_cast<const T*>(p.scores) + static_cast<int64_t>(row) * p.stride +
                  (p.row_starts != nullptr ? p.row_starts[row] : 0);
  const uint32_t warp = threadIdx.x / 32;
  for (uint32_t t = threadIdx.x; t < n_s; t += kBlock) {
    const uint32_t c = t / E;
    const uint32_t e = t % E;
    uint32_t pos = static_cast<uint32_t>(
        (static_cast<uint64_t>(c) * static_cast<uint64_t>(length)) / W) + e;
    pos = min(pos, static_cast<uint32_t>(length) - 1);
    const uint16_t key = sortable_f16(to_float(rowp[pos]));
    s_keys[t] = key;
    atomicAdd(&s_hist[warp][key >> 8], 1);
  }
  __syncthreads();
  // reduce the privatized histograms into s_scan[0]
  if (threadIdx.x < kRadix) {
    int acc = 0;
    for (uint32_t w = 0; w < kWarps; ++w) acc += s_hist[w][threadIdx.x];
    s_scan[0][threadIdx.x] = acc;
  }
  __syncthreads();

  // sample crossing count: E[#captured] ~ target
  uint32_t j = (p.target * n_s + static_cast<uint32_t>(length) - 1) / static_cast<uint32_t>(length);
  j = max(j, 1u);
  j = min(j, n_s);

  __syncthreads();
  suffix_scan_257(s_scan[0], s_scan[1]); // s_scan[0][i] = #{coarse >= i}
  if (threadIdx.x < kRadix) {
    // largest coarse bin with sample suffix >= j (exists: suffix[0] = n_s >= j)
    if (s_scan[0][threadIdx.x] >= static_cast<int>(j) &&
        s_scan[0][threadIdx.x + 1] < static_cast<int>(j))
      s_tc = static_cast<int>(threadIdx.x);
  }
  __syncthreads();
  const int tc = s_tc;
  if (tc <= 0) { // threshold below everything: capture all
    if (threadIdx.x == 0) p.v_s[row] = __int_as_float(0xff800000); // -inf
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  if (threadIdx.x == 0) s_above = s_scan[0][tc + 1]; // sample count with coarse > tc
  __syncthreads();
  const int above = s_above;

  // sub-bin pass over the sampled keys in the boundary coarse bin
  for (uint32_t t = threadIdx.x; t < n_s; t += kBlock) {
    const uint16_t key = s_keys[t];
    if ((key >> 8) == static_cast<uint16_t>(tc)) atomicAdd(&s_sub[key & 0xFFu], 1);
  }
  __syncthreads();
  // total(t_sub) = above + subsuf[t_sub], non-increasing, total(0) >= j
  // (suffix[tc] >= j by tc's choice). Largest t_sub with total >= j.
  if (threadIdx.x < kRadix) s_scan[0][threadIdx.x] = s_sub[threadIdx.x];
  s_scan[0][kRadix] = 0;
  s_scan[1][kRadix] = 0;
  __syncthreads();
  suffix_scan_257(s_scan[0], s_scan[1]); // s_scan[0][i] = #{sub >= i} within tc
  if (threadIdx.x < kRadix) {
    if (above + s_scan[0][threadIdx.x] >= static_cast<int>(j) &&
        above + s_scan[0][threadIdx.x + 1] < static_cast<int>(j))
      s_tsub = static_cast<int>(threadIdx.x);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const uint32_t K_s = (static_cast<uint32_t>(tc) << 8) | static_cast<uint32_t>(s_tsub);
    // conservative: capture {x >= value(K_s - 1)} >= {key >= K_s}
    p.v_s[row] = (K_s > 0) ? key_to_value(static_cast<uint16_t>(K_s - 1))
                           : __int_as_float(0xff800000);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// S2: the one read -- warp-ballot compaction capture
// ---------------------------------------------------------------------------

template <bool kUsePDL, typename T>
__global__ __launch_bounds__(kBlock) void ballot_capture_kernel(const BallotTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.y;
  const int32_t length = p.lengths[row];
  if (length <= static_cast<int32_t>(p.topk)) { // naive (also covers <= 0)
    if (blockIdx.x == 0) {
      int32_t* out_row = p.out + static_cast<int64_t>(row) * p.topk;
      for (uint32_t j = threadIdx.x; j < p.topk; j += kBlock)
        out_row[j] = (static_cast<int32_t>(j) < length) ? static_cast<int32_t>(j) : -1;
    }
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int64_t lo = static_cast<int64_t>(blockIdx.x) * p.cap_chunk; // window-local
  if (lo >= length) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int64_t hi = min(lo + p.cap_chunk, static_cast<int64_t>(length));
  const T* rowp = reinterpret_cast<const T*>(p.scores) + static_cast<int64_t>(row) * p.stride +
                  (p.row_starts != nullptr ? p.row_starts[row] : 0);
  const float v_s = p.v_s[row];

  __shared__ uint2 s_ent[kWarps][kWarpElems]; // (key, pos) per-warp staging
  __shared__ int s_chist[kRadix];             // captured-keys' coarse bins (for V)
  __shared__ uint32_t s_wtot[kWarps];
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  uint32_t wc = 0; // this warp's region cursor
  for (uint32_t i = threadIdx.x; i < kRadix; i += kBlock) s_chist[i] = 0;
  if (lane == 0) s_wtot[warp] = 0;
  __syncthreads();

  // warp-local flush: histogram the region, reserve a global range, copy out
  auto flush_warp = [&]() {
    if (wc == 0) return;
    for (uint32_t i = lane; i < wc; i += 32) {
      atomicAdd(&s_chist[s_ent[warp][i].x >> 8], 1);
      const uint32_t base = 0; // filled below
      (void)base;
    }
    // reserve AFTER histogramming (both read s_ent)
    const uint32_t base = (lane == 0)
        ? atomicAdd(reinterpret_cast<unsigned*>(&p.counters[row * 4 + 0]), wc)
        : 0;
    const uint32_t b = __shfl_sync(0xFFFFFFFFu, base, 0);
    const int64_t rbase = static_cast<int64_t>(row) * p.cap + b;
    for (uint32_t i = lane; i < wc; i += 32) {
      if (b + i < p.cap) {
        p.cand_pos[rbase + i] = static_cast<int32_t>(s_ent[warp][i].y);
        p.cand_key[rbase + i] = static_cast<int32_t>(s_ent[warp][i].x);
      }
    }
    if (lane == 0) s_wtot[warp] += wc;
    wc = 0;
  };

  // head (to 16B alignment) -- all threads participate in the ballot
  {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(rowp + lo);
    uint32_t head = static_cast<uint32_t>((16u - (addr & 15u)) & 15u) / sizeof(T);
    if (lo + head > hi) head = static_cast<uint32_t>(hi - lo);
    bool pr = false;
    uint32_t k = 0;
    if (threadIdx.x < head) {
      const float x = to_float(rowp[lo + threadIdx.x]);
      pr = !(x < v_s);
      k = pr ? sortable_f16(x) : 0;
    }
    const uint32_t m = __ballot_sync(0xFFFFFFFFu, pr);
    if (pr) {
      const uint32_t rank = __popc(m & ((1u << lane) - 1u));
      s_ent[warp][wc + rank] = make_uint2(k, static_cast<uint32_t>(lo + threadIdx.x));
    }
    wc += __popc(m);
    // body: float4 groups, software-pipelined (two loads in flight/thread)
    const uint32_t n4 = (static_cast<uint32_t>(hi - lo) - head) >> 2;
    const uint32_t iters = (n4 + kBlock - 1) / kBlock;
    const T* body = rowp + lo + head;
    float pre[4] = {0.f, 0.f, 0.f, 0.f};
    if (threadIdx.x < n4) load4<T>(body + threadIdx.x * 4, pre);
#pragma unroll 1
    for (uint32_t k2 = 0; k2 < iters; ++k2) {
      const uint32_t g = threadIdx.x + k2 * kBlock;
      const bool have = g < n4;
      const uint32_t gn = g + kBlock;
      float x[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) x[e] = pre[e];
      if (gn < n4) load4<T>(body + gn * 4, pre);
      const uint32_t base = static_cast<uint32_t>(lo) + head + g * 4;
      bool pred[4];
      uint32_t key[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        pred[e] = have && !(x[e] < v_s);
        key[e] = pred[e] ? sortable_f16(x[e]) : 0;
      }
      if (wc + 128 > kWarpElems) flush_warp();  // uniform across the warp
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const uint32_t m4 = __ballot_sync(0xFFFFFFFFu, pred[e]);
        if (m4 == 0) continue;
        if (pred[e]) {
          const uint32_t rank = __popc(m4 & ((1u << lane) - 1u));
          s_ent[warp][wc + rank] = make_uint2(key[e], base + e);
        }
        wc += __popc(m4);
      }
    }
    // tail
    const uint32_t tail_start = head + (((static_cast<uint32_t>(hi - lo) - head) >> 2) << 2);
    const uint32_t tail_n = static_cast<uint32_t>(hi - lo) - tail_start;
    bool tpr = false;
    uint32_t tk = 0;
    if (threadIdx.x < tail_n) {
      const float x = to_float(rowp[lo + tail_start + threadIdx.x]);
      tpr = !(x < v_s);
      tk = tpr ? sortable_f16(x) : 0;
    }
    if (wc + 32 > kWarpElems) flush_warp();  // tail overflow guard
    const uint32_t tm = __ballot_sync(0xFFFFFFFFu, tpr);
    if (tpr) {
      const uint32_t rank = __popc(tm & ((1u << lane) - 1u));
      s_ent[warp][wc + rank] = make_uint2(tk, static_cast<uint32_t>(lo + tail_start + threadIdx.x));
    }
    wc += __popc(tm);
  }

  // ---- final flush + per-CTA coarse histogram out ----
  flush_warp();
  __syncthreads();
  if (threadIdx.x < kRadix) {
    const int v = s_chist[threadIdx.x];
    if (v != 0)
      atomicAdd(&p.hist[static_cast<int64_t>(row) * kRadix + threadIdx.x], v);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// V: verify + plan (fast path) / flag the fallback
// ---------------------------------------------------------------------------

template <bool kUsePDL>
__global__ __launch_bounds__(kBlock) void ballot_plan_kernel(const BallotTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.x;
  // page-table row (prefill): needed by the transform for every row
  if (p.cu_seqlens != nullptr) {
    for (uint32_t i = threadIdx.x; i < p.prefill_bs; i += kBlock) {
      const int32_t a = p.cu_seqlens[i];
      const int32_t b = p.cu_seqlens[i + 1];
      if (static_cast<int32_t>(row) >= a && static_cast<int32_t>(row) < b)
        p.seq[row] = static_cast<int32_t>(i);
    }
  }
  const int32_t length = p.lengths[row];
  if (length <= static_cast<int32_t>(p.topk)) { // naive: S2 wrote the output
    if (threadIdx.x == 0) {
      p.fast[row] = 1;
      p.plan[row * 4 + 0] = -1;
      p.plan[row * 4 + 1] = length;
      p.plan[row * 4 + 2] = 0;
      p.plan[row * 4 + 3] = length;
      if (p.stats != nullptr) {
        p.stats[row * 4 + 0] = length;
        p.stats[row * 4 + 1] = length;
        p.stats[row * 4 + 2] = 0;
        p.stats[row * 4 + 3] = 0;
      }
    }
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const uint32_t n = static_cast<uint32_t>(p.counters[row * 4 + 0]); // exact capture count
  const bool ok = (n >= p.topk && n <= p.cap);
  if (!ok) {
    if (threadIdx.x == 0) p.fast[row] = 0;
    if (threadIdx.x < kRadix) // F1 rebuilds the full-row histogram
      p.hist[static_cast<int64_t>(row) * kRadix + threadIdx.x] = 0;
    if (threadIdx.x == 0 && p.stats != nullptr) {
      p.stats[row * 4 + 0] = static_cast<int32_t>(n);
      p.stats[row * 4 + 1] = static_cast<int32_t>(min(n, p.cap));
      p.stats[row * 4 + 2] = 0;
      p.stats[row * 4 + 3] = 1 | (n > p.cap ? 2 : 4); // fallback | over | under
    }
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  // fast path: exact select over the captured list (fg's WS2 machinery).
  // The coarse histogram of the captured keys was built during the capture
  // pass; here we only consume it (no per-entry atomics in this kernel).
  __shared__ int s_scan[2][kRadix + 1];
  __shared__ int s_sub[kRadix];
  __shared__ int s_tc, s_ngt;
  s_scan[0][kRadix] = 0;
  s_scan[1][kRadix] = 0;
  for (uint32_t i = threadIdx.x; i < kRadix; i += kBlock)
    s_sub[i] = 0;
  if (threadIdx.x == 0) p.fast[row] = 1;
  __syncthreads();

  const int n_sto = static_cast<int>(n); // n <= cap: everything was stored
  if (threadIdx.x < kRadix)
    s_scan[0][threadIdx.x] = p.hist[static_cast<int64_t>(row) * kRadix + threadIdx.x];
  __syncthreads();
  suffix_scan_257(s_scan[0], s_scan[1]); // s_scan[0][i] = #{coarse >= i}
  if (threadIdx.x < kRadix) {
    const int topk = static_cast<int>(p.topk);
    // '>=': the exactly-covered case must select a bin, not bail (fg WS2);
    // plateau choices are tie-equivalent.
    if (s_scan[0][threadIdx.x] >= topk && s_scan[0][threadIdx.x + 1] <= topk) {
      s_tc = static_cast<int>(threadIdx.x);
      s_ngt = s_scan[0][threadIdx.x + 1];
    }
  }
  __syncthreads();
  const int tc = s_tc;
  const int n_gt = s_ngt;
  const int32_t* cand_key = p.cand_key + static_cast<int64_t>(row) * p.cap;
  for (int i = threadIdx.x; i < n_sto; i += kBlock) {
    const int key = cand_key[i];
    if ((key >> 8) == tc) atomicAdd(&s_sub[key & 0xFFu], 1);
  }
  __syncthreads();
  if (threadIdx.x < kRadix)
    p.hist2[static_cast<int64_t>(row) * kRadix + threadIdx.x] = s_sub[threadIdx.x];
  if (threadIdx.x == 0) {
    p.plan[row * 4 + 0] = tc;
    p.plan[row * 4 + 1] = n_gt;
    p.plan[row * 4 + 2] = static_cast<int>(p.topk) - n_gt;
    p.plan[row * 4 + 3] = static_cast<int>(n);
    p.counters[row * 4 + 1] = static_cast<int32_t>(n); // n_stored
    p.counters[row * 4 + 2] = 0;                       // sel0 out-slot base (warm)
    p.counters[row * 4 + 3] = 0;                       // residual appends
    if (p.stats != nullptr) {
      p.stats[row * 4 + 0] = static_cast<int32_t>(n);
      p.stats[row * 4 + 1] = static_cast<int32_t>(n);
      p.stats[row * 4 + 2] = static_cast<int>(p.topk) - n_gt;
      p.stats[row * 4 + 3] = 0;
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

/// Stream [beg, end) of a row with vectorized loads (16B-aligned body,
/// scalar head/tail). fn(int64_t pos, float x) per element. No ballots here
/// (fn is called by a subset of threads only).
template <typename T, typename F>
SGL_DEVICE void stream_aligned(const T* rowp, int64_t beg, int64_t end, F&& fn) {
  const uintptr_t addr = reinterpret_cast<uintptr_t>(rowp + beg);
  uint32_t head = static_cast<uint32_t>((16u - (addr & 15u)) & 15u) / sizeof(T);
  if (beg + head > end) head = static_cast<uint32_t>(end - beg);
  if (threadIdx.x < head) fn(beg + threadIdx.x, to_float(rowp[beg + threadIdx.x]));
  const uint32_t n4 = (static_cast<uint32_t>(end - beg) - head) >> 2;
  const T* body = rowp + beg + head;
  for (uint32_t g = threadIdx.x; g < n4; g += kBlock) {
    float x[4];
    load4<T>(body + g * 4, x);
    const int64_t base = beg + head + static_cast<int64_t>(g) * 4;
#pragma unroll
    for (int e = 0; e < 4; ++e) fn(base + e, x[e]);
  }
  const uint32_t tail_start = head + (n4 << 2);
  const uint32_t tail_n = static_cast<uint32_t>(end - beg) - tail_start;
  if (threadIdx.x < tail_n)
    fn(beg + tail_start + threadIdx.x, to_float(rowp[beg + tail_start + threadIdx.x]));
}

// ---------------------------------------------------------------------------
// F1: fallback coarse histogram (fg's K1, gated on !fast, window offset)
// ---------------------------------------------------------------------------

template <bool kUsePDL, typename T>
__global__ __launch_bounds__(kBlock) void ballot_hist_kernel(const BallotTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.y;
  if (p.fast[row]) { // fast path already selected this row
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int32_t length = p.lengths[row];
  if (length <= static_cast<int32_t>(p.topk)) { // unreachable (naive => fast)
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  constexpr uint32_t kFB = 8;  // chunks per block: shrinks the gated grid 8x
  const int64_t cbase = static_cast<int64_t>(blockIdx.x) * kFB;
  if (cbase * kChunk >= length) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  __shared__ int s_hist[kRadix];
  for (uint32_t i = threadIdx.x; i < kRadix; i += kBlock) s_hist[i] = 0;
  __syncthreads();

  const T* rowp = reinterpret_cast<const T*>(p.scores) + static_cast<int64_t>(row) * p.stride +
                  (p.row_starts != nullptr ? p.row_starts[row] : 0);
  for (uint32_t ci = 0; ci < kFB; ++ci) {
    const int64_t beg = (cbase + ci) * kChunk;
    if (beg >= length) break;
    const int64_t end = min(beg + kChunk, static_cast<int64_t>(length));
    stream_aligned<T>(rowp, beg, end, [&](int64_t pos, float x) {
      atomicAdd(&s_hist[coarse_bin(x)], 1);
    });
  }
  __syncthreads();
  if (threadIdx.x < kRadix) {
    const int v = s_hist[threadIdx.x];
    if (v != 0)
      atomicAdd(&p.hist[static_cast<int64_t>(row) * kRadix + threadIdx.x], v);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// F2: fallback plan (fg's K2, gated)
// ---------------------------------------------------------------------------

template <bool kUsePDL>
__global__ __launch_bounds__(kBlock) void ballot_plan2pre_kernel(const BallotTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.x;
  if (p.fast[row]) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  __shared__ int s_scan[2][kRadix + 1];
  s_scan[0][kRadix] = 0;
  s_scan[1][kRadix] = 0;

  // zero hist2 + counters for F3; hist[row] is re-zeroed below for the next
  // call (it is only consumed by this kernel).
  if (threadIdx.x < kRadix)
    p.hist2[static_cast<int64_t>(row) * kRadix + threadIdx.x] = 0;
  if (threadIdx.x == 0) {
    p.counters[row * 4 + 0] = 0;
    p.counters[row * 4 + 1] = 0;
  }

  const int32_t length = p.lengths[row];
  if (length <= static_cast<int32_t>(p.topk)) { // unreachable (naive => fast)
    if (threadIdx.x == 0) {
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

// ---------------------------------------------------------------------------
// F3: fallback gather (fg's K3, gated, window offset)
// ---------------------------------------------------------------------------

template <bool kUsePDL, typename T>
__global__ __launch_bounds__(kBlock) void ballot_gather_kernel(const BallotTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.y;
  if (p.fast[row]) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int32_t length = p.lengths[row];
  if (length <= static_cast<int32_t>(p.topk)) { // unreachable (naive => fast)
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int32_t t = p.plan[row * 4 + 0];
  const int32_t r = p.plan[row * 4 + 2];
  constexpr uint32_t kFB = 8;  // chunks per block: shrinks the gated grid 8x
  const int64_t cbase = static_cast<int64_t>(blockIdx.x) * kFB;
  if (cbase * kChunk >= length) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const T* rowp = reinterpret_cast<const T*>(p.scores) + static_cast<int64_t>(row) * p.stride +
                  (p.row_starts != nullptr ? p.row_starts[row] : 0);
  int32_t* out_row = p.out + static_cast<int64_t>(row) * p.topk;

  __shared__ int s_gt_pos[kChunk];
  __shared__ int s_eq_pos[kChunk];
  __shared__ int s_eq_sub[kChunk];
  __shared__ int s_sub[kRadix];
  __shared__ int s_gt_n, s_eq_n, s_gt_base, s_eq_base;

  auto process = [&](int64_t pos, float x) {
    const int32_t bin = coarse_bin(x);
    if (bin > t) {
      s_gt_pos[atomicAdd(&s_gt_n, 1)] = static_cast<int32_t>(pos);
    } else if (bin == t && r > 0) {
      const int slot = atomicAdd(&s_eq_n, 1);
      s_eq_pos[slot] = static_cast<int32_t>(pos);
      const int sub = static_cast<int>(sortable_f16(x) & 0xFFu);
      s_eq_sub[slot] = sub;
      atomicAdd(&s_sub[sub], 1);
    }
  };
#pragma unroll 1
  for (uint32_t ci = 0; ci < kFB; ++ci) {
    const int64_t beg = (cbase + ci) * kChunk;
    if (beg >= length) break;
    const int64_t end = min(beg + kChunk, static_cast<int64_t>(length));
    if (threadIdx.x == 0) {
      s_gt_n = 0;
      s_eq_n = 0;
    }
    for (uint32_t i = threadIdx.x; i < kRadix; i += kBlock) s_sub[i] = 0;
    __syncthreads();
    stream_aligned<T>(rowp, beg, end, process);
    __syncthreads();
    if (threadIdx.x == 0) {
      s_gt_base = atomicAdd(&p.counters[row * 4 + 0], s_gt_n);
      s_eq_base = atomicAdd(&p.counters[row * 4 + 1], s_eq_n);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < s_gt_n; i += kBlock)
      out_row[s_gt_base + i] = s_gt_pos[i];

    if (r > 0 && s_eq_base < static_cast<int32_t>(p.cap)) {
      const int n_store = min(s_eq_n, static_cast<int32_t>(p.cap) - s_eq_base);
      int32_t* cand = p.cand_pos + static_cast<int64_t>(row) * p.cap + s_eq_base;
      int32_t* cand_sub = p.cand_key + static_cast<int64_t>(row) * p.cap + s_eq_base;
      for (int i = threadIdx.x; i < n_store; i += kBlock) {
        cand[i] = s_eq_pos[i];
        cand_sub[i] = s_eq_sub[i];
      }
      if (n_store == s_eq_n) {
        for (uint32_t b = threadIdx.x; b < kRadix; b += kBlock)
          if (s_sub[b] != 0)
            atomicAdd(&p.hist2[static_cast<int64_t>(row) * kRadix + b], s_sub[b]);
      } else {
        for (int i = threadIdx.x; i < n_store; i += kBlock)
          atomicAdd(&p.hist2[static_cast<int64_t>(row) * kRadix + s_eq_sub[i]], 1);
      }
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// P2: round-0 threshold from hist2 (fg's plan2; warm := fast)
// ---------------------------------------------------------------------------

template <bool kUsePDL>
__global__ __launch_bounds__(kBlock) void ballot_plan2_kernel(const BallotTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.x;
  const int32_t length = p.lengths[row];
  const int32_t r0 = p.plan[row * 4 + 2];
  const int32_t n_gt = p.plan[row * 4 + 1];
  const bool warm = (p.fast[row] != 0);
  // S3 slot counters. Initialized BEFORE the early exit: warm rows with
  // r0 == 0 still run select0 (it writes ALL output slots), base 0; fallback
  // rows: F3 already wrote [0, n_gt), so their base is n_gt.
  if (threadIdx.x == 0) {
    p.counters[row * 4 + 2] = warm ? 0 : n_gt;
    p.counters[row * 4 + 3] = 0;
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
    // '>=': the exactly-covered case must select the boundary bin instead of
    // bailing (fg plan2); unreachable on the fallback path (n_eq > r strictly).
    if (s_scan[0][threadIdx.x] >= r0 && s_scan[0][threadIdx.x + 1] <= r0) {
      const int t2 = static_cast<int>(threadIdx.x);
      const int n_gt2 = s_scan[0][t2 + 1];
      p.plan2[row * 4 + 0] = t2;
      p.plan2[row * 4 + 1] = n_gt + n_gt2;
      p.plan2[row * 4 + 2] = r0 - n_gt2;
      p.plan2[row * 4 + 3] = n_gt2;
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// S3: round-0 selection over the candidate list (fg's select0; warm := fast)
// ---------------------------------------------------------------------------

template <bool kUsePDL>
__global__ __launch_bounds__(kBlock) void ballot_select0_kernel(const BallotTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.y;
  const bool warm = (p.fast[row] != 0);
  const int32_t r0 = p.plan[row * 4 + 2];
  if (!warm && r0 <= 0) { // fallback: nothing left to select (F3 wrote all)
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
  if (!warm && t2 < 0) { // inconsistent (proved impossible on the fallback path)
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

  const int32_t tc = p.plan[row * 4 + 0]; // warm: coarse threshold bin
  const int32_t* cand = p.cand_pos + static_cast<int64_t>(row) * p.cap;
  const int32_t* cand_sub = p.cand_key + static_cast<int64_t>(row) * p.cap;
  for (int i = static_cast<int>(beg) + threadIdx.x; i < end; i += kBlock) {
    const int32_t pos = cand[i];
    const int key = cand_sub[i];
    if (warm) {
      const int c = key >> 8;
      if (c > tc) {
        s_gt_pos[atomicAdd(&s_gt_n, 1)] = pos;
      } else if (c == tc && t2 >= 0) {
        const int low = key & 0xFFu;
        if (low > t2) {
          s_gt_pos[atomicAdd(&s_gt_n, 1)] = pos;
        } else if (low == t2 && r2 > 0) {
          s_eq_pos[atomicAdd(&s_eq_n, 1)] = pos;
        }
      }
    } else {
      if (key > t2) {
        s_gt_pos[atomicAdd(&s_gt_n, 1)] = pos;
      } else if (key == t2 && r2 > 0) {
        s_eq_pos[atomicAdd(&s_eq_n, 1)] = pos;
      }
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

// ---------------------------------------------------------------------------
// S4: exact fp32 radix refinement over the small residual list (fg's refine)
// ---------------------------------------------------------------------------

template <bool kUsePDL, typename T>
__global__ __launch_bounds__(kBlock) void ballot_refine_kernel(const BallotTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.x;
  // self-clean hist2 for the next call (before any early exit)
  for (uint32_t i = threadIdx.x; i < kRadix; i += kBlock)
    p.hist2[static_cast<int64_t>(row) * kRadix + i] = 0;
  const int32_t length = p.lengths[row];
  const int32_t r0 = p.plan[row * 4 + 2];
  const int32_t r2 = p.plan2[row * 4 + 2];
  const int32_t n2 = p.counters[row * 4 + 3];
  if (length <= static_cast<int32_t>(p.topk) || r0 <= 0 || r2 <= 0 || n2 <= 0) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const T* rowp = reinterpret_cast<const T*>(p.scores) + static_cast<int64_t>(row) * p.stride +
                  (p.row_starts != nullptr ? p.row_starts[row] : 0);
  int32_t* out_row = p.out + static_cast<int64_t>(row) * p.topk;
  const int32_t* src = p.cand_b + static_cast<int64_t>(row) * p.cap;
  int32_t* dst = p.cand_pos + static_cast<int64_t>(row) * p.cap; // reuse (S3 done with it)

  __shared__ int s_hist[2][kRadix + 1];
  __shared__ int s_ctr, s_next_cnt, s_last_remain, s_t;
  int topk_rem = r2;

  for (uint32_t i = threadIdx.x; i <= kRadix; i += kBlock) s_hist[0][i] = 0;
  if (threadIdx.x == 0) s_hist[1][kRadix] = 0;
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
      if (suf[threadIdx.x] >= topk_rem && suf[threadIdx.x + 1] <= topk_rem) {
        s_t = static_cast<int>(threadIdx.x);
        s_last_remain = topk_rem - suf[threadIdx.x + 1];
      }
    }
    __syncthreads();
    const int t = s_t;
    if (t < 0) { // inconsistent (proved impossible); bail without writing
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
        if (round == 3) { // final: take any topk_rem of the ties (fg semantics)
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

// ---------------------------------------------------------------------------
// TF: page-table transform (prefill entry only; production semantics)
// ---------------------------------------------------------------------------

template <bool kUsePDL>
__global__ __launch_bounds__(kBlock) void ballot_transform_kernel(const BallotTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.y;
  int32_t* out_row = p.out + static_cast<int64_t>(row) * p.topk;
  const int32_t* pt_row = p.page_table + static_cast<int64_t>(p.seq[row]) * p.pt_stride;
  // stage this block's slots in smem first: no in-plane global read-after-write
  // hazard within the kernel (reads and writes are separated by a barrier)
  __shared__ int32_t s_raw[kBlock];
  const uint32_t t0 = blockIdx.x * kBlock;
  if (t0 + threadIdx.x < p.topk) s_raw[threadIdx.x] = out_row[t0 + threadIdx.x];
  __syncthreads();
  if (t0 + threadIdx.x < p.topk) {
    const int32_t raw = s_raw[threadIdx.x];
    if (raw < 0) out_row[t0 + threadIdx.x] = -1;
    else if (raw >= static_cast<int32_t>(p.stride)) out_row[t0 + threadIdx.x] = -2 - raw;  // expose stale (<-2)
    else out_row[t0 + threadIdx.x] = pt_row[raw];
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

} // namespace

template <bool kUsePDL>
struct TopKBallot {
  static constexpr uint32_t kMaxRows = 65535; // grid.y limit

  template <typename T>
  static void run_t(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView workspace,
      const uint32_t cap,
      const uint32_t target,
      const tvm::ffi::Optional<tvm::ffi::TensorView> row_starts,
      const tvm::ffi::Optional<tvm::ffi::TensorView> stats,
      const tvm::ffi::Optional<tvm::ffi::TensorView> page_table,
      const tvm::ffi::Optional<tvm::ffi::TensorView> cu_seqlens,
      const bool prefill) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"score_stride"};
    auto W = SymbolicSize{"workspace_ints"};
    auto K = SymbolicSize{"topk"};
    auto BS = SymbolicSize{"prefill_bs"};
    auto BSp1 = SymbolicSize{"prefill_bs_plus_1"};
    auto P = SymbolicSize{"page_table_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, -1}).with_strides({S, 1}).with_dtype<T>().with_device(device).verify(scores);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device).verify(out);
    TensorMatcher({W}).with_dtype<int32_t>().with_device(device).verify(workspace);

    const int32_t* rs_ptr = nullptr;
    if (row_starts.has_value()) {
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(row_starts.value());
      rs_ptr = static_cast<const int32_t*>(row_starts.value().data_ptr());
    }
    int32_t* stats_ptr = nullptr;
    if (stats.has_value()) {
      TensorMatcher({B, 4}).with_dtype<int32_t>().with_device(device).verify(stats.value());
      stats_ptr = static_cast<int32_t*>(stats.value().data_ptr());
    }
    const int32_t* pt_ptr = nullptr;
    const int32_t* cu_ptr = nullptr;
    int64_t pt_stride = 0;
    uint32_t prefill_bs = 0;
    if (prefill) {
      TensorMatcher({BS, -1}).with_strides({P, 1}).with_dtype<int32_t>().with_device(device).verify(page_table.value());
      TensorMatcher({BSp1}).with_dtype<int32_t>().with_device(device).verify(cu_seqlens.value());
      RuntimeCheck(BSp1.unwrap() >= 2, "cu_seqlens must have >= 2 entries");
      pt_ptr = static_cast<const int32_t*>(page_table.value().data_ptr());
      cu_ptr = static_cast<const int32_t*>(cu_seqlens.value().data_ptr());
      pt_stride = P.unwrap();
      prefill_bs = static_cast<uint32_t>(BS.unwrap());
    }

    const auto batch = static_cast<uint32_t>(B.unwrap());
    const auto stride = S.unwrap();
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");
    RuntimeCheck(batch > 0 && batch <= kMaxRows, "batch too large for grid.y");
    RuntimeCheck(cap > topk, "cap must exceed topk");

    // [v_s | plan | plan2 | counters | fast | seq | hist | hist2 | cand_pos | cand_key | cand_b]
    int64_t o = 0;
    auto* ws = static_cast<int32_t*>(workspace.data_ptr());
    float* v_s = reinterpret_cast<float*>(ws + o);        o += batch;
    int32_t* plan = ws + o;                               o += 4 * batch;
    int32_t* plan2 = ws + o;                              o += 4 * batch;
    int32_t* counters = ws + o;                           o += 4 * batch;
    int32_t* fast = ws + o;                               o += batch;
    int32_t* seq = ws + o;                                o += batch;
    int32_t* hist = ws + o;                               o += static_cast<int64_t>(batch) * kRadix;
    int32_t* hist2 = ws + o;                              o += static_cast<int64_t>(batch) * kRadix;
    int32_t* cand_pos = ws + o;                           o += static_cast<int64_t>(batch) * cap;
    int32_t* cand_key = ws + o;                           o += static_cast<int64_t>(batch) * cap;
    int32_t* cand_b = ws + o;                             o += static_cast<int64_t>(batch) * cap;
    RuntimeCheck(static_cast<int64_t>(W.unwrap()) >= o, "workspace too small");

    BallotTopKParams p{
        .scores = scores.data_ptr(),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .row_starts = rs_ptr,
        .out = static_cast<int32_t*>(out.data_ptr()),
        .v_s = v_s,
        .hist = hist,
        .hist2 = hist2,
        .plan = plan,
        .plan2 = plan2,
        .counters = counters,
        .fast = fast,
        .seq = seq,
        .page_table = pt_ptr,
        .cu_seqlens = cu_ptr,
        .cand_pos = cand_pos,
        .cand_key = cand_key,
        .cand_b = cand_b,
        .stats = stats_ptr,
        .stride = stride,
        .pt_stride = pt_stride,
        .topk = topk,
        .cap = cap,
        .target = target,
        .prefill_bs = prefill_bs,
        .cap_chunk = prefill ? 0xFFFFFFFFu : 4096u,
    };

    const uint32_t chunks = prefill ? 1u : static_cast<uint32_t>((stride + 4095) / 4096);
    const uint32_t sub_chunks = static_cast<uint32_t>((stride + kChunk - 1) / kChunk);
    const uint32_t slices = (cap + kSlice - 1) / kSlice;
    const dim3 grid_chunks(chunks, batch);
    const dim3 grid_fb((sub_chunks + 7) / 8, batch);  // gated fallback: 8 sub-chunks/block
    const dim3 grid_slices(slices, batch);

    LaunchKernel(batch, kBlock, device.unwrap())(ballot_sample_kernel<kUsePDL, T>, p);
    LaunchKernel(grid_chunks, kBlock, device.unwrap())(ballot_capture_kernel<kUsePDL, T>, p);
    LaunchKernel(batch, kBlock, device.unwrap())(ballot_plan_kernel<kUsePDL>, p);
    LaunchKernel(grid_fb, kBlock, device.unwrap())(ballot_hist_kernel<kUsePDL, T>, p);
    LaunchKernel(batch, kBlock, device.unwrap())(ballot_plan2pre_kernel<kUsePDL>, p);
    LaunchKernel(grid_fb, kBlock, device.unwrap())(ballot_gather_kernel<kUsePDL, T>, p);
    LaunchKernel(batch, kBlock, device.unwrap())(ballot_plan2_kernel<kUsePDL>, p);
    LaunchKernel(grid_slices, kBlock, device.unwrap())(ballot_select0_kernel<kUsePDL>, p);
    LaunchKernel(batch, kBlock, device.unwrap())(ballot_refine_kernel<kUsePDL, T>, p);
    if (prefill) {
      const dim3 grid_tf((topk + kBlock - 1) / kBlock, batch);
      // no PDL on the transform: it consumes S4's output; a plain launch is
      // stream-ordered (fully serialized), avoiding the early-trigger window
      LaunchKernel(grid_tf, kBlock, device.unwrap())(ballot_transform_kernel<kUsePDL>, p);
    }
  }

  static void run(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView workspace,
      const uint32_t cap,
      const uint32_t target,
      const tvm::ffi::Optional<tvm::ffi::TensorView> stats) {
    const tvm::ffi::Optional<tvm::ffi::TensorView> no_rs, no_pt, no_cu;
    const auto dt = scores.dtype();
    if (dt.code == DLDataTypeCode::kDLFloat && dt.bits == 32 && dt.lanes == 1) {
      run_t<fp32_t>(scores, lengths, out, workspace, cap, target, no_rs, stats, no_pt, no_cu, false);
    } else if (dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16 && dt.lanes == 1) {
      run_t<bf16_t>(scores, lengths, out, workspace, cap, target, no_rs, stats, no_pt, no_cu, false);
    } else {
      host::RuntimeCheck(false, "scores must be fp32 or bf16");
    }
  }

  static void run_prefill(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView workspace,
      const uint32_t cap,
      const uint32_t target,
      const tvm::ffi::Optional<tvm::ffi::TensorView> row_starts,
      const tvm::ffi::Optional<tvm::ffi::TensorView> stats,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView cu_seqlens) {
    const tvm::ffi::Optional<tvm::ffi::TensorView> no_pt;
    const auto dt = scores.dtype();
    host::RuntimeCheck(dt.code == DLDataTypeCode::kDLFloat && dt.bits == 32 && dt.lanes == 1,
                       "prefill scores must be fp32");
    run_t<fp32_t>(scores, lengths, out, workspace, cap, target, row_starts, stats,
                  page_table, cu_seqlens, true);
  }
};

} // namespace sglang
