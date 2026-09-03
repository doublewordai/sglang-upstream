/**
 * \file topk_ballot.cuh
 * \brief One-read exact top-k (k <= 2048) for DSA indexers — v2 (ka-topk-select).
 *
 * Successor to the ka-topk-issue design (v1, 10-kernel chain, 1.2-1.7x the fg
 * chain at decode). The chrome traces said the bottleneck was the SELECT phase
 * (plan-V 14.6 us + plan2 1.9 + select0 7.5 + refine 6.4 at b=16/L=1M) and a
 * capture at 2.0-2.1 TB/s vs the 1-pass kernel's 3.3. v2 restructures:
 *
 *   S1 sample    (B, 512thr): windowed two-level 16-bit threshold (unchanged
 *                 from v1: coarse 256-bin + sub-bin; kills the degenerate
 *                 fp16-histogram class that makes the 1-pass fall back).
 *                 Zeroes the capture counter + coarse histogram.
 *   S2 capture   (chunks, B; 1024thr): THE read. Double-buffered float4
 *                 stream (the 1-pass kernel's load structure); per element a
 *                 single compare; a match (~0.6% of elements) does a global
 *                 atomicAdd append of (sortable_u32 key, pos) plus one hist
 *                 atomicAdd (key>>24) — NO per-element smem atomics, NO warp
 *                 ballots (v1's VOTE+popc sat between loads and cost ~40% of
 *                 the streaming rate). The exact match count survives past
 *                 the cap (count counts, stores truncate).
 *   S3 select    (B; 1024thr): ONE fused kernel replacing v1's V/F1/F2/F3/
 *                 P2/S3/S4 chain. Fast path: 4-round radix select over the
 *                 captured list (round 0 consumes the capture-built coarse
 *                 histogram; boundary entries shrink through a ping-pong
 *                 residual; fp32-exact). Miss rows (n < topk or n > cap,
 *                 ~never on real logits) run the fg-class 2-pass fallback
 *                 INSIDE this kernel (2 extra reads of the row).
 *
 * Chain = 3 kernels, PDL-linked (real programmatic launches, unlike v1 whose
 * launches never set the PDL attribute). Semantics = fg's decode contract:
 * output [B, topk] raw positions, arbitrary order, -1 padding for length <=
 * topk rows; exactness class = fg's (only the fallback path can cap-truncate,
 * and only when one fp16-coarse bin holds > cap candidates; the fast path is
 * always exact whenever taken).
 *
 * Self-cleaning across calls / CUDA-graph replays: S1 zeroes counters[0] and
 * hist; every other consumed word is rewritten by its producer within the
 * same call (plan fields live in smem only).
 */
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include "topk_keys.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace {

constexpr uint32_t kSampleBlock = 1024;  // S1
constexpr uint32_t kSampleWarps = kSampleBlock / 32;
constexpr uint32_t kCapBlock = 1024;     // S2
constexpr uint32_t kSelBlock = 1024;     // S3

struct BallotTopKParams {
  const void* __restrict__ scores;        // [B, stride] T
  const int32_t* __restrict__ lengths;    // [B]
  const int32_t* __restrict__ row_starts; // [B] or nullptr
  int32_t* __restrict__ out;              // [B, topk]
  float* __restrict__ v_s;                // [B] capture threshold (S1 out)
  int32_t* __restrict__ hist;             // [B, 256] coarse (key32>>24)
  int32_t* __restrict__ counters;         // [B, 4] {n, -, -, -}
  int32_t* __restrict__ stats;            // optional [B, 4] {n, n_stored, r, flags}
  uint2* __restrict__ cand;               // [B, cap] (key32, pos)
  uint2* __restrict__ resA;               // [B, cap] residual ping
  uint2* __restrict__ resB;               // [B, cap] residual pong
  int64_t stride;
  uint32_t topk;
  uint32_t cap;
  uint32_t target;      // capture-count target the sample aims at
  uint32_t cap_chunk;   // elements per capture CTA
};

// --- key conversions, to_float/load4, suffix_scan_257: topk_keys.cuh ---

// ---------------------------------------------------------------------------
// S1: sample -> two-level (16-bit key) threshold, conservative value
// ---------------------------------------------------------------------------

template <bool kUsePDL, typename T>
__global__ __launch_bounds__(kSampleBlock) void ballot_sample_kernel(const BallotTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.x;
  if (threadIdx.x == 0) { // counter + coarse hist for S2 (self-cleaning)
    p.counters[row * 4 + 0] = 0;
    p.counters[row * 4 + 1] = 0;
  }
  for (uint32_t i = threadIdx.x; i < kRadix; i += kSampleBlock)
    p.hist[static_cast<int64_t>(row) * kRadix + i] = 0;
  const int32_t length = p.lengths[row];
  // capture-all iff the whole row fits the sample bound (and the cap): exact,
  // never misses; no sample needed. (v1 captured-all up to `cap`; that costs
  // up to `cap` appends — pointless when a sampled threshold captures 3x topk.)
  if (length <= static_cast<int32_t>(p.topk) ||
      static_cast<uint32_t>(length) <= min(p.cap, kSampleMax)) {
    if (threadIdx.x == 0) p.v_s[row] = __int_as_float(0xff800000); // -inf
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  // windowed sample (W windows of E consecutive elements): alias-resistant
  const uint32_t n_s = min(kSampleMax, max(kSampleMin, static_cast<uint32_t>(length) / 8)) & ~63u;
  const uint32_t W = 64;
  const uint32_t E = n_s / W;
  __shared__ uint16_t s_keys[kSampleMax];
  __shared__ int s_hist[kSampleWarps][kRadix];  // warp-privatized
  __shared__ int s_scan[kRadix + 1];
  __shared__ int s_sub[kRadix];
  __shared__ int s_scratch[273];
  __shared__ int s_tc, s_tsub, s_above;
  for (uint32_t i = threadIdx.x; i < kSampleWarps * kRadix; i += kSampleBlock)
    (&s_hist[0][0])[i] = 0;
  for (uint32_t i = threadIdx.x; i < kRadix; i += kSampleBlock)
    s_sub[i] = 0;
  if (threadIdx.x == 0) { s_tc = 0; s_tsub = 0; s_above = 0; }
  __syncthreads();

  const T* rowp = reinterpret_cast<const T*>(p.scores) + static_cast<int64_t>(row) * p.stride +
                  (p.row_starts != nullptr ? p.row_starts[row] : 0);
  const uint32_t warp = threadIdx.x / 32;
  for (uint32_t t = threadIdx.x; t < n_s; t += kSampleBlock) {
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
  if (threadIdx.x < kRadix) {
    int acc = 0;
    for (uint32_t w = 0; w < kSampleWarps; ++w) acc += s_hist[w][threadIdx.x];
    s_scan[threadIdx.x] = acc;
  }
  __syncthreads();

  // sample crossing count: E[#captured] ~ target
  uint32_t j = (p.target * n_s + static_cast<uint32_t>(length) - 1) / static_cast<uint32_t>(length);
  j = max(j, 1u);
  j = min(j, n_s);

  suffix_scan_256_ip(s_scan, s_scratch); // s_scan[i] = #{coarse >= i}
  if (threadIdx.x < kRadix) {
    if (s_scan[threadIdx.x] >= static_cast<int>(j) &&
        s_scan[threadIdx.x + 1] < static_cast<int>(j))
      s_tc = static_cast<int>(threadIdx.x);
  }
  __syncthreads();
  const int tc = s_tc;
  if (tc <= 0) { // threshold below everything: capture all
    if (threadIdx.x == 0) p.v_s[row] = __int_as_float(0xff800000); // -inf
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  if (threadIdx.x == 0) s_above = s_scan[tc + 1];
  __syncthreads();
  const int above = s_above;

  // sub-bin pass over the sampled keys in the boundary coarse bin
  for (uint32_t t = threadIdx.x; t < n_s; t += kSampleBlock) {
    const uint16_t key = s_keys[t];
    if ((key >> 8) == static_cast<uint16_t>(tc)) atomicAdd(&s_sub[key & 0xFFu], 1);
  }
  __syncthreads();
  if (threadIdx.x < kRadix) s_scan[threadIdx.x] = s_sub[threadIdx.x];
  __syncthreads();
  suffix_scan_256_ip(s_scan, s_scratch); // s_scan[i] = #{sub >= i} within tc
  if (threadIdx.x < kRadix) {
    if (above + s_scan[threadIdx.x] >= static_cast<int>(j) &&
        above + s_scan[threadIdx.x + 1] < static_cast<int>(j))
      s_tsub = static_cast<int>(threadIdx.x);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const uint32_t K_s = (static_cast<uint32_t>(tc) << 8) | static_cast<uint32_t>(s_tsub);
    // conservative: capture {x >= value(K_s - 1)} >= {key >= K_s}
    p.v_s[row] = (K_s > 0) ? key16_to_value(static_cast<uint16_t>(K_s - 1))
                           : __int_as_float(0xff800000);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// Streaming helpers
// ---------------------------------------------------------------------------

/// Stream [lo, hi) of a row with vectorized loads (16B-aligned head/body/tail,
/// software-pipelined 2-deep). fn(int64_t pos, float x) per element; `pos` is
/// window-local. All threads must reach the call (uniform loop bounds).
template <typename T, typename F>
SGL_DEVICE void stream_chunk(const T* rowp, int64_t lo, int64_t hi, F&& fn) {
  const uint32_t tx = threadIdx.x;
  const uintptr_t addr = reinterpret_cast<uintptr_t>(rowp + lo);
  uint32_t head = static_cast<uint32_t>((16u - (addr & 15u)) & 15u) / sizeof(T);
  if (lo + head > hi) head = static_cast<uint32_t>(hi - lo);
  if (tx < head) fn(lo + tx, to_float(rowp[lo + tx]));
  const uint32_t n4 = (static_cast<uint32_t>(hi - lo) - head) >> 2;
  const T* body = rowp + lo + head;
  const uint32_t iters = (n4 + kCapBlock - 1) / kCapBlock;
  float pre[4] = {0.f, 0.f, 0.f, 0.f};
  if (tx < n4) load4<T>(body + tx * 4, pre);
#pragma unroll 1
  for (uint32_t k = 0; k < iters; ++k) {
    const uint32_t g = tx + k * kCapBlock;
    const bool have = g < n4;
    const uint32_t gn = g + kCapBlock;
    float x[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) x[e] = pre[e];
    if (gn < n4) load4<T>(body + gn * 4, pre);
    const int64_t base = lo + head + static_cast<int64_t>(g) * 4;
#pragma unroll
    for (int e = 0; e < 4; ++e)
      if (have) fn(base + e, x[e]);
  }
  const uint32_t tail_start = head + (n4 << 2);
  const uint32_t tail_n = static_cast<uint32_t>(hi - lo) - tail_start;
  if (tx < tail_n) fn(lo + tail_start + tx, to_float(rowp[lo + tail_start + tx]));
}

/// Stream [0, len) of a row for the select kernel (kSelBlock threads).
template <typename T, typename F>
SGL_DEVICE void stream_row(const T* rowp, int64_t len, F&& fn) {
  const uint32_t tx = threadIdx.x;
  const uintptr_t addr = reinterpret_cast<uintptr_t>(rowp);
  uint32_t head = static_cast<uint32_t>((16u - (addr & 15u)) & 15u) / sizeof(T);
  if (head > len) head = static_cast<uint32_t>(len);
  if (tx < head) fn(static_cast<int64_t>(tx), to_float(rowp[tx]));
  const uint32_t n4 = (static_cast<uint32_t>(len) - head) >> 2;
  const T* body = rowp + head;
  for (uint32_t g = tx; g < n4; g += kSelBlock) {
    float x[4];
    load4<T>(body + g * 4, x);
    const int64_t base = head + static_cast<int64_t>(g) * 4;
#pragma unroll
    for (int e = 0; e < 4; ++e) fn(base + e, x[e]);
  }
  const uint32_t tail_start = head + (n4 << 2);
  const uint32_t tail_n = static_cast<uint32_t>(len) - tail_start;
  if (tx < tail_n) fn(static_cast<int64_t>(tail_start + tx), to_float(rowp[tail_start + tx]));
}

// ---------------------------------------------------------------------------
// S2: the one read — compare + smem-staged append (one global atomic per CTA)
// ---------------------------------------------------------------------------

constexpr uint32_t kStage = 8192;  // per-CTA staging entries (64 KB smem)

template <bool kUsePDL, typename T>
__global__ __launch_bounds__(kCapBlock, 2) void ballot_capture_kernel(const BallotTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.y;
  const int32_t length = p.lengths[row];
  if (length <= static_cast<int32_t>(p.topk)) { // naive (also covers <= 0)
    if (blockIdx.x == 0) {
      int32_t* out_row = p.out + static_cast<int64_t>(row) * p.topk;
      for (uint32_t j = threadIdx.x; j < p.topk; j += kCapBlock)
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
  const int64_t hi = min(lo + static_cast<int64_t>(p.cap_chunk), static_cast<int64_t>(length));
  const T* rowp = reinterpret_cast<const T*>(p.scores) + static_cast<int64_t>(row) * p.stride +
                  (p.row_starts != nullptr ? p.row_starts[row] : 0);
  const float v_s = p.v_s[row];
  uint2* cand = p.cand + static_cast<int64_t>(row) * p.cap;
  unsigned* pcount = reinterpret_cast<unsigned*>(&p.counters[row * 4 + 0]);
  unsigned* pstored = reinterpret_cast<unsigned*>(&p.counters[row * 4 + 1]);
  int* phist = p.hist + static_cast<int64_t>(row) * kRadix;

  // per-CTA staging: matches append into smem (rare; a per-CTA counter is
  // contention-free at ~100 hits/chunk), flushed ONCE with a single global
  // atomicAdd + bulk copy. A per-match GLOBAL atomic measured ~150 ns
  // effective on one address (bench_stream.py) — 870 us for 256 MB at a 2.3%
  // match rate; this structure removes it from the per-element path.
  __shared__ uint2 s_buf[kStage];
  __shared__ int s_cnt;
  __shared__ int s_hist[kRadix];
  __shared__ int s_base;
  if (threadIdx.x == 0) s_cnt = 0;
  for (uint32_t i = threadIdx.x; i < kRadix; i += kCapBlock) s_hist[i] = 0;
  __syncthreads();

  stream_chunk<T>(rowp, lo, hi, [&](int64_t pos, float x) {
    if (!(x < v_s)) { // NaN-inclusive, like fg
      // cap the staging: once past kStage this CTA's overflow already dooms
      // the row to the fallback — don't hammer the counter
      if (s_cnt < static_cast<int>(kStage)) {
        const uint32_t key = sortable_u32(x);
        const int slot = atomicAdd(&s_cnt, 1);
        if (slot < static_cast<int>(kStage)) {
          s_buf[slot] = make_uint2(key, static_cast<uint32_t>(pos));
          atomicAdd(&s_hist[key >> 24], 1);
        }
      }
    }
  });
  __syncthreads();
  const int nst = s_cnt;
  if (nst > 0) {
    if (threadIdx.x == 0) s_base = static_cast<int>(atomicAdd(pcount, static_cast<unsigned>(nst)));
    __syncthreads();
    const int base = s_base;
    const int ncopy = min(nst, static_cast<int>(kStage));
    for (int i = threadIdx.x; i < ncopy; i += kCapBlock)
      if (base + i < static_cast<int>(p.cap)) cand[base + i] = s_buf[i];
    if (threadIdx.x == 0)
      atomicAdd(pstored, static_cast<unsigned>(ncopy));
    for (uint32_t b = threadIdx.x; b < kRadix; b += kCapBlock) {
      const int v = s_hist[b];
      if (v != 0) atomicAdd(&phist[b], v);
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// S3: fused select — 4-round radix over the captured list (fast) or the
// fg-class 2-pass fallback (miss rows), all in ONE kernel.
// ---------------------------------------------------------------------------

template <bool kUsePDL, typename T>
__global__ __launch_bounds__(kSelBlock) void ballot_select_kernel(const BallotTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t row = blockIdx.x;
  const int32_t length = p.lengths[row];
  const uint32_t topk = p.topk;
  if (length <= static_cast<int32_t>(topk)) { // naive: S2 wrote the output
    if (threadIdx.x == 0 && p.stats != nullptr) {
      p.stats[row * 4 + 0] = length;
      p.stats[row * 4 + 1] = length;
      p.stats[row * 4 + 2] = 0;
      p.stats[row * 4 + 3] = 0;
    }
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const uint32_t n = static_cast<uint32_t>(p.counters[row * 4 + 0]); // exact count
  const uint32_t n_stored = static_cast<uint32_t>(p.counters[row * 4 + 1]);
  const bool fast = (n >= topk) && (n <= p.cap) && (n_stored >= n);
  int32_t* out_row = p.out + static_cast<int64_t>(row) * p.topk;
  const uint2* cand = p.cand + static_cast<int64_t>(row) * p.cap;
  const T* rowp = reinterpret_cast<const T*>(p.scores) + static_cast<int64_t>(row) * p.stride +
                  (p.row_starts != nullptr ? p.row_starts[row] : 0);

  __shared__ int s_scan[kRadix + 1];
  __shared__ int s_h[2][kRadix + 1];
  __shared__ int s_scratch[273];
  __shared__ int s_t, s_ngt, s_fill, s_res, s_nres, s_ctr, s_tie;
  if (threadIdx.x == 0) { s_t = -1; s_ngt = 0; s_fill = 0; s_res = 0; s_nres = 0; }
  __syncthreads();

  // ---- round-0 histogram: capture-built (fast) or full-row byte-0 of the
  // 32-bit key (fallback). NOTE: the fallback MUST use the same key space as
  // rounds 1-3 (sortable_u32): mixing a 16-bit-key split with 32-bit-key
  // refinement mis-sorts values that round up across a power of two in fp16
  // (e.g. 1.99992 -> fp16 2.0 -> coarse bin 0xC0 while its u32 key sorts
  // below the whole bin). The 1-pass/fg fallbacks are safe because they
  // select the residual by fp32 VALUE, not by a second key space.
  if (fast) {
    if (threadIdx.x < kRadix)
      s_h[0][threadIdx.x] = p.hist[static_cast<int64_t>(row) * kRadix + threadIdx.x];
  } else {
    for (uint32_t i = threadIdx.x; i < kRadix; i += kSelBlock) s_h[0][i] = 0;
    __syncthreads();
    stream_row<T>(rowp, static_cast<int64_t>(length), [&](int64_t pos, float x) {
      (void)pos;
      atomicAdd(&s_h[0][(sortable_u32(x) >> 24)], 1);
    });
  }
  __syncthreads();

  // ---- round-0 boundary bin (crossing at topk) ----
  suffix_scan_256_ip(s_h[0], s_scratch); // s_h[0][i] = #{bin >= i}
  if (threadIdx.x < kRadix) {
    // '>=' left / '<=' right: the exactly-covered case must select a bin. Any
    // plateau winner is CONSISTENT because n_gt is derived AFTER the barrier
    // from the resolved bin (writing the pair from the winner races: two
    // winners can tear t and ngt).
    if (s_h[0][threadIdx.x] >= static_cast<int>(topk) &&
        s_h[0][threadIdx.x + 1] <= static_cast<int>(topk)) {
      s_t = static_cast<int>(threadIdx.x);
    }
  }
  __syncthreads();
  const int t0 = s_t;      // >= 0: hist sums to >= topk on both paths
  const int ngt0 = s_h[0][t0 + 1];

  // ---- round-0 scatter: emit above-boundary to out; boundary -> residual
  //      (+ round-1 histogram of the residual, byte 1 of the 32-bit key) ----
  for (uint32_t i = threadIdx.x; i < kRadix; i += kSelBlock) s_h[1][i] = 0;
  __syncthreads();
  if (fast) {
    for (uint32_t i = threadIdx.x; i < n; i += kSelBlock) {
      const uint2 e = cand[i];
      const int b = static_cast<int>(e.x >> 24);
      if (b > t0) {
        const int s = atomicAdd(&s_fill, 1);
        out_row[s] = static_cast<int32_t>(e.y);
      } else if (b == t0) {
        const int s = atomicAdd(&s_res, 1);
        if (s < static_cast<int>(p.cap)) {
          p.resA[static_cast<int64_t>(row) * p.cap + s] = e;
          atomicAdd(&s_h[1][(e.x >> 16) & 0xFFu], 1);
        }
      }
    }
  } else {
    stream_row<T>(rowp, static_cast<int64_t>(length), [&](int64_t pos, float x) {
      const uint32_t key = sortable_u32(x);
      const int c = static_cast<int>(key >> 24);
      if (c > t0) {
        const int s = atomicAdd(&s_fill, 1);
        out_row[s] = static_cast<int32_t>(pos);
      } else if (c == t0) {
        const int s = atomicAdd(&s_res, 1);
        if (s < static_cast<int>(p.cap)) {
          p.resA[static_cast<int64_t>(row) * p.cap + s] = make_uint2(key, static_cast<uint32_t>(pos));
          atomicAdd(&s_h[1][(key >> 16) & 0xFFu], 1);
        }
      }
    });
  }
  __syncthreads();
  int fill = s_fill; // == ngt0 <= topk
  int need = static_cast<int>(topk) - fill;
  int m = min(s_res, static_cast<int>(p.cap));

  // ---- rounds 1..3: fp32-exact radix over the shrinking residual ----
  const uint2* src = p.resA + static_cast<int64_t>(row) * p.cap;
  uint2* dst = p.resB + static_cast<int64_t>(row) * p.cap;
  int parity = 1; // s_h[1] holds the round-1 histogram
  for (int round = 1; round <= 3 && need > 0; ++round) {
    const int shift = 24 - 8 * round;
    suffix_scan_256_ip(s_h[parity], s_scratch); // s_h[parity][i] = #{bin >= i}
    if (threadIdx.x == 0) { s_t = -1; s_ngt = 0; s_ctr = 0; s_tie = 0; }
    __syncthreads();
    if (threadIdx.x < kRadix) {
      if (s_h[parity][threadIdx.x] >= need && s_h[parity][threadIdx.x + 1] <= need) {
        s_t = static_cast<int>(threadIdx.x);
      }
    }
    __syncthreads();
    const int t = s_t;    // >= 0: residual >= need (invariant from round r-1)
    const int ngt = s_h[parity][t + 1];
    if (round < 3) {
      for (uint32_t i = threadIdx.x; i < kRadix; i += kSelBlock) s_h[1 - parity][i] = 0;
      if (threadIdx.x == 0) s_nres = 0;
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < static_cast<uint32_t>(m); i += kSelBlock) {
      const uint2 e = src[i];
      const int b = static_cast<int>((e.x >> shift) & 0xFFu);
      if (b > t) {
        const int s = atomicAdd(&s_ctr, 1);
        out_row[fill + s] = static_cast<int32_t>(e.y);
      } else if (b == t) {
        if (round < 3) {
          const int s = atomicAdd(&s_nres, 1);
          dst[s] = e;
          atomicAdd(&s_h[1 - parity][(e.x >> (shift - 8)) & 0xFFu], 1);
        } else {
          // final round: byte-equal keys are the exact fp32 tie class; any
          // `need - ngt` of them complete the output (fg's tie rule)
          const int s = atomicAdd(&s_tie, 1);
          if (s < need - ngt) out_row[fill + ngt + s] = static_cast<int32_t>(e.y);
        }
      }
    }
    __syncthreads();
    fill += ngt;
    need -= ngt;
    if (round == 3 || need <= 0) break;
    m = s_nres;
    src = dst;
    dst = (dst == p.resB + static_cast<int64_t>(row) * p.cap)
              ? p.resA + static_cast<int64_t>(row) * p.cap
              : p.resB + static_cast<int64_t>(row) * p.cap;
    parity = 1 - parity;
  }

  if (threadIdx.x == 0 && p.stats != nullptr) {
    p.stats[row * 4 + 0] = static_cast<int32_t>(n);
    p.stats[row * 4 + 1] = static_cast<int32_t>(min(n, p.cap));
    p.stats[row * 4 + 2] = static_cast<int32_t>(ngt0);
    p.stats[row * 4 + 3] = fast ? 0
        : (1 | (n > p.cap ? 2 : (n < topk ? 4 : 0)));
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

    const auto batch = static_cast<uint32_t>(B.unwrap());
    const auto stride = S.unwrap();
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");
    RuntimeCheck(batch > 0 && batch <= kMaxRows, "batch too large for grid.y");
    RuntimeCheck(cap > topk, "cap must exceed topk");

    // [v_s (1) | counters (4) | hist (256)] + 3 x cap uint2 lists, per row.
    // The uint2 arrays must be 8-byte aligned: pad the int32 header to even.
    int64_t o = 0;
    auto* ws = static_cast<int32_t*>(workspace.data_ptr());
    float* v_s = reinterpret_cast<float*>(ws + o);        o += batch;
    int32_t* counters = ws + o;                           o += 4 * batch;
    int32_t* hist = ws + o;                               o += static_cast<int64_t>(batch) * kRadix;
    o = (o + 1) & ~static_cast<int64_t>(1);
    uint2* cand = reinterpret_cast<uint2*>(ws + o);       o += 2 * static_cast<int64_t>(batch) * cap;
    uint2* resA = reinterpret_cast<uint2*>(ws + o);       o += 2 * static_cast<int64_t>(batch) * cap;
    uint2* resB = reinterpret_cast<uint2*>(ws + o);       o += 2 * static_cast<int64_t>(batch) * cap;
    RuntimeCheck(static_cast<int64_t>(W.unwrap()) >= o, "workspace too small");

    BallotTopKParams p{
        .scores = scores.data_ptr(),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .row_starts = rs_ptr,
        .out = static_cast<int32_t*>(out.data_ptr()),
        .v_s = v_s,
        .hist = hist,
        .counters = counters,
        .stats = stats_ptr,
        .cand = cand,
        .resA = resA,
        .resB = resB,
        .stride = stride,
        .topk = topk,
        .cap = cap,
        .target = target,
        .cap_chunk = 4096u,
    };

    // ~264 CTAs total (2 per SM x 132 SMs): one wave, long streams per CTA.
    // Small batches get small chunks (spread one row across the SMs); large
    // batches get whole-row-ish chunks (fewer waves, longer pipelines).
    const int64_t want = static_cast<int64_t>(stride) * static_cast<int64_t>(batch) / 264 + 1;
    uint32_t chunk = 4096;
    while (chunk < want && chunk < (1u << 20)) chunk <<= 1;
    p.cap_chunk = chunk;

    const uint32_t chunks = static_cast<uint32_t>((stride + p.cap_chunk - 1) / p.cap_chunk);
    const dim3 grid_capture(chunks, batch);

    LaunchKernel(batch, kSampleBlock, device.unwrap()).enable_pdl(kUsePDL)(
        ballot_sample_kernel<kUsePDL, T>, p);
    LaunchKernel(grid_capture, kCapBlock, device.unwrap()).enable_pdl(kUsePDL)(
        ballot_capture_kernel<kUsePDL, T>, p);
    LaunchKernel(batch, kSelBlock, device.unwrap()).enable_pdl(kUsePDL)(
        ballot_select_kernel<kUsePDL, T>, p);
  }

  static void run(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView workspace,
      const uint32_t cap,
      const uint32_t target,
      const tvm::ffi::Optional<tvm::ffi::TensorView> stats) {
    const tvm::ffi::Optional<tvm::ffi::TensorView> no_rs;
    const auto dt = scores.dtype();
    if (dt.code == DLDataTypeCode::kDLFloat && dt.bits == 32 && dt.lanes == 1) {
      run_t<fp32_t>(scores, lengths, out, workspace, cap, target, no_rs, stats);
    } else if (dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16 && dt.lanes == 1) {
      run_t<bf16_t>(scores, lengths, out, workspace, cap, target, no_rs, stats);
    } else {
      host::RuntimeCheck(false, "scores must be fp32 or bf16");
    }
  }
};

} // namespace sglang
