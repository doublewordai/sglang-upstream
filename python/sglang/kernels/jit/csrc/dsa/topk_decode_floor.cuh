/**
 * \file topk_decode_floor.cuh
 * \brief Byte-floor decode-time top-k (k <= 2048) for DSA indexers: ONE
 * persistent launch, each logits row read (almost) once, optional fused
 * page-table transform.
 *
 * Replaces the 6-launch PDL chain of topk_decode_fg.cuh (two full reads per
 * row) with a single persistent kernel using two grid-wide spin barriers:
 *
 *   P0 plan   owner CTA (CTA r < B) samples its row (16 windows x 128 elems
 *             = 2048 elems, independent loads) into a smem coarse histogram
 *             and picks the capture threshold t_s = largest coarse bin whose
 *             scaled sample suffix >= min(6144, L/2). Naive rows (length <=
 *             topk) are deferred to P2.
 *   ---- grid barrier 1 ----
 *   P1 stream  THE read: block-cyclic (row, 4096-chunk) over the full grid;
 *             per element only convert + compare + ballot, and for the ~1%
 *             of elements with coarse bin >= t_s a warp-aggregated append
 *             (one smem atomicAdd per warp group) of (f16-key << 32 | pos)
 *             into the per-row global candidate list (smem-staged, one
 *             global atomicAdd per chunk flush). NO per-element histogram:
 *             the exact captured count IS count(bin >= t_s). Overflows past
 *             the cap set a flag.
 *   ---- grid barrier 2 ----
 *   P2 select owner CTA per row:
 *             fast path (no overflow and n_captured >= topk): the captured
 *             list provably contains the true top-k (count(bin >= t_s) >=
 *             topk means the top-k all have bin >= t_s = the capture
 *             threshold -- checked against the EXACT counter, never the
 *             sample, so exactness never depends on sample quality).
 *             overflow: select on the capped list (fg's documented
 *             inexactness class, cap = min(65536, stride)).
 *             under-capture (n_captured < topk, only when the sample
 *             overestimated): the owner alone re-streams its row building
 *             the full coarse histogram and capturing bin >= t_s - 1;
 *             if that still overflows or falls short, a second re-stream
 *             captures bin >= t_sel exactly (rare; slow; always exact up
 *             to the cap).
 *             Selection = fg's K4a+K4b fused: coarse hist of the list ->
 *             t_sel (largest bin with suffix >= topk; t_sel >= t_s) ->
 *             bin > t_sel -> output; bin == t_sel -> 256-bin sub-histogram
 *             (low byte of the f16 key) -> sub > t2 -> output; sub == t2 ->
 *             residual -> fg's exact 4-round fp32 radix refinement.
 *             Optional fused transform: out[j] = page_table[row][pos] for
 *             pos >= 0 else -1 (transform_index_page_table_decode's
 *             arithmetic, integer-exact).
 *
 * Semantics are topk_decode_fg's (= sgl_kernel topk_kernel's):
 *   - scores [B, stride] fp32/bf16, unit inner stride, only [0, length) read;
 *     -inf sorts below everything, +inf above, NaN via the fp16 key like fg;
 *   - output [B, topk] int32 raw positions in [0, length), arbitrary order;
 *     length <= topk rows emit i for i < length else -1 (negative lengths
 *     emit all -1); with page_table set those positions go through the table;
 *   - tie rule (fg's documented rule): boundary ties -- equal fp32 values at
 *     the k-th boundary -- are resolved arbitrarily (atomic arrival order);
 *   - inexactness class (fg's): if the threshold bin holds more candidates
 *     than the per-row cap, an arbitrary capped subset is refined. Unlike
 *     fg, overflow never leaves stale output slots: unwritten slots are -1.
 *
 * Grid: G = clamp(ceil(B*stride/4096), B, 132*4) CTAs of 256 threads,
 * co-resident by construction (__launch_bounds__(256, 4), <= 34 KB smem), so
 * the spin barriers cannot deadlock on an exclusive GPU. CUDA-graph safe:
 * no host syncs, fixed launch shape from (B, stride, cap); every consumed
 * workspace word is written before use within the same replay and re-zeroed
 * by its consumer (self-cleaning); the barrier words are reset by the
 * barrier closers (counts) and by the LAST exiting CTA (releases + exit
 * counter) before it fires its PDL trigger, so a PDL-overlapped successor
 * always observes a clean workspace.
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
constexpr uint32_t kBlock = 256;
constexpr uint32_t kChunk = 4096;  // elements per P1 unit (16/thread)
constexpr uint32_t kWindow = 128;  // sample window (elems)
constexpr uint32_t kWindows = 16;  // windows per row -> 2048 sample elems
constexpr uint32_t kStage = 4096;  // u64 candidate staging entries (>= kChunk)
constexpr uint32_t kRStage = 2048; // i32 residual staging entries
constexpr uint32_t kMaxTopK = 2048;
constexpr uint32_t kCTAsPerSM = 4; // co-residency for the spin barriers
constexpr uint32_t kMaxGrid = 132 * kCTAsPerSM;
constexpr int64_t kSampleTarget = 6144; // capture-size target (~3x topk)

// smem layout (int32 words):
//   [0, 256)                  s_hist   (P0b sample hist, P2 list/fallback hist)
//   [256, 256 + 2*kStage)     s_cand   (P1 u64 staging)
//   P2 aliases the s_cand region:
//     [256, 256+2048)         s_outA
//     [2304, 2304+2048)       s_outB
//     [4352, 4352+256)        s_sub
//     [4608, 4608+514)        s_scan (2 x 257)
//     [5122, 5122+2048)       s_rstage
//     [7170, 7170+514)        s_hist2 (2 x 257, K4b)
constexpr uint32_t kSmemInts = 2 * kStage + kRadix + 32;

struct FloorTopKParams {
  const void* __restrict__ scores;        // [B, stride] T
  const int32_t* __restrict__ lengths;    // [B]
  int32_t* __restrict__ out;              // [B, topk]
  const int32_t* __restrict__ page_table; // [B, pt_stride] or null (raw out)
  int32_t* __restrict__ plan;             // [B, 8] {t_s}
  int32_t* __restrict__ counters;         // [B, 8] {n_captured, flags}
  unsigned long long* __restrict__ cand;  // [B, cap] (f16key << 32) | pos
  int32_t* __restrict__ resid;            // [B, cap]
  int32_t* __restrict__ resid2;           // [B, cap]
  int32_t* __restrict__ barrier;          // [8] {cnt1, rel1, cnt2, rel2, exit}
  int32_t* __restrict__ stats;            // optional [B, 4] {n_eq, n_eq_stored, r, flags}
  int64_t stride;
  int64_t pt_stride;
  uint32_t batch;
  uint32_t topk;
  uint32_t cap;
  uint32_t aligned;
};

// --- key conversions: bit-identical to topk_decode_fg / production ---

SGL_DEVICE uint16_t sortable_f16(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
}

SGL_DEVICE uint32_t sortable_u32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

SGL_DEVICE unsigned long long pack_key_pos(uint16_t key, int64_t pos) {
  return (static_cast<unsigned long long>(key) << 32) | static_cast<unsigned long long>(pos);
}

/// Conservative capture threshold for coarse bin t: the float value of the
/// f16-sortable key ONE BELOW the bin's smallest key. `x >= this` is a
/// SUPERSET of {key(x) >= t<<8} (floats just below the bin's lowest key
/// still round UP into it under RNE, so the bin's own value would miss
/// them); the exact key is recomputed per captured element anyway.
SGL_DEVICE float capture_threshold_value(int t) {
  if (t <= 0) return __int_as_float(0xff800000u); // -inf: capture everything
  const uint16_t key_lo = static_cast<uint16_t>((t << 8) - 1);
  const uint16_t bits =
      (key_lo & 0x8000u) ? static_cast<uint16_t>(key_lo & 0x7FFFu)
                         : static_cast<uint16_t>(~key_lo);
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

/// Suffix sums over 257 smem ints (256 bins + [256] = 0 sentinel),
/// Hillis-Steele. Identical to topk_decode_fg's.
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

/// Grid-wide barrier over all gridDim.x CTAs (co-resident by construction).
/// count is reset by the last arriver; release is reset by the last CTA to
/// exit the kernel (see the exit protocol below), so the pair is clean for
/// the next replay. Volatile-spin + fences (contest-kernel-proven pattern).
SGL_DEVICE void grid_sync(int* count, volatile int* release) {
  const int expected = static_cast<int>(gridDim.x);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (atomicAdd(count, 1) == expected - 1) {
      *count = 0;         // everyone has arrived; safe to reset
      __threadfence();    // order this CTA's (and others') writes before release
      *release = 1;
    } else {
      while (*release == 0) {
        __nanosleep(64);
      }
    }
  }
  __syncthreads();
}

template <bool kUsePDL, typename T>
__global__ __launch_bounds__(kBlock, kCTAsPerSM) void floor_topk_kernel(const FloorTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>(); // P0 reads the predecessor's logits
  const uint32_t g = blockIdx.x;
  const uint32_t G = gridDim.x;
  const uint32_t b = p.batch;
  const int tid = static_cast<int>(threadIdx.x);

  __shared__ __align__(16) int s_raw[kSmemInts];
  int* s_hist = s_raw;                                             // [256]
  unsigned long long* s_cand =                                     // [kStage]
      reinterpret_cast<unsigned long long*>(s_raw + kRadix);
  int* s_outA = s_raw + kRadix;                                    // [2048]
  int* s_outB = s_raw + kRadix + kMaxTopK;                         // [2048]
  int* s_sub = s_raw + kRadix + 2 * kMaxTopK;                      // [256]
  int (*s_scan)[kRadix + 1] =
      reinterpret_cast<int (*)[kRadix + 1]>(s_raw + kRadix + 2 * kMaxTopK + kRadix);
  int* s_rstage = s_raw + kRadix + 2 * kMaxTopK + kRadix + 2 * (kRadix + 1); // [2048]
  int (*s_hist2)[kRadix + 1] =
      reinterpret_cast<int (*)[kRadix + 1]>(s_raw + kRadix + 2 * kMaxTopK + kRadix +
                                            2 * (kRadix + 1) + kRStage);
  __shared__ int s_stage_n, s_cur_row, s_flush_base;
  __shared__ int s_outA_n, s_outB_n, s_eq_n, s_rstage_n, s_resid_n;
  __shared__ int s_t, s_t2, s_n_eq, s_r, s_filled;
  __shared__ int s_last_remain, s_ctr, s_next_cnt;

  const T* rowp_of = reinterpret_cast<const T*>(p.scores);

  // ---------------- P0: owner samples its row and plans t_s ----------------
  if (g < b) {
    const uint32_t r = g;
    const int32_t L = p.lengths[r];
    if (L > static_cast<int32_t>(p.topk)) {
      const T* rowp = rowp_of + static_cast<int64_t>(r) * p.stride;
      for (uint32_t i = tid; i < kRadix; i += kBlock) s_hist[i] = 0;
      __syncthreads();
      const int64_t wstride = L / kWindows; // >= 128 because L > topk >= 2048
      // 2048 independent sample loads (8 per thread)
      for (int i = tid; i < static_cast<int>(kWindows) * static_cast<int>(kWindow);
           i += kBlock) {
        const int wi = i / kWindow;
        const int off = i - wi * kWindow;
        const int64_t pos = static_cast<int64_t>(wi) * wstride + off;
        if (pos < L) atomicAdd(&s_hist[sortable_f16(to_float(rowp[pos])) >> 8], 1);
      }
      __syncthreads();
      s_scan[0][kRadix] = 0;
      s_scan[1][kRadix] = 0;
      if (tid < kRadix) s_scan[0][tid] = s_hist[tid];
      __syncthreads();
      suffix_scan_257(s_scan[0], s_scan[1]);
      if (tid == 0) s_t = 0; // default: capture everything (tiny rows)
      __syncthreads();
      if (tid < kRadix) {
        // t_s = largest coarse bin whose scaled sample suffix >= target
        const int64_t sample_n = static_cast<int64_t>(kWindows) * kWindow;
        const int64_t target = min(kSampleTarget, static_cast<int64_t>(L) / 2);
        const int64_t cur = static_cast<int64_t>(s_scan[0][tid]) * L / sample_n;
        const int64_t next = static_cast<int64_t>(s_scan[0][tid + 1]) * L / sample_n;
        if (cur >= target && next < target) s_t = static_cast<int>(tid);
      }
      __syncthreads();
      if (tid == 0) {
        p.plan[static_cast<int64_t>(r) * 8 + 0] = s_t;
        // capture test value: {x >= v_s} is a superset of {bin(x) >= t_s}
        // (NaN handled in the stream by an isnan test)
        p.plan[static_cast<int64_t>(r) * 8 + 1] =
            __float_as_uint(capture_threshold_value(s_t));
      }
    }
  }

  grid_sync(p.barrier + 0, reinterpret_cast<volatile int*>(p.barrier + 1));

  // ---------------- P1: the single streaming read + capture ----------------
  {
    if (tid == 0) {
      s_stage_n = 0;
      s_cur_row = -1;
    }
    __syncthreads();

    // flush the candidate staging into cand[row]; all threads (uniform call)
    auto flush_stage = [&]() {
      __syncthreads(); // settle in-flight appends before reading s_stage_n
      const int n = s_stage_n;
      const int row = s_cur_row;
      if (n > 0) {
        if (tid == 0) {
          s_flush_base = atomicAdd(&p.counters[static_cast<int64_t>(row) * 8 + 0], n);
        }
        __syncthreads();
        const int base = s_flush_base;
        const int room = static_cast<int>(p.cap) - base;
        const int n_store = min(n, max(room, 0));
        if (n_store < n) atomicOr(&p.counters[static_cast<int64_t>(row) * 8 + 1], 2);
        unsigned long long* dst = p.cand + static_cast<int64_t>(row) * p.cap + base;
        for (int i = tid; i < n_store; i += kBlock) dst[i] = s_cand[i];
        __syncthreads();
      }
      if (tid == 0) s_stage_n = 0;
      __syncthreads();
    };

    auto chunks_of_row = [&](uint32_t r) -> uint32_t {
      const int32_t L = p.lengths[r];
      if (L <= static_cast<int32_t>(p.topk)) return 0;
      return static_cast<uint32_t>((L + kChunk - 1) / kChunk);
    };
    uint32_t total = 0;
    for (uint32_t r = 0; r < b; ++r) total += chunks_of_row(r);

    for (uint32_t u = g; u < total; u += G) {
      // map flat chunk index -> (row, chunk); every CTA visits rows in
      // increasing order, so staging can be flushed on row switch
      uint32_t r = 0, ci = 0, acc = 0;
      for (uint32_t rr = 0; rr < b; ++rr) {
        const uint32_t n = chunks_of_row(rr);
        if (u < acc + n) { r = rr; ci = u - acc; break; }
        acc += n;
      }
      if (s_cur_row != static_cast<int>(r)) {
        flush_stage();
        if (tid == 0) s_cur_row = static_cast<int>(r);
        __syncthreads();
      }
      const int32_t L = p.lengths[r];
      const int64_t beg = static_cast<int64_t>(ci) * kChunk;
      const int64_t end = min(beg + kChunk, static_cast<int64_t>(L));
      const T* rowp = rowp_of + static_cast<int64_t>(r) * p.stride;
      const float v_s = __uint_as_float(
          static_cast<uint32_t>(p.plan[static_cast<int64_t>(r) * 8 + 1]));

      // capture test = one float compare (+ isnan); the exact f16 key is
      // computed only for captured elements. ONE smem atomicAdd per float4
      // group, reserving ALL its kept slots at once. Appends per chunk
      // <= kChunk <= kStage.
      auto process4 = [&](int64_t pos, const float (&x)[4]) {
        int cnt = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e)
          if (x[e] >= v_s || x[e] != x[e]) ++cnt;
        if (cnt != 0) {
          const int base = atomicAdd(&s_stage_n, cnt);
          int w = 0;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            if (x[e] >= v_s || x[e] != x[e]) {
              s_cand[base + w] = pack_key_pos(sortable_f16(x[e]), pos + e);
              ++w;
            }
          }
        }
      };
      auto process1 = [&](int64_t pos, float x) {
        if (x >= v_s || x != x) {
          const int slot = atomicAdd(&s_stage_n, 1);
          s_cand[slot] = pack_key_pos(sortable_f16(x), pos);
        }
      };

      if (p.aligned) {
        const int64_t n4 = (end - beg) >> 2;
        for (int64_t grp = tid; grp < n4; grp += kBlock) {
          float x[4];
          load4<T>(rowp + beg + grp * 4, x);
          process4(beg + grp * 4, x);
        }
        for (int64_t i = beg + (n4 << 2) + tid; i < end; i += kBlock)
          process1(i, to_float(rowp[i])); // <= 3 appends: fits kStage headroom
      } else {
        for (int64_t i = beg + tid; i < end; i += kBlock) process1(i, to_float(rowp[i]));
      }
      flush_stage(); // chunk boundary: appends this chunk <= kChunk <= kStage
    }
    flush_stage();
  }

  grid_sync(p.barrier + 2, reinterpret_cast<volatile int*>(p.barrier + 3));

  // ---------------- P2: verify + select (owner CTA per row) ----------------
  if (g < b) {
    const uint32_t r = g;
    const int32_t L = p.lengths[r];
    int32_t* out_row = p.out + static_cast<int64_t>(r) * p.topk;
    const T* rowp = rowp_of + static_cast<int64_t>(r) * p.stride;
    const int32_t* pt = p.page_table;
    const int64_t pt_row = static_cast<int64_t>(r) * p.pt_stride;

    // output write with optional fused page-table transform
    auto write_out = [&](int slot, int32_t pos) {
      out_row[slot] = (pt != nullptr && pos >= 0) ? pt[pt_row + pos] : pos;
    };

    if (L <= static_cast<int32_t>(p.topk)) {
      // naive path: deterministic, fg-identical
      for (uint32_t j = tid; j < p.topk; j += kBlock) {
        const int32_t pos = (static_cast<int32_t>(j) < L) ? static_cast<int32_t>(j) : -1;
        write_out(j, pos);
      }
      if (tid == 0) {
        p.counters[static_cast<int64_t>(r) * 8 + 0] = 0;
        p.counters[static_cast<int64_t>(r) * 8 + 1] = 0;
        if (p.stats != nullptr) {
          p.stats[static_cast<int64_t>(r) * 4 + 0] = L;
          p.stats[static_cast<int64_t>(r) * 4 + 1] = L;
          p.stats[static_cast<int64_t>(r) * 4 + 2] = 0;
          p.stats[static_cast<int64_t>(r) * 4 + 3] = 0;
        }
      }
    } else {
      int flags = 0;
      const int t_s = p.plan[static_cast<int64_t>(r) * 8 + 0];
      const int n_cap = p.counters[static_cast<int64_t>(r) * 8 + 0];
      const int ovfl = p.counters[static_cast<int64_t>(r) * 8 + 1] & 2;

      int n_list;
      bool have_full_hist = false;
      if (!ovfl && n_cap >= static_cast<int>(p.topk)) {
        // fast path: the P1 list provably contains the true top-k
        n_list = n_cap;
      } else {
        // under-capture (sample overestimated) or cap overflow: the owner
        // alone re-streams the row building the FULL coarse histogram,
        // capturing >= t_s - 1 (under-capture) or nothing (overflow; the
        // tight t_sel capture follows in tier-2)
        flags |= 1;
        const bool fb_overflow = ovfl != 0;
        const uint32_t t_f1 = fb_overflow ? 255u : static_cast<uint32_t>(max(t_s - 1, 0));
        const float v_f1 = fb_overflow
                               ? __int_as_float(0x7f800000) // +inf: capture nothing
                               : capture_threshold_value(static_cast<int>(t_f1));
        for (uint32_t i = tid; i < kRadix; i += kBlock) s_hist[i] = 0;
        if (tid == 0) s_stage_n = 0;
        __syncthreads();
        int64_t kept = 0;
        auto fb_process = [&](int64_t pos, float x) {
          const uint16_t key = sortable_f16(x);
          atomicAdd(&s_hist[key >> 8], 1);
          if (x >= v_f1 || x != x) {
            const int slot = atomicAdd(&s_stage_n, 1);
            if (slot < static_cast<int>(kStage)) s_cand[slot] = pack_key_pos(key, pos);
          }
        };
        // (chunked to flush the staging; full hist always accumulated)
        for (int64_t c0 = 0; c0 < L; c0 += kChunk) {
          const int64_t beg = c0;
          const int64_t end = min(c0 + kChunk, static_cast<int64_t>(L));
          if (p.aligned) {
            const int64_t n4 = (end - beg) >> 2;
            for (int64_t grp = tid; grp < n4; grp += kBlock) {
              float x[4];
              load4<T>(rowp + beg + grp * 4, x);
#pragma unroll
              for (int e = 0; e < 4; ++e) fb_process(beg + grp * 4 + e, x[e]);
            }
            for (int64_t i = beg + (n4 << 2) + tid; i < end; i += kBlock)
              fb_process(i, to_float(rowp[i]));
          } else {
            for (int64_t i = beg + tid; i < end; i += kBlock) fb_process(i, to_float(rowp[i]));
          }
          __syncthreads();
          // local flush into cand
          const int n = s_stage_n;
          const int n_store = min(n, static_cast<int>(p.cap) - static_cast<int>(kept));
          if (n_store < n) flags |= 2;
          unsigned long long* dst = p.cand + static_cast<int64_t>(r) * p.cap + kept;
          for (int i = tid; i < n_store; i += kBlock) dst[i] = s_cand[i];
          kept += n;
          if (tid == 0) s_stage_n = 0;
          __syncthreads();
        }
        // full hist -> t_sel
        s_scan[0][kRadix] = 0;
        s_scan[1][kRadix] = 0;
        for (uint32_t i = tid; i < kRadix; i += kBlock) s_scan[0][i] = s_hist[i];
        __syncthreads();
        suffix_scan_257(s_scan[0], s_scan[1]);
        if (tid == 0) s_t = -1;
        __syncthreads();
        if (tid < kRadix) {
          if (s_scan[0][tid] >= static_cast<int>(p.topk) &&
              s_scan[0][tid + 1] < static_cast<int>(p.topk))
            s_t = static_cast<int>(tid);
        }
        __syncthreads();
        have_full_hist = true;
        const int64_t count_f1 = (t_f1 == 0)
                                     ? static_cast<int64_t>(L)
                                     : static_cast<int64_t>(s_scan[0][t_f1]);
        if (fb_overflow || kept > static_cast<int64_t>(p.cap) || count_f1 < static_cast<int>(p.topk)) {
          // tier-2: capture >= t_sel exactly (slow path; pathological only)
          const uint32_t t_f2 = static_cast<uint32_t>(max(s_t, 0));
          const float v_f2 = capture_threshold_value(static_cast<int>(t_f2));
          if (tid == 0) s_stage_n = 0;
          __syncthreads();
          kept = 0;
          for (int64_t c0 = 0; c0 < L; c0 += kChunk) {
            const int64_t beg = c0;
            const int64_t end = min(c0 + kChunk, static_cast<int64_t>(L));
            for (int64_t i = beg + tid; i < end; i += kBlock) {
              const float x = to_float(rowp[i]);
              if (x >= v_f2 || x != x) {
                const uint16_t key = sortable_f16(x);
                const int slot = atomicAdd(&s_stage_n, 1);
                if (slot < static_cast<int>(kStage)) s_cand[slot] = pack_key_pos(key, i);
              }
            }
            __syncthreads();
            const int n = s_stage_n;
            const int n_store = min(n, static_cast<int>(p.cap) - static_cast<int>(kept));
            if (n_store < n) flags |= 2;
            unsigned long long* dst = p.cand + static_cast<int64_t>(r) * p.cap + kept;
            for (int i = tid; i < n_store; i += kBlock) dst[i] = s_cand[i];
            kept += n;
            if (tid == 0) s_stage_n = 0;
            __syncthreads();
          }
        }
        n_list = static_cast<int>(min(kept, static_cast<int64_t>(p.cap)));
      }

      // ---- coarse hist of the list -> t_sel (skip if full hist known) ----
      const unsigned long long* list = p.cand + static_cast<int64_t>(r) * p.cap;
      if (!have_full_hist) {
        for (uint32_t i = tid; i < kRadix; i += kBlock) s_hist[i] = 0;
        __syncthreads();
        for (int i = tid; i < n_list; i += kBlock) {
          const unsigned long long e = list[i];
          atomicAdd(&s_hist[static_cast<uint16_t>(e >> 32) >> 8], 1);
        }
        __syncthreads();
        s_scan[0][kRadix] = 0;
        s_scan[1][kRadix] = 0;
        for (uint32_t i = tid; i < kRadix; i += kBlock) s_scan[0][i] = s_hist[i];
        __syncthreads();
        suffix_scan_257(s_scan[0], s_scan[1]);
        if (tid == 0) s_t = -1;
        __syncthreads();
        if (tid < kRadix) {
          if (s_scan[0][tid] >= static_cast<int>(p.topk) &&
              s_scan[0][tid + 1] < static_cast<int>(p.topk))
            s_t = static_cast<int>(tid);
        }
        __syncthreads();
      }
      const int t_sel = s_t;
      const int r_need0 = (t_sel >= 0) ? static_cast<int>(p.topk) - s_scan[0][t_sel + 1] : 0;
      int r_need = r_need0;
      if (tid == 0) s_n_eq = (t_sel >= 0) ? s_scan[0][t_sel] - s_scan[0][t_sel + 1] : 0;
      if (t_sel < 0) {
        // impossible when n_list >= topk; capped-overflow corner only
        flags |= 4;
        for (uint32_t j = tid; j < p.topk; j += kBlock) write_out(j, -1);
        r_need = 0;
      }

      if (t_sel >= 0) {
        // ---- pass 1: bin > t_sel -> outA; bin == t_sel -> sub-hist ----
        if (tid == 0) { s_outA_n = 0; s_eq_n = 0; s_filled = 0; }
        for (uint32_t i = tid; i < kRadix; i += kBlock) s_sub[i] = 0;
        __syncthreads();
        for (int i = tid; i < n_list; i += kBlock) {
          const unsigned long long e = list[i];
          const uint16_t key = static_cast<uint16_t>(e >> 32);
          const int32_t pos = static_cast<int32_t>(e & 0xffffffffu);
          const int bin = key >> 8;
          if (bin > t_sel) {
            s_outA[atomicAdd(&s_outA_n, 1)] = pos;
          } else if (bin == t_sel) {
            atomicAdd(&s_sub[key & 0xFFu], 1);
            atomicAdd(&s_eq_n, 1);
          }
        }
        __syncthreads();
        const int nA = s_outA_n;

        // sub-bin threshold t2 (fg plan2 semantics); absent crossing
        // (eq exhausted below r_need, cap-overflow class) -> -1 = take all
        s_scan[0][kRadix] = 0;
        s_scan[1][kRadix] = 0;
        for (uint32_t i = tid; i < kRadix; i += kBlock) s_scan[0][i] = s_sub[i];
        __syncthreads();
        suffix_scan_257(s_scan[0], s_scan[1]);
        if (tid == 0) s_t2 = -1;
        __syncthreads();
        if (tid < kRadix) {
          if (s_scan[0][tid] > r_need && s_scan[0][tid + 1] <= r_need) s_t2 = static_cast<int>(tid);
        }
        __syncthreads();
        const int t2 = s_t2;

        // ---- pass 2: bin == t_sel: sub > t2 -> outB, sub == t2 -> resid ----
        if (tid == 0) { s_outB_n = 0; s_rstage_n = 0; s_resid_n = 0; }
        __syncthreads();
        for (int base = 0; base < n_list; base += kRStage) {
          const int stop = min(base + static_cast<int>(kRStage), n_list);
          for (int i = base + tid; i < stop; i += kBlock) {
            const unsigned long long e = list[i];
            const uint16_t key = static_cast<uint16_t>(e >> 32);
            if ((key >> 8) == t_sel) {
              const int32_t pos = static_cast<int32_t>(e & 0xffffffffu);
              const int sub = key & 0xFFu;
              if (sub > t2) {
                s_outB[atomicAdd(&s_outB_n, 1)] = pos;
              } else if (sub == t2) {
                const int slot = atomicAdd(&s_rstage_n, 1);
                if (slot < static_cast<int>(kRStage)) s_rstage[slot] = pos;
              }
            }
          }
          __syncthreads();
          {
            const int n = s_rstage_n;
            int32_t* dst = p.resid + static_cast<int64_t>(r) * p.cap + s_resid_n;
            for (int i = tid; i < n; i += kBlock) dst[i] = s_rstage[i];
            if (tid == 0) { s_resid_n += n; s_rstage_n = 0; }
          }
          __syncthreads();
        }
        __syncthreads();
        const int nB = s_outB_n;
        const int r2 = (t2 >= 0) ? r_need - s_scan[0][t2 + 1] : 0;

        // write A and B blocks (transform fused)
        for (int i = tid; i < nA; i += kBlock) write_out(i, s_outA[i]);
        for (int i = tid; i < nB; i += kBlock) write_out(nA + i, s_outB[i]);
        if (tid == 0) s_filled = nA + nB;

        // ---- K4b: exact fp32 radix refinement over the residual ----
        const int n2 = (t2 >= 0 && r2 > 0) ? s_resid_n : 0;
        if (r2 > 0 && n2 <= 0) flags |= 4;
        if (n2 > 0) {
          const int32_t* src = p.resid + static_cast<int64_t>(r) * p.cap;
          int32_t* dst = p.resid2 + static_cast<int64_t>(r) * p.cap;
          for (uint32_t i = tid; i <= kRadix; i += kBlock) s_hist2[0][i] = 0;
          if (tid == 0) s_hist2[1][kRadix] = 0;
          __syncthreads();
          for (int i = tid; i < n2; i += kBlock)
            atomicAdd(&s_hist2[0][(sortable_u32(to_float(rowp[src[i]])) >> 24) & 0xFFu], 1);
          if (tid == 0) s_ctr = nA + nB; // out slot base
          __syncthreads();
          int src_n = n2;
          int topk_rem = r2;
#pragma unroll 1
          for (int round = 0; round < 4; ++round) {
            const int shift = 24 - 8 * round;
            if (tid == 0) { s_t = -1; s_next_cnt = 0; }
            __syncthreads();
            suffix_scan_257(s_hist2[round & 1], s_hist2[1 - (round & 1)]);
            if (tid < kRadix) {
              const int* suf = s_hist2[round & 1];
              if (suf[threadIdx.x] > topk_rem && suf[threadIdx.x + 1] <= topk_rem) {
                s_t = static_cast<int>(threadIdx.x);
                s_last_remain = topk_rem - suf[threadIdx.x + 1];
              }
            }
            __syncthreads();
            const int t = s_t;
            if (t < 0) { // impossible; bail defensively
              flags |= 4;
              break;
            }
            topk_rem -= s_hist2[round & 1][t + 1];
            const bool last = (round == 3) || (topk_rem == 0);
            if (!last) {
              for (uint32_t i = tid; i <= kRadix; i += kBlock)
                s_hist2[1 - (round & 1)][i] = 0;
              __syncthreads();
            }
            const int base_out = nA + nB;
            for (int i = tid; i < src_n; i += kBlock) {
              const int32_t pos = src[i];
              const float x = to_float(rowp[pos]);
              const int32_t bin = static_cast<int32_t>((sortable_u32(x) >> shift) & 0xFFu);
              if (bin > t) {
                write_out(atomicAdd(&s_ctr, 1), pos);
              } else if (bin == t) {
                if (round == 3) { // final ties: arbitrary (fg's documented rule)
                  const int old = atomicAdd(&s_last_remain, -1);
                  if (old > 0) write_out(base_out + r2 - old, pos);
                } else if (topk_rem > 0) {
                  const int slot = atomicAdd(&s_next_cnt, 1);
                  dst[slot] = pos;
                  atomicAdd(&s_hist2[1 - (round & 1)][(sortable_u32(x) >> (shift - 8)) & 0xFFu], 1);
                }
              }
            }
            __syncthreads();
            if (last) {
              if (tid == 0) s_filled = max(s_filled, base_out + r2);
              break;
            }
            src_n = s_next_cnt;
            const int32_t* tmp = src;
            src = dst;
            dst = const_cast<int32_t*>(tmp);
          }
        }

        // cap-overflow class: guarantee no stale slots (fg leaves them stale)
        __syncthreads();
        const int filled = s_filled;
        for (int j = filled + tid; j < static_cast<int>(p.topk); j += kBlock)
          write_out(j, -1);
      }

      // self-clean: counters (hist is smem-only now)
      if (tid == 0) {
        p.counters[static_cast<int64_t>(r) * 8 + 0] = 0;
        p.counters[static_cast<int64_t>(r) * 8 + 1] = 0;
        if (p.stats != nullptr) {
          p.stats[static_cast<int64_t>(r) * 4 + 0] = s_n_eq;
          p.stats[static_cast<int64_t>(r) * 4 + 1] = s_eq_n;
          p.stats[static_cast<int64_t>(r) * 4 + 2] = r_need;
          p.stats[static_cast<int64_t>(r) * 4 + 3] = flags;
        }
      }
    }
  }

  // ---------------- exit protocol ----------------
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    const int expected = static_cast<int>(gridDim.x);
    if (atomicAdd(p.barrier + 4, 1) == expected - 1) {
      // last CTA out: reset the barrier release words + exit counter for the
      // next replay (nobody spins on them any more), before the PDL trigger.
      p.barrier[1] = 0;
      p.barrier[3] = 0;
      p.barrier[4] = 0;
      __threadfence();
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

} // namespace

template <bool kUsePDL>
struct TopKDecodeFloor {
  static constexpr uint32_t kMaxRows = kMaxGrid;

  template <typename T>
  static void run_t(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView out,
      const tvm::ffi::Optional<tvm::ffi::TensorView> page_table,
      const tvm::ffi::TensorView workspace,
      const uint32_t cap,
      const tvm::ffi::Optional<tvm::ffi::TensorView> stats) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"score_stride"};
    auto W = SymbolicSize{"workspace_ints"};
    auto K = SymbolicSize{"topk"};
    auto P = SymbolicSize{"page_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, -1}).with_strides({S, 1}).with_dtype<T>().with_device(device).verify(scores);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device).verify(out);
    TensorMatcher({W}).with_dtype<int32_t>().with_device(device).verify(workspace);

    const int32_t* pt_ptr = nullptr;
    int64_t pt_stride = 0;
    if (page_table.has_value()) {
      TensorMatcher({B, -1}).with_strides({P, 1}).with_dtype<int32_t>().with_device(device)
          .verify(page_table.value());
      pt_ptr = static_cast<const int32_t*>(page_table.value().data_ptr());
      pt_stride = P.unwrap();
    }

    int32_t* stats_ptr = nullptr;
    if (stats.has_value()) {
      TensorMatcher({B, 4}).with_dtype<int32_t>().with_device(device).verify(stats.value());
      stats_ptr = static_cast<int32_t*>(stats.value().data_ptr());
    }

    const auto batch = static_cast<uint32_t>(B.unwrap());
    const auto stride = static_cast<int64_t>(S.unwrap());
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");
    RuntimeCheck(batch > 0 && batch <= kMaxRows, "batch too large for the persistent grid");
    RuntimeCheck(cap > topk, "cap must exceed topk");

    // [barrier 8 | plan B*8 | counters B*8 | (pad) | cand B*cap u64
    //  | resid B*cap | resid2 B*cap]
    const int64_t n_head = 8;
    const int64_t n_small = n_head + static_cast<int64_t>(batch) * (8 + 8);
    const int64_t cand_off = n_small + (n_small & 1); // 8B-align the u64 list
    const int64_t need = cand_off + 4 * static_cast<int64_t>(batch) * cap;
    RuntimeCheck(static_cast<int64_t>(W.unwrap()) >= need, "workspace too small");

    auto* ws = static_cast<int32_t*>(workspace.data_ptr());
    const auto* scores_ptr = static_cast<const T*>(scores.data_ptr());
    const bool aligned =
        (reinterpret_cast<uintptr_t>(scores_ptr) % 16 == 0) && (stride % 4 == 0);

    FloorTopKParams p{
        .scores = scores_ptr,
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .out = static_cast<int32_t*>(out.data_ptr()),
        .page_table = pt_ptr,
        .plan = ws + 8,
        .counters = ws + 8 + static_cast<int64_t>(batch) * 8,
        .cand = reinterpret_cast<unsigned long long*>(ws + cand_off),
        .resid = ws + cand_off + 2 * static_cast<int64_t>(batch) * cap,
        .resid2 = ws + cand_off + 3 * static_cast<int64_t>(batch) * cap,
        .barrier = ws,
        .stats = stats_ptr,
        .stride = stride,
        .pt_stride = pt_stride,
        .batch = batch,
        .topk = topk,
        .cap = cap,
        .aligned = aligned ? 1u : 0u,
    };

    // persistent grid: enough CTAs to spread the stream, all co-resident
    const int64_t chunks = (static_cast<int64_t>(batch) * stride + kChunk - 1) / kChunk;
    uint32_t grid = static_cast<uint32_t>(min(chunks, static_cast<int64_t>(kMaxGrid)));
    grid = max(grid, batch);
    LaunchKernel(grid, kBlock, device.unwrap()).enable_pdl(kUsePDL)(floor_topk_kernel<kUsePDL, T>, p);
  }

  static void run(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView out,
      const tvm::ffi::Optional<tvm::ffi::TensorView> page_table,
      const tvm::ffi::TensorView workspace,
      const uint32_t cap,
      const tvm::ffi::Optional<tvm::ffi::TensorView> stats) {
    const auto dt = scores.dtype();
    if (dt.code == DLDataTypeCode::kDLFloat && dt.bits == 32 && dt.lanes == 1) {
      run_t<fp32_t>(scores, lengths, out, page_table, workspace, cap, stats);
    } else if (dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16 && dt.lanes == 1) {
      run_t<bf16_t>(scores, lengths, out, page_table, workspace, cap, stats);
    } else {
      host::RuntimeCheck(false, "scores must be fp32 or bf16");
    }
  }
};

} // namespace sglang
