/**
 * \file topk_decode_floor_dbg_vcap.cuh
 * \brief Byte-floor decode-time top-k (k <= 2048) for DSA indexers: ONE
 * persistent launch, each logits row read (almost) once, optional fused
 * page-table transform.
 *
 * Replaces the 6-launch PDL chain of topk_decode_fg.cuh (two full reads per
 * row) with a single persistent kernel using two grid-wide spin barriers:
 *
 *   P0 plan   owner CTA (CTA r < B) samples its row (16 coalesced 128-elem
 *             windows = 2048 elems) into a smem coarse histogram and picks
 *             the capture threshold t_s (largest coarse bin whose scaled
 *             sample suffix >= ~6k). Rows with length <= cap capture
 *             everything (t_s = 0) and can never fall back. Naive rows
 *             (length <= topk) are deferred to P2.
 *   ---- grid barrier 1 ----
 *   P1 stream  THE read: block-cyclic (row, 4096-chunk) over the full grid.
 *             Each CTA accumulates the EXACT full coarse histogram in smem
 *             (flushed per row-switch with global atomics) and captures every
 *             element with coarse bin >= t_s into the per-row global candidate
 *             list ((f16-key << 32) | pos, u64, smem-staged, one global
 *             atomicAdd per flush). Overflows past the cap set a flag.
 *   ---- grid barrier 2 ----
 *   P2 select owner CTA per row: suffix-scans the exact full histogram to
 *             t_sel = largest bin with suffix >= topk, n_gt, r, n_eq (this
 *             always satisfies t_sel >= t_s, so {bin > t_sel} and {bin ==
 *             t_sel} are fully contained in the captured set). Fast path iff
 *             no overflow AND suffix(t_s) >= topk -- then the captured list
 *             is a superset of the true top-k, checked against the EXACT
 *             histogram (never the sample), so exactness never depends on
 *             sample quality. Otherwise the owner alone re-streams its row
 *             capturing bin >= t_sel (a second full read; only when the
 *             sample mispredicted or the threshold bin overflows the cap).
 *             Selection = fg's K4a+K4b fused: bin > t_sel -> output;
 *             bin == t_sel -> 256-bin sub-histogram (low byte of the f16
 *             key) -> sub > t2 -> output; sub == t2 -> residual -> fg's
 *             exact 4-round fp32 radix refinement. Optional fused transform:
 *             out[j] = page_table[row][pos] for pos >= 0 else -1 (the
 *             arithmetic of transform_index_page_table_decode, integer-exact).
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
 *     than the per-row cap, an arbitrary capped subset is refined (default
 *     cap = min(65536, stride), 16x production's 4096). Unlike fg, overflow
 *     never leaves stale output slots: unwritten slots are filled with -1.
 *
 * Grid: G = clamp(ceil(B*stride/4096), B, 132*4) CTAs of 256 threads,
 * co-resident by construction (__launch_bounds__(256, 4), 34 KB smem), so
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
constexpr uint32_t kStage = 4096;  // u64 candidate staging entries
constexpr uint32_t kRStage = 2048; // i32 residual staging entries
constexpr uint32_t kMaxTopK = 2048;
constexpr uint32_t kCTAsPerSM = 4; // co-residency for the spin barriers
constexpr uint32_t kMaxGrid = 132 * kCTAsPerSM;
constexpr int64_t kSampleTarget = 6144; // capture-size target (~3x topk)
// staging flush guard: headroom for one tile of appends (4/thread) + tail
constexpr int kStageGuard = 4 * kBlock + 8;

// smem layout (int32 words):
//   [0, 256)                  s_hist   (P0b sample hist, P1 exact hist)
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
  int32_t* __restrict__ hist;             // [B, 256] exact full coarse hist
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


// debug: global timestamp in ns (coherent across CTAs)
SGL_DEVICE unsigned long long dbg_timer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

template <bool kUsePDL, typename T>
__global__ __launch_bounds__(kBlock, kCTAsPerSM) void floor_topk_kernel(const FloorTopKParams p) {
  device::PDLWaitPrimary<kUsePDL>(); // P0 reads the predecessor's logits
  const unsigned long long dbg_t0 = (threadIdx.x == 0) ? dbg_timer() : 0;
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
  __shared__ int s_stage_n, s_cur_row, s_flush_base, s_do_flush;
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
      for (uint32_t wi = 0; wi < kWindows; ++wi) {
        const int64_t base = static_cast<int64_t>(wi) * wstride;
        for (int64_t i = base + tid; i < base + kWindow; i += kBlock)
          atomicAdd(&s_hist[sortable_f16(to_float(rowp[i])) >> 8], 1);
      }
      __syncthreads();
      s_scan[0][kRadix] = 0;
      s_scan[1][kRadix] = 0;
      if (tid < kRadix) s_scan[0][tid] = s_hist[tid];
      __syncthreads();
      suffix_scan_257(s_scan[0], s_scan[1]);
      if (tid == 0) s_t = 0; // default: capture everything (small rows)
      __syncthreads();
      if (tid < kRadix) {
        // t_s = largest coarse bin whose scaled sample suffix >= target
        const int64_t sample_n = static_cast<int64_t>(kWindows) * kWindow;
        const int64_t cur = static_cast<int64_t>(s_scan[0][tid]) * L / sample_n;
        const int64_t next = static_cast<int64_t>(s_scan[0][tid + 1]) * L / sample_n;
        if (cur >= kSampleTarget && next < kSampleTarget) s_t = static_cast<int>(tid);
      }
      __syncthreads();
      int t_s = s_t;
      if (static_cast<int64_t>(L) <= static_cast<int64_t>(p.cap)) t_s = 0; // exact, never falls back
      if (tid == 0) p.plan[static_cast<int64_t>(r) * 8 + 0] = t_s;
    }
    if (tid == 0 && p.stats != nullptr) p.stats[static_cast<int64_t>(r) * 8 + 4] = (int32_t)(dbg_timer() - dbg_t0);
  }

  grid_sync(p.barrier + 0, reinterpret_cast<volatile int*>(p.barrier + 1));

  // ---------------- P1: the single streaming read + capture ----------------
  {
    for (uint32_t i = tid; i < kRadix; i += kBlock) s_hist[i] = 0;
    if (tid == 0) {
      s_stage_n = 0;
      s_cur_row = -1;
      s_do_flush = 0;
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

    // flush the exact coarse histogram into hist[row]; all threads
    auto flush_hist = [&]() {
      const int row = s_cur_row;
      if (false && row >= 0 && tid < kRadix) {
        const int v = s_hist[tid];
        if (v != 0) {
          atomicAdd(&p.hist[static_cast<int64_t>(row) * kRadix + tid], v);
          s_hist[tid] = 0;
        }
      }
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
      // increasing order, so per-row smem state can be flushed on switch
      uint32_t r = 0, ci = 0, acc = 0;
      for (uint32_t rr = 0; rr < b; ++rr) {
        const uint32_t n = chunks_of_row(rr);
        if (u < acc + n) { r = rr; ci = u - acc; break; }
        acc += n;
      }
      if (s_cur_row != static_cast<int>(r)) {
        flush_stage();
        flush_hist();
        if (tid == 0) s_cur_row = static_cast<int>(r);
        __syncthreads();
      }
      const int32_t L = p.lengths[r];
      const int64_t beg = static_cast<int64_t>(ci) * kChunk;
      const int64_t end = min(beg + kChunk, static_cast<int64_t>(L));
      const T* rowp = rowp_of + static_cast<int64_t>(r) * p.stride;
      const uint32_t t_s = static_cast<uint32_t>(p.plan[static_cast<int64_t>(r) * 8 + 0]);

      auto process = [&](int64_t pos, float x) {
        const uint16_t key = sortable_f16(x);
        if ((key >> 8) >= t_s) {
          const int slot = atomicAdd(&s_stage_n, 1);
          s_cand[slot] = (static_cast<unsigned long long>(key) << 32) |
                         static_cast<unsigned long long>(pos);
        }
      };

      // no syncs inside the chunk: worst case one append per element = kChunk
      // = kStage, flushed at the chunk boundary (a uniform loop iteration)
      if (p.aligned) {
        const int64_t n4 = (end - beg) >> 2;
        for (int64_t g = tid; g < n4; g += kBlock) {
          float x[4];
          load4<T>(rowp + beg + g * 4, x);
#pragma unroll
          for (int e = 0; e < 4; ++e) process(beg + g * 4 + e, x[e]);
        }
        for (int64_t i = beg + (n4 << 2) + tid; i < end; i += kBlock)
          process(i, to_float(rowp[i]));
      } else {
        for (int64_t i = beg + tid; i < end; i += kBlock) process(i, to_float(rowp[i]));
      }
      flush_stage(); // chunk boundary: appends this chunk <= kChunk <= kStage
    }
    flush_hist();
  }

  grid_sync(p.barrier + 2, reinterpret_cast<volatile int*>(p.barrier + 3));
  const unsigned long long dbg_t2 = (threadIdx.x == 0) ? dbg_timer() : 0;

  // ---------------- P2: verify + select (owner CTA per row) ----------------
  if (g < b) {
    if (threadIdx.x == 0 && p.stats != nullptr) p.stats[static_cast<int64_t>(g) * 8 + 5] = (int32_t)(dbg_t2 - dbg_t0);
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
      // exact full-histogram suffix scan -> t_sel, n_eq, r
      s_scan[0][kRadix] = 0;
      s_scan[1][kRadix] = 0;
      if (tid < kRadix) s_scan[0][tid] = p.hist[static_cast<int64_t>(r) * kRadix + tid];
      __syncthreads();
      suffix_scan_257(s_scan[0], s_scan[1]);
      if (tid == 0) { s_t = -1; s_filled = 0; }
      __syncthreads();
      if (tid < kRadix) {
        if (s_scan[0][tid] >= static_cast<int>(p.topk) &&
            s_scan[0][tid + 1] < static_cast<int>(p.topk)) {
          s_t = static_cast<int>(tid);
          s_n_eq = s_scan[0][tid] - s_scan[0][tid + 1];
          s_r = static_cast<int>(p.topk) - s_scan[0][tid + 1];
        }
      }
      __syncthreads();
      const int t_sel = s_t;
      const int r_need = s_r;
      int flags = 0;
      if (t_sel < 0) {
        // impossible (suffix(0) = L > topk, suffix(256) = 0); bail defensively
        // but keep the workspace clean for the next replay
        flags |= 4;
        for (uint32_t j = tid; j < p.topk; j += kBlock) write_out(j, -1);
        if (tid < kRadix) p.hist[static_cast<int64_t>(r) * kRadix + tid] = 0;
        if (tid == 0) {
          p.counters[static_cast<int64_t>(r) * 8 + 0] = 0;
          p.counters[static_cast<int64_t>(r) * 8 + 1] = 0;
          if (p.stats != nullptr) {
            p.stats[static_cast<int64_t>(r) * 4 + 0] = 0;
            p.stats[static_cast<int64_t>(r) * 4 + 1] = 0;
            p.stats[static_cast<int64_t>(r) * 4 + 2] = 0;
            p.stats[static_cast<int64_t>(r) * 4 + 3] = flags;
          }
        }
      } else {
        const int t_s = p.plan[static_cast<int64_t>(r) * 8 + 0];
        const int suffix_ts = s_scan[0][t_s];
        const int ovfl = p.counters[static_cast<int64_t>(r) * 8 + 1] & 2;
        const int n_cap = p.counters[static_cast<int64_t>(r) * 8 + 0];

        int n_list;
        if (!ovfl && suffix_ts >= static_cast<int>(p.topk)) {
          n_list = n_cap; // == suffix(t_s): fast path, captured ⊇ true top-k
        } else {
          // fallback: this owner alone re-streams the row, capturing bin >= t_sel
          flags |= 1;
          const uint32_t t_sel_u = static_cast<uint32_t>(t_sel);
          if (tid == 0) { s_stage_n = 0; s_do_flush = 0; }
          __syncthreads();
          int64_t kept = 0;
          auto fb_store = [&](int n) {
            const int n_store = min(n, static_cast<int>(p.cap) - static_cast<int>(kept));
            if (n_store < n) flags |= 2; // cap overflow (fg's inexactness class)
            unsigned long long* dst = p.cand + static_cast<int64_t>(r) * p.cap + kept;
            for (int i = tid; i < n_store; i += kBlock) dst[i] = s_cand[i];
            kept += n;
          };
          auto fb_process = [&](int64_t pos, float x) {
            const uint16_t key = sortable_f16(x);
            if ((key >> 8) >= t_sel_u) {
              const int slot = atomicAdd(&s_stage_n, 1);
              s_cand[slot] = (static_cast<unsigned long long>(key) << 32) |
                             static_cast<unsigned long long>(pos);
            }
          };
          if (p.aligned) {
            const int64_t n4all = L >> 2;
            for (int64_t c0 = 0; c0 < L; c0 += kChunk) {
              const int64_t beg = c0;
              const int64_t end = min(c0 + kChunk, static_cast<int64_t>(L));
              const int64_t n4 = (end - beg) >> 2;
              for (int64_t g = tid; g < n4; g += kBlock) {
                float x[4];
                load4<T>(rowp + beg + g * 4, x);
#pragma unroll
                for (int e = 0; e < 4; ++e) fb_process(beg + g * 4 + e, x[e]);
              }
              for (int64_t i = beg + (n4 << 2) + tid; i < end; i += kBlock)
                fb_process(i, to_float(rowp[i]));
              __syncthreads();
              fb_store(s_stage_n);
              if (tid == 0) s_stage_n = 0;
              __syncthreads();
            }
          } else {
            for (int64_t c0 = 0; c0 < L; c0 += kChunk) {
              const int64_t beg = c0;
              const int64_t end = min(c0 + kChunk, static_cast<int64_t>(L));
              for (int64_t i = beg + tid; i < end; i += kBlock)
                fb_process(i, to_float(rowp[i]));
              __syncthreads();
              fb_store(s_stage_n);
              if (tid == 0) s_stage_n = 0;
              __syncthreads();
            }
          }
          n_list = static_cast<int>(min(kept, static_cast<int64_t>(p.cap)));
          if (kept > static_cast<int64_t>(p.cap)) flags |= 2;
        }

        // ---- selection over cand[r][0, n_list) ----
        if (tid == 0) { s_outA_n = 0; s_eq_n = 0; }
        for (uint32_t i = tid; i < kRadix; i += kBlock) s_sub[i] = 0;
        __syncthreads();
        const unsigned long long* list = p.cand + static_cast<int64_t>(r) * p.cap;
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

        // sub-bin threshold t2 (fg K4a/K4-plan2 semantics); absent crossing
        // (eq exhausted below r_need, cap-overflow class only) -> -1 = take all
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

        // pass 2: bin == t_sel: sub > t2 -> outB, sub == t2 -> residual list
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
        if (r2 > 0 && n2 <= 0) flags |= 4; // residual empty but r2 > 0: inconsistent
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

        // self-clean: hist + counters (consumed above)
        if (tid < kRadix) p.hist[static_cast<int64_t>(r) * kRadix + tid] = 0;
        if (tid == 0) {
          p.counters[static_cast<int64_t>(r) * 8 + 0] = 0;
          p.counters[static_cast<int64_t>(r) * 8 + 1] = 0;
          if (p.stats != nullptr) {
            p.stats[static_cast<int64_t>(r) * 8 + 6] = (int32_t)(dbg_timer() - dbg_t0);
            p.stats[static_cast<int64_t>(r) * 8 + 0] = s_n_eq;
            p.stats[static_cast<int64_t>(r) * 4 + 1] = s_eq_n;
            p.stats[static_cast<int64_t>(r) * 4 + 2] = r_need;
            p.stats[static_cast<int64_t>(r) * 4 + 3] = flags;
          }
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
      TensorMatcher({B, 8}).with_dtype<int32_t>().with_device(device).verify(stats.value());
      stats_ptr = static_cast<int32_t*>(stats.value().data_ptr());
    }

    const auto batch = static_cast<uint32_t>(B.unwrap());
    const auto stride = static_cast<int64_t>(S.unwrap());
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");
    RuntimeCheck(batch > 0 && batch <= kMaxRows, "batch too large for the persistent grid");
    RuntimeCheck(cap > topk, "cap must exceed topk");

    // [barrier 8 | hist B*256 | plan B*8 | counters B*8 | (pad) | cand B*cap
    //  u64 | resid B*cap | resid2 B*cap]
    const int64_t n_head = 8;
    const int64_t n_small = n_head + static_cast<int64_t>(batch) * (kRadix + 8 + 8);
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
        .hist = ws + 8,
        .plan = ws + 8 + static_cast<int64_t>(batch) * kRadix,
        .counters = ws + 8 + static_cast<int64_t>(batch) * (kRadix + 8),
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
