/**
 * \file topk_prefill_fused.cuh
 * \brief Single-kernel exact top-k + page-table transform for the DSA prefill
 * indexer (ka-topk-select v2).
 *
 * The production 1-pass kernel (topk_prefill_1pass.cuh) is ONE 1024-thread
 * block per row: windowed 12-bit sample -> threshold, one double-buffered
 * float4 read with per-match smem atomicAdd append, in-smem select
 * (handle_tie), in-kernel 2-pass fallback, page-table transform. It sustains
 * ~3.3 TB/s (90% of the 3.665 TB/s measured HBM roofline) — but its 12-bit
 * sample threshold degenerates on real fp8 logits (17-23 fallback rows per
 * 8192-row chunk at L=950k, each +2 full reads).
 *
 * This kernel is the 1-pass structure with ONE change: the sample threshold
 * is the ka-topk two-level 16-bit key threshold (coarse 256-bin + sub-bin =
 * full fp16 key), which slices crowded fp16 bins — the degenerate-histogram
 * class measured by prefill-energy and confirmed by ka-topk-issue is
 * eliminated (0 fallback rows on real logits). The fallback (rare) is the
 * fg-class 2-pass + overflow sub-pass: coarse-16 histogram (read 2) ->
 * gather boundary bin (read 3); if the boundary bin overflows the tie buffer,
 * a sub-bin pass (read 4) selects within it; exact except when a single fp16
 * VALUE's mass exceeds kCap (fg's documented class).
 *
 * Semantics identical to fast_topk_transform_prefill_1pass (production):
 * dst[row, t] = src_page_table[seq(row)][pos_t] (page size 1), -1 padding,
 * naive <= topk rows keep the first `length` table entries (production
 * quirk), row_starts windows, cu_seqlens_q or row_to_page (PAGETABLE_HOIST)
 * source resolution.
 */
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/deepseek_v4/topk_impl.cuh>

#include "topk_keys.cuh"

#include <cuda_fp16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>

namespace sglang {

namespace impl = device::topk;
using impl::TieValue;
using impl::warp_inclusive_sum;

namespace {

constexpr uint32_t kBlockSize = impl::kWarpSize * 32;  // 1024
constexpr uint32_t kNumWarps = kBlockSize / impl::kWarpSize;
constexpr uint32_t kCap = 8192;      // candidate buffer capacity (smem)
constexpr uint32_t kSampleWindows = 64;
constexpr uint32_t kTieItems = kCap / kBlockSize;  // 8
using vec_t = device::AlignedVector<float, 4>;

// --- sortable_f16 / coarse16 / key16_to_value / suffix_scan_257: topk_keys.cuh ---

// ---------------------------------------------------------------------------
// Shared memory (~93 KB dynamic)
// ---------------------------------------------------------------------------

struct alignas(16) MatchBin {
  uint32_t bin;
  uint32_t above_count;
  uint32_t equal_count;
  uint32_t _pad;
};

struct TieHandleSmem {
  alignas(128) uint32_t counter;
  alignas(128) uint32_t counter_final;
  MatchBin match;
  uint32_t warp_sum[kNumWarps];
  uint32_t histogram[2][kRadix];
};

struct FusedSmem {
  union {
    struct {  // sample phase: 16 KB keys + 32 KB warp-privatized coarse
      uint16_t keys[kSampleMax];
      int coarse[kNumWarps][kRadix];
    } sample;
    struct {  // collect phase: 64 KB
      uint32_t count;
      uint32_t pad[63];
      TieValue cand[kCap];
    } collect;
    struct {  // fallback gather: 64 KB
      uint32_t count_gt;
      uint32_t count_eq;
      uint32_t pad2[62];
      TieValue tie[kCap];
    } fb;
  } u;
  int scan[2][kRadix + 1];         // sample scans + fallback coarse/sub hists
  int32_t out_indices[kMaxTopK];   // select output: 8 KB
  int tbin, ngt, tsub, ngt_sub;    // plan scalars
  TieHandleSmem tie;               // ~2.2 KB (radix_tie_select scratch)
};
static_assert(sizeof(FusedSmem) <= 96 * 1024, "smem budget");

// ---------------------------------------------------------------------------
// In-smem select: the 1-pass's tie machinery, verbatim (kMaxNumTie = kCap)
// ---------------------------------------------------------------------------

SGL_DEVICE void tie_emit(int32_t* out, const uint32_t pos, const uint32_t raw_idx) {
  out[pos] = static_cast<int32_t>(raw_idx);
}

/// Exact radix select over tie candidates: each thread owns kItems strided
/// elements (inactive beyond num_ties). Requires num_ties <= kItems*kBlockSize.
template <uint32_t kItems>
SGL_DEVICE void radix_tie_select(
    const TieValue* tie_buffer,
    int32_t* out,
    const uint32_t base,
    const uint32_t num_ties,
    const uint32_t topk,
    TieHandleSmem* smem) {
  const auto tx = threadIdx.x;
  const auto lane_id = tx % impl::kWarpSize;
  const auto warp_id = tx / impl::kWarpSize;

  bool active[kItems];
  uint32_t key[kItems];
  uint32_t idx[kItems];
  uint32_t write_pos[kItems];
#pragma unroll
  for (uint32_t i = 0; i < kItems; ++i) {
    const auto t = tx + i * kBlockSize;
    active[i] = t < num_ties;
    const auto tie = active[i] ? tie_buffer[t] : TieValue::invalid();
    key[i] = impl::extract_exact_bin(tie.value);
    idx[i] = tie.idx;
    write_pos[i] = topk;
  }
  uint32_t topk_remain = topk;
  if (tx < kRadix) smem->histogram[0][tx] = 0;
  if (tx == kRadix) smem->counter = smem->counter_final = 0;
  __syncthreads();
  uint32_t total_active = num_ties;

#pragma unroll 1
  for (int round = 0; round < 4; ++round) {
    const uint32_t shift = 24 - round * 8;
    const auto hist_idx = round % 2;
    const auto histogram = smem->histogram[hist_idx];

#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      if (active[i]) atomicAdd(&histogram[(key[i] >> shift) & 0xFFu], 1);
    }
    if (round < 3 && tx < kRadix) {
      smem->histogram[hist_idx ^ 1][tx] = 0;
    }
    __syncthreads();

    uint32_t hist_val = 0;
    uint32_t warp_inc = 0;
    if (tx < kRadix) {
      hist_val = histogram[tx];
      warp_inc = warp_inclusive_sum(lane_id, hist_val);
      if (lane_id == impl::kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
    }
    __syncthreads();
    if (tx < kRadix) {
      const auto inter = device::warp::reduce_sum(lane_id < warp_id ? smem->warp_sum[lane_id] : 0);
      const auto prefix = inter + warp_inc;      // inclusive prefix through this bin
      const auto above = total_active - prefix;  // elements in bins ABOVE this one
      if (above < topk_remain && above + hist_val >= topk_remain) {
        smem->match = {tx, above, hist_val, 0};
      }
    }
    __syncthreads();

    const auto [threshold_bin, above_count, equal_count, __] = smem->match;
    if (round < 3) total_active = equal_count;
    topk_remain -= above_count;

    // scatter
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      if (!active[i]) continue;
      const uint32_t bin = (key[i] >> shift) & 0xFFu;
      if (bin > threshold_bin) {
        write_pos[i] = atomicAdd(&smem->counter, 1);
        active[i] = false;
      } else if (bin < threshold_bin) {
        active[i] = false;
      } else if (round == 3) {
        write_pos[i] = topk - topk_remain + atomicAdd(&smem->counter_final, 1);
      }
      // bin == thr && round < 3: stay active for next round
    }

    if (round == 3 || topk_remain == 0) break;
    __syncthreads();
  }

#pragma unroll
  for (uint32_t i = 0; i < kItems; ++i) {
    if (write_pos[i] < topk) tie_emit(out, base + write_pos[i], idx[i]);
  }
}

/// Select the top-`topk` (by (value, idx)) of tie_buffer[0..num_ties) into
/// out[base .. base+topk). Slots below `base` are assumed already filled.
SGL_DEVICE void handle_tie(
    const TieValue* tie_buffer,
    int32_t* out,
    const uint32_t base,
    const uint32_t num_ties,
    const uint32_t topk,
    TieHandleSmem* smem) {
  constexpr auto is_greater = [](const TieValue& a, const TieValue& b) {
    return (a.value > b.value) || (a.value == b.value && a.idx < b.idx);
  };
  const auto tx = threadIdx.x;
  const auto lane_id = tx % impl::kWarpSize;
  const auto warp_id = tx / impl::kWarpSize;
  static_assert(kNumWarps == impl::kWarpSize);

  if (num_ties <= topk) {
    for (uint32_t t = tx; t < num_ties; t += kBlockSize) {
      tie_emit(out, base + t, tie_buffer[t].idx);
    }
    for (uint32_t t = num_ties + tx; t < topk; t += kBlockSize) {
      tie_emit(out, base + t, base + t);  // pad (unreachable in this kernel)
    }
  } else if (num_ties <= impl::kWarpSize) {
    if (lane_id >= num_ties || warp_id >= num_ties) return;  // some threads idle
    const uint32_t mask = (1ull << num_ties) - 1u;
    const auto tie = tie_buffer[lane_id];
    const auto target = tie_buffer[warp_id];
    const auto rank = impl::warp_sum_bool(is_greater(tie, target), mask);
    if (lane_id == 0 && rank < topk) tie_emit(out, base + rank, target.idx);
  } else if (num_ties <= impl::kWarpSize * 2) {
    const auto warp_id_0 = warp_id;
    const auto warp_id_1 = warp_id + impl::kWarpSize;
    const auto lane_id_1 = lane_id + impl::kWarpSize;
    const auto invalid = TieValue::invalid();
    const auto tie_0 = tie_buffer[lane_id];
    const auto tie_1 = lane_id_1 < num_ties ? tie_buffer[lane_id_1] : invalid;
    const auto target_0 = tie_buffer[warp_id_0];
    const auto target_1 = tie_buffer[warp_id_1];
    {
      const auto rank = impl::warp_sum_bool(is_greater(tie_0, target_0)) +
                        impl::warp_sum_bool(is_greater(tie_1, target_0));
      if (lane_id == 0 && rank < topk) tie_emit(out, base + rank, target_0.idx);
    }
    if (warp_id_1 < num_ties) {
      const auto rank = impl::warp_sum_bool(is_greater(tie_0, target_1)) +
                        impl::warp_sum_bool(is_greater(tie_1, target_1));
      if (lane_id == 0 && rank < topk) tie_emit(out, base + rank, target_1.idx);
    }
  } else if (num_ties <= impl::kWarpSize * 4) {
    const auto invalid = TieValue::invalid();
    const TieValue tie[] = {
        tie_buffer[lane_id + 0 * impl::kWarpSize],
        tie_buffer[lane_id + 1 * impl::kWarpSize],
        lane_id + 2 * impl::kWarpSize < num_ties ? tie_buffer[lane_id + 2 * impl::kWarpSize] : invalid,
        lane_id + 3 * impl::kWarpSize < num_ties ? tie_buffer[lane_id + 3 * impl::kWarpSize] : invalid,
    };
    const TieValue target[] = {
        tie_buffer[warp_id + 0 * impl::kWarpSize],
        tie_buffer[warp_id + 1 * impl::kWarpSize],
        tie_buffer[warp_id + 2 * impl::kWarpSize],
        tie_buffer[warp_id + 3 * impl::kWarpSize],
    };
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (i >= 2 && warp_id + i * impl::kWarpSize >= num_ties) break;
      uint32_t rank = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        rank += impl::warp_sum_bool(is_greater(tie[j], target[i]));
      }
      if (lane_id == 0 && rank < topk) tie_emit(out, base + rank, target[i].idx);
    }
  } else if (num_ties <= kBlockSize) {
    radix_tie_select<1>(tie_buffer, out, base, num_ties, topk, smem);
  } else {
    radix_tie_select<kTieItems>(tie_buffer, out, base, num_ties, topk, smem);
  }
}

// ---------------------------------------------------------------------------
// Streaming window iteration (1-pass's stream_window, verbatim)
// ---------------------------------------------------------------------------

template <typename F>
SGL_DEVICE void stream_window(const float* __restrict__ row, const uint32_t length, F&& fn) {
  const auto tx = threadIdx.x;
  const uintptr_t base_addr = reinterpret_cast<uintptr_t>(row);
  const uint32_t misalign = (16u - (base_addr & 15u)) & 15u;
  uint32_t head = misalign >> 2;
  head = head < length ? head : length;
  const uint32_t body = (length - head) & ~3u;
  const uint32_t tail = length - head - body;

  if (tx < head) fn(row[tx], tx);
  const uint32_t nvec = body / 4;
  const float* const body_ptr = row + head;

  vec_t next_vec;
  uint32_t vi = tx;
  if (vi < nvec) next_vec.load(body_ptr, vi);
  while (vi < nvec) {
    const auto cur = next_vec;
    const auto base = head + vi * 4;
    vi += kBlockSize;
    if (vi < nvec) next_vec.load(body_ptr, vi);
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      fn(cur[j], base + j);
    }
  }
  if (tx < tail) fn(row[head + body + tx], head + body + tx);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

struct PrefillFusedParams {
  const float* __restrict__ input;             // [B, input_stride]
  const int32_t* __restrict__ row_starts;      // [B] or nullptr
  const int32_t* __restrict__ lengths;         // [B]
  const int32_t* __restrict__ src_page_table;  // [prefill_bs, src_stride]
  const int32_t* __restrict__ cu_seqlens_q;    // [prefill_bs + 1] or nullptr
  const int32_t* __restrict__ row_to_page;     // [B] or nullptr (PAGETABLE_HOIST)
  int32_t* __restrict__ dst;                   // [B, topk]
  int32_t* __restrict__ stats;                 // [B, 4] or nullptr
  int64_t input_stride;
  int64_t src_stride;
  uint32_t prefill_bs;
  uint32_t topk;
  uint32_t target;      // capture-count target (3x topk, clamped)
  uint32_t cap;         // fast-path bound (<= kCap)
};

__global__ __launch_bounds__(kBlockSize, 2) void topk_prefill_fused_kernel(
    const __grid_constant__ PrefillFusedParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto* s = reinterpret_cast<FusedSmem*>(smem_raw);
  const uint32_t bid = blockIdx.x;
  const uint32_t tx = threadIdx.x;

  // ---- resolve the source page-table row (1-pass logic, verbatim) ----
  __shared__ const int32_t* s_src;
  if (p.row_to_page != nullptr) {
    if (tx == 0) {
      s_src = p.src_page_table + static_cast<int64_t>(p.row_to_page[bid]) * p.src_stride;
    }
  } else if (p.prefill_bs <= kBlockSize) {
    if (tx < p.prefill_bs) {
      if (bid >= (uint32_t)p.cu_seqlens_q[tx] && bid < (uint32_t)p.cu_seqlens_q[tx + 1]) {
        s_src = p.src_page_table + static_cast<int64_t>(tx) * p.src_stride;
      }
    }
  } else {
    for (uint32_t i = tx; i < p.prefill_bs; i += kBlockSize) {
      if (bid >= (uint32_t)p.cu_seqlens_q[i] && bid < (uint32_t)p.cu_seqlens_q[i + 1]) {
        s_src = p.src_page_table + static_cast<int64_t>(i) * p.src_stride;
      }
    }
  }
  __syncthreads();
  const int32_t* src_row = s_src;

  const int32_t length_i32 = p.lengths[bid];
  const int32_t row_start = p.row_starts != nullptr ? p.row_starts[bid] : 0;
  const float* row = p.input + static_cast<int64_t>(bid) * p.input_stride + row_start;
  int32_t* out = p.dst + static_cast<int64_t>(bid) * p.topk;

  // Naive path (length <= topk), production bit-for-bit incl. the quirk
  if (length_i32 <= 0 || (uint32_t)length_i32 <= p.topk) {
    for (uint32_t i = tx; i < p.topk; i += kBlockSize) {
      out[i] = (int32_t)i < length_i32 ? src_row[i] : -1;
    }
    if (p.stats != nullptr && tx == 0) {
      p.stats[bid * 4 + 0] = length_i32;
      p.stats[bid * 4 + 1] = length_i32;
      p.stats[bid * 4 + 2] = 0;
      p.stats[bid * 4 + 3] = 0;
    }
    return;
  }
  const uint32_t length = static_cast<uint32_t>(length_i32);
  const uint32_t topk = p.topk;

  // ---- Phase 1: windowed two-level 16-bit sample -> conservative v_s ----
  float v_s;
  if (length <= kSampleMax) {
    v_s = -FLT_MAX;  // capture everything; count = length <= kSampleMax <= cap
  } else {
    const uint32_t n_s = min(kSampleMax, max(kSampleMin, length / 8)) & ~63u;
    const uint32_t E = n_s / kSampleWindows;
    for (uint32_t i = tx; i < kNumWarps * kRadix; i += kBlockSize)
      (&s->u.sample.coarse[0][0])[i] = 0;
    s->scan[0][kRadix] = 0;
    s->scan[1][kRadix] = 0;
    if (tx == 0) s->tbin = 0;
    __syncthreads();
    const uint32_t warp = tx / 32;
    for (uint32_t t = tx; t < n_s; t += kBlockSize) {
      const uint32_t c = t / E;
      const uint32_t e = t % E;
      uint32_t off = static_cast<uint32_t>(
          (static_cast<uint64_t>(c) * static_cast<uint64_t>(length)) / kSampleWindows) + e;
      off = min(off, length - 1);
      const uint16_t key = sortable_f16(row[off]);
      s->u.sample.keys[t] = key;
      atomicAdd(&s->u.sample.coarse[warp][key >> 8], 1);
    }
    __syncthreads();
    if (tx < kRadix) {
      int acc = 0;
      for (uint32_t w = 0; w < kNumWarps; ++w) acc += s->u.sample.coarse[w][tx];
      s->scan[0][tx] = acc;
    }
    __syncthreads();
    // the coarse rows are dead now: reuse row 0 as the sub-bin histogram
    // (MUST be re-zeroed first — it still holds warp 0's coarse counts)
    for (uint32_t i = tx; i < kRadix; i += kBlockSize) s->u.sample.coarse[0][i] = 0;
    __syncthreads();
    uint32_t j = (p.target * n_s + length - 1) / length;
    j = max(j, 1u);
    j = min(j, n_s);
    suffix_scan_257(s->scan[0], s->scan[1]); // scan[0][i] = #{coarse >= i}
    if (tx < kRadix) {
      if (s->scan[0][tx] >= (int)j && s->scan[0][tx + 1] < (int)j)
        s->tbin = (int)tx;
    }
    __syncthreads();
    const int tc = s->tbin;
    if (tc <= 0) {
      v_s = -FLT_MAX;  // threshold below everything: capture all
    } else {
      if (tx == 0) s->ngt = s->scan[0][tc + 1];
      __syncthreads();
      const int above = s->ngt;
      // sub-bin pass over the sampled keys in the boundary coarse bin
      for (uint32_t t = tx; t < n_s; t += kBlockSize) {
        const uint16_t key = s->u.sample.keys[t];
        if ((key >> 8) == (uint16_t)tc)
          atomicAdd(&s->u.sample.coarse[0][key & 0xFFu], 1); // sub-hist (coarse dead)
      }
      __syncthreads();
      if (tx < kRadix) s->scan[0][tx] = s->u.sample.coarse[0][tx];
      s->scan[0][kRadix] = 0;
      s->scan[1][kRadix] = 0;
      __syncthreads();
      suffix_scan_257(s->scan[0], s->scan[1]); // scan[0][i] = #{sub >= i}
      if (tx == 0) s->tsub = -1;
      __syncthreads();
      if (tx < kRadix) {
        if (above + s->scan[0][tx] >= (int)j && above + s->scan[0][tx + 1] < (int)j)
          s->tsub = (int)tx;
      }
      __syncthreads();
      const uint32_t K_s =
          (static_cast<uint32_t>(tc) << 8) | static_cast<uint32_t>(max(s->tsub, 0));
      v_s = (K_s > 0) ? key16_to_value(static_cast<uint16_t>(K_s - 1)) : -FLT_MAX;
    }
    __syncthreads(); // sample structs done; collect reuses the union
  }

  // ---- Phase 2: the single collect pass ----
  if (tx == 0) s->u.collect.count = 0;
  __syncthreads();
  stream_window(row, length, [&](float x, uint32_t idx) {
    if (!(x < v_s)) { // NaN-inclusive
      const auto pos = atomicAdd(&s->u.collect.count, 1);
      if (pos < kCap) s->u.collect.cand[pos] = TieValue{x, idx};
    }
  });
  __syncthreads();
  const uint32_t count = s->u.collect.count;

  if (count >= topk && count <= p.cap) {
    // ---- Phase 3a: exact in-smem select over the full candidate set ----
    handle_tie(s->u.collect.cand, s->out_indices, 0, count, topk, &s->tie);
    if (p.stats != nullptr && tx == 0) {
      p.stats[bid * 4 + 0] = static_cast<int32_t>(count);
      p.stats[bid * 4 + 1] = static_cast<int32_t>(min(count, p.cap));
      p.stats[bid * 4 + 2] = 0;
      p.stats[bid * 4 + 3] = 0;
    }
  } else {
    // ---- Phase 3b: fg-class fallback (reads 2-3, +1 only on tie overflow) ----
    for (uint32_t i = tx; i < kRadix; i += kBlockSize) s->scan[0][i] = 0;
    if (tx == 0) { s->tbin = -1; s->ngt = 0; }
    __syncthreads();
    stream_window(row, length, [&](float x, uint32_t) {
      atomicAdd(&s->scan[0][coarse16(x)], 1);
    });
    __syncthreads();
    suffix_scan_257(s->scan[0], s->scan[1]);
    if (tx < kRadix) {
      if (s->scan[0][tx] >= (int)topk && s->scan[0][tx + 1] <= (int)topk) {
        s->tbin = (int)tx;
        s->ngt = s->scan[0][tx + 1];
      }
    }
    __syncthreads();
    const int t0 = s->tbin;   // >= 0: hist sums to length > topk
    const int ngt0 = s->ngt;
    const int remain = (int)topk - ngt0; // > 0 unless the plateau hit exactly

    // gather (read 3): coarse > t0 -> out; coarse == t0 -> tie buffer
    // (+ full sub-hist of the boundary bin into scan[0]... built below on overflow)
    if (tx == 0) { s->u.fb.count_gt = 0; s->u.fb.count_eq = 0; }
    __syncthreads();
    stream_window(row, length, [&](float x, uint32_t idx) {
      const int c = coarse16(x);
      if (c > t0) {
        const auto pos = atomicAdd(&s->u.fb.count_gt, 1);
        if (pos < topk) s->out_indices[pos] = static_cast<int32_t>(idx);
      } else if (c == t0 && remain > 0) {
        const auto pos = atomicAdd(&s->u.fb.count_eq, 1);
        if (pos < kCap) s->u.fb.tie[pos] = TieValue{x, idx};
      }
    });
    __syncthreads();
    const uint32_t above = s->u.fb.count_gt;             // <= topk (crossing)
    const uint32_t eq_stored = min(s->u.fb.count_eq, kCap);

    if (s->u.fb.count_eq <= kCap) {
      // no overflow: the tie buffer holds the WHOLE boundary bin -> fp32-exact
      if (remain > 0)
        handle_tie(s->u.fb.tie, s->out_indices, above, eq_stored, (uint32_t)remain, &s->tie);
    } else {
      // boundary bin overflowed the tie buffer: sub-bin pass (read 4) over the
      // FULL boundary bin, then regather (read 5... folded: hist pass first)
      for (uint32_t i = tx; i < kRadix; i += kBlockSize) s->scan[0][i] = 0;
      __syncthreads();
      stream_window(row, length, [&](float x, uint32_t) {
        if (coarse16(x) == t0)
          atomicAdd(&s->scan[0][sortable_f16(x) & 0xFFu], 1);
      });
      __syncthreads();
      s->scan[0][kRadix] = 0;
      s->scan[1][kRadix] = 0;
      __syncthreads();
      suffix_scan_257(s->scan[0], s->scan[1]); // scan[0][i] = #{sub >= i} (full bin)
      if (tx < kRadix) { s->tsub = -1; s->ngt_sub = 0; }
      __syncthreads();
      if (tx < kRadix) {
        if (s->scan[0][tx] >= remain && s->scan[0][tx + 1] <= remain) {
          s->tsub = (int)tx;
          s->ngt_sub = s->scan[0][tx + 1];
        }
      }
      __syncthreads();
      const int tsub = s->tsub;      // >= 0 (sub-hist sums to count_eq > kCap >= remain)
      const int ngt_sub = s->ngt_sub;
      const int rem2 = remain - ngt_sub; // >= 0
      // regather the boundary bin by sub-key (read 5): sub > tsub -> out;
      // sub == tsub (single fp16 VALUE) -> tie buffer (fg's tie class)
      if (tx == 0) { s->u.fb.count_gt = 0; s->u.fb.count_eq = 0; }
      __syncthreads();
      stream_window(row, length, [&](float x, uint32_t idx) {
        if (coarse16(x) == t0) {
          const int sub = static_cast<int>(sortable_f16(x) & 0xFFu);
          if (sub > tsub) {
            // #{sub > tsub} == ngt_sub over the full bin (sub-hist was full)
            const auto pos = atomicAdd(&s->u.fb.count_gt, 1);
            if (pos < (uint32_t)ngt_sub) s->out_indices[above + pos] = static_cast<int32_t>(idx);
          } else if (sub == tsub && rem2 > 0) {
            const auto pos = atomicAdd(&s->u.fb.count_eq, 1);
            if (pos < kCap) s->u.fb.tie[pos] = TieValue{x, idx};
          }
        }
      });
      __syncthreads();
      const uint32_t eq2 = min(s->u.fb.count_eq, kCap);
      if (rem2 > 0)
        handle_tie(s->u.fb.tie, s->out_indices, above + (uint32_t)ngt_sub, eq2,
                   (uint32_t)rem2, &s->tie);
    }
    if (p.stats != nullptr && tx == 0) {
      p.stats[bid * 4 + 0] = static_cast<int32_t>(count);
      p.stats[bid * 4 + 1] = static_cast<int32_t>(min(count, p.cap));
      p.stats[bid * 4 + 2] = static_cast<int32_t>(remain);
      p.stats[bid * 4 + 3] = 1 | (count > p.cap ? 2 : 4);
    }
  }
  __syncthreads();

  // ---- Phase 4: page-table transform (page_size = 1) ----
  for (uint32_t t = tx; t < topk; t += kBlockSize) {
    const int32_t raw = s->out_indices[t];
    out[t] = raw < 0 ? -1 : src_row[raw];
  }
}

} // namespace

struct TopKBallotPrefill {
  static void transform(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView dst,
      const tvm::ffi::TensorView src_page_table,
      const tvm::ffi::Optional<tvm::ffi::TensorView> cu_seqlens_q,
      const tvm::ffi::Optional<tvm::ffi::TensorView> row_starts,
      const tvm::ffi::Optional<tvm::ffi::TensorView> stats,
      const tvm::ffi::Optional<tvm::ffi::TensorView> row_to_page,
      const uint32_t target) {
    using namespace host;
    auto B = SymbolicSize{"num_rows"};
    auto L = SymbolicSize{"max_seq_len"};
    auto S = SymbolicSize{"score_stride"};
    auto K = SymbolicSize{"topk"};
    auto BS = SymbolicSize{"prefill_bs"};
    auto BSp1 = SymbolicSize{"prefill_bs_plus_1"};
    auto P = SymbolicSize{"page_table_stride"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, L}).with_strides({S, 1}).with_dtype<float>().with_device(device_).verify(scores);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(lengths);
    TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device_).verify(dst);
    TensorMatcher({BS, -1}).with_strides({P, 1}).with_dtype<int32_t>().with_device(device_).verify(src_page_table);

    const int32_t* cu_seqlens_ptr = nullptr;
    if (cu_seqlens_q.has_value()) {
      TensorMatcher({BSp1}).with_dtype<int32_t>().with_device(device_).verify(cu_seqlens_q.value());
      cu_seqlens_ptr = static_cast<const int32_t*>(cu_seqlens_q.value().data_ptr());
    }
    const int32_t* row_to_page_ptr = nullptr;
    if (row_to_page.has_value()) {
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(row_to_page.value());
      row_to_page_ptr = static_cast<const int32_t*>(row_to_page.value().data_ptr());
    }
    const int32_t* row_starts_ptr = nullptr;
    if (row_starts.has_value()) {
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(row_starts.value());
      row_starts_ptr = static_cast<const int32_t*>(row_starts.value().data_ptr());
    }
    int32_t* stats_ptr = nullptr;
    if (stats.has_value()) {
      TensorMatcher({B, 4}).with_dtype<int32_t>().with_device(device_).verify(stats.value());
      stats_ptr = static_cast<int32_t*>(stats.value().data_ptr());
    }

    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");
    RuntimeCheck(BS.unwrap() >= 1, "prefill_bs must be >= 1");
    RuntimeCheck(BS.unwrap() <= B.unwrap(), "prefill_bs must be <= num_rows");
    RuntimeCheck(
        cu_seqlens_ptr != nullptr || row_to_page_ptr != nullptr,
        "one of cu_seqlens_q / row_to_page must be provided");
    if (cu_seqlens_ptr != nullptr) {
      RuntimeCheck(BSp1.unwrap() == BS.unwrap() + 1, "invalid cu_seqlens_q shape");
    }
    static_assert(sizeof(FusedSmem) % 128 == 0);

    [[maybe_unused]] static const bool smem_ok = [] {
      const auto err = ::cudaFuncSetAttribute(
          topk_prefill_fused_kernel,
          ::cudaFuncAttributeMaxDynamicSharedMemorySize,
          sizeof(FusedSmem));
      RuntimeCheck(err == cudaSuccess, "cudaFuncSetAttribute failed");
      return true;
    }();

    const PrefillFusedParams params{
        .input = static_cast<const float*>(scores.data_ptr()),
        .row_starts = row_starts_ptr,
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .src_page_table = static_cast<const int32_t*>(src_page_table.data_ptr()),
        .cu_seqlens_q = cu_seqlens_ptr,
        .row_to_page = row_to_page_ptr,
        .dst = static_cast<int32_t*>(dst.data_ptr()),
        .stats = stats_ptr,
        .input_stride = S.unwrap(),
        .src_stride = P.unwrap(),
        .prefill_bs = static_cast<uint32_t>(BS.unwrap()),
        .topk = topk,
        .target = target,
        .cap = kCap,
    };
    const auto num_rows = static_cast<uint32_t>(B.unwrap());
    if (num_rows == 0) return;
    const auto device = device_.unwrap();
    LaunchKernel(num_rows, kBlockSize, device, sizeof(FusedSmem))
        (topk_prefill_fused_kernel, params);
  }
};

} // namespace sglang
