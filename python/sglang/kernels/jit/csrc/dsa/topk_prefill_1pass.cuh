/**
 * \file topk_prefill_1pass.cuh
 * \brief Single-pass top-k + page-table transform for the DSA prefill indexer.
 *
 * Replaces `topk_transform_prefill_kernel` (sgl_kernel aot, 2-pass radix select
 * that reads the logits row twice) for the prefill-shaped PAGED case:
 * one 1024-thread block per expanded query row, top-k (<= 2048) of
 * score[row, row_start : row_start + length], output
 * dst[row, t] = src_page_table[seq(row)][pos_t] (page_size = 1 table),
 * -1 padding. Semantics mirror the production kernel (see
 * aot/csrc/elementwise/topk.cu), including the naive <= topk path.
 *
 * Algorithm (1 full read of the row + a 0.78% sample):
 *  1. SAMPLE: 64 chunks x 128 elements spread over the window -> fp16-coarse
 *     12-bit histogram -> threshold bin whose suffix count crosses
 *     j = ceil(kTargetCount * m / length) -> T = bin lower bound.
 *     E[#elements >= T] ~= kTargetCount (5120), CAP = 8192.
 *  2. COLLECT (the one full pass): x >= T -> atomicAdd(count), store {x, idx}
 *     while pos < CAP. `count` is the EXACT number of elements >= T.
 *  3. SELECT: if topk <= count <= CAP the buffer holds every element >= T, so
 *     the top-k is an exact radix select (full fp32 key) over the buffer.
 *     Otherwise (T mis-estimated, ~1e-3/row on real data; degenerate rows) fall
 *     back to a 2-pass coarse-radix select over the row (v2 TopKStreaming
 *     style): exact except when > CAP distinct fp32 values share one fp16 bin.
 *  4. TRANSFORM: dst[t] = src_pt[pos_t] (or -1).
 *
 * Tie handling: exact fp32 ties at the selection boundary are resolved
 * arbitrarily (insert order is atomic); any tie-consistent set is produced.
 * In-window +/-inf are handled (the -inf fp16 bin lower bound is -inf).
 * NaN logits are NOT supported (same as the DeepSeek-V4 v2 kernel).
 *
 * Reuses helpers from <sgl_kernel/deepseek_v4/topk_impl.cuh>; the tie
 * machinery is a local copy of TopKConfig::handle_tie / radix_tie_select
 * resized to kMaxNumTie = kCap = 8192.
 */
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek_v4/topk_impl.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>

namespace sglang {

namespace impl = device::topk;
using impl::TieValue;
using impl::extract_exact_bin;
using impl::extract_coarse_bin;
using impl::coarse_bin_lower_bound;
using impl::warp_inclusive_sum;
using impl::warp_sum_bool;

namespace {

constexpr uint32_t kBlockSize = impl::kWarpSize * 32;  // 1024
constexpr uint32_t kNumWarps = kBlockSize / impl::kWarpSize;
constexpr uint32_t kMaxTopK = 2048;
constexpr uint32_t kCap = 8192;      // candidate buffer capacity
constexpr uint32_t kHistBits = 12;   // fp16-coarse bins (matches v2 streaming)
constexpr uint32_t kHistSize = 1 << kHistBits;
constexpr uint32_t kSampleChunks = 64;
constexpr uint32_t kSampleChunkElems = 128;
constexpr uint32_t kSampleCount = kSampleChunks * kSampleChunkElems;  // 8192
constexpr uint32_t kTargetCount = 5120;  // E[#cand] target: P(count < topk) and
                                         // P(count > kCap) both ~ 3e-4 at L=1M
constexpr uint32_t kRadixSize = 256;
using vec_t = device::AlignedVector<float, 4>;

// ---------------------------------------------------------------------------
// Shared memory layout (~90.3 KB dynamic)
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
  uint32_t histogram[2][kRadixSize];
};

struct PrefillSmem {
  union {
    uint32_t sample_hist[kHistSize];  // sample phase: 16 KB
    struct {                          // collect phase: 64 KB
      uint32_t count;
      TieValue cand[kCap];
    } collect;
    struct {  // fallback pass 2: 64 KB (overlaps collect)
      uint32_t count_gt;
      uint32_t count_eq;
      TieValue tie[kCap];
    } fb;
  } u;
  uint32_t coarse_hist[kHistSize];  // fallback pass 1: 16 KB
  int32_t out_indices[kMaxTopK];    // select output: 8 KB
  TieHandleSmem tie;                // ~2.2 KB
  alignas(128) uint32_t threshold_bin;
};
static_assert(sizeof(PrefillSmem) <= 96 * 1024, "smem budget");

// ---------------------------------------------------------------------------
// Selection over the candidate buffer (local copy of v2's TopKConfig
// handle_tie / radix_tie_select, kMaxNumTie -> kCap, emit -> int32 raw out)
// ---------------------------------------------------------------------------

constexpr uint32_t kTieItems = kCap / kBlockSize;  // 8

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
    key[i] = extract_exact_bin(tie.value);
    idx[i] = tie.idx;
    write_pos[i] = topk;
  }
  uint32_t topk_remain = topk;
  if (tx < kRadixSize) smem->histogram[0][tx] = 0;
  if (tx == kRadixSize) smem->counter = smem->counter_final = 0;
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
    if (round < 3 && tx < kRadixSize) {
      smem->histogram[hist_idx ^ 1][tx] = 0;
    }
    __syncthreads();

    uint32_t hist_val = 0;
    uint32_t warp_inc = 0;
    if (tx < kRadixSize) {
      hist_val = histogram[tx];
      warp_inc = warp_inclusive_sum(lane_id, hist_val);
      if (lane_id == impl::kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
    }
    __syncthreads();
    if (tx < kRadixSize) {
      const auto inter = device::warp::reduce_sum(lane_id < warp_id ? smem->warp_sum[lane_id] : 0);
      const auto prefix = inter + warp_inc;      // inclusive prefix through this bin
      const auto above = total_active - prefix;  // elements in bins ABOVE this one
      // 3. Find threshold bin
      if (above < topk_remain && above + hist_val >= topk_remain) {
        smem->match = {tx, above, hist_val, 0};
      }
    }
    __syncthreads();

    const auto [threshold_bin, above_count, equal_count, __] = smem->match;
    if (round < 3) total_active = equal_count;
    topk_remain -= above_count;

    // 4. Scatter
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
      tie_emit(out, base + t, base + t);  // pad (only reachable in fallback)
    }
  } else if (num_ties <= impl::kWarpSize) {
    if (lane_id >= num_ties || warp_id >= num_ties) return;  // some threads idle
    /// NOTE: use long long to avoid mask overflow when num_tie == 32
    const uint32_t mask = (1ull << num_ties) - 1u;
    const auto tie = tie_buffer[lane_id];
    const auto target = tie_buffer[warp_id];
    const auto rank = warp_sum_bool(is_greater(tie, target), mask);
    if (lane_id == 0 && rank < topk) tie_emit(out, base + rank, target.idx);
  } else if (num_ties <= impl::kWarpSize * 2) {
    // 64 x 64 topk implementation: each thread takes 2 elements
    const auto warp_id_0 = warp_id;
    const auto warp_id_1 = warp_id + impl::kWarpSize;
    const auto lane_id_1 = lane_id + impl::kWarpSize;
    const auto invalid = TieValue::invalid();
    const auto tie_0 = tie_buffer[lane_id];
    const auto tie_1 = lane_id_1 < num_ties ? tie_buffer[lane_id_1] : invalid;
    const auto target_0 = tie_buffer[warp_id_0];
    const auto target_1 = tie_buffer[warp_id_1];
    {
      const auto rank = warp_sum_bool(is_greater(tie_0, target_0)) + warp_sum_bool(is_greater(tie_1, target_0));
      if (lane_id == 0 && rank < topk) tie_emit(out, base + rank, target_0.idx);
    }
    if (warp_id_1 < num_ties) {
      const auto rank = warp_sum_bool(is_greater(tie_0, target_1)) + warp_sum_bool(is_greater(tie_1, target_1));
      if (lane_id == 0 && rank < topk) tie_emit(out, base + rank, target_1.idx);
    }
  } else if (num_ties <= impl::kWarpSize * 4) {
    // 128 x 128 topk implementation: local sort + merge
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
        rank += warp_sum_bool(is_greater(tie[j], target[i]));
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
// Histogram threshold search (v2 TopKRadixBase::find_threshold, inlined for a
// plain uint32_t[kHistSize] array). Finds the bin with
// above < k <= above + count (suffix sums from the top).
// ---------------------------------------------------------------------------

SGL_DEVICE void find_threshold_bin(
    const uint32_t* hist,  // [kHistSize], sums to total
    const uint32_t total,  // histogram total (sample size or row length)
    const uint32_t k,      // crossing count
    uint32_t* warp_sum_scratch,
    uint32_t* out_bin) {
  constexpr uint32_t kItems = kHistSize / kBlockSize;  // 4
  const auto tx = threadIdx.x;
  const auto lane_id = tx % impl::kWarpSize;
  const auto warp_id = tx / impl::kWarpSize;

  uint32_t orig[kItems];
  uint32_t local_sum = 0;
#pragma unroll
  for (uint32_t i = 0; i < kItems; ++i) {
    orig[i] = hist[tx * kItems + i];  // thread tx owns consecutive bins
    local_sum += orig[i];
  }
  const auto warp_inc = warp_inclusive_sum(lane_id, local_sum);
  const auto warp_exc = warp_inc - local_sum;
  if (lane_id == impl::kWarpSize - 1) warp_sum_scratch[warp_id] = warp_inc;
  __syncthreads();

  const auto tmp = warp_sum_scratch[lane_id];
  uint32_t prefix_sum = device::warp::reduce_sum(lane_id < warp_id ? tmp : 0);
  prefix_sum += warp_exc;
#pragma unroll
  for (uint32_t i = 0; i < kItems; ++i) {
    prefix_sum += orig[i];
    const auto above = total - prefix_sum;
    if (above < k && above + orig[i] >= k) {
      *out_bin = tx * kItems + i;
    }
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// Streaming window iteration: fn(value, window_idx) for each element of
// [row, row + length), vectorized float4 over the 16B-aligned body with scalar
// head/tail, double-buffered vector loads.
// ---------------------------------------------------------------------------

template <typename F>
SGL_DEVICE void stream_window(const float* __restrict__ row, const uint32_t length, F&& fn) {
  const auto tx = threadIdx.x;
  const uintptr_t base_addr = reinterpret_cast<uintptr_t>(row);
  static_assert(sizeof(float) == 4);
  const uint32_t misalign = (16u - (base_addr & 15u)) & 15u;  // bytes
  uint32_t head = misalign >> 2;                              // elements (4B aligned)
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

struct PrefillTopKParams {
  const float* __restrict__ input;             // [B, input_stride]
  const int32_t* __restrict__ row_starts;      // [B] or nullptr
  const int32_t* __restrict__ lengths;         // [B]
  const int32_t* __restrict__ src_page_table;  // [prefill_bs, src_stride]
  const int32_t* __restrict__ cu_seqlens_q;    // [prefill_bs + 1]
  int32_t* __restrict__ dst;                   // [B, topk]
  int32_t* __restrict__ stats;                 // [B, 2] or nullptr {count, fallback}
  int64_t input_stride;
  int64_t src_stride;
  uint32_t prefill_bs;
  uint32_t topk;
};

__global__ __launch_bounds__(kBlockSize, 2) void topk_transform_prefill_1pass_kernel(
    const __grid_constant__ PrefillTopKParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto* s = reinterpret_cast<PrefillSmem*>(smem_raw);
  const uint32_t bid = blockIdx.x;
  const uint32_t tx = threadIdx.x;

  // Resolve the source page-table row for this block (production semantics:
  // the unique seq s with cu_seqlens_q[s] <= bid < cu_seqlens_q[s+1]).
  __shared__ const int32_t* s_src;
  if (p.prefill_bs <= kBlockSize) {
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

  // Naive path (length <= topk), matching production bit-for-bit (including
  // its row_start-ignoring quirk): keep the first `length` page-table entries.
  if (length_i32 <= 0 || (uint32_t)length_i32 <= p.topk) {
    for (uint32_t i = tx; i < p.topk; i += kBlockSize) {
      out[i] = (int32_t)i < length_i32 ? src_row[i] : -1;
    }
    return;
  }
  const uint32_t length = static_cast<uint32_t>(length_i32);
  const uint32_t topk = p.topk;

  // ---- Phase 1: sample -> threshold T ------------------------------------
  float T;
  if (length <= kSampleCount) {
    T = -FLT_MAX;  // collect everything; count = length <= kCap -> exact
  } else {
    constexpr uint32_t kItems = kHistSize / kBlockSize;
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) s->u.sample_hist[tx + i * kBlockSize] = 0;
    if (tx == 0) s->threshold_bin = 0;
    __syncthreads();
    // 64 chunks x 128 elements, chunk c at window offset floor(c*length/64);
    // consecutive threads hit consecutive elements (coalesced scalar loads).
    for (uint32_t t = tx; t < kSampleCount; t += kBlockSize) {
      const uint32_t c = t / kSampleChunkElems;
      const uint32_t e = t % kSampleChunkElems;
      const uint32_t off = (c * length) / kSampleChunks + e;
      atomicAdd(&s->u.sample_hist[extract_coarse_bin<kHistBits>(row[off])], 1);
    }
    __syncthreads();
    // j = ceil(kTargetCount * m / length): E[count(>= T)] ~= kTargetCount.
    const uint32_t j = (kTargetCount * kSampleCount + length - 1) / length;
    find_threshold_bin(s->u.sample_hist, kSampleCount, j < 1 ? 1 : j, s->tie.warp_sum, &s->threshold_bin);
    T = coarse_bin_lower_bound<kHistBits>(s->threshold_bin);
  }

  // ---- Phase 2: the single collect pass ----------------------------------
  if (tx == 0) s->u.collect.count = 0;
  __syncthreads();
  stream_window(row, length, [&](float x, uint32_t idx) {
    if (x >= T) {
      const auto pos = atomicAdd(&s->u.collect.count, 1);
      if (pos < kCap) s->u.collect.cand[pos] = TieValue{x, idx};
    }
  });
  __syncthreads();
  const uint32_t count = s->u.collect.count;

  if (count >= topk && count <= kCap) {
    // ---- Phase 3a: exact select over the full candidate set --------------
    handle_tie(s->u.collect.cand, s->out_indices, 0, count, topk, &s->tie);
    if (p.stats != nullptr && tx == 0) {
      p.stats[bid * 2 + 0] = static_cast<int32_t>(count);
      p.stats[bid * 2 + 1] = 0;
    }
  } else {
    // ---- Phase 3b: fallback 2-pass coarse radix select over the row ------
    constexpr uint32_t kItems = kHistSize / kBlockSize;
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) s->coarse_hist[tx + i * kBlockSize] = 0;
    if (tx == 0) s->threshold_bin = 0;
    __syncthreads();
    stream_window(row, length, [&](float x, uint32_t) {
      atomicAdd(&s->coarse_hist[extract_coarse_bin<kHistBits>(x)], 1);
    });
    __syncthreads();
    find_threshold_bin(s->coarse_hist, length, topk, s->tie.warp_sum, &s->threshold_bin);
    const uint32_t thr = s->threshold_bin;
    const float v_hi = coarse_bin_lower_bound<kHistBits>(thr + 1);
    const float v_lo = coarse_bin_lower_bound<kHistBits>(thr);
    if (tx == 0) {
      s->u.fb.count_gt = 0;
      s->u.fb.count_eq = 0;
    }
    __syncthreads();
    stream_window(row, length, [&](float x, uint32_t idx) {
      if (x >= v_hi) {
        const auto pos = atomicAdd(&s->u.fb.count_gt, 1);
        if (pos < topk) s->out_indices[pos] = static_cast<int32_t>(idx);
      } else if (x >= v_lo) {
        const auto pos = atomicAdd(&s->u.fb.count_eq, 1);
        if (pos < kCap) s->u.fb.tie[pos] = TieValue{x, idx};
      }
    });
    __syncthreads();
    const uint32_t above = s->u.fb.count_gt;
    const uint32_t eq = min(s->u.fb.count_eq, kCap);
    const uint32_t remain = above < topk ? topk - above : 0;
    handle_tie(s->u.fb.tie, s->out_indices, above, eq, remain, &s->tie);
    if (p.stats != nullptr && tx == 0) {
      p.stats[bid * 2 + 0] = static_cast<int32_t>(count);
      p.stats[bid * 2 + 1] = 1;
    }
  }
  __syncthreads();

  // ---- Phase 4: page-table transform (page_size = 1) ---------------------
  for (uint32_t t = tx; t < topk; t += kBlockSize) {
    const int32_t raw = s->out_indices[t];
    out[t] = raw < 0 ? -1 : src_row[raw];
  }
}

}  // namespace

struct TopKPrefill1PassKernel {
  static void transform(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView dst,
      const tvm::ffi::TensorView src_page_table,
      const tvm::ffi::TensorView cu_seqlens_q,
      const tvm::ffi::Optional<tvm::ffi::TensorView> row_starts,
      const tvm::ffi::Optional<tvm::ffi::TensorView> stats) {
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

    TensorMatcher({B, L})  // scores
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device(device_)
        .verify(scores);
    TensorMatcher({B})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(lengths);
    TensorMatcher({B, K})  // dst
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(dst);
    TensorMatcher({BS, -1})  // src_page_table (page_size = 1)
        .with_strides({P, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(src_page_table);
    TensorMatcher({BSp1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(cu_seqlens_q);

    const int32_t* row_starts_ptr = nullptr;
    if (row_starts.has_value()) {
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(row_starts.value());
      row_starts_ptr = static_cast<const int32_t*>(row_starts.value().data_ptr());
    }
    int32_t* stats_ptr = nullptr;
    if (stats.has_value()) {
      TensorMatcher({B, 2}).with_dtype<int32_t>().with_device(device_).verify(stats.value());
      stats_ptr = static_cast<int32_t*>(stats.value().data_ptr());
    }

    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");
    RuntimeCheck(BS.unwrap() >= 1, "prefill_bs must be >= 1");
    RuntimeCheck(BS.unwrap() <= B.unwrap(), "prefill_bs must be <= num_rows");
    RuntimeCheck(BSp1.unwrap() == BS.unwrap() + 1, "invalid cu_seqlens_q shape");
    static_assert(sizeof(PrefillSmem) % 128 == 0);

    // Opt into > 48 KB dynamic shared memory once per process.
    [[maybe_unused]] static const bool smem_ok = [] {
      const auto err = ::cudaFuncSetAttribute(
          topk_transform_prefill_1pass_kernel,
          ::cudaFuncAttributeMaxDynamicSharedMemorySize,
          sizeof(PrefillSmem));
      RuntimeCheck(err == cudaSuccess, "cudaFuncSetAttribute failed");
      return true;
    }();

    const PrefillTopKParams params{
        .input = static_cast<const float*>(scores.data_ptr()),
        .row_starts = row_starts_ptr,
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .src_page_table = static_cast<const int32_t*>(src_page_table.data_ptr()),
        .cu_seqlens_q = static_cast<const int32_t*>(cu_seqlens_q.data_ptr()),
        .dst = static_cast<int32_t*>(dst.data_ptr()),
        .stats = stats_ptr,
        .input_stride = S.unwrap(),
        .src_stride = P.unwrap(),
        .prefill_bs = static_cast<uint32_t>(BS.unwrap()),
        .topk = topk,
    };
    const auto num_rows = static_cast<uint32_t>(B.unwrap());
    if (num_rows == 0) return;
    const auto device = device_.unwrap();
    LaunchKernel(num_rows, kBlockSize, device, sizeof(PrefillSmem))
        (topk_transform_prefill_1pass_kernel, params);
  }
};

}  // namespace sglang
