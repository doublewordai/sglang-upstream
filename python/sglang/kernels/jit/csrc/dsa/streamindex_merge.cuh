/**
 * \file streamindex_merge.cuh
 * \brief StreamIndex-style partition-merge top-k for the DSA prefill indexer.
 *
 * Key-axis chunked variant of lane/topk-1pass: the [q x L] fp32 logits tensor
 * never exists. The scorer (deep_gemm fp8_mqa_logits, or any producer) is run
 * per key chunk [c0, c1) producing a small tile [q, c1-c0]; an extract kernel
 * appends every in-window element >= the row's running threshold M into a
 * persistent per-row candidate buffer (TieValue {value, global kv pos}), and
 * compacts (exact top-2048 select, reusing topk-1pass's radix machinery) when
 * the buffer fills. A final kernel runs the exact top-2048 select over the
 * candidates and applies the production page-table transform.
 *
 * Exactness (top-2048, ties at the boundary arbitrary as in any select):
 *  - M starts at -FLT_MAX ("nothing dropped yet") and only rises at
 *    compaction, where it becomes min of the kept top-2048.
 *  - The extract pass appends ALL in-window x >= M (no sampling, no
 *    truncation: the window is streamed in kSeg-sized segments and compaction
 *    fires between segments whenever cursor >= kCompactAt, so
 *    cursor <= kCompactAt + kSeg = kCandCap always).
 *  - Induction: buffer always contains the top-min(2048, appended) of all
 *    appended elements, and every non-appended (dropped) element x has
 *    >= 2048 kept elements >= x (the drops happen only when > 2048 candidates
 *    compete, and M is the 2048-th largest of everything appended so far).
 *    Hence top-2048(row) \subseteq candidates at the end.
 *
 * Determinism: candidate sets and tie-break (value, idx) are data-determined;
 * the smem atomic append order never affects the selected set.
 *
 * Semantics mirror topk_prefill_1pass.cuh / the production kernel, including
 * the naive length <= topk path (which ignores row_start, as production does)
 * and the -1 padding. NaN logits are NOT supported (same as v2/topk-1pass).
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
using impl::warp_inclusive_sum;
using impl::warp_sum_bool;

namespace {

constexpr uint32_t kBlockSize = impl::kWarpSize * 32;  // 1024
constexpr uint32_t kNumWarps = kBlockSize / impl::kWarpSize;
constexpr uint32_t kMaxTopK = 2048;
constexpr uint32_t kCandCap = 8192;    // per-row candidate capacity (8 B each)
constexpr uint32_t kCompactAt = 6144;  // compact when cursor reaches this
constexpr uint32_t kSeg = 2048;        // streaming segment between compaction checks
static_assert(kCompactAt + kSeg == kCandCap, "cursor bound");
constexpr uint32_t kRadixSize = 256;
using vec_t = device::AlignedVector<float, 4>;

// ---------------------------------------------------------------------------
// Shared memory
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

struct MergeSmem {
  alignas(128) TieValue cand[kCandCap];  // 64 KB staging (compaction / final)
  int32_t out_idx[kMaxTopK];             // 8 KB select output
  TieHandleSmem tie;                     // ~2.2 KB
  alignas(128) uint32_t cursor;          // live cursor during streaming
  uint32_t n_appends;
  uint32_t n_compacts;
  alignas(128) float m_new;              // compaction output threshold
  float warp_min[kNumWarps];
};
static_assert(sizeof(MergeSmem) <= 96 * 1024, "smem budget");

// ---------------------------------------------------------------------------
// Exact select over the candidate buffer (local copy of v2's TopKConfig
// handle_tie / radix_tie_select, kMaxNumTie -> kCandCap; identical to
// lane/topk-1pass topk_prefill_1pass.cuh -- proven machinery, unchanged)
// ---------------------------------------------------------------------------

constexpr uint32_t kTieItems = kCandCap / kBlockSize;  // 8

SGL_DEVICE void tie_emit(int32_t* out, const uint32_t pos, const uint32_t raw_idx) {
  out[pos] = static_cast<int32_t>(raw_idx);
}

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
      if (above < topk_remain && above + hist_val >= topk_remain) {
        smem->match = {tx, above, hist_val, 0};
      }
    }
    __syncthreads();

    const auto [threshold_bin, above_count, equal_count, __] = smem->match;
    if (round < 3) total_active = equal_count;
    topk_remain -= above_count;

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
    }

    if (round == 3 || topk_remain == 0) break;
    __syncthreads();
  }

#pragma unroll
  for (uint32_t i = 0; i < kItems; ++i) {
    if (write_pos[i] < topk) tie_emit(out, base + write_pos[i], idx[i]);
  }
}

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
      tie_emit(out, base + t, base + t);  // pad (unreachable in this driver)
    }
  } else if (num_ties <= impl::kWarpSize) {
    if (lane_id >= num_ties || warp_id >= num_ties) return;
    const uint32_t mask = (1ull << num_ties) - 1u;
    const auto tie = tie_buffer[lane_id];
    const auto target = tie_buffer[warp_id];
    const auto rank = warp_sum_bool(is_greater(tie, target), mask);
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
      const auto rank = warp_sum_bool(is_greater(tie_0, target_0)) + warp_sum_bool(is_greater(tie_1, target_0));
      if (lane_id == 0 && rank < topk) tie_emit(out, base + rank, target_0.idx);
    }
    if (warp_id_1 < num_ties) {
      const auto rank = warp_sum_bool(is_greater(tie_0, target_1)) + warp_sum_bool(is_greater(tie_1, target_1));
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
// Streaming window iteration (from topk-1pass, unchanged)
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
// Extract + merge kernel: one block per row, one key chunk
// ---------------------------------------------------------------------------

constexpr uint32_t kRowsPerCTA = 8;  // amortize CTA fixed costs (launch, state)

struct ExtractParams {
  const float* __restrict__ tile;       // [q, tile_stride] (chunk-local logits)
  const int32_t* __restrict__ ks;       // [q] global window starts
  const int32_t* __restrict__ ke;       // [q] global window ends
  int64_t tile_stride;
  uint32_t num_rows;
  uint32_t c0;                          // global key range of this chunk
  uint32_t c1;
  TieValue* __restrict__ cand;          // [q, kCandCap]
  uint32_t* __restrict__ cursor;        // [q]
  float* __restrict__ thresh;           // [q]
  uint32_t* __restrict__ stats;         // [q, 3] {appends, compactions, cursor}
};

__global__ __launch_bounds__(kBlockSize, 2) void streamindex_extract_kernel(
    const __grid_constant__ ExtractParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto* s = reinterpret_cast<MergeSmem*>(smem_raw);
  const uint32_t tx = threadIdx.x;

  const uint32_t row0 = blockIdx.x * kRowsPerCTA;
  for (uint32_t rr = 0; rr < kRowsPerCTA; ++rr) {
  const uint32_t row = row0 + rr;
  if (row >= p.num_rows) break;

  const int32_t ks_g = p.ks[row];
  const int32_t ke_g = p.ke[row];
  // true window of this row inside this chunk (tile-local coords)
  uint32_t a, b;
  {
    const int32_t lo = ks_g > (int32_t)p.c0 ? ks_g : (int32_t)p.c0;
    const int32_t hi = ke_g < (int32_t)p.c1 ? ke_g : (int32_t)p.c1;
    a = lo > (int32_t)p.c0 ? (uint32_t)(lo - (int32_t)p.c0) : 0u;
    b = hi > (int32_t)p.c0 ? (uint32_t)(hi - (int32_t)p.c0) : 0u;
    b = b > a ? b : a;  // empty window -> a == b (nothing streamed)
  }
  if (b <= a) continue;  // state untouched (cursor/M from previous chunks stands)

  float M = p.thresh[row];
  if (tx == 0) {
    s->cursor = p.cursor[row];
    s->n_appends = 0;
    s->n_compacts = 0;
  }
  __syncthreads();

  const float* trow = p.tile + static_cast<int64_t>(row) * p.tile_stride;
  TieValue* crow = p.cand + static_cast<int64_t>(row) * kCandCap;
  const uint32_t kv_base = p.c0;  // global kv position of tile column 0

  for (uint32_t base = a; base < b; base += kSeg) {
    const uint32_t len = kSeg < (b - base) ? kSeg : (b - base);
    stream_window(trow + base, len, [&](float x, uint32_t j) {
      if (x >= M) {
        const auto pos = atomicAdd(&s->cursor, 1);
        // pos < kCandCap by the kCompactAt + kSeg == kCandCap invariant
        crow[pos] = TieValue{x, kv_base + base + j};
        atomicAdd(&s->n_appends, 1u);
      }
    });
    __syncthreads();
    if (s->cursor >= kCompactAt) {
      // ---- compaction: exact top-2048 of cand[0..cursor) ----
      // Stage a position-indexed copy (idx := array position) so the select
      // emits array positions; the ORIGINAL (value, kv pos) pairs stay in
      // gmem crow for the race-free gather below.
      const uint32_t n = s->cursor;  // >= kCompactAt > kMaxTopK
      for (uint32_t t = tx; t < n; t += kBlockSize)
        s->cand[t] = TieValue{crow[t].value, t};
      __syncthreads();
      handle_tie(s->cand, s->out_idx, 0, n, kMaxTopK, &s->tie);
      __syncthreads();
      float mn = FLT_MAX;
      for (uint32_t t = tx; t < kMaxTopK; t += kBlockSize) {
        const TieValue v = crow[static_cast<uint32_t>(s->out_idx[t])];
        s->cand[t] = v;
        mn = fminf(mn, v.value);
      }
      __syncthreads();
      for (uint32_t t = tx; t < kMaxTopK; t += kBlockSize) crow[t] = s->cand[t];
      // block min -> smem
      const auto lane_id = tx % impl::kWarpSize;
      const auto warp_id = tx / impl::kWarpSize;
#pragma unroll
      for (uint32_t off = 16; off > 0; off >>= 1)
        mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, off));
      if (lane_id == 0) s->warp_min[warp_id] = mn;
      __syncthreads();
      if (tx == 0) {
        float bm = s->warp_min[0];
#pragma unroll
        for (uint32_t w = 1; w < kNumWarps; ++w) bm = fminf(bm, s->warp_min[w]);
        s->m_new = bm;
        p.thresh[row] = bm;
        s->cursor = kMaxTopK;
        s->n_compacts += 1;
      }
      __syncthreads();
      M = s->m_new;
    }
  }
  __syncthreads();
  if (tx == 0) {
    p.cursor[row] = s->cursor;
    if (p.stats != nullptr) {
      // accumulate across chunks (caller zeroes once per run)
      p.stats[row * 3 + 0] += s->n_appends;
      p.stats[row * 3 + 1] += s->n_compacts;
      p.stats[row * 3 + 2] = s->cursor;
    }
  }
  __syncthreads();  // before the next row reuses smem state
  }
}

// ---------------------------------------------------------------------------
// Final kernel: exact top-2048 over candidates + page-table transform
// ---------------------------------------------------------------------------

struct FinalParams {
  const TieValue* __restrict__ cand;             // [q, kCandCap]
  const uint32_t* __restrict__ cursor;           // [q]
  const int32_t* __restrict__ ks;                // [q] global window starts
  const int32_t* __restrict__ ke;                // [q] global window ends
  const int32_t* __restrict__ src_page_table;    // [prefill_bs, src_stride]
  const int32_t* __restrict__ cu_seqlens_q;      // [prefill_bs + 1]
  int32_t* __restrict__ dst;                     // [q, topk]
  int64_t src_stride;
  uint32_t prefill_bs;
  uint32_t topk;
};

__global__ __launch_bounds__(kBlockSize, 2) void streamindex_final_kernel(
    const __grid_constant__ FinalParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto* s = reinterpret_cast<MergeSmem*>(smem_raw);
  const uint32_t row = blockIdx.x;
  const uint32_t tx = threadIdx.x;

  // Resolve the source page-table row (production semantics).
  __shared__ const int32_t* s_src;
  if (p.prefill_bs <= kBlockSize) {
    if (tx < p.prefill_bs) {
      if (row >= (uint32_t)p.cu_seqlens_q[tx] && row < (uint32_t)p.cu_seqlens_q[tx + 1]) {
        s_src = p.src_page_table + static_cast<int64_t>(tx) * p.src_stride;
      }
    }
  } else {
    for (uint32_t i = tx; i < p.prefill_bs; i += kBlockSize) {
      if (row >= (uint32_t)p.cu_seqlens_q[i] && row < (uint32_t)p.cu_seqlens_q[i + 1]) {
        s_src = p.src_page_table + static_cast<int64_t>(i) * p.src_stride;
      }
    }
  }
  __syncthreads();
  const int32_t* src_row = s_src;

  const int32_t row_start = p.ks[row];
  const int32_t length_i32 = p.ke[row] - row_start;
  int32_t* out = p.dst + static_cast<int64_t>(row) * p.topk;

  // Naive path (length <= topk), matching production bit-for-bit (including
  // its row_start-ignoring quirk): keep the first `length` page-table entries.
  if (length_i32 <= 0 || (uint32_t)length_i32 <= p.topk) {
    for (uint32_t i = tx; i < p.topk; i += kBlockSize) {
      out[i] = (int32_t)i < length_i32 ? src_row[i] : -1;
    }
    return;
  }
  const uint32_t topk = p.topk;
  const uint32_t n = p.cursor[row];  // >= topk for every non-naive row

  const TieValue* crow = p.cand + static_cast<int64_t>(row) * kCandCap;
  for (uint32_t t = tx; t < n; t += kBlockSize) s->cand[t] = crow[t];
  __syncthreads();
  handle_tie(s->cand, s->out_idx, 0, n, topk, &s->tie);
  __syncthreads();
  for (uint32_t t = tx; t < topk; t += kBlockSize) {
    // out_idx[t] holds the selected TieValue.idx = GLOBAL kv position
    // (handle_tie emits idx values, not buffer positions). Production
    // transforms with the page-1 table indexed by the WINDOW-LOCAL position
    // (dst = pt[pos - row_start]; verified against the production kernel on
    // row_starts != 0 shapes -- matches topk-1pass's mirror of production).
    const int32_t raw = s->out_idx[t];
    out[t] = raw >= row_start ? src_row[raw - row_start] : -1;
  }
}

}  // namespace

struct StreamIndexMergeKernel {
  static void extract(
      const tvm::ffi::TensorView tile,
      const tvm::ffi::TensorView ks,
      const tvm::ffi::TensorView ke,
      const tvm::ffi::TensorView cand,
      const tvm::ffi::TensorView cursor,
      const tvm::ffi::TensorView thresh,
      int64_t c0,
      int64_t c1,
      const tvm::ffi::Optional<tvm::ffi::TensorView> stats) {
    using namespace host;
    auto Q = SymbolicSize{"num_rows"};
    auto T = SymbolicSize{"tile_width"};
    auto TS = SymbolicSize{"tile_stride"};
    auto C = SymbolicSize{"cand_cap"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({Q, T})  // tile
        .with_strides({TS, 1})
        .with_dtype<float>()
        .with_device(device_)
        .verify(tile);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(ks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(ke);
    TensorMatcher({Q, C})  // cand: reinterpret as TieValue pairs
        .with_strides({C, 1})
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(cand);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(cursor);
    TensorMatcher({Q}).with_dtype<float>().with_device(device_).verify(thresh);

    uint32_t* stats_ptr = nullptr;
    if (stats.has_value()) {
      TensorMatcher({Q, 3}).with_dtype<int32_t>().with_device(device_).verify(stats.value());
      stats_ptr = static_cast<uint32_t*>(stats.value().data_ptr());
    }

    RuntimeCheck(C.unwrap() == kCandCap, "cand capacity must be 8192");
    RuntimeCheck(c1 > c0, "empty chunk range");
    RuntimeCheck(T.unwrap() == (uint64_t)(c1 - c0), "tile width must match chunk");

    static_assert(sizeof(MergeSmem) % 128 == 0);
    [[maybe_unused]] static const bool smem_ok = [] {
      const auto err = ::cudaFuncSetAttribute(
          streamindex_extract_kernel,
          ::cudaFuncAttributeMaxDynamicSharedMemorySize,
          sizeof(MergeSmem));
      RuntimeCheck(err == cudaSuccess, "cudaFuncSetAttribute failed");
      return true;
    }();

    const ExtractParams params{
        .tile = static_cast<const float*>(tile.data_ptr()),
        .ks = static_cast<const int32_t*>(ks.data_ptr()),
        .ke = static_cast<const int32_t*>(ke.data_ptr()),
        .tile_stride = TS.unwrap(),
        .num_rows = static_cast<uint32_t>(Q.unwrap()),
        .c0 = static_cast<uint32_t>(c0),
        .c1 = static_cast<uint32_t>(c1),
        .cand = reinterpret_cast<TieValue*>(cand.data_ptr()),
        .cursor = static_cast<uint32_t*>(cursor.data_ptr()),
        .thresh = static_cast<float*>(thresh.data_ptr()),
        .stats = stats_ptr,
    };
    const auto num_rows = static_cast<uint32_t>(Q.unwrap());
    if (num_rows == 0) return;
    const auto grid = (num_rows + kRowsPerCTA - 1) / kRowsPerCTA;
    LaunchKernel(grid, kBlockSize, device_.unwrap(), sizeof(MergeSmem))
        (streamindex_extract_kernel, params);
  }

  static void final(
      const tvm::ffi::TensorView cand,
      const tvm::ffi::TensorView cursor,
      const tvm::ffi::TensorView ks,
      const tvm::ffi::TensorView ke,
      const tvm::ffi::TensorView dst,
      const tvm::ffi::TensorView src_page_table,
      const tvm::ffi::TensorView cu_seqlens_q,
      int64_t topk) {
    using namespace host;
    auto Q = SymbolicSize{"num_rows"};
    auto C = SymbolicSize{"cand_cap"};
    auto K = SymbolicSize{"topk"};
    auto BS = SymbolicSize{"prefill_bs"};
    auto BSp1 = SymbolicSize{"prefill_bs_plus_1"};
    auto P = SymbolicSize{"page_table_stride"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({Q, C})
        .with_strides({C, 1})
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(cand);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(cursor);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(ks);
    TensorMatcher({Q}).with_dtype<int32_t>().with_device(device_).verify(ke);
    TensorMatcher({Q, K}).with_dtype<int32_t>().with_device(device_).verify(dst);
    TensorMatcher({BS, -1})
        .with_strides({P, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(src_page_table);
    TensorMatcher({BSp1}).with_dtype<int32_t>().with_device(device_).verify(cu_seqlens_q);

    RuntimeCheck(topk > 0 && topk <= (int64_t)kMaxTopK, "topk must be in (0, 2048]");
    RuntimeCheck(BS.unwrap() >= 1, "prefill_bs must be >= 1");
    RuntimeCheck(BS.unwrap() <= Q.unwrap(), "prefill_bs must be <= num_rows");
    RuntimeCheck(BSp1.unwrap() == BS.unwrap() + 1, "invalid cu_seqlens_q shape");
    RuntimeCheck(C.unwrap() == kCandCap, "cand capacity must be 8192");

    static_assert(sizeof(MergeSmem) % 128 == 0);
    [[maybe_unused]] static const bool smem_ok = [] {
      const auto err = ::cudaFuncSetAttribute(
          streamindex_final_kernel,
          ::cudaFuncAttributeMaxDynamicSharedMemorySize,
          sizeof(MergeSmem));
      RuntimeCheck(err == cudaSuccess, "cudaFuncSetAttribute failed");
      return true;
    }();

    const FinalParams params{
        .cand = reinterpret_cast<const TieValue*>(cand.data_ptr()),
        .cursor = static_cast<const uint32_t*>(cursor.data_ptr()),
        .ks = static_cast<const int32_t*>(ks.data_ptr()),
        .ke = static_cast<const int32_t*>(ke.data_ptr()),
        .src_page_table = static_cast<const int32_t*>(src_page_table.data_ptr()),
        .cu_seqlens_q = static_cast<const int32_t*>(cu_seqlens_q.data_ptr()),
        .dst = static_cast<int32_t*>(dst.data_ptr()),
        .src_stride = P.unwrap(),
        .prefill_bs = static_cast<uint32_t>(BS.unwrap()),
        .topk = static_cast<uint32_t>(topk),
    };
    const auto num_rows = static_cast<uint32_t>(Q.unwrap());
    if (num_rows == 0) return;
    LaunchKernel(num_rows, kBlockSize, device_.unwrap(), sizeof(MergeSmem))
        (streamindex_final_kernel, params);
  }
};

}  // namespace sglang
