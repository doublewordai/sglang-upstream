#pragma once

// D2H (device -> host-pinned) copy kernels using warp-coalesced SM stores.
//
// Why: on GH200 the copy-engine D2H path over C2C is capped at ~170 GB/s no
// matter the alignment (measured 169.6 GB/s, lanes d2h-stores/gather/
// lpddr-budget), while SM stores whose warp instructions are coalesced
// sustain 381-384 GB/s into the same pinned pool (2.25x). Per-thread
// contiguous stores (thread owns [i*S, i*S+S)) collapse to ~53 GB/s for
// S >= 64 B and partial-line stores (16 B per 128 B line) trigger
// read-modify-write line fills (7.1x penalty) — so BOTH kernels here keep
// 16 B vectors consecutive across the warp for every store instruction.
//
// Two entry points:
//  * D2HSegStoreKernel::run  — pre-tiled contiguous segments (host builds
//    <=4 MB tiles so grid parallelism is independent of segment count).
//  * D2HRowsStoreKernel::run — indexed all-layer gather-scatter (the AOT
//    remainder path): flat grid-stride over (token, layer, vec) with vec
//    fastest; item bytes and layer count are template parameters so the
//    div/mod strength-reduce to multiplies.
//
// Byte-identical to the copy-engine / AOT paths over the same inputs.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace sglang {
namespace d2h_store {

constexpr uint32_t kBlockSize = 256;

__global__ __launch_bounds__(kBlockSize) void seg_store_kernel(
    const int64_t* __restrict__ src_addrs,
    const int64_t* __restrict__ dst_addrs,
    const int64_t* __restrict__ tile_bytes,
    const uint32_t n_tiles) {
  const uint32_t t = blockIdx.x;
  if (t >= n_tiles) return;
  const uint4* __restrict__ src = reinterpret_cast<const uint4*>(src_addrs[t]);
  uint4* __restrict__ dst = reinterpret_cast<uint4*>(dst_addrs[t]);
  const int64_t n_vec = tile_bytes[t] >> 4;
  for (int64_t v = threadIdx.x; v < n_vec; v += kBlockSize) {
    dst[v] = src[v];
  }
}

template <typename T, int64_t kItemBytes, uint32_t kNumLayers>
__global__ __launch_bounds__(kBlockSize) void rows_store_kernel(
    const uint64_t* __restrict__ src_ptr_table,
    const uint64_t* __restrict__ dst_ptr_table,
    const T* __restrict__ src_indices,
    const T* __restrict__ dst_indices,
    const int64_t total_vecs) {
  constexpr int64_t kVecsPerRow = kItemBytes / 16;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kBlockSize;
  for (int64_t v = static_cast<int64_t>(blockIdx.x) * kBlockSize + threadIdx.x; v < total_vecs;
       v += stride) {
    // v = (token * kNumLayers + layer) * kVecsPerRow + vec_in_row
    const int64_t row = v / kVecsPerRow;
    const int64_t j = v - row * kVecsPerRow;
    const int64_t layer = row % kNumLayers;
    const int64_t token = row / kNumLayers;
    const int64_t src_row = static_cast<int64_t>(src_indices[token]);
    const int64_t dst_row = static_cast<int64_t>(dst_indices[token]);
    reinterpret_cast<uint4* __restrict__>(dst_ptr_table[layer])[dst_row * kVecsPerRow + j] =
        reinterpret_cast<const uint4* __restrict__>(src_ptr_table[layer])[src_row * kVecsPerRow + j];
  }
}

struct D2HSegStoreKernel {
  static void run(
      const tvm::ffi::TensorView src_addrs,
      const tvm::ffi::TensorView dst_addrs,
      const tvm::ffi::TensorView tile_bytes) {
    using namespace host;

    auto N = SymbolicSize{"num tiles"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({N})  //
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device_)
        .verify(src_addrs)
        .verify(dst_addrs)
        .verify(tile_bytes);

    const auto n = static_cast<uint32_t>(N.unwrap());
    if (n == 0) return;
    const auto device = device_.unwrap();
    LaunchKernel(n, kBlockSize, device)(
        seg_store_kernel,
        static_cast<const int64_t*>(src_addrs.data_ptr()),
        static_cast<const int64_t*>(dst_addrs.data_ptr()),
        static_cast<const int64_t*>(tile_bytes.data_ptr()),
        n);
  }
};

template <int64_t kItemBytes, uint32_t kNumLayers>
struct D2HRowsStoreKernel {
  template <typename T>
  static void run_impl(
      const tvm::ffi::TensorView src_ptr_table,
      const tvm::ffi::TensorView dst_ptr_table,
      const tvm::ffi::TensorView src_indices,
      const tvm::ffi::TensorView dst_indices) {
    using namespace host;

    auto L = SymbolicSize{"num layers (>= k)"};
    auto N = SymbolicSize{"num tokens"};
    auto indices_dtype = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({L})  //
        .with_dtype<uint64_t>()
        .with_device<kDLGPU>(device_)
        .verify(src_ptr_table)
        .verify(dst_ptr_table);
    TensorMatcher({N})  //
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLGPU>(device_)
        .verify(src_indices)
        .verify(dst_indices);

    // ptr tables may be longer than kNumLayers (packed MTP draft pools); the
    // kernels only touch layers [0, kNumLayers), like the AOT launcher.
    RuntimeCheck(L.unwrap() >= kNumLayers, "D2H rows store: layer count mismatch");
    RuntimeCheck(kItemBytes % 16 == 0, "D2H rows store: item bytes must be 16B aligned");

    const auto n = N.unwrap();
    if (n == 0) return;
    const int64_t total_vecs = n * static_cast<int64_t>(kNumLayers) * (kItemBytes / 16);
    const auto blocks = std::min(div_ceil(total_vecs, static_cast<int64_t>(kBlockSize)), int64_t{4096});
    const auto device = device_.unwrap();
    const auto kernel = rows_store_kernel<T, kItemBytes, kNumLayers>;
    LaunchKernel(static_cast<uint32_t>(blocks), kBlockSize, device)(
        kernel,
        static_cast<const uint64_t*>(src_ptr_table.data_ptr()),
        static_cast<const uint64_t*>(dst_ptr_table.data_ptr()),
        static_cast<const T*>(src_indices.data_ptr()),
        static_cast<const T*>(dst_indices.data_ptr()),
        total_vecs);
  }

  static void run(
      const tvm::ffi::TensorView src_ptr_table,
      const tvm::ffi::TensorView dst_ptr_table,
      const tvm::ffi::TensorView src_indices,
      const tvm::ffi::TensorView dst_indices) {
    if (src_indices.dtype().bits == 32) {
      run_impl<int32_t>(src_ptr_table, dst_ptr_table, src_indices, dst_indices);
    } else {
      run_impl<int64_t>(src_ptr_table, dst_ptr_table, src_indices, dst_indices);
    }
  }
};

}  // namespace d2h_store
}  // namespace sglang
