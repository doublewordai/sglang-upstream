/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// JIT dispatch entry for the SM90 native-fp8 sparse MLA decode kernel.

#pragma once

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "kernel.cuh"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace sglang {

static inline void _sdk_set_device_and_stream(
    int device_id, int64_t cuda_stream, cudaStream_t* out) {
  cudaSetDevice(device_id);
  *out = reinterpret_cast<cudaStream_t>(cuda_stream);
}

// Main split-KV kernel: q bf16 [b, 64, 576], kv raw rows uint8 [rows, 656],
// indices int32 [b, topk] (negative = masked), seqlens int32 [b].
void sparse_mla_fp8_decode_dispatch(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView kv,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView seqlens,
    tvm::ffi::TensorView partial_o,
    tvm::ffi::TensorView partial_ml,
    int64_t num_splits,
    int64_t topk,
    int64_t tail_sentinel,
    double sm_scale,
    int64_t cuda_stream) {
  SparseMlaFp8DecodeParams params;
  _sdk_set_device_and_stream(q.device().device_id, cuda_stream, &params.stream);
  params.num_reqs = (int)q.shape()[0];
  params.num_heads = (int)q.shape()[1];
  params.num_splits = (int)num_splits;
  params.topk = (int)topk;
  params.d_v = 512;
  params.sm_scale_log2e = (float)sm_scale * (float)M_LOG2E;
  params.q = reinterpret_cast<const uint8_t*>(q.data_ptr());
  params.kv = reinterpret_cast<const uint8_t*>(kv.data_ptr());
  params.indices = static_cast<const int*>(indices.data_ptr());
  params.seqlens = static_cast<const int*>(seqlens.data_ptr());
  params.partial_o = static_cast<float*>(partial_o.data_ptr());
  params.partial_ml = static_cast<float*>(partial_ml.data_ptr());
  params.tail_sentinel = (int)tail_sentinel;
  sm90::decode::SparseMlaFp8DecodeKernel::run(params);
}

// Combine: partial (m, l, O) -> out bf16 [b, 64, 512].
void sparse_mla_fp8_decode_combine(
    tvm::ffi::TensorView partial_o,
    tvm::ffi::TensorView partial_ml,
    tvm::ffi::TensorView out,
    int64_t num_splits,
    int64_t cuda_stream) {
  SparseMlaFp8CombineParams params;
  _sdk_set_device_and_stream(out.device().device_id, cuda_stream, &params.stream);
  params.num_reqs = (int)(out.shape()[0]);
  params.num_heads = (int)(out.shape()[1]);
  params.d_v = (int)(out.shape()[2]);
  params.num_splits = (int)num_splits;
  params.partial_o = static_cast<float*>(partial_o.data_ptr());
  params.partial_ml = static_cast<float*>(partial_ml.data_ptr());
  params.out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());
  sm90::decode::run_combine(params);
}

}  // namespace sglang
