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

#pragma once

#include <cutlass/bfloat16.h>
#include <cstdint>
#include <cuda_runtime.h>

struct SparseMlaFp8DecodeParams {
  int num_reqs;      // b (flattened query rows; MTP verify = bs * n_draft)
  int num_splits;    // P: partitions per request
  int topk;          // index list length per request (multiple of 64)
  int num_heads;     // 64
  int d_v;           // 512
  float sm_scale_log2e;

  const uint8_t* __restrict__ q;         // [num_reqs, num_heads, 576] bf16
  const uint8_t* __restrict__ kv;        // [rows, 656] raw row bytes (fp8 | scales | rope)
  const int* __restrict__ indices;       // [num_reqs, topk]  (negative = masked)
  const int* __restrict__ seqlens;       // [num_reqs] valid index count (scheduler hint)
  float* __restrict__ partial_o;         // [num_reqs, P, num_heads, d_v]
  float* __restrict__ partial_ml;        // [num_reqs, P, num_heads, 2] (m, l) in exp2 units
  int tail_sentinel;                     // 1: rows beyond seqlens are all -1 (skip blocks)
  float* __restrict__ debug;             // optional [num_reqs, 12] (nullable): E pre-pass dump
  // q pre-quantization outputs (qprep kernel writes; main kernel may read them
  // instead of params.q when use_qprep is set)
  uint8_t* __restrict__ q_fp8_out;       // [num_reqs, 64, 512]
  cutlass::bfloat16_t* __restrict__ q_rope_out;  // [num_reqs, 64, 64]
  float* __restrict__ q_scale_out;       // [num_reqs, 64]
  int use_qprep;                         // 1: main kernel loads pre-quantized q
  int fused;                             // 1: in-kernel combine behind an atomic barrier
  int* __restrict__ counter;             // [1] i32 fused barrier counter (qprep zeroes it)
  cutlass::bfloat16_t* __restrict__ out_fused;  // [num_reqs, 64, 512] fused-mode output

  cudaStream_t stream;
};

struct SparseMlaFp8CombineParams {
  int num_reqs, num_splits, num_heads, d_v;
  const float* __restrict__ partial_o;
  const float* __restrict__ partial_ml;
  cutlass::bfloat16_t* __restrict__ out;  // [num_reqs, num_heads, d_v]
  cudaStream_t stream;
};
