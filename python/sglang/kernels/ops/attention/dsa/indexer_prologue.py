"""Lane indexer-prologue: DSA indexer prologue fused kernels with the Hadamard
rotation KEPT INSIDE (production arithmetic preserved).

Q: rope -> Hadamard -> fp8 quant (pow2 "ue8m0" scale) + head-gate fold, one
   kernel per the whole q tail (input = wq_b output, output = q_fp8 + weights).
K: LayerNorm -> rope -> Hadamard -> fp8 quant + paged index-k store, one
   kernel (input = wk slice of the merged wk_weights_proj GEMM output).

CUDA only (JIT). See jit/csrc/dsa/indexer_prologue.cuh for the arithmetic
contract and the bit-exactness notes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_q_indexer_prologue_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "indexer_prologue_q",
        *args,
        cuda_files=["dsa/indexer_prologue.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedQIndexerPrologueKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_k_indexer_prologue_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "indexer_prologue_k",
        *args,
        cuda_files=["dsa/indexer_prologue.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedKIndexerPrologueKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_k_indexer_prologue_store_module(dtype: torch.dtype, page_size: int) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl(), page_size)
    return load_jit(
        f"indexer_prologue_k_store_p{page_size}",
        *args,
        cuda_files=["dsa/indexer_prologue.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedKIndexerPrologueStoreKernel<{args}>::forward"),
        ],
    )


def fused_q_indexer_prologue(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    head_gate_scale: float,
    softmax_scale: float,
    hadamard_scale: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Q prologue: RoPE(leading 64, interleaved) + 128-pt Hadamard + fp8 quant
    (pow2 scale) + head-gate fold. Returns (q_fp8 [B,H,128], weights [B,H,1]
    fp32) with weights[b,h] = ((w[b,h]*head_gate_scale) * q_scale) * softmax_scale.

    q_input: (B, H, 128) bf16 contiguous (wq_b output).
    weight:  (B, H) bf16, row stride may differ (merged-GEMM slice).
    """
    q_fp8 = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    module = _jit_q_indexer_prologue_module(q_input.dtype)
    module.forward(
        q_input,
        q_fp8,
        weight,
        weights_out,
        float(head_gate_scale),
        float(softmax_scale),
        float(hadamard_scale),
        cos_sin_cache,
        positions,
    )
    return q_fp8, weights_out


def fused_k_indexer_prologue(
    k_input: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: float,
    hadamard_scale: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """K prologue without cache store: LayerNorm + RoPE + Hadamard -> bf16.

    k_input: (B, 128) bf16, row stride may differ (wk slice of the merged GEMM).
    """
    k_out = torch.empty(k_input.shape, dtype=k_input.dtype, device=k_input.device)
    module = _jit_k_indexer_prologue_module(k_input.dtype)
    module.forward(
        k_input,
        k_out,
        ln_weight,
        ln_bias,
        cos_sin_cache,
        positions,
        float(eps),
        float(hadamard_scale),
    )
    return k_out


def fused_k_indexer_prologue_store(
    k_input: torch.Tensor,
    cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: float,
    hadamard_scale: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    page_size: int,
) -> None:
    """K prologue + fused store: LayerNorm + RoPE + Hadamard + fp8 quant
    (fp32 scale max(1e-4, amax)/448) + paged index-k write, one launch.

    cache: (num_pages, 132*page_size) uint8. out_cache_loc: (B,) int64.
    """
    if not out_cache_loc.is_contiguous():
        out_cache_loc = out_cache_loc.contiguous()
    module = _jit_k_indexer_prologue_store_module(k_input.dtype, page_size)
    module.forward(
        k_input,
        cache,
        out_cache_loc,
        ln_weight,
        ln_bias,
        cos_sin_cache,
        positions,
        float(eps),
        float(hadamard_scale),
    )
