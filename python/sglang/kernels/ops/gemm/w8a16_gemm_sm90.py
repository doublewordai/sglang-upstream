"""SM90 W8A16 mixed-dtype GEMM (fp8 e4m3 weights x bf16 activations,
in-kernel dequant with per-N x K-group-128 scales), JIT-built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _w8a16_sm90_cuda_flags() -> list[str]:
    return [
        "-DNDEBUG",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-DCUTLASS_TEST_LEVEL=0",
        "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]


@cache_once
def _jit_w8a16_sm90_module(variant: int) -> Module:
    args = make_cpp_args(variant)
    return load_jit(
        "w8a16_gemm_sm90",
        *args,
        cuda_files=["gemm/w8a16_gemm_sm90.cuh"],
        cuda_wrappers=[
            ("w8a16_gemm_sm90", f"W8A16GemmSm90Kernel<{args}>::run"),
        ],
        extra_dependencies=["cutlass"],
        extra_cuda_cflags=_w8a16_sm90_cuda_flags(),
    )


def expand_block_scales(weight_scale_inv: torch.Tensor, n: int) -> torch.Tensor:
    """[N/128, K/128] block scales -> [K/128, N] row-major per-N expanded."""
    return weight_scale_inv.repeat_interleave(128, dim=0)[:n].t().contiguous()


def w8a16_gemm_sm90(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_b: torch.Tensor,
    variant: int = 0,
) -> torch.Tensor:
    """W8A16 GEMM.

    Args:
        mat_a: [M, K] bf16 row-major activation (M <= 32).
        mat_b: [N, K] fp8 e4m3 row-major weight (as stored).
        scales_b: [K // 128, N] fp32 row-major per-N expanded block scales
            (see expand_block_scales).
        variant: tile/schedule (0..3).
    Returns:
        [M, N] bf16.
    """
    out = torch.empty(
        (mat_a.shape[0], mat_b.shape[0]), dtype=torch.bfloat16, device=mat_a.device
    )
    module = _jit_w8a16_sm90_module(variant)
    module.w8a16_gemm_sm90(out, mat_a, mat_b, scales_b)
    return out
