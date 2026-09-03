"""SM90 fp8 blockwise-scaled GEMM (CUTLASS ex-67 recipe), JIT-built.

Numerics recipe identical to the production DeepGEMM path: fp8 e4m3 x fp8 e4m3,
fp32 accumulation, per-token-group-128 activation scales, 128x128 weight block
scales, bf16 output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _fp8_blockwise_sm90_cuda_flags() -> list[str]:
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
def _jit_fp8_blockwise_sm90_module(variant: int) -> Module:
    args = make_cpp_args(variant)
    return load_jit(
        "fp8_blockwise_scaled_mm_sm90",
        *args,
        cuda_files=["gemm/fp8_blockwise/fp8_blockwise_scaled_mm_sm90.cuh"],
        cuda_wrappers=[
            ("fp8_blockwise_sm90", f"Fp8BlockwiseSm90Kernel<{args}>::run"),
        ],
        extra_dependencies=["cutlass"],
        extra_cuda_cflags=_fp8_blockwise_sm90_cuda_flags(),
    )


def fp8_blockwise_scaled_mm_sm90(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    variant: int = 0,
) -> torch.Tensor:
    """SM90 fp8 blockwise GEMM.

    Args:
        mat_a: [M, K] fp8 e4m3 row-major activation, M >= 1.
        scales_a: [M, K // 128] fp32, column-major (stride (1, pitch)) where
            pitch >= M is 4-aligned — the production per-token-group quant
            output buffer as-is (pitch = pad4(M)).
        mat_b: [N, K] fp8 e4m3 row-major weight (as stored).
        scales_b: [K // 128, ceil(N // 128)] fp32 row-major —
            weight_scale_inv.t().contiguous().
        variant: tile/schedule/orientation (0..7).
    Returns:
        [M, N] bf16.
    """
    out = torch.empty(
        (mat_a.shape[0], mat_b.shape[0]), dtype=torch.bfloat16, device=mat_a.device
    )
    module = _jit_fp8_blockwise_sm90_module(variant)
    module.fp8_blockwise_sm90(out, mat_a, mat_b, scales_a, scales_b)
    return out
