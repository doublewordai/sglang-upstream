"""dgf_norm_quant — fused (add-)RMSNorm + per-token-group-128 fp8 quant (JIT).

Lane decode-glue-fusion. One kernel replaces [fused_add_rmsnorm; quant; (scale
layout)] at the decode-layer norm sites, and the kMoeIn variant replaces
[sh_out.add_(routed, alpha); next-layer fused_add_rmsnorm; quant] at the MoE
combine epilogue. Quant arithmetic and scale layouts replicate the production
per_token_group_quant exactly (bf16-domain amax, scale=amax/448, TMA-aligned
col-major scales + optional row-major); the norm reduction can differ from the
flashinfer CuTe kernel by 1 bf16 ulp on ~1e-5 of elements.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)


@cache_once
def _dgf_norm_quant_module(
    k: int, add: bool, moe_in: bool, dual: bool, use_pdl: bool
):
    args = make_cpp_args(k, add, moe_in, dual, use_pdl)
    return load_jit(
        "dgf_norm_quant",
        *args,
        cuda_files=["norm/dgf_norm_quant.cuh"],
        cuda_wrappers=[
            ("dgf_norm_quant", f"sglang::DGFNormQuantKernel<{args}>::run"),
        ],
        # match the production per_token_group_quant build: div.approx for
        # 448/amax (bit-identical fp8 codes given identical h)
        extra_cuda_cflags=["--use_fast_math"],
    )


def _alloc_scales(T: int, K: int, dual: bool, device):
    """Production scale layouts: col-major TMA-aligned [K/128, T_pad4] viewed
    as logical [T, K/128] (strides (1, T_pad4)); optional row-major [T, K/128]."""
    t_pad4 = (T + 3) // 4 * 4
    s_col = torch.empty(K // 128, t_pad4, dtype=torch.float32, device=device)
    s_col_view = s_col.transpose(0, 1)[:T, :]
    s_row = (
        torch.empty(T, K // 128, dtype=torch.float32, device=device) if dual else None
    )
    return s_col, s_col_view, s_row, t_pad4


def dgf_rmsnorm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
    shared: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    dual_scale: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Fused (add-)RMSNorm + group-128 fp8 quant.

    - residual given, shared None:  res' = res + x; h = rmsnorm(res')*w  (input_ln site)
    - residual + shared given:       res' = res + shared + alpha*x      (MoE epilogue)
    - neither:                       h = rmsnorm(x)*w                   (q_a site)

    Returns (h, res', q, s_col_view[, s_row if dual_scale]).
    """
    T, K = x.shape
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert K % 128 == 0 and K % (256 * 8) == 0, f"K={K} unsupported"
    moe_in = shared is not None
    add = residual is not None and not moe_in
    res = residual if residual is not None else x  # unused dummy when plain
    sh = shared if shared is not None else x  # unused dummy when not moe_in
    h = torch.empty_like(x)
    q = torch.empty(T, K, dtype=torch.float8_e4m3fn, device=x.device)
    s_col, s_col_view, s_row, t_pad4 = _alloc_scales(T, K, dual_scale, x.device)
    mod = _dgf_norm_quant_module(K, add, moe_in, dual_scale, is_arch_support_pdl())
    mod.dgf_norm_quant(
        h, res, x, sh, weight, q, s_col,
        s_row if s_row is not None else s_col,  # dummy when not dual
        float(alpha), T, t_pad4, float(eps),
    )
    out = (h, res, q, s_col_view)
    if dual_scale:
        out = out + (s_row,)
    return out
