"""Full-grid decode-time top-k for DSA indexers (see jit/csrc/dsa/topk_decode_fg.cuh).

Drop-in replacement for ``sgl_kernel.fast_topk_v2`` on decode / spec-verify
shapes (few rows, long rows): same output semantics (raw positions in
``[0, length)``, ``-1`` padding for ``length <= topk`` rows, arbitrary order,
production's exact 4-round radix refinement over the coarse threshold bin),
but each row is processed by the full grid (two-phase histogram + gather,
row read exactly twice) instead of one 1024-thread block per row.
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
def _jit_topk_decode_fg_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "dsa_topk_decode_fg",
        *args,
        cuda_files=["dsa/topk_decode_fg.cuh"],
        cuda_wrappers=[("topk_decode_fg", f"TopKDecodeFG<{args}>::run")],
    )


# Persistent per-(B, cap, device) workspace: hist + hist2 + plan + counters
# (small) and the two capped candidate lists. Zero-initialized once; the
# kernels leave hist/counters zeroed for the next call (K2 re-zeros hist
# after reading it and zeroes hist2/counters before K3), so the buffers are
# self-cleaning across calls and CUDA-graph replays.
_WORKSPACE_CACHE = {}

_DEFAULT_CAP = 65536  # 16x production's 4096-entry smem candidate cap


def _get_workspace(batch: int, cap: int, device: torch.device) -> torch.Tensor:
    key = (batch, cap, str(device))
    ws = _WORKSPACE_CACHE.get(key)
    if ws is None:
        n_ints = batch * (2 * 256 + 4 + 4 + 4) + 3 * batch * cap
        ws = torch.zeros(n_ints, dtype=torch.int32, device=device)
        _WORKSPACE_CACHE[key] = ws
    return ws


def topk_decode_fg(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    cap: int = _DEFAULT_CAP,
    return_stats: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Select the top-``topk`` positions per row of ``scores`` (decode shape).

    Args:
        scores: ``[B, stride]`` fp32 (or bf16) with unit inner stride; only
            ``[0, lengths[row])`` is read.
        lengths: ``[B]`` int32 (non-negative; ``<= topk`` rows emit
            ``0..length-1`` then ``-1``).
        topk: ``0 < topk <= 2048``.
        cap: per-row candidate-list capacity for the coarse threshold bin
            (production equivalent: 4096).
        return_stats: also return ``[B, 4]`` int32
            ``{n_eq, eq_appended, r, inconsistent}``.

    Returns:
        ``[B, topk]`` int32 positions (arbitrary order), plus stats if asked.
    """
    assert scores.dim() == 2 and scores.stride(1) == 1
    assert 0 < topk <= 2048
    batch = scores.shape[0]
    if batch == 0:
        out = torch.empty((0, topk), dtype=torch.int32, device=scores.device)
        return (out, torch.empty((0, 4), dtype=torch.int32, device=scores.device)) if return_stats else out
    if lengths.dtype != torch.int32 or lengths.stride(0) != 1:
        lengths = lengths.to(dtype=torch.int32).contiguous()

    out = scores.new_full((batch, topk), -1, dtype=torch.int32)
    stats = (
        torch.zeros((batch, 4), dtype=torch.int32, device=scores.device)
        if return_stats
        else None
    )
    ws = _get_workspace(batch, cap, scores.device)
    module = _jit_topk_decode_fg_module()
    module.topk_decode_fg(scores, lengths, out, ws, cap, stats)
    return (out, stats) if return_stats else out
