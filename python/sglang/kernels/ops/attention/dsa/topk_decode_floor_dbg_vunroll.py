"""Byte-floor decode-time top-k for DSA indexers (see jit/csrc/dsa/topk_decode_floor_dbg_vunroll.cuh).

One persistent launch (two in-kernel grid barriers) that reads each logits row
ONCE (plus a 2048-element sample and, on rows where the sample mispredicts the
threshold, a rare fg-equivalent second read), replacing the 6-launch
``topk_decode_fg`` chain. Same output semantics (raw positions in ``[0,
length)``, ``-1`` padding for ``length <= topk`` rows, arbitrary order, fg's
tie rule), with an optional fused page-table transform (the arithmetic of
``transform_index_page_table_decode``).
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
def _jit_topk_decode_floor_dbg_vunroll_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "dsa_topk_decode_floor_dbg_vunroll_dbg",
        *args,
        cuda_files=["dsa/topk_decode_floor_dbg_vunroll.cuh"],
        cuda_wrappers=[("topk_decode_floor_dbg_vunroll", f"TopKDecodeFloor<{args}>::run")],
    )


# Persistent per-(B, cap, device) workspace, zero-initialized once. The kernel
# is self-cleaning: every consumed word (hist, counters, barrier state) is
# re-zeroed by its consumer within the same call/replay.
_WORKSPACE_CACHE = {}

_DEFAULT_CAP = 65536  # 16x production's 4096-entry smem candidate cap


def _get_workspace(batch: int, cap: int, device: torch.device) -> torch.Tensor:
    key = (batch, cap, str(device))
    ws = _WORKSPACE_CACHE.get(key)
    if ws is None:
        n_ints = 8 + batch * (256 + 8 + 8) + 4 * batch * cap + 2
        ws = torch.zeros(n_ints, dtype=torch.int32, device=device)
        _WORKSPACE_CACHE[key] = ws
    return ws


def topk_decode_floor_dbg_vunroll(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    cap: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    return_stats: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Select the top-``topk`` positions per row of ``scores`` (decode shape).

    Args:
        scores: ``[B, stride]`` fp32 (or bf16) with unit inner stride; only
            ``[0, lengths[row])`` is read.
        lengths: ``[B]`` int32 (non-negative; ``<= topk`` rows emit
            ``0..length-1`` then ``-1``).
        topk: ``0 < topk <= 2048``.
        cap: per-row candidate-list capacity for the threshold bin
            (fg/production equivalent: 4096; default min(65536, stride)).
        page_table: optional ``[B, pt_stride]`` int32 (page_size=1 table);
            when given, output positions are transformed through it
            (``out = page_table[row, pos]``, ``-1`` passthrough) in the same
            launch.
        return_stats: also return ``[B, 8]`` (debug: 4-6 = phase timings ns) int32
            ``{n_eq, n_eq_stored, r, flags}`` (flags: 1=fallback re-read,
            2=cap overflow, 4=inconsistent).

    Returns:
        ``[B, topk]`` int32 positions (arbitrary order), plus stats if asked.
    """
    assert scores.dim() == 2 and scores.stride(1) == 1
    assert 0 < topk <= 2048
    batch = scores.shape[0]
    if batch == 0:
        out = torch.empty((0, topk), dtype=torch.int32, device=scores.device)
        return (out, torch.empty((0, 8), dtype=torch.int32, device=scores.device)) if return_stats else out
    if cap is None:
        cap = min(_DEFAULT_CAP, max(topk + 1, scores.shape[1]))
    if lengths.dtype != torch.int32 or lengths.stride(0) != 1:
        lengths = lengths.to(dtype=torch.int32).contiguous()
    if page_table is not None:
        assert page_table.dtype == torch.int32 and page_table.dim() == 2
        assert page_table.shape[0] == batch and page_table.stride(1) == 1

    out = torch.empty((batch, topk), dtype=torch.int32, device=scores.device)
    stats = (
        torch.zeros((batch, 8), dtype=torch.int32, device=scores.device)
        if return_stats
        else None
    )
    ws = _get_workspace(batch, cap, scores.device)
    module = _jit_topk_decode_floor_dbg_vunroll_module()
    module.topk_decode_floor_dbg_vunroll(scores, lengths, out, page_table, ws, cap, stats)
    return (out, stats) if return_stats else out
