"""Warp-ballot one-read exact top-k for DSA indexers (see jit/csrc/dsa/topk_ballot.cuh).

Two entries:
  - ``topk_ballot`` (decode): same contract as ``topk_decode_fg`` — raw
    positions, arbitrary order, ``-1`` padding for ``length <= topk`` rows.
  - ``topk_transform_prefill_ballot`` (prefill): same contract as
    ``fast_topk_transform_prefill_1pass`` / ``sgl_kernel.fast_topk_transform_fused``
    — window-local selection + page-table (page_size=1) transform.

One full read of each logits row (plus an ~0.8% sample) instead of the fg
chain's two; per-element work is a compare + warp ballot (no per-element smem
atomics, no histogram in the streaming pass); the select reuses the fg
machinery over the captured candidate list. Miss rows (sample mispredicted the
capture count) fall back device-side to the fg 2-pass chain — exact, fg's cap
class.
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
def _jit_topk_ballot_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "dsa_topk_ballot",
        *args,
        cuda_files=["dsa/topk_ballot.cuh"],
        cuda_wrappers=[
            ("topk_ballot", f"TopKBallot<{args}>::run"),
            ("topk_ballot_prefill", f"TopKBallot<{args}>::run_prefill"),
        ],
    )


# Persistent per-(B, cap, device) workspace, zero-initialized once. The kernels
# are self-cleaning (see the .cuh header): every consumed word is re-zeroed by
# its consumer within the same call/replay.
_WORKSPACE_CACHE = {}

_DEFAULT_CAP_DECODE = 65536  # 16x production's 4096-entry smem candidate cap


def _prefill_cap(batch: int) -> int:
    # production prefill calls the select per scorer row-chunk (q=512); the
    # brief's q=8192 shape is the whole-layer tile. Keep the workspace under
    # ~1 GB at any shape.
    return 16384 if batch <= 2048 else 8192


def _get_workspace(batch: int, cap: int, device: torch.device) -> torch.Tensor:
    key = (batch, cap, str(device))
    ws = _WORKSPACE_CACHE.get(key)
    if ws is None:
        # (v_s 1 + plan 4 + plan2 4 + counters 4 + fast 1 + seq 1 + 2*256 hist)
        # per row + 3 capped candidate lists
        n_ints = batch * (15 + 2 * 256) + 3 * batch * cap
        ws = torch.zeros(n_ints, dtype=torch.int32, device=device)
        _WORKSPACE_CACHE[key] = ws
    return ws


def _target(topk: int, cap: int) -> int:
    # expected capture count for the sample: 3x topk keeps P(miss) ~ 0 under
    # sample-quantile noise (sd ~= target*sqrt(1/(p*n_sample))), while staying
    # well under cap. Misses fall back exactly, so this only gates speed.
    return max(1, min(3 * topk, cap // 2))


def topk_ballot(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    cap: Optional[int] = None,
    return_stats: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Select the top-``topk`` positions per row of ``scores`` (decode shape).

    Args:
        scores: ``[B, stride]`` fp32 (or bf16) with unit inner stride; only
            ``[0, lengths[row])`` is read.
        lengths: ``[B]`` int32 (non-negative; ``<= topk`` rows emit
            ``0..length-1`` then ``-1``).
        topk: ``0 < topk <= 2048``.
        cap: per-row candidate-list capacity (default 65536; fg's class at
            16x production's cap).
        return_stats: also return ``[B, 4]`` int32
            ``{n_captured, n_stored, r, flags}`` (flags: 1 = fallback,
            2 = capture overflow, 4 = capture underflow).

    Returns:
        ``[B, topk]`` int32 positions (arbitrary order), plus stats if asked.
    """
    assert scores.dim() == 2 and scores.stride(1) == 1
    assert 0 < topk <= 2048
    batch = scores.shape[0]
    if batch == 0:
        out = torch.empty((0, topk), dtype=torch.int32, device=scores.device)
        return (out, torch.empty((0, 4), dtype=torch.int32, device=scores.device)) if return_stats else out
    if cap is None:
        cap = _DEFAULT_CAP_DECODE
    if lengths.dtype != torch.int32 or lengths.stride(0) != 1:
        lengths = lengths.to(dtype=torch.int32).contiguous()

    out = torch.empty((batch, topk), dtype=torch.int32, device=scores.device)
    stats = (
        torch.zeros((batch, 4), dtype=torch.int32, device=scores.device)
        if return_stats
        else None
    )
    ws = _get_workspace(batch, cap, scores.device)
    module = _jit_topk_ballot_module()
    module.topk_ballot(scores, lengths, out, ws, cap, _target(topk, cap), stats)
    return (out, stats) if return_stats else out


def topk_transform_prefill_ballot(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table_size_1: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
    out_stats: Optional[torch.Tensor] = None,
    cap: Optional[int] = None,
) -> torch.Tensor:
    """Same contract as ``sgl_kernel.fast_topk_transform_fused`` (prefill).

    Args:
        score: [B, L] fp32 logits, unit inner stride.
        lengths: [B] int32 per-row window length (positions
            [row_starts[i], row_starts[i] + lengths[i]) are selectable; the
            rest are never read). May be <= 0 (row outputs all -1).
        page_table_size_1: [prefill_bs, >= L] int32 page table (page size 1).
        cu_seqlens_q: [prefill_bs + 1] int32 cumulative q lengths mapping each
            row to its page-table row.
        topk: selection size, 0 < topk <= 2048.
        row_starts: optional [B] int32 window start per row.
        out_stats: optional [B, 4] int32 diagnostics
            {n_captured, n_stored, r, flags}.
        cap: per-row candidate capacity (default 16384 for B <= 2048, else
            8192).

    Returns:
        [B, topk] int32: dst[i, t] = page_table_size_1[seq(i)][pos_t], -1 pad.
    """
    assert topk <= 2048
    assert score.dim() == 2 and score.stride(1) == 1
    assert score.dtype == torch.float32
    batch = score.shape[0]
    if batch == 0:
        return score.new_empty((0, topk), dtype=torch.int32)
    if cap is None:
        cap = _prefill_cap(batch)
    if lengths.dtype != torch.int32 or lengths.stride(0) != 1:
        lengths = lengths.to(dtype=torch.int32).contiguous()
    if row_starts is not None and (row_starts.dtype != torch.int32 or row_starts.stride(0) != 1):
        row_starts = row_starts.to(dtype=torch.int32).contiguous()
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_q.stride(0) != 1:
        cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32).contiguous()
    assert page_table_size_1.dtype == torch.int32 and page_table_size_1.stride(1) == 1

    dst = score.new_empty((batch, topk), dtype=torch.int32)
    ws = _get_workspace(batch, cap, score.device)
    module = _jit_topk_ballot_module()
    module.topk_ballot_prefill(
        score, lengths, dst, ws, cap, _target(topk, cap), row_starts, out_stats,
        page_table_size_1, cu_seqlens_q,
    )
    return dst
