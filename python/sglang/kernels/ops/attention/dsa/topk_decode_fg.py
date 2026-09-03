"""Full-grid decode-time top-k for DSA indexers (see jit/csrc/dsa/topk_decode_fg.cuh).

Drop-in replacement for ``sgl_kernel.fast_topk_v2`` on decode / spec-verify
shapes (few rows, long rows): same output semantics (raw positions in
``[0, length)``, ``-1`` padding for ``length <= topk`` rows, arbitrary order,
production's exact 4-round radix refinement over the coarse threshold bin),
but each row is processed by the full grid (two-phase histogram + gather,
row read exactly twice) instead of one 1024-thread block per row.

Warm start (``warmstart=True``): the caller-carried threshold side-channel
turns the 2-pass select into a single streaming pass + exact refine when the
selection is temporally stable. One fp32 per (request, layer) is carried
across steps: the previous step's k-th logit minus a ``delta``-sigma margin.
A miss (candidate count outside ``[topk, cap]``) falls back to the full
2-pass chain device-side; garbage seeds (NaN, +-inf, stale) only ever miss,
never produce a wrong answer. The thresholds buffer is persistent per
(``warm_key``, batch, device) so CUDA-graph replay evolves it across steps;
key it by the indexer layer id when several layers share a batch size.
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
        cuda_wrappers=[
            ("topk_decode_fg", f"TopKDecodeFG<{args}>::run"),
            ("topk_decode_fg_ws", f"TopKDecodeFG<{args}>::run_ws"),
        ],
    )


# Persistent per-(B, cap, device, warm) workspace: hist + hist2 + plan + counters
# (+ ws_flags + sums when warm) and the capped candidate lists. Zero-initialized
# once; both chains leave the per-row state zeroed for the next call of the
# SAME mode (the warm chain re-zeros in thr_out, the plain chain in K2), so
# warm and plain workspaces are cached separately and must not be shared.
_WORKSPACE_CACHE = {}

# Persistent per-(warm_key, B, device) threshold side-channel: +inf on creation
# (the first call misses and falls back), then evolved by the kernels. The
# caller (HiSparse coordinator / dsa_indexer layer) owns the (request, layer)
# mapping; warm_key must identify the layer when layers share a batch size.
_THRESHOLD_CACHE = {}

_DEFAULT_CAP = 65536  # 16x production's 4096-entry smem candidate cap
_DEFAULT_DELTA = 0.3  # seed margin in units of the row's logit sigma


def _get_workspace(batch: int, cap: int, device: torch.device, warm: bool) -> torch.Tensor:
    key = (batch, cap, str(device), warm)
    ws = _WORKSPACE_CACHE.get(key)
    if ws is None:
        # (2*256 hist + 4 plan + 4 plan2 + 4 counters + 3 warm slots) per row
        # (the plain mode only needs 12 of the 15 small slots; the extra 3
        # keep both layouts the same size) + 3 capped candidate lists.
        n_ints = batch * (2 * 256 + 4 + 4 + 4 + 3) + 3 * batch * cap
        ws = torch.zeros(n_ints, dtype=torch.int32, device=device)
        _WORKSPACE_CACHE[key] = ws
    return ws


def _get_thresholds(batch: int, warm_key: int, device: torch.device) -> torch.Tensor:
    key = (warm_key, batch, str(device))
    thr = _THRESHOLD_CACHE.get(key)
    if thr is None:
        thr = torch.full((batch,), float("inf"), dtype=torch.float32, device=device)
        _THRESHOLD_CACHE[key] = thr
    return thr


def topk_decode_fg(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    cap: int = _DEFAULT_CAP,
    return_stats: bool = False,
    warmstart: bool = False,
    delta: float = _DEFAULT_DELTA,
    warm_key: int = 0,
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
            ``{n_eq, eq_appended, r, inconsistent}`` (plain) or ``[B, 6]``
            with ``{..., warm_ok, n_total}`` appended (warmstart).
        warmstart: carry the previous call's k-th value minus a ``delta``-sigma
            margin as this call's threshold seed (one fp32 per row, persisted
            per ``warm_key``); exact 1-pass select on a hit, full 2-pass
            fallback on a miss.
        delta: seed margin in units of the row's logit sigma.
        warm_key: thresholds-cache key (use the indexer layer id so layers
            sharing a batch size keep independent side-channels).

    Returns:
        ``[B, topk]`` int32 positions (arbitrary order), plus stats if asked.
    """
    assert scores.dim() == 2 and scores.stride(1) == 1
    assert 0 < topk <= 2048
    batch = scores.shape[0]
    if batch == 0:
        out = torch.empty((0, topk), dtype=torch.int32, device=scores.device)
        ncol = 6 if warmstart else 4
        return (
            (out, torch.empty((0, ncol), dtype=torch.int32, device=scores.device))
            if return_stats
            else out
        )
    if lengths.dtype != torch.int32 or lengths.stride(0) != 1:
        lengths = lengths.to(dtype=torch.int32).contiguous()

    out = torch.empty((batch, topk), dtype=torch.int32, device=scores.device)
    module = _jit_topk_decode_fg_module()
    if warmstart:
        stats = (
            torch.zeros((batch, 6), dtype=torch.int32, device=scores.device)
            if return_stats
            else None
        )
        ws = _get_workspace(batch, cap, scores.device, warm=True)
        thr = _get_thresholds(batch, warm_key, scores.device)
        module.topk_decode_fg_ws(scores, lengths, out, ws, cap, stats, thr, delta)
    else:
        stats = (
            torch.zeros((batch, 4), dtype=torch.int32, device=scores.device)
            if return_stats
            else None
        )
        ws = _get_workspace(batch, cap, scores.device, warm=False)
        module.topk_decode_fg(scores, lengths, out, ws, cap, stats)
    return (out, stats) if return_stats else out
