"""Single-pass fused top-k + page-table transform for the DSA prefill indexer.

Drop-in replacement for ``sgl_kernel.fast_topk_transform_fused`` in the
prefill-shaped PAGED case (the call sites that would launch
``topk_transform_prefill_kernel``). Reads each logits row once (plus a 0.78%
sample) instead of twice; see ``jit/csrc/dsa/topk_prefill_1pass.cuh``.

lane/pagetable-gather: ``row_to_page`` (SGLANG_DSA_PAGETABLE_HOIST) lets the
caller pass the per-step page table whole plus a per-row table-row map
instead of a pre-materialized ``page_table_1[batch_idx]`` [rows, L] copy —
identical outputs, no gather.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _jit_topk_prefill_1pass_module():
    return load_jit(
        "dsa_topk_prefill_1pass_v1",
        cuda_files=["dsa/topk_prefill_1pass.cuh"],
        cuda_wrappers=[
            ("topk_transform_prefill_1pass", "TopKPrefill1PassKernel::transform"),
        ],
    )


def fast_topk_transform_prefill_1pass(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table_size_1: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
    out_stats: Optional[torch.Tensor] = None,
    row_to_page: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Same contract as ``sgl_kernel.fast_topk_transform_fused``.

    Args:
        score: [B, L] fp32 logits, unit inner stride, row stride % 4 == 0.
        lengths: [B] int32 per-row window length (positions
            [row_starts[i], row_starts[i] + lengths[i]) are selectable; the
            rest are never read). May be <= 0 (row outputs all -1).
        page_table_size_1: [prefill_bs, >= L] int32 page table (page size 1).
        cu_seqlens_q: [prefill_bs + 1] int32 cumulative q lengths mapping each
            row to its page-table row. May be None when row_to_page is given.
        topk: selection size, 0 < topk <= 2048.
        row_starts: optional [B] int32 window start per row.
        out_stats: optional [B, 2] int32 diagnostics, filled with
            {candidate_count, used_fallback} per row.
        row_to_page: optional [B] int32 page-table row per score row
            (SGLANG_DSA_PAGETABLE_HOIST). When given, the kernel reads
            page_table_size_1[row_to_page[i]] directly and cu_seqlens_q is
            ignored — the caller must NOT have materialized a [B, L] copy.
    Returns:
        [B, topk] int32: dst[i, t] = page_table_size_1[seq(i)][pos_t], -1 pad.
    """
    assert topk <= 2048, "prefill 1-pass top-k supports topk <= 2048"
    assert score.dim() == 2
    dst = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    module = _jit_topk_prefill_1pass_module()
    module.topk_transform_prefill_1pass(
        score, lengths, dst, page_table_size_1, cu_seqlens_q, row_starts, out_stats, row_to_page
    )
    return dst
