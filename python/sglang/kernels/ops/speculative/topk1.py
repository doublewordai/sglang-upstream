from __future__ import annotations

import torch
import triton
import triton.language as tl

_DRAFT_TOPK1_BLOCK = 8192


@triton.jit
def _draft_topk1_partial_argmax_kernel(
    logits,
    partial_vals,
    partial_indices,
    partial_sumexp,
    logits_row_stride,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    COMPUTE_SUMEXP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # int64 row base: row * stride overflows int32 once bs * vocab reaches 2^31.
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    vals = tl.load(
        logits + row * logits_row_stride + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    # Keep NaNs on valid lanes from selecting the masked tail.
    vals = tl.where(vals == vals, vals, -1e30)

    max_val = tl.max(vals, axis=0)
    local_index = tl.argmax(vals, axis=0)
    out_offset = row * num_splits + split
    tl.store(partial_vals + out_offset, max_val)
    tl.store(partial_indices + out_offset, split * BLOCK + local_index)
    if COMPUTE_SUMEXP:
        # sum of exp(val - local_max): masked lanes contribute exp(-inf)=0;
        # the finalize kernel rescales by exp(local_max - global_max).
        sumexp = tl.sum(tl.exp(vals - max_val), axis=0)
        tl.store(partial_sumexp + out_offset, sumexp)


@triton.jit
def _draft_topk1_finalize_kernel(
    partial_vals,
    partial_indices,
    partial_sumexp,
    topk_p,
    topk_index,
    positions,
    draft_tokens,
    draft_probs,
    draft_tokens_stride,
    draft_probs_stride,
    draft_token_column,
    num_splits: tl.constexpr,
    WRITE_DRAFT_TOKEN: tl.constexpr,
    WRITE_PROBS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )

    split = tl.argmax(vals, axis=0)
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    tl.store(topk_index + row, index)
    tl.store(topk_p + row, 1.0)
    if WRITE_DRAFT_TOKEN:
        tl.store(draft_tokens + row * draft_tokens_stride + draft_token_column, index)
    if WRITE_PROBS:
        # P(argmax token) = exp(max - logsumexp) = 1 / sum(exp(v - max)).
        sumexp = tl.sum(
            tl.load(
                partial_sumexp + row * num_splits + offsets,
                mask=mask,
                other=0.0,
            )
            * tl.exp(vals - tl.max(vals, axis=0)),
            axis=0,
        )
        tl.store(
            draft_probs + row * draft_probs_stride + draft_token_column,
            1.0 / sumexp,
        )

    position = tl.load(positions + row)
    tl.store(positions + row, position + 1)


def draft_topk1_postprocess(
    next_token_logits: torch.Tensor,
    positions: torch.Tensor,
    draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0,
    draft_probs: torch.Tensor | None = None,
):
    """Argmax draft logits for topk=1 and advance positions.

    PyTorch eager argmax reduces each row with too little parallelism for the
    GLM/DSV4 vocab widths in CUDA graph replay. This split reduction exposes
    the vocab dimension across CTAs, then finalizes one token per row.

    If ``draft_tokens`` is given, the finalize kernel also stores the argmax
    into ``draft_tokens[:, draft_token_column]``, mutating the caller-owned
    buffer in place. ``topk_p`` is returned as constant 1.0: topk=1 drafting
    is greedy and the chain probabilities are unused downstream.

    If ``draft_probs`` is given ([rows, num_steps] float32, same row stride
    contract as ``draft_tokens``), the finalize kernel also stores the argmax
    probability P(argmax token) into
    ``draft_probs[:, draft_token_column]`` -- the per-step confidence source
    for adaptive verify scheduling (lane/adaptive-spec).
    """
    assert next_token_logits.ndim == 2
    assert next_token_logits.stride(1) == 1
    assert positions.ndim == 1
    assert positions.is_contiguous()
    assert positions.shape[0] == next_token_logits.shape[0]
    assert positions.device == next_token_logits.device
    write_draft_token = draft_tokens is not None
    if write_draft_token:
        assert draft_tokens.ndim == 2
        assert draft_tokens.dtype == torch.long
        assert draft_tokens.device == next_token_logits.device
        assert draft_tokens.shape[0] == next_token_logits.shape[0]
        assert draft_tokens.stride(1) == 1
        assert 0 <= draft_token_column < draft_tokens.shape[1]
    write_probs = draft_probs is not None
    if write_probs:
        assert draft_probs.ndim == 2
        assert draft_probs.dtype == torch.float32
        assert draft_probs.device == next_token_logits.device
        assert draft_probs.shape[0] == next_token_logits.shape[0]
        assert draft_probs.stride(1) == 1
        assert 0 <= draft_token_column < draft_probs.shape[1]

    bs, vocab_size = next_token_logits.shape
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=next_token_logits.device)
    topk_index = torch.empty(
        (bs, 1), dtype=torch.int64, device=next_token_logits.device
    )
    if bs == 0:
        return topk_p, topk_index

    block = _DRAFT_TOPK1_BLOCK
    num_splits = triton.cdiv(vocab_size, block)
    partial_vals = torch.empty(
        (bs, num_splits), dtype=torch.float32, device=next_token_logits.device
    )
    partial_indices = torch.empty(
        (bs, num_splits), dtype=torch.int32, device=next_token_logits.device
    )
    # Dummy operand for the disabled sumexp slot: the pointer must be valid
    # even though the kernel never dereferences it (gated off by
    # COMPUTE_SUMEXP / WRITE_PROBS).
    partial_sumexp = (
        torch.empty(
            (bs, num_splits), dtype=torch.float32, device=next_token_logits.device
        )
        if write_probs
        else partial_vals
    )

    _draft_topk1_partial_argmax_kernel[(bs, num_splits)](
        next_token_logits,
        partial_vals,
        partial_indices,
        partial_sumexp,
        next_token_logits.stride(0),
        vocab_size,
        num_splits,
        COMPUTE_SUMEXP=write_probs,
        BLOCK=block,
        num_warps=8,
    )
    # Dummy operands for the disabled slots: the pointers must be valid even
    # though the kernel never dereferences them (gated off by the constexprs).
    _draft_topk1_finalize_kernel[(bs,)](
        partial_vals,
        partial_indices,
        partial_sumexp,
        topk_p,
        topk_index,
        positions,
        draft_tokens if write_draft_token else topk_index,
        draft_probs if write_probs else topk_p,
        draft_tokens.stride(0) if write_draft_token else 0,
        draft_probs.stride(0) if write_probs else 0,
        draft_token_column,
        num_splits,
        WRITE_DRAFT_TOKEN=write_draft_token,
        WRITE_PROBS=write_probs,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return topk_p, topk_index
