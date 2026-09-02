"""Decode-shaped DSA indexer logits kernel (Triton).

Replaces ``deep_gemm.fp8_paged_mqa_logits`` at decode / target-verify shapes
(few query rows, long contexts). Same semantics:

    logits[i, p] = scale_kv(p) * sum_h w[i, h] * relu(dot_fp8(q[i, h, :], k[p, :]))

with fp32 accumulation, per-token fp32 ``scale_kv`` stored in the page tail of
the index-K pool, relu applied to the per-head fp8 dot BEFORE the per-head
weight, and positions ``p >= context_len(row)`` left unwritten
(``clean_logits=False`` semantics: the select only reads ``[0, len)``).

Pool layout (matches ``index_buf_accessor.SetKAndS`` and DeepGEMM's reader;
the ``[page, 64, 1, 132]`` view used by callers is a shape lie -- only the
8448-byte page stride is meaningful): per 64-token page, the first 8192 bytes
are the k values (token-major, 128 contiguous fp8 bytes per token), followed
by 64 fp32 scales.

Difference vs the DeepGEMM kernel: all query rows of a request are resident
and every index-K row is read exactly ONCE per request, no matter how many
query rows (draft tokens) the request has. The DeepGEMM SM90 paged kernel
processes query rows in atoms of 1 (decode) or 2 (verify) rows; the sglang
SM90 path always uses the split form (one request per query row), i.e. at
target-verify each draft token re-reads the request's whole index-K.

Kernel structure: grid (token tiles, B). Each program streams one TILE_T-token
tile of a request's index-K (k tile [T, 128] fp8 loaded once, kept as the
tl.dot A operand), then loops over the request's query rows in groups of RG:
``tl.dot(k, trans(q_group))`` -> [T, RG*32] fp32, relu, per-head weights,
head reduction over the last axis, per-token scale, store [T, RG] transposed.

CUDA-graph safe: no host syncs, no atomics, fixed grid per shape, output
allocated per call (captured into the graph pool, like DeepGEMM's internal
allocation).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_mqa_logits_kernel(
    q_ptr,  # fp8e4m3 [R*H, D] contiguous; row (r*next_n + i)*H + h
    w_ptr,  # fp32    [R*H]   contiguous
    k_ptr,  # fp8e4m3 view of pool base (byte == element addressing)
    s_ptr,  # fp32    view of pool base (element = 4 bytes)
    bt_ptr,  # int32  [B, bt_stride] page table
    ctx_ptr,  # int32 [B*next_n] per-query-row context lengths
    out_ptr,  # fp32  [R, out_stride]
    next_n,  # query rows per request
    bt_stride,
    out_stride,
    H: tl.constexpr,
    D: tl.constexpr,
    PS: tl.constexpr,  # page size (tokens)
    KB: tl.constexpr,  # k bytes per token (D, fp8)
    PAGEB: tl.constexpr,  # pool bytes per page (PS * (KB + 4))
    TILE_T: tl.constexpr,  # tokens per program (multiple of PS)
    RG: tl.constexpr,  # query rows per inner iteration
    N_ITER: tl.constexpr,  # ceil(next_n / RG)
):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)

    # per-request max context length (last row of the request has the max len)
    ctx_max = tl.load(ctx_ptr + pid_b * next_n + next_n - 1)
    t0 = pid_t * TILE_T
    if t0 >= ctx_max:
        return

    tok = t0 + tl.arange(0, TILE_T)  # [T]
    m_tok = tok < ctx_max
    # page id per token (block table row is shared by all rows of the request)
    pages = tl.load(bt_ptr + pid_b * bt_stride + (tok // PS), mask=m_tok, other=0)
    pages = pages.to(tl.int64)
    tip = tok % PS
    k_base = pages * PAGEB + tip * KB  # [T] int64, k row base byte

    # K tile [T, D] fp8 (contiguous D bytes per token) + scales [T]
    d = tl.arange(0, D)
    k = tl.load(k_ptr + k_base[:, None] + d[None, :], mask=m_tok[:, None], other=0.0)
    sc = tl.load(s_ptr + pages * (PAGEB // 4) + (PS * KB // 4) + tip, mask=m_tok, other=0.0)  # [T]

    m = tl.arange(0, RG * H)  # flat (row-in-group * H + head)
    ri = tl.arange(0, RG)  # query row index within the group
    q_base = pid_b * next_n * H * D
    w_base = pid_b * next_n * H

    for g in tl.static_range(N_ITER):
        rows = g * RG + m // H  # query row index within the request
        m_row = rows < next_n
        q = tl.load(
            q_ptr + q_base + g * RG * H * D + m[:, None] * D + d[None, :],
            mask=m_row[:, None],
            other=0.0,
        )  # [RG*H, D] fp8
        w = tl.load(w_ptr + w_base + g * RG * H + m, mask=m_row, other=0.0)  # [RG*H]
        ri_g = g * RG + ri  # [RG] query row indices within the request
        m_row_g = ri_g < next_n
        ctx = tl.load(ctx_ptr + pid_b * next_n + ri_g, mask=m_row_g, other=0)  # [RG]

        dots = tl.dot(k, tl.trans(q))  # [T, RG*H] fp32
        dots = tl.maximum(dots, 0.0) * w[None, :]
        r3 = tl.reshape(dots, (TILE_T, RG, H))
        acc = tl.sum(r3, axis=2) * sc[:, None]  # [T, RG]
        out_off = (pid_b * next_n + ri_g)[None, :].to(tl.int64) * out_stride + tok[:, None]
        tl.store(
            out_ptr + out_off,
            acc,
            mask=m_row_g[None, :] & (tok[:, None] < ctx[None, :]),
        )


def _pow2ceil(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


# (tile_t, rg, num_warps, num_stages) chosen from the L=1M sweep (bisect_mine.py):
# next_n==1 is memory-bound (small dot) and measured at parity with the
# DeepGEMM kernel (the flag routes only next_n>=2 to this kernel); larger
# next_n is tensor-core bound and prefers RG=2 (N=64 wgmma) with more warps as
# the row count grows.
def _pick_cfg(next_n: int):
    if next_n <= 1:
        return dict(tile_t=128, rg=1, num_warps=4, num_stages=2)
    if next_n <= 7:
        return dict(tile_t=256, rg=2, num_warps=4, num_stages=3)
    if next_n < 32:
        return dict(tile_t=128, rg=2, num_warps=4, num_stages=2)
    return dict(tile_t=256, rg=2, num_warps=8, num_stages=2)


def decode_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_seq_len: int,
    out: torch.Tensor | None = None,
    tile_t: int | None = None,
    rg: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """Decode-shaped fp8 paged MQA logits (see module docstring).

    Args:
        q_fp8: ``[B, next_n, H, D]`` (or ``[R, 1, H, D]`` split form) fp8e4m3,
            contiguous.
        kv_cache: ``[P, 64, 1, 132]`` uint8 pool view (real layout: per 64-token
            page, 8192 B of k values then 64 fp32 scales; only the page stride
            is taken from this view).
        weights: ``[R, H]`` (or ``[B, next_n, H]``) fp32, contiguous.
        context_lens: ``[B, next_n]`` int32 per-query-row lengths.
        block_tables: ``[B, max_pages]`` int32, one row per REQUEST (de-expand
            before calling if you have per-query-row tables).
        max_seq_len: logits row width (``block_tables.shape[1] * 64``).

    Returns:
        ``[R, max_seq_len]`` fp32 view with row stride padded to 256 elements
        (same padded-view contract as DeepGEMM; positions ``>= len`` unwritten).
    """
    assert q_fp8.dim() == 4 and q_fp8.shape[3] == 128 and q_fp8.shape[2] == 32
    B, next_n, H, D = q_fp8.shape
    R = B * next_n
    assert kv_cache.dim() == 4 and kv_cache.shape[1:] == (64, 1, 132)
    assert kv_cache.dtype == torch.uint8 and kv_cache.stride(1) == 132
    assert block_tables.shape[0] == B and block_tables.dtype == torch.int32
    assert context_lens.shape == (B, next_n) and context_lens.dtype == torch.int32
    if weights.dim() == 3:
        weights = weights.view(R, H)
    assert weights.shape == (R, H) and weights.is_contiguous()

    cfg = _pick_cfg(next_n)
    if tile_t is not None:
        cfg["tile_t"] = tile_t
    if rg is not None:
        cfg["rg"] = rg
    if num_warps is not None:
        cfg["num_warps"] = num_warps
    if num_stages is not None:
        cfg["num_stages"] = num_stages

    stride = _align(max(max_seq_len, 1), 256)
    if out is None:
        out = torch.empty((R, stride), dtype=torch.float32, device=q_fp8.device)
    else:
        assert out.shape[0] == R and out.stride(0) >= max_seq_len and out.stride(1) == 1

    if R == 0 or B == 0:
        return out[:, :max_seq_len]

    rg_c = min(_pow2ceil(cfg["rg"]), _pow2ceil(max(next_n, 1)))
    n_iter = (next_n + rg_c - 1) // rg_c

    grid = (triton.cdiv(max_seq_len, cfg["tile_t"]), B)
    _decode_mqa_logits_kernel[grid](
        q_fp8,
        weights,
        kv_cache.view(torch.float8_e4m3fn),
        kv_cache.view(torch.float32),
        block_tables,
        context_lens,
        out,
        next_n,
        block_tables.stride(0),
        out.stride(0),
        H=H,
        D=D,
        PS=64,
        KB=128,
        PAGEB=64 * 132,
        TILE_T=cfg["tile_t"],
        RG=rg_c,
        N_ITER=n_iter,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out[:, :max_seq_len]
