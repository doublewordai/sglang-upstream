# Copyright 2023-2026 SGLang Team
# Fused post-logits pipeline (lane fused-sampling): penalties + temperature +
# softmax + token selection (greedy / gumbel-max sampled) in one Triton kernel,
# and a fused greedy speculative-verify (argmax + tree walk + bonus).
#
# Bit-exactness contract (vs the reference eager path):
#  - greedy: argmax over fp32 logits, ties -> lowest index (matches torch.argmax).
#  - sampled (temperature-only "simple case"): identical RNG consumption and
#    arithmetic to torch.multinomial(probs, 1) == (probs / q).argmax(-1), where
#    q = torch.empty_like(probs).exponential_(1.0) draws philox4x32-10 uniforms
#    with curand's exact grid mapping (see _philox_uniform below).  The softmax
#    denominator accumulation order differs from ATen's block softmax; decisions
#    are bit-exact whenever the top-2 score margin exceeds ~2 ulp (measured).
#  - verify (greedy spec): argmax identical to above; the tree walk is a port of
#    the existing Triton verify_tree_greedy kernel.

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice as _libdevice
except ImportError:  # older layouts
    from triton.language.extra.cuda import libdevice as _libdevice

# ---------------------------------------------------------------------------
# Philox4x32-10 (curand-compatible), vectorized over a Triton block.
# ---------------------------------------------------------------------------

PHILOX_M4x32_0 = tl.constexpr(0xD2511F53)
PHILOX_M4x32_1 = tl.constexpr(0xCD9E8D57)


@triton.jit
def _philox_round(c0, c1, c2, c3, k0, k1):
    m0 = tl.full((), 0xD2511F53, tl.uint64)
    m1 = tl.full((), 0xCD9E8D57, tl.uint64)
    p0 = c0.to(tl.uint64) * m0
    p1 = c2.to(tl.uint64) * m1
    lo0 = (p0 & 0xFFFFFFFF).to(tl.uint32)
    hi0 = (p0 >> 32).to(tl.uint32)
    lo1 = (p1 & 0xFFFFFFFF).to(tl.uint32)
    hi1 = (p1 >> 32).to(tl.uint32)
    return (
        hi1 ^ c1 ^ k0,
        lo1,
        hi0 ^ c3 ^ k1,
        lo0,
    )


@triton.jit
def _philox4x32_10(c0, c1, c2, c3, k0, k1):
    """10-round Philox4x32 (curand_Philox4x32_10); returns uint32 tensors."""
    c0 = c0.to(tl.uint32)
    c1 = c1.to(tl.uint32)
    c2 = c2.to(tl.uint32)
    c3 = c3.to(tl.uint32)
    k0 = k0.to(tl.uint32)
    k1 = k1.to(tl.uint32)
    w0 = tl.full((), 0x9E3779B9, tl.uint32)
    w1 = tl.full((), 0xBB67AE85, tl.uint32)
    for i in range(10):
        c0, c1, c2, c3 = _philox_round(c0, c1, c2, c3, k0, k1)
        if i < 9:
            k0 = k0 + w0
            k1 = k1 + w1
    return c0, c1, c2, c3


@triton.jit
def _exponential_from_philox(
    L, ctr0_lo, ctr0_hi, seed_lo, seed_hi, S
):
    """q ~ Exp(1) for global element indices L, replicating torch's
    distribution_elementwise_grid_stride_kernel + exponential transform.

    S = 256 * grid of the reference launch (see _ref_grid_stride).  For
    element L: idx = L % S, m = L // S, and the m-th uniform of thread idx is
    component (m % 4) of philox(ctr=(off/4 + m//4, 0, idx, 0), key=seed).
    ctr0_lo/hi = (off/4) split into 32-bit halves (curand carry semantics)."""
    ctr0_lo = ctr0_lo.to(tl.uint32)
    ctr0_hi = ctr0_hi.to(tl.uint32)
    seed_lo = seed_lo.to(tl.uint32)
    seed_hi = seed_hi.to(tl.uint32)
    idx = (L % S).to(tl.uint32)
    m = L // S
    k = (m // 4).to(tl.uint32)
    ii = m % 4
    c0 = ctr0_lo + k
    carry = tl.where(c0 < ctr0_lo, 1, 0).to(tl.uint32)
    c1 = ctr0_hi + carry
    z = tl.zeros_like(c0)
    o0, o1, o2, o3 = _philox4x32_10(c0, c1, idx, z, seed_lo, seed_hi)
    # curand_uniform: x * 2^-32 + 2^-33  (in (0, 1])
    x = tl.where(ii == 0, o0, tl.where(ii == 1, o1, tl.where(ii == 2, o2, o3)))
    u = x.to(tl.float32) * 2.3283064365386963e-10 + 1.1641532182693481e-10
    # at::transformation::exponential (CUDA path): -log(u), with u >= 1-eps/2
    # clamped so q = eps/2 (positive, tiny) instead of 0.  NOTE: the reference
    # kernel's at::log compiles to the FAST __logf (== lg2.approx.f32 * ln2),
    # NOT accurate logf (verified bitwise vs torch exponential_ on GH200).
    NEG_HALF_EPS: tl.constexpr = -5.9604644775390625e-08
    lg = tl.where(u >= 0.99999994039535522, NEG_HALF_EPS, _libdevice.fast_logf(u))
    return -lg


# ---------------------------------------------------------------------------
# Fused decode sampling, split-K full-width design (lane fused-sampling).
#   greedy:  argmax kernel (packed atomic, exact, lowest-idx ties) + finish
#   sampled: max kernel -> exp/partial-sum kernel -> philox sample/argmax
#            kernel -> finish.  The finish kernel unpacks the winner and
#            zeroes the atomic buffers for the next step (graph-friendly).
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["rows_per_req", "V"])
def _fused_argmax_rows_kernel(
    logits_ptr,  # [rows, V] fp32 (penalties NOT applied)
    add_pen_ptr,  # [rows/rows_per_req, V] fp32 or dummy
    scale_pen_ptr,
    temps_ptr,  # [rows] fp32 or dummy (sampled: t = x / T before argmax? NO:
    # greedy argmax runs on penalized logits, no temperature)
    out_packed_ptr,  # [rows] uint64, pre-zeroed
    rows_per_req,
    V,
    BLOCK: tl.constexpr,
    HAS_ADD_PEN: tl.constexpr,
    HAS_SCALE_PEN: tl.constexpr,
):
    pid = tl.program_id(0)
    nchunks = tl.cdiv(V, BLOCK)
    row = pid // nchunks
    chunk = pid % nchunks
    base = row.to(tl.int64) * V
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < V
    x = tl.load(logits_ptr + base + offs, mask=mask, other=float("-inf"))
    if HAS_ADD_PEN:
        pbase = (row // rows_per_req).to(tl.int64) * V
        x += tl.load(add_pen_ptr + pbase + offs, mask=mask, other=0.0)
    if HAS_SCALE_PEN:
        pbase = (row // rows_per_req).to(tl.int64) * V
        sc = tl.load(scale_pen_ptr + pbase + offs, mask=mask, other=1.0)
        x = tl.where(x < 0, x * sc, x / sc)
    x = tl.where(mask, x, float("-inf"))
    m = tl.max(x, 0)
    cand = x == m
    idx = tl.min(tl.where(cand, offs.to(tl.int64), V + 1), 0)
    ov = _f32_to_ordered_u32(m).to(tl.uint64)
    notidx = (~idx.to(tl.uint64)) & 0xFFFFFFFF
    tl.atomic_max(out_packed_ptr + row, (ov << 32) | notidx)


@triton.jit(do_not_specialize=["V", "S", "seed_lo", "seed_hi", "ctr0_lo", "ctr0_hi"])
def _fused_rowmax_kernel(
    logits_ptr,
    temps_ptr,  # [rows] fp32
    add_pen_ptr,
    scale_pen_ptr,
    rowmax_ptr,  # [rows] uint32 (ordered-transformed), pre-zeroed
    V,
    S,
    seed_lo,
    seed_hi,
    ctr0_lo,
    ctr0_hi,
    rng_ptr,
    BLOCK: tl.constexpr,
    HAS_ADD_PEN: tl.constexpr,
    HAS_SCALE_PEN: tl.constexpr,
    RNG_FROM_PTR: tl.constexpr,
):
    # per-chunk max of t = penalized(x) / T  ->  atomic max (exact)
    pid = tl.program_id(0)
    nchunks = tl.cdiv(V, BLOCK)
    row = pid // nchunks
    chunk = pid % nchunks
    base = row.to(tl.int64) * V
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < V
    x = tl.load(logits_ptr + base + offs, mask=mask, other=float("-inf"))
    if HAS_ADD_PEN:
        x += tl.load(add_pen_ptr + base + offs, mask=mask, other=0.0)
    if HAS_SCALE_PEN:
        sc = tl.load(scale_pen_ptr + base + offs, mask=mask, other=1.0)
        x = tl.where(x < 0, x * sc, x / sc)
    T = tl.load(temps_ptr + row)
    t = x / T
    t = tl.where(mask, t, float("-inf"))
    m = tl.max(t, 0)
    tl.atomic_max(rowmax_ptr + row, _f32_to_ordered_u32(m))


@triton.jit(do_not_specialize=["V"])
def _fused_expsum_kernel(
    logits_ptr,
    temps_ptr,
    add_pen_ptr,
    scale_pen_ptr,
    rowmax_ptr,  # [rows] uint32 (ordered)
    zpart_ptr,  # [rows, KP] fp32 partial sums
    V,
    KP: tl.constexpr,  # number of partial-sum chunks (this kernel's chunking)
    BLOCK: tl.constexpr,
    HAS_ADD_PEN: tl.constexpr,
    HAS_SCALE_PEN: tl.constexpr,
):
    pid = tl.program_id(0)
    nchunks = tl.cdiv(V, BLOCK)
    row = pid // nchunks
    chunk = pid % nchunks
    base = row.to(tl.int64) * V
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < V
    x = tl.load(logits_ptr + base + offs, mask=mask, other=0.0)
    if HAS_ADD_PEN:
        x += tl.load(add_pen_ptr + base + offs, mask=mask, other=0.0)
    if HAS_SCALE_PEN:
        sc = tl.load(scale_pen_ptr + base + offs, mask=mask, other=1.0)
        x = tl.where(x < 0, x * sc, x / sc)
    T = tl.load(temps_ptr + row)
    t = x / T
    mo = tl.load(rowmax_ptr + row)
    # invert ordered transform: b = (o ^ 0x80000000) if o & 0x80000000 else ~o
    m = tl.where(
        (mo & 0x80000000) != 0,
        (mo ^ 0x80000000).to(tl.int32, bitcast=True),
        (~mo).to(tl.int32, bitcast=True),
    ).to(tl.float32, bitcast=True)
    e = _libdevice.exp(t - m)
    e = tl.where(mask, e, 0.0)
    tl.store(zpart_ptr + row.to(tl.int64) * KP + chunk, tl.sum(e, 0))


@triton.jit(do_not_specialize=["V", "S", "seed_lo", "seed_hi", "ctr0_lo", "ctr0_hi"])
def _fused_sample_argmax_kernel(
    logits_ptr,
    temps_ptr,
    add_pen_ptr,
    scale_pen_ptr,
    rowmax_ptr,
    zpart_ptr,  # [rows, KP] fp32
    out_packed_ptr,  # [rows] uint64, pre-zeroed
    V,
    S,  # reference exponential_ grid stride
    seed_lo,
    seed_hi,
    ctr0_lo,
    ctr0_hi,
    rng_ptr,
    KP: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_ADD_PEN: tl.constexpr,
    HAS_SCALE_PEN: tl.constexpr,
    RNG_FROM_PTR: tl.constexpr,
):
    if RNG_FROM_PTR:
        seed_lo = tl.load(rng_ptr + 0)
        seed_hi = tl.load(rng_ptr + 1)
        ctr0_lo = tl.load(rng_ptr + 2)
        ctr0_hi = tl.load(rng_ptr + 3)
    seed_lo = seed_lo.to(tl.uint32)
    seed_hi = seed_hi.to(tl.uint32)
    ctr0_lo = ctr0_lo.to(tl.uint32)
    ctr0_hi = ctr0_hi.to(tl.uint32)

    pid = tl.program_id(0)
    nchunks = tl.cdiv(V, BLOCK)
    row = pid // nchunks
    chunk = pid % nchunks
    base = row.to(tl.int64) * V

    # Z = sum of partials in fixed chunk order (deterministic)
    Z = 0.0
    for k in range(KP):
        Z += tl.load(zpart_ptr + row.to(tl.int64) * KP + k)

    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < V
    x = tl.load(logits_ptr + base + offs, mask=mask, other=0.0)
    if HAS_ADD_PEN:
        x += tl.load(add_pen_ptr + base + offs, mask=mask, other=0.0)
    if HAS_SCALE_PEN:
        sc = tl.load(scale_pen_ptr + base + offs, mask=mask, other=1.0)
        x = tl.where(x < 0, x * sc, x / sc)
    T = tl.load(temps_ptr + row)
    t = x / T
    mo = tl.load(rowmax_ptr + row)
    m = tl.where(
        (mo & 0x80000000) != 0,
        (mo ^ 0x80000000).to(tl.int32, bitcast=True),
        (~mo).to(tl.int32, bitcast=True),
    ).to(tl.float32, bitcast=True)
    e = _libdevice.exp(t - m)
    p = e / Z
    L = base + offs
    q = _exponential_from_philox(L, ctr0_lo, ctr0_hi, seed_lo, seed_hi, S)
    score = p / q
    score = tl.where(mask, score, float("-inf"))
    # per-chunk winner (value, global idx); ties inside chunk -> lowest idx
    best = tl.max(score, 0)
    cand = score == best
    idx = tl.min(tl.where(cand, offs.to(tl.int64), V + 1), 0)
    ov = _f32_to_ordered_u32(best).to(tl.uint64)
    notidx = (~idx.to(tl.uint64)) & 0xFFFFFFFF
    tl.atomic_max(out_packed_ptr + row, (ov << 32) | notidx)


@triton.jit(do_not_specialize=["N"])
def _fused_finish_kernel(
    out_packed_ptr,  # [rows] uint64
    out_ptr,  # [rows] int32/int64 token ids
    rowmax_ptr,  # [rows] uint32 (zeroed here for the next step)
    N,
    OUT_INT64: tl.constexpr,
    ZERO_ROWMAX: tl.constexpr,
):
    i = tl.program_id(0)
    if i < N:
        packed = tl.load(out_packed_ptr + i)
        idx = (~packed) & 0xFFFFFFFF
        if OUT_INT64:
            tl.store(out_ptr + i, idx.to(tl.int64))
        else:
            tl.store(out_ptr + i, idx.to(tl.int32))
        tl.store(out_packed_ptr + i, tl.zeros((), tl.uint64))
        if ZERO_ROWMAX:
            tl.store(rowmax_ptr + i, tl.zeros((), tl.uint32))


_SCRATCH: dict = {}


def _decode_scratch(dev, rows, kp):
    key = (dev, rows, kp)
    if key not in _SCRATCH:
        _SCRATCH[key] = {
            "rowmax": torch.zeros(rows, dtype=torch.int32, device=dev),  # u32 view
            "out_packed": torch.zeros(rows, dtype=torch.int64, device=dev),  # u64 view
            "zpart": torch.empty((rows, kp), dtype=torch.float32, device=dev),
        }
    return _SCRATCH[key]


def _u32_view(t):
    return t.view(torch.uint32) if t.dtype == torch.int32 else t


def _u64_view(t):
    return t.view(torch.uint64) if t.dtype == torch.int64 else t


def fused_decode_sample(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    additive_penalties: Optional[torch.Tensor],
    scaling_penalties: Optional[torch.Tensor],
    greedy: bool,
) -> torch.Tensor:
    """Returns next-token ids; bit-exact vs the reference path (greedy:
    torch.argmax; sampled: softmax + multinomial simple case)."""
    b, V = logits.shape
    dev = logits.device
    logits = logits.contiguous()
    temps = temperatures.reshape(-1).contiguous()
    out_dtype = torch.int64 if greedy else torch.int32
    out = torch.empty((b,), dtype=out_dtype, device=dev)

    seed = off = 0
    S = 1
    rng_ptr = _rng_buf(dev)
    rng_from_ptr = False
    if not greedy:
        numel = b * V
        S = _ref_grid_stride(numel)
        delta = ((numel - 1) // (S * 4) + 1) * 4
        # The reference exponential_ consumes the CURRENT offset, then the
        # generator advances by delta.  Read first, then advance.
        seed, off = _read_philox_state(dev)
        _advance_philox_offset(dev, delta)
        if torch.cuda.is_current_stream_capturing():
            # graph-safe: bake pointer reads into the capture; the replay
            # wrapper refreshes the pinned staging buffer (and advances the
            # generator) before each replay
            rng_from_ptr = True
            stage = _rng_stage(dev)
            stage[0] = seed & 0xFFFFFFFF
            stage[1] = (seed >> 32) & 0xFFFFFFFF
            stage[2] = (off // 4) & 0xFFFFFFFF
            stage[3] = ((off // 4) >> 32) & 0xFFFFFFFF
            rng_ptr.copy_(stage, non_blocking=True)

    has_add = additive_penalties is not None
    has_scale = scaling_penalties is not None
    dummy = logits

    if greedy:
        sc = _decode_scratch(dev, b, 1)
        out_packed = _u64_view(sc["out_packed"])
        BLOCK = 8192
        nchunks = triton.cdiv(V, BLOCK)
        _fused_argmax_rows_kernel[(b * nchunks,)](
            logits,
            additive_penalties if has_add else dummy,
            scaling_penalties if has_scale else dummy,
            temps,
            out_packed,
            1,
            V,
            BLOCK=BLOCK,
            HAS_ADD_PEN=has_add,
            HAS_SCALE_PEN=has_scale,
            num_warps=32,
        )
        _fused_finish_kernel[(b,)](
            out_packed, out, _u32_view(sc["rowmax"]), b,
            OUT_INT64=True, ZERO_ROWMAX=False,
        )
        return out

    # sampled: 3 full-width passes + finish
    sc = _decode_scratch(dev, b, triton.cdiv(V, 8192))
    rowmax = _u32_view(sc["rowmax"])
    out_packed = _u64_view(sc["out_packed"])
    zpart = sc["zpart"]
    KP = zpart.shape[1]

    BLOCK_AB = 8192
    nchunks_ab = triton.cdiv(V, BLOCK_AB)
    _fused_rowmax_kernel[(b * nchunks_ab,)](
        logits, temps,
        additive_penalties if has_add else dummy,
        scaling_penalties if has_scale else dummy,
        rowmax, V, S, seed & 0xFFFFFFFF, seed >> 32,
        (off // 4) & 0xFFFFFFFF, (off // 4) >> 32, rng_ptr,
        BLOCK=BLOCK_AB, HAS_ADD_PEN=has_add, HAS_SCALE_PEN=has_scale,
        RNG_FROM_PTR=rng_from_ptr, num_warps=32,
    )
    _fused_expsum_kernel[(b * nchunks_ab,)](
        logits, temps,
        additive_penalties if has_add else dummy,
        scaling_penalties if has_scale else dummy,
        rowmax, zpart, V,
        KP=KP, BLOCK=BLOCK_AB, HAS_ADD_PEN=has_add, HAS_SCALE_PEN=has_scale,
        num_warps=32,
    )
    BLOCK_C = 2048
    nchunks_c = triton.cdiv(V, BLOCK_C)
    _fused_sample_argmax_kernel[(b * nchunks_c,)](
        logits, temps,
        additive_penalties if has_add else dummy,
        scaling_penalties if has_scale else dummy,
        rowmax, zpart, out_packed, V, S,
        seed & 0xFFFFFFFF, seed >> 32,
        (off // 4) & 0xFFFFFFFF, (off // 4) >> 32, rng_ptr,
        KP=KP, BLOCK=BLOCK_C, HAS_ADD_PEN=has_add, HAS_SCALE_PEN=has_scale,
        RNG_FROM_PTR=rng_from_ptr, num_warps=16,
    )
    _fused_finish_kernel[(b,)](
        out_packed, out, rowmax, b,
        OUT_INT64=False, ZERO_ROWMAX=True,
    )
    return out


# ---------------------------------------------------------------------------
# Kernel 3: fused greedy-verify phase 1 — full-width argmax over [rows, V]
# with per-request penalty broadcast; packed atomic max (exact, ties -> low idx)
# followed by a tiny decode kernel.
# ---------------------------------------------------------------------------


@triton.jit
def _f32_to_ordered_u32(x):
    """Monotonic map fp32 bits -> uint32 (total order matches float order)."""
    b = x.to(tl.int32, bitcast=True).to(tl.uint32)
    return tl.where((b & 0x80000000) != 0, ~b, b | 0x80000000)


@triton.jit(do_not_specialize=["rows_per_req", "V"])
def fused_verify_argmax_kernel(
    logits_ptr,  # [rows, V] fp32
    add_pen_ptr,  # [bs, V] fp32 (per request; row i uses add_pen[i // rows_per_req])
    scale_pen_ptr,
    out_packed_ptr,  # [rows] uint64, pre-zeroed
    rows_per_req,
    V,
    BLOCK: tl.constexpr,
    HAS_ADD_PEN: tl.constexpr,
    HAS_SCALE_PEN: tl.constexpr,
):
    pid = tl.program_id(0)
    nchunks = tl.cdiv(V, BLOCK)
    row = pid // nchunks
    chunk = pid % nchunks
    start = chunk * BLOCK

    base = row.to(tl.int64) * V
    offs = start + tl.arange(0, BLOCK)
    mask = offs < V
    x = tl.load(logits_ptr + base + offs, mask=mask, other=float("-inf"))
    if HAS_ADD_PEN:
        pbase = (row // rows_per_req).to(tl.int64) * V
        x += tl.load(add_pen_ptr + pbase + offs, mask=mask, other=0.0)
    if HAS_SCALE_PEN:
        pbase = (row // rows_per_req).to(tl.int64) * V
        s = tl.load(scale_pen_ptr + pbase + offs, mask=mask, other=1.0)
        x = tl.where(x < 0, x * s, x / s)
    x = tl.where(mask, x, float("-inf"))

    m = tl.max(x, 0)
    cand = x == m
    idx = tl.min(tl.where(cand, offs.to(tl.int64), V + 1), 0)

    # pack: ordered value in high 32 bits, ~idx in low 32 (ties -> lowest idx)
    ov = _f32_to_ordered_u32(m).to(tl.uint64)
    notidx = (~idx.to(tl.uint64)) & 0xFFFFFFFF
    packed = (ov << 32) | notidx
    tl.atomic_max(out_packed_ptr + row, packed)


@triton.jit(do_not_specialize=["N"])
def fused_verify_argmax_decode_kernel(
    out_packed_ptr,  # [rows] uint64
    out_ptr,  # [rows] int64/int32 target_predict
    N,
    OUT_INT64: tl.constexpr,
):
    i = tl.program_id(0)
    if i < N:
        packed = tl.load(out_packed_ptr + i)
        idx = (~packed) & 0xFFFFFFFF
        if OUT_INT64:
            tl.store(out_ptr + i, idx.to(tl.int64))
        else:
            tl.store(out_ptr + i, idx.to(tl.int32))


# ---------------------------------------------------------------------------
# Kernel 4: fused greedy-verify phase 2 — tree walk + bonus token.
# Port of sglang/kernels/ops/speculative/spec_tree.py verify_tree_greedy +
# fill_bonus_tokens, with target_predict consumed in-register.
# ---------------------------------------------------------------------------


@triton.jit
def fused_verify_walk_kernel(
    predicts_ptr,  # [bs * ndt] int32 (pre-zeroed)
    accept_index_ptr,  # [bs, nst] int32 (pre-filled -1)
    accept_token_num_ptr,  # [bs] int32 (num correct drafts)
    bonus_ptr,  # [bs] int32
    candidates_ptr,  # [bs, ndt]
    retrieve_index_ptr,  # [bs, ndt]
    retrieve_next_token_ptr,  # [bs, ndt]
    retrieve_next_sibling_ptr,  # [bs, ndt]
    target_predict_ptr,  # [bs * ndt] int32/int64
    ndt: tl.constexpr,  # num_draft_tokens
    nst: tl.constexpr,  # num_speculative_tokens (accept_index width)
):
    bx = tl.program_id(0)

    last_accept_retrieve_idx = tl.load(retrieve_index_ptr + bx * ndt)
    tl.store(accept_index_ptr + bx * nst, last_accept_retrieve_idx)
    num_accept_tokens = tl.cast(0, last_accept_retrieve_idx.dtype)
    cur_index = tl.cast(0, last_accept_retrieve_idx.dtype)

    should_continue = 1
    for j in range(1, nst):
        if should_continue:
            cur_index = tl.load(retrieve_next_token_ptr + bx * ndt + cur_index)

            target_row = last_accept_retrieve_idx // ndt
            target_col = last_accept_retrieve_idx % ndt
            target_token = tl.load(
                target_predict_ptr + target_row * ndt + target_col
            )

            found_match = 0
            for _ in range(ndt):
                if found_match == 0:
                    is_valid = cur_index != -1
                    safe_cur_index = cur_index * is_valid
                    safe_index = bx * ndt + safe_cur_index
                    draft_index = tl.load(retrieve_index_ptr + safe_index)
                    draft_token = tl.load(candidates_ptr + safe_index)
                    token_match = is_valid & (draft_token == target_token)
                    tl.store(
                        predicts_ptr + last_accept_retrieve_idx,
                        target_token,
                        mask=token_match,
                    )
                    next_num_accept_tokens = num_accept_tokens + 1
                    tl.store(
                        accept_index_ptr + bx * nst + next_num_accept_tokens,
                        draft_index,
                        mask=token_match,
                    )
                    num_accept_tokens = num_accept_tokens + token_match
                    last_accept_retrieve_idx = (
                        token_match * draft_index
                        + (~token_match) * last_accept_retrieve_idx
                    )
                    found_match = token_match * 1 + (~is_valid) * (-1)
                    cur_index = tl.load(
                        retrieve_next_sibling_ptr + safe_index,
                        mask=~token_match & is_valid,
                        other=cur_index,
                    )
            if found_match != 1:
                should_continue = 0

    tl.store(accept_token_num_ptr + bx, num_accept_tokens)

    target_row = last_accept_retrieve_idx // ndt
    target_col = last_accept_retrieve_idx % ndt
    final_target = tl.load(target_predict_ptr + target_row * ndt + target_col)
    tl.store(predicts_ptr + last_accept_retrieve_idx, final_target)
    # bonus = predict[accept_index[bx, num_accept]] (accept_lens = drafts + 1)
    bonus_slot = accept_index_ptr + bx * nst + num_accept_tokens
    tl.store(bonus_ptr + bx, final_target.to(tl.int32))
    _ = bonus_slot  # (kept for symmetry; bonus == final_target here)


# ---------------------------------------------------------------------------
# Host-side launchers
# ---------------------------------------------------------------------------


def _ref_grid_stride(numel: int, sm_count: int = 132, blocks_per_sm: int = 8) -> int:
    """Replicates ATen calc_execution_policy for exponential_ on fp32:
    block 256, grid = min(SMs * blocks_per_sm, ceil(numel / 256))."""
    grid = min(sm_count * blocks_per_sm, (numel + 255) // 256)
    return 256 * grid


def philox_counter_offset(numel: int, sm_count: int = 132, blocks_per_sm: int = 8):
    S = _ref_grid_stride(numel, sm_count, blocks_per_sm)
    return ((numel - 1) // (S * 4) + 1) * 4


def _read_philox_state(device) -> Tuple[int, int]:
    gen = torch.cuda.default_generators[device.index or 0]
    v = gen.get_state()[:16].view(torch.int64)
    return int(v[0]), int(v[1])


def _advance_philox_offset(device, delta: int):
    """Advance the generator's philox offset by delta (little-endian layout:
    state[:8] = seed uint64, state[8:16] = offset uint64)."""
    gen = torch.cuda.default_generators[device.index or 0]
    st = gen.get_state()
    v = st[:16].view(torch.int64)
    seed = int(v[0])
    off = int(v[1]) + delta
    v[1] = off
    gen.set_state(st)
    return seed, off


_RNG_BUF: dict = {}


def _rng_buf(dev):
    if dev not in _RNG_BUF:
        _RNG_BUF[dev] = torch.zeros(4, dtype=torch.int64, device=dev)
    return _RNG_BUF[dev]


_RNG_STAGE: dict = {}


def _rng_stage(dev):
    # pinned staging buffer so the H2D refresh is graph-capturable
    if dev not in _RNG_STAGE:
        _RNG_STAGE[dev] = torch.zeros(4, dtype=torch.int64, pin_memory=True)
    return _RNG_STAGE[dev]


def fused_verify_greedy(
    logits: torch.Tensor,  # [bs * ndt, V] fp32 (penalties NOT applied)
    additive_penalties: Optional[torch.Tensor],  # [bs, V] or None
    scaling_penalties: Optional[torch.Tensor],
    candidates: torch.Tensor,  # [bs, ndt]
    retrieve_index: torch.Tensor,  # [bs, ndt]
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    ndt: int,
    nst: int,
):
    """Greedy target-verify: fused penalties + argmax + tree walk + bonus.
    Returns (predicts, accept_index, accept_token_num, bonus)."""
    rows, V = logits.shape
    bs = rows // ndt
    dev = logits.device

    out_packed = torch.zeros((rows,), dtype=torch.uint64, device=dev)
    nchunks = triton.cdiv(V, 16384)
    dummy = logits
    fused_verify_argmax_kernel[(rows * nchunks,)](
        logits,
        additive_penalties if additive_penalties is not None else dummy,
        scaling_penalties if scaling_penalties is not None else dummy,
        out_packed,
        ndt,
        V,
        BLOCK=16384,
        HAS_ADD_PEN=additive_penalties is not None,
        HAS_SCALE_PEN=scaling_penalties is not None,
        num_warps=8,
    )
    target_predict = torch.empty((rows,), dtype=torch.int32, device=dev)
    fused_verify_argmax_decode_kernel[(rows,)](
        out_packed, target_predict, rows, OUT_INT64=False
    )

    predicts = torch.zeros((rows,), dtype=torch.int32, device=dev)
    accept_index = torch.full((bs, nst), -1, dtype=torch.int32, device=dev)
    accept_token_num = torch.empty((bs,), dtype=torch.int32, device=dev)
    bonus = torch.empty((bs,), dtype=torch.int32, device=dev)
    fused_verify_walk_kernel[(bs,)](
        predicts,
        accept_index,
        accept_token_num,
        bonus,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        ndt=ndt,
        nst=nst,
        num_warps=1,
    )
    return predicts, accept_index, accept_token_num, bonus
