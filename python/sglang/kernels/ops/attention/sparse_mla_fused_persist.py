"""Persistent fused split+combine sparse MLA decode (SM90/GH200, Triton).

Single launch, grid = min(#SMs, work) CTAs. The grid is capped at co-resident
capacity, so every CTA runs concurrently — that is what makes both combine
modes deadlock-free at ANY batch (the contest kernel's spin barrier deadlocked
only because its grid (b x P) exceeded resident CTAs at b=16).

Work items (request t, head-chunk hc, split s) are assigned as contiguous
blocks per CTA (CTA c takes [c*k, (c+1)*k), k = ceil(W/G)). Each item runs the
contest stride-partition online-softmax over its 2048/P indices and writes
split partials to a workspace. Two in-kernel combine modes:

  COMBINE=0 "last-arriver" (default, no spin): atomic_add on a per-group
    counter; the CTA that sees old == P-1 merges all P partials and writes the
    output, then subtracts P so the counter is 0 again at the next launch
    (CUDA-graph replayable with no memset node).
  COMBINE=1 "spin + D-parallel" (contest structure, safe here because the grid
    is co-resident): every split CTA waits for done[g] == P then combines its
    own D-slice. Requires k == 1 (a CTA must never block its own group's later
    item); the launcher falls back to COMBINE=0 otherwise. Replay safety: a
    second counter counts retired combiners; the last one zeroes both (the
    reset is ordered after every CTA's poll exit, so no poller can miss it).

ORDER=0 assigns items request-major (b=1 reproduces the contest CTA layout);
ORDER=1 split-major co-schedules the same split across requests so rows shared
between requests (shared prefixes) are served from L2 once.

Semantics match production flash_mla_with_kvcache sparse fp8 decode: score
every non-negative index, mask only < 0 sentinels, seqlens only size the
schedule, all-sentinel rows output exactly 0.
"""
from typing import Optional

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
ROW = tl.constexpr(656)  # bytes per pool row
ROW_F32 = tl.constexpr(656 // 4)  # 164
ROW_BF16 = tl.constexpr(656 // 2)  # 328
NOPE = tl.constexpr(512)
ROPE = tl.constexpr(64)
SCALE_OFF_F32 = tl.constexpr(512 // 4)  # fp32-element offset of scales in a row
ROPE_OFF_BF16 = tl.constexpr((512 + 16) // 2)  # 264


@triton.jit
def _sdf_fwd(
    Q_ptr, Pool8_ptr, PoolS_ptr, PoolR_ptr, Idx_ptr,
    Pm_ptr, Pl_ptr, Pacc_ptr, Cnt_ptr, Out_ptr,
    sm_scale_log2e,
    stride_qt, stride_it,
    num_ctas,
    TOPK: tl.constexpr,
    Hc: tl.constexpr,   # heads per chunk (16/32/64)
    HC: tl.constexpr,   # chunks = 64 // Hc
    P: tl.constexpr,    # row splits
    B: tl.constexpr,    # requests
    BLOCK_N: tl.constexpr,
    COMBINE: tl.constexpr,
    ORDER: tl.constexpr,
    PDT: tl.constexpr,  # partial acc dtype: 0 fp32, 1 bf16
    DEQ: tl.constexpr,  # 0 bf16-dequant dots (contest), 1 native-fp8 group dots
):
    pid = tl.program_id(0)
    SPLIT: tl.constexpr = TOPK // P
    NEG_INF: tl.constexpr = float("-inf")

    W: tl.constexpr = B * HC * P
    k_items = (W + num_ctas - 1) // num_ctas
    w_lo = pid * k_items
    w_hi = tl.minimum(W, w_lo + k_items)

    offs_d = tl.arange(0, NOPE)
    offs_pe = tl.arange(0, ROPE)
    offs_h = tl.arange(0, Hc)
    offs_n = tl.arange(0, BLOCK_N)

    for w in range(w_lo, w_hi):
        # ---- decode work item -> (t, hcc, s) ----
        if ORDER == 0:
            t = w // (HC * P)
            r0 = w % (HC * P)
            hcc = r0 // P
            s = r0 % P
        else:
            s = w // (B * HC)
            r0 = w % (B * HC)
            t = r0 // HC
            hcc = r0 % HC
        g = t * HC + hcc  # combine group

        # ---- q chunk ----
        offs_hh = hcc * Hc + offs_h
        if DEQ == 0:
            qn = tl.load(
                Q_ptr + t * stride_qt + offs_hh[:, None] * (NOPE + ROPE) + offs_d[None, :],
                eviction_policy="evict_last",
            )
            qp = tl.load(
                Q_ptr + t * stride_qt + offs_hh[:, None] * (NOPE + ROPE) + NOPE + offs_pe[None, :],
                eviction_policy="evict_last",
            )
        else:
            # native-fp8 path: per-row fp8 quant of the nope latent (rope stays bf16)
            offs_g = tl.arange(0, 128)
            qg0 = tl.load(Q_ptr + t * stride_qt + offs_hh[:, None] * (NOPE + ROPE) + 0 * 128 + offs_g[None, :], eviction_policy="evict_last").to(tl.float32)
            qg1 = tl.load(Q_ptr + t * stride_qt + offs_hh[:, None] * (NOPE + ROPE) + 1 * 128 + offs_g[None, :], eviction_policy="evict_last").to(tl.float32)
            qg2 = tl.load(Q_ptr + t * stride_qt + offs_hh[:, None] * (NOPE + ROPE) + 2 * 128 + offs_g[None, :], eviction_policy="evict_last").to(tl.float32)
            qg3 = tl.load(Q_ptr + t * stride_qt + offs_hh[:, None] * (NOPE + ROPE) + 3 * 128 + offs_g[None, :], eviction_policy="evict_last").to(tl.float32)
            qp = tl.load(
                Q_ptr + t * stride_qt + offs_hh[:, None] * (NOPE + ROPE) + NOPE + offs_pe[None, :],
                eviction_policy="evict_last",
            )
            aq = tl.maximum(tl.max(tl.abs(qg0), 1), tl.maximum(tl.max(tl.abs(qg1), 1),
                         tl.maximum(tl.max(tl.abs(qg2), 1), tl.max(tl.abs(qg3), 1))))
            s_q = tl.maximum(aq, 1e-30) / 448.0
            q8_0 = (qg0 / s_q[:, None]).to(tl.float8e4nv)
            q8_1 = (qg1 / s_q[:, None]).to(tl.float8e4nv)
            q8_2 = (qg2 / s_q[:, None]).to(tl.float8e4nv)
            q8_3 = (qg3 / s_q[:, None]).to(tl.float8e4nv)

        # ---- split phase: stride partition {s, s+P, ...} of the 2048 indices ----
        m_i = tl.full([Hc], NEG_INF, dtype=tl.float32)
        l_i = tl.zeros([Hc], dtype=tl.float32)
        if DEQ == 0:
            acc = tl.zeros([Hc, NOPE], dtype=tl.float32)
        else:
            acc0 = tl.zeros([Hc, 128], dtype=tl.float32)
            acc1 = tl.zeros([Hc, 128], dtype=tl.float32)
            acc2 = tl.zeros([Hc, 128], dtype=tl.float32)
            acc3 = tl.zeros([Hc, 128], dtype=tl.float32)

        offs_sp = s + tl.arange(0, SPLIT) * P
        idx_scan = tl.load(Idx_ptr + t * stride_it + offs_sp)
        num_valid = tl.sum((idx_scan >= 0).to(tl.int32), axis=0)
        max_bn = ((num_valid + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

        for bn in range(0, max_bn, BLOCK_N):
            pos = s + (bn + offs_n) * P
            idx = tl.load(Idx_ptr + t * stride_it + pos)
            # a split owns stride-partition positions {s + i*P, i < SPLIT}; when
            # SPLIT < BLOCK_N the block would read other splits' positions and
            # beyond topk - mask to this split's own range
            valid = (idx >= 0) & ((bn + offs_n) < SPLIT)
            safe = tl.where(valid, idx, 0).to(tl.int64)

            if DEQ == 0:
                kc8 = tl.load(
                    Pool8_ptr + safe[:, None] * ROW + offs_d[None, :],
                    mask=valid[:, None], other=0.0, eviction_policy="evict_first",
                )
                kp = tl.load(
                    PoolR_ptr + safe[:, None] * ROW_BF16 + ROPE_OFF_BF16 + offs_pe[None, :],
                    mask=valid[:, None], other=0.0, eviction_policy="evict_first",
                )
                sc0 = tl.load(PoolS_ptr + safe * ROW_F32 + SCALE_OFF_F32 + 0, mask=valid, other=0.0)
                sc1 = tl.load(PoolS_ptr + safe * ROW_F32 + SCALE_OFF_F32 + 1, mask=valid, other=0.0)
                sc2 = tl.load(PoolS_ptr + safe * ROW_F32 + SCALE_OFF_F32 + 2, mask=valid, other=0.0)
                sc3 = tl.load(PoolS_ptr + safe * ROW_F32 + SCALE_OFF_F32 + 3, mask=valid, other=0.0)

                # exact per-group dequant to bf16 (contest adaptation)
                kcf = kc8.to(tl.float32)
                kcf = tl.where(offs_d[None, :] < 128, kcf * sc0[:, None],
                      tl.where(offs_d[None, :] < 256, kcf * sc1[:, None],
                      tl.where(offs_d[None, :] < 384, kcf * sc2[:, None],
                               kcf * sc3[:, None])))
                kc = kcf.to(tl.bfloat16)

                logits = tl.dot(qn, tl.trans(kc))
                logits = tl.dot(qp, tl.trans(kp), acc=logits)
                logits = logits * sm_scale_log2e
            else:
                kp = tl.load(
                    PoolR_ptr + safe[:, None] * ROW_BF16 + ROPE_OFF_BF16 + offs_pe[None, :],
                    mask=valid[:, None], other=0.0, eviction_policy="evict_first",
                )
                sc0 = tl.load(PoolS_ptr + safe * ROW_F32 + SCALE_OFF_F32 + 0, mask=valid, other=0.0)
                sc1 = tl.load(PoolS_ptr + safe * ROW_F32 + SCALE_OFF_F32 + 1, mask=valid, other=0.0)
                sc2 = tl.load(PoolS_ptr + safe * ROW_F32 + SCALE_OFF_F32 + 2, mask=valid, other=0.0)
                sc3 = tl.load(PoolS_ptr + safe * ROW_F32 + SCALE_OFF_F32 + 3, mask=valid, other=0.0)
                k0 = tl.load(Pool8_ptr + safe[:, None] * ROW + 0 * 128 + offs_g[None, :], mask=valid[:, None], other=0.0, eviction_policy="evict_first")
                k1 = tl.load(Pool8_ptr + safe[:, None] * ROW + 1 * 128 + offs_g[None, :], mask=valid[:, None], other=0.0, eviction_policy="evict_first")
                k2 = tl.load(Pool8_ptr + safe[:, None] * ROW + 2 * 128 + offs_g[None, :], mask=valid[:, None], other=0.0, eviction_policy="evict_first")
                k3 = tl.load(Pool8_ptr + safe[:, None] * ROW + 3 * 128 + offs_g[None, :], mask=valid[:, None], other=0.0, eviction_policy="evict_first")

                # QK: raw fp8 group dots, exact descale on the fp32 accumulator
                nope = tl.dot(q8_0, tl.trans(k0)) * sc0[None, :]
                nope += tl.dot(q8_1, tl.trans(k1)) * sc1[None, :]
                nope += tl.dot(q8_2, tl.trans(k2)) * sc2[None, :]
                nope += tl.dot(q8_3, tl.trans(k3)) * sc3[None, :]
                logits = (tl.dot(qp, tl.trans(kp)) + nope * s_q[:, None]) * sm_scale_log2e
            logits = tl.where(valid[None, :], logits, NEG_INF)

            m_new = tl.maximum(m_i, tl.max(logits, axis=1))
            m_new_safe = tl.where(m_new == NEG_INF, 0.0, m_new)
            alpha = tl.where(m_i == NEG_INF, 0.0, tl.exp2(m_i - m_new_safe))
            p = tl.exp2(logits - m_new_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            if DEQ == 0:
                acc = acc * alpha[:, None]
                acc = tl.dot(p.to(tl.bfloat16), kc, acc=acc)
            else:
                # PV: p-tilde = p * s_v[j, g] per output group, V raw fp8 widened
                acc0 = acc0 * alpha[:, None]
                acc1 = acc1 * alpha[:, None]
                acc2 = acc2 * alpha[:, None]
                acc3 = acc3 * alpha[:, None]
                acc0 = tl.dot((p * sc0[None, :]).to(tl.bfloat16), k0.to(tl.bfloat16), acc=acc0)
                acc1 = tl.dot((p * sc1[None, :]).to(tl.bfloat16), k1.to(tl.bfloat16), acc=acc1)
                acc2 = tl.dot((p * sc2[None, :]).to(tl.bfloat16), k2.to(tl.bfloat16), acc=acc2)
                acc3 = tl.dot((p * sc3[None, :]).to(tl.bfloat16), k3.to(tl.bfloat16), acc=acc3)
            m_i = m_new

        # ---- store partials ----
        pbase = (g * P + s) * Hc
        tl.store(Pm_ptr + pbase + offs_h, m_i, cache_modifier=".cg")
        tl.store(Pl_ptr + pbase + offs_h, l_i, cache_modifier=".cg")
        if DEQ == 0:
            aoff = (pbase + offs_h[:, None]) * NOPE + offs_d[None, :]
            if PDT == 0:
                tl.store(Pacc_ptr + aoff, acc, cache_modifier=".cg")
            else:
                tl.store(Pacc_ptr + aoff, acc.to(tl.bfloat16), cache_modifier=".cg")
        else:
            aoff = (pbase + offs_h[:, None]) * NOPE + offs_g[None, :]
            if PDT == 0:
                tl.store(Pacc_ptr + aoff + 0 * 128, acc0, cache_modifier=".cg")
                tl.store(Pacc_ptr + aoff + 1 * 128, acc1, cache_modifier=".cg")
                tl.store(Pacc_ptr + aoff + 2 * 128, acc2, cache_modifier=".cg")
                tl.store(Pacc_ptr + aoff + 3 * 128, acc3, cache_modifier=".cg")
            else:
                tl.store(Pacc_ptr + aoff + 0 * 128, acc0.to(tl.bfloat16), cache_modifier=".cg")
                tl.store(Pacc_ptr + aoff + 1 * 128, acc1.to(tl.bfloat16), cache_modifier=".cg")
                tl.store(Pacc_ptr + aoff + 2 * 128, acc2.to(tl.bfloat16), cache_modifier=".cg")
                tl.store(Pacc_ptr + aoff + 3 * 128, acc3.to(tl.bfloat16), cache_modifier=".cg")

        if COMBINE == 0:
            # ---- last-arriver combine (no spin) ----
            old = tl.atomic_add(Cnt_ptr + g, 1, sem="acq_rel", scope="gpu")
            if old == P - 1:
                m_gl = tl.full([Hc], NEG_INF, dtype=tl.float32)
                l_gl = tl.zeros([Hc], dtype=tl.float32)
                acc_c = tl.zeros([Hc, NOPE], dtype=tl.float32)
                for si in range(0, P):
                    pb = (g * P + si) * Hc
                    m_si = tl.load(Pm_ptr + pb + offs_h)
                    l_si = tl.load(Pl_ptr + pb + offs_h)
                    if PDT == 0:
                        a_si = tl.load(
                            Pacc_ptr + (pb + offs_h[:, None]) * NOPE + offs_d[None, :]
                        )
                    else:
                        a_si = tl.load(
                            Pacc_ptr + (pb + offs_h[:, None]) * NOPE + offs_d[None, :]
                        ).to(tl.float32)
                    m_new = tl.maximum(m_gl, m_si)
                    m_ns = tl.where(m_new == NEG_INF, 0.0, m_new)
                    al = tl.where(m_gl == NEG_INF, 0.0, tl.exp2(m_gl - m_ns))
                    be = tl.where(m_si == NEG_INF, 0.0, tl.exp2(m_si - m_ns))
                    acc_c = acc_c * al[:, None] + a_si * be[:, None]
                    l_gl = l_gl * al + l_si * be
                    m_gl = m_new
                l_safe = tl.where(l_gl == 0.0, 1.0, l_gl)
                o = acc_c / l_safe[:, None]
                tl.store(
                    Out_ptr + t * (64 * NOPE) + offs_hh[:, None] * NOPE + offs_d[None, :],
                    o.to(tl.bfloat16),
                )
                # counter back to 0 for the next launch (graph replay safe)
                tl.atomic_add(Cnt_ptr + g, -P, sem="release", scope="gpu")
        else:
            # ---- spin + D-parallel combine (k == 1 only; launcher enforces) ----
            tl.atomic_add(Cnt_ptr + 2 * g, 1, sem="acq_rel", scope="gpu")
            dv = tl.atomic_add(Cnt_ptr + 2 * g, 0, sem="acquire", scope="gpu")
            while dv < P:
                dv = tl.atomic_add(Cnt_ptr + 2 * g, 0, sem="acquire", scope="gpu")
            BLOCK_D: tl.constexpr = NOPE // P
            offs_dd = s * BLOCK_D + tl.arange(0, BLOCK_D)
            m_gl = tl.full([Hc], NEG_INF, dtype=tl.float32)
            l_gl = tl.zeros([Hc], dtype=tl.float32)
            acc_c = tl.zeros([Hc, BLOCK_D], dtype=tl.float32)
            for si in range(0, P):
                pb = (g * P + si) * Hc
                m_si = tl.load(Pm_ptr + pb + offs_h)
                l_si = tl.load(Pl_ptr + pb + offs_h)
                if PDT == 0:
                    a_si = tl.load(
                        Pacc_ptr + (pb + offs_h[:, None]) * NOPE + offs_dd[None, :]
                    )
                else:
                    a_si = tl.load(
                        Pacc_ptr + (pb + offs_h[:, None]) * NOPE + offs_dd[None, :]
                    ).to(tl.float32)
                m_new = tl.maximum(m_gl, m_si)
                m_ns = tl.where(m_new == NEG_INF, 0.0, m_new)
                al = tl.where(m_gl == NEG_INF, 0.0, tl.exp2(m_gl - m_ns))
                be = tl.where(m_si == NEG_INF, 0.0, tl.exp2(m_si - m_ns))
                acc_c = acc_c * al[:, None] + a_si * be[:, None]
                l_gl = l_gl * al + l_si * be
                m_gl = m_new
            l_safe = tl.where(l_gl == 0.0, 1.0, l_gl)
            o = acc_c / l_safe[:, None]
            tl.store(
                Out_ptr + t * (64 * NOPE) + offs_hh[:, None] * NOPE + offs_dd[None, :],
                o.to(tl.bfloat16),
            )
            # retire; the last retiree zeroes both counters (ordered after
            # every CTA's poll exit, so no poller can miss the reset)
            old2 = tl.atomic_add(Cnt_ptr + 2 * g + 1, 1, sem="acq_rel", scope="gpu")
            if old2 == P - 1:
                tl.atomic_xchg(Cnt_ptr + 2 * g, 0, sem="release", scope="gpu")
                tl.atomic_xchg(Cnt_ptr + 2 * g + 1, 0, sem="release", scope="gpu")


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------

_WS_CACHE: dict = {}
_SM_COUNT: Optional[int] = None
_LAST_KERNEL = None


def last_kernel_info() -> Optional[dict]:
    """Occupancy introspection of the most recent launch (for the co-residency
    guard: a persistent grid larger than resident capacity hangs, it does not
    degrade). Best-effort across Triton versions."""
    k = _LAST_KERNEL
    if k is None:
        return None
    try:
        md = getattr(k, "metadata", None)
        return dict(
            n_regs=getattr(k, "n_regs", None),
            n_spills=getattr(k, "n_spills", None),
            shared=getattr(md, "shared", None) if md is not None else None,
            max_ctas_per_sm=_resident_ctas(k),
        )
    except Exception:
        return None


def _resident_ctas(k) -> Optional[int]:
    n_regs = getattr(k, "n_regs", None)
    md = getattr(k, "metadata", None)
    shared = getattr(md, "shared", 0) if md is not None else 0
    if n_regs is None:
        return None
    # Hopper: 65536 32-bit regs / SM, 232448 B smem / SM
    threads = 32 * (getattr(md, "num_warps", 8) if md is not None else 8)
    lim = 65536 // max(1, n_regs * threads)
    if shared:
        lim = min(lim, max(1, 232448 // shared))
    return lim


def sm_count(dev=None) -> int:
    global _SM_COUNT
    if _SM_COUNT is None:
        _SM_COUNT = torch.cuda.get_device_properties(
            dev if dev is not None else torch.cuda.current_device()
        ).multi_processor_count
    return _SM_COUNT


def default_cfg(b: int) -> dict:
    """Persistent config: fill the SMs with one work item per CTA (W <= #SMs)."""
    if b <= 1:
        hc, p = 16, 32
    elif b <= 2:
        hc, p = 16, 16
    elif b <= 4:
        hc, p = 16, 8
    elif b <= 8:
        hc, p = 16, 4
    elif b <= 16:
        hc, p = 32, 4
    elif b <= 32:
        hc, p = 64, 4
    elif b <= 64:
        hc, p = 64, 2
    else:
        hc, p = 64, 1
    return dict(Hc=hc, P=p, BLOCK_N=64, warps=8, stages=2,
                COMBINE=0, ORDER=0 if b <= 1 else 1, PDT=0, DEQ=0)


def resolve_cfg(b: int, cfg: Optional[dict]) -> dict:
    c = dict(default_cfg(b))
    if cfg:
        c.update(cfg)
    c["HC"] = 64 // c["Hc"]
    return c


def get_ws(b: int, hc: int, p: int, pdt: int, dev) -> dict:
    hc_chunks = 64 // hc
    key = (b, hc, p, pdt, dev.index if dev.index is not None else 0)
    ws = _WS_CACHE.get(key)
    if ws is None:
        ngroups = b * hc_chunks
        acc_dt = torch.float32 if pdt == 0 else torch.bfloat16
        ws = dict(
            pm=torch.empty(ngroups * p * hc, dtype=torch.float32, device=dev),
            pl=torch.empty(ngroups * p * hc, dtype=torch.float32, device=dev),
            pacc=torch.empty(ngroups * p * hc * 512, dtype=acc_dt, device=dev),
            cnt=torch.zeros(ngroups * 2, dtype=torch.int32, device=dev),
            out=torch.empty(b, 64, 512, dtype=torch.bfloat16, device=dev),
        )
        _WS_CACHE[key] = ws
    return ws


def sdf_fwd(q, pool, idx, sm_scale, out=None, cfg=None, ws=None):
    """q [b, 64, 576] bf16 contiguous; pool [L, 656] fp8-e4m3 view of the DSA
    pool; idx [b, 2048] i32; returns out [b, 64, 512] bf16.
    CUDA-graph safe at fixed shapes (workspace + self-resetting counters)."""
    B = q.shape[0]
    assert q.shape[1] == 64 and q.shape[2] == 576 and q.is_contiguous()
    L = pool.shape[0]
    assert pool.stride(1) == 1 and pool.stride(0) == 656
    # strided-view hazard (supervisor warning 04:2xZ): idx may be a view with
    # non-unit last-dim stride - refuse loudly instead of computing garbage
    assert idx.stride(1) == 1, f"idx last-dim stride {idx.stride(1)} != 1"
    assert idx.shape[1] == 2048
    dev = q.device
    c = resolve_cfg(B, cfg)
    Hc, HC, P = c["Hc"], c["HC"], c["P"]

    W = B * HC * P
    G = min(sm_count(dev), W)
    k = (W + G - 1) // G
    if c["COMBINE"] == 1 and k > 1:
        c["COMBINE"] = 0  # spin mode needs one item per CTA

    if ws is None:
        ws = get_ws(B, Hc, P, c["PDT"], dev)
    if out is None:
        out = ws["out"]

    pool_s = pool.view(torch.float32)
    pool_r = pool.view(torch.bfloat16)

    h = _sdf_fwd[(G,)](
        q, pool, pool_s, pool_r, idx,
        ws["pm"], ws["pl"], ws["pacc"], ws["cnt"], out,
        sm_scale * LOG2E,
        q.stride(0), idx.stride(0),
        G,
        TOPK=2048, Hc=Hc, HC=HC, P=P, B=B,
        BLOCK_N=c["BLOCK_N"], COMBINE=c["COMBINE"], ORDER=c["ORDER"], PDT=c["PDT"],
        DEQ=c["DEQ"],
        num_warps=c["warps"], num_stages=c["stages"],
    )
    global _LAST_KERNEL
    _LAST_KERNEL = h
    return out
