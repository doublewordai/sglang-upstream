"""StreamIndex-style partition-merge top-k for the DSA prefill indexer.

Key-axis chunked replacement for the prefill `mqa_logits + topk_transform`
pair: the scorer runs per key chunk producing a small [q, W] tile, an extract
kernel merges tile candidates into a persistent per-row candidate buffer
(exact top-2048 machinery from lane/topk-1pass), and a final kernel selects
and applies the production page-table transform. The [q, L] fp32 logits tensor
never exists; peak extra memory is one tile (q x W x 4 B, double-buffered when
pipelined) plus the candidate state (q x 8192 x 8 B).

Driver entry points:
- ``streamindex_topk_prefill``: full pipeline (deep_gemm.fp8_mqa_logits
  key-chunked on one stream, extraction pipelined on a second stream).
- ``streamindex_select_from_logits``: select machinery only, fed slices of a
  materialized [q, L] logits tensor (for select-only benchmarking vs
  production / topk-1pass on identical data).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

CAND_CAP = 8192  # per-row candidate capacity (TieValue {f32 value, u32 pos})
NEG_FLT_MAX = -3.4028234663852886e38  # -FLT_MAX


@cache_once
def _jit_streamindex_module():
    return load_jit(
        "dsa_streamindex_merge_v1",
        cuda_files=["dsa/streamindex_merge.cuh"],
        cuda_wrappers=[
            ("streamindex_extract", "StreamIndexMergeKernel::extract"),
            ("streamindex_final", "StreamIndexMergeKernel::final"),
        ],
    )


def _alloc_state(q: int, device, stats: bool):
    cand = torch.empty((q, CAND_CAP), dtype=torch.int64, device=device)
    cursor = torch.zeros(q, dtype=torch.int32, device=device)
    thresh = torch.full((q,), NEG_FLT_MAX, dtype=torch.float32, device=device)
    st = torch.zeros(q, 3, dtype=torch.int32, device=device) if stats else None
    return cand, cursor, thresh, st


def _merge_tiles(
    tile_fn,
    n_chunks: int,
    q: int,
    ks: torch.Tensor,
    ke: torch.Tensor,
    page_table_size_1: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    cand: torch.Tensor,
    cursor: torch.Tensor,
    thresh: torch.Tensor,
    stats: Optional[torch.Tensor],
    stream_ext: Optional[torch.cuda.Stream],
):
    """Run extract over each tile produced by tile_fn(ci) -> (tile, c0, c1),
    then the final select. tile_fn must enqueue the tile's production on the
    CURRENT stream and return a tensor valid on stream_ext (via events)."""
    module = _jit_streamindex_module()
    dev = cand.device
    cur = torch.cuda.current_stream(dev) if stream_ext is None else stream_ext
    # Deferred tile release: keep each tile referenced until its extraction's
    # event has fired. tile.record_stream() alone did NOT protect the freed
    # block from immediate reuse by the next scorer call (measured: pipelined
    # runs corrupted mid-stream chunks; keep-alive and sync-after-extract both
    # fixed it; deferred free keeps the pipeline -- see lane worklog).
    pending = []  # (tile, extract-done event)
    for ci in range(n_chunks):
        tile, c0, c1 = tile_fn(ci)
        with torch.cuda.stream(cur):
            module.streamindex_extract(
                tile, ks, ke, cand, cursor, thresh, int(c0), int(c1), stats
            )
        if stream_ext is None:
            # single stream: freeing at rebinding is ordered and safe
            del tile
        else:
            ext_ev = torch.cuda.Event()
            ext_ev.record(stream_ext)
            pending.append((tile, ext_ev))
            pending = [(t, e) for (t, e) in pending if not e.query()]
            if len(pending) > 2:
                # back-pressure: bound in-flight tiles (the host would otherwise
                # run the whole loop ahead of the device and hold every tile)
                pending[0][1].synchronize()
                pending = [(t, e) for (t, e) in pending if not e.query()]
    for _t, e in pending:
        e.synchronize()
    pending = []
    dst = torch.empty((q, topk), dtype=torch.int32, device=dev)
    with torch.cuda.stream(cur):
        module.streamindex_final(cand, cursor, ks, ke, dst, page_table_size_1,
                                 cu_seqlens_q, int(topk))
    if stream_ext is not None:
        # hand dst back to the caller's stream only after it is produced
        done = torch.cuda.Event()
        done.record(stream_ext)
        torch.cuda.current_stream(dev).wait_event(done)
    return dst


def streamindex_topk_prefill(
    q_fp8: torch.Tensor,
    kv: Tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    page_table_size_1: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    W: int = 8192,
    pipeline: bool = True,
    stats: bool = False,
    scorer=None,
) -> torch.Tensor:
    """Full pipeline. Same output contract as ``fast_topk_transform_fused``
    (PAGED, page_size=1 table): dst[i, t] = page_table_size_1[seq(i)][pos_t].

    Args:
        q_fp8: [q, H, D] fp8 indexer queries; kv: (data [L, D] fp8, scales
            [L] fp32); weights: [q, H] fp32; ks/ke: [q] int32 GLOBAL windows.
        W: key-axis chunk width. pipeline: overlap extraction of chunk c-1
            with scoring of chunk c on a second CUDA stream.
        scorer: callable(q_fp8, kv_slice, weights, ks_local, ke_local) ->
            tile; defaults to deep_gemm.fp8_mqa_logits (clean_logits=False).
    """
    assert 0 < topk <= 2048
    q, L = q_fp8.shape[0], kv[0].shape[0]
    dev = q_fp8.device
    if scorer is None:
        import deep_gemm

        def scorer(qq, kkv, ww, ksl, kel):
            return deep_gemm.fp8_mqa_logits(qq, kkv, ww, ksl, kel, clean_logits=False)

    cand, cursor, thresh, st = _alloc_state(q, dev, stats)

    c0s = list(range(0, L, W))
    # per-chunk LOCAL windows for the scorer (kv is sliced; deep_gemm treats
    # ks/ke as indices into the passed kv; empty rows -> zero-length [ks, ks))
    ks_all = torch.clamp(ks[None, :] - torch.tensor(c0s, device=dev, dtype=torch.int32)[:, None], min=0)
    ke_all = torch.clamp(ke[None, :] - torch.tensor(c0s, device=dev, dtype=torch.int32)[:, None], min=0, max=W)
    ke_all = torch.maximum(ke_all, ks_all)

    stream_mqa = torch.cuda.Stream(dev) if pipeline else None
    stream_ext = torch.cuda.Stream(dev) if pipeline else None
    events = [torch.cuda.Event() for _ in c0s]
    if stream_mqa is not None:
        # state init + ks_all/ke_all ran on the CURRENT stream; the worker
        # streams must not race ahead of it (was a real bug: pipelined runs
        # read garbage cursor/thresh -> all rows wrong; fixed 2026-09-02).
        setup_done = torch.cuda.Event()
        setup_done.record()
        stream_mqa.wait_event(setup_done)
        stream_ext.wait_event(setup_done)

    def tile_fn(ci):
        c0 = c0s[ci]
        c1 = min(c0 + W, L)
        s = torch.cuda.current_stream(dev) if stream_mqa is None else stream_mqa
        with torch.cuda.stream(s):
            tile = scorer(q_fp8, (kv[0][c0:c1], kv[1][c0:c1]), weights,
                          ks_all[ci], ke_all[ci])
        if stream_mqa is not None:
            events[ci].record(stream_mqa)
            stream_ext.wait_event(events[ci])
        return tile, c0, c1

    if stream_mqa is not None:
        return _merge_tiles(tile_fn, len(c0s), q, ks, ke,
                            page_table_size_1, cu_seqlens_q, topk,
                            cand, cursor, thresh, st, stream_ext)
    return _merge_tiles(tile_fn, len(c0s), q, ks, ke, page_table_size_1,
                        cu_seqlens_q, topk, cand, cursor, thresh, st, None)


def streamindex_select_from_logits(
    logits: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    page_table_size_1: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    W: int = 8192,
    stats: bool = False,
) -> torch.Tensor:
    """Select machinery only, over precomputed [q, L] fp32 logits (tiles are
    strided views; no scorer). For benchmarking the partition-merge select
    against production / topk-1pass on identical data."""
    assert logits.dtype == torch.float32 and logits.dim() == 2
    q, L = logits.shape
    cand, cursor, thresh, st = _alloc_state(q, logits.device, stats)
    c0s = list(range(0, L, W))

    def tile_fn(ci):
        c0 = c0s[ci]
        c1 = min(c0 + W, L)
        return logits[:, c0:c1], c0, c1

    dst = _merge_tiles(tile_fn, len(c0s), q, ks, ke, page_table_size_1,
                       cu_seqlens_q, topk, cand, cursor, thresh, st, None)
    if stats:
        return dst, st
    return dst
