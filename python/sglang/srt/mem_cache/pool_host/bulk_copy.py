"""Bulk (batched-memcpy) host<->device KV copy helpers.

The per-row UVA gather/scatter kernels used by the HiCache fallback paths move
656 B rows one at a time and sustain only ~10-25 GB/s over C2C (SuperInfer's
measurement class). These helpers instead coalesce a page-granular index set
into contiguous runs and submit every run as ONE copy-engine segment via
cudaMemcpyBatchAsync (CUDA >= 12.8), which reaches the C2C roofline for
segments >= ~2-8 MB and still beats the kernels at page granularity.

Segment-size curve measured on GH200 (nid010170, sequential memcpyAsync,
512 MB per op): 64 KB -> 7 GB/s, 512 KB -> 55, 1 MB -> 109, 2 MB -> 213/147,
8 MB -> 376/163, 128 MB -> 416/169; H2D ceiling 419, D2H ceiling 170 GB/s.

All copies are byte-for-byte identical to the kernel paths (same source rows
to the same destination slots); only the transport differs.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Iterator, Tuple

import torch

logger = logging.getLogger(__name__)

_CUDA_ERROR_INVALID_VALUE = 1
_CUDA_ERROR_NEWER_DRIVER = 36
_CUDA_ERROR_NOT_SUPPORTED = 801

_MEMLOC_INVALID = 0
_MEMLOC_DEVICE = 1
_MEMLOC_HOST = 2

_SRC_ORDER_STREAM = 1  # cudaMemcpySrcAccessOrderStream

# Staging budget for the H2D bulk load path (tokens are chunked to fit).
BULK_STAGING_BYTES = 512 * 1024 * 1024
# Maximum segments submitted per cudaMemcpyBatchAsync call.
BULK_MAX_SEGMENTS_PER_CALL = 16384
# Runs shorter than this (bytes) go to the fallback kernel instead of the
# batched CE: measured batched-CE rates are ~29-47 GB/s at 32-41 KB segments
# (WORSE than the AOT kernel's 35-47), ~72-75 at 128 KB, 154-160 at ~800 KB,
# and roofline (414 H2D / 170 D2H) at >= 8 MB. 128 KB keeps page-granular
# random runs on the (unchanged) kernel path — no regression — while any
# multi-page joint run rides the CE.
BULK_MIN_SEGMENT_BYTES = 128 * 1024


class _CudaMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CudaMemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_int),
        ("srcLocHint", _CudaMemLocation),
        ("dstLocHint", _CudaMemLocation),
        ("flags", ctypes.c_uint),
    ]


_cudart = None
_batch_fn = None
_batch_unavailable = False


def _get_batch_fn():
    """Resolve cudaMemcpyBatchAsync from libcudart once; None if absent."""
    global _cudart, _batch_fn, _batch_unavailable
    if _batch_unavailable:
        return None
    if _batch_fn is not None:
        return _batch_fn
    if _cudart is None:
        _cudart = ctypes.CDLL("libcudart.so")
    fn = getattr(_cudart, "cudaMemcpyBatchAsync", None)
    if fn is None:
        _batch_unavailable = True
        logger.info(
            "cudaMemcpyBatchAsync not found in libcudart; bulk copies use the "
            "per-run memcpy fallback"
        )
        return None
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # dsts
        ctypes.POINTER(ctypes.c_void_p),  # srcs
        ctypes.POINTER(ctypes.c_size_t),  # sizes
        ctypes.c_size_t,  # count
        ctypes.c_void_p,  # attrs
        ctypes.POINTER(ctypes.c_size_t),  # attrsIdxs
        ctypes.c_size_t,  # numAttrs
        ctypes.c_void_p,  # stream
    ]
    _batch_fn = fn
    return fn


def batched_memcpy_async(
    dst_addrs: torch.Tensor,
    src_addrs: torch.Tensor,
    sizes: torch.Tensor,
    stream,
    *,
    src_is_device: bool,
    dst_is_device: bool,
) -> bool:
    """Submit len(sizes) copies in ONE cudaMemcpyBatchAsync call.

    dst_addrs/src_addrs/sizes: int64 CPU tensors (addresses/bytes). All copies
    share one attribute set (srcAccessOrder=Stream + location hints).
    Returns False if the API is unavailable on this driver (no side effects);
    any other error raises.
    """
    fn = _get_batch_fn()
    n = int(sizes.numel())
    if n == 0:
        return True
    if fn is None:
        return False
    assert dst_addrs.dtype == torch.int64 and src_addrs.dtype == torch.int64
    assert sizes.dtype == torch.int64
    assert dst_addrs.is_cpu and src_addrs.is_cpu and sizes.is_cpu

    device_id = torch.cuda.current_device()
    attrs = _CudaMemcpyAttributes()
    attrs.srcAccessOrder = _SRC_ORDER_STREAM
    if src_is_device:
        attrs.srcLocHint = _CudaMemLocation(_MEMLOC_DEVICE, device_id)
    else:
        attrs.srcLocHint = _CudaMemLocation(_MEMLOC_HOST, 0)
    if dst_is_device:
        attrs.dstLocHint = _CudaMemLocation(_MEMLOC_DEVICE, device_id)
    else:
        attrs.dstLocHint = _CudaMemLocation(_MEMLOC_HOST, 0)
    attrs.flags = 0
    attrs_idxs = (ctypes.c_size_t * 1)(0)

    dsts_ptr = ctypes.cast(int(dst_addrs.data_ptr()), ctypes.POINTER(ctypes.c_void_p))
    srcs_ptr = ctypes.cast(int(src_addrs.data_ptr()), ctypes.POINTER(ctypes.c_void_p))
    sizes_ptr = ctypes.cast(int(sizes.data_ptr()), ctypes.POINTER(ctypes.c_size_t))

    raw_stream = stream.cuda_stream if hasattr(stream, "cuda_stream") else stream
    err = fn(
        dsts_ptr,
        srcs_ptr,
        sizes_ptr,
        ctypes.c_size_t(n),
        ctypes.byref(attrs),
        attrs_idxs,
        ctypes.c_size_t(1),
        ctypes.c_void_p(raw_stream),
    )
    if err != 0:
        if err in (
            _CUDA_ERROR_INVALID_VALUE,
            _CUDA_ERROR_NEWER_DRIVER,
            _CUDA_ERROR_NOT_SUPPORTED,
        ):
            _cudart.cudaGetLastError()
            logger.info(
                "cudaMemcpyBatchAsync unavailable (err=%d); using per-run "
                "memcpy fallback",
                err,
            )
            _batch_unavailable = True
            return False
        raise RuntimeError(f"cudaMemcpyBatchAsync failed with cudaError {err}")
    return True


def find_runs(idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Maximal runs of consecutive values at consecutive positions.

    Returns (start_values, lengths), both int64 CPU tensors. A "run" is a
    maximal slice idx[s:e] with idx[i+1] == idx[i] + 1 for all i in [s, e).
    """
    if idx.numel() == 0:
        empty = torch.empty(0, dtype=torch.int64)
        return empty, empty.clone()
    brk = (torch.diff(idx) != 1).nonzero().flatten() + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), brk])
    ends = torch.cat([brk, torch.full((1,), idx.numel(), dtype=torch.int64)])
    lens = ends - starts
    return idx[starts], lens


def find_joint_runs(
    a: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Maximal runs where BOTH a and b advance by exactly 1 per position.

    Returns (a_starts, b_starts, lengths, positions) where positions[i] is the
    index of the run's first element in the original tensors.
    """
    n = a.numel()
    if n == 0:
        empty = torch.empty(0, dtype=torch.int64)
        return empty, empty.clone(), empty.clone(), empty.clone()
    inside = torch.zeros(n, dtype=torch.bool)
    if n > 1:
        inside[1:] = (torch.diff(a) == 1) & (torch.diff(b) == 1)
    start_mask = torch.zeros(n, dtype=torch.bool)
    start_mask[0] = True
    if n > 1:
        start_mask[1:] = ~inside[1:]
    pos = start_mask.nonzero().flatten()
    ends = torch.cat([pos[1:], torch.full((1,), n, dtype=torch.int64)])
    lens = ends - pos
    return a[pos], b[pos], lens, pos


def chunk_runs(
    starts: torch.Tensor, lens: torch.Tensor, cap: int
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """Greedy chunks of runs with total tokens <= cap (long runs are split).

    Yields (chunk_starts, chunk_lens, chunk_staging_offsets, filled) where
    staging_offsets[i] is the token offset of run i inside the chunk.
    """
    n = lens.numel()
    i = 0
    while i < n:
        s = int(starts[i])
        length = int(lens[i])
        if length > cap:
            # split the overlong run into cap-sized pieces
            n_full = length // cap
            for k in range(n_full):
                yield (
                    torch.tensor([s + k * cap], dtype=torch.int64),
                    torch.tensor([cap], dtype=torch.int64),
                    torch.tensor([0], dtype=torch.int64),
                    cap,
                )
            rem = length - n_full * cap
            if rem > 0:
                yield (
                    torch.tensor([s + n_full * cap], dtype=torch.int64),
                    torch.tensor([rem], dtype=torch.int64),
                    torch.tensor([0], dtype=torch.int64),
                    rem,
                )
            i += 1
            continue
        acc_s, acc_l, acc_o = [], [], []
        total = 0
        while i < n:
            s = int(starts[i])
            length = int(lens[i])
            if length > cap:
                break
            if total + length > cap:
                break
            acc_s.append(s)
            acc_l.append(length)
            acc_o.append(total)
            total += length
            i += 1
        if total > 0:
            yield (
                torch.tensor(acc_s, dtype=torch.int64),
                torch.tensor(acc_l, dtype=torch.int64),
                torch.tensor(acc_o, dtype=torch.int64),
                total,
            )


# Ops smaller than this (tokens) keep the per-row kernel path even when the
# bulk flag is on (the index .cpu() sync costs more than the copy).
BULK_MIN_TOKENS = 64
