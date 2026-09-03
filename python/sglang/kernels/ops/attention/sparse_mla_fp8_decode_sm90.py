"""JIT-compiled native-fp8 sparse MLA DECODE attention kernel for SM90 (GH200).

Consumes the production fp8 KV rows (656 B: fp8 nope 512 | 4 x fp32 group
scales | bf16 rope 64) directly with fp8 WGMMA, keeping the stored per-group
scales exact (descaled on the QK accumulators; folded into the P matrix for
PV). Split-KV across the top-2048 rows + a combine kernel. See the lane
SPEC.md for the numerics contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _sdk_cuda_flags() -> list[str]:
    # Same flag set as sparse_mla_q8kv8_prefill_sm90 (verified by that lane's
    # per-flag ablation on SM90; --use_fast_math only affects the exp2f path).
    return [
        "-O3",
        "-DNDEBUG",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "--use_fast_math",
        "-Xptxas=-v",
        "-lineinfo",
    ]


@cache_once
def _jit_sparse_mla_fp8_decode_module() -> Module:
    return load_jit(
        "sparse_mla_fp8_decode_sm90",
        cuda_files=[
            "sparse_mla_fp8_decode_sm90/entry.cuh",
        ],
        cuda_wrappers=[
            ("dispatch", "sparse_mla_fp8_decode_dispatch"),
            ("combine", "sparse_mla_fp8_decode_combine"),
            ("qprep", "sparse_mla_fp8_qprep_dispatch"),
        ],
        extra_cuda_cflags=_sdk_cuda_flags(),
        extra_dependencies=["cutlass"],
    )


_resolved_entries: tuple | None = None


def _get_entries() -> tuple:
    global _resolved_entries
    if _resolved_entries is None:
        m = _jit_sparse_mla_fp8_decode_module()
        _resolved_entries = (
            m["dispatch"],
            m["combine"],
            m["qprep"],
        )
    return _resolved_entries


_get_current_stream_raw = torch._C._cuda_getCurrentRawStream


def sparse_mla_fp8_decode_fwd(
    q: torch.Tensor,            # [b, 64, 576] bf16 (contiguous)
    kv: torch.Tensor,           # [rows, 656] uint8 view of the DSA pool
    indices: torch.Tensor,      # [b, topk] int32 (negative = masked)
    seqlens: torch.Tensor,      # [b] int32 (scheduler hint; see tail_sentinel)
    sm_scale: float,
    num_splits: int = 16,
    tail_sentinel: bool = True,
    *,
    partial_o: torch.Tensor | None = None,   # [b, P, 64, 512] f32
    partial_ml: torch.Tensor | None = None,  # [b, P, 64, 2] f32
    out: torch.Tensor | None = None,         # [b, 64, 512] bf16
    debug: torch.Tensor | None = None,       # [b, 12] f32 (E pre-pass dump)
    qprep: tuple | None = None,              # (q_fp8, q_rope, q_scale) pre-quantized
    fused: bool = False,                     # single-launch in-kernel combine (caps P)
    counter: torch.Tensor | None = None,     # [1] i32 barrier counter (reused across calls)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (out, partial_o, partial_ml)."""
    assert q.dtype == torch.bfloat16 and q.is_contiguous()
    assert q.shape[1] == 64 and q.shape[2] == 576
    b = q.shape[0]
    topk = indices.shape[1]
    assert indices.dtype == torch.int32 and indices.is_contiguous()
    assert kv.dtype == torch.uint8 and kv.is_contiguous() and kv.shape[1] == 656
    assert seqlens.dtype == torch.int32 and seqlens.shape[0] == b

    dev = q.device
    dispatch_fn, combine_fn, qprep_fn = _get_entries()
    stream = _get_current_stream_raw(dev.index or 0)

    if fused:
        # co-residency cap: all b*P CTAs must be resident (1 CTA/SM here)
        num_splits = min(num_splits, 32, 132 // b)
        if qprep is None:
            q_fp8 = torch.empty((b, 64, 512), dtype=torch.uint8, device=dev)
            q_rope = torch.empty((b, 64, 64), dtype=torch.bfloat16, device=dev)
            q_scale = torch.empty((b, 64), dtype=torch.float32, device=dev)
        else:
            q_fp8, q_rope, q_scale = qprep
        if partial_o is None:
            partial_o = torch.empty((b, num_splits, 64, 512), dtype=torch.float32, device=dev)
        if partial_ml is None:
            partial_ml = torch.empty((b, num_splits, 64, 2), dtype=torch.float32, device=dev)
        if out is None:
            out = torch.empty((b, 64, 512), dtype=torch.bfloat16, device=dev)
        if counter is None:
            counter = torch.zeros(1, dtype=torch.int32, device=dev)
        debug = torch.empty(0, dtype=torch.float32, device=dev)
        qprep_fn(q, q_fp8, q_rope, q_scale, counter, b, 1.0, stream)
        dispatch_fn(
            q, kv, indices, seqlens, partial_o, partial_ml, debug, q_fp8, q_rope, q_scale,
            counter, out, num_splits, topk, 1 if tail_sentinel else 0, 1, sm_scale, stream,
        )
        return out, partial_o, partial_ml

    if partial_o is None:
        partial_o = torch.empty((b, num_splits, 64, 512), dtype=torch.float32, device=dev)
    if partial_ml is None:
        partial_ml = torch.empty((b, num_splits, 64, 2), dtype=torch.float32, device=dev)
    if out is None:
        out = torch.empty((b, 64, 512), dtype=torch.bfloat16, device=dev)
    if debug is None:
        debug = torch.empty(0, dtype=torch.float32, device=dev)
    if qprep is None:
        q_fp8 = torch.empty(0, dtype=torch.uint8, device=dev)
        q_rope = torch.empty(0, dtype=torch.bfloat16, device=dev)
        q_scale = torch.empty(0, dtype=torch.float32, device=dev)
    else:
        q_fp8, q_rope, q_scale = qprep
    dispatch_fn(
        q, kv, indices, seqlens, partial_o, partial_ml, debug, q_fp8, q_rope, q_scale,
        torch.empty(0, dtype=torch.int32, device=dev),
        torch.empty(0, dtype=torch.bfloat16, device=dev),
        num_splits, topk, 1 if tail_sentinel else 0, 0, sm_scale, stream,
    )
    combine_fn(partial_o, partial_ml, out, num_splits, stream)
    return out, partial_o, partial_ml


def sparse_mla_fp8_qprep(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-quantize q [b, 64, 576] bf16 -> (q_fp8 [b,64,512] u8, q_rope [b,64,64]
    bf16, s_q [b,64] f32). Pass the tuple as qprep= to the fwd to skip
    per-CTA re-quantization."""
    assert q.dtype == torch.bfloat16 and q.is_contiguous()
    b = q.shape[0]
    dev = q.device
    q_fp8 = torch.empty((b, 64, 512), dtype=torch.uint8, device=dev)
    q_rope = torch.empty((b, 64, 64), dtype=torch.bfloat16, device=dev)
    q_scale = torch.empty((b, 64), dtype=torch.float32, device=dev)
    _, _, qprep_fn = _get_entries()
    counter = torch.zeros(1, dtype=torch.int32, device=dev)
    qprep_fn(q, q_fp8, q_rope, q_scale, counter, b, 1.0, _get_current_stream_raw(dev.index or 0))
    return q_fp8, q_rope, q_scale
