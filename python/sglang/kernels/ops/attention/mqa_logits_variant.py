"""Tunable block configuration for DeepGEMM's fp8 MQA logits (prefill indexer).

Lane mqa-tune.  DeepGEMM hardcodes BLOCK_Q = 128/num_heads, BLOCK_KV = 256,
stage pipeline 3x3 and 512 math threads for ``sm90_fp8_mqa_logits`` with no
runtime override.  This module JIT-compiles the *same, unmodified* kernel
template (from the deep_gemm package) with a caller-selected block
configuration (bit-exact; the block layout does not change any output
element's arithmetic) and exposes it with the same calling convention as
``deep_gemm.fp8_mqa_logits(..., clean_logits=False)``: it allocates the padded
logits buffer internally and returns a [seq_len, seq_len_kv] view.

Selected with ``SGLANG_DSA_MQA_LOGITS_VARIANT``:
  * unset / "off" / "none"  -> caller falls back to deep_gemm.fp8_mqa_logits
  * "best"                  -> (4, 192, 3, 5, 384)   [measured +5-7% on GH200]
  * "BQ,BKV,QS,KVS,MT"      -> explicit configuration, e.g. "4,192,3,5,384"
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# Measured best on GH200 (nid010171) at GLM-5.3 prefill shapes
# (q in {2048, 8192} x L in {262144, 524288, 1048576}): ~780 TF/s vs ~735 TF/s
# for the production 4/256/3/3/512 config. See lanes/mqa-tune/results.md.
BEST_CONFIG: Tuple[int, int, int, int, int] = (4, 192, 3, 5, 384)

# Configurations measured bit-exact and spill-free on GH200 (lanes/mqa-tune
# sweep, ptxas-verified register usage). Others are rejected: the register
# file caps usable accumulators at 65536/(128+math_threads) per thread, and
# e.g. BLOCK_Q=8 at 512 math threads compiles but spills catastrophically.
VALID_CONFIGS = {
    (4, 192, 3, 5, 384),  # best
    (4, 192, 3, 3, 384),
    (4, 192, 3, 4, 384),
    (4, 256, 2, 4, 512),
    (4, 256, 3, 3, 512),  # production config (for A/B)
}


@cache_once
def _jit_mqa_logits_variant_module(
    bq: int, bkv: int, qs: int, kvs: int, mt: int, nsm: int
) -> Module:
    args = make_cpp_args(bq, bkv, qs, kvs, mt, nsm)
    return load_jit(
        "mqa_logits_variant",
        *args,
        cuda_files=["dsa/mqa_logits_variant.cuh"],
        cuda_wrappers=[
            (
                "run_mqa_logits_variant",
                f"MQALogitsVariantKernel<{args}>::run",
            )
        ],
        extra_dependencies=["cutlass"],  # provides deep_gemm's include dir
        extra_ldflags=["-lcuda"],
    )


def parse_mqa_logits_variant(raw: str) -> Optional[Tuple[int, int, int, int, int]]:
    """Parse a SGLANG_DSA_MQA_LOGITS_VARIANT value into (bq, bkv, qs, kvs, mt)."""
    raw = (raw or "").strip().lower()
    if raw in ("", "off", "none", "false", "0"):
        return None
    if raw == "best":
        return BEST_CONFIG
    parts = [int(x) for x in raw.replace(";", ",").split(",")]
    if len(parts) != 5:
        raise ValueError(
            f"SGLANG_DSA_MQA_LOGITS_VARIANT must be 'off', 'best' or "
            f"'BQ,BKV,QS,KVS,MT' (5 ints), got {raw!r}"
        )
    bq, bkv, qs, kvs, mt = parts
    cfg = (bq, bkv, qs, kvs, mt)
    if bq * 32 > 256 or bkv != mt // 2 or mt % 128 != 0 or mt > 512 or bkv < 64:
        # Kernel constraints: BLOCK_KV == num_math_threads / 2 (epilogue
        # mapping), math threads multiple of 128 (warpgroups) and at most 512
        # (register file), WGMMA N = BLOCK_Q * 32 <= 256.
        raise ValueError(
            f"invalid mqa logits variant config {raw!r}: need BLOCK_KV == MT/2, "
            f"MT % 128 == 0, MT <= 512, BLOCK_Q*32 <= 256"
        )
    if cfg not in VALID_CONFIGS:
        raise ValueError(
            f"mqa logits variant config {raw!r} was not measured; valid configs: "
            f"{sorted(VALID_CONFIGS)} (see grace-1m lane mqa-tune results.md)"
        )
    return cfg


_VARIANT_CACHE: dict = {"raw": None, "cfg": None}


def get_mqa_logits_variant_config(
    raw: Optional[str] = None,
) -> Optional[Tuple[int, int, int, int, int]]:
    if raw is None:
        return None
    if _VARIANT_CACHE["raw"] != raw:
        _VARIANT_CACHE["raw"] = raw
        _VARIANT_CACHE["cfg"] = parse_mqa_logits_variant(raw)
    return _VARIANT_CACHE["cfg"]


def fp8_mqa_logits_variant(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    num_sms: int,
    bq: int,
    bkv: int,
    qs: int,
    kvs: int,
    mt: int,
) -> torch.Tensor:
    """Drop-in for deep_gemm.fp8_mqa_logits(q, (kv, kv_scales), w, ks, ke,
    clean_logits=False) with a selectable block configuration.

    Returns a [seq_len, seq_len_kv] view of a padded fp32 buffer, exactly like
    the deep_gemm call (row stride is 1024B-aligned and covers
    seq_len_kv + BLOCK_KV of unreduced tail values).
    """
    seq_len, num_heads, head_dim = q.shape
    seq_len_kv = kv.shape[0]
    assert num_heads == 32 and head_dim == 128, "variant supports 32 heads x 128 dims only"

    stride_logits = (seq_len_kv + bkv + 255) // 256 * 256
    aligned_seq_len = (seq_len + bq - 1) // bq * bq
    logits = torch.empty(
        (aligned_seq_len, stride_logits), dtype=torch.float32, device=q.device
    )
    mod = _jit_mqa_logits_variant_module(bq, bkv, qs, kvs, mt, num_sms)
    mod.run_mqa_logits_variant(q, kv, kv_scales, weights, ks, ke, logits)
    return logits[:seq_len, :seq_len_kv]
