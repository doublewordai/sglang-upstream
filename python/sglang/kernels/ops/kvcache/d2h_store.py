"""D2H (device -> host) SM-store copy ops.

The copy-engine D2H path is capped at ~170 GB/s over C2C on GH200; warp-
coalesced SM stores reach 381-384 GB/s into the same pinned pool (measured,
lane d2h-stores, nid010161 GPU0, 2026-09-03). These JIT kernels provide the
winning pattern for the hicache/hisparse bulk backup paths.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)


@cache_once
def jit_d2h_store_module(*, item_size: int, num_layers: int) -> Module:
    """Segment store is item-agnostic; rows store is specialized on both."""
    args = make_cpp_args(item_size, num_layers)
    return load_jit(
        "d2h_store",
        *args,
        cuda_files=[
            "kvcacheio/d2h_store.cuh",
        ],
        cuda_wrappers=[
            ("launch_seg_store", "&sglang::d2h_store::D2HSegStoreKernel::run"),
            (
                "launch_rows_store",
                f"&sglang::d2h_store::D2HRowsStoreKernel<{args}>::run",
            ),
        ],
    )


def can_use_d2h_sm_store(*, item_size: int, num_layers: int) -> bool:
    if item_size % 16 != 0 or item_size <= 0 or num_layers <= 0:
        return False
    try:
        jit_d2h_store_module(item_size=item_size, num_layers=num_layers)
        return True
    except Exception as e:
        logger.warning(f"Failed to load D2H SM-store JIT kernel: {e}")
        return False
