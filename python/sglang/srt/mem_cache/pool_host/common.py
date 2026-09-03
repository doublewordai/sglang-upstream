from __future__ import annotations

import json
import logging
import os
from collections import defaultdict

import torch

from sglang.srt.mem_cache.storage.mmap import alloc_mmap

logger = logging.getLogger(__name__)


class HostTensorAllocator:
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"HostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        return alloc_mmap(dims, dtype)


class ShmHostTensorAllocator(HostTensorAllocator):
    def __init__(self):
        super().__init__()
        self.fds = []
        self.mms = []

    @property
    def fd(self):
        return self.fds[0] if self.fds else None

    @property
    def mm(self):
        return self.mms[0] if self.mms else None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"ShmHostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        from sglang.srt.mem_cache.storage.mmap import alloc_shm

        tensor, fd, mm = alloc_shm(dims, dtype)
        self.fds.append(fd)
        self.mms.append(mm)
        return tensor

    def __del__(self):
        for fd in getattr(self, "fds", []):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.fds = []


def get_allocator_from_storage(allocator_type):
    if allocator_type == "mooncake":
        try:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeHostTensorAllocator,
            )

            return MooncakeHostTensorAllocator()
        except ImportError:
            logger.warning(
                "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
                "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
                "Fallback to use default allocator."
            )
            return HostTensorAllocator()
    elif allocator_type == "mori":
        try:
            from sglang.srt.mem_cache.storage.umbp.umbp_host_allocator import (
                UMBPHostTensorAllocator,
            )

            return UMBPHostTensorAllocator()
        except (ImportError, RuntimeError) as exc:
            logger.warning(
                "UMBPHostTensorAllocator unavailable (%s). "
                "Falling back to torch.empty-based allocator.",
                exc,
            )
            return HostTensorAllocator()
    elif allocator_type == "shm":
        return ShmHostTensorAllocator()
    else:
        return HostTensorAllocator()


def get_allocator_type(server_args) -> str:
    backend = getattr(server_args, "hicache_storage_backend", None)
    if backend == "shm":
        return "shm"
    if backend == "dynamic":
        extra_config_str = getattr(
            server_args, "hicache_storage_backend_extra_config", None
        )
        if extra_config_str:
            try:
                config = json.loads(extra_config_str)
                if config.get("allocator") == "shm":
                    return "shm"
            except Exception:
                pass
    return backend or "default"


def _cuda_host_register(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    n_bytes = buffer.numel() * buffer.element_size()
    rc = cudart.cudaHostRegister(buffer.data_ptr(), n_bytes, 0)
    if int(rc) != 0:
        raise RuntimeError(
            f"cudaHostRegister failed (rc={int(rc)}, "
            f"{cudart.cudaGetErrorString(rc)}) for ptr={buffer.data_ptr():#x} "
            f"size={n_bytes}; host buffer is not pinned and device transfers "
            f"may silently return stale data."
        )


def _cuda_host_unregister(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    rc = cudart.cudaHostUnregister(buffer.data_ptr())
    if int(rc) != 0:
        # Best-effort on shutdown: warn, don't raise -- a leak is reclaimed at exit.
        logger.warning(
            "cudaHostUnregister failed (rc=%d, %s) for ptr=%#x",
            int(rc),
            cudart.cudaGetErrorString(rc),
            buffer.data_ptr(),
        )


_REGISTER_RETRY_SPACERS: list = []


def alloc_with_host_register(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.

    cudaHostRegister on hugepage-backed pools fails intermittently with
    cudaErrorInvalidValue depending on where the kernel placed the mapping
    (v16-memory-plan rig: identical boots fail/pass; a fixed VA fails
    deterministically). When a hugepage mode is active, retry the whole
    alloc+register with a small VA spacer between attempts so the kernel
    hands out a different placement each try.
    """
    import gc
    import mmap as _mmap

    from sglang.srt.environ import envs

    import math

    _n_bytes = int(math.prod(dims)) * torch.empty([], dtype=dtype).element_size()
    # Size gate (v16-memory-plan): cudaHostRegister on SMALL hugetlb pools
    # (< ~32 GiB) fails deterministically at every placement tested (3.33 GB
    # indexer pool x8 addresses, 8.19 GB x19); only the large decode host pool
    # registers reliably. Small pools stay on base pages.
    _SMALL_HUGEPAGE_LIMIT = 32 * (1 << 30)
    hugepage_mode = (
        bool((envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip())
        and _n_bytes >= _SMALL_HUGEPAGE_LIMIT
    )
    if not hugepage_mode and (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip():
        _saved = os.environ.get("SGLANG_HUGEPAGE_SIZE")
        os.environ.pop("SGLANG_HUGEPAGE_SIZE", None)
        try:
            buffer = allocator.allocate(dims, dtype=dtype, device=device)
            if pin_memory:
                _cuda_host_register(buffer)
            return buffer
        finally:
            os.environ["SGLANG_HUGEPAGE_SIZE"] = _saved
    attempts = 8 if hugepage_mode else 1
    # Placement ladder: attempt 0 kernel-chosen; then fixed hints descending
    # through the low VA region where registrations empirically succeed
    # (at/below the CUDA device arena); far-below fallbacks last.
    ladder = [
        None,
        0x400E00000000,
        0x400A00000000,
        0x400600000000,
        0x400400000000,
        0x300000000000,
        0x200000000000,
        0x10000000000,
    ]
    for i in range(attempts):
        hint = ladder[i % len(ladder)]
        if hint is not None:
            os.environ["SGLANG_HUGEPAGE_MMAP_HINT"] = hex(hint)
        else:
            os.environ.pop("SGLANG_HUGEPAGE_MMAP_HINT", None)
        try:
            buffer = allocator.allocate(dims, dtype=dtype, device=device)
        finally:
            os.environ.pop("SGLANG_HUGEPAGE_MMAP_HINT", None)
        if not pin_memory:
            return buffer
        try:
            _cuda_host_register(buffer)
            return buffer
        except RuntimeError as e:
            if i == attempts - 1:
                raise
            logger.warning(
                "cudaHostRegister failed on hugepage pool (attempt %d/%d): %s; "
                "retrying with a VA spacer",
                i + 1,
                attempts,
                str(e)[:120],
            )
            del buffer
            gc.collect()
            # Shift the kernel's mmap placement: a small anonymous mapping
            # takes the top of the freed hole, so the next pool mapping
            # lands at a different address.
            _REGISTER_RETRY_SPACERS.append(_mmap.mmap(-1, (2 + i * 4) * 1024 * 1024))
    raise RuntimeError("unreachable")


def alloc_with_pin_memory(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
    },
)
