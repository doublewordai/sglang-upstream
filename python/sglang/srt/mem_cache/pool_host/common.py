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


def _cuda_host_register(buffer: torch.Tensor, reg_bytes: int | None = None) -> None:
    """Pin the host mapping behind ``buffer`` with cudaHostRegister.

    ``reg_bytes`` (when given) is the page-rounded size of the underlying
    mmap -- on 2 MiB hugetlb mappings the driver only accepts sizes that are
    a multiple of the huge page size (or leave < 64 KiB -- one base page --
    uncovered in the final huge page; measured on GH200 driver 590.44.01 /
    KMD 565.57.01, 64 KiB base pages). Registering the exact tensor byte
    count (e.g. 85,146,826,752 for a 1,664,064-token GLM decode pool) fails
    with cudaErrorInvalidValue at every placement; the page-rounded mapping
    size always satisfies the rule. The padding bytes belong to the same
    mapping and are never touched by the pool.
    """
    cudart = torch.cuda.cudart()
    n_bytes = reg_bytes if reg_bytes is not None else buffer.numel() * buffer.element_size()
    rc = cudart.cudaHostRegister(buffer.data_ptr(), n_bytes, 0)
    if int(rc) != 0:
        # A partially-populated mapping cannot be pinned: while pinning, the
        # driver faults the still-missing pages and fails with
        # cudaErrorInvalidValue at the population frontier (hugetlb-pin
        # follow-up, 2026-09-03: staging-2 C8e died on exactly this). Name
        # the real cause instead of a bare CUDA error.
        from sglang.srt.mem_cache.storage.mmap.mmap_allocator import (
            HostPoolPopulationError,
            populated_bytes_in_range,
        )

        populated = populated_bytes_in_range(buffer.data_ptr(), n_bytes)
        if populated is not None and populated < n_bytes - (2 * 1024 * 1024):
            raise HostPoolPopulationError(
                f"cudaHostRegister failed (rc={int(rc)}, "
                f"{cudart.cudaGetErrorString(rc)}) for ptr={buffer.data_ptr():#x} "
                f"size={n_bytes}: the range is only {populated / 2**30:.1f} of "
                f"{n_bytes / 2**30:.1f} GiB populated -- huge pages in it cannot "
                "be faulted (mempolicy-bound NUMA node out of free memory). "
                "See the allocation-time check in mmap_allocator for remediation."
            )
        raise RuntimeError(
            f"cudaHostRegister failed (rc={int(rc)}, "
            f"{cudart.cudaGetErrorString(rc)}) for ptr={buffer.data_ptr():#x} "
            f"size={n_bytes}; host buffer is not pinned and device transfers "
            f"may silently return stale data."
        )


def _cuda_host_unregister(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    # A chunked registration (fallback path) unregisters each chunk start.
    chunks = getattr(buffer, "_sglang_register_chunks", None)
    if chunks:
        for start, _sz in reversed(chunks):
            rc = cudart.cudaHostUnregister(start)
            if int(rc) != 0:
                logger.warning(
                    "cudaHostUnregister failed (rc=%d, %s) for chunk ptr=%#x",
                    int(rc), cudart.cudaGetErrorString(rc), start,
                )
        return
    rc = cudart.cudaHostUnregister(buffer.data_ptr())
    if int(rc) != 0:
        # Best-effort on shutdown: warn, don't raise -- a leak is reclaimed at exit.
        logger.warning(
            "cudaHostUnregister failed (rc=%d, %s) for ptr=%#x",
            int(rc),
            cudart.cudaGetErrorString(rc),
            buffer.data_ptr(),
        )


def _register_chunked(buffer: torch.Tensor, reg_bytes: int, page: int) -> None:
    """Fallback: pin ``reg_bytes`` as page-multiple cudaHostRegister chunks.

    The mapping stays ONE VA range; only the pinning is split. Chunks are
    recorded on the tensor so _cuda_host_unregister can undo them.
    """
    cudart = torch.cuda.cudart()
    ptr = buffer.data_ptr()
    for chunk in (16 * 2**30, 4 * 2**30, 2**30):
        if reg_bytes <= chunk:
            continue
        chunks = []
        off = 0
        while off < reg_bytes:
            sz = min(chunk, reg_bytes - off)
            rc = cudart.cudaHostRegister(ptr + off, sz, 0)
            if int(rc) != 0:
                for start, _ in reversed(chunks):
                    cudart.cudaHostUnregister(start)
                from sglang.srt.mem_cache.storage.mmap.mmap_allocator import (
                    HostPoolPopulationError,
                    populated_bytes_in_range,
                )

                populated = populated_bytes_in_range(ptr, reg_bytes)
                if populated is not None and populated < reg_bytes - (2 * 1024 * 1024):
                    raise HostPoolPopulationError(
                        f"cudaHostRegister chunk failed (rc={int(rc)}, "
                        f"{cudart.cudaGetErrorString(rc)}) at ptr={ptr + off:#x} "
                        f"size={sz} (chunk {chunk >> 30} GiB): the pool is only "
                        f"{populated / 2**30:.1f} of {reg_bytes / 2**30:.1f} GiB "
                        "populated -- this chunk straddles the population "
                        "frontier; huge pages beyond it cannot be faulted "
                        "(mempolicy-bound NUMA node out of free memory)."
                    )
                raise RuntimeError(
                    f"cudaHostRegister chunk failed (rc={int(rc)}, "
                    f"{cudart.cudaGetErrorString(rc)}) at ptr={ptr + off:#x} "
                    f"size={sz} (chunk {chunk >> 30} GiB)"
                )
            chunks.append((ptr + off, sz))
            off += sz
        buffer._sglang_register_chunks = chunks
        return
    raise RuntimeError("_register_chunked: reg_bytes not page-multiple")


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

    Registration size law (measured, hugetlb-pin lane, GH200 driver
    590.44.01 / KMD 565.57.01): on a 2 MiB-hugetlb mapping cudaHostRegister
    returns cudaErrorInvalidValue unless the size is a multiple of the huge
    page size, or leaves < 64 KiB (one base page) uncovered in the final
    huge page. The failure is a pure function of the SIZE (identical
    placements pass/fail with different sizes; misaligned starts are fine),
    which is why the old VA-placement retry ladder could not fix it. The
    allocator page-rounds the mapping, so we register the page-rounded
    mapping size (attached by alloc_mmap as _sglang_mmap_alloc_bytes)
    instead of the tensor's exact byte count; a chunked registration is the
    fallback if a whole-range register still fails.
    """
    from sglang.srt.environ import envs

    import math

    _n_bytes = int(math.prod(dims)) * torch.empty([], dtype=dtype).element_size()
    # Size gate (v16-memory-plan): keep SMALL pools on base pages even when
    # SGLANG_HUGEPAGE_SIZE is set. The register-size fix makes small hugetlb
    # pools pin fine too (verified 8 GiB; the 3.33 GiB indexer pool with
    # padding), so the threshold is now tunable via SGLANG_HUGEPAGE_MIN_BYTES
    # (bytes; default keeps the v16 behavior: only pools >= 32 GiB are
    # hugetlb-backed).
    _hugepage_min_bytes = envs.SGLANG_HUGEPAGE_MIN_BYTES.get()
    _env_set = bool((envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip())
    hugepage_mode = _env_set and _n_bytes >= _hugepage_min_bytes
    if _env_set and not hugepage_mode:
        _saved = os.environ.get("SGLANG_HUGEPAGE_SIZE")
        os.environ.pop("SGLANG_HUGEPAGE_SIZE", None)
        try:
            buffer = allocator.allocate(dims, dtype=dtype, device=device)
            if pin_memory:
                _cuda_host_register(buffer)
            return buffer
        finally:
            os.environ["SGLANG_HUGEPAGE_SIZE"] = _saved

    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if not pin_memory:
        return buffer
    # Page-rounded register size: the mmap behind a hugepage-mode pool is
    # exactly ceil(n/page)*page bytes (alloc_mmap); register that whole
    # mapping so the range end lands on a page boundary. For base/THP pools
    # the attribute is also set (PMD/PAGESIZE rounding) and registering the
    # rounded mapping is equally safe.
    reg_bytes = getattr(buffer, "_sglang_mmap_alloc_bytes", None)
    if reg_bytes is None or reg_bytes < _n_bytes:
        reg_bytes = _n_bytes
    try:
        _cuda_host_register(buffer, reg_bytes)
        return buffer
    except RuntimeError as e:
        from sglang.srt.mem_cache.storage.mmap.mmap_allocator import (
            HostPoolPopulationError,
        )

        if isinstance(e, HostPoolPopulationError) or not hugepage_mode:
            # Partially-populated pools are unfixable by chunking (the pages
            # cannot be faulted at all); surface the capacity cause directly.
            raise
        logger.warning(
            "cudaHostRegister failed on the page-rounded hugepage pool "
            "(size=%d): %s; falling back to chunked registration",
            reg_bytes, str(e)[:160],
        )
    page = 2 * 1024 * 1024
    if (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip().upper() == "1GB":
        page = 1024 * 1024 * 1024
    _register_chunked(buffer, reg_bytes, page)
    return buffer


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
