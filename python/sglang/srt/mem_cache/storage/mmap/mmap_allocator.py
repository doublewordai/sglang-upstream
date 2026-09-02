import ctypes
import ctypes.util
import errno
import logging
import math
import mmap
import os
import uuid
import weakref

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Load libc once at module level so munmap is callable safely at GC/shutdown time.
# Resolve the SONAME via find_library so the allocator also works on systems
# whose libc is not named "libc.so.6" (e.g. musl / Alpine).
try:
    _libc_name = ctypes.util.find_library("c") or "libc.so.6"
    _libc = ctypes.CDLL(_libc_name, use_errno=True)
    _libc.mmap.restype = ctypes.c_void_p
    _libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    _libc.munmap.restype = ctypes.c_int
    _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
except OSError:
    _libc = None

# MAP_POPULATE is in Python's mmap module only since 3.11.
_MAP_POPULATE = getattr(mmap, "MAP_POPULATE", 0x08000)
# MAP_HUGETLB and MAP_HUGE_* are Linux-specific and not in Python's mmap module.
_MAP_HUGETLB = 0x40000
_MAP_NORESERVE = 0x4000
_MAP_HUGE_2MB = 21 << 26  # 0x1400000
_MAP_HUGE_1GB = 30 << 26  # 0x78000000
_MAP_FAILED = ctypes.c_void_p(-1).value
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
_MADV_NOHUGEPAGE = getattr(mmap, "MADV_NOHUGEPAGE", 15)
_MADV_HUGEPAGE = 14
# MADV_COLLAPSE: kernel 6.1+; synchronously collapse a range to PMD THPs.
_MADV_COLLAPSE = 25


def _thp_anon_huge_kb(ptr: int) -> int:
    """AnonHugePages (kB) of the smaps VMA containing ptr (0 if not found)."""
    import re

    tgt = False
    try:
        with open("/proc/self/smaps") as f:
            for line in f:
                m = re.match(r"^([0-9a-f]+)-([0-9a-f]+) ", line)
                if m:
                    tgt = int(m.group(1), 16) <= ptr < int(m.group(2), 16)
                    continue
                if tgt and line.startswith("AnonHugePages:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _alloc_hugepage(n_bytes: int, alloc_bytes: int, extra_flags: int) -> ctypes.Array:
    """Call mmap via libc with hugepage flags and return an owning ctypes array.

    MAP_NORESERVE is required on hosts whose hugetlb pool is empty
    (nr_hugepages=0): reservation-based mmaps fail with ENOMEM, while
    NORESERVE faults allocate "surplus" huge pages from the buddy allocator
    up to nr_overcommit_hugepages (verified on Isambard's 6.4 64k-page
    kernel: 1 GiB -> 512 surplus 2 MiB pages, HugePages_Surp 0->512).
    MAP_POPULATE forces every fault at mmap time, so exhaustion surfaces as
    a clean ENOMEM here instead of a later SIGBUS.

    munmap fires automatically via weakref.finalize when the array is
    garbage-collected (i.e. when the tensor that wraps it is freed).
    """
    ptr = _libc.mmap(
        None,
        alloc_bytes,
        mmap.PROT_READ | mmap.PROT_WRITE,
        mmap.MAP_SHARED
        | mmap.MAP_ANONYMOUS
        | _MAP_POPULATE
        | _MAP_NORESERVE
        | extra_flags,
        -1,
        0,
    )
    if ptr is None or ptr == _MAP_FAILED:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    array = (ctypes.c_uint8 * n_bytes).from_address(ptr)
    weakref.finalize(array, _libc.munmap, ctypes.c_void_p(ptr), alloc_bytes)
    return array


def _alloc_plain(alloc_bytes: int) -> mmap.mmap:
    """Anonymous page-size mapping, populated before it is returned.

    With SGLANG_MAP_HOST_POOL_PRIVATE the pages are private anonymous ones with
    huge pages off: once pinned, memory compaction skips them cheaply, whereas
    pinned MAP_SHARED pages are isolated, unmapped and remapped on every failed
    migration attempt, which stalls device access through the pinned mapping.
    """
    if envs.SGLANG_MAP_HOST_POOL_PRIVATE.get():
        mm = mmap.mmap(
            -1,
            alloc_bytes,
            flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        mm.madvise(_MADV_NOHUGEPAGE)
        mm.madvise(_MADV_POPULATE_WRITE)
        return mm
    mm = mmap.mmap(
        -1,
        alloc_bytes,
        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | _MAP_POPULATE,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    try:
        # MADV_POPULATE_WRITE guarantees pages are populated and writable,
        # throwing an error on failure (e.g. out of memory).
        mm.madvise(_MADV_POPULATE_WRITE)
    except OSError:
        # Fall back to MAP_POPULATE if MADV_POPULATE_WRITE is not supported (<5.14 kernel).
        pass
    return mm


def _thp_pmd_size() -> int:
    """PMD (transparent huge page) size in bytes; 2 MiB fallback."""
    try:
        return int(open("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size").read())
    except (OSError, ValueError):
        return 2 * 1024 * 1024


def _mmap_libc(size: int, hint: int, flags: int) -> int:
    """Raw libc mmap; returns the address (raises OSError on failure)."""
    ptr = _libc.mmap(
        ctypes.c_void_p(hint) if hint else None,
        size,
        mmap.PROT_READ | mmap.PROT_WRITE,
        mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | flags,
        -1,
        0,
    )
    if ptr is None or ptr == _MAP_FAILED:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e))
    return ptr


def _alloc_thp(n_bytes: int, alloc_bytes: int, strict: bool) -> ctypes.Array:
    """Private anonymous mapping backed by PMD transparent huge pages.

    THP is the fallback when the hugetlb pool cannot be reserved (no root):
    on the 6.4 64k-page Isambard kernel THP is enabled=always/defrag=madvise
    with a 512 MiB PMD size, so MADV_HUGEPAGE + MADV_POPULATE_WRITE faults in
    512 MiB pages directly and MADV_COLLAPSE picks up any stragglers. Like
    hugetlbfs pages, THP pages are not LRU-compactable, so the kcompactd
    pinned-page storm fixed by SGLANG_MAP_HOST_POOL_PRIVATE cannot recur.

    The mapping is forced to start on a PMD boundary (MAP_FIXED_NOREPLACE
    probe ladder, else kernel-chosen address trimmed head+tail): a
    page-aligned start can lose up to one whole PMD (25% of a 2 GiB pool) to
    the alignment gap, which otherwise silently caps THP coverage.
    """
    pmd = _thp_pmd_size()
    ptr = None
    if _libc is not None:
        noreplace = getattr(mmap, "MAP_FIXED_NOREPLACE", 0x100000)
        for base in (0x10000000000, 0x400000000000, 0x800000000000):
            hint = base
            for _ in range(64):
                try:
                    ptr = _mmap_libc(alloc_bytes, hint, noreplace)
                    break
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        ptr = None
                        break
                    hint += pmd
            if ptr is not None:
                break
    if ptr is None:
        if _libc is not None:
            over = alloc_bytes + pmd
            raw = _mmap_libc(over, 0, 0)
            aligned = (raw + pmd - 1) & ~(pmd - 1)
            if aligned != raw:
                _libc.munmap(ctypes.c_void_p(raw), aligned - raw)
            tail = raw + over - (aligned + alloc_bytes)
            if tail > 0:
                _libc.munmap(ctypes.c_void_p(aligned + alloc_bytes), tail)
            ptr = aligned
        else:
            mm = mmap.mmap(
                -1, alloc_bytes,
                flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            ptr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    # enable + fault in THPs
    _libc.madvise(ctypes.c_void_p(ptr), alloc_bytes, _MADV_HUGEPAGE)
    _libc.madvise(ctypes.c_void_p(ptr), alloc_bytes, _MADV_POPULATE_WRITE)
    try:
        _libc.madvise(ctypes.c_void_p(ptr), alloc_bytes, _MADV_COLLAPSE)
    except OSError as e:  # <6.1 kernels: coverage check below still applies
        logger.warning("MADV_COLLAPSE unavailable (%s); THP coverage may be partial", e)
    huge_kb = _thp_anon_huge_kb(ptr)
    coverage = huge_kb * 1024 / alloc_bytes
    if strict and coverage < 0.98:
        _libc.munmap(ctypes.c_void_p(ptr), alloc_bytes)
        raise RuntimeError(
            f"SGLANG_HUGEPAGE_SIZE=THP with SGLANG_HUGEPAGE_STRICT=1 but only "
            f"{coverage:.1%} of {alloc_bytes >> 20} MiB is THP-backed "
            f"(AnonHugePages={huge_kb} kB); refusing to fall back silently."
        )
    if coverage < 0.5:
        logger.warning(
            "THP host pool only %.1f%% hugepage-backed (AnonHugePages=%d kB of %d MiB)",
            coverage * 100, huge_kb, alloc_bytes >> 20,
        )
    else:
        logger.info(
            "THP host pool %.1f%% hugepage-backed (AnonHugePages=%d kB of %d MiB)",
            coverage * 100, huge_kb, alloc_bytes >> 20,
        )
    array = (ctypes.c_uint8 * n_bytes).from_address(ptr)
    weakref.finalize(array, _libc.munmap, ctypes.c_void_p(ptr), alloc_bytes)
    return array


def _mm_ptr(mm: mmap.mmap) -> int:
    """Address of the mmap via a ctypes buffer view (mmap exposes no ptr)."""
    return ctypes.addressof(ctypes.c_char.from_buffer(mm))


def alloc_mmap(dims: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Allocate a host tensor via anonymous mmap.

    SGLANG_HUGEPAGE_SIZE selects the page backing:
      ""    : base pages (MAP_SHARED anon, or MAP_PRIVATE + MADV_NOHUGEPAGE
              when SGLANG_MAP_HOST_POOL_PRIVATE=1 -- the production default)
      "2MB"/"1GB": hugetlbfs via MAP_HUGETLB|MAP_NORESERVE (works with an
              empty hugetlb pool through surplus/overcommit pages)
      "THP": MAP_PRIVATE anon + MADV_HUGEPAGE + MADV_POPULATE_WRITE +
              MADV_COLLAPSE (PMD transparent huge pages, no root needed)
    With SGLANG_HUGEPAGE_STRICT=1 a requested hugepage mode that cannot be
    satisfied raises instead of silently falling back to base pages.

    MAP_POPULATE (hugetlb) / MADV_POPULATE_WRITE (anon) are required so
    cudaHostRegister pins real, pre-faulted physical pages (otherwise pinning
    can race with COW or page faults and the device ends up reading stale
    data).

    The tensor owns the mapping; munmap fires when the tensor is freed.
    """
    # Re-read per call (not cached) so that envs.SGLANG_HUGEPAGE_SIZE.override()
    # works correctly in tests.
    hugepage_size = (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip().upper()
    strict = envs.SGLANG_HUGEPAGE_STRICT.get()
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()

    if hugepage_size == "THP":
        page_size = _thp_pmd_size()  # PMD multiple so coverage is not tail-capped
        alloc_bytes = math.ceil(n_bytes / page_size) * page_size
        array = _alloc_thp(n_bytes, alloc_bytes, strict)
        return torch.frombuffer(array, dtype=dtype, count=math.prod(dims)).reshape(dims)

    if hugepage_size == "":
        page_size, extra_flags = mmap.PAGESIZE, 0
    elif hugepage_size == "2MB":
        page_size, extra_flags = 2 * 1024 * 1024, _MAP_HUGETLB | _MAP_HUGE_2MB
    elif hugepage_size == "1GB":
        page_size, extra_flags = 1024 * 1024 * 1024, _MAP_HUGETLB | _MAP_HUGE_1GB
    else:
        msg = (
            f"Unrecognized SGLANG_HUGEPAGE_SIZE={envs.SGLANG_HUGEPAGE_SIZE.get()!r}; "
            "expected '', '2MB', '1GB' or 'THP'."
        )
        if strict:
            raise ValueError(msg)
        logger.warning("%s Falling back to plain page-size mmap.", msg)
        page_size, extra_flags = mmap.PAGESIZE, 0

    alloc_bytes = math.ceil(n_bytes / page_size) * page_size

    if extra_flags:
        if _libc is None:
            msg = (
                "Hugepage mmap requested but libc could not be loaded; "
                f"SGLANG_HUGEPAGE_SIZE={hugepage_size} is ignored."
            )
            if strict:
                raise RuntimeError(msg)
            logger.error(msg)
        else:
            try:
                array = _alloc_hugepage(n_bytes, alloc_bytes, extra_flags)
                return torch.frombuffer(
                    array, dtype=dtype, count=math.prod(dims)
                ).reshape(dims)
            except OSError as e:
                msg = (
                    f"Hugepage mmap via libc failed ({e}); "
                    f"SGLANG_HUGEPAGE_SIZE={hugepage_size} is ignored."
                )
                if strict:
                    raise
                logger.error(msg)
        alloc_bytes = math.ceil(n_bytes / mmap.PAGESIZE) * mmap.PAGESIZE

    # Plain mmap path -- used directly when no hugepages requested, or as fallback.
    # torch.frombuffer keeps a reference to mm inside the tensor storage, so mm
    # stays alive until the tensor is freed and mmap.mmap.__del__ calls munmap.
    mm = _alloc_plain(alloc_bytes)
    return torch.frombuffer(mm, dtype=dtype, count=math.prod(dims)).reshape(dims)


def alloc_shm(dims: tuple, dtype: torch.dtype) -> tuple[torch.Tensor, int, mmap.mmap]:
    """Allocate a host tensor via shared memory (/dev/shm).

    Returns a tuple of (tensor, fd, mm).
    The caller is responsible for keeping the fd open if they need to share it,
    and closing it when they are done.
    """
    hugepage_size = (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip().upper()
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()

    # Note: hugepages are not directly supported with /dev/shm mmap files
    # without mounting hugetlbfs there, so we fall back to plain page size.
    if hugepage_size != "":
        logger.warning(
            "Hugepages are not supported with SHM allocator. "
            "Falling back to plain page-size mmap."
        )

    page_size = mmap.PAGESIZE
    alloc_bytes = math.ceil(n_bytes / page_size) * page_size

    # Create an anonymous shared memory file descriptor via memfd_create
    fd = None
    try:
        # MFD_CLOEXEC is standard on Linux 3.17+
        fd = os.memfd_create(
            f"sglang_host_pool_{uuid.uuid4().hex}",
            flags=getattr(os, "MFD_CLOEXEC", 1),
        )
    except (AttributeError, OSError):
        # Fallback to creating a file in /dev/shm if memfd_create is not supported
        shm_path = f"/dev/shm/sglang_host_pool_{uuid.uuid4().hex}.mmap"
        try:
            fd = os.open(shm_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
            try:
                os.unlink(shm_path)
            except OSError:
                pass
        except Exception as e:
            raise OSError(f"Failed to create shm file: {e}")

    try:
        os.ftruncate(fd, alloc_bytes)
        mm = mmap.mmap(
            fd,
            alloc_bytes,
            flags=mmap.MAP_SHARED | _MAP_POPULATE,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        try:
            # MADV_POPULATE_WRITE guarantees pages are populated and writable,
            # throwing an error on failure (e.g. out of memory).
            mm.madvise(_MADV_POPULATE_WRITE)
        except OSError:
            # Fall back to MAP_POPULATE if MADV_POPULATE_WRITE is not supported (<5.14 kernel).
            pass
    except Exception as e:
        if fd is not None:
            os.close(fd)
        raise e

    tensor = torch.frombuffer(mm, dtype=dtype, count=math.prod(dims)).reshape(dims)
    return tensor, fd, mm
