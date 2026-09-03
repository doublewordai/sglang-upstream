import ctypes
import ctypes.util
import errno
import logging
import math
import mmap
import os
import re
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
    _libc.madvise.restype = ctypes.c_int
    _libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _libc.prctl.restype = ctypes.c_int
    _libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 5
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
_PR_SET_THP_DISABLE = 41
_PR_GET_THP_DISABLE = 42


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
    # Optional placement hint (v16-memory-plan): cudaHostRegister on hugepage
    # pools is placement-sensitive (rc=1 when the mapping lands inside a
    # CUDA-driver VA reservation); the register-retry loop in pool_host/common
    # sets this to walk the pool through candidate addresses.
    hint = os.environ.get("SGLANG_HUGEPAGE_MMAP_HINT", "")
    addr = ctypes.c_void_p(int(hint, 16)) if hint else None
    flags = (
        mmap.MAP_SHARED
        | mmap.MAP_ANONYMOUS
        | _MAP_POPULATE
        | _MAP_NORESERVE
        | extra_flags
    )
    ptr = None
    if addr is not None:
        noreplace = getattr(mmap, "MAP_FIXED_NOREPLACE", 0x100000)
        ptr = _libc.mmap(
            addr, alloc_bytes, mmap.PROT_READ | mmap.PROT_WRITE, flags | noreplace, -1, 0
        )
        if ptr is None or ptr == _MAP_FAILED:
            if ctypes.get_errno() == 17:  # EEXIST: hint occupied
                ptr = None
            else:
                errno = ctypes.get_errno()
                raise OSError(errno, os.strerror(errno))
    if ptr is None:
        ptr = _libc.mmap(
            None,
            alloc_bytes,
            mmap.PROT_READ | mmap.PROT_WRITE,
            flags,
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


def _prctl_thp_disabled(on: bool) -> int:
    """Set/clear PR_SET_THP_DISABLE; return the PREVIOUS state (0/1).

    Used by the THP mode so an engine running with SGLANG_DISABLE_THP=1 can
    still fault in a THP-backed pool: the process flag gates fault-time THP
    allocation and MADV_COLLAPSE, but not explicit hugetlb mappings.
    """
    prev = int(_libc.prctl(_PR_GET_THP_DISABLE, *([ctypes.c_ulong(0)] * 5)))
    want = 1 if on else 0
    if prev != want:
        rc = _libc.prctl(
            _PR_SET_THP_DISABLE, ctypes.c_ulong(want), *([ctypes.c_ulong(0)] * 4)
        )
        if rc != 0:
            e = ctypes.get_errno()
            raise OSError(e, f"prctl(PR_SET_THP_DISABLE,{on}) {os.strerror(e)}")
    return prev


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
    # enable + fault in THPs (madvise argtypes are declared above; without them
    # a >2 GiB length silently wraps to a C int and the advice never applies).
    # Clear any process-wide PR_SET_THP_DISABLE first (SGLANG_DISABLE_THP),
    # restore it afterwards.
    prev_disabled = _prctl_thp_disabled(False)
    try:
        rc = _libc.madvise(ctypes.c_void_p(ptr), alloc_bytes, _MADV_HUGEPAGE)
        if rc != 0:
            e = ctypes.get_errno()
            _libc.munmap(ctypes.c_void_p(ptr), alloc_bytes)
            raise OSError(e, f"madvise(MADV_HUGEPAGE) {os.strerror(e)}")
        rc = _libc.madvise(ctypes.c_void_p(ptr), alloc_bytes, _MADV_POPULATE_WRITE)
        if rc != 0:
            e = ctypes.get_errno()
            _libc.munmap(ctypes.c_void_p(ptr), alloc_bytes)
            raise OSError(e, f"madvise(MADV_POPULATE_WRITE) {os.strerror(e)}")
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
    finally:
        _prctl_thp_disabled(bool(prev_disabled))
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


# --- Boot-time host-pool NUMA locality check -------------------------------
# lpddr-budget measured a 2.0x decode-gather slowdown (99.5 -> 50.0 GB/s
# wide-copy, 90.6 -> 47.7 GB/s composed) when a host pool lands on a remote
# Grace NUMA node. Placement is decided at first touch: alloc_mmap faults
# every page at mmap time, so locality follows the calling thread's memory
# policy (sglang binds each rank with numactl --cpunodebind=N --membind=N in
# utils/numa_utils.configure_subprocess). This check prints pct_local per
# pool right after population so a placement regression (e.g. binding
# dropped, local node exhausted) cannot pass silently at boot.
_NUMA_LOCALITY_MIN_BYTES = 128 * 1024 * 1024  # only log pool-sized mappings
_NUMA_LOCALITY_WARN_PCT = 99.0


def _parse_cpulist(text: str) -> set:
    """Parse a /sys cpulist ("0-3,7") into a set of ints."""
    out = set()
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def _numa_local_node() -> "tuple[int | None, bool]":
    """(local_node, affinity_contained) from the calling thread's CPU affinity.

    Returns the NUMA node that contains the smallest allowed CPU, and whether
    the whole affinity mask lies inside that node (it does for a properly
    numactl-bound rank). (None, False) on non-NUMA systems or when the
    affinity spans nodes.
    """
    try:
        allowed = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return None, False
    if not allowed:
        return None, False
    first = min(allowed)
    try:
        entries = sorted(os.listdir("/sys/devices/system/node"))
    except OSError:
        return None, False
    node, contained = None, False
    for d in entries:
        if not re.fullmatch(r"node\d+", d):
            continue
        try:
            with open(f"/sys/devices/system/node/{d}/cpulist") as f:
                cpus = _parse_cpulist(f.read())
        except (OSError, ValueError):
            continue
        if cpus and first in cpus:
            node = int(d[4:])
            contained = allowed <= cpus
            break
    return node, contained


def _numa_pages_by_node(ptr: int, n_bytes: int):
    """(bytes_by_node, vma_policies) for populated pages overlapping
    [ptr, ptr+n_bytes), from /proc/self/numa_maps. (None, None) if the
    file is unavailable."""
    byt = {}
    policies = []
    try:
        with open("/proc/self/numa_maps") as f:
            lines = f.readlines()
    except OSError:
        return None, None
    starts = [int(line.split()[0], 16) for line in lines]
    for i, line in enumerate(lines):
        start = starts[i]
        end = starts[i + 1] if i + 1 < len(lines) else start + 1
        if end <= ptr or start >= ptr + n_bytes:
            continue
        toks = line.split()
        pol = toks[1] if len(toks) > 1 else "?"
        if len(toks) > 2 and toks[2].startswith("("):
            pol += " " + toks[2]  # "prefer (many):0-3"
        policies.append(pol)
        ps_kb = None
        for t in toks:
            if t.startswith("kernelpagesize_kB="):
                ps_kb = int(t.split("=", 1)[1])
        if ps_kb is None:
            continue
        for t in toks:
            m = re.fullmatch(r"N(\d+)=(\d+)", t)
            if m:
                byt[int(m.group(1))] = (
                    byt.get(int(m.group(1)), 0) + int(m.group(2)) * ps_kb * 1024
                )
    return byt, policies


class HostPoolPopulationError(RuntimeError):
    """A host-pool mapping is not fully populated: huge pages in it cannot
    be faulted (typically the mempolicy-bound NUMA node is out of free
    memory). The pool is unusable -- pinning or first touch of the missing
    pages fails -- so allocation must abort with the real cause instead of
    surfacing later as cudaErrorInvalidValue (staging-2 C8e, 2026-09-03).
    """


def populated_bytes_in_range(ptr: int, n_bytes: int) -> "int | None":
    """Resident bytes overlapping [ptr, ptr+n_bytes), from /proc/self/numa_maps.

    None when numa_maps cannot be read (callers treat as "cannot verify").
    """
    byt, _ = _numa_pages_by_node(ptr, n_bytes)
    if byt is None:
        return None
    return sum(byt.values())


def _node_free_bytes(node: int) -> "int | None":
    """MemFree of a NUMA node in bytes (best-effort)."""
    try:
        with open(f"/sys/devices/system/node/node{node}/meminfo") as f:
            for line in f:
                if "MemFree:" in line:
                    return int(line.split()[3]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def verify_host_pool_populated(
    ptr: int, n_bytes: int, what: str = "host pool", tol: int = 2 * 1024 * 1024
) -> None:
    """Abort when a MAP_POPULATE'd host pool is not fully resident.

    Measured on Isambard GH200 (kernel 6.4; hugetlb-pin follow-up, 2026-09-03):
    under numactl --membind (MPOL_BIND) a hugetlb surplus fault fails once the
    bound NUMA node runs out of free memory. MAP_POPULATE silently ignores
    those faults, mmap() succeeds with a PARTIALLY-populated mapping, and the
    loss only surfaces later: cudaHostRegister returns cudaErrorInvalidValue
    on the chunk containing the population frontier (its GUP cannot fault the
    missing pages), and a plain first touch of them would SIGBUS. staging-2
    arm C8e died exactly like this: 4 decode ranks/node x 86.9 GiB pools,
    frontiers 21-78 GiB, whole-pool + 16 GiB-chunk registers rc=1.
    """
    populated = populated_bytes_in_range(ptr, n_bytes)
    if populated is None or populated >= n_bytes - tol:
        return
    byt, policies = _numa_pages_by_node(ptr, n_bytes)
    pol = ",".join(policies) if policies else "?"
    free_str = ""
    for n in sorted(byt or {}):
        fb = _node_free_bytes(n)
        if fb is not None:
            free_str += f" N{n} MemFree={fb / 2**30:.1f}GiB"
    raise HostPoolPopulationError(
        f"{what}: only {populated / 2**30:.1f} of {n_bytes / 2**30:.1f} GiB "
        f"could be faulted in (mempolicy={pol};{free_str or ' no per-node free info'})."
        " The mapping is unusable from here: cudaHostRegister of it fails with"
        " cudaErrorInvalidValue at this population frontier and touching the"
        " missing pages would SIGBUS. Free node memory, relax the membind"
        " policy, or shrink the pool (hisparse host_to_device_ratio /"
        " max_total_tokens)."
    )


_NUMA_UNSET = object()
_gpu_numa_node_cache = _NUMA_UNSET


def _gpu_numa_node():
    """NUMA node of the current CUDA device via NVML (best-effort, cached).

    None when CUDA is not initialized, pynvml is unavailable, or the query
    fails (CPU-only hosts, unit tests); callers fall back to the
    affinity-derived node.
    """
    global _gpu_numa_node_cache
    if _gpu_numa_node_cache is not _NUMA_UNSET:
        return _gpu_numa_node_cache
    node = None
    try:
        import torch

        if torch.cuda.is_initialized():
            from sglang.srt.utils.numa_utils import _query_numa_node_for_gpu

            nodes = _query_numa_node_for_gpu(torch.cuda.current_device())
            if nodes:
                node = nodes[0]
    except Exception:
        node = None
    _gpu_numa_node_cache = node
    return node


def log_host_pool_numa_locality(ptr: int, n_bytes: int, name: str = "") -> None:
    """Print pct_local (share of a host pool's populated pages on the
    GPU-local NUMA node) once at allocation; warn below 99%.

    Raise instead of warn when SGLANG_NUMA_LOCALITY_STRICT=1. Silently no-ops
    for mappings below _NUMA_LOCALITY_MIN_BYTES, on non-NUMA hosts, or when
    /proc/self/numa_maps cannot be read.
    """
    if n_bytes < _NUMA_LOCALITY_MIN_BYTES:
        return
    byt, policies = _numa_pages_by_node(ptr, n_bytes)
    if not byt:
        return
    total = sum(byt.values())
    if total == 0:
        return
    by_str = " ".join(f"N{n}={b / 2**30:.1f}GiB" for n, b in sorted(byt.items()))
    pol_str = ",".join(policies) or "?"
    prefix = f"[host-pool numa] {name}: " if name else "[host-pool numa] "
    local, contained = _numa_local_node()
    gpu_node = _gpu_numa_node()
    if gpu_node is not None:
        # the true locality target is the GPU's own node (NVML); the
        # affinity node is only a proxy for it
        target, tgt = gpu_node, f"gpu_node=N{gpu_node}"
    elif local is not None:
        target, tgt = local, f"local_node=N{local}"
    else:
        logger.info(
            "%s%.1f GiB at %#x: no local node determined (affinity spans "
            "nodes and GPU node unavailable); by_node: %s (policy=%s)",
            prefix, total / 2**30, ptr, by_str, pol_str,
        )
        return
    notes = []
    if local is not None and gpu_node is not None and local != gpu_node:
        notes.append(f"rank bound to N{local} but its GPU is on N{gpu_node}")
    if not contained:
        notes.append("affinity spans nodes")
    notes_s = ("; " + "; ".join("WARNING " + n for n in notes)) if notes else ""
    pct = 100.0 * byt.get(target, 0) / total
    line = (
        f"{prefix}{total / 2**30:.1f} GiB at {ptr:#x}: {tgt} "
        f"pct_local={pct:.2f}% by_node: {by_str} (policy={pol_str}{notes_s})"
    )
    if pct >= _NUMA_LOCALITY_WARN_PCT:
        logger.info(line)
        return
    logger.warning(
        "%s -- host pool is NOT NUMA-local; decode gather/swap-in from this "
        "pool runs at up to 2x reduced bandwidth (lpddr-budget: 99.5->50.0 "
        "GB/s wide-copy on a remote node). Check the rank's numactl binding "
        "(SGLANG_NUMA_BIND_V2 / SGLANG_AUTO_NUMA_BIND).", line,
    )
    if envs.SGLANG_NUMA_LOCALITY_STRICT.get():
        raise RuntimeError(line)


def alloc_mmap(dims: tuple, dtype: torch.dtype, name: str = "") -> torch.Tensor:
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
        tensor = torch.frombuffer(array, dtype=dtype, count=math.prod(dims)).reshape(dims)
        tensor._sglang_mmap_alloc_bytes = alloc_bytes
        log_host_pool_numa_locality(tensor.data_ptr(), alloc_bytes, name)
        return tensor

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
                tensor = torch.frombuffer(
                    array, dtype=dtype, count=math.prod(dims)
                ).reshape(dims)
                tensor._sglang_mmap_alloc_bytes = alloc_bytes
                # NUMA-locality line FIRST so the binding is visible even
                # when population fails; then verify MAP_POPULATE completed
                # (it silently stops when a mempolicy-bound NUMA node runs
                # out of memory; a partially-populated pool must fail HERE
                # with the real cause -- see verify_host_pool_populated).
                log_host_pool_numa_locality(tensor.data_ptr(), alloc_bytes, name)
                verify_host_pool_populated(
                    tensor.data_ptr(), alloc_bytes, "hugetlb host pool", tol=page_size
                )
                return tensor
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
    tensor = torch.frombuffer(mm, dtype=dtype, count=math.prod(dims)).reshape(dims)
    tensor._sglang_mmap_alloc_bytes = alloc_bytes
    # Same population check as the hugetlb path: the shared variant swallows
    # MADV_POPULATE_WRITE failures, and a short pool breaks pinning later.
    verify_host_pool_populated(
        tensor.data_ptr(), alloc_bytes, "host pool", tol=mmap.PAGESIZE
    )
    log_host_pool_numa_locality(tensor.data_ptr(), alloc_bytes, name)
    return tensor


def alloc_shm(
    dims: tuple, dtype: torch.dtype, name: str = ""
) -> tuple[torch.Tensor, int, mmap.mmap]:
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
    log_host_pool_numa_locality(tensor.data_ptr(), alloc_bytes, name)
    return tensor, fd, mm
