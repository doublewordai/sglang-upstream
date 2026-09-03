# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Unit tests for the SGLANG_HUGEPAGE_SIZE host-pool page-backing modes.

Covers the mmap_allocator paths on kernels where they are available:
  base (""), THP ("THP"), hugetlbfs 2 MiB ("2MB"), plus SGLANG_HUGEPAGE_STRICT
  error handling. Each successful mode must produce a writable tensor of the
  requested shape whose bytes round-trip bit-exactly, and (when a GPU is
  present) survive cudaHostRegister + H2D/D2H round-trips unchanged.

Skips (not fails) modes the host kernel cannot provide (e.g. an empty
hugetlb pool without surplus/overcommit support, or THP disabled), so the
test is runnable on any host; page backing is ASSERTED from /proc/self/smaps
for every mode that does run, i.e. a silent fallback to base pages fails.

Run:  python -m pytest test/test_host_pool_pages.py -x -q
"""

import ctypes
import mmap
import re
import sys
from contextlib import contextmanager

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mmap import alloc_mmap

try:
    import pytest
except ImportError:  # frozen prod venvs have no pytest: run standalone
    class _Skip(Exception):
        pass

    class _PytestShim:
        @staticmethod
        def skip(msg):
            raise _Skip(msg)

        @staticmethod
        @contextmanager
        def raises(exc):
            @contextmanager
            def _cm():
                try:
                    yield
                except exc:
                    return
                raise AssertionError(f"expected {exc.__name__}")
            with _cm():
                yield

    pytest = _PytestShim()

GIB = 2**30
MIB = 2**20

_MADV_COLLAPSE = 25


def _smaps_entry(ptr: int) -> dict:
    out, tgt = {}, False
    with open("/proc/self/smaps") as f:
        for line in f:
            m = re.match(r"^([0-9a-f]+)-([0-9a-f]+) ", line)
            if m:
                tgt = int(m.group(1), 16) <= ptr < int(m.group(2), 16)
                if tgt:
                    out["start"] = int(m.group(1), 16)
                continue
            if tgt:
                for k in (
                    "AnonHugePages:",
                    "KernelPageSize:",
                    "MMUPageSize:",
                    "VmFlags:",
                ):
                    if line.startswith(k):
                        out[k] = line.split(None, 1)[1].strip()
        # fallthrough: VmFlags is near the end of each entry
    return out


def _ptr_of(t: torch.Tensor) -> int:
    return t.data_ptr()


def _pattern_roundtrip(t: torch.Tensor) -> bool:
    """Write + read back a pattern at head, tail and stride; True iff intact.

    The strided samples are drawn from the INTERIOR (excluding the head/tail
    regions) so the three writes can never alias: for pathological sizes
    (e.g. 3,574,813,696 = 4096*872757 + 1024) the last strided position of
    the old v[::stride] selection landed exactly on the tail region and the
    test clobbered its own pattern.
    """
    v = t.flatten().view(torch.uint8)
    h = min(1024, v.numel())
    v[:h] = torch.arange(h, dtype=torch.uint8)
    if v.numel() > h:
        v[-h:] = torch.arange(h, dtype=torch.uint8) + 1
    stride = max(1, (v.numel() - 2 * h) // 4096)
    sel = v[h : v.numel() - h : stride]
    sel.copy_(torch.arange(sel.numel(), dtype=torch.uint8))
    assert v[0].item() == 0
    if v.numel() > h:
        assert v[-1].item() == 0, (v[-1].item(),)
    assert bool(torch.equal(v[:h], torch.arange(h, dtype=torch.uint8)))
    if v.numel() > h:
        assert bool(torch.equal(v[-h:], torch.arange(h, dtype=torch.uint8) + 1))
    assert bool(torch.equal(sel, torch.arange(sel.numel(), dtype=torch.uint8)))
    return True


def _gpu_roundtrip(t: torch.Tensor) -> None:
    if not torch.cuda.is_available():
        return
    cudart = torch.cuda.cudart()
    rc = cudart.cudaHostRegister(t.data_ptr(), t.numel(), 0)
    assert int(rc) == 0, f"cudaHostRegister failed rc={int(rc)}"
    try:
        n = min(t.numel(), 64 * MIB)
        host = t.flatten().view(torch.uint8)[:n]
        host[: 8 * MIB] = torch.arange(8 * MIB, dtype=torch.uint8) % 251
        dev = host.cuda(non_blocking=False)
        back = dev.cpu()
        assert torch.equal(back, host), "GPU H2D/D2H roundtrip mismatch"
    finally:
        cudart.cudaHostUnregister(t.data_ptr())


def test_base_pages_default():
    with envs.SGLANG_HUGEPAGE_SIZE.override(""), envs.SGLANG_MAP_HOST_POOL_PRIVATE.override(True):
        t = alloc_mmap((256, 1024, 576), torch.uint8)
    assert t.shape == (256, 1024, 576) and t.dtype == torch.uint8
    sm = _smaps_entry(_ptr_of(t))
    assert int(sm["KernelPageSize:"].split()[0]) * 1024 == mmap.PAGESIZE
    assert "nh" in sm["VmFlags:"], "base mode must disable THP (MADV_NOHUGEPAGE)"
    assert _pattern_roundtrip(t)
    _gpu_roundtrip(t)


def test_thp_mode_backed_by_huge_pages():
    try:
        with envs.SGLANG_HUGEPAGE_SIZE.override("THP"), envs.SGLANG_HUGEPAGE_STRICT.override(True):
            t = alloc_mmap((2 * GIB,), torch.uint8)
    except (RuntimeError, OSError) as e:
        pytest.skip(f"THP unavailable on this host: {e}")
    sm = _smaps_entry(_ptr_of(t))
    huge_kb = int(sm["AnonHugePages:"].split()[0])
    assert huge_kb * 1024 >= 0.98 * 2 * GIB, f"THP coverage too low: {sm}"
    assert _pattern_roundtrip(t)
    _gpu_roundtrip(t)


def test_hugetlb_2m_mode():
    before = 0
    for line in open("/proc/meminfo"):
        if line.startswith("HugePages_Surp:"):
            before = int(line.split()[1])
    try:
        with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"), envs.SGLANG_HUGEPAGE_STRICT.override(True):
            t = alloc_mmap((1 * GIB,), torch.uint8)
    except OSError as e:
        pytest.skip(f"2 MiB hugetlbfs unavailable on this host: {e}")
    sm = _smaps_entry(_ptr_of(t))
    assert int(sm["MMUPageSize:"].split()[0]) * 1024 == 2 * MIB, sm
    assert "ht" in sm["VmFlags:"], sm
    surp = 0
    for line in open("/proc/meminfo"):
        if line.startswith("HugePages_Surp:"):
            surp = int(line.split()[1])
    assert surp >= before + (1 * GIB) // (2 * MIB) - 1, "expected surplus huge pages"
    assert _pattern_roundtrip(t)
    _gpu_roundtrip(t)


def test_thp_mode_works_under_disable_thp():
    """SGLANG_DISABLE_THP engines can still allocate a THP-backed pool."""
    from sglang.srt.utils.thp import get_thp_disabled, set_thp_disabled

    prev = get_thp_disabled()
    set_thp_disabled(True)
    try:
        assert get_thp_disabled() == 1
        with envs.SGLANG_HUGEPAGE_SIZE.override("THP"), envs.SGLANG_HUGEPAGE_STRICT.override(True):
            try:
                t = alloc_mmap((1 * GIB,), torch.uint8)
            except (RuntimeError, OSError) as e:
                pytest.skip(f"THP unavailable on this host: {e}")
        # prctl must be restored to disabled after the allocation
        assert get_thp_disabled() == 1
        sm = _smaps_entry(_ptr_of(t))
        huge_kb = int(sm["AnonHugePages:"].split()[0])
        assert huge_kb * 1024 >= 0.98 * 1 * GIB, f"THP coverage too low under disable flag: {sm}"
        assert _pattern_roundtrip(t)
        _gpu_roundtrip(t)
    finally:
        set_thp_disabled(bool(prev))


def test_strict_rejects_unknown_size():
    with envs.SGLANG_HUGEPAGE_SIZE.override("3MB"), envs.SGLANG_HUGEPAGE_STRICT.override(True):
        with pytest.raises(ValueError):
            alloc_mmap((1024,), torch.uint8)


def _alloc_and_register(n_bytes: int, min_bytes: int) -> torch.Tensor:
    """Engine path: HostTensorAllocator + alloc_with_host_register."""
    from sglang.srt.mem_cache.pool_host.common import (
        HostTensorAllocator,
        alloc_with_host_register,
    )

    with (
        envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
        envs.SGLANG_HUGEPAGE_STRICT.override(True),
        envs.SGLANG_HUGEPAGE_MIN_BYTES.override(min_bytes),
    ):
        return alloc_with_host_register(
            (n_bytes,), torch.uint8, "cpu", pin_memory=True,
            allocator=HostTensorAllocator(),
        )


def test_hugetlb_register_exact_indexer_size():
    """The 3,574,813,696 B indexer-pool byte count failed cudaHostRegister
    deterministically at every placement before the register-size fix
    (v16-memory-plan: 19 placements). With the page-rounded register it must
    pin through the engine path."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister
    try:
        t = _alloc_and_register(3_574_813_696, 0)
    except OSError as e:
        pytest.skip(f"2 MiB hugetlbfs unavailable: {e}")
    try:
        sm = _smaps_entry(_ptr_of(t))
        assert int(sm["MMUPageSize:"].split()[0]) * 1024 == 2 * MIB, sm
        assert getattr(t, "_sglang_mmap_alloc_bytes", 0) >= 3_574_813_696
        assert _pattern_roundtrip(t)
    finally:
        _cuda_host_unregister(t)


def test_hugetlb_register_exact_staging_decode_size():
    """The staging-2 C8b decode host pool: (78, 1664064, 1, 656) uint8 =
    85,146,826,752 B -- the exact byte count whose cudaHostRegister failure
    (rc=1 invalid argument, 8 ladder attempts x 2 boots) blocked the v16
    hugetlb arm on nid0111xx. The page-rounded register must pin it."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister

    dims = (78, 1_664_064, 1, 656)
    assert dims[0] * dims[1] * dims[2] * dims[3] == 85_146_826_752
    from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister
    try:
        t = _alloc_and_register(85_146_826_752, 0)
    except OSError as e:
        pytest.skip(f"2 MiB hugetlbfs unavailable: {e}")
    try:
        sm = _smaps_entry(_ptr_of(t))
        assert int(sm["MMUPageSize:"].split()[0]) * 1024 == 2 * MIB, sm
        assert getattr(t, "_sglang_mmap_alloc_bytes", 0) >= 85_146_826_752
        # bit-exact by construction: only pinning changed; spot-check the tensor
        v = t.view(torch.uint8).flatten()
        v[:1024] = torch.arange(1024, dtype=torch.uint8)
        assert v[1023].item() == 1023
    finally:
        _cuda_host_unregister(t)


def test_hugetlb_default_gate_small_pool_stays_base():
    """Default SGLANG_HUGEPAGE_MIN_BYTES (32 GiB) keeps small pools on base
    pages even with SGLANG_HUGEPAGE_SIZE=2MB (v16 sizing behavior)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from sglang.srt.mem_cache.pool_host.common import (
        HostTensorAllocator,
        alloc_with_host_register,
        _cuda_host_unregister,
    )

    with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"), envs.SGLANG_HUGEPAGE_STRICT.override(True):
        t = alloc_with_host_register(
            (1 * GIB,), torch.uint8, "cpu", pin_memory=True,
            allocator=HostTensorAllocator(),
        )
    try:
        sm = _smaps_entry(_ptr_of(t))
        assert int(sm["MMUPageSize:"].split()[0]) * 1024 == mmap.PAGESIZE, sm
    finally:
        _cuda_host_unregister(t)


def test_non_strict_unknown_size_falls_back():
    with envs.SGLANG_HUGEPAGE_SIZE.override("3MB"), envs.SGLANG_HUGEPAGE_STRICT.override(False):
        t = alloc_mmap((1024,), torch.uint8)
    assert t.shape == (1024,)
    assert _pattern_roundtrip(t)


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = skips = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}", flush=True)
        except Exception as e:
            if type(e).__name__ == "Skipped" or "_Skip" in type(e).__name__:
                skips += 1
                print(f"SKIP {t.__name__}: {e}", flush=True)
            else:
                fails += 1
                print(f"FAIL {t.__name__}", flush=True)
                traceback.print_exc()
    print(f"{len(tests) - fails - skips} pass, {skips} skip, {fails} fail", flush=True)
    sys.exit(1 if fails else 0)
