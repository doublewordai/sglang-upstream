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
    """Write + read back a pattern at head, tail and stride; True iff intact."""
    v = t.view(torch.uint8)
    v[:1024] = torch.arange(1024, dtype=torch.uint8)
    v[-1024:] = torch.arange(1024, dtype=torch.uint8) + 1
    stride = max(1, v.numel() // 4096)
    sel = v[::stride]
    sel.copy_(torch.arange(sel.numel(), dtype=torch.uint8))
    assert v[0].item() == 0 and v[-1].item() == 0
    assert bool(torch.equal(v[::stride], torch.arange(sel.numel(), dtype=torch.uint8)))
    return True


def _gpu_roundtrip(t: torch.Tensor) -> None:
    if not torch.cuda.is_available():
        return
    n = min(t.numel(), 64 * MIB)
    host = t.view(torch.uint8)[:n]
    host[: 8 * MIB] = torch.arange(8 * MIB, dtype=torch.uint8) % 251
    cudart = torch.cuda.cudart()
    rc = cudart.cudaHostRegister(t.data_ptr(), t.numel(), 0)
    assert int(rc) == 0, f"cudaHostRegister failed rc={int(rc)}"
    try:
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


def test_strict_rejects_unknown_size():
    with envs.SGLANG_HUGEPAGE_SIZE.override("3MB"), envs.SGLANG_HUGEPAGE_STRICT.override(True):
        with pytest.raises(ValueError):
            alloc_mmap((1024,), torch.uint8)


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
