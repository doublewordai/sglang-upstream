# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Process-wide THP (transparent hugepage) discipline for engine processes.

Nodes commonly run THP=always. Only the hisparse host pool carries
MADV_NOHUGEPAGE; every other large anonymous host allocation (torch host
tensors, staging buffers, glibc malloc arenas) is then THP-eligible, which
feeds the khugepaged/kcompactd interference class against pinned engine
memory (the storm that stalls GPU access through ATS translations).
PR_SET_THP_DISABLE is inherited by forked children, so setting it once in
the launcher process covers the whole engine process tree. Explicit hugetlb
host pools (SGLANG_HUGEPAGE_SIZE=2MB) are unaffected by the flag.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

PR_SET_THP_DISABLE = 41
PR_GET_THP_DISABLE = 42

_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
_libc.prctl.restype = ctypes.c_int
_libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 5


def get_thp_disabled() -> int:
    """1 if PR_SET_THP_DISABLE is set for this thread group, else 0."""
    return int(_libc.prctl(PR_GET_THP_DISABLE, *([ctypes.c_ulong(0)] * 5)))


def set_thp_disabled(on: bool) -> None:
    rc = _libc.prctl(
        PR_SET_THP_DISABLE, ctypes.c_ulong(1 if on else 0), *([ctypes.c_ulong(0)] * 4)
    )
    if rc != 0:
        raise RuntimeError(f"prctl(PR_SET_THP_DISABLE, {on}) failed rc={rc}")


def maybe_disable_thp() -> bool:
    """Behind SGLANG_DISABLE_THP=1, disable THP for this process tree.

    Call once at engine init, before large host allocations. Returns whether
    THP is (now) disabled. Idempotent.
    """
    if not envs.SGLANG_DISABLE_THP.get():
        return False
    if get_thp_disabled() != 1:
        set_thp_disabled(True)
        logger.info(
            "SGLANG_DISABLE_THP=1: PR_SET_THP_DISABLE set for this process tree "
            "(anonymous host allocations stay on base pages; hugetlb pools via "
            "SGLANG_HUGEPAGE_SIZE are unaffected)"
        )
    return True
