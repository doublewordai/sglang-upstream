import functools
import json
import os
import warnings
from contextlib import contextmanager
from enum import IntEnum
from typing import Any, Callable, Dict, Optional


@functools.lru_cache(maxsize=1)
def _default_hip() -> bool:
    """Lazy ROCm/HIP detection for platform-conditional env defaults.

    Avoids importing torch at environ import time (this module is intentionally
    stdlib-only and loaded very early). Resolved on first EnvField.get() that uses
    it as a default, by which point torch is already imported in any real run;
    falls back to False if torch is unavailable.
    """
    try:
        import torch

        return torch.version.hip is not None
    except Exception:
        return False


def _default_cache_subdir(name: str) -> str:
    """A directory under SGLANG_CACHE_DIR, for env defaults that track it.

    Pass as a callable default: SGLANG_CACHE_DIR is declared further down the
    Envs body, and resolving late also lets tests override it.
    """
    return os.path.join(os.path.expanduser(envs.SGLANG_CACHE_DIR.get()), name)


class EnvField:
    _allow_set_name = True

    def __init__(self, default: Any):
        self.default = default
        # NOTE: environ can only accept str values, so we need a flag to indicate
        # whether the env var is explicitly set to None.
        self._set_to_none = False

    def __set_name__(self, owner, name):
        assert EnvField._allow_set_name, "Usage like `a = envs.A` is not allowed"
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def _resolve_default(self) -> Any:
        # Support a callable default for lazily/platform-computed defaults
        # (e.g. EnvBool(_default_hip)); evaluated only when the env is unset.
        return self.default() if callable(self.default) else self.default

    def get(self) -> Any:
        value = os.getenv(self.name)

        # Explicitly set to None
        if self._set_to_none:
            assert value == str(None)
            return None

        # Not set, return default
        if value is None:
            return self._resolve_default()

        try:
            return self.parse(value)
        except ValueError as e:
            default = self._resolve_default()
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{default}"'
            )
            return default

    def is_set(self):
        return self.name in os.environ

    def set(self, value: Any):
        self._set_to_none = value is None
        os.environ[self.name] = str(value)

    @contextmanager
    def override(self, value: Any):
        backup_present = self.name in os.environ
        backup_value = os.environ.get(self.name)
        backup_set_to_none = self._set_to_none
        self.set(value)
        yield
        if backup_present:
            os.environ[self.name] = backup_value
        else:
            os.environ.pop(self.name, None)
        self._set_to_none = backup_set_to_none

    def clear(self):
        os.environ.pop(self.name, None)
        self._set_to_none = False

    def __bool__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )

    def __len__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )


class EnvTuple(EnvField):
    def parse(self, value: str) -> tuple[str, ...]:
        return tuple(s.strip() for s in value.split(",") if s.strip())


class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value


class EnvJSON(EnvField):
    def parse(self, value: str | None) -> list | dict | None:
        if not value:
            return None
        if os.path.exists(value):
            with open(value) as f:
                return json.load(f)
        return json.loads(value)


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid integer value')


class _DeprecatedEnvFallback:
    """Mixin for EnvField subclasses: if the canonical env var is not set,
    check *deprecated_name* and emit DeprecationWarning before reading it.

    Usage:
        SGLANG_DSA_FUSE_TOPK = EnvBoolWithAlias(True, deprecated_name="SGLANG_NSA_FUSE_TOPK")
    """

    def __init__(self, default: Any, deprecated_name: str):
        super().__init__(default)
        self.deprecated_name = deprecated_name

    def get(self) -> Any:
        if os.getenv(self.name) is None:
            fallback = os.getenv(self.deprecated_name)
            if fallback is not None:
                warnings.warn(
                    f"Environment variable '{self.deprecated_name}' is deprecated; "
                    f"use '{self.name}' instead. "
                    "The alias will be removed in a future release.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                os.environ[self.name] = fallback
        return super().get()


class EnvBoolWithAlias(_DeprecatedEnvFallback, EnvBool):
    pass


class EnvIntWithAlias(_DeprecatedEnvFallback, EnvInt):
    pass


class EnvFloat(EnvField):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid float value')


class GateGemvMode(IntEnum):
    """Small-batch Inkling gate linear implementation.

    OFF: always the cublas GEMM
    PAIR: PDL-chained GEMV and gate JIT kernels
    FUSED: single-launch GEMV + gate epilogue (last-block ticket)
    """

    OFF = 0
    PAIR = 1
    FUSED = 2


class ToolStrictLevel(IntEnum):
    """
    Defines the strictness levels for tool call parsing and validation.

    OFF: No strict validation
    FUNCTION: Enables structural tag constraints for all tools
    PARAMETER: Enforces strict parameter validation for all tools
    """

    OFF = 0
    FUNCTION = 1
    PARAMETER = 2


class InvariantCheckLevel(IntEnum):
    """Signal level for value/index validity checks (see invariants.py).

    OFF: data layer only (sanitize/containment); no detection, no signal.
    WARN: detect + throttled log/count; degrade, never crash (prod on-demand).
    STRICT: detect + crash on GUARD/FATAL violations (CI default).

    The data layer is unconditional and independent of this level; only the
    detection + signal layer is gated here.
    """

    OFF = 0
    WARN = 1
    STRICT = 2


class DsparkFoldedSampling(IntEnum):
    """Sampling support in the graph-folded DSpark draft proposal: OFF =
    greedy-only folding, AUTO = on when its buffers fit in free GPU memory,
    FORCE = always."""

    OFF = 0
    AUTO = 1
    FORCE = 2


class Envs:
    # Organization principles for this registry:
    # - Put every field in exactly one topical section. Prefer an existing
    #   section; add a new one only when no current section is a clear fit.
    # - Group by the behavior and owning call sites, not by name similarity
    #   alone. Keep closely related lifecycle or feature knobs adjacent.
    # - Keep each section focused and below 30 fields. Split growing sections
    #   by subsystem or lifecycle instead of creating catch-all groups.
    # - Order broad runtime subsystems before shared storage and backends; keep
    #   platform- and model-specific integrations in dedicated later sections.
    # - Use the same three-line section header everywhere; do not add ad hoc
    #   one-line headings or append unrelated fields at the end of a section.
    # - Keep vendor-specific aliases with their owning integration, and keep
    #   test/debug knobs with the feature or test workflow they exercise.
    # - Keep explanatory comments attached to their field when moving it.
    # - For organization-only changes, AST-check that field names, descriptor
    #   types, and defaults are unchanged and that only field order moved.

    # ===================================================================
    # Runtime configuration and process identity
    # ===================================================================
    # Per-role config-namespace bookkeeping: off / record / enforce (value is
    # validated fail-loud in runtime_context, which resolves it once at import
    # so the read stays dynamo-prunable).
    SGLANG_ROLE_NAMESPACES = EnvStr("off")
    # Record mode: append each newly observed (role, namespace) pair to this
    # file so the audit survives signal-killed workers.
    SGLANG_ROLE_NAMESPACES_OUT = EnvStr(None)
    IS_H200 = EnvBool(False)
    SGLANG_ENABLE_TORCH_INFERENCE_MODE = EnvBool(False)

    # ===================================================================
    # Model configuration, discovery, and weight loading
    # ===================================================================
    SGLANG_USE_MODELSCOPE = EnvBool(False)
    # Controls weight-file ordering for load-time I/O optimization.
    #   -1 : no sorting, no staggering; preserves original file order.
    #    0 : sort files only; maximizes ordering but may reduce cross-rank I/O concurrency.
    #   k>0: sort files and stagger per-rank order with factor k.
    #        Files are processed in groups of (tp_size * k), and rank r starts each
    #        group at offset (r * k), improving multi-rank I/O concurrency while
    #        keeping access relatively ordered.
    SGLANG_SORT_WEIGHT_FILES = EnvInt(0)
    SGLANG_DISABLED_MODEL_ARCHS = EnvTuple(tuple())
    SGLANG_PREFETCH_BLOCK_SIZE_MB = EnvInt(16)
    SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION = EnvBool(False)
    SGLANG_ENABLE_WEIGHT_LOADER_V2 = EnvBool(False)
    # Copy rank-local MoE slices into independent CPU storage before H2D when
    # they reference a larger mmap-backed checkpoint storage.
    SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D = EnvBool(False)
    # fast-boot: stage safetensors mmap views into anon CPU storage before H2D
    # (removes the 2-6x file-page re-read of pageable H2D; byte-identical).
    SGLANG_WEIGHT_LOAD_STAGE_VIEWS = EnvBool(False)
    SGLANG_LOAD_SNAPSHOT_USE_ZMQ = EnvBool(False)
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvBool(False)
    HF_HUB_DISABLE_XET = EnvBool(False)
    # In seconds. If a warmup forward batch takes longer than this, the server will crash to prevent hanging.
    # Recommend to increase warmup timeout to 1800 to accommodate some kernel JIT precache e.g. deep gemm
    SGLANG_WARMUP_TIMEOUT = EnvFloat(-1)
    SGLANG_EXTERNAL_MODEL_PACKAGE = EnvStr("")
    SGLANG_EXTERNAL_MM_MODEL_ARCH = EnvStr("")
    SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE = EnvStr("")

    # ===================================================================
    # HTTP server and health
    # ===================================================================
    # Decompress request bodies tagged with `x-body-compressed`.
    SGLANG_ENABLE_REQUEST_DECOMPRESSION = EnvBool(False)
    # Override parsed request fields from headers.
    SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES = EnvBool(False)
    DISABLE_OPENAPI_DOC = EnvBool(False)
    SGLANG_TIMEOUT_KEEP_ALIVE = EnvInt(5)
    # Uvicorn multiprocess supervisor pings each worker on this interval; default 5s is
    # too short when many workers cold-start and load tokenizers in parallel.
    SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT = EnvInt(10)
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION = EnvBool(True)

    # ===================================================================
    # Logging
    # ===================================================================
    SGLANG_LOG_GC = EnvBool(False)
    SGLANG_LOG_FORWARD_ITERS = EnvBool(False)
    SGLANG_LOG_DECODE_GRAPH_KEY = EnvBool(False)
    SGLANG_LOG_MS = EnvBool(False)
    SGLANG_LOG_REQUEST_EXCEEDED_MS = EnvInt(-1)
    SGLANG_LOG_REQUEST_HEADERS = EnvTuple(tuple())
    SGLANG_LOG_SCHEDULER_STATUS_TARGET = EnvStr("")
    SGLANG_LOG_SCHEDULER_STATUS_INTERVAL = EnvFloat(60.0)

    # ===================================================================
    # IPC, broadcasters, and ports
    # ===================================================================
    SGLANG_USE_PICKLE_IPC = EnvBool(True)
    # Log top-level PickleWrapper frames unwrapped on msgpack IPC decode.
    SGLANG_LOG_PICKLE_IPC_OBJECTS = EnvBool(False)
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER = EnvBool(True)
    SGLANG_TCP_STORE_PORT = EnvInt(29600)
    # Base port hint for ephemeral sockets (ZMQ, SHM broadcaster, etc.).
    # When set, get_open_port() and shm_broadcast search upwards from this
    # value instead of asking the OS for a random port.  Useful to keep all
    # SGLang ports in a predictable range behind a firewall.
    SGLANG_PORT = EnvInt(None)
    SGLANG_BACKUP_PORT_BASE = EnvInt(10000)

    # ===================================================================
    # CI and test execution
    # ===================================================================
    SGLANG_IS_IN_CI = EnvBool(False)
    SGLANG_IS_IN_CI_AMD = EnvBool(False)
    SGLANG_TEST_MAX_RETRY = EnvInt(None)
    # Expand jit_kernel test grids to their full parameter ranges (nightly).
    SGLANG_JIT_KERNEL_RUN_FULL_TESTS = EnvBool(False)
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK = EnvBool(False)

    # ===================================================================
    # Crash diagnostics and shutdown
    # ===================================================================
    SGLANG_CUDA_COREDUMP = EnvBool(False)
    # None = unset, letting get_dump_dir() resolve the base (RUNNER_TEMP in CI,
    # else /tmp); see debug_utils/cuda_coredump.py.
    SGLANG_CUDA_COREDUMP_DIR = EnvStr(None)
    SGLANG_FORCE_SHUTDOWN = EnvBool(False)
    SGLANG_PYSPY_DUMP_BEFORE_CRASH = EnvBool(True)
    SGLANG_CUDA_COREDUMP_BEFORE_CRASH = EnvBool(True)
    SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS = EnvFloat(60.0)

    # ===================================================================
    # Constrained decoding and grammar
    # ===================================================================
    SGLANG_GRAMMAR_POLL_INTERVAL = EnvFloat(0.005)
    SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = EnvInt(10000)
    SGLANG_DISABLE_OUTLINES_DISK_CACHE = EnvBool(False)

    # ===================================================================
    # Fault injection and regression tests
    # ===================================================================
    SGLANG_TEST_STUCK_DETOKENIZER = EnvFloat(0)
    SGLANG_TEST_STUCK_DP_CONTROLLER = EnvFloat(0)
    SGLANG_TEST_STUCK_SCHEDULER_INIT = EnvFloat(0)
    SGLANG_TEST_STUCK_TOKENIZER = EnvFloat(0)
    SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS = EnvInt(0)
    SGLANG_TEST_REQUEST_TIME_STATS = EnvBool(False)
    SGLANG_TEST_DISAGG_FAILURE_PROB = EnvFloat(0.0)
    SGLANG_TEST_RETRACT = EnvBool(False)
    SGLANG_TEST_RETRACT_INTERVAL = EnvInt(3)
    SGLANG_TEST_RETRACT_NO_PREFILL_BS = EnvInt(2**31)
    # Scheduler: force lazy extra_buffer prealloc to fail at decode boundaries
    SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL = EnvBool(False)
    # KL tests: skip the cache-hit count assertion (e.g. when alloc failure reduces hits)
    SGLANG_TEST_SKIP_CACHE_HIT_ASSERT = EnvBool(False)

    # ===================================================================
    # PD and scripted-runtime tests
    # ===================================================================
    SGLANG_TEST_PD_DISAGG_BACKEND = EnvStr("mooncake")
    SGLANG_TEST_PD_DISAGG_DEVICES = EnvStr(None)
    SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB = EnvFloat(0.0)
    SGLANG_TEST_SCRIPTED_RUNTIME = EnvBool(False)
    SGLANG_TEST_SCRIPTED_RUNTIME_IPC_ADDR = EnvStr(None)
    SGLANG_TEST_SCRIPTED_RUNTIME_OUT_OF_BAND_ERROR_PATH = EnvStr(None)
    SGLANG_TEST_SCRIPTED_RUNTIME_SYS_PATH_ENTRY = EnvStr(None)

    # ===================================================================
    # Profiling, tracing, and metrics
    # ===================================================================
    SGLANG_PROFILE_WITH_STACK = EnvBool(True)
    SGLANG_PROFILE_RECORD_SHAPES = EnvBool(True)
    SGLANG_PROFILE_V2 = EnvBool(False)
    SGLANG_ENABLE_NVTX_SCHEDULER = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_ENABLE_NVTX"
    )
    SGLANG_ENABLE_NVTX_OPERATIONS = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_OPERATIONS_ENABLE_PROFILE"
    )
    SGLANG_RECORD_STEP_TIME = EnvBool(False)
    SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE = EnvBool(False)
    # Opt-in: emit one CUDA-graph capture trace per captured batch size (per-bs).
    # SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE (single combined trace) takes
    # precedence when both are set.
    SGLANG_GRAPH_BATCH_CAPTURE = EnvBool(False)
    SGLANG_TORCH_PROFILER_DIR = EnvStr("/tmp")
    # Allocator-history buffer for /start_profile activities=["MEM"]; the
    # default truncates long windows (each entry is one alloc/free event).
    SGLANG_MEM_PROFILE_MAX_ENTRIES = EnvInt(100000)
    SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS = EnvInt(500)
    SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE = EnvInt(64)
    SGLANG_TRACE_ASYNC = EnvBool(False)
    SGLANG_TRACE_ASYNC_FLUSH_THRESHOLD = EnvInt(100)
    SGLANG_ENABLE_METRICS_DEVICE_TIMER = EnvBool(False)
    SGLANG_ENABLE_METRICS_DP_ATTENTION = EnvBool(False)

    # ===================================================================
    # Debugging and invariant checks
    # ===================================================================
    SGLANG_DETECT_SLOW_RANK = EnvBool(False)
    SGLANG_DEBUG_MEMORY_POOL = EnvBool(False)
    # NaN-fill the unified memory pool at boot (debug repro switch).
    SGLANG_DEBUG_POISON_POOL = EnvBool(False)
    SGLANG_DEBUG_REVERT_PR = EnvInt(0)
    SGLANG_PHASE_CHECKER_DEBUG = EnvBool(False)
    SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(True)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY = EnvInt(0)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE = EnvBool(True)
    # Physical KV-page checks: committed<=allocated + no page alias.
    SGLANG_CHECK_KV_PAGE_INVARIANTS = EnvBool(False)
    SGLANG_TBO_DEBUG = EnvBool(False)
    # Timing probe: run the swap-in fully but skip the host->device KV bytes,
    # measuring the "IO is free" floor. GARBAGE OUTPUT -- benchmarking only.
    SGLANG_DEBUG_HISPARSE_SKIP_IO = EnvBool(False)
    SGLANG_DSA_IN_GRAPH_METADATA = EnvBool(False)
    SGLANG_HISPARSE_FAST_BACKUP = EnvBool(False)
    # mtp-debug lane: log the first target-verify step's hisparse page table
    # and the draft pool's transferred rows (diagnosis of spec x hisparse).
    SGLANG_MTP_DEBUG = EnvBool(False)
    # Bulk host<->device HiCache transfers: coalesce page-granular index sets
    # into contiguous runs and move them with cudaMemcpyBatchAsync (copy
    # engine) instead of per-row UVA gather/scatter kernels. Byte-identical
    # copies; affects the HiCache H2D load path (page_first host pools) and
    # the layer_first D2H backup path (hisparse staging backup).
    SGLANG_HICACHE_BULK_COPY = EnvBool(True)
    # D2H bulk backup via warp-coalesced SM stores instead of the copy engine
    # (the CE D2H path is capped at ~170 GB/s over C2C on GH200; coalesced SM
    # stores sustain ~383 GB/s into the same pinned pool). Replaces the
    # segment copies AND the remainder kernel of the bulk backup when
    # SGLANG_HICACHE_BULK_COPY is also on; byte-identical; falls back to the
    # merged copy-engine path when the JIT module is unavailable.
    SGLANG_D2H_SM_STORES = EnvBool(False)
    # warm-local-prefill (lane warm-local-prefill): decode-rank local extend of
    # warm-turn appends. Enabled per-request via rid prefix "WLP-"; these gate
    # eligibility (max new-span tokens, min matched fraction of the prompt) and
    # tracing.
    SGLANG_WLP_ENABLE = EnvBool(False)
    SGLANG_WLP_MAX_NEW_TOKENS = EnvInt(4096)
    SGLANG_WLP_MIN_MATCH_FRAC = EnvFloat(0.5)
    # Allow prefix_len == 0 (full local prefill of a short prompt on the decode
    # rank): rig weight-equality probe + the local-refill fallback variant.
    SGLANG_WLP_ALLOW_COLD = EnvBool(False)
    SGLANG_WLP_TRACE = EnvBool(False)
    # wlp-fused-topk: consume prod's fused PAGED top-k output (slot-resolved
    # logical locs) in the WLP retained-prefix extend. ON (default): the
    # union swap-in discriminates prefix/delta selections in the loc domain
    # via logical_to_host_row -- the same fused/1PASS kernel (and therefore
    # the same selection) as the prefill arm's warm extends. OFF: WLP extends
    # force the unfused top-k (raw positions; the original lane-validated
    # path; exact vs the PD path only when SGLANG_DSA_FUSE_TOPK=0 on both
    # arms).
    SGLANG_WLP_FUSED_TOPK = EnvBool(True)
    # Master switch for all async-asserted invariant probes (NaN, Inf, OOB,
    # page alignment). Off in prod; tests turn it on to fail-fast on
    # numerical / index violations instead of getting silent NaN cascades.
    SGLANG_ENABLE_ASYNC_ASSERT = EnvBool(False)
    # Signal level for value/index validity checks (nan/inf/oob/...); see
    # invariants.py. OFF (prod default) runs only the free data layer, WARN
    # adds throttled logging, STRICT (CI default) crashes on violations.
    # Supersedes SGLANG_ENABLE_ASYNC_ASSERT, which is bridged as STRICT until
    # every callsite migrates.
    SGLANG_INVARIANT_CHECK = EnvInt(InvariantCheckLevel.OFF)

    # ===================================================================
    # Runtime simulations
    # ===================================================================
    SGLANG_SIMULATE_ACC_LEN = EnvFloat(-1)
    SGLANG_SIMULATE_ACC_METHOD = EnvStr("match-expected")
    SGLANG_SIMULATE_ACC_TOKEN_MODE = EnvStr("fixed")
    SGLANG_SIMULATE_UNIFORM_EXPERTS = EnvBool(False)
    SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS = EnvBool(False)

    # ===================================================================
    # DSpark speculative decoding
    # ===================================================================
    SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER = EnvBool(False)
    SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS = EnvBool(False)
    SGLANG_DSPARK_DEBUG_DUMP = EnvTuple(tuple())
    SGLANG_DSPARK_LOG_SPS_PRED_INTERVAL = EnvInt(0)
    SGLANG_DSPARK_STS_COLLECT_PATH = EnvStr("")
    SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH = EnvStr("")
    SGLANG_DSPARK_BLOCK_ACCEPT_ONLINE_INTERVAL = EnvInt(0)
    SGLANG_DSPARK_ENABLE_SPS_RECORD = EnvBool(False)
    SGLANG_DSPARK_FAST_KERNEL = EnvBool(True)
    SGLANG_DSPARK_FP32_LM_HEAD = EnvBool(False)
    SGLANG_DSPARK_FAST_SAMPLING = EnvBool(True)
    SGLANG_DSPARK_FOLDED_SAMPLING = EnvInt(DsparkFoldedSampling.AUTO)
    SGLANG_DSPARK_FOLDED_PROPOSAL = EnvBool(True)
    SGLANG_DSPARK_STACKED_CTX_KV = EnvBool(True)
    SGLANG_DSPARK_EMBED_IN_GRAPH = EnvBool(True)
    SGLANG_DSPARK_OPT_MARKOV_W2_BF16 = EnvBool(True)
    SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD = EnvBool(True)
    SGLANG_DSPARK_ENABLE_MULTI_STREAM = EnvBool(True)
    SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS = EnvInt(2)

    # ===================================================================
    # Memory pools and KV-cache sizing
    # ===================================================================
    SGLANG_NATIVE_MOVE_KV_CACHE = EnvBool(False)
    # Disable lazy compaction in the unified memory pool allocator and
    # fall back to the per-free eager compaction. Used for production
    # A/B and quick rollback. Default False (lazy compaction on).
    SGLANG_DISABLE_LAZY_COMPACTION = EnvBool(False)
    # Sort the multi-ended allocator's free list after a merge (perf A/B knob).
    SGLANG_SORT_FREE_LIST_AFTER_MERGE = EnvBool(False)
    # Periodically log lazy-compaction stats per sub-pool (observability only).
    SGLANG_LOG_LAZY_COMPACTION_STATS = EnvBool(False)
    SGLANG_LOG_LAZY_COMPACTION_STATS_INTERVAL_SEC = EnvInt(30)
    # HND KV layout folds (page, head) into one paged index for per-kv-head sparse
    # page tables (DP attn); paged backends like trtllm_mha consume it directly.
    SGLANG_USE_HND_KVCACHE = EnvBool(False)
    # Size the KV pool after CUDA-graph capture.
    SGLANG_ENABLE_POST_CAPTURE_KV_SIZING = EnvBool(False)

    # ===================================================================
    # Scheduler token budgeting and admission
    # ===================================================================
    SGLANG_INIT_NEW_TOKEN_RATIO = EnvFloat(0.7)
    SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR = EnvFloat(0.14)
    SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS = EnvInt(600)
    SGLANG_RETRACT_DECODE_STEPS = EnvInt(20)
    SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION = EnvInt(4096)
    SGLANG_MAX_NEW_TOKENS_LIMIT = EnvInt(None)
    SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR = EnvFloat(0.75)
    SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES = EnvInt(None)
    SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK = EnvFloat(None)
    SGLANG_DATA_PARALLEL_BUDGET_INTERVAL = EnvInt(1)
    # Compact extend-attention scheduler tile-budget admission (AMD/HIP-only).
    # Budget <= 0 disables; >0 sets the max prefix-extend tiles per batch.
    SGLANG_PREFILL_TILE_BUDGET = EnvInt(0)
    # Tile-budget mode: "compact" (default, counts actual per-request tiles) or
    # "legacy" (rectangular grid, max_extend_len-shaped).
    # Internal/testing only - users should not need to change this.
    SGLANG_PREFILL_TILE_BUDGET_MODE = EnvStr("compact")
    SGLANG_PREFILL_DELAYER_MAX_PREFILL_BS_WINDOW_SIZE = EnvInt(16)

    # ===================================================================
    # Scheduler polling, timeouts, and output
    # ===================================================================
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT = EnvInt(1000)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE = EnvInt(1)
    # in seconds. Set if you observe high memory accumulation over a long serving period.
    SGLANG_EMPTY_CACHE_INTERVAL = EnvFloat(-1)
    SGLANG_SCHEDULER_MAX_RECV_PER_POLL = EnvInt(-1)
    SGLANG_SCHEDULER_SKIP_ALL_GATHER = EnvBool(False)
    SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE = EnvBool(False)
    SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION = EnvBool(False)
    SGLANG_REQ_WAITING_TIMEOUT = EnvFloat(-1)  # in seconds
    SGLANG_REQ_RUNNING_TIMEOUT = EnvFloat(-1)  # in seconds
    # For non-streaming requests, the scheduler still flushes intermediate
    # output batches to the tokenizer manager every N decoded tokens so that
    # `first_token_time`/TTFT can be recorded. Lower this (e.g. to 1) to get
    # an accurate TTFT for benchmarking; the upstream default of 50 trades
    # off some TTFT-metric accuracy for less IPC overhead.
    SGLANG_FORCE_STREAM_INTERVAL = EnvInt(50)

    # ===================================================================
    # Overlap scheduler and pipeline parallelism
    # ===================================================================
    SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP = EnvBool(False)
    # Force delay_sample_func for all overlap decode (not just grammar mode),
    # allowing CPU result processing to overlap with subsequent forward computation
    # and reducing the impact of sampling overhead on the critical path.
    SGLANG_ENABLE_DELAY_SAMPLE = EnvBool(False)
    # Fuse the post-logits decode pipeline (temperature -> softmax -> token
    # selection) into one Triton kernel, bit-exact vs the eager reference
    # (greedy: torch.argmax tie-break; sampled: torch.multinomial's gumbel-max
    # composite with identical philox RNG consumption). Falls back to the
    # reference path for top-k/top-p/min-p, seeded/deterministic sampling,
    # logprob requests, and non-CUDA devices.
    SGLANG_FUSED_SAMPLING = EnvBool(False)
    # Force-enable the WAR (write-after-read) barrier for the overlap scheduler
    # even when is_cuda() is False (e.g. AMD/ROCm). On CUDA the barrier is
    # already enabled regardless of this flag (see start_event_loop).
    SGLANG_ENABLE_WAR_BARRIER = EnvBool(False)
    # Force the WAR barrier to wait for the whole forward instead of the
    # read-done fastpath event.
    SGLANG_FORCE_COARSE_WAR_BARRIER = EnvBool(False)
    # Enable prefill read-done publication after compliant metadata initialization.
    SGLANG_ENABLE_PREFILL_WAR_READ_DONE = EnvBool(False)
    # PP: skip output send/recv when the entire batch consists of non-final chunked prefill requests,
    # since process_batch_result_prefill discards next_token_ids for those anyway.
    SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM = EnvBool(False)
    SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH = EnvBool(False)

    # ===================================================================
    # Radix and sparse KV caches
    # ===================================================================
    SGLANG_EXPERIMENTAL_CPP_RADIX_TREE = EnvBool(False)
    SGLANG_RADIX_FORCE_MISS = EnvBool(False)
    SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD = EnvInt(8192)
    SGLANG_MAX_KV_CHUNK_CAPACITY = EnvInt(128 * 1024)
    # Kill-switch for the shared-index (IndexShare) swap-in prefetch
    # (auto-enabled for GLM-5.2-style DSA); set True to A/B synchronous swap-in.
    SGLANG_DISABLE_HISPARSE_PREFETCH = EnvBool(False)

    # MTP verify path's multi-position shared-index prefetch (union swap-in
    # per skip layer). Default True (merge policy 2026-09-03: exact and
    # measured-positive lands as the default path): the per-position fork
    # measures 2.6-3.2x vs the synchronous per-position swap-in and is
    # bit-identical. Set 0 to restore the synchronous fallback (speculation
    # itself is default-off in prod, so this only affects spec runs).
    SGLANG_HISPARSE_SPEC_PREFETCH = EnvBool(True)
    # Debug: verify each step swap-in selection contains distinct token
    # positions (kernel miss-compaction invariant; see hisparse_coordinator).
    SGLANG_DEBUG_HISPARSE_CHECK_TOPK = EnvBool(False)
    # draft-prefetch lane probe: per verify step, stash the draft seed top-k,
    # the target's per-layer/per-position top-k, the per-layer resident tables,
    # per-(layer,position) swap-in CUDA-event timings and miss counts, then
    # append one JSON line per step to SGLANG_DPF_PROBE_OUT. Works with the
    # eager and CUDA-graph verify paths; spec (target_verify) only.
    SGLANG_DPF_PROBE = EnvBool(False)
    SGLANG_DPF_PROBE_OUT = EnvStr("")
    SGLANG_DPF_PROBE_REQS = EnvInt(2)
    SGLANG_DPF_PROBE_RAW_STEPS = EnvInt(0)
    # draft-prefetch: after the draft step, prefetch (draft seed top-k -
    # resident) rows into each layer's device buffer on the coordinator's
    # side stream, ordered by layer; the verify swap-ins wait per layer.
    # Exact by construction (cache fill only).
    SGLANG_DPF_PREFETCH = EnvBool(False)
    # Diagnostic: log HiSparse verify-step miss counts (per draft position,
    # last anchor group's plans) every N verify steps; 0 = off.
    SGLANG_HISPARSE_MISS_LOG = EnvInt(0)
    # Plan-then-IO swap-in split: the fused kernel plans only and a
    # full-GPU-grid kernel copies the recorded miss plan (warp per row).
    # Set False to A/B the fused in-kernel copy (pre-wide-gather path).
    SGLANG_HISPARSE_WIDE_GATHER = EnvBool(True)
    # mk-batch-curve: size the narrow copy_cache_planned fallback grid by bytes
    # (one CTA per 64 KiB of worst-case miss bytes, capped at the SM count)
    SGLANG_HISPARSE_RIGHTSIZE_COPY_GRID = EnvBool(False)

    # HiSparse IO streams (write-staging / decode-backup / shared-index
    # prefetch / the swap-in gather) bound to a CUDA green context holding this
    # many SMs (0 = normal streams, feature off). Bounded SM footprint for the
    # hicache-like side traffic so it cannot steal SMs from the decode critical
    # path; CUDA-graph capture works across primary+green streams (verified on
    # GH200, driver 565.57.01 / CUDA 13 userspace).
    SGLANG_HISPARSE_GREEN_CTX_SMS = EnvInt(0)
    # Additionally run the swap-in gather itself (anchor/verify calls, which
    # sit on the attention critical path) on the green-context stream with a
    # fork+join. Off by default: at decode batch sizes the fused swap-in grid
    # is b x 960 threads (<= 5 SMs), so SM isolation buys nothing there while
    # the fork/join adds latency; on by default would change step timing.
    SGLANG_HISPARSE_SWAPIN_GREEN_CTX = EnvBool(False)
    # Defer the decode hisparse map/eager-backup chain (map_last_loc_to_buffer)
    # from prepare_for_decode to the forward launch (run_batch), i.e. past the
    # per-step DP-attention metadata all_gather. A rank still running that chain
    # is a straggler entering the collective; every other rank then waits in
    # c10d::Work::wait with its GPU idle (the synchronized 8-12 ms decode
    # stalls; see lanes hiccup/hiccup-2). Default off.
    SGLANG_HISPARSE_DEFER_DECODE_MAP = EnvBool(False)
    # Measurement-only (hiccup-3): path to append per-call eager-backup rows to.
    SGLANG_HISPARSE_BACKUP_LOG = EnvStr("")
    # Opt-in: allocate DSA index-K only on the layers that compute top-k
    # (shared-index models) under HiSparse / PD disaggregation as well. Set it
    # on both PD arms; the NIXL transport carries nothing for 0-byte layers.
    SGLANG_DSA_ELIDE_SHARED_INDEX_K = EnvBool(False)
    SGLANG_DSA_ELIDE_PREFILL_HICACHE = EnvBool(False)
    SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS = EnvBool(True)
    # Decode batches between SWA out-of-window evictions.
    SGLANG_SWA_EVICTION_INTERVAL = EnvInt(128)
    SGLANG_ENABLE_UNIFIED_RADIX_TREE = EnvBool(False)
    # Registered TreeCore backend serving the unified radix cache.
    SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND = EnvStr("python")
    # TODO(DSV4): @ispobock this has bug on main branch when retract
    SGLANG_OPT_SWA_RADIX_CACHE_COMPACT = EnvBool(False)
    SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT = EnvBool(False)
    SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW = EnvBool(False)

    # ===================================================================
    # PD disaggregation runtime
    # ===================================================================
    # NOTE: For SGLANG_DISAGGREGATION_THREAD_POOL_SIZE, the effective default is
    # computed dynamically at runtime based on cpu_count; see disaggregation backends.
    SGLANG_DISAGGREGATION_THREAD_POOL_SIZE = EnvInt(None)
    SGLANG_DISAGGREGATION_QUEUE_SIZE = EnvInt(4)
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_ZMQ_SEND_TIMEOUT = EnvInt(1)
    SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL = EnvFloat(5.0)
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE = EnvInt(2)
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_NIXL_BACKEND = EnvStr("UCX")
    SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS = EnvStr("{}")
    SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX = EnvBool(True)
    SGLANG_DISAGGREGATION_ZMQ_MAX_SOCKETS = EnvInt(16384)
    SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER = EnvBool(False)
    SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK = EnvBool(False)
    SGLANG_DISAGGREGATION_SAMPLING_MASK_MAX_TOKENS = EnvInt(0)
    SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL = EnvInt(120)

    # ===================================================================
    # Distributed and model-parallel runtime
    # ===================================================================
    SGLANG_ENABLE_CP_V2 = EnvBool(False)
    SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS = EnvBool(False)
    # Comma-separated bundle indices for Ray Custom PG mode (e.g., "0,1,2,7").
    SGLANG_RAY_BUNDLE_INDICES = EnvStr("")
    # Override the distributed init method used by torch.distributed.init_process_group.
    # Set to "env://" to use an externally-created TCPStore via MASTER_ADDR/MASTER_PORT.
    SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE = EnvStr(None)
    SGLANG_IS_FIRST_RANK_ON_NODE = EnvBool(True)
    SGLANG_SYNC_TOKEN_IDS_ACROSS_TP = EnvBool(False)
    SGLANG_ENABLE_COLOCATED_BATCH_GEN = EnvBool(False)
    SGLANG_SHARED_EXPERT_TP1 = EnvBool(False)
    # Replicate the input embedding across TP ranks instead of sharding it
    # along the vocab dimension (saves an all-reduce/all-gather in the embed
    # lookup at the cost of replicated embedding weights). Drives both the
    # target and every draft that shares its embedding (see
    # get_embedding_tp_kwargs); they must stay in lock-step. Currently only
    # applies to the Deepseek-V2 family (Deepseek V3.1, Kimi K2.5) + drafts.
    SGLANG_ENABLE_EMBED_REPLICATION = EnvBool(False)

    # ===================================================================
    # Tool calling and native web search
    # ===================================================================
    SGLANG_FORWARD_UNKNOWN_TOOLS = EnvBool(False)
    # Native web search (Exa). EXA_API_KEY is the vendor BYOK credential
    # (kept as-is, not renamed to SGLANG_*); the SGLANG_EXA_* knobs tune the
    # request defaults for the built-in GPT-OSS web_search tool.
    EXA_API_KEY = EnvStr(None)
    SGLANG_EXA_NUM_RESULTS = EnvInt(10)
    SGLANG_EXA_SEARCH_TYPE = EnvStr("auto")
    SGLANG_EXA_INCLUDE_HIGHLIGHTS = EnvBool(True)
    SGLANG_TOOL_STRICT_LEVEL = EnvInt(ToolStrictLevel.OFF)

    # ===================================================================
    # HiCache storage backends and mmap allocation
    # ===================================================================
    SGLANG_HICACHE_HF3FS_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE = EnvInt(None)
    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR = EnvStr(None)
    # File-backend LRU eviction (opt-in; sizes accept SI/IEC suffixes, "0" disables).
    SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE = EnvStr(None)
    SGLANG_HICACHE_FILE_BACKEND_EVICTION_RATIO = EnvFloat(0.9)
    SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE = EnvStr("0")
    # Enable client-side metadata caching to optimize filesystem checks (e.g. for Lustre/NFS/FUSE)
    SGLANG_HICACHE_FILE_BACKEND_ENABLE_METADATA_CACHE = EnvBool(False)
    # Positive cache TTL for filesystem metadata lookups (-1 disables positive expiration)
    SGLANG_HICACHE_FILE_BACKEND_METADATA_TTL = EnvFloat(5.0)
    SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR = EnvStr(None)
    SGLANG_HICACHE_BLOB_BACKEND_STORAGE_DIR = EnvStr(None)
    # Enable O_DIRECT when opening NIXL POSIX backend files (bypasses OS page cache).
    # Disable with SGLANG_HICACHE_NIXL_USE_DIRECT_IO=0 or via the
    # "use_direct_io": false key in --hicache-storage-backend-extra-config.
    SGLANG_HICACHE_NIXL_USE_DIRECT_IO = EnvBool(True)
    SGLANG_HUGEPAGE_SIZE = EnvStr("")
    # Fail hard instead of silently falling back to base pages when the
    # SGLANG_HUGEPAGE_SIZE backing cannot be provided (bad size string,
    # hugetlb mmap failure, THP coverage below ~98%).
    SGLANG_HUGEPAGE_STRICT = EnvBool(False)
    # Pools smaller than this many bytes stay on base pages even when
    # SGLANG_HUGEPAGE_SIZE is set (v16-memory-plan sizing: only the large
    # decode hisparse host pool is hugetlb-backed; hicache/shadow pools on
    # base pages). The register-size fix makes small hugetlb pools pin fine,
    # so this is tunable; default keeps the v16 behavior (32 GiB).
    SGLANG_HUGEPAGE_MIN_BYTES = EnvInt(32 * (1 << 30))
    # Disable transparent hugepages for the whole engine process tree at init
    # (prctl PR_SET_THP_DISABLE, inherited by children). Stops khugepaged/
    # kcompactd churn on non-pool host allocations while the pools themselves
    # use explicit hugepages via SGLANG_HUGEPAGE_SIZE.
    SGLANG_DISABLE_THP = EnvBool(False)
    # Back host KV pools with MAP_PRIVATE anonymous pages (huge pages off) instead
    # of MAP_SHARED ones: kernel memory compaction skips pinned anonymous pages but
    # unmaps pinned shared ones on every failed migration, stalling GPU access.
    SGLANG_MAP_HOST_POOL_PRIVATE = EnvBool(False)
    # Raise (instead of warn) at boot when a host KV pool has <99% of its
    # populated pages on the GPU-local NUMA node (mmap_allocator check).
    SGLANG_NUMA_LOCALITY_STRICT = EnvBool(False)

    # ===================================================================
    # KV-transfer staging and Mooncake transport
    # ===================================================================
    # Staging buffer for heterogeneous TP KV transfer
    SGLANG_DISAGG_STAGING_BUFFER = EnvBool(False)
    SGLANG_DISAGG_STAGING_POOL_SIZE_MB = EnvInt(4096)
    # TODO(yangminl): remove SGLANG_STAGING_USE_TORCH and the torch fallback in
    # staging_buffer.py once Triton kernels are fully validated in production.
    SGLANG_STAGING_USE_TORCH = EnvBool(False)
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL = EnvStr(None)
    ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE = EnvBool(False)
    ASCEND_NPU_PHY_ID = EnvInt(-1)
    SGLANG_MOONCAKE_SEND_AUX_TCP = EnvBool(False)
    SGLANG_ENABLE_FAILED_SESSION_PROBE = EnvBool(False)
    SGLANG_FAILED_SESSION_PROBE_INTERVAL_S = EnvFloat(30.0)

    # ===================================================================
    # Mooncake store
    # ===================================================================
    SGLANG_HICACHE_MOONCAKE_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_MOONCAKE_REUSE_TE = EnvBool(True)
    MOONCAKE_MASTER = EnvStr(None)
    MOONCAKE_CLIENT = EnvStr(None)
    MOONCAKE_LOCAL_HOSTNAME = EnvStr("localhost")
    MOONCAKE_TE_META_DATA_SERVER = EnvStr("P2PHANDSHAKE")
    MOONCAKE_GLOBAL_SEGMENT_SIZE = EnvStr("4gb")
    MOONCAKE_PROTOCOL = EnvStr("rdma")
    MOONCAKE_DEVICE = EnvStr("")
    MOONCAKE_MASTER_METRICS_PORT = EnvInt(9003)
    MOONCAKE_CHECK_SERVER = EnvBool(False)
    MOONCAKE_STANDALONE_STORAGE = EnvBool(False)
    MOONCAKE_ENABLE_SSD_OFFLOAD = EnvBool(False)
    MOONCAKE_OFFLOAD_FILE_STORAGE_PATH = EnvStr(None)
    MOONCAKE_TENANT_ID = EnvStr("default")

    # ===================================================================
    # MoRI transport and expert dispatch
    # ===================================================================
    # Send CPU-resident AUX data via RDMA instead of ZMQ TCP (default: TCP).
    SGLANG_MORI_SEND_AUX_RDMA = EnvBool(False)
    # Number of RDMA Queue Pairs (QPs) used per transfer operation. Higher
    # values can increase parallelism and bandwidth utilization.
    SGLANG_MORI_QP_PER_TRANSFER = EnvInt(4)
    # Number of RDMA work requests posted in a single batch to each QP. Larger
    # batch sizes reduce per-operation overhead and improve throughput at the
    # cost of higher latency. -1 selects automatic sizing based on the number
    # of merged work requests and available endpoints.
    SGLANG_MORI_POST_BATCH_SIZE = EnvInt(-1)
    # Number of worker threads in the RDMA executor thread pool. More workers
    # can improve parallelism for large batch transfers across multiple QPs,
    # but excessive threads may cause contention.
    SGLANG_MORI_NUM_WORKERS = EnvInt(4)
    # Number of sharded synchronous worker threads that drain KV transfers.
    # Also the bound on outstanding (posted-but-not-completed) transfers, so it
    # is the primary throttle keeping the RDMA send queue from overflowing.
    SGLANG_MORI_TRANSFER_SHARDS = EnvInt(8)
    # Poll cadence (ms) at which a transfer worker wakes to check the SLA while
    # waiting for completion; real completion still wakes it immediately.
    SGLANG_MORI_WAIT_POLL_MS = EnvInt(1000)
    # Per-transfer SLA (ms) before a KV transfer is failed; 0 disables the SLA
    # and relies on the RDMA retry-exceeded timeout only.
    SGLANG_MORI_TRANSFER_TIMEOUT_MS = EnvInt(0)
    SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(4096)

    # ===================================================================
    # AMD, ROCm, and AITER
    # ===================================================================
    SGLANG_USE_AITER = EnvBool(False)
    SGLANG_USE_AITER_AG = EnvBool(True)
    # Use reduce_scatter (instead of all_reduce + dp_scatter) for the equal-chunk
    # MAX_LEN DP-MoE combine. Default ON for ROCm/HIP (uses the aiter custom
    # symmetric-memory kernel), OFF elsewhere (would fall back to RCCL); override
    # explicitly to force on/off on any platform.
    SGLANG_DP_USE_REDUCE_SCATTER = EnvBool(_default_hip)
    # Quantize the variable-length DP-MoE gather payload (SGLANG_DP_USE_GATHERV
    # path, prefill/extend only) to fp8-e4m3 with per-token-group-128 scales:
    # halves the gathered hidden-state bytes over NCCL; the combine
    # (reduce_scatterv) leg stays bf16 (NCCL SUM cannot run on fp8).  Lossy on
    # the wire — same group quantization the MoE expert GEMMs apply to their
    # input anyway, but router/shared-expert reads see rounded values, so this
    # stays accuracy-gated and default OFF.
    SGLANG_ENABLE_DP_GATHER_FP8 = EnvBool(False)
    SGLANG_USE_AITER_UNIFIED_ATTN = EnvBool(False)
    # Select the gate/up tile layout for AITER MoE: True -> interleave
    # (matches FlyDSL `gate_mode="interleave"` kernels), False -> separated
    # (matches `gate_mode="separated"`, the layout used by gptoss_fp4 tuned
    # configs and by Mxfp4MoEMethod's post-fix weight shuffle).
    SGLANG_USE_AITER_MOE_GU_ITLV = EnvBool(True)
    # Fold `silu(gate) * up` into the triton MoE up-GEMM epilogue. W13 rows are
    # permuted in place at load so gate/up land in adjacent columns of the same
    # output tile, which removes intermediate_cache1 and the standalone
    # activation launch per MoE layer. Opt-in because the in-place permute is
    # not compatible with runtime weight updates or EPLB expert rearrangement,
    # both of which assume the checkpoint's halves layout.
    SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED = EnvBool(False)
    # Fuse the `residual_add + RMSNorm + zero-pad` triplet that appears
    # before the MoE block for models whose MoE input hidden_size must be
    # padded up to a stride (e.g. GPT-OSS MXFP4 needs pad to multiple of
    # 256). When False (default) the pad runs as a separate
    # torch.nn.functional.pad call inside the MoE method. When True, the
    # aiter Triton kernel `fused_add_rmsnorm_pad` produces a padded
    # post-attention layernorm output in one launch and the MoE method
    # skips the explicit pad. Currently only takes effect on the
    # post_attention_layernorm path with aiter backend and TP=1.
    SGLANG_AITER_FUSE_RMSNORM_PAD = EnvBool(False)
    # Physical layout for MHA KV cache. "nhd" (default) keeps the existing
    # (size, head_num, head_dim) per-token storage that
    # `aiter.mha.mha_batch_prefill_func`/`unified_attention` consume directly.
    # "vectorized_5d" allocates K as (num_blocks, H_kv, head_dim/x, page_size, x)
    # and V as (num_blocks, H_kv, page_size/x, head_dim, x) (x = 16 / dtype_size),
    # matching the SHUFFLE layout that aiter's CK FmhaBatchPrefill kernel and
    # `aiter.ops.triton.gluon.pa_decode_gluon` both consume natively. This is
    # the SHUFFLE KV layout that enables pa_decode_gluon for full-attn
    # decode without runtime permutes.
    SGLANG_AITER_KV_CACHE_LAYOUT = EnvStr("nhd")
    SGLANG_ROCM_FUSED_DECODE_MLA = EnvBool(False)
    SGLANG_ROCM_DISABLE_LINEARQUANT = EnvBool(False)
    USE_ROCM_AITER_ROPE_BACKEND = EnvStr("0")
    # Enable dual-stream MoE (shared experts vs routed experts) on the
    # ROCm/AITER path. Requires GPU_MAX_HW_QUEUES>=5 to avoid HW-queue serialization.
    SGLANG_ROCM_USE_MULTI_STREAM = EnvBool(False)
    SGLANG_HACK_FLASHMLA_BACKEND = EnvStr("tilelang")
    SGLANG_USE_AITER_FP8_PER_TOKEN = EnvBool(False)

    # DSV4 Aiter flags
    SGLANG_OPT_USE_AITER_SILU_MUL = EnvBool(False)
    SGLANG_OPT_USE_FUSED_QK_NORM_ROPE = EnvBool(True)
    SGLANG_OPT_USE_AITER_INDEXER = EnvBool(False)

    # ===================================================================
    # Apple Silicon and MLX
    # ===================================================================
    SGLANG_USE_MLX = EnvBool(False)
    SGLANG_MLX_USE_CUSTOM_ROPE = EnvBool(False)
    SGLANG_MLX_FUSE_SWIGLU = EnvBool(False)
    # Number of decode steps between periodic mx.clear_cache() calls.
    # Set to 0 to disable cache clearing entirely.
    SGLANG_MLX_CLEAR_CACHE_STEPS = EnvInt(256)
    # MLX buffer-cache cap in GB.
    SGLANG_MLX_CACHE_LIMIT_GB = EnvFloat(None)

    # ===================================================================
    # Ascend NPU
    # ===================================================================
    SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT = EnvBool(False)
    SGLANG_NPU_USE_MULTI_STREAM = EnvBool(False)
    SGLANG_NPU_USE_MLAPO = EnvBool(False)
    # Forward native implementation for activation gelu tanh for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GELUTANH = EnvBool(False)
    # Forward native implementation for gemma rms norm for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM = EnvBool(False)
    # Delay all-gather after qlora for better performance for Deepseek v3.2
    SGLANG_USE_AG_AFTER_QLORA = EnvBool(False)
    # Enable int4x2 weights loading
    SGLANG_NPU_W4A4_NEW_PACKING = EnvBool(False)
    # Keep K3 shared experts and dense MLPs sharded over attention TP.
    SGLANG_K3_SHARED_EXPERTS_ATTN_TP = EnvBool(False)
    SGLANG_K3_DENSE_MLP_ATTN_TP = EnvBool(False)
    # Use the graph-safe Triton-Ascend kernel for masked speculative KV commits.
    SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_NPU_USE_TRITON_KV_CACHE_STORE"
    )
    # Quantize x to int8 in the dispatch operator (vendor alias consumed by the
    # Ascend DeepEP library; the MTP draft-build scopes override it to False).
    DEEP_NORMAL_MODE_USE_INT8_QUANT = EnvBool(False)
    SGLANG_ZBAL_LOCAL_MEM_SIZE = EnvInt(0)
    SGLANG_ZBAL_BOOTSTRAP_URL = EnvStr("")

    # ===================================================================
    # MUSA
    # ===================================================================
    SGLANG_MUSA_FA3_FORCE_UPDATE_METADATA = EnvBool(False)

    # ===================================================================
    # Quantization
    # ===================================================================
    SGLANG_INT4_WEIGHT = EnvBool(False)
    SGLANG_CPU_QUANTIZATION = EnvBool(False)
    SGLANG_USE_DYNAMIC_MXFP4_LINEAR = EnvBool(False)
    SGLANG_FORCE_FP8_MARLIN = EnvBool(False)
    SGLANG_MOE_NVFP4_DISPATCH = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE = EnvBool(False)
    SGLANG_QUANT_ALLOW_DOWNCASTING = EnvBool(False)
    SGLANG_FP8_IGNORED_LAYERS = EnvStr("")
    SGLANG_FP4_IGNORED_LAYERS = EnvStr("")
    # On by default; set SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE=0 as a kill switch.
    # Consults the tuned per-(N, K, M) Triton tile config table in
    # apply_fp8_linear. When a tuned config exists for this GPU / weight shape /
    # token count, run the Triton w8a8 FP8 GEMM with it; otherwise keep the
    # default CUTLASS path. Only takes effect on a GPU with a matching
    # dtype=fp8_w8a8_channelwise config JSON under
    # kernels/ops/quantization/configs/ (currently L40S), so it is a no-op on
    # any other GPU / untuned shape even when enabled.
    SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE = EnvBool(True)

    # ===================================================================
    # Humming quantization
    # ===================================================================
    SGLANG_HUMMING_ONLINE_QUANT_CONFIG = EnvJSON(None)
    SGLANG_HUMMING_INPUT_QUANT_CONFIG = EnvJSON(None)
    SGLANG_HUMMING_USE_F16_ACCUM = EnvBool(False)
    SGLANG_HUMMING_MOE_GEMM_TYPE = EnvStr("")

    # ===================================================================
    # FlashInfer, FlashMLA, and TRT-LLM
    # ===================================================================
    SGLANG_IS_FLASHINFER_AVAILABLE = EnvBool(True)
    SGLANG_FLASHINFER_USE_PAGED = EnvBool(False)
    # Default to the pick from flashinfer
    SGLANG_FLASHINFER_WORKSPACE_SIZE = EnvInt(384 * 1024 * 1024)
    # Per-rank dispatch capacity of the FlashInfer MoE A2A dispatcher. Unset
    # means each call site keeps its own default.
    SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(None)
    # Enable per-token FP32 activation scaling for serialized ModelOpt FP4 with
    # FlashInfer TRT-LLM or CuTe DSL v2 MoE.
    SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION = EnvBool(False)
    # Launch the TRT-LLM MoE grouped GEMMs with PDL only at or below this
    # token count.
    SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS = EnvInt(8192)
    # Use FlashInfer's fused atomic CUTLASS/CuTe DSL MoE finalize.
    SGLANG_FLASHINFER_MOE_FUSED_FINALIZE = EnvBool(True)
    # Master switch for the experimental TRT-LLM LoRA fast path; when OFF (default) every
    # fine-grained opt switch reads False, keeping non-experimental paths byte-identical.
    SGLANG_EXPERIMENTAL_LORA_OPTI = EnvBool(False)
    # SGLang needs to know FlashInfer NVFP4 4over6 config to compute the global scale factor.
    FLASHINFER_NVFP4_4OVER6 = EnvBool(False)
    FLASHINFER_NVFP4_4OVER6_E4M3_USE_256 = EnvBool(False)
    # Skip-softmax threshold scale factor for TRT-LLM attention (prefill and decode separately).
    # None = standard attention. See https://arxiv.org/abs/2512.12087
    SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    # SM120 FlashMLA decode backend: "flashinfer" (default), "triton", or "torch".
    SGLANG_SM120_FLASHMLA_BACKEND = EnvStr("flashinfer")
    SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE = EnvInt(4096)
    SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE = EnvInt(2048)
    SGLANG_FLASHINFER_AUTOTUNE_CACHE = EnvBool(True)
    # Also autotune one EXTEND-shaped dummy at max_prefill_tokens during
    # warmup. Opt-in: the extra forward needs transient activation headroom
    # that small-VRAM or tightly-packed configs may not have.
    SGLANG_FLASHINFER_AUTOTUNE_EXTEND = EnvBool(False)

    # ===================================================================
    # Triton and Torch compilation
    # ===================================================================
    SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS = EnvBool(False)
    SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE = EnvBool(False)
    # Compact extend-attention query-tile grid: AMD/HIP-only optimization
    # (parity with flash-attn's ragged-aware launch). The feature checks _is_hip
    # explicitly in code; this env var allows override (0=force off, 1=force on).
    SGLANG_TRITON_COMPACT_EXTEND_ATTENTION = EnvBool(True)
    # Raise if Triton loads a kernel after the engine starts serving. This
    # verifies that startup warmup covers every kernel specialization used at
    # serving time.
    SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY = EnvBool(False)
    SGLANG_TRITON_SLOW_COMPILE_THRESHOLD_SECS = EnvFloat(1.0)
    SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB = EnvFloat(1.0)
    # gfx950 MLA decode stage-1: pick the launch geometry and split count per batch.
    # Reorders the fp32 accumulation, so off by default.
    SGLANG_MLA_DECODE_TUNE = EnvBool(False)
    SGLANG_ENABLE_TORCH_COMPILE = EnvBool(False)
    SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE = EnvInt(4096)
    SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE = EnvInt(256)

    # ===================================================================
    # Expert parallel load balancing
    # ===================================================================
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_CANARY = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS = EnvBool(False)
    SGLANG_LOG_EXPERT_LOCATION_METADATA = EnvBool(False)
    SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR = EnvStr("/tmp")
    SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL = EnvInt(0)
    # Chunk size for the rebalance expert-weight P2P exchange; set
    # >= num_physical_experts to submit a single batch_isend_irecv.
    SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE = EnvIntWithAlias(
        32, deprecated_name="SGLANG_EPLB_ROCM_P2P_BATCH_CHUNK_SIZE"
    )

    # ===================================================================
    # DeepGEMM
    # ===================================================================
    SGLANG_ENABLE_JIT_DEEPGEMM = EnvBool(True)
    SGLANG_DEEPGEMM_STANDARD_LAYOUT = EnvStr("auto")
    SGLANG_DEEPGEMM_MASKED_MEMORY_BUDGET_FRACTION = EnvFloat(0.25)
    # Cap the DeepGEMM masked grouped-GEMM per-expert padded capacity at
    # round_up(max(masked_m), 256) instead of round_up(rank_tokens, 256):
    # shrinks the [num_local_experts, m, *] MoE intermediates ~4x under
    # load imbalance (they otherwise OOM saturated --moe-runner-backend
    # deep_gemm serving).  Costs one D2H sync per MoE layer.
    SGLANG_OPT_DG_MASKED_M_CAP = EnvBool(False)
    # Drop dp-attention MAX_LEN pad rows from MoE dispatch (StandardDispatcher
    # post-translation topk_ids -> -1): pad rows otherwise run the router on
    # stale hidden values and burn expert compute whose outputs are discarded;
    # colliding pad top-ks also inflate the DeepGEMM masked-GEMM workspace to
    # OOM at saturation.  Capture-safe (reads only global_num_tokens_gpu).
    SGLANG_OPT_MASK_DP_PAD_MOE = EnvBool(False)
    SGLANG_JIT_DEEPGEMM_PRECOMPILE = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_FAST_WARMUP = EnvBool(False)
    SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS = EnvInt(4)
    SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE = EnvBool(False)
    # Resolved lazily so it tracks SGLANG_CACHE_DIR, which is defined below.
    SGLANG_DG_CACHE_DIR = EnvStr(lambda: _default_cache_subdir("deep_gemm"))
    SGLANG_DG_USE_NVRTC = EnvBool(False)
    SGLANG_USE_DEEPGEMM_BMM = EnvBool(False)
    SGLANG_DEEPGEMM_SANITY_CHECK = EnvBool(False)
    SGLANG_DEEPGEMM_PDL = EnvBool(True)
    SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP = EnvBool(False)

    # ===================================================================
    # Cache directories
    # ===================================================================
    SGLANG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/sglang"))
    # JIT kernel build cache. None = unset, resolving to ~/.cache/sglang/jit;
    # point it at a persistent mount to share builds across CI jobs.
    SGLANG_JIT_CACHE_DIR = EnvStr(None)
    # Log, at INFO, which dependency changed whenever a module is rebuilt.
    SGLANG_JIT_CACHE_DEBUG = EnvBool(False)
    # How many builds to keep per module variant. None = unset = keep all, which
    # is what makes reverting an edit an instant hit instead of a rebuild; set
    # it to trade that away for disk (1 keeps only the most recent build).
    SGLANG_JIT_CACHE_KEEP = EnvInt(None)

    # ===================================================================
    # Expert-parallel dispatch and MoE execution
    # ===================================================================
    # Deprecated in favor of '--deepep-dispatcher-output-dtype bf16' but still
    # read by several call sites; do not use in new code.
    SGLANG_DEEPEP_BF16_DISPATCH = EnvBool(False)
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS = EnvInt(32)
    # Lane deepep-v2: V2 ElasticBuffer sizing (max tokens per rank in a step;
    # chunked-prefill-size per DP rank on the prefill arm).
    SGLANG_DEEPEP_V2_NUM_MAX_TOKENS_PER_RANK = EnvInt(8192)
    SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO = EnvBool(False)
    # Force dynamic Waterfill with runtime EP all-reduce instead of the default
    # static local-batch path.
    SGLANG_DISABLE_STATIC_WATERFILL = EnvBool(False)
    SGLANG_NIXL_EP_BF16_DISPATCH = EnvBool(False)
    SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_PPLX_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_ENABLE_MOE_DEFERRED_FINALIZE = EnvBool(True)
    # DeepSeek/GLM MoE (deepseek_v2.py): quantize the (dp-gathered) MoE input
    # to per-token-group-128 fp8 ONCE and feed both the fused shared-expert
    # GEMM (cutlass w8a8 linear) and the routed experts' triton fused runner,
    # instead of quantizing the same [T, hidden] tensor twice with different
    # scale layouts. Only engages on CUDA with fp8 block-128 weights, the
    # standard dispatcher, and the triton MoE runner; falls back silently
    # otherwise.
    SGLANG_OPT_MOE_QUANT_ONCE = EnvBool(False)

    # GLM-5.3 small-M decode GEMMs: route the fp8 block-128 W8A8 GEMM
    # through the JIT CUTLASS sm90 blockwise kernel (ex-67 recipe, swapAB
    # orientation, variant 5 = cooperative 128x16x128) instead of DeepGEMM
    # for the measured-winning (N, K) shapes at small M (GH200 bench,
    # lane w8a16-gemm 2026-09-02; outputs bit-exact vs DeepGEMM):
    #   o_proj 16384->6144  0.86x/0.92x/0.92x at M=1/4/16
    #   d_dn   12288->6144  0.91x/0.94x/0.94x
    #   sh_gu   6144->4096  0.96x/0.93x/1.10x (M<=4 only)
    # No effect on other shapes/M or on non-sm90 CUDA.
    SGLANG_GLM_FP8_BLOCKWISE_SMALLM_GEMM = EnvBool(False)

    # Megakernel MoE (doublewordai/megakernel): per-rank decode token capacity
    SGLANG_MEGAKERNEL_NUM_MAX_TOKENS_PER_RANK = EnvInt(64)

    # ===================================================================
    # DeepGEMM Mega MoE
    # ===================================================================
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK = EnvInt(8192)
    # When set, the mega-MoE x slot is packed E2M1 (FP4) instead of FP8 E4M3.
    # Halves symm-buffer footprint and unlocks the MXF4 mainloop downstream.
    # Setting this also exports DG_USE_FP4_ACTS=1 so DeepGEMM's symm-buffer
    # sizing + fp8_fp4_mega_moe pick up the FP4 layout.
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS = EnvBool(False)
    # Switches the L1+L2 mainloops from kind::mxf8f6f4 (K=32 with-padding) to
    # kind::mxf4 (K=64 dense) inside fp8_fp4_mega_moe. No effect unless
    # SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS is also set; DeepGEMM asserts
    # this combination on the host side.
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND = EnvBool(False)

    # ===================================================================
    # Top-k kernels
    # ===================================================================
    SGLANG_OPT_USE_FUSED_HASH_TOPK = EnvBool(True)
    # Opt-in: route DeepSeek-V3 grouped topk through the unified Triton router
    # instead of the flashinfer/AOT grouped kernels. Off by default (flashinfer is
    # the tuned production path); the Triton path is bit-exact on DeepSeek-V3.2 e2e
    # and benchmarks at parity, so this is a consolidation escape hatch, not a perf flip.
    SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK = EnvBool(False)
    SGLANG_OPT_USE_TOPK_V2 = EnvBool(True)

    # ===================================================================
    # Kernel selection and fused backends
    # ===================================================================
    SGLANG_USE_SGL_FA3_KERNEL = EnvBool(True)
    # Force every sglang.kernels BaseFusedOp onto one backend (a KernelBackend
    # value, e.g. "torch" / "torch_compile" / "triton" / "aot"); unset =
    # auto-select by priority. "torch" flips all fused ops to their pure-torch
    # reference implementations for numerical-bug bisection.
    SGLANG_FORCE_FUSED_OP_BACKEND = EnvStr(None)
    USE_TRITON_W8A8_FP8_KERNEL = EnvBool(False)
    SGLANG_MOE_PADDING = EnvBool(False)

    # ===================================================================
    # Logits and log-probability processing
    # ===================================================================
    SGLANG_RETURN_ORIGINAL_LOGPROB = EnvBool(False)
    # Sanitize NaN logits before sampling kernels and log a throttled warning
    # (see sanitize_nan_logits).
    SGLANG_SANITIZE_NAN_LOGITS = EnvBool(False)
    SGLANG_ENABLE_LOGPROB_CHUNK = EnvBoolWithAlias(
        True, deprecated_name="SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK"
    )
    SGLANG_LOGPROB_CHUNK_SIZE = EnvIntWithAlias(
        2048, deprecated_name="SGLANG_LOGITS_PROCESSER_CHUNK_SIZE"
    )
    # Compute input logprobs from logits via per-row logsumexp instead of
    # materializing the full-vocab log-softmax. Escape hatch only; the two
    # paths are mathematically identical.
    SGLANG_ENABLE_FAST_INPUT_LOGPROBS = EnvBool(True)

    # ===================================================================
    # Deterministic inference and all-reduce
    # ===================================================================
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE = EnvBool(False)
    # Use 1-stage all-reduce kernel on AMD (deterministic, fixed accumulation order)
    # If not set: auto (enabled when --enable-deterministic-inference is on)
    # Set to 1: force enable (even without --enable-deterministic-inference)
    # Set to 0: force disable (use default Aiter AR even with --enable-deterministic-inference)
    SGLANG_USE_1STAGE_ALLREDUCE = EnvBool(False)
    # NCCL channel count pinned on CUDA so the all-reduce reduces a token the
    # same way whatever else shares its batch. Raise it to buy back bandwidth
    # on links that can drive more channels.
    SGLANG_DETERMINISTIC_NCCL_NCHANNELS = EnvInt(8)
    SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2 = EnvBool(True)
    # Default per-direction workspace cap for CustomAllReduceV2; explicit
    # constructor sizes take precedence over this.
    SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB = EnvInt(16 * 1024)
    SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PULL_SIZE_KB = EnvInt(None)
    SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB = EnvInt(None)
    # Allow CustomAllReduceV2 on a process group that spans nodes (MNNVL
    # fabric). Requires torch symmetric memory to rendezvous across nodes
    # (fabric handles + IMEX). Graph zero-copy input registration is not
    # supported in this mode and is disabled; all-reduce inside CUDA graphs
    # falls back to eager pull from the symm workspace. Auto-enabled on
    # MNNVL-fabric devices (GB200/GB300) when nnodes > 1; set 0/1 to
    # override in either direction.
    SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE = EnvBool(False)

    # ===================================================================
    # RoPE cache
    # ===================================================================
    SGLANG_SPEC_EXPANSION_SAFETY_FACTOR = EnvInt(2)
    SGLANG_ROPE_CACHE_FP32 = EnvBool(False)
    SGLANG_ROPE_CACHE_SAFETY_MARGIN = EnvInt(256)
    SGLANG_ROPE_CACHE_ALIGN = EnvInt(128)

    # ===================================================================
    # Speculative decoding
    # ===================================================================
    SGLANG_ENABLE_OVERLAP_PLAN_STREAM = EnvBool(False)
    # A/B: keep the DFLASH draft greedy head eager (not folded in-graph).
    SGLANG_DFLASH_EAGER_DRAFT_SAMPLER = EnvBool(False)
    SGLANG_RAGGED_VERIFY_MODE = EnvStr("static")
    # EAGLE adaptive verify (lane/adaptive-spec): per-request verify lengths from
    # draft confidence + an SPS cost table. Requires SGLANG_RAGGED_VERIFY_MODE=compact.
    SGLANG_EAGLE_ADAPTIVE_VERIFY = EnvBool(False)
    # Lane M3: force every request's verify_len (grid timing measurement).
    SGLANG_EAGLE_FORCE_VERIFY_LEN = EnvInt(0)
    # Lane M3: append per-step verify timing rows to this jsonl path.
    SGLANG_EAGLE_VERIFY_TIMING = EnvStr("")
    # Lane A/B: cap swap-in positions per layer in the ragged verify page
    # table build (measurement only; unsafe for mixed-vl graph replays).
    SGLANG_EAGLE_SWAPIN_MAXPOS = EnvInt(0)
    # Path to an SPS cost table JSON (SpsCostTable or SpsAdditiveCostTable format,
    # see dspark_sps.py) for the EAGLE adaptive verify budget scheduler. Without
    # it the schedule degenerates to verify-all (full width through the ragged
    # graphs) and a warning is logged.
    SGLANG_EAGLE_SPS_TABLE = EnvStr("")
    SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE = EnvBool(False)
    # Lane debug: log ragged verify page-table shapes/values (one line per call).
    SGLANG_RAGGED_DEBUG = EnvBool(False)
    # Skip draft_extend while adaptive spec is at steps=0 (drafting disabled).
    # Saves the per-step draft forward, but the draft KV goes stale: an upshift
    # back to steps>0 starts from a cold draft state (low accept until it recovers).
    SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND = EnvBool(False)
    # Kill-switch for the draft-extend cuda graph. Draft extend then always runs
    # eager. Escape hatch for setups where the capture's memory pool costs more
    # than the graph saves (e.g. DeepEP MoE workspace captured at full dispatch
    # capacity).
    SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH = EnvBool(False)
    # Use the split-KV (flash-decode) kernel for EAGLE target-verify on the
    # Triton backend (ROCm). Only active at speculative topk == 1; falls back to
    # extend_attention_fwd for unsupported cases or when set false (e.g. for
    # debugging). Correctness is unaffected; this only changes performance.
    SGLANG_ENABLE_SPLITKV_VERIFY = EnvBool(True)
    SGLANG_NGRAM_FORCE_GREEDY_VERIFY = EnvBool(False)

    # ===================================================================
    # Multimodal processing
    # ===================================================================
    SGLANG_VLM_CACHE_SIZE_MB = EnvInt(100)
    SGLANG_IMAGE_MAX_PIXELS = EnvInt(16384 * 28 * 28)
    SGLANG_RESIZE_RESAMPLE = EnvStr("")
    SGLANG_MM_BUFFER_SIZE_MB = EnvInt(0)
    SGLANG_MM_PRECOMPUTE_HASH = EnvBool(False)
    SGLANG_VIT_ENABLE_CUDA_GRAPH = EnvBool(False)
    # Use the fully-vectorized ViT position-embedding interpolation (no per-image
    # Python loop / CPU<->GPU sync). Bit-exact with the legacy implementation;
    # set False to fall back to the per-image loop.
    SGLANG_VIT_ENABLE_VECTORIZED_POS_EMBED = EnvBool(True)
    SGLANG_MM_SKIP_COMPUTE_HASH = EnvBool(False)
    # For pre-tokenized (list[int]) multimodal prompts,
    # preserve the user's original tokens to avoid retokenization drift.
    SGLANG_MM_AVOID_RETOKENIZE = EnvBool(True)

    # ===================================================================
    # Multimodal CUDA IPC transport
    # ===================================================================
    SGLANG_USE_CUDA_IPC_TRANSPORT = EnvBool(False)
    # Reuse the mapping for the already-allocated bounded CUDA IPC pool. This
    # has no effect unless CUDA IPC feature transport is explicitly selected.
    SGLANG_USE_IPC_POOL_HANDLE_CACHE = EnvBool(True)
    SGLANG_MM_FEATURE_CACHE_MB = EnvInt(1 * 1024)
    SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC = EnvFloat(0.05)

    # ===================================================================
    # Mamba state and cache
    # ===================================================================
    SGLANG_MAMBA_CONV_DTYPE = EnvStr("bfloat16")
    SGLANG_MAMBA_SSM_DTYPE = EnvStr(None)
    # Kill-switch for the fused per-slot conv clear/copy kernel (MambaPool);
    # falls back to the per-conv-type Python loop.
    SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS = EnvBool(False)
    # Opt-in: on the unified radix tree, leave the matched-prefix mamba evictable
    # during decode (it is already COW'd to the request's own slot) and shrink the
    # mamba pool ratio accordingly. Frees one resident slot per running request,
    # raising max_running_requests. Off = original locking + ratio (escape hatch).
    SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK = EnvBool(False)

    # ===================================================================
    # CUDA graphs and execution buffers
    # ===================================================================
    SGLANG_USE_BREAKABLE_CUDA_GRAPH = EnvBool(False)
    # Guards CUDA graph executable dedup via cudaGraphExecUpdate.
    SGLANG_ENABLE_CUDA_GRAPH_DEDUP = EnvBool(False)
    SGLANG_MEMORY_SAVER_CUDA_GRAPH = EnvBool(False)
    # Eager forward wraps the ForwardBatch's own tensors instead of copying them
    # into the CUDA graph buffer registry (no per-iter device-to-device copy).
    SGLANG_EAGER_INPUT_NO_COPY = EnvBool(False)

    # ===================================================================
    # Tokenizer, request state, embeddings, and reasoning controls
    # ===================================================================
    SGLANG_EMBEDDINGS_SPARSE_HEAD = EnvStr(None)
    # Think tokens budget: negative means unlimited, >= 0 caps thinking tokens
    SGLANG_MAX_THINK_TOKENS = EnvInt(-1)
    SGLANG_PATCH_TOKENIZER = EnvBool(True)
    SGLANG_REQUEST_STATE_WAIT_TIMEOUT = EnvInt(4)
    SGLANG_DEFAULT_THINKING = EnvBool(False)

    # ===================================================================
    # Encoder pipeline and disaggregation
    # ===================================================================
    SGLANG_ENCODER_GRPC_TIMEOUT_SECS = EnvInt(60)
    # Encoder receiver selection: http|grpc (used by EPD paths).
    SGLANG_ENCODER_MM_RECEIVER_MODE = EnvStr("http")
    SGLANG_ENCODER_RECV_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_SEND_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_HTTP_TIMEOUT = EnvFloat(1800.0)
    SGLANG_ENCODER_REQ_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_DISPATCH_MIN_ITEMS = EnvInt(2)
    SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU = EnvBool(False)
    SGLANG_ENCODER_MAX_BATCH_SIZE = EnvInt(8)
    SGLANG_ENCODER_PREPROC_WORKERS = EnvInt(8)
    # EncoderBootstrapServer health-check tuning.  Interval == 0 disables it.
    SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL = EnvFloat(10.0)
    SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_TIMEOUT = EnvFloat(2.0)
    # Seconds before permanently dropping an unhealthy encoder (0 = keep probing).
    SGLANG_ENCODER_BOOTSTRAP_EVICTED_TTL = EnvFloat(600.0)
    # Persistent receiver-side GPU embedding pool size for mooncake EPD transport.
    # 0 disables (per-request register/deregister). 4096 = 4GB default per TP
    SGLANG_EMBEDDING_POOL_SIZE_MB = EnvInt(4096)
    SGLANG_ENCODER_DP_WORKER_MAX_INFLIGHT = EnvInt(64)

    # ===================================================================
    # Native gRPC server
    # ===================================================================
    # Native gRPC server. SGLANG_GRPC_PORT is the env fallback for the
    # --grpc-port CLI flag; setting either enables the native server alongside
    # HTTP. The worker-threads knob stays env-only (internal tuning, no CLI
    # surface).
    SGLANG_GRPC_PORT = EnvInt(None)
    SGLANG_GRPC_WORKER_THREADS = EnvInt(4)

    # ===================================================================
    # NUMA and CPU affinity
    # ===================================================================
    SGLANG_SET_CPU_AFFINITY = EnvBool(False)
    SGLANG_NUMA_BIND_V2 = EnvBool(True)
    SGLANG_AUTO_NUMA_BIND = EnvBool(True)
    SGLANG_CRASH_ON_NUMA_BIND_FAILURE = EnvBool(False)

    # ===================================================================
    # DeepSeek V4
    # ===================================================================

    # Model and Quantization
    # Set False when using FP4-to-FP8 converted DeepSeek V4 checkpoint.
    SGLANG_DSV4_FP4_EXPERTS = EnvBool(True)
    # Set True to dequantize the FP4 experts to FP8 at runtime
    SGLANG_DSV4_FP4_DEQUANT = EnvBool(False)
    # Flash-0731 also accepts "low"; the active profile is checkpoint-resolved.
    SGLANG_DSV4_REASONING_EFFORT = EnvStr("")
    # Quantize the SWA fp8 KV cache from bf16-rounded values (matches
    # trainer-side QAT and the DSA-CP path) instead of fp32 registers.
    SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE = EnvBool(False)

    # Kernels and indexer
    SGLANG_OPT_DEEPGEMM_HC_PRENORM = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_MHC_PRE = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_MHC_POST = EnvBool(True)
    SGLANG_OPT_USE_FLASHINFER_MHC = EnvBool(False)
    SGLANG_OPT_FUSE_MHC_POST_PRE = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_INDEXER = EnvBool(False)
    SGLANG_OPT_DSV4_NONPAGED_INDEXER = EnvBool(True)
    # Per-rank local query rows (after DP-attention sharding when enabled),
    # not request ISL.
    SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS = EnvInt(8192)
    SGLANG_OPT_USE_JIT_INDEXER_METADATA = EnvBool(True)
    SGLANG_OPT_USE_ONLINE_COMPRESS = EnvBool(False)
    SGLANG_EXPERIMENTAL_ONLINE_C128_MTP = EnvBool(False)
    SGLANG_DSV4_COMPRESS_STATE_DTYPE = EnvStr("float32")
    SGLANG_FP8_PAGED_MQA_LOGITS_TORCH = EnvBool(False)
    SGLANG_OPT_FLASHMLA_SPARSE_PREFILL = EnvBool(True)

    # cache, GEMM, and distributed
    SGLANG_OPT_FP8_WO_A_GEMM = EnvBool(True)
    SGLANG_OPT_BF16_FP32_GEMM_ALGO = EnvStr("cublas")
    SGLANG_OPT_FUSE_WQA_WKV = EnvBool(True)
    SGLANG_OPT_USE_MULTI_STREAM_OVERLAP = EnvBool(True)

    # ===================================================================
    # Inkling
    # ===================================================================
    SGLANG_OPT_USE_FUSED_GATE_TOPK = EnvBool(True)
    # Inside the fused gate: use the CUDA JIT top-k+renorm kernel (v2) instead
    # of the triton kernel when the production Inkling shape applies.
    SGLANG_OPT_USE_GATE_TOPK_JIT = EnvBool(True)
    # Inside the fused gate: replace the cublas gate linear with the
    # expert-per-block GEMV JIT kernel at small token counts (GateGemvMode).
    SGLANG_OPT_GATE_GEMV_MODE = EnvInt(GateGemvMode.PAIR)
    # Capture all multi-layer EAGLE draft-extend steps and the in-graph chain
    # rotation into ONE CUDA graph instead of one captured graph per step.
    SGLANG_ENABLE_SINGLE_CG_DRAFT = EnvBool(True)
    # Draft sampler uses the Gumbel-max trick (argmax(probs / Exp(1))) instead of
    # torch.multinomial, whose device-side validity assert breaks draft-graph replay.
    SGLANG_OPT_USE_GUMBEL_SAMPLE = EnvBool(True)
    # Multi-layer chain-MTP boundary-KV fix: widen the draft-extend window to
    # rewrite rejected-draft KV rows before reuse (acc_len repair; on by default).
    SGLANG_ENABLE_MTP_BOUNDARY_KV_FIX = EnvBool(True)
    SGLANG_OPT_USE_INKLING_MULTI_STREAM_OVERLAP = EnvBool(True)
    SGLANG_OPT_USE_INKLING_SHEARED_BIAS = EnvBool(True)
    # Use feature-stacked GEMMs for the no-LoRA BF16 shared sink. Eligible LoRA
    # serving enables this layout independently of the flag.
    SGLANG_OPT_LINEARIZED_SHARED_SINK = EnvBool(True)
    # Use the autotuned JIT all-reduce, falling back to torch multimem for
    # shapes where it wins.
    SGLANG_OPT_USE_INKLING_CUSTOM_AR = EnvBool(True)
    # Fuse small-batch decode all-reduce, MLP convolution, and attention norm.
    # Requires the custom all-reduce; other shapes use the unfused path.
    SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV_NORM = EnvBool(True)
    # Fuse eligible extend all-reduce, convolution, and cache updates.
    # Supports scattered or full-width state and requires the custom all-reduce.
    SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV = EnvBool(True)
    # Fuse eligible convolution, QK norm, window, and KV-store prologue work.
    # Non-BF16 caches retain the backend KV store.
    SGLANG_OPT_USE_INKLING_FUSED_ATTN_PROLOGUE = EnvBool(True)
    # Override shared-expert selection: true uses grouped GEMM, false uses BMM.
    # When unset, selection follows model, quantization, and LoRA requirements.
    SGLANG_OPT_USE_INKLING_SHARED_FUSED_MOE = EnvBool(True)
    # Fold the conditional long-context log-scaling tau into its producers
    # instead of separate output-sized scale kernels: the fused attn
    # prologue's q path (bit-exact, before MXFP8 quantization there) and the
    # rel_logits projection's r OPERAND (the diagonal scale commutes through
    # the einsum, shrinking the pass by rel_extent/d_rel = 64x; rounding moves
    # before the GEMM). Flag-off keeps the standalone apply_log_scaling_tau
    # on the outputs.
    # Fold the MoE shared-expert partials into the custom AR kernels instead
    # of a separate {routed + shared} torch.add per MoE layer; some buckets
    # keep a pre-add during the AR stage-in. torch.add numerics
    # (bit-identical). Requires SGLANG_OPT_USE_INKLING_CUSTOM_AR.
    SGLANG_OPT_USE_INKLING_FUSED_AR_SHARED = EnvBool(True)
    SGLANG_OPT_USE_INKLING_FUSED_LOG_TAU = EnvBool(True)
    # Dispatch the rel_logits projection around einsum's hidden compaction
    # copy of the strided r operand (a view into the packed qkvr output):
    # zero-copy strided-batched matmul at small t, JIT row-compact + einsum
    # above the band, single-launch tau-folded kernel in the small-t tau
    # band. Bit-identical to the plain einsum; flag-off restores it.
    SGLANG_OPT_USE_INKLING_REL_PROJ_DISPATCH = EnvBool(True)
    # Quantize and store MXFP8 K/V data and scales in one fused kernel.
    SGLANG_OPT_INKLING_MXFP8_FUSED_QUANT_STORE = EnvBool(True)
    # Default reasoning effort in [0.0, 0.99] when omitted by a request.
    # An empty string falls back to the protocol default (0.9); the effort
    # directive is always emitted.
    SGLANG_INKLING_DEFAULT_REASONING_EFFORT = EnvStr("0.9")
    SGLANG_INKLING_RS_MM_PREPROCESS = EnvBool(True)

    # ===================================================================
    # DSA backend (GLM 5 and DeepSeek V3.2)
    # ===================================================================
    SGLANG_DSA_FUSE_TOPK = EnvBoolWithAlias(
        True, deprecated_name="SGLANG_NSA_FUSE_TOPK"
    )
    # Full-grid decode-time top-k (jit/csrc/dsa/topk_decode_fg.cuh) replacing
    # sgl_kernel fast_topk_v2 on decode/verify shapes (row_starts is None,
    # batch <= 64, fp32/bf16). Same selection semantics; each row is read
    # exactly twice by the whole grid instead of one block per row.
    SGLANG_DSA_TOPK_DECODE_FG = EnvBool(True)
    # Byte-floor decode-time top-k (jit/csrc/dsa/topk_decode_floor.cuh): one
    # persistent launch with in-kernel grid barriers reading each row once
    # (plus a small sample and a rare fg-equivalent fallback re-read),
    # replacing both fast_topk_v2 and the fg chain on decode/verify shapes
    # (same gate domain as SGLANG_DSA_TOPK_DECODE_FG; precedence: floor > fg).
    # Same selection semantics as the fg kernel.
    SGLANG_DSA_TOPK_DECODE_FLOOR = EnvBool(False)
    # Warm-start the full-grid decode top-k: carry the previous decode step's
    # k-th logit minus a delta-sigma margin per (request, layer) as the
    # threshold seed; 1 streaming pass + exact refine on a hit, full 2-pass
    # fallback on a miss (requires SGLANG_DSA_TOPK_DECODE_FG).
    SGLANG_DSA_TOPK_WARMSTART = EnvBool(False)
    SGLANG_DSA_TOPK_WARMSTART_DELTA = EnvFloat(0.3)

    # Decode-shaped Triton paged-MQA logits kernel
    # (kernels/ops/attention/dsa/decode_mqa_logits.py) replacing the DeepGEMM
    # split call at target-verify: every index-K row is read once per request
    # for all its draft tokens instead of once per query row.
    SGLANG_DSA_DECODE_MQA_LOGITS_TRITON = EnvBool(False)
    SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC = EnvBool(False)
    SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK = EnvStr(None)
    # Prefill-shaped PAGED DSA top-k: use the single-pass JIT kernel
    # (jit/csrc/dsa/topk_prefill_1pass.cuh) instead of the 2-pass
    # sgl_kernel topk_transform_prefill_kernel. Reads the logits once.
    SGLANG_DSA_TOPK_PREFILL_1PASS = EnvBool(True)
    # lane/pagetable-gather: in the chunked-mqa PAGED prefill top-k path,
    # pass the per-step page table to the 1-pass kernel whole plus a per-row
    # table-row map (row_to_page) instead of materializing
    # page_table_1[batch_idx] as a [rows, L] int32 copy per row-chunk per
    # topk layer (~31 GB/layer of HBM writes at L~950k, 5x redundant across
    # the step). Identity-exact; requires SGLANG_DSA_TOPK_PREFILL_1PASS=1.
    SGLANG_DSA_PAGETABLE_HOIST = EnvBool(False)
    # lane/streamindex-topk: key-chunked scorer + partition-merge candidate
    # maintenance for the prefill indexer top-k; the [q, L] logits tensor
    # never exists (exact top-2048, tie-consistent at the boundary).
    SGLANG_DSA_TOPK_STREAMINDEX = EnvBool(False)
    SGLANG_DSA_TOPK_STREAMINDEX_W = EnvInt(8192)
    SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD = EnvIntWithAlias(
        2048, deprecated_name="SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD"
    )
    SGLANG_DSA_HIP_DISABLE_PRESHUFFLE = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_NSA_HIP_DISABLE_PRESHUFFLE"
    )
    SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION = EnvFloat(0.2)
    # Lane mqa-tune: block configuration for the prefill indexer logits
    # kernel sm90_fp8_mqa_logits ('off' | 'best' | 'BQ,BKV,QS,KVS,MT').
    # Same DeepGEMM kernel template, bit-exact; see lanes/mqa-tune.
    SGLANG_DSA_MQA_LOGITS_VARIANT = EnvStr("off")
    SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM = EnvBool(False)
    SGLANG_DSA_TOPK_BROADCAST = EnvBool(False)
    SGLANG_DISABLE_DSA_INDEXER_FUSION = EnvBool(False)
    # GLM-5.3 / DeepSeek-V2 small-M decode: materialize bf16 copies of the
    # qkv_a and indexer (wq_b, wk, weights_proj) projections at load time and
    # route M<=16 forwards through the dsv3_fused_a bf16 GEMV, skipping the
    # per-token-group fp8 activation quant on those paths. Default OFF.
    SGLANG_GLM_DSV3_BF16_SMALLM_GEMV = EnvBool(False)

    # Opt-in: quantize the (otherwise bf16) LM head / draft shared head to
    # blockwise fp8 [128, 128] at weight-load time (amax/448 fp32 scales, the
    # same recipe as the GLM-5.3 FP8 checkpoint's other weights) and run the
    # LM-head GEMM through the production w8a8 block-fp8 path. Halves the
    # ~1.9 GB weight read that dominates the decode/verify LM-head cost
    # (measured 1.87x faster at M=64 on GH200, lanes/lm-head-gemm).
    # NOT bit-exact vs the bf16 GEMM (quantization error characterized in
    # lanes/lm-head-gemm); requires N % 128 == 0 and K % 128 == 0 per rank,
    # otherwise the head silently stays bf16. Takes precedence over
    # --enable-fp32-lm-head when both are set.
    SGLANG_LM_HEAD_FP8 = EnvBool(False)

    # lane/indexer-prologue: keep the Hadamard inside the fused DSA indexer
    # prologue kernels (production arithmetic).
    SGLANG_DSA_INDEXER_FUSION_KEEP_HADAMARD = EnvBool(False)
    # Opt-in perf path for --dsa-prefill-backend flashmla_sparse_q8: fuse the
    # absorbed q bmm with the nope/rope concat + fp8 cast so q is written
    # directly in fp8 ("born fp8") and the standalone concat-cast kernel
    # disappears.  Not bit-exact vs the default path (same rounding stages,
    # different GEMM accumulation order), hence default OFF until accuracy-
    # gated (oracle + full-set gsm8k).
    SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q = EnvBool(False)
    # Opt-in perf path for --dsa-prefill-backend flashmla_sparse_q8: pass a
    # per-row valid-topk count (derived from the trailing -1 pad run of the
    # topk indices) so the kernel skips whole pad-only topk blocks instead of
    # computing masked zero contributions.  Bit-exact by construction: skipped
    # blocks contain only -1 pads, and -1 entries inside the consumed range
    # still take the in-kernel clamp+mask path.
    SGLANG_ENABLE_DSA_Q8KV8_TOPK_LENGTH = EnvBool(False)
    # Opt-in: run the born-fp8 q-prep (absorbed bmm + concat + fp8 cast,
    # ~173us/layer-call) on alt_stream underneath the DSA indexer — the two
    # chains fork independently from the q_a_layernorm output.  Requires
    # SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q; eager-prefill-only via the born
    # predicate.  Coarse per-layer join keeps the single-slot born-q buffer
    # WAR-safe.
    SGLANG_ENABLE_DSA_Q8KV8_QPREP_OVERLAP = EnvBool(False)
    # Opt-in: fuse the Q8KV8 non-prefix KV prep — cast-concat k/k_rope
    # directly into the persistent fp8 kv buffer and zero the pad band in one
    # Triton kernel (replaces bf16 _cat + copy_ cast + zero_ tail).
    SGLANG_ENABLE_DSA_Q8KV8_KV_CAT_FUSION = EnvBool(False)
    # Opt-in: route flashmla_kv DECODE (and target-verify) attention to the
    # lane sparse-decode-kernel native-fp8 SM90 kernel (JIT).  Numerics:
    # prod-class error vs the fp32 oracle (q per-row fp8 + exact per-group KV
    # descale on the QK accumulator + bf16 P/V); NOT bit-exact vs FlashMLA.
    # Perf (b=1, GH200): ~36 us vs prod 17 us fused-graph — currently SLOWER
    # than the production kernel; kept for development/A-B, default OFF.
    SGLANG_DSA_DECODE_FP8_NATIVE = EnvBool(False)
    # Opt-in: route flashmla_kv DECODE (and target-verify) attention to the
    # lane sparse-decode-fused persistent Triton kernel: single launch,
    # in-kernel split+combine (no second launch / combine kernel), grid
    # capped at co-resident capacity so it is deadlock-free at any batch.
    # Numerics: prod-class vs the fp32 oracle (0.17-0.21% of RMS), NOT
    # bit-exact vs FlashMLA (different dequant path and split order).
    SGLANG_DSA_DECODE_FUSED_PERSISTENT = EnvBool(False)
    # Opt-in (lane/sparse-attn): with --dsa-prefill-backend flashmla_auto on
    # SM90 + fp8 KV, route EXTEND prefill batches to the native-fp8 Q8KV8
    # sparse prefill kernel (flashmla_sparse_q8) instead of the fp8-KV
    # flashmla_kv decode kernel.  Measured 1.38-1.74x faster end-to-end at
    # GLM-5.3 prefill shapes on GH200 (q=2048..8192, L=0.5M..1M); numerics
    # differ (per-tensor fp8 requant of KV + fp8 q + fp8 P: ~9x the error of
    # the bf16-MMA path vs an fp32 oracle, ~2.7% of output RMS mean), hence
    # default OFF.  Non-EXTEND batches keep flashmla_kv.
    SGLANG_DSA_PREFILL_Q8KV8_AUTO = EnvBool(False)
    # Q8KV8 born-fp8 q-prep codegen: "auto" = per-K Triton dispatch (default);
    # "cuda" = the hand-written SM90 WGMMA kernel (bitwise identical to the
    # Triton two_dot variant, 1.16-1.38x faster across GLM/DS shapes).
    SGLANG_OPT_Q8KV8_QPREP_VARIANT = EnvStr("auto")

    # ===================================================================
    # MiniMax M3
    # ===================================================================
    SGLANG_OPT_USE_BF16_ROUTER_GEMM = EnvBool(True)
    SGLANG_OPT_USE_MINIMAX_DENSE_SPARSE_DECODE = EnvBool(False)
    SGLANG_DISABLE_MSA = EnvBool(False)
    SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH = EnvBool(False)
    # Kill switch for the derived fp8 attention-GEMM mode (m3_fp8_attn_gemm_enabled):
    # forces the pre-fp8 behavior (bf16 indexer + widening sparse path, bf16 q)
    # even when kv_cache_dtype fp8_e4m3 + trtllm_mha + SM100 would activate it.
    SGLANG_DISABLE_M3_FP8_ATTN_GEMM = EnvBool(False)
    # MiniMax-M3 sparse decode indexer: single JIT radix-select kernel replaces the 2-stage split-K Triton topk.
    SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX = EnvBool(True)
    # Fused JIT store (minimax_store_kv_index) of main+index K/V instead of separate
    # set_*_buffer copies; falls back when main/index dtypes differ or non-CUDA.
    SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE = EnvBool(True)
    # MiniMax-M3 MXFP8 MoE experimental fusion toggles (default off; A/B only).
    SGLANG_MINIMAX_M3_FUSED_SWIGLU_MXFP8 = EnvBool(False)
    SGLANG_MINIMAX_M3_FUSED_MOE_COMBINE = EnvBool(False)
    # MiniMax M3 NPU prefill MAIN-attention: route the sparse main attention through
    # the native Ascend FA op `torch.ops.npu.npu_fused_infer_attention_score` (FIA)
    # with a per-query CUSTOM block_table
    SGLANG_MINIMAX_NPU_PREFILL_FIA = EnvBool(True)
    # MiniMax-M3 NPU sparse INDEXER (decode + verify topk block selection): route
    # through the native AscendC packed indexer op instead of the Triton indexer.
    SGLANG_MINIMAX_NPU_NATIVE_INDEXER = EnvBool(False)
    # MiniMax-M3 NPU sparse MAIN-attention (decode-main + verify-main): route the
    # sparse main attention through the native AscendC sparse-attention op with the
    # cached block_table override.
    SGLANG_MINIMAX_NPU_NATIVE_ATTN = EnvBool(False)
    # MiniMax-M3 on ROCm force-disables custom all-reduce in its model override
    # (arg_groups/overrides.py) when aiter all-reduce fusion is off. Set this to
    # opt back in and keep custom/quick all-reduce enabled -- e.g. to run the
    # INT4 quick-reduce path via ROCM_QUICK_REDUCE_QUANTIZATION={INT4,INT6,INT8}.
    SGLANG_M3_ALLOW_CUSTOM_AR = EnvBool(False)

    # ===================================================================
    # Kimi K3
    # ===================================================================
    # MNNVL fused all-reduce (bf16, TP8): zero-copy 1shot multicast-push for
    # small messages and in-place NVLS 2shot on symmetric-memory tensors for
    # large ones, with an optional fused residual add. Covers the KDA o_proj
    # output and the latent|shared MoE reduce; everything else falls back to
    # the regular all-reduce path. Auto-enabled on SM100/SM103 when
    # CustomAllReduceV2 with multicast is available; set 0/1 to override in
    # either direction. See srt/layers/k3_ar_fusion.py.
    SGLANG_K3_AR_FUSION = EnvBool(False)
    # K3 SP-MoE fused residual + reduce-scatter and matching all-gather over
    # CustomAllReduceV2's MNNVL push workspace. Auto-probed for the validated
    # TP8 GB300 configuration; set 0/1 to override. See
    # srt/layers/k3_sp_collective.py.
    SGLANG_K3_SP_COLLECTIVE = EnvBool(False)
    # Keep K3's post-MoE residual stream token-sharded between consecutive
    # SP-MoE layers. The next attention-residual aggregation and snapshot
    # bank write run on the local shard, then only the normalized attention
    # input is all-gathered. Requires SGLANG_K3_SP_COLLECTIVE.
    SGLANG_K3_SP_ATTN_RES = EnvBool(False)
    # Fused o_proj GEMM + all-reduce (bf16, TP 2..8, SM100+): one
    # kernel computes the TP-local o_proj partial and the cross-rank sum over
    # a P2P comm region, replacing the GEMM + NCCL AR pair at M <= 512.
    SGLANG_K3_GEMM_AR = EnvBool(False)
    # Merge the router gate and routed_expert_down_proj weights so the K3 MoE
    # front reads hidden_states once, and run the top-k plus the bf16 cast in one
    # epilogue kernel. See kernels/ops/moe/moe_front.py. Default on.
    SGLANG_K3_FUSED_FRONT = EnvBool(True)
    SGLANG_KIMI_K3_VIT_CUDA_GRAPH_CACHE_CAPACITY = EnvInt(2)
    SGLANG_KIMI_K3_VIT_CUDA_GRAPH_MIN_HITS = EnvInt(2)
    SGLANG_KIMI_K3_VIT_CUDA_GRAPH_MAX_SEQLEN = EnvInt(6144)

    # ===================================================================
    # Symmetric memory
    # ===================================================================
    SGLANG_SYMM_MEM_PREALLOC_GB_SIZE = EnvInt(-1)
    SGLANG_DEBUG_SYMM_MEM = EnvBool(False)

    # ===================================================================
    # Plugin system
    # ===================================================================
    SGLANG_PLATFORM = EnvStr("")
    SGLANG_PLUGINS = EnvStr("")

    # ===================================================================
    # KV-Canary and Token-Oracle (testing only)
    # ===================================================================
    SGLANG_KV_CANARY_RING_CAPACITY = EnvInt(1024)
    SGLANG_KV_CANARY_STATS_PRINT_EVERY_N_STEPS = EnvInt(100)
    SGLANG_KV_CANARY_ENABLE_WRITE_INPUT_ASSERT = EnvBool(False)
    SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS = EnvInt(50)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_UNUSED_CACHE_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_POST_FORWARD_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_TARGET_GROUP = EnvStr(None)
    SGLANG_KV_CANARY_PERTURB_NEXT_TOKEN_SWAP_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE = EnvBool(False)
    SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT = EnvBool(False)
    SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS_INTERVAL = EnvInt(0)
    SGLANG_KV_CANARY_ENABLE_MHA_V = EnvBool(False)

    # ===================================================================
    # Rust server
    # ===================================================================
    SGLANG_RUST_SERVER = EnvBool(False)
    # Build a missing Rust extension from source (auto), require a bundled or
    # cached extension (never), or rebuild the local cache entry (force).
    SGLANG_RUST_BUILD_MODE = EnvStr("auto")
    # Most batched requests one /generate HTTP call may expand into.
    SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ = EnvInt(4096)


envs = Envs()
EnvField._allow_set_name = False


class _DeprecatedEnv:
    """One deprecated env var: warn if it is set, and optionally forward its
    (possibly transformed) value to a replacement env var."""

    def __init__(
        self,
        replacement: Optional[str] = None,
        transform: Optional[Callable[[str], str]] = None,
        note: Optional[str] = None,
    ):
        self.replacement = replacement
        self.transform = transform
        self.note = note

    def apply(self, old_name: str):
        if old_name not in os.environ:
            return
        message = f"Environment variable {old_name} is deprecated."
        if self.replacement is not None:
            message += f" Please use {self.replacement} instead."
        if self.note is not None:
            message += f" {self.note}"
        warnings.warn(message)
        if self.replacement is not None:
            value = os.environ[old_name]
            if self.transform is not None:
                value = self.transform(value)
            os.environ[self.replacement] = value


def _ms_to_s(value: str) -> str:
    return str(float(value) / 1000.0)


def _invert_bool(value: str) -> str:
    return "0" if value.lower() in ("true", "1", "yes", "y") else "1"


# The single registry for deprecated environment variables, processed once at
# import by _handle_deprecated_envs(). Add new deprecations here instead of
# ad-hoc warnings. For a rename where the old name must keep working through a
# descriptor, use EnvBoolWithAlias / EnvIntWithAlias instead.
_DEPRECATED_ENVS: Dict[str, _DeprecatedEnv] = {
    # Renamed: the value is forwarded to the replacement.
    "SGLANG_GC_LOG": _DeprecatedEnv(replacement="SGLANG_LOG_GC"),
    "SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH": _DeprecatedEnv(
        replacement="SGLANG_MOE_NVFP4_DISPATCH"
    ),
    "SGLANG_ENABLE_THINKING": _DeprecatedEnv(replacement="SGLANG_DEFAULT_THINKING"),
    "SGLANG_REASONING_EFFORT": _DeprecatedEnv(
        replacement="SGLANG_DSV4_REASONING_EFFORT"
    ),
    "SGLANG_USE_JIT_ALL_REDUCE": _DeprecatedEnv(
        replacement="SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"
    ),
    # The legacy DISABLE flags have the opposite polarity of their replacement.
    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": _DeprecatedEnv(
        replacement="SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", transform=_invert_bool
    ),
    # Renamed with a unit change.
    "SGLANG_QUEUED_TIMEOUT_MS": _DeprecatedEnv(
        replacement="SGLANG_REQ_WAITING_TIMEOUT",
        transform=_ms_to_s,
        note="Note the unit change: milliseconds -> seconds.",
    ),
    "SGLANG_FORWARD_TIMEOUT_MS": _DeprecatedEnv(
        replacement="SGLANG_REQ_RUNNING_TIMEOUT",
        transform=_ms_to_s,
        note="Note the unit change: milliseconds -> seconds.",
    ),
    # Removed without replacement.
    "SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2": _DeprecatedEnv(),
    # Superseded by the unified JIT per_token_group_quant, the default CUDA path.
    "SGLANG_OPT_USE_JIT_PER_TOKEN_GROUP_QUANT": _DeprecatedEnv(),
    "SGLANG_MASKED_GEMM_FAST_ACT": _DeprecatedEnv(),
    "SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN": _DeprecatedEnv(),
    # sconv-family kernels always use the CUDA-JIT ports when supported; no toggle.
    "SGLANG_OPT_USE_CUDA_SCONV": _DeprecatedEnv(),
    # DSV4 compressor V2 is always used.
    "SGLANG_OPT_USE_COMPRESSOR_V2": _DeprecatedEnv(),
    # Replaced by CLI flags.
    "SGLANG_ENABLE_GRPC": _DeprecatedEnv(
        note="Please use '--grpc-port' to enable the native gRPC server."
    ),
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": _DeprecatedEnv(
        note="Please use '--enable-prefill-delayer' instead."
    ),
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": _DeprecatedEnv(
        note="Please use '--prefill-delayer-max-delay-passes' instead."
    ),
    "SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK": _DeprecatedEnv(
        note="Please use '--prefill-delayer-token-usage-low-watermark' instead."
    ),
    "SGLANG_CUTLASS_MOE": _DeprecatedEnv(
        note="Please use '--moe-runner-backend=cutlass' and/or "
        "'--speculative-moe-runner-backend=cutlass' instead."
    ),
    "SGLANG_DFLASH_PREFILL_REFILL_TARGET": _DeprecatedEnv(
        note="DFlash now auto-enables the min-free-slots delay; unset this env. "
        "To override the threshold, use '--min-free-slots-delay'."
    ),
}


def _handle_deprecated_envs():
    for old_name, deprecation in _DEPRECATED_ENVS.items():
        deprecation.apply(old_name)

    # Rewrite the legacy SGL_ prefix to SGLANG_ (names not covered above).
    for key, value in list(os.environ.items()):
        if key.startswith("SGL_") and key not in _DEPRECATED_ENVS:
            new_key = key.replace("SGL_", "SGLANG_", 1)
            warnings.warn(
                f"Environment variable {key} is deprecated, please use {new_key}"
            )
            os.environ[new_key] = value


def third_party_cache_defaults() -> Dict[str, str]:
    base = os.path.expanduser(envs.SGLANG_CACHE_DIR.get())
    return {
        "TRITON_CACHE_DIR": os.path.join(base, "triton"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(base, "inductor"),
        "CUDA_CACHE_PATH": os.path.join(base, "nv"),
        # FlashInfer appends ".cache/flashinfer" to this base itself, so this
        # is the base dir rather than the final cache dir.
        "FLASHINFER_WORKSPACE_BASE": base,
    }


def redirect_third_party_caches():
    """Point third-party JIT caches at SGLANG_CACHE_DIR, so a run's compiled
    kernels can be cleaned, warmed or volume-mounted as one directory.

    Must be called early. The redirect silently does nothing if either of
    these has already happened:

    - FlashInfer was imported. It resolves its workspace at import time.
    - Inductor made its first ``cache_dir()`` call. That call setdefaults
      TORCHINDUCTOR_CACHE_DIR itself.
    """
    for key, value in third_party_cache_defaults().items():
        os.environ.setdefault(key, value)


_handle_deprecated_envs()

# Trigger auto-injection of CUDA coredump env vars when SGLANG_CUDA_COREDUMP=1.
# Best-effort; for strict guarantees, set CUDA_* env vars in the shell before
# launching Python. Imported conditionally to keep the default import of this
# module free of non-stdlib side effects.
if envs.SGLANG_CUDA_COREDUMP.get():
    import sglang.srt.debug_utils.cuda_coredump  # noqa: F401, E402  # isort: skip
