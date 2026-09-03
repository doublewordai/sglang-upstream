# to be combined with the sparse coordinator class and sparse algorithm family

import logging
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch

from sglang.kernels.ops.kvcache.hisparse import (
    copy_cache_planned_mla,
    copy_cache_planned_wide_mla,
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
)
from sglang.srt.configs.model_config import dsa_layer_skips_topk, is_deepseek_dsa
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseDSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.utils import get_device_module, is_hip

device_module = get_device_module()

_is_hip = is_hip()

logger = logging.getLogger(__name__)


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseTokenStats(NamedTuple):
    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float


def resolve_shared_index_layers(
    *,
    hf_text_config,
    pp_size: int,
    is_speculative: bool,
) -> Optional[List[bool]]:
    """Per-layer "reuses the previous layer's top-k index" pattern, or None.

    Mirrors DeepseekV2AttentionMLA's skip_topk derivation (index_topk_pattern /
    index_topk_freq / cli_factor); None when the model has no sharing or the
    prefetch cannot run (PP, kill-switch). Speculative decoding is supported:
    the verify path plans one multi-position IO group per skip layer
    (see swap_in_selected_pages).
    """
    if not is_deepseek_dsa(hf_text_config):
        return None
    num_layers = hf_text_config.num_hidden_layers
    cli_factor = getattr(hf_text_config, "cli_factor", 1) or 1
    if cli_factor > 1:
        pattern = [i % cli_factor != 0 for i in range(num_layers)]
    else:
        pattern = [dsa_layer_skips_topk(hf_text_config, i) for i in range(num_layers)]
    if not any(pattern):
        return None
    if pp_size != 1:
        # Under PP a rank's first layers can be skip layers whose anchor lives
        # on the previous rank; _build_prefetch_groups would mis-group them.
        # Prod never runs hisparse with PP (only the pp_size=1 decode arm enables
        # it), so keep the synchronous fallback rather than guessing boundaries.
        logger.warning(
            "HiSparse shared-index prefetch is unsupported under pipeline "
            "parallelism (pp_size=%d); falling back to synchronous swap-in.",
            pp_size,
        )
        return None
    if envs.SGLANG_DISABLE_HISPARSE_PREFETCH.get():
        logger.info(
            "HiSparse shared-index prefetch disabled via "
            "SGLANG_DISABLE_HISPARSE_PREFETCH; using synchronous swap-in."
        )
        return None
    return pattern


def _build_prefetch_groups(
    is_shared_index_layer: List[bool],
) -> Tuple[Dict[int, List[int]], List[int]]:
    """Group consecutive shared-index (skip) layers under their anchor layer.

    Returns (groups, slot): anchor layer_id -> ordered skip layers, and each
    skip layer's position in its group (indexes the per-slot prefetch events).
    """
    groups: Dict[int, List[int]] = {}
    slot = [0] * len(is_shared_index_layer)
    anchor = None
    for i, is_shared in enumerate(is_shared_index_layer):
        if not is_shared:
            anchor = i  # compute layer; anchors the skip layers after it
            continue
        assert anchor is not None, (
            f"shared-index (skip) layer {i} has no preceding compute layer; "
            "the model's index-topk pattern is invalid"
        )
        group = groups.setdefault(anchor, [])
        slot[i] = len(group)
        group.append(i)
    return groups, slot


class HiSparseCoordinator:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: Union[
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ],
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
        host_to_device_ratio: int = 2,
        swap_in_block_size: int = 960,
        shared_index_layers: Optional[List[bool]] = None,
        num_draft_tokens: int = 1,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device
        self.swap_in_block_size = swap_in_block_size
        # MTP/EAGLE verify positions per target forward (1 = plain decode).
        self.num_draft_tokens = max(1, int(num_draft_tokens))
        # Timing probe: skip the host->device KV bytes to measure the "IO is
        # free" floor. Produces garbage output; benchmarking only.
        self.skip_io = envs.SGLANG_DEBUG_HISPARSE_SKIP_IO.get()
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio
        # pd/mtp-hisparse: the MTP draft model's device-resident KV pool
        # (logical-indexed). Set by the scheduler after both exist; see
        # remap_draft_kv_from_host_rows.
        self.draft_pool = None

        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        if self.is_dsv4_hisparse:
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
            page_size = self.mem_pool_device.page_size
            num_host_pages = (
                self.token_to_kv_pool_allocator.size_full // self.compress_ratio
                + page_size
                - 1
            ) // page_size
            self.mem_pool_host = DeepSeekV4PagedHostPool(
                pool_name="dsv4_hisparse_c4",
                device_buffers=self.mem_pool_device.kv_buffer,
                item_bytes=self.mem_pool_device.bytes_per_page_padded,
                num_host_pages=num_host_pages,
                slot_page_size=page_size,
                layout="layer_first",
            )
            self.item_size_bytes = (
                self.mem_pool_device.kv_cache_total_dim
                * self.mem_pool_device.store_dtype.itemsize
            )
        else:
            assert isinstance(
                self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
            )
            self.mem_pool_device: HiSparseDSATokenToKVPool = (
                self.token_to_kv_pool_allocator.get_kvcache()
            )
            self.mem_pool_host = MLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=self.mem_pool_device.page_size,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
            self.item_size_bytes = self.mem_pool_host.token_stride_size
        self.page_size = self.mem_pool_device.page_size

        # Plan-then-IO split (lanes/gather): the fused swap-in kernel plans only
        # (its hit/LRU/evict logic is unchanged -- IO was strictly after
        # planning) and records the miss plan; a full-GPU-grid kernel then
        # copies the planned rows (warp per row). The per-warp copy inside the
        # fused kernel leaves the C2C link starved at small batch (one block
        # per request). DSA/MLA linear layout only; the DSv4 page-padded path
        # keeps the fused kernel.
        self._wide_gather = (
            envs.SGLANG_HISPARSE_WIDE_GATHER.get() and not self.is_dsv4_hisparse
        )
        self._sm_count = torch.cuda.get_device_properties(
            device
        ).multi_processor_count

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        max_compressed_context_len = (
            max_context_len + self.compress_ratio - 1
        ) // self.compress_ratio

        # to have an extra page for new tokens
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_req_slots, self.padded_buffer_size),
            dtype=torch.int64,
            device=device,
        )
        self.req_device_buffer_size = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        self.req_to_host_pool = torch.full(
            (max_num_req_slots, max_compressed_context_len + self.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.req_to_host_pool_allocated_len = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )

        self.green_ctx = None
        self._swapin_stream = None
        green_sms = int(envs.SGLANG_HISPARSE_GREEN_CTX_SMS.get() or 0)
        if green_sms > 0 and not is_hip():
            try:
                import torch as _torch

                dev_idx = (
                    _torch.device(device).index
                    if _torch.device(device).index is not None
                    else _torch.cuda.current_device()
                )
                self.green_ctx = _torch.cuda.GreenContext.create(
                    green_sms, dev_idx
                )
                logger.info(
                    "HiSparse: green context with %d SMs for IO streams "
                    "(backup/prefetch/swap-in)",
                    green_sms,
                )
            except Exception as e:  # pragma: no cover - driver-dependent
                logger.warning(
                    "HiSparse: GreenContext.create(%d) failed (%r); using "
                    "normal streams",
                    green_sms,
                    e,
                )
                self.green_ctx = None
        self.write_staging_stream = self._make_io_stream()
        self.decode_backup_stream = self._make_io_stream()
        # A dedicated green stream for the swap-in gather (issued inside the
        # captured attention flow; fork+join events are capture-safe). Normal
        # path keeps swap-in on the caller's current stream.
        if self.green_ctx is not None:
            self._swapin_stream = self._make_io_stream()
            self._swapin_on_green = bool(
                envs.SGLANG_HISPARSE_SWAPIN_GREEN_CTX.get()
            )
        else:
            self._swapin_on_green = False

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        # Pinned D2H for the spec-v2 verify hooks: pageable .cpu() reads are
        # stream-ordered behind queued kernels (e.g. the draft-extend replay)
        # and pay pageable-staging overhead; these copies run on a private
        # stream gated on the verify-sample event instead.
        self._d2h_stream = device_module.Stream()
        self._d2h_event = device_module.Event()
        self._d2h_pinned = None
        self._verify_sample_done = None
        self.ack_staging_queue: List[HiSparseAct] = []
        # warm-local-prefill: req_pool_idx -> device slot tensor for the
        # extend union swap-in (registered at batch build, freed at staging).
        self._extend_scratch: dict = {}
        self.wlp_trace = envs.SGLANG_WLP_TRACE.get()
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # initialize data structures for swap-in kernel
        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_req_slots, 1)
            .contiguous()
        )
        self._device_buffer_arange_i32 = torch.arange(
            self.device_buffer_size, dtype=torch.int32, device=device
        )

        # Pre-allocated output buffer for swap_in_selected_pages (CUDA-graph safe)
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.raw_indices_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)
        # lane/adaptive-spec: cached sacrificial ghost-row loc (see
        # ragged_ghost_cache_loc); None until first use / after clear().
        self._ragged_ghost_loc: Optional[int] = None

        # Miss-plan buffers for the plan-then-IO swap-in split: the planning
        # kernel (every layer when the wide gather is on; each group's anchor
        # under shared-index prefetch) records (host row, device slot) per
        # miss, and the wide-grid copy replays them. One buffer set suffices:
        # readers are ordered after the writer on the issuing stream(s).
        self._miss_src = torch.zeros(
            (max_num_req_slots, self.top_k), dtype=torch.int64, device=device
        )
        self._miss_dst = torch.zeros(
            (max_num_req_slots, self.top_k), dtype=torch.int32, device=device
        )
        self._miss_count = torch.zeros(
            (max_num_req_slots,), dtype=torch.int32, device=device
        )

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        self._skip_first_backup = [False] * max_num_req_slots

        self._init_shared_index_prefetch(
            shared_index_layers=shared_index_layers,
            layer_num=layer_num,
            max_num_req_slots=max_num_req_slots,
        )

    def _init_shared_index_prefetch(
        self,
        shared_index_layers: Optional[List[bool]],
        layer_num: int,
        max_num_req_slots: int,
    ) -> None:
        """Set up the plan-then-IO prefetch for shared-index (IndexShare) models:
        the anchor's kernel records its miss plan and skip layers replay it on
        `prefetch_stream`, overlapping their IO with the intervening compute."""
        if shared_index_layers is not None and len(shared_index_layers) != layer_num:
            # Attention-layer count differs from num_hidden_layers (e.g. Longcat
            # doubles it): pattern would be misindexed, fall back to synchronous.
            logger.warning(
                "HiSparse shared-index prefetch disabled: pattern length %d != "
                "KV pool layer_num %d; using synchronous swap-in.",
                len(shared_index_layers),
                layer_num,
            )
            shared_index_layers = None
        self._is_shared_index_layer = list(shared_index_layers or [False] * layer_num)
        self.enable_prefetch = any(self._is_shared_index_layer)
        self._prefetch_groups, self._prefetch_slot = _build_prefetch_groups(
            self._is_shared_index_layer
        )
        # Diagnostic cadence (verify steps): log per-position miss counts every
        # N steps from commit_verify_tokens (outside the CUDA graph).
        self._miss_log_every = int(envs.SGLANG_HISPARSE_MISS_LOG.get())
        self._miss_log_seen = 0
        self._miss_src_v = None
        self._miss_dst_v = None
        self._miss_count_v = None
        self._verify_slot_table = None
        self._prefetch_events_v = None
        if not self.enable_prefetch:
            return

        # Small fixed grid for the copy-only kernel: low SM footprint so the
        # copies overlap compute with little contention.
        self._prefetch_copy_blocks = 4
        max_group_size = max(len(g) for g in self._prefetch_groups.values())
        self.prefetch_stream = self._make_io_stream()
        self._prefetch_events = [device_module.Event() for _ in range(max_group_size)]
        logger.info(
            "HiSparse: shared-index prefetch (plan-then-IO) enabled; %d anchor "
            "group(s), %d skip layer(s) of %d total%s.",
            len(self._prefetch_groups),
            sum(self._is_shared_index_layer),
            layer_num,
            (
                f"; MTP verify: {self.num_draft_tokens}-position plan replay"
                if self.num_draft_tokens > 1
                else ""
            ),
        )

    def _make_io_stream(self):
        """A stream for hisparse IO, bound to the green context when the
        SGLANG_HISPARSE_GREEN_CTX_SMS flag is set (else a normal stream)."""
        if self.green_ctx is not None:
            return self.green_ctx.Stream()
        return device_module.Stream()

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def destroy(self) -> None:
        # Drain in-flight transfers so the buffer is idle, then unregister it.
        # See HostKVCache.destroy for why the explicit unregister matters.
        self.write_staging_stream.synchronize()
        self.decode_backup_stream.synchronize()
        if self._swapin_stream is not None:
            self._swapin_stream.synchronize()
        if self.enable_prefetch:
            # Skip-layer copies read the pinned host pool on the prefetch stream.
            self.prefetch_stream.synchronize()
        self.mem_pool_host.destroy()

    def get_token_stats(self) -> HiSparseTokenStats:
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        host_capacity = self.mem_pool_host.size
        host_tokens = host_capacity - self.mem_pool_host.available_size()
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

    def admit_request_into_staging(self, req: Req, adopted_len: int = 0) -> None:
        """Back up freshly computed KV to the host pool, then stage the request.

        With ``adopted_len > 0`` (warm-local-prefill path), the first
        ``adopted_len`` tokens are a retained prefix whose host rows already
        exist (adopted via :meth:`adopt_prefix`): only the delta
        ``[adopted_len, extend_range.end)`` gets new host pages and a device->
        host backup.
        """
        req.hisparse_staging = True

        prefill_start = adopted_len
        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, prefill_start : req.extend_range.end
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        prefill_len = adopted_len + len(device_indices)
        host_indices = self.mem_pool_host.alloc_paged_token_slots(
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            self.host_token_len(prefill_start),
            self.host_token_len(prefill_len) - self.host_token_len(prefill_start),
        )

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        # The extend scratch (union swap-in slots) is no longer needed: the
        # forward pass that consumed it has completed (this runs from batch
        # result processing, after the forward's copy_done sync).
        self.free_extend_scratch(req.req_pool_idx)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def admit_request_direct(self, req: Req) -> None:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging DMA entirely. Only allocates a small device buffer
        (4KB) for decode-time swap-in, then marks the request as ready.
        Host indices were already written to req_to_host_pool.

        Metadata fixups after alloc_device_buffer():
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.
        """
        self.alloc_device_buffer(req)
        self.remap_draft_kv_from_host_rows(req)

        host_len = self.host_token_len(req.kv.kv_allocated_len)
        if host_len <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            self._preload_to_device_buffer(req)
        else:
            # Long sequence: reset device_buffer_tokens to -1 so the kernel
            # sees all slots as empty -> every top-k lookup is a miss -> host load.
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1

        req.hisparse_staging = False
        self._skip_first_backup[req.req_pool_idx] = True
        logger.debug("HiSparse: admitting request %s directly", req.rid)

    def remap_draft_kv_from_host_rows(self, req: Req) -> None:
        """pd/mtp-hisparse: move the transferred draft-layer KV to its logical locs.

        Under hisparse the PD transfer writes EVERY layer (target latent and
        draft) at the decode arm's *host-row* indices -- the main KV's
        destination is the pinned host pool, and the draft pool's VRAM buffer
        is appended to the same index list. The draft model, however, reads
        its device-resident pool at *logical* locs (req_to_token). Move the
        just-transferred delta rows host_rows -> logical, via a staging
        buffer (the two index spaces alias the same pool rows).
        """
        if self.draft_pool is None:
            return
        idx = req.req_pool_idx
        adopted = getattr(self, "req_adopted_len", None)
        start = int(adopted[idx]) if adopted is not None else 0
        # The transfer wrote exactly [start, host_token_len(kv_allocated_len))
        # rows (token-granular, matching the prealloc host alloc); the page
        # tail beyond it was never written.
        end = min(
            int(self.req_to_host_pool_allocated_len[idx]),
            self.host_token_len(req.kv.kv_allocated_len),
        )
        if end <= start:
            return
        host_rows = self.req_to_host_pool[idx, start:end].to(torch.int64)
        logical = self.req_to_token_pool.req_to_token[idx, start:end].to(torch.int64)
        for layer_id in range(self.draft_pool.layer_num):
            buf = self.draft_pool.get_key_buffer(layer_id)
            # fp8/byte-typed rows have no index_copy_cuda; move them as raw
            # bytes (the row holds nope_fp8|scales|rope_bf16 contiguously).
            if buf.dtype != torch.uint8 and buf.element_size() == 1:
                buf = buf.view(torch.uint8)
            staged = buf.index_select(0, host_rows)
            buf.index_copy_(0, logical, staged)
        if envs.SGLANG_MTP_DEBUG.get():
            logger.info(
                "HiSparse: remapped draft KV rows [%d, %d) host %s -> logical %s "
                "for %s",
                start,
                end,
                host_rows[:4].tolist(),
                logical[:4].tolist(),
                req.rid,
            )

    def host_token_len(self, kv_allocated_len: int) -> int:
        if self.is_dsv4_hisparse:
            return kv_allocated_len // self.compress_ratio
        return kv_allocated_len

    def _preload_to_device_buffer(self, req: Req) -> None:
        """Preload all tokens from host pool into the device buffer."""
        n = self.host_token_len(req.kv.kv_allocated_len)
        host_indices = self.req_to_host_pool[req.req_pool_idx, :n]
        device_locs = self.req_to_device_buffer[req.req_pool_idx, :n]

        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def alloc_device_buffer(self, req: Req) -> None:
        if self.is_dsv4_hisparse:
            allocated_len = req.extend_range.end
            alloc_size = self.padded_buffer_size
        else:
            allocated_len = req.kv.kv_allocated_len
            page_size = self.mem_pool_device.page_size
            # Allocate only enough for current tokens (page-aligned).
            # When prefill already fills device_buffer_size, include the reserved page.
            alloc_size = min(
                ((allocated_len + page_size - 1) // page_size) * page_size,
                self.device_buffer_size,
            )
            if alloc_size == self.device_buffer_size:
                alloc_size = self.padded_buffer_size

        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.req_pool_idx, :allocated_len]
            )
        )
        compressed_len = len(compressed_logical_indices)

        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            compressed_logical_indices, alloc_size
        )
        if buffer_indices is None:
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(compressed_len=%d, alloc_size=%d)",
                req.rid,
                compressed_len,
                alloc_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

        buffer_indices = buffer_indices.to(torch.int32)
        self.req_to_device_buffer[req.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.req_pool_idx] = alloc_size

        self.req_device_buffer_tokens[
            :, req.req_pool_idx, : self.device_buffer_size
        ] = self._device_buffer_arange_i32
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )

    def _grow_device_buffers(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Grow device buffers for requests whose sequence length exceeds current capacity."""
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]
        # pd/mtp-hisparse: grow up to the padded buffer. An MTP verify window
        # can cross device_buffer_size in one step (positions at/after dbs
        # take reserved-page slots within [dbs, dbs+page)), so a request stays
        # "growable" until it holds the full padded buffer; plain decode is
        # unchanged (it crosses at seq_len == dbs, where new_cap already
        # becomes padded_buffer_size).
        short_reqs_cpu = seq_lens_cpu < self.padded_buffer_size
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)

        if torch.any(needs_grow_cpu):
            page_size = self.mem_pool_device.page_size
            grow_indices = torch.where(needs_grow_cpu)[0]

            # Compute all grow sizes on CPU, then do a single bulk allocation
            req_idxs = []
            old_caps = []
            new_caps = []
            grow_sizes = []
            total_grow = 0
            for i in grow_indices.tolist():
                req_idx = int(req_pool_indices_cpu[i])
                current_cap = int(current_caps[i])
                seq_len = int(seq_lens_cpu[i])

                new_cap = min(
                    ((seq_len + page_size - 1) // page_size) * page_size,
                    self.device_buffer_size,
                )
                if new_cap == self.device_buffer_size:
                    new_cap = self.padded_buffer_size
                grow_size = new_cap - current_cap
                if grow_size <= 0:
                    continue
                req_idxs.append(req_idx)
                old_caps.append(current_cap)
                new_caps.append(new_cap)
                grow_sizes.append(grow_size)
                total_grow += grow_size

            if total_grow > 0:
                all_new_indices = (
                    self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                        total_grow
                    )
                )
                if all_new_indices is None:
                    logger.error(
                        "HiSparse: _grow_device_buffers bulk alloc failed "
                        "(total_grow=%d)",
                        total_grow,
                    )
                    raise RuntimeError(
                        f"HiSparse _grow_device_buffers failed (total_grow={total_grow})"
                    )

                offset = 0
                for req_idx, current_cap, new_cap, grow_size in zip(
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]
                    offset += grow_size
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk
                    self.req_device_buffer_size[req_idx] = new_cap

        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]

    def has_ongoing_staging(self) -> bool:
        return len(self.ack_staging_queue) > 0

    def collect_ready_reqs(self) -> List[Req]:
        ready_reqs: List[Req] = []
        if len(self.ack_staging_queue) == 0:
            return ready_reqs

        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            # prepare device buffer and update req
            self.alloc_device_buffer(req)
            # warm-local-prefill: alloc_device_buffer assumes the staging
            # path's contiguous-from-zero layout; correct the content map for
            # an adopted prefix (slot j holds token prefix_len + j).
            adopted = int(getattr(req, "wlp_adopted_len", 0) or 0)
            if adopted > 0:
                self._fixup_wlp_buffer_tokens(req, adopted)
            self._skip_first_backup[req.req_pool_idx] = True
            req.hisparse_staging = False
            finish_count -= 1
            ready_reqs.append(req)
        return ready_reqs

    def _fixup_wlp_buffer_tokens(self, req: Req, prefix_len: int) -> None:
        """Correct the device-buffer content map after a warm-local extend.

        ``alloc_device_buffer`` seeds ``req_device_buffer_tokens`` with
        ``arange`` ("slot j holds token j"), which is only true for the
        staging path where the whole prompt was just prefilled from zero.
        After a warm-local extend the buffer holds the delta's device slots:
        slot j holds token ``prefix_len + j`` for ``j < min(delta, alloc)``;
        any remaining slots are empty.
        """
        idx = req.req_pool_idx
        allocated_len = int(req.kv.kv_allocated_len)
        delta = allocated_len - prefix_len
        alloc_size = int(self.req_device_buffer_size[idx])
        n = max(0, min(int(delta), alloc_size))
        if n > 0:
            toks = self._device_buffer_arange_i32[:n] + prefix_len
            self.req_device_buffer_tokens[:, idx, :n] = toks
        if n < self.device_buffer_size:
            self.req_device_buffer_tokens[
                :, idx, n : self.device_buffer_size
            ] = -1

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        if not self.is_dsv4_hisparse:
            # Grow device buffers if needed and resolve the latest-token slot.
            reserved_buffer_loc = self._grow_device_buffers(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)

            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
                out_cache_loc
            )
            # ROCm: the decode remap creates a temporary hisparse device slot per
            # new token (via the page_size==1 allocator path). Free the stale
            # slot before pointing the mapping at the reserved device-buffer slot,
            # otherwise the temporary slots leak and corrupt later swap-in lookups.
            # CUDA keeps the original behavior: the swap-in kernel consumes only
            # top_k_device_locs, so stale mapping entries are harmless there.
            if _is_hip:
                previous_locs = self.mem_pool_device._translate_loc_to_hisparse_device(
                    compressed_locs
                )
                stale_locs = previous_locs[
                    (previous_locs > 0) & (previous_locs != reserved_buffer_loc)
                ]
                if stale_locs.numel() > 0:
                    self.token_to_kv_pool_allocator.free_hisparse_indices(stale_locs)

            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = reserved_buffer_loc
            return

        active_reqs = seq_lens % self.compress_ratio == 0
        if not torch.any(active_reqs):
            return

        active_seq_lens = seq_lens[active_reqs]
        active_out_cache_loc = out_cache_loc[active_reqs]
        active_req_pool_indices = req_pool_indices[active_reqs]

        compressed_seq_lens = active_seq_lens // self.compress_ratio
        reserved_positions = (compressed_seq_lens - 1).clamp(
            max=self.device_buffer_size
        )
        reserved_buffer_loc = self.req_to_device_buffer[
            active_req_pool_indices, reserved_positions
        ]

        self.req_device_buffer_token_locs[
            :, active_req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)

        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            active_out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            reserved_buffer_loc
        )

    # ------------------------------------------------------------------
    # MTP target-verify (pd/mtp-hisparse)
    #
    # A verify step writes num_draft_tokens new tokens per request at positions
    # [seq_len, seq_len + n). Decode's single reserved slot generalises to a
    # window: positions below device_buffer_size take their 1:1 buffer slot,
    # positions at or past it take the reserved page slot
    # device_buffer_size + (pos - max(seq_len, device_buffer_size)). The swap-in
    # kernel binds the same window (num_newest = p + 1 for draft position p).
    # After acceptance the committed tokens are backed up to host rows, exactly
    # like decode's eager backup of the previous token, before the next step
    # reuses the reserved slots.
    # ------------------------------------------------------------------
    def _pinned_read(self, tensors) -> list:
        """Async-copy small int64 GPU tensors into one pinned buffer on the
        private D2H stream (gated on the verify-sample event when set), then
        event-synchronize and return the pinned views. Values read here were
        produced no later than the last event this host thread synchronized
        on, so the wait is transfer latency, not GPU work."""
        n = sum(t.numel() for t in tensors)
        if self._d2h_pinned is None or self._d2h_pinned.numel() < n:
            self._d2h_pinned = torch.empty(
                max(n, 256), dtype=torch.int64, pin_memory=True
            )
        ev = self._verify_sample_done
        if ev is not None:
            self._d2h_stream.wait_event(ev)
        else:
            self._d2h_stream.wait_stream(device_module.current_stream())
        with device_module.stream(self._d2h_stream):
            off = 0
            for t in tensors:
                self._d2h_pinned[off : off + t.numel()].copy_(
                    t.to(torch.int64), non_blocking=True
                )
                off += t.numel()
        self._d2h_event.record(self._d2h_stream)
        self._d2h_event.synchronize()
        off = 0
        views = []
        for t in tensors:
            views.append(self._d2h_pinned[off : off + t.numel()])
            off += t.numel()
        return views

    def record_verify_sample_done(self) -> None:
        """Fence the verify sample output (called right after eagle_sample,
        before the caller launches draft-extend)."""
        ev = device_module.Event()
        ev.record()
        self._verify_sample_done = ev

    def _verify_slot_locs(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Physical device-buffer slots of positions [seq_len, seq_len + n) per request: [bs, n]."""
        dbs = self.device_buffer_size
        positions = seq_lens.view(-1, 1) + torch.arange(
            num_tokens, device=seq_lens.device, dtype=seq_lens.dtype
        ).view(1, -1)
        reserved_base = torch.clamp(seq_lens, min=dbs).view(-1, 1)
        slot_idx = torch.where(positions < dbs, positions, dbs + (positions - reserved_base))
        return self.req_to_device_buffer[req_pool_indices.view(-1, 1), slot_idx]

    def map_verify_locs_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        num_draft_tokens: int,
    ) -> None:
        """Before an MTP target-verify forward: steer the n new tokens' KV writes
        into the device buffer (1:1 region or reserved page) and grow buffers."""
        assert not self.is_dsv4_hisparse, "MTP verify under hisparse: DSA only"
        n = int(num_draft_tokens)
        page_size = self.mem_pool_device.page_size
        assert n <= page_size, (
            f"MTP verify needs num_draft_tokens ({n}) <= hisparse page size ({page_size})"
        )
        assert self.device_buffer_size >= n * self.top_k, (
            f"MTP verify needs device_buffer_size ({self.device_buffer_size}) >= "
            f"num_draft_tokens * top_k ({n} * {self.top_k})"
        )
        bs = seq_lens.shape[0]
        if seq_lens_cpu is None or req_pool_indices_cpu is None:
            # Pinned async D2H on the private stream: the values are relay-
            # published and ready, and this must not wait behind queued
            # kernels on the forward stream.
            pin_seq, pin_req = self._pinned_read([seq_lens, req_pool_indices])
            if seq_lens_cpu is None:
                seq_lens_cpu = pin_seq
            if req_pool_indices_cpu is None:
                req_pool_indices_cpu = pin_req
        self.wait_for_pending_backup()
        # Grow 1:1 buffers to cover the last new position; allocates the reserved
        # page for requests crossing device_buffer_size.
        self._grow_device_buffers(
            seq_lens + n, req_pool_indices, seq_lens_cpu + n, req_pool_indices_cpu
        )
        locs = self._verify_slot_locs(seq_lens, req_pool_indices, n)  # [bs, n]
        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            locs.reshape(-1)
        )

    def map_verify_locs_to_buffer_ragged(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        ragged_layout,
        num_draft_tokens: int,
    ) -> None:
        """Ragged variant of map_verify_locs_to_buffer (lane/adaptive-spec):
        each request's window is verify_lens[i] tokens (front-packed rows).

        ``out_cache_loc`` is the compact tier-padded tensor; only the first
        total_verify_tokens rows are real. Ghost rows' locs are pre-steered to
        the sacrificial row by ragged_ghost_cache_loc and are not mapped here.
        """
        assert not self.is_dsv4_hisparse, "MTP verify under hisparse: DSA only"
        n = int(num_draft_tokens)
        page_size = self.mem_pool_device.page_size
        assert n <= page_size, (
            f"MTP verify needs num_draft_tokens ({n}) <= hisparse page size ({page_size})"
        )
        assert self.device_buffer_size >= n * self.top_k, (
            f"MTP verify needs device_buffer_size ({self.device_buffer_size}) >= "
            f"num_draft_tokens * top_k ({n} * {self.top_k})"
        )
        verify_lens_cpu_list = (
            ragged_layout.verify_lens_cpu
            if ragged_layout.verify_lens_cpu is not None
            else ragged_layout.verify_lens.detach().to("cpu").tolist()
        )
        total = int(sum(verify_lens_cpu_list))
        if total <= 0:
            return
        verify_lens = ragged_layout.verify_lens.to(
            device=seq_lens.device, dtype=torch.int64
        )
        verify_lens_cpu = torch.tensor(
            verify_lens_cpu_list, dtype=torch.int64
        )
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.cpu()
        if req_pool_indices_cpu is None:
            req_pool_indices_cpu = req_pool_indices.cpu()
        self.wait_for_pending_backup()
        # Grow 1:1 buffers to cover each request's last new position.
        self._grow_device_buffers(
            seq_lens + verify_lens,
            req_pool_indices,
            seq_lens_cpu + verify_lens_cpu,
            req_pool_indices_cpu,
        )
        # Per-(request, within) window slots for the real rows only.
        starts = torch.cumsum(verify_lens, dim=0) - verify_lens
        rows = torch.arange(total, device=seq_lens.device, dtype=torch.int64)
        req_id = torch.searchsorted(
            torch.cumsum(verify_lens, dim=0), rows, right=True
        )
        safe_req = req_id.clamp(max=verify_lens.shape[0] - 1)
        within = rows - starts[safe_req]
        positions = seq_lens[safe_req] + within
        reserved_base = torch.clamp(seq_lens[safe_req], min=self.device_buffer_size)
        slot_idx = torch.where(
            positions < self.device_buffer_size,
            positions,
            self.device_buffer_size + (positions - reserved_base),
        )
        locs = self.req_to_device_buffer[req_pool_indices[safe_req], slot_idx]
        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            out_cache_loc[:total]
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            locs.to(torch.int64)
        )

    def ragged_ghost_cache_loc(self, num_draft_tokens: int) -> int:
        """Full-pool loc for ghost (tier-padding) rows, steered once to the
        first spare reserved-page row (device_buffer_size + n), which no
        position ever maps to (the verify window is at most n tokens and the
        reserved page holds page_size >= n slots). Uses the mapping's slack
        tail (size_full + page_size - 1): never allocated, only reset by
        clear(), after which the lazy flag re-steers.

        Returns 0 (plain loc-0 ghost convention, as DSpark) when the reserved
        page has no spare row.
        """
        n = int(num_draft_tokens)
        if self._ragged_ghost_loc is not None:
            return self._ragged_ghost_loc
        mapping = self.mem_pool_device.full_to_hisparse_device_index_mapping
        page_size = self.mem_pool_device.page_size
        if page_size <= n:
            logger.warning(
                "ragged ghost steering unavailable (page_size %d <= n %d); "
                "ghost rows write through mapping[0] (loc-0 convention).",
                page_size,
                n,
            )
            self._ragged_ghost_loc = 0
            return 0
        ghost_loc = int(mapping.shape[0]) - 2
        mapping[ghost_loc] = self.device_buffer_size + n
        self._ragged_ghost_loc = ghost_loc
        return ghost_loc

    def commit_verify_tokens(
        self,
        seq_lens: torch.Tensor,
        accept_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """After acceptance: back up the accepted tokens [seq_len, seq_len + accept)
        from their buffer slots to host rows (the swap-in kernel serves them from
        host once the reserved slots are reused)."""
        bs = seq_lens.shape[0]
        if seq_lens_cpu is None or req_pool_indices_cpu is None:
            # Pinned async D2H on the private stream, gated on the
            # verify-sample event: the read must not be stream-ordered behind
            # the draft-extend replay queued on the forward stream.
            pin_seq, pin_req, pin_acc = self._pinned_read(
                [seq_lens, req_pool_indices, accept_lens]
            )
            if seq_lens_cpu is None:
                seq_lens_cpu = pin_seq
            if req_pool_indices_cpu is None:
                req_pool_indices_cpu = pin_req
            accept_cpu = pin_acc.tolist()
        else:
            accept_cpu = accept_lens.to("cpu", non_blocking=False).tolist()
        accept_cpu = [int(a) for a in accept_cpu]
        # Compute every slot in [seq_len, seq_len + max_accept) with ONE
        # _verify_slot_locs launch (the old loop ran ~6 small ops per request),
        # then flatten the per-req accepted prefixes with a mask; row-major
        # flatten matches the host-row allocation order below.
        max_accept = max(1, max(accept_cpu, default=1))
        slot_locs = self._verify_slot_locs(seq_lens, req_pool_indices, max_accept)
        col = torch.arange(max_accept, device=seq_lens.device).view(1, -1)
        accept_gpu = accept_lens.to(torch.int64).view(-1, 1)
        device_locs = slot_locs[col < accept_gpu].to(torch.int64)
        host_locs_list = []
        for i, a in enumerate(accept_cpu):
            if a <= 0:
                continue
            req_idx = int(req_pool_indices_cpu[i])
            start_pos = int(seq_lens_cpu[i])
            host_locs = self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                req_idx,
                start_pos,
                a,
            )
            host_locs_list.append(host_locs)
        if not host_locs_list:
            return
        host_locs = torch.cat(host_locs_list)
        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    def _fast_backup_eligible(
        self,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> bool:
        """Steady-state conditions for the GPU-side backup fast path.

        True when no first-step skip is pending (each request's first decode step
        after staging must take the generic path once) and every request's host
        pool already covers this step's position (no page growth needed). Both
        checks read CPU state only -- no device sync.
        """
        page_size = self.mem_pool_host.page_size
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                return False
            page_end = (
                (int(seq_lens_cpu[i]) - 1 + page_size - 1) // page_size * page_size
            )
            if page_end > int(self.req_to_host_pool_allocated_len[req_idx]):
                return False
        return True

    def _fast_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> None:
        """GPU-side steady-state backup of the previous token (compress_ratio == 1).

        backup_indices == the whole batch, so backup_req_indices/prev positions
        are plain arithmetic and both loc vectors are single gathers. The launch
        block mirrors the generic path exactly (same stream waits and events).
        """
        prev_pos = seq_lens - 1 - 1  # (seq_len - 1) // 1 - 1
        buffer_slot = prev_pos.clamp(max=self.device_buffer_size)
        host_locs = self.req_to_host_pool[req_pool_indices, prev_pos]
        device_locs = self.req_to_device_buffer[req_pool_indices, buffer_slot]

        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            host_locs.record_stream(self.decode_backup_stream)
            device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Back up the previous compressed token to host memory.

        Each newly produced compressed token (one per `compress_ratio` decode
        steps) must be backed up to host so the swap-in kernel can later
        recover it.

        Two cases are skipped:
        - The first decode step right after staging: all prefill tokens were
          already backed up during staging, so there is nothing new to save.
        - Steps where `(seq_len - 1) % compress_ratio != 0`: no new compressed
          token was produced this step.
        """
        # [lane attn-streams] Steady-state fast path (SGLANG_HISPARSE_FAST_BACKUP,
        # compress_ratio == 1): every request backs up its previous token, so the
        # index list is the whole batch and both loc vectors are two gathers --
        # no python-list -> pageable-H2D round-trip (measured ~2.2 ms host/step on
        # GH200) and no per-request python loop. Falls back to the generic path on
        # first-step skips or host-page growth.
        if (
            envs.SGLANG_HISPARSE_FAST_BACKUP.get()
            and self.compress_ratio == 1
            and self._fast_backup_eligible(seq_lens_cpu, req_pool_indices_cpu)
        ):
            self._fast_backup_previous_token(seq_lens, req_pool_indices)
            return

        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up),
        # and skip non-aligned steps that did not produce a new compressed token.
        backup_indices = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                self._skip_first_backup[req_idx] = False
                continue
            if (int(seq_lens_cpu[i]) - 1) % self.compress_ratio == 0:
                backup_indices.append(i)

        if not backup_indices:
            return

        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device=self.device
        )
        backup_req_indices = req_pool_indices[backup_indices_gpu]

        # The previous compressed token's position and its device buffer slot:
        #  compressed_pos = (seq_len - 1) // compress_ratio - 1
        #  - short: slot = compressed_pos          (within the regular buffer)
        #  - long:  slot = device_buffer_size      (the reserved slot)
        prev_seq_lens = seq_lens[backup_indices_gpu] - 1
        compressed_prev_seq_lens = prev_seq_lens // self.compress_ratio
        actual_compressed_pos = compressed_prev_seq_lens - 1

        buffer_slot = actual_compressed_pos.clamp(max=self.device_buffer_size)

        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]

        host_locs_list = []
        for i in backup_indices:
            req_idx = int(req_pool_indices_cpu[i])
            start_pos = (int(seq_lens_cpu[i]) - 1) // self.compress_ratio - 1
            host_locs = self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                req_idx,
                start_pos,
                1,
            )
            host_locs_list.append(host_locs)
        host_locs = torch.cat(host_locs_list)

        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
            if actual_compressed_pos.is_cuda:
                actual_compressed_pos.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    def wait_for_pending_backup(self) -> None:
        if not self._has_pending_backup:
            return
        self._backup_done_event.wait(device_module.current_stream())
        self._has_pending_backup = False

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA kernel) instead.

        Note: dsv4 hisparse is not supported — DeepSeekV4SingleKVPoolHost has no
        load_to_device_per_layer and indices live in compressed space. Currently
        only used as a kernel oracle in test_hisparse_unit.py (non-dsv4 path).

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
        """
        assert (
            not self.is_dsv4_hisparse
        ), "naive_load_topk is not implemented for dsv4 hisparse"
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)

            assert torch.all(
                selected_tokens >= 0
            ), f"Req {req_idx}: selected tokens contain negative positions"
            assert torch.all(selected_tokens < seq_len), (
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                device_indices = torch.empty(
                    top_n, dtype=torch.int64, device=self.device
                )

                is_latest_token = selected_tokens == (seq_len - 1)
                needs_host_load = ~is_latest_token

                device_indices[is_latest_token] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]

                num_to_load = int(needs_host_load.sum().item())
                if num_to_load > 0:
                    tokens_to_load = selected_tokens[needs_host_load]
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]

                    invalid_mask = host_locs < 0
                    if torch.any(invalid_mask):
                        bad_positions = tokens_to_load[invalid_mask].tolist()
                        raise AssertionError(
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]
                    device_indices[needs_host_load] = buffer_locs

                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)

        return top_k_indices

    # ------------------------------------------------------------------
    # warm-local-prefill: extend-time union swap-in (host pages -> scratch)
    # ------------------------------------------------------------------

    def register_extend_scratch(self, req_pool_idx: int, max_slots: int) -> torch.Tensor:
        """Reserve hisparse device slots for one warm-local extend's union
        swap-in. Freed by :meth:`free_extend_scratch` at staging admission
        (or abort). Returns the slot tensor."""
        assert req_pool_idx not in self._extend_scratch, (
            f"extend scratch already registered for req {req_pool_idx}"
        )
        page = self.page_size
        need = (int(max_slots) + page - 1) // page * page
        locs = self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(need)
        if locs is None:
            raise RuntimeError(
                f"WLP: extend scratch alloc failed (need {need} device slots)"
            )
        self._extend_scratch[req_pool_idx] = locs
        return locs

    def free_extend_scratch(self, req_pool_idx: int) -> None:
        locs = self._extend_scratch.pop(req_pool_idx, None)
        if locs is not None and locs.numel() > 0:
            self.token_to_kv_pool_allocator.free_hisparse_indices(locs)

    def _grow_extend_scratch(self, req_pool_idx: int, min_slots: int) -> torch.Tensor:
        """Grow the registered scratch to at least ``min_slots`` slots."""
        locs = self._extend_scratch[req_pool_idx]
        have = int(locs.numel())
        if have >= min_slots:
            return locs
        page = self.page_size
        need = (min_slots - have + page - 1) // page * page
        extra = self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(need)
        if extra is None:
            raise RuntimeError(
                f"WLP: extend scratch grow failed (have {have}, need {min_slots})"
            )
        locs = torch.cat([locs, extra])
        self._extend_scratch[req_pool_idx] = locs
        return locs

    def extend_swap_in_page_table(
        self,
        req_pool_indices: torch.Tensor,
        topk_positions: torch.Tensor,
        prefix_lens: torch.Tensor,
        translated: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Build the extend attention page table when the matched prefix is
        host-resident (retained).

        ``topk_positions``: (num_queries, top_k) per-query selected positions
        (-1 padded). ``translated``: the PAGED-transform page table (logical
        slot per selection). For selections in the retained prefix
        (position < prefix_len) the KV lives in host pages: the union of those
        positions across the chunk's queries is loaded into the extend
        scratch for this layer and the table points at the scratch slots;
        delta selections translate through the device mapping as usual.
        Selection order per query is preserved, so the attention kernel sees
        the same index sequence the prefill arm's device-resident path sees.
        """
        num_reqs = int(req_pool_indices.numel())
        if num_reqs != 1:
            raise NotImplementedError(
                "warm-local-prefill extend supports single-request batches "
                f"(got {num_reqs})"
            )
        r = int(req_pool_indices[0].item())
        prefix_len = int(prefix_lens[0].item())

        pos = topk_positions.to(torch.int64)
        valid = pos >= 0
        pos_c = pos.clamp(min=0)
        is_prefix = pos_c < prefix_len

        flat = pos_c[valid & is_prefix].reshape(-1)
        if flat.numel() > 0:
            union = torch.unique(flat)  # sorted ascending
            u = int(union.numel())
            if self.wlp_trace and layer_id == 0:
                mb = u * self.item_size_bytes / 1e6
                logger.info(
                    "WLP swapin rid_layer=0 prefix=%d union=%d c2c_mb=%.2f",
                    prefix_len,
                    u,
                    mb,
                )
            locs = self._extend_scratch.get(r)
            if locs is None or int(locs.numel()) < u:
                locs = self._grow_extend_scratch(r, max(u, 1))
            scratch = locs[:u]
            host_rows = self.req_to_host_pool[r, union]
            if not bool((host_rows >= 0).all()):
                missing = union[(host_rows < 0).nonzero(as_tuple=True)[0][:8]]
                raise AssertionError(
                    f"WLP: req {r} layer {layer_id}: no host rows for prefix "
                    f"positions {missing.tolist()} (prefix_len={prefix_len})"
                )
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_rows,
                scratch,
                layer_id,
                io_backend="kernel",
            )
            idx = torch.searchsorted(union, pos_c.reshape(-1)).reshape(pos.shape)
            # searchsorted returns len(union) for positions past the union
            # (delta selections); clamp before the gather -- those entries are
            # discarded by the where-mask below but the gather still evaluates.
            prefix_locs = scratch[idx.clamp(max=u - 1)]
        else:
            prefix_locs = None

        trans_c = translated.to(torch.int64).clamp(min=0)
        device_locs = self.mem_pool_device.translate_loc_to_hisparse_device(
            trans_c
        )
        if prefix_locs is not None:
            table = torch.where(valid & is_prefix, prefix_locs, device_locs)
        else:
            table = device_locs
        table = torch.where(valid, table, -1)
        return table.to(torch.int32)

    def abort_staging_request(self, req: Req) -> None:
        """Remove a request from the staging queue and free its host + device resources.

        Must be called when aborting a request that has been admitted into staging
        but has not yet completed (i.e. req.hisparse_staging is True).
        """
        # Remove from staging queue
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        # Wait for any in-flight staging DMA to complete before freeing
        self.write_staging_stream.synchronize()

        prefill_len = req.extend_range.end
        adopted = int(getattr(req, "wlp_adopted_len", 0) or 0)
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, adopted:prefill_len
        ]
        self.token_to_kv_pool_allocator.free_hisparse(allocated_locs)

        # Free host memory that was allocated during admit_request_into_staging.
        # With an adopted (retained) prefix, rows [0, adopted) belong to the
        # radix tree and must survive the abort.
        if adopted > 0:
            host_len = self.host_token_len(
                int(self.req_to_host_pool_allocated_len[req.req_pool_idx])
            )
            host_indices = self.req_to_host_pool[
                req.req_pool_idx, self.host_token_len(adopted) : host_len
            ]
            host_indices = host_indices[host_indices >= 0]
            if host_indices.numel() > 0:
                self.mem_pool_host.free(host_indices)
            self.req_to_host_pool[
                req.req_pool_idx, self.host_token_len(adopted) :
            ] = -1
            self.req_to_host_pool_allocated_len[req.req_pool_idx] = (
                self.host_token_len(adopted)
            )
        else:
            host_indices = self.mem_pool_host.allocated_host_indices(
                self.req_to_host_pool,
                req.req_pool_idx,
                self.req_to_host_pool_allocated_len[req.req_pool_idx],
            )
            if host_indices.numel() > 0:
                self.mem_pool_host.free(host_indices)
            self.req_to_host_pool[req.req_pool_idx, :] = -1
            self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        # Any extend scratch still registered for this req goes back to the
        # device pool (the staging backup never consumed it).
        self.free_extend_scratch(req.req_pool_idx)
        self._skip_first_backup[req.req_pool_idx] = False
        req.hisparse_staging = False

    def retract_req(self, req: Req) -> None:
        if req.hisparse_staging:
            self.abort_staging_request(req)
        else:
            self.request_finished(req)

    def request_finished(self, req: Req):
        # release resources only after the execution of a potential overlapped batch
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()

        # Use kv_allocated_len (not seqlen): under speculative decoding the
        # allocator can over-allocate beyond the committed seqlen, and those
        # extra slots may carry stale mapping entries pointing at buffer slots
        # we just freed via free_hisparse_indices(all_hi). If left set, the
        # subsequent release_kv_cache -> allocator.free -> free_hisparse path
        # re-frees them (double-free into the page allocator's free list).
        allocated_len = req.kv.kv_allocated_len

        # release memory -- only free actually-allocated buffer indices
        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        if current_cap > 0:
            side_buf_hi = self.req_to_device_buffer[req.req_pool_idx, :current_cap]
            all_hi = torch.unique(side_buf_hi[side_buf_hi > 0])
            if all_hi.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(all_hi)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ]
        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            allocated_locs
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = 0

        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        # clear req info
        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req.req_pool_idx] = False

    # ------------------------------------------------------------------
    # Prefix retention (radix-over-hisparse)
    #
    # The radix tree retains finished requests' logical kv indices; their MLA
    # latent stays in the pinned host pool and their indexer keys stay valid in
    # the (full-logical-size, device-resident) index buffer because retained
    # logical indices are never reused. One CPU side table maps each retained
    # logical index to the host row that mirrors it; adoption of a matched
    # prefix is a gather from that table into the new request's per-request
    # host-row table. Device resources (buffer slots, mapping rows) are still
    # released at finish exactly as before.
    # ------------------------------------------------------------------
    def init_retention(self) -> None:
        assert self.compress_ratio == 1, (
            "hisparse prefix retention assumes compress_ratio == 1 "
            "(host rows are keyed positionally by token index)"
        )
        n = int(self.token_to_kv_pool_allocator.size_full) + self.page_size + 1
        self.logical_to_host_row = torch.full((n,), -1, dtype=torch.int64)
        self.req_adopted_len = torch.zeros(
            self.req_to_host_pool.shape[0], dtype=torch.int64
        )
        self.token_to_kv_pool_allocator.register_retention_cb(self.on_logical_free)

    def reset_retention(self) -> None:
        """Drop every retained row (tree reset / flush_cache). The allocator's
        clear() rebuilds its free lists without the retention hook, so the
        rows must be returned to the host pool here."""
        rows = self.logical_to_host_row[self.logical_to_host_row >= 0]
        if rows.numel() > 0:
            self.mem_pool_host.free(rows)
        self.logical_to_host_row.fill_(-1)
        self.req_adopted_len.zero_()

    def on_logical_free(self, free_index: torch.Tensor) -> None:
        """Allocator hook: a logical index leaving the allocator takes its
        retained host row (if any) with it."""
        idx = free_index.to(device="cpu", dtype=torch.int64)
        rows = self.logical_to_host_row[idx]
        held = rows >= 0
        if bool(held.any()):
            self.mem_pool_host.free(rows[held])
            self.logical_to_host_row[idx[held]] = -1

    def flush_pending_to_host(self, req: Req, target_len: int) -> int:
        """Synchronously complete the host mirror up to `target_len` tokens.

        The eager backup lags one compressed token (the newest lives only in
        the request's reserved device-buffer slot). Returns the number of
        tokens whose host mirror is verified complete — the retained prefix is
        truncated to this length, so an unrecoverable tail shortens the cache
        instead of ever publishing a hole.
        """
        idx = req.req_pool_idx
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()

        have = int(self.req_to_host_pool_allocated_len[idx])
        want = self.host_token_len(target_len)
        if want <= have:
            return min(target_len, have)

        # Recover pending positions from the per-request device buffer. With
        # the reserved-slot scheme only positions >= device_buffer_size share a
        # slot, and only the newest of those is still resident.
        pending = list(range(have, want))
        recoverable = []
        for p in pending:
            slot = min(p, self.device_buffer_size)
            if slot == self.device_buffer_size and p != want - 1:
                break  # older long-seq positions were overwritten; stop here
            if int(self.req_to_device_buffer[idx, slot]) <= 0:
                break
            recoverable.append((p, slot))
        if not recoverable:
            return have

        positions = [p for p, _ in recoverable]
        slots = torch.tensor(
            [s for _, s in recoverable], dtype=torch.int64, device=self.device
        )
        device_locs = self.req_to_device_buffer[idx, slots]
        host_locs = self.mem_pool_host.alloc_paged_token_slots(
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            idx,
            positions[0],
            len(positions),
        )
        self.mem_pool_host.backup_from_device_all_layer(
            self.mem_pool_device,
            host_locs,
            device_locs,
            io_backend="kernel",
        )
        device_module.current_stream().synchronize()
        return min(target_len, positions[-1] + 1)

    def host_rows_snapshot(self, req: Req) -> torch.Tensor:
        """Copy of this request's host rows (positional), before release."""
        idx = req.req_pool_idx
        n = int(self.req_to_host_pool_allocated_len[idx])
        return self.req_to_host_pool[idx, :n].clone()

    def retain_rows(self, logical_values: torch.Tensor, rows: torch.Tensor) -> None:
        """Move host-row ownership for `logical_values` to the side table."""
        assert logical_values.numel() == rows.numel()
        idx = logical_values.to(device="cpu", dtype=torch.int64)
        assert bool(
            (self.logical_to_host_row[idx] < 0).all()
        ), "retain_rows: logical index already holds a retained row"
        self.logical_to_host_row[idx] = rows.to(device="cpu", dtype=torch.int64)

    def adopt_prefix(self, req: Req, prefix_indices: torch.Tensor) -> None:
        """Point a new request's host-row table at a retained prefix."""
        idx = req.req_pool_idx
        n = prefix_indices.numel()
        if n == 0:
            self.req_adopted_len[idx] = 0
            return
        rows = self.logical_to_host_row[
            prefix_indices.to(device="cpu", dtype=torch.int64)
        ]
        assert bool((rows >= 0).all()), (
            "adopt_prefix: matched prefix has un-retained host rows"
        )
        self.req_to_host_pool[idx, :n] = rows
        self.req_to_host_pool_allocated_len[idx] = n
        self.req_adopted_len[idx] = n
        logger.info(
            "HiSparse retention: adopted %d-token host-resident prefix for %s",
            n,
            req.rid,
        )

    def unadopt_prefix(self, req: Req) -> None:
        """Roll back adopt_prefix: clear this request's host-row table.

        The rows themselves stay owned by the radix side table
        (``logical_to_host_row``); only the per-request view is dropped, so a
        retried pre-allocation re-adopts them cleanly. Used when host-pool
        exhaustion aborts a partially-completed pre-allocation.
        """
        idx = req.req_pool_idx
        n = int(self.req_adopted_len[idx])
        if n > 0:
            self.req_to_host_pool[idx, :n] = -1
        self.req_to_host_pool_allocated_len[idx] = 0
        self.req_adopted_len[idx] = 0

    def release_for_retention(self, req: Req) -> None:
        """request_finished minus the host-row free: device buffer slots,
        mapping rows and per-request tables are released; host bytes survive
        (ownership passes to the radix tree via retain_rows)."""
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()

        allocated_len = req.kv.kv_allocated_len
        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        if current_cap > 0:
            side_buf_hi = self.req_to_device_buffer[req.req_pool_idx, :current_cap]
            all_hi = torch.unique(side_buf_hi[side_buf_hi > 0])
            if all_hi.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(all_hi)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ]
        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            allocated_locs
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = 0

        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        self.req_adopted_len[req.req_pool_idx] = 0
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req.req_pool_idx] = False

    def free_unretained_rows(self, rows: torch.Tensor) -> None:
        """Free host rows that were never moved to the side table."""
        rows = rows[rows >= 0]
        if rows.numel() > 0:
            self.mem_pool_host.free(rows)

    def _run_swap_in_kernel(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        record_plan: bool = False,
        num_newest: int = 1,
        plan_slot: Optional[int] = None,
    ) -> torch.Tensor:
        """Run the swap-in kernel for one layer; return its slot table.

        record_plan (set on the anchor of a shared-index group) also records the
        miss plan for the skip layers to replay: into self._miss_{src,dst,count}
        for decode (plan_slot=None) or into the per-position
        self._miss_{src,dst,count}_v[plan_slot] for MTP verify.
        num_newest: the tokens [seq_len - num_newest, seq_len) were written this
        step and resolve to the reserved page (MTP target-verify; 1 = decode).

        With the wide gather enabled the fused kernel runs in plan-only mode
        (skip_io; hit/LRU/evict decisions and the recorded plan are identical --
        the elided IO was strictly after planning) and the planned rows are
        then copied by a full-GPU-grid kernel.
        """
        num_reqs = req_pool_indices.size(0)
        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]
        if num_newest != 1:
            assert not self.is_dsv4_hisparse, "MTP verify swap-in: DSA hisparse only"

        swap_in_fn = (
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
        )
        plan_only = self._wide_gather
        plan = (
            dict(
                miss_src=self._miss_src[:num_reqs],
                miss_dst=self._miss_dst[:num_reqs],
                miss_count=self._miss_count[:num_reqs],
            )
            if record_plan or plan_only
            else {}
        )
        swap_in_fn(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            skip_io=self.skip_io or plan_only,
            **plan,
            **({} if self.is_dsv4_hisparse else {"num_newest": num_newest}),
        )
        if plan_only and not self.skip_io:
            self._run_wide_copy_kernel(num_reqs, layer_id)
        return top_k_indices

    def _wide_copy_blocks(self, num_reqs: int) -> int:
        """Grid for the wide plan-driven copy: one warp per worst-case row
        (num_reqs * top_k rows / warps per block), capped at 4 blocks per SM
        (the measured sweet spot for a per-layer random gather; more blocks
        cannot help once every SM has rows in flight)."""
        warps_per_block = 256 // 32
        wanted = (num_reqs * self.top_k + warps_per_block - 1) // warps_per_block
        return min(wanted, 4 * self._sm_count)

    def _run_wide_copy_kernel(self, num_reqs: int, layer_id: int) -> None:
        """Copy this layer's recorded miss plan host->device with a
        full-GPU-grid gather (warp per planned row; see lanes/gather for why
        the C2C link needs the whole grid at small batch)."""
        copy_cache_planned_wide_mla(
            miss_src=self._miss_src[:num_reqs],
            miss_dst=self._miss_dst[:num_reqs],
            miss_count=self._miss_count[:num_reqs],
            num_real_reqs=self.num_real_reqs,
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_blocks=self._wide_copy_blocks(num_reqs),
        )

    def _run_copy_only_kernel(self, num_reqs: int, skip_layer: int) -> None:
        """Replay the anchor's recorded miss plan into a skip layer's buffers
        (IO-only; the anchor's slot table stays valid -- lockstep layout)."""
        if self._wide_gather and not self.skip_io:
            self._run_wide_copy_kernel(num_reqs, skip_layer)
            return
        copy_cache_planned_mla(
            miss_src=miss_src,
            miss_dst=miss_dst,
            miss_count=miss_count,
            num_real_reqs=self.num_real_reqs,
            host_cache=self.mem_pool_host.kv_buffer[skip_layer],
            device_buffer=self.mem_pool_device.kv_buffer[skip_layer],
            item_size_bytes=self.item_size_bytes,
            num_blocks=self._prefetch_copy_blocks,
            is_dsv4_layout=self.is_dsv4_hisparse,
            skip_io=self.skip_io,
        )

    def _maybe_green_swap_in(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        record_plan: bool = False,
        num_newest: int = 1,
    ) -> torch.Tensor:
        """Swap-in on the green-context stream when the flag is set (fork+join
        with events; capture-safe inside CUDA graphs), else on the current
        stream. Same kernels/args/order either way, so outputs are identical."""
        if self._swapin_stream is None or not self._swapin_on_green:
            return self._run_swap_in_kernel(
                req_pool_indices,
                compressed_seq_lens,
                top_k_result,
                layer_id,
                record_plan=record_plan,
                num_newest=num_newest,
            )
        cur = device_module.current_stream()
        self._swapin_stream.wait_stream(cur)
        with device_module.stream(self._swapin_stream):
            out = self._run_swap_in_kernel(
                req_pool_indices,
                compressed_seq_lens,
                top_k_result,
                layer_id,
                record_plan=record_plan,
                num_newest=num_newest,
            )
        cur.wait_stream(self._swapin_stream)
        return out

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        num_newest: int = 1,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices.

        With prefetch enabled, anchors plan (and, unless the wide gather is
        on, copy) synchronously, recording the miss plan, and prefetch their
        skip layers' copies; skip layers just wait. num_newest > 1 is an MTP
        target-verify position (prefetch is off under speculative decoding, so
        it always takes the direct path).
        """
        if not self.enable_prefetch or num_newest != 1:
            return self._maybe_green_swap_in(
                req_pool_indices,
                compressed_seq_lens,
                top_k_result,
                layer_id,
                num_newest=num_newest,
            )

        num_reqs = req_pool_indices.size(0)
        # num_newest == 1 is ambiguous between plain decode and verify
        # position 0 (both pass 1): under speculation (num_draft_tokens > 1)
        # the target only runs TARGET_VERIFY forwards, so route to the verify
        # path; without speculation there are no verify calls at all.
        if num_newest != 1 or self.num_draft_tokens > 1:
            return self._verify_swap_in(
                req_pool_indices,
                compressed_seq_lens,
                top_k_result,
                layer_id,
                num_newest,
                num_reqs,
            )

        if self._is_shared_index_layer[layer_id]:
            # Skip layer: wait for its prefetched copy; the anchor's slot table
            # applies (shared index + lockstep buffers).
            slot = self._prefetch_slot[layer_id]
            self._prefetch_events[slot].wait(device_module.current_stream())
            return self.top_k_device_locs_buffer[:num_reqs]

        # Anchor: swap in synchronously (recording the plan), then prefetch the
        # skip layers' copies on the side stream.
        group = self._prefetch_groups.get(layer_id)
        anchor_locs = self._maybe_green_swap_in(
            req_pool_indices,
            compressed_seq_lens,
            top_k_result,
            layer_id,
            record_plan=group is not None,
        )
        if group:
            # Fork: the prefetch stream must observe the anchor's plan (produced
            # on the current stream) before replaying it.
            self.prefetch_stream.wait_stream(device_module.current_stream())
            with device_module.stream(self.prefetch_stream):
                for skip_layer in group:
                    self._run_copy_only_kernel(num_reqs, skip_layer)
                    self._prefetch_events[self._prefetch_slot[skip_layer]].record(
                        self.prefetch_stream
                    )
        return anchor_locs

    def _verify_swap_in(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        num_newest: int,
        num_reqs: int,
    ) -> torch.Tensor:
        """MTP target-verify swap-in for one (layer, position) call.

        The caller (dsa_backend's verify branch) invokes this per layer and per
        draft position p = num_newest-1 (p = 0 passes num_newest=1, same as
        decode — swap_in_selected_pages routes on num_draft_tokens).
        Anchors run the fused kernel per position (synchronously -- their own
        attention needs the rows) recording per-position plans; the last
        position's call issues the whole group's multi-position copies on the
        prefetch stream. Skip layers wait for their group's copies and return
        the anchor's stashed per-position slot table (lockstep layout).
        """
        assert self.num_draft_tokens > 1, "verify swap-in without draft tokens"
        p = num_newest - 1
        if self._is_shared_index_layer[layer_id]:
            slot = self._prefetch_slot[layer_id]
            self._prefetch_events_v[slot].wait(device_module.current_stream())
            return self._verify_slot_table[p][:num_reqs]

        group = self._prefetch_groups.get(layer_id)
        anchor_locs = self._run_swap_in_kernel(
            req_pool_indices,
            compressed_seq_lens,
            top_k_result,
            layer_id,
            record_plan=group is not None,
            num_newest=num_newest,
            plan_slot=p,
        )
        # Stash this position's slot table before the next position's kernel
        # overwrites top_k_device_locs_buffer (skip layers replay it).
        self._verify_slot_table[p][:num_reqs].copy_(anchor_locs)
        if group and num_newest == self.num_draft_tokens:
            # All positions' plans are recorded: fork once and issue one
            # multi-position IO group per skip layer. The per-position copies
            # are independent (a token missed by position p is a hit for every
            # other position, so each planned row belongs to exactly one plan).
            self.prefetch_stream.wait_stream(device_module.current_stream())
            with device_module.stream(self.prefetch_stream):
                for skip_layer in group:
                    for q in range(self.num_draft_tokens):
                        self._run_copy_only_kernel(num_reqs, skip_layer, plan_slot=q)
                    self._prefetch_events_v[self._prefetch_slot[skip_layer]].record(
                        self.prefetch_stream
                    )
        return anchor_locs
