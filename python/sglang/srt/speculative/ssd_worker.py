from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from sglang.srt.layers.logits_processor import CaptureHiddenMode, LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import (
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.ssd_utils import (
    build_ssd_verify_payload,
    extract_ssd_bonus_tokens,
)

logger = logging.getLogger(__name__)


def _group_verified_tokens(
    verified_id: torch.Tensor, accept_lengths: Sequence[int]
) -> List[List[int]]:
    flat_tokens = [int(x) for x in verified_id.detach().cpu().reshape(-1).tolist()]
    grouped: List[List[int]] = []
    offset = 0
    for accepted in accept_lengths:
        width = int(accepted) + 1
        grouped.append(flat_tokens[offset : offset + width])
        offset += width
    return grouped


def _tensor_to_int_list(values: Optional[torch.Tensor]) -> Optional[List[int]]:
    if values is None:
        return None
    return [int(x) for x in values.detach().cpu().reshape(-1).tolist()]


@dataclass
class PendingAcceptance:
    accepted_tokens: List[int]
    bonus_token: int


class SSDWorker(EAGLEWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        self._model_runner = target_worker.model_runner
        self.model_config = target_worker.model_config
        self.hot_token_id = None
        self.adaptive_controller = None
        self._pending_acceptance: dict[str, PendingAcceptance] = {}

        if not server_args.ssd_orchestrator_addr:
            raise ValueError(
                "SSD speculative decoding requires --ssd-orchestrator-addr."
            )

        try:
            from ssd_client import SSDClient
        except ImportError as exc:
            raise ImportError(
                "Failed to import ssd_client. Build/install the Rust-backed SSD client "
                "from the sibling ssd-sglang workspace before enabling "
                "speculative_algorithm=SSD."
            ) from exc

        self.client = SSDClient(server_args.ssd_orchestrator_addr)

    def _current_token(self, req) -> int:
        if req.output_ids:
            return int(req.output_ids[-1])
        if req.origin_input_ids:
            return int(req.origin_input_ids[-1])
        raise ValueError(
            "SSD speculative decoding cannot determine the current token "
            f"for request {req.rid}."
        )

    def _prefix_tokens(self, req) -> List[int]:
        return [int(x) for x in (req.origin_input_ids + req.output_ids)]

    def _query_remote_drafts(self, batch: ScheduleBatch) -> EagleVerifyInput:
        payloads = []
        for idx, req in enumerate(batch.reqs):
            pending = self._pending_acceptance.pop(
                req.rid,
                PendingAcceptance(
                    accepted_tokens=[],
                    bonus_token=self._current_token(req),
                ),
            )
            draft_result = self.client.query_drafts(
                req.rid,
                pending.accepted_tokens,
                pending.bonus_token,
                self.speculative_num_draft_tokens,
                self.topk,
            )
            # When bonus was suppressed (j > 0 previous round), bonus_token is 0.
            # Use last accepted token as the fallback for the draft tree.
            if pending.accepted_tokens:
                current_token = pending.accepted_tokens[-1]
            elif pending.bonus_token != 0:
                current_token = pending.bonus_token
            else:
                current_token = self._current_token(req)
            # Verify masks are sized against the KV-cached prefix, which excludes
            # the token being verified in this forward.
            prefix_len = int(batch.seq_lens_cpu[idx])
            payloads.append(
                build_ssd_verify_payload(
                    current_token=current_token,
                    prefix_len=prefix_len,
                    num_draft_tokens=self.speculative_num_draft_tokens,
                    draft_result=draft_result,
                )
            )

        return EagleVerifyInput(
            draft_token=torch.tensor(
                [token for payload in payloads for token in payload["draft_tokens"]],
                dtype=torch.long,
                device=batch.device,
            ),
            custom_mask=torch.tensor(
                [mask for payload in payloads for mask in payload["custom_mask"]],
                dtype=torch.bool,
                device=batch.device,
            ),
            positions=torch.tensor(
                [pos for payload in payloads for pos in payload["positions"]],
                dtype=torch.int64,
                device=batch.device,
            ),
            retrieve_index=torch.tensor(
                [payload["retrieve_index"] for payload in payloads],
                dtype=torch.long,
                device=batch.device,
            ),
            retrieve_next_token=torch.tensor(
                [payload["retrieve_next_token"] for payload in payloads],
                dtype=torch.long,
                device=batch.device,
            ),
            retrieve_next_sibling=torch.tensor(
                [payload["retrieve_next_sibling"] for payload in payloads],
                dtype=torch.long,
                device=batch.device,
            ),
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=None,
            seq_lens_cpu=batch.seq_lens_cpu.clone(),
        )

    def _cache_acceptance(
        self,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        verified_id: torch.Tensor,
        accept_lengths: Sequence[int],
        suppress_mask: List[bool],
    ) -> None:
        candidates = spec_info.draft_token.view(
            len(batch.reqs), self.speculative_num_draft_tokens
        )
        bonus_tokens = extract_ssd_bonus_tokens(verified_id, accept_lengths)
        verified_tokens = verified_id.detach().cpu().tolist()
        token_offset = 0
        for idx, req in enumerate(batch.reqs):
            accepted = int(accept_lengths[idx])
            accepted_tokens = []
            if accepted > 0:
                accepted_tokens = [
                    int(x)
                    for x in candidates[
                        idx, 1 : accepted + 1
                    ].detach().cpu().tolist()
                ]
            req_verified_tokens = verified_tokens[token_offset : token_offset + accepted + 1]
            token_offset += accepted + 1

            if suppress_mask[idx]:
                # Bonus suppressed: send accepted drafts only, no bonus.
                # The orchestrator will advance the trie by the accepted tokens.
                self._pending_acceptance[req.rid] = PendingAcceptance(
                    accepted_tokens=accepted_tokens,
                    bonus_token=0,
                )
            else:
                self._pending_acceptance[req.rid] = PendingAcceptance(
                    accepted_tokens=accepted_tokens,
                    bonus_token=bonus_tokens[idx],
                )

            logger.info(
                "SSD verify rid=%s accepted_len=%d accepted_tokens=%s bonus_token=%d "
                "verified_tokens=%s suppressed=%s",
                req.rid,
                accepted,
                accepted_tokens,
                self._pending_acceptance[req.rid].bonus_token,
                req_verified_tokens,
                suppress_mask[idx],
            )

    def _suppress_bonus_tokens(
        self,
        batch: ScheduleBatch,
        verify_output: EagleVerifyOutput,
    ) -> tuple[List[int], List[bool]]:
        """Post-process verify output to suppress bonus tokens for j > 0 requests.

        When j > 0 draft tokens are accepted, the bonus token invalidates the
        pre-computed speculation trie. We undo its effects: pop from output_ids,
        roll back KV tracking and seq_lens, and free the cache slot.

        Returns:
            ssd_accept_lengths: Adjusted accept lengths (j-1 for j>0, 0 for j=0)
                that preserve the convention accept_length + 1 = total tokens.
            suppress_mask: Per-request bool indicating if the bonus was suppressed.
        """
        accept_lengths = verify_output.accept_length_per_req_cpu
        suppress_mask: List[bool] = []
        slots_to_free = []

        for i, req in enumerate(batch.reqs):
            j = int(accept_lengths[i])
            suppress = j > 0 and not req.finished()
            suppress_mask.append(suppress)
            if suppress:
                # The bonus token's KV slot is at position seq_lens[i] - 1
                # (verify already advanced seq_lens by j + 1).
                bonus_pos = int(batch.seq_lens[i].item()) - 1
                bonus_slot = self.req_to_token_pool.req_to_token[
                    batch.req_pool_indices[i], bonus_pos
                ]
                slots_to_free.append(bonus_slot)

                req.output_ids.pop()
                req.kv_committed_len -= 1
                req.kv_allocated_len -= 1

        # Free bonus KV cache slots in one batched call.
        if slots_to_free:
            self.token_to_kv_pool_allocator.free(
                torch.stack(slots_to_free)
            )

        # Roll back seq_lens for suppressed requests.
        suppress_tensor = torch.tensor(
            suppress_mask, device=batch.device, dtype=batch.seq_lens.dtype
        )
        batch.seq_lens.sub_(suppress_tensor)
        for i, suppress in enumerate(suppress_mask):
            if suppress:
                batch.seq_lens_cpu[i] -= 1

        # Adjusted accept lengths: max(j - 1, 0) preserves the convention
        # that accept_length + 1 = total tokens produced per request.
        ssd_accept_lengths = [max(int(j) - 1, 0) for j in accept_lengths]

        return ssd_accept_lengths, suppress_mask

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        # Extend means prefill
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            (
                logits_output,
                next_token_ids,
                _seq_lens_cpu,
                can_run_cuda_graph,
            ) = self.forward_target_extend(batch)
            next_token_ids_cpu = next_token_ids.detach().cpu().tolist()
            for req, next_token_id in zip(batch.reqs, next_token_ids_cpu):
                prefix_tokens = self._prefix_tokens(req)
                registered = self.client.register_request(
                    req.rid,
                    prefix_tokens,
                    self.speculative_num_draft_tokens,
                    self.topk,
                    self.speculative_num_steps,
                )
                if not registered:
                    raise RuntimeError(
                        f"SSD orchestrator rejected register_request for {req.rid}."
                    )
                self._pending_acceptance[req.rid] = PendingAcceptance(
                    accepted_tokens=[],
                    bonus_token=int(next_token_id),
                )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=can_run_cuda_graph,
            )
        else:
                
            set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)
            spec_info = self._query_remote_drafts(batch)
            set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)
            set_time_batch(batch.reqs, "set_spec_verify_start_time", trace_only=True)

            logits_output, verify_output, _model_worker_batch, can_run_cuda_graph = self.verify(
                batch, spec_info
            )

            # Suppress bonus tokens for j > 0 requests. This undoes the bonus
            # token's effects on output_ids, KV cache, and seq_lens so that the
            # pre-computed speculation trie remains valid.
            ssd_accept_lengths, suppress_mask = self._suppress_bonus_tokens(
                batch, verify_output
            )

            self._cache_acceptance(
                batch,
                spec_info,
                verify_output.verified_id,
                verify_output.accept_length_per_req_cpu,
                suppress_mask,
            )
            verified_tokens = _group_verified_tokens(
                verify_output.verified_id, verify_output.accept_length_per_req_cpu
            )
            output_tokens = [
                tokens[:-1] if suppress else tokens
                for tokens, suppress in zip(verified_tokens, suppress_mask)
            ]
            logger.info(
                "Forward trace path=ssd-target-verify mode=%s rids=%s output_tokens=%s "
                "verified_tokens=%s kv_cache_seq_lens_after=%s suppressed_bonus=%s",
                batch.forward_mode.name,
                [req.rid for req in batch.reqs],
                output_tokens,
                verified_tokens,
                [int(x) for x in batch.seq_lens_cpu.tolist()],
                suppress_mask,
            )

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(ssd_accept_lengths),
                accept_length_per_req_cpu=ssd_accept_lengths,
                can_run_cuda_graph=can_run_cuda_graph,
            )

    def _mamba_verify_update(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
        spec_info: EagleVerifyInput,
        seq_lens_pre_verify: torch.Tensor,
    ):
        """SSD override: adjust mamba state to target the last accepted draft
        token instead of the bonus token for j > 0 requests.

        The parent implementation selects the mamba state at step j (the bonus).
        For SSD bonus suppression we need the state at step j-1 (the last
        accepted draft), so accepted_steps is decremented by 1 for j > 0.
        """
        if batch.forward_mode.is_idle():
            return

        device = logits_output.hidden_states.device
        accept_lengths_cpu = res.accept_length_per_req_cpu

        accepted_length = (
            torch.tensor(accept_lengths_cpu, device=device, dtype=torch.int64) + 1
        )
        cumulative_accepted_lengths = torch.cumsum(accepted_length, dim=0)
        accepted_indices_start = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=cumulative_accepted_lengths.dtype,
                    device=cumulative_accepted_lengths.device,
                ),
                cumulative_accepted_lengths[:-1],
            ]
        )
        accepted_indices_offset = torch.arange(
            0,
            len(batch.seq_lens) * batch.spec_info.draft_token_num,
            step=batch.spec_info.draft_token_num,
            dtype=accepted_indices_start.dtype,
            device=accepted_indices_start.device,
        )

        if spec_info.topk > 1 and res.accepted_indices.shape[0] > 0:
            # For topk > 1: pick the second-to-last accepted index for j > 0
            # (last accepted draft) instead of the last (bonus).
            last_indices = cumulative_accepted_lengths - 1
            ssd_mask = torch.tensor(
                [j > 0 for j in accept_lengths_cpu], device=device, dtype=torch.bool
            )
            ssd_adjusted_indices = torch.where(
                ssd_mask,
                torch.clamp(last_indices - 1, min=accepted_indices_start),
                last_indices,
            )
            accepted_steps = (
                res.accepted_indices[ssd_adjusted_indices] - accepted_indices_offset
            )
        else:
            accepted_steps = accepted_length - 1
            # For j > 0: step back by 1 (suppress bonus).
            ssd_mask = torch.tensor(
                [j > 0 for j in accept_lengths_cpu], device=device, dtype=torch.bool
            )
            accepted_steps = torch.where(
                ssd_mask,
                torch.clamp(accepted_steps - 1, min=0),
                accepted_steps,
            )

        if batch.mamba_track_indices is not None:
            mamba_track_interval = self.server_args.mamba_track_interval
            # Use SSD-adjusted seq_lens for tracking boundary checks.
            # batch.seq_lens already includes the bonus (+1); subtract 1 for j > 0.
            ssd_seq_lens = batch.seq_lens - ssd_mask.to(batch.seq_lens.dtype)
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != ssd_seq_lens // mamba_track_interval
            )
            tracking_point = (
                ssd_seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(
                tracking_point - seq_lens_pre_verify - 1, min=0
            )
            mamba_steps_to_track = torch.where(
                to_track_mask,
                res.accepted_indices[to_track_ith + accepted_indices_start]
                - accepted_indices_offset,
                -1,
            )
        else:
            mamba_steps_to_track = None

        logger.info(
            "Mamba trace path=ssd-target-verify rids=%s cache_src_steps=%s "
            "track_cache_indices=%s track_src_steps=%s",
            [req.rid for req in batch.reqs],
            _tensor_to_int_list(accepted_steps),
            _tensor_to_int_list(batch.mamba_track_indices),
            _tensor_to_int_list(mamba_steps_to_track),
        )

        self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def forward_draft_extend(self, *args, **kwargs):
        return None

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        return None

    def finish_request(self, request_id: str) -> bool:
        self._pending_acceptance.pop(request_id, None)
        finished = bool(self.client.finish_request(request_id))
        if not finished:
            raise RuntimeError(
                f"SSD orchestrator rejected finish_request for {request_id}."
            )
        return finished
