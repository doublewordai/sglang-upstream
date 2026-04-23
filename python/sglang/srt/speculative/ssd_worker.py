from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from sglang.srt.layers.logits_processor import CaptureHiddenMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.ssd_utils import build_ssd_verify_payload
from sglang.srt.speculative.speculative_utils import set_time_batch

logger = logging.getLogger(__name__)


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
        self.model_runner = target_worker.model_runner
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
        for req in batch.reqs:
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
            payloads.append(
                build_ssd_verify_payload(
                    current_token=pending.bonus_token,
                    prefix_len=len(req.origin_input_ids) + len(req.output_ids),
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
    ) -> None:
        candidates = spec_info.draft_token.view(
            len(batch.reqs), self.speculative_num_draft_tokens
        )
        verified_tokens = verified_id.detach().cpu().tolist()
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
            self._pending_acceptance[req.rid] = PendingAcceptance(
                accepted_tokens=accepted_tokens,
                bonus_token=int(verified_tokens[idx]),
            )

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
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

        set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)
        spec_info = self._query_remote_drafts(batch)
        set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)
        set_time_batch(batch.reqs, "set_spec_verify_start_time", trace_only=True)

        logits_output, verify_output, _model_worker_batch, can_run_cuda_graph = self.verify(
            batch, spec_info
        )

        self._cache_acceptance(
            batch,
            spec_info,
            verify_output.verified_id,
            verify_output.accept_length_per_req_cpu,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.verified_id,
            num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
            accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
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
