from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol

from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    read_adaptive_policy_name,
)
from sglang.srt.speculative.spec_telemetry import capture_draft_confidence

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
    from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
        EAGLEDraftExtendCudaGraphRunner,
    )


@dataclass
class SpecRuntimeState:
    """A complete set of runtime resources bound to a specific speculative
    decoding configuration.

    Each decode round runs three stages — draft, verify, extend — and every
    stage has shape-dependent resources (attention backends and CUDA graphs)
    that must match the current configuration.  Switching adaptive steps
    means swapping the entire state atomically.

    ``speculative_num_steps == 0`` is the speculation-off state: all
    draft-side members are None and the target members are plain-decode
    resources (1 token per request, ForwardMode.DECODE).
    """

    # -- Configuration (determines shapes for all stages) --
    speculative_num_steps: int
    speculative_num_draft_tokens: int

    # -- Draft stage: draft model multi-step autoregressive generation --
    draft_attn_backend: "AttentionBackend | None"
    cuda_graph_runner: "EAGLEDraftCudaGraphRunner | None"

    # -- Verify stage: target model one-pass tree verification --
    target_attn_backend: "AttentionBackend"
    target_graph_runner: "DecodeCudaGraphRunner | CPUGraphRunner | None"

    # -- Extend stage: draft model KV cache catch-up after verify --
    draft_extend_attn_backend: "AttentionBackend | None"
    cuda_graph_runner_for_draft_extend: "EAGLEDraftExtendCudaGraphRunner | None"


class SpeculativeStepPolicy(Protocol):
    """Minimal interface AdaptiveController needs from a step policy.

    Implemented by ``AdaptiveSpeculativeParams`` (acceptance-EMA, the
    default) and ``PricedSpeculativeParams`` (goodput-priced). *rids*,
    *round_idx*, and *num_steps* are richer priced-policy inputs; the EMA
    policy accepts and ignores them.
    """

    @property
    def candidate_steps(self) -> list[int]: ...

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None: ...

    def get_steps_for_batch(
        self,
        batch_size: int,
        rids: list[str] | None = None,
        round_idx: int | None = None,
    ) -> int: ...

    def on_verify_complete(
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int,
        rids: list[str] | None = None,
        num_steps: int | None = None,
    ) -> int | None: ...

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None: ...


class AdaptiveSpecWorker(Protocol):
    """Protocol that a worker must implement to use AdaptiveController."""

    speculative_num_steps: int

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ) -> SpecRuntimeState: ...

    def apply_runtime_state(self, state: SpecRuntimeState) -> None: ...


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.

    Works with any worker that implements AdaptiveSpecWorker protocol:
      - build_adaptive_runtime_state(steps, draft_tokens) → runtime state
      - apply_runtime_state(state) → apply it to the worker

    The worker only needs to:
      1. Call register() for the initial state, then init_states()
         once during startup.
      2. Call on_verify_complete(num_correct_drafts_per_req) after each decode verify.
    """

    def __init__(self, worker: AdaptiveSpecWorker, config_path: str | None = None):
        self.worker = worker
        self.policy_name = read_adaptive_policy_name(config_path)
        self.params: SpeculativeStepPolicy
        if self.policy_name == "priced":
            from sglang.srt.speculative.priced_spec_params import (
                PricedSpeculativeParams,
            )

            self.params = PricedSpeculativeParams(
                initial_steps=worker.speculative_num_steps,
                cfg_path=config_path,
            )
        else:
            self.params = AdaptiveSpeculativeParams(
                initial_steps=worker.speculative_num_steps,
                cfg_path=config_path,
            )
        self._states: dict[int, SpecRuntimeState] = {}

    @property
    def needs_request_context(self) -> bool:
        """Whether the worker should pass rids/round_idx into the per-round
        activate decision (priced policy only)."""
        return self.policy_name == "priced"

    @property
    def wants_draft_confidences(self) -> bool:
        """Whether the worker should read back draft confidences each round
        and feed them to the policy. Gated on the capture env so the EMA
        policy (and uninstrumented priced runs) never pay the GPU sync."""
        return self.policy_name == "priced" and capture_draft_confidence()

    @property
    def candidate_steps(self) -> list[int]:
        return self.params.candidate_steps

    @property
    def g0_max_gap(self) -> int:
        """Per-request drafter-gap cap for g=0 (speculation-off) phases.
        Only the priced policy can select g=0; others fall back to the
        default bound (never reached)."""
        return getattr(self.params, "g0_max_gap", 1024)

    def register(self, state: SpecRuntimeState, steps: int | None = None) -> None:
        """Register a pre-built runtime state.

        *steps* defaults to state.speculative_num_steps when not given.
        """
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state

    def init_states(self, cuda_graph_bs: list[int] | None = None) -> None:
        """Build and register runtime states for all candidate steps."""
        self.params.set_cuda_graph_bs(cuda_graph_bs)

        for steps in self.candidate_steps:
            if steps in self._states:
                continue

            pruned_bs = self.params.cuda_graph_bs_for_step(steps)
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                cuda_graph_bs=pruned_bs,
            )
            self._states[steps] = state

        if self.policy_name == "priced":
            # Decisions only over candidates whose runtime states exist.
            self.params.set_available_steps(sorted(self._states))

        # Start on the initial step.
        self._activate(self.worker.speculative_num_steps)

    def activate_step_by_batch(
        self,
        batch_size: int,
        rids: list[str] | None = None,
        round_idx: int | None = None,
    ) -> None:
        target = self.params.get_steps_for_batch(
            batch_size, rids=rids, round_idx=round_idx
        )
        if target != self.worker.speculative_num_steps:
            self._activate(target)

    def on_verify_complete(
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int,
        rids: list[str] | None = None,
        num_steps: int | None = None,
    ) -> None:
        """Feed verify results; switch runtime state if the policy warrants it."""
        new_step = self.params.on_verify_complete(
            num_correct_drafts_per_req,
            batch_size,
            rids=rids,
            num_steps=num_steps,
        )
        if new_step is not None:
            self._activate(new_step)

    def observe_draft_confidences(
        self, rids: list[str], confidences: Optional[list[list[float]]]
    ) -> None:
        """Hand the just-finished round's per-request draft confidences to the
        policy (used as acceptance estimates for the next round's decision).
        Only meaningful when ``wants_draft_confidences`` is True."""
        if confidences is None or self.policy_name != "priced":
            return
        self.params.observe_draft_confidences(rids, confidences)

    def _activate(self, speculative_num_steps: int) -> None:
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)
