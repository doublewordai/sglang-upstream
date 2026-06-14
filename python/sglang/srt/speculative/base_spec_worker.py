from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker


class BaseDraftWorker(ABC):
    @abstractmethod
    def draft():
        pass

    @abstractmethod
    def draft_extend():
        pass


class BaseSpecWorker(ABC):
    @property
    @abstractmethod
    def target_worker(self) -> TpModelWorker:
        pass

    @property
    @abstractmethod
    def draft_worker(self) -> BaseDraftWorker:
        pass

    @property
    def spec_v2_attn_backends(self) -> tuple:
        """Attn backends touched by spec_v2 forward; OR-ed by decide_needs_cpu_seq_lens.
        Default returns target only; subclasses extend with draft backends."""
        return (self.target_worker.model_runner.attn_backend,)

    @abstractmethod
    def clear_cache_pool(self):
        # TODO: move this abstract method to BaseTpWorker and call through self.model_runner
        pass

    def on_verify_complete_cpu(
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int = 0,
        rids: list[str] | None = None,
        num_steps: int | None = None,
    ) -> None:
        """Hook called after verify finishes and accept counts are on CPU.

        Default no-op. Adaptive-aware workers override this to feed the
        controller without forcing a GPU→CPU sync in the worker hot path.
        *rids* and *num_steps* (the active draft steps the result was
        produced with) feed per-request policies; EMA-style policies
        ignore them.
        """
        pass

    def activate_step_by_batch(
        self,
        batch_size: int,
        rids: list[str] | None = None,
        round_idx: int | None = None,
    ) -> None:
        """Activate the optimal adaptive step for the current batch size.

        Default no-op. Adaptive-aware workers override this to switch
        the runtime state before each draft round. *rids* and *round_idx*
        (a monotonically increasing worker forward-round index) feed
        per-request priced policies; EMA-style policies ignore them.
        """
        pass
