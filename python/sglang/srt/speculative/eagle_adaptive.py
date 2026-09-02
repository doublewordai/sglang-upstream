"""Adaptive (variable-length) verify scheduling for EAGLE/NextN chains.

DSpark-style per-request verify budgets (see the DSpark blog + dspark_planner.py)
applied to our EAGLE/NextN chain draft (topk=1):

- Confidence per draft step k = P_draft(drafted token k), read from the draft's
  own logits (topk_p for k=0 from draft-extend; the draft_topk1_postprocess
  finalize kernel for k>=1). Survival probs = cumprod(confidence).
- The SPS-argmax budget (compute_verify_token_budget) picks how many extra
  draft-token slots the step gets; ScheduleVerifyLensTopk distributes them to
  requests by survival probability (per-request position-prefix by construction,
  since survival is non-increasing in k).
- The resulting verify_lens [bs] ride a RaggedVerifyLayout so the target verify
  runs front-packed on token-keyed CUDA graphs, and the acceptance kernels see
  a tree truncated at verify_len (unverified nodes do not exist).

Outputs stay exact: every committed token is still the target's argmax at its
position; the verify width only changes how many tokens each request advances
per step.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    ScheduleVerifyLensTopk,
)
from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    compute_verify_token_budget,
    ragged_capture_num_tokens,
)
from sglang.srt.speculative.dspark_components.dspark_sps import (
    load_sps_table_from_path,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    RaggedVerifyMode,
    read_ragged_verify_mode,
)

logger = logging.getLogger(__name__)


def eagle_ragged_verify_enabled() -> bool:
    """Compact ragged-verify graphs are active for this process."""
    try:
        return read_ragged_verify_mode() is RaggedVerifyMode.COMPACT
    except ValueError:
        return False


class EagleAdaptiveVerifyScheduler:
    """Per-step verify-length scheduler for EAGLE chains.

    ``sps_table``: an SpsCostTable / SpsAdditiveCostTable JSON (dspark_sps.py
    format) giving step time vs (bs, total verify tokens). Without a table the
    schedule degenerates to verify-all (full width through the ragged graphs),
    which is still a useful constant-layout test of the ragged plumbing.
    """

    def __init__(self, *, num_steps: int, num_draft_tokens: int) -> None:
        if num_draft_tokens != num_steps + 1:
            raise ValueError(
                "adaptive verify supports topk=1 chains: num_draft_tokens "
                f"({num_draft_tokens}) must equal num_steps + 1 ({num_steps + 1})"
            )
        self.num_steps = num_steps
        self.num_draft_tokens = num_draft_tokens
        self.cfg = DSparkScheduleConfig(
            gamma=num_steps,
            min_verify_len=1,
            max_verify_len=num_draft_tokens,
        )
        self.last_budget: Optional[int] = None
        self.last_verify_lens: Optional[torch.Tensor] = None

        table_path = envs.SGLANG_EAGLE_SPS_TABLE.get()
        if table_path:
            self.sps_table = load_sps_table_from_path(table_path)
            logger.info(
                "EAGLE adaptive verify: SPS table loaded from %s",
                table_path,
            )
        else:
            self.sps_table = None
            logger.warning(
                "EAGLE adaptive verify: SGLANG_EAGLE_SPS_TABLE is unset; the "
                "budget degenerates to verify-all (full-width ragged layout)."
            )

    def schedule_verify_lens(self, *, confidence: torch.Tensor) -> torch.Tensor:
        """confidence [bs, num_steps] (P_draft of each drafted token) →
        verify_lens [bs] int32 in [1, num_draft_tokens]."""
        if self.sps_table is None:
            return torch.full(
                (confidence.shape[0],),
                self.num_draft_tokens,
                dtype=torch.int32,
                device=confidence.device,
            )
        survival = torch.cumprod(confidence.to(torch.float32), dim=1)
        decision = compute_verify_token_budget(
            history_survival_probs=survival,
            sps_table=self.sps_table,
            cfg=self.cfg,
        )
        self.last_budget = decision.budget
        verify_lens = ScheduleVerifyLensTopk.execute(
            confidence=confidence, budget=decision.budget, cfg=self.cfg
        )
        self.last_verify_lens = verify_lens
        return verify_lens


def build_eagle_ragged_layout(
    *,
    verify_lens: torch.Tensor,
    model_runner,
    device: torch.device,
) -> Optional[RaggedVerifyLayout]:
    """verify_lens [bs] (device, int32) → RaggedVerifyLayout rounded up to the
    captured token grid. None when the total exceeds the largest captured tier
    (caller falls back to the uniform non-ragged verify path).
    """
    capture_num_tokens = ragged_capture_num_tokens(model_runner=model_runner)
    verify_lens_cpu = [int(v) for v in verify_lens.detach().to("cpu").tolist()]
    total = sum(verify_lens_cpu)
    if not verify_lens_cpu:
        return None
    if capture_num_tokens is not None:
        if total > capture_num_tokens[-1]:
            return None
        grid = capture_num_tokens
    else:
        grid = [total]
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=verify_lens_cpu, device=device, grid=grid
    )
