# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MoE expert-parallel layer backed by the megakernel (github.com/doublewordai/megakernel).

One kernel launch per layer performs the dispatch, the fused GEMM1 + SwiGLU, GEMM2 and the combine
over CXI, so no token dispatcher and no MoE runner are involved.  One transport and one kernel object
serve every MoE layer of the process.  Supports GLM-5.2 FP8 block-quantized experts at EP16 on GH200
nodes; ``Megakernel`` checks the geometry.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)

_kernel = None


def _get_kernel(hidden_size: int, top_k: int, num_local_experts: int, intermediate_size: int):
    """One CXI transport and one kernel object per process, shared by every MoE layer."""
    global _kernel
    if _kernel is None:
        from megakernel import Megakernel, Transport

        tp = get_tp_group()

        def all_gather(obj):
            out = [None] * tp.world_size
            dist.all_gather_object(out, obj, group=tp.cpu_group)
            return out

        transport = Transport(
            world_size=tp.world_size,
            rank=tp.rank_in_group,
            local_rank=torch.cuda.current_device(),
            hidden_size=hidden_size,
            top_k=top_k,
            num_local_experts=num_local_experts,
            max_tokens_per_rank=envs.SGLANG_MEGAKERNEL_NUM_MAX_TOKENS_PER_RANK.get(),
            all_gather=all_gather,
        )
        _kernel = Megakernel(transport, intermediate_size)
    return _kernel


class MegakernelMoE(FusedMoE):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )
        assert num_fused_shared_experts == 0, "the megakernel routes only the routed experts"
        assert activation == "silu", "the megakernel fuses SwiGLU"
        self._kernel = None

    def prepare_megakernel_weights(self) -> None:
        """Called from Fp8MoEMethod.process_weights_after_loading: interleave gate/up rows in
        place (see megakernel.weights) and bring up the shared transport and kernel."""
        from megakernel import interleave_gate_up_inplace

        interleave_gate_up_inplace(self.w13_weight.data, self.w13_weight_scale_inv.data)
        w2_scale = self.w2_weight_scale_inv.data
        if w2_scale.dtype != torch.float32 or not w2_scale.is_contiguous():
            self.w2_weight_scale_inv.data = w2_scale.contiguous().float()
        self._kernel = _get_kernel(
            self.hidden_size, self.top_k, self.num_local_experts, self.intermediate_size_per_partition
        )

    def forward_impl(self, hidden_states: torch.Tensor, topk_output: TopKOutput, **kwargs):
        assert self._kernel is not None, "prepare_megakernel_weights() must run after weight loading"
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device
        if num_tokens > 0:
            x_q, x_scale = sglang_per_token_group_quant_fp8(hidden_states.contiguous(), 128)
            topk_ids = topk_output.topk_ids.to(torch.int32).contiguous()
            topk_weights = topk_output.topk_weights.to(torch.float32).contiguous()
        else:
            # An idle data-parallel rank still owns experts: it runs the layer for the other
            # ranks' tokens with valid (unused) pointers.
            x_q = torch.empty(1, self.hidden_size, dtype=torch.float8_e4m3fn, device=device)
            x_scale = torch.empty(1, self.hidden_size // 128, dtype=torch.float32, device=device)
            topk_ids = torch.full((1, self.top_k), -1, dtype=torch.int32, device=device)
            topk_weights = torch.zeros(1, self.top_k, dtype=torch.float32, device=device)
        return self._kernel.forward(
            x_q,
            x_scale,
            topk_ids,
            topk_weights,
            num_tokens,
            self.w13_weight.data,
            self.w13_weight_scale_inv.data,
            self.w2_weight.data,
            self.w2_weight_scale_inv.data,
        )
