#!/usr/bin/env python3
"""prefill-moe M3 patch: fused SiLU*mul + fp8 group-quant (legacy-exact act) for
the deepep-normal contiguous prefill path + shared/dense experts.

Flag-gated by SGLANG_PMOE_FUSED_ACT_QUANT (default off).

1. jit/csrc/gemm/per_token_group_quant.cuh: new kLegacyExactAct template param —
   the fused-silu branch computes silu in fp32 with the same expression the
   act_and_mul_kernel uses (under --use_fast_math both compile to
   __expf + div.approx) and a single round of the fp32 product to bf16, making
   the fused kernel bit-exact with the legacy act_and_mul + quant pair.
2. ops/quantization/per_token_group_quant.py: thread legacy_exact_act through
   _jit_module / custom op / public API.
3. srt/layers/moe/moe_runner/deep_gemm.py (_run_contiguous_gemm): fused call
   replacing legacy silu_and_mul + row-major quant + tma_align (one kernel,
   emits TMA-aligned col-major scales directly).
4. srt/models/deepseek_v2.py (DeepseekV2MLP.forward): shared/dense experts —
   fused act+quant, down_proj consumes the (fp8, scale) tuple.
"""
import pathlib, sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")


def patch(rel, old, new, count=1):
    p = ROOT / rel
    s = p.read_text()
    assert s.count(old) == count, f"{rel}: pattern count {s.count(old)} != {count}\n---\n{old[:300]}"
    p.write_text(s.replace(old, new))
    print(f"patched {rel}")


# ---- 1. csrc: QuantTrait param ----
patch(
    "python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh",
    """template <
    typename InputType_,
    typename QuantType_,
    uint32_t kGroupSize_,
    bool kUe8m0_,
    bool kRowMajor_,
    bool kAligned_,
    bool kFuseSiluAndMul_>
struct QuantTrait {
  // rename
  using InputType = InputType_;
  using QuantType = QuantType_;
  static constexpr uint32_t kGroupSize = kGroupSize_;
  static constexpr bool kUe8m0 = kUe8m0_;
  static constexpr bool kRowMajor = kRowMajor_;
  static constexpr bool kAligned = kAligned_;
  static constexpr bool kFuseSiluAndMul = kFuseSiluAndMul_;""",
    """template <
    typename InputType_,
    typename QuantType_,
    uint32_t kGroupSize_,
    bool kUe8m0_,
    bool kRowMajor_,
    bool kAligned_,
    bool kFuseSiluAndMul_,
    bool kLegacyExactAct_ = false>
struct QuantTrait {
  // rename
  using InputType = InputType_;
  using QuantType = QuantType_;
  static constexpr uint32_t kGroupSize = kGroupSize_;
  static constexpr bool kUe8m0 = kUe8m0_;
  static constexpr bool kRowMajor = kRowMajor_;
  static constexpr bool kAligned = kAligned_;
  static constexpr bool kFuseSiluAndMul = kFuseSiluAndMul_;
  static constexpr bool kLegacyExactAct = kLegacyExactAct_;""",
)

# ---- 1b. csrc: fused branch legacy-exact expression ----
patch(
    "python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh",
    """    if constexpr (kFuseSiluAndMul) {
      in_vec_t up;
      up.load(token_in + group_offset + params.hidden_size, lane_id);
#pragma unroll
      for (uint32_t i = 0; i < kVecSize / 2; ++i) {
        const auto gate = cast<float2>(in[i]);
        const auto act = cast<T2>(float2{detail::silu(gate.x), detail::silu(gate.y)});
        in[i] = __hmul2(act, up[i]);
      }
    }""",
    """    if constexpr (kFuseSiluAndMul) {
      in_vec_t up;
      up.load(token_in + group_offset + params.hidden_size, lane_id);
#pragma unroll
      for (uint32_t i = 0; i < kVecSize / 2; ++i) {
        if constexpr (kLegacyExactAct) {
          // prefill-moe lane: bit-exact with act_and_mul_kernel — same fp32
          // expression (under --use_fast_math both sides compile expf to
          // __expf and '/' to div.approx) and a single round of the fp32
          // product to T, exactly like device::cast<T>(act * up_f32).
          const auto gate = cast<float2>(in[i]);
          const auto upf = cast<float2>(up[i]);
          in[i] = cast<T2>(float2{
              (gate.x / (1.0f + expf(-gate.x))) * upf.x,
              (gate.y / (1.0f + expf(-gate.y))) * upf.y});
        } else {
          const auto gate = cast<float2>(in[i]);
          const auto act = cast<T2>(float2{detail::silu(gate.x), detail::silu(gate.y)});
          in[i] = __hmul2(act, up[i]);
        }
      }
    }""",
)

# ---- 1c. csrc: launcher templates ----
for launcher in ("PerTokenGroupQuantFlatKernel", "PerTokenGroupQuantMaskedKernel"):
    patch(
        "python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh",
        f"""    bool kAligned,
    bool kFuseSiluAndMul,
    bool kUsePDL>
struct {launcher} {{
  using Trait = QuantTrait<InputType, QuantType, kGroupSize, kUe8m0, kRowMajor, kAligned, kFuseSiluAndMul>;""",
        f"""    bool kAligned,
    bool kFuseSiluAndMul,
    bool kUsePDL,
    bool kLegacyExactAct = false>
struct {launcher} {{
  using Trait = QuantTrait<InputType, QuantType, kGroupSize, kUe8m0, kRowMajor, kAligned, kFuseSiluAndMul, kLegacyExactAct>;""",
    )

# ---- 2. python API ----
patch(
    "python/sglang/kernels/ops/quantization/per_token_group_quant.py",
    """def _jit_module(
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    group_size: int,
    scale_ue8m0: bool,
    row_major: bool,
    aligned: bool,
    fuse_silu_and_mul: bool,
    masked_layout: bool,
    use_pdl: bool,
) -> Module:""",
    """def _jit_module(
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    group_size: int,
    scale_ue8m0: bool,
    row_major: bool,
    aligned: bool,
    fuse_silu_and_mul: bool,
    masked_layout: bool,
    use_pdl: bool,
    legacy_exact_act: bool = False,
) -> Module:""",
)

patch(
    "python/sglang/kernels/ops/quantization/per_token_group_quant.py",
    """    assert group_size in _SUPPORTED_GROUP_SIZES
    trait_args = make_cpp_args(
        in_dtype,
        out_dtype,
        group_size,
        scale_ue8m0,
        row_major,
        aligned,
        fuse_silu_and_mul,
        use_pdl,
    )""",
    """    assert group_size in _SUPPORTED_GROUP_SIZES
    assert not legacy_exact_act or fuse_silu_and_mul
    trait_args = make_cpp_args(
        in_dtype,
        out_dtype,
        group_size,
        scale_ue8m0,
        row_major,
        aligned,
        fuse_silu_and_mul,
        use_pdl,
    )
    if legacy_exact_act:
        # in-place: list.__add__ would return a plain list whose __str__ is a
        # python literal, breaking the cuda_wrappers template rendering
        trait_args += make_cpp_args(True)""",
)

patch(
    "python/sglang/kernels/ops/quantization/per_token_group_quant.py",
    """def _per_token_group_quant_custom_op(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    expected_m: Optional[int] = None,
) -> None:
    num_groups = output_q.shape[-1] // group_size
    row_major, aligned = _infer_scale_layout(output_s, scale_ue8m0, num_groups)
    module = _jit_module(
        input.dtype,
        output_q.dtype,
        int(group_size),
        bool(scale_ue8m0),
        row_major,
        aligned,
        bool(fuse_silu_and_mul),
        masked_m is not None,
        is_arch_support_pdl(),
    )""",
    """def _per_token_group_quant_custom_op(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    expected_m: Optional[int] = None,
    legacy_exact_act: bool = False,
) -> None:
    num_groups = output_q.shape[-1] // group_size
    row_major, aligned = _infer_scale_layout(output_s, scale_ue8m0, num_groups)
    module = _jit_module(
        input.dtype,
        output_q.dtype,
        int(group_size),
        bool(scale_ue8m0),
        row_major,
        aligned,
        bool(fuse_silu_and_mul),
        masked_m is not None,
        is_arch_support_pdl(),
        bool(legacy_exact_act),
    )""",
)

patch(
    "python/sglang/kernels/ops/quantization/per_token_group_quant.py",
    """    masked_m: Optional[torch.Tensor] = None,
    expected_m: Optional[int] = None,
    *,
    out_dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:""",
    """    masked_m: Optional[torch.Tensor] = None,
    expected_m: Optional[int] = None,
    *,
    out_dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    legacy_exact_act: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:""",
)

patch(
    "python/sglang/kernels/ops/quantization/per_token_group_quant.py",
    """    _per_token_group_quant_custom_op(
        input=input,
        output_q=output_q,
        output_s=output_s,
        group_size=group_size,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
        expected_m=expected_m,
    )
    return output_q, output_s""",
    """    _per_token_group_quant_custom_op(
        input=input,
        output_q=output_q,
        output_s=output_s,
        group_size=group_size,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
        expected_m=expected_m,
        legacy_exact_act=legacy_exact_act,
    )
    return output_q, output_s""",
)

# ---- 3. moe_runner wiring ----
# init the align flag before the activation chain
patch(
    "python/sglang/srt/layers/moe/moe_runner/deep_gemm.py",
    """        if self.config.activation == "situ":
            situ_beta = self.config.gemm1_alpha""",
    """        # prefill-moe lane: the fused branch below emits TMA-aligned scales
        # directly; every other branch leaves them row-major for the trailing
        # tma_align_input_scale.
        down_scale_needs_align = True
        if self.config.activation == "situ":
            situ_beta = self.config.gemm1_alpha""",
)

patch(
    "python/sglang/srt/layers/moe/moe_runner/deep_gemm.py",
    """_is_hip = is_hip()""",
    """_PMOE_FUSED_ACT_QUANT = get_bool_env_var("SGLANG_PMOE_FUSED_ACT_QUANT")

_is_hip = is_hip()""",
)

patch(
    "python/sglang/srt/layers/moe/moe_runner/deep_gemm.py",
    """        else:
            from sglang.kernels.ops.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )

            if self.swiglu_limit is not None:
                gateup_output = _apply_swiglu_limit(
                    gateup_output, swiglu_limit=self.swiglu_limit
                )

            if not _is_musa:
                down_input = torch.empty(
                    (all_tokens, N // 2),
                    device=gateup_output.device,
                    dtype=torch.bfloat16,
                )
                _legacy_silu_and_mul(gateup_output.view(-1, N), down_input)
            else:
                down_input = _silu_and_mul_musa(gateup_output.view(-1, N))
            del gateup_output

            down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
                down_input,
                scale_block_size,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            del down_input""",
    """        elif (
            _PMOE_FUSED_ACT_QUANT
            and self.config.activation == "silu"
            and self.swiglu_limit is None
            and not _is_musa
            and not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        ):
            # prefill-moe lane: one fused SiLU*mul + fp8 group-quant kernel
            # emitting the TMA-aligned col-major scale layout directly
            # (replaces legacy act_and_mul + row-major quant + tma_align;
            # bit-exact, see lane dir test_fused_act_quant.py).
            from sglang.kernels.ops.quantization.per_token_group_quant import (
                per_token_group_quant,
            )

            down_input_fp8, down_input_scale = per_token_group_quant(
                gateup_output,
                group_size=scale_block_size,
                scale_ue8m0=False,
                fuse_silu_and_mul=True,
                column_major_scales=True,
                legacy_exact_act=True,
            )
            del gateup_output
            down_scale_needs_align = False
        else:
            from sglang.kernels.ops.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )

            if self.swiglu_limit is not None:
                gateup_output = _apply_swiglu_limit(
                    gateup_output, swiglu_limit=self.swiglu_limit
                )

            if not _is_musa:
                down_input = torch.empty(
                    (all_tokens, N // 2),
                    device=gateup_output.device,
                    dtype=torch.bfloat16,
                )
                _legacy_silu_and_mul(gateup_output.view(-1, N), down_input)
            else:
                down_input = _silu_and_mul_musa(gateup_output.view(-1, N))
            del gateup_output

            down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
                down_input,
                scale_block_size,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            del down_input""",
)

# remove the now double tma_align for the non-fused path (the fused path already
# emits the aligned layout): make the trailing align conditional.
patch(
    "python/sglang/srt/layers/moe/moe_runner/deep_gemm.py",
    """        if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            down_input_scale = tma_align_input_scale(down_input_scale)

        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (down_input_fp8, down_input_scale),
            w2_weight_fp8,
            down_output,
            m_indices,""",
    """        if (
            deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES
            and down_scale_needs_align
        ):
            down_input_scale = tma_align_input_scale(down_input_scale)

        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (down_input_fp8, down_input_scale),
            w2_weight_fp8,
            down_output,
            m_indices,""",
)

# ---- 4. shared/dense experts (DeepseekV2MLP.forward) ----
patch(
    "python/sglang/srt/models/deepseek_v2.py",
    """        else:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x""",
    """        else:
            if (
                _PMOE_FUSED_ACT_QUANT
                and self.swiglu_limit is None
                and self.down_proj.weight.dtype == torch.uint8
                and hasattr(self.down_proj, "weight_scale_inv")
            ):
                # prefill-moe lane: fused SiLU*mul + fp8 group quant (bit-exact
                # with act_fn + the linear's inner quant; TMA-aligned scales).
                from sglang.kernels.ops.quantization.per_token_group_quant import (
                    per_token_group_quant,
                )

                x, _ = self.down_proj(
                    per_token_group_quant(
                        gate_up,
                        group_size=128,
                        scale_ue8m0=False,
                        fuse_silu_and_mul=True,
                        column_major_scales=True,
                        legacy_exact_act=True,
                    )
                )
                return x
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x""",
)

print("all patches applied")
