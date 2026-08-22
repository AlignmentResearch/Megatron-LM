# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core.activations import squared_relu
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _LinearWithLoRA(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.base = nn.Linear(input_size, output_size, bias=False)
        self.lora_a = nn.Linear(input_size, 2, bias=False)
        self.lora_b = nn.Linear(2, output_size, bias=False)
        self.base.weight.requires_grad_(False)
        self.forward_calls = 0

    def forward(self, inputs: torch.Tensor, _tokens_per_expert: list[int]):
        self.forward_calls += 1
        return self.base(inputs) + self.lora_b(self.lora_a(inputs)), None


class _TinyGroupedMLP(TEGroupedMLP):
    def __init__(self, recompute: bool):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(
            fp8=None,
            fp4=None,
            moe_apply_probs_on_input=False,
            use_te_activation_func=False,
            bias_activation_fusion=False,
            activation_func=squared_relu,
            use_fused_weighted_squared_relu=False,
            gated_linear_unit=False,
        )
        self.linear_fc1 = _LinearWithLoRA(3, 8)
        self.linear_fc2 = _LinearWithLoRA(8, 3)
        self.expert_fc1_activation_recompute = recompute
        self.activation_recompute = False
        self.offload_expert_fc1 = False
        self.offload_moe_act = False


def _config(**kwargs) -> TransformerConfig:
    config_kwargs = {
        "num_layers": 1,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_moe_experts": 2,
        "moe_ffn_hidden_size": 32,
        "recompute_granularity": "selective",
        "recompute_modules": ["expert_fc1_act"],
    }
    config_kwargs.update(kwargs)
    return TransformerConfig(**config_kwargs)


def test_expert_fc1_activation_recompute_requires_grouped_gemm():
    with pytest.raises(ValueError, match="requires moe_grouped_gemm"):
        _config(moe_grouped_gemm=False)


def test_expert_fc1_activation_recompute_requires_transformer_engine():
    with pytest.raises(ValueError, match="requires transformer_engine"):
        _config(moe_grouped_gemm=True, transformer_impl="local")


@pytest.mark.parametrize("conflict", ["moe", "moe_act"])
def test_expert_fc1_activation_recompute_rejects_nested_moe_recompute(conflict: str):
    with pytest.raises(ValueError, match="cannot be combined"):
        _config(
            moe_grouped_gemm=True,
            transformer_impl="transformer_engine",
            recompute_modules=["expert_fc1_act", conflict],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.internal
def test_expert_fc1_activation_recompute_preserves_lora_and_input_gradients():
    Utils.initialize_model_parallel(1, 1)
    try:
        torch.manual_seed(7)
        reference_model = _TinyGroupedMLP(recompute=False).cuda()
        recompute_model = deepcopy(reference_model)
        recompute_model.expert_fc1_activation_recompute = True

        reference_input = torch.randn(4, 3, device="cuda", requires_grad=True)
        recompute_input = reference_input.detach().clone().requires_grad_(True)
        tokens_per_expert = torch.tensor([2, 2], device="cuda", dtype=torch.int64)
        reference_probs = torch.rand(4, device="cuda", requires_grad=True)
        recompute_probs = reference_probs.detach().clone().requires_grad_(True)

        reference_output, _ = reference_model(
            reference_input, tokens_per_expert, reference_probs
        )
        recompute_output, _ = recompute_model(
            recompute_input, tokens_per_expert, recompute_probs
        )
        torch.testing.assert_close(recompute_output, reference_output)

        reference_output.square().sum().backward()
        recompute_output.square().sum().backward()

        torch.testing.assert_close(recompute_input.grad, reference_input.grad)
        torch.testing.assert_close(recompute_probs.grad, reference_probs.grad)
        for reference_name, reference_parameter in reference_model.named_parameters():
            recompute_parameter = recompute_model.get_parameter(reference_name)
            assert (
                recompute_parameter.requires_grad == reference_parameter.requires_grad
            )
            if reference_parameter.requires_grad:
                torch.testing.assert_close(
                    recompute_parameter.grad, reference_parameter.grad
                )
            else:
                assert recompute_parameter.grad is None
                assert reference_parameter.grad is None

        assert reference_model.linear_fc1.forward_calls == 1
        assert reference_model.linear_fc2.forward_calls == 1
        assert recompute_model.linear_fc1.forward_calls == 2
        assert recompute_model.linear_fc2.forward_calls == 1
    finally:
        Utils.destroy_model_parallel()
