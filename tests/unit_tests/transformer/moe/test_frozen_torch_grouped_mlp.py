from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


def _config(*, backend: str = "transformer_engine") -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_grouped_gemm=True,
        moe_expert_gemm_backend=backend,
        add_bias_linear=False,
        gated_linear_unit=False,
        activation_func=F.gelu,
        bias_activation_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        moe_router_load_balancing_type="sinkhorn",
        moe_router_topk=1,
    )


@pytest.mark.parametrize(
    "override,error",
    [
        ({"num_moe_experts": None}, "requires num_moe_experts"),
        ({"moe_grouped_gemm": False}, "requires moe_grouped_gemm=True"),
        ({"bf16": False, "params_dtype": torch.float32}, "requires BF16 parameters"),
        ({"add_bias_linear": True}, "does not support expert bias"),
    ],
)
def test_torch_grouped_expert_gemm_config_validation(override, error):
    kwargs = {
        "num_layers": 1,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_moe_experts": 2,
        "moe_grouped_gemm": True,
        "moe_expert_gemm_backend": "torch",
        "add_bias_linear": False,
        "bf16": True,
        "params_dtype": torch.bfloat16,
    }
    kwargs.update(override)

    with pytest.raises(ValueError, match=error):
        TransformerConfig(**kwargs)


@pytest.mark.skipif(
    not is_te_min_version("1.9.0.dev0")
    or not torch.cuda.is_available()
    or not hasattr(torch, "_grouped_mm")
    or torch.cuda.get_device_capability()[0] < 10,
    reason="torch grouped expert GEMM requires TE and a compatible CUDA device",
)
class TestFrozenTorchGroupedMLP:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @staticmethod
    def _model(config: TransformerConfig) -> MoELayer:
        model = MoELayer(
            config,
            get_gpt_layer_with_transformer_engine_submodules(
                config.num_moe_experts, moe_grouped_gemm=True
            ).mlp.submodules,
        )
        return Float16Module(config, model).module.cuda()

    def test_forward_input_gradient_and_checkpoint_parity(self):
        baseline = self._model(_config())
        torch_model = self._model(_config(backend="torch"))
        torch_model.load_state_dict(baseline.state_dict())
        torch_linears = (
            torch_model.experts.linear_fc1,
            torch_model.experts.linear_fc2,
        )
        for linear in torch_linears:
            for parameter in linear.parameters():
                parameter.requires_grad = False
        state_keys_before = set(torch_model.state_dict())

        baseline_input = torch.rand(
            (32, 2, 16), dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        torch_input = baseline_input.detach().clone().requires_grad_(True)
        baseline_output, _ = baseline(baseline_input)
        torch_output, _ = torch_model(torch_input)
        grad_output = torch.randn_like(baseline_output)
        baseline_output.backward(grad_output)
        torch_output.backward(grad_output)

        torch.testing.assert_close(torch_output, baseline_output, rtol=0.02, atol=0.02)
        torch.testing.assert_close(
            torch_input.grad, baseline_input.grad, rtol=0.02, atol=0.02
        )
        assert set(torch_model.state_dict()) == state_keys_before
        assert not any("_torch_grouped_weight" in key for key in state_keys_before)
        for linear in torch_linears:
            assert linear.torch_grouped_mm_prepared
            assert linear._buffers[
                "_torch_grouped_weight"
            ].untyped_storage().data_ptr() == (
                linear.weight0.untyped_storage().data_ptr()
            )

    def test_no_tokens(self):
        config = deepcopy(_config(backend="torch"))
        model = self._model(config)
        for parameter in model.experts.parameters():
            parameter.requires_grad = False
        hidden_states = torch.empty(
            (0, 16), dtype=torch.bfloat16, device="cuda", requires_grad=True
        )

        output, _ = model.experts(
            hidden_states,
            tokens_per_expert=torch.zeros(4, dtype=torch.int32, device="cuda"),
            permuted_probs=torch.empty(0, dtype=torch.float32, device="cuda"),
        )
        output.sum().backward()

        assert output.shape == (0, 16)
        assert hidden_states.grad is not None

    def test_zero_token_groups(self):
        baseline = self._model(_config())
        torch_model = self._model(_config(backend="torch"))
        torch_model.load_state_dict(baseline.state_dict())
        for parameter in torch_model.experts.parameters():
            parameter.requires_grad = False

        baseline_input = torch.rand(
            (5, 16), dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        torch_input = baseline_input.detach().clone().requires_grad_(True)
        tokens_per_expert = torch.tensor([2, 0, 3, 0], device="cuda")
        permuted_probs = torch.rand(5, dtype=torch.float32, device="cuda")
        baseline_output, _ = baseline.experts(
            baseline_input, tokens_per_expert, permuted_probs
        )
        torch_output, _ = torch_model.experts(
            torch_input, tokens_per_expert, permuted_probs
        )
        grad_output = torch.randn_like(baseline_output)
        baseline_output.backward(grad_output)
        torch_output.backward(grad_output)

        torch.testing.assert_close(torch_output, baseline_output, rtol=0.02, atol=0.02)
        torch.testing.assert_close(
            torch_input.grad, baseline_input.grad, rtol=0.02, atol=0.02
        )
