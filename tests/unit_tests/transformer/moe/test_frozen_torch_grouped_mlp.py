from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.extensions.transformer_engine import _TorchGroupedFusedLoRA
from megatron.core.extensions.transformer_engine import _run_torch_grouped_mm
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

    def test_reprepares_after_middle_expert_weight_replacement(self):
        model = self._model(_config(backend="torch"))
        linear = model.experts.linear_fc1
        for parameter in linear.parameters():
            parameter.requires_grad = False
        linear.prepare_torch_grouped_mm()
        replacement = torch.randn_like(linear.weight1)

        linear.weight1.data = replacement

        assert not linear.torch_grouped_mm_prepared
        linear.prepare_torch_grouped_mm()
        assert linear.torch_grouped_mm_prepared
        torch.testing.assert_close(linear.weight1, replacement)

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


def test_fused_expert_lora_matches_shared_adapter_and_gradients_on_cpu():
    torch.manual_seed(7)
    group_sizes = [2, 0, 3]
    offsets = torch.tensor(group_sizes, dtype=torch.int32).cumsum(0)
    base_weight = torch.randn(3, 7, 5, dtype=torch.double)
    reference_a = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
    reference_b = torch.randn(7, 3, dtype=torch.double, requires_grad=True)
    reference_input = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
    fused_input = reference_input.detach().clone().requires_grad_(True)
    fused_a = reference_a.detach().clone().requires_grad_(True)
    fused_b = reference_b.detach().clone().requires_grad_(True)
    scale = 2.5
    base_output = _run_torch_grouped_mm(
        reference_input, base_weight.transpose(1, 2), offsets
    )
    reference = base_output + scale * reference_input.matmul(
        reference_a.transpose(0, 1)
    ).matmul(reference_b.transpose(0, 1))
    augmented_weight = base_weight.new_empty((3, 10, 5))
    augmented_weight[:, :7, :].copy_(base_weight)
    augmented_weight[:, 7:, :].copy_(fused_a)
    counters = {
        "forward_calls": 0,
        "backward_calls": 0,
        "base_only_forward_calls": 0,
    }
    fused = _TorchGroupedFusedLoRA.apply(
        fused_input,
        augmented_weight,
        fused_a,
        fused_b,
        offsets,
        7,
        scale,
        counters,
    )
    grad_output = torch.randn_like(reference)
    reference_grads = torch.autograd.grad(
        reference, (reference_input, reference_a, reference_b), grad_output
    )
    fused_grads = torch.autograd.grad(
        fused, (fused_input, fused_a, fused_b), grad_output
    )

    torch.testing.assert_close(fused, reference)
    for actual, expected in zip(fused_grads, reference_grads):
        torch.testing.assert_close(actual, expected)
    assert counters == {
        "forward_calls": 1,
        "backward_calls": 1,
        "base_only_forward_calls": 0,
    }


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not hasattr(torch, "_grouped_mm")
    or torch.cuda.get_device_capability()[0] < 10,
    reason="production fused expert LoRA parity requires a Blackwell CUDA device",
)
@pytest.mark.parametrize(
    ("input_features", "output_features"),
    [(1024, 2688), (2688, 1024)],
    ids=["super-fc1", "super-fc2"],
)
def test_fused_expert_lora_super_shapes_over_repeated_updates(input_features, output_features):
    generator = torch.Generator(device="cuda").manual_seed(20260822)
    group_sizes = [0 if index % 17 == 0 else 1 + (index * 13) % 4 for index in range(128)]
    offsets = torch.tensor(group_sizes, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    rows = sum(group_sizes)
    rank = 32
    scale = 64.0 / rank
    base_weight = torch.randn(
        (len(group_sizes), output_features, input_features),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    reference_a = torch.randn(
        (rank, input_features),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    reference_b = torch.randn(
        (output_features, rank),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    fused_a = reference_a.detach().clone().requires_grad_(True)
    fused_b = reference_b.detach().clone().requires_grad_(True)
    augmented_weight = base_weight.new_empty(
        (len(group_sizes), output_features + rank, input_features)
    )
    augmented_weight[:, :output_features, :].copy_(base_weight)
    augmented_weight[:, output_features:, :].copy_(fused_a)
    reference_optimizer = torch.optim.SGD([reference_a, reference_b], lr=0.01)
    fused_optimizer = torch.optim.SGD([fused_a, fused_b], lr=0.01)
    counters = {"forward_calls": 0, "backward_calls": 0, "base_only_forward_calls": 0}

    for _ in range(3):
        reference_optimizer.zero_grad()
        fused_optimizer.zero_grad()
        reference_inputs = []
        fused_inputs = []
        reference_outputs = []
        fused_outputs = []
        for _ in range(2):
            reference_input = torch.randn(
                (rows, input_features),
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
                requires_grad=True,
            )
            fused_input = reference_input.detach().clone().requires_grad_(True)
            base_output = torch._grouped_mm(
                reference_input, base_weight.transpose(1, 2), offs=offsets
            )
            reference_output = base_output + scale * reference_input.matmul(
                reference_a.transpose(0, 1)
            ).matmul(reference_b.transpose(0, 1))
            fused_output = _TorchGroupedFusedLoRA.apply(
                fused_input,
                augmented_weight,
                fused_a,
                fused_b,
                offsets,
                output_features,
                scale,
                counters,
            )
            torch.testing.assert_close(fused_output, reference_output, rtol=0.02, atol=0.02)
            reference_inputs.append(reference_input)
            fused_inputs.append(fused_input)
            reference_outputs.append(reference_output)
            fused_outputs.append(fused_output)

        grad_output = torch.randn(
            reference_outputs[0].shape, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        torch.autograd.backward(reference_outputs, [grad_output, grad_output])
        torch.autograd.backward(fused_outputs, [grad_output, grad_output])
        for fused_input, reference_input in zip(fused_inputs, reference_inputs):
            torch.testing.assert_close(fused_input.grad, reference_input.grad, rtol=0.02, atol=1.0)
        torch.testing.assert_close(fused_a.grad, reference_a.grad, rtol=0.02, atol=0.02)
        torch.testing.assert_close(fused_b.grad, reference_b.grad, rtol=0.02, atol=0.02)
        reference_optimizer.step()
        fused_optimizer.step()
        torch.testing.assert_close(fused_a, reference_a, rtol=0.02, atol=0.02)
        torch.testing.assert_close(fused_b, reference_b, rtol=0.02, atol=0.02)

    assert any(size == 0 for size in group_sizes)
    assert counters == {"forward_calls": 6, "backward_calls": 6, "base_only_forward_calls": 0}
