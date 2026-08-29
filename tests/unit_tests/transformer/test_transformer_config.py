# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.extensions import transformer_engine as transformer_engine_extension
from megatron.core.transformer.transformer_config import TransformerConfig


def _make_overlap_config(mtp_num_layers: int | None) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=True,
        bf16=True,
        mtp_num_layers=mtp_num_layers,
    )


@pytest.mark.parametrize("mtp_num_layers", [None, 0, 1])
def test_ep_a2a_overlap_accepts_supported_mtp_layer_counts(mtp_num_layers: int | None):
    config = _make_overlap_config(mtp_num_layers)

    assert config.mtp_num_layers == mtp_num_layers


@pytest.mark.parametrize("mtp_num_layers", [-1, 2])
def test_ep_a2a_overlap_rejects_unsupported_mtp_layer_counts(mtp_num_layers: int):
    with pytest.raises(AssertionError, match="MTP supports at most one layer"):
        _make_overlap_config(mtp_num_layers)


def _make_torch_grouped_expert_config(**overrides) -> TransformerConfig:
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
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        ({"delay_wgrad_compute": True}, "incompatible with delay_wgrad_compute=True"),
        (
            {"overlap_dispatch_backward_with_experts_wgrad": True},
            "incompatible with overlap_dispatch_backward_with_experts_wgrad=True",
        ),
        (
            {"transformer_impl": "local"},
            "requires transformer_impl='transformer_engine' to construct TEGroupedMLP",
        ),
    ],
)
def test_torch_grouped_expert_backend_rejects_incompatible_modes(overrides, error_match):
    with pytest.raises(ValueError, match=error_match):
        _make_torch_grouped_expert_config(**overrides)


def test_torch_grouped_expert_backend_rejects_sequential_mlp_fallback(monkeypatch):
    monkeypatch.setattr(transformer_engine_extension, "TEColumnParallelGroupedLinear", None)

    with pytest.raises(ValueError, match="requires Transformer Engine >= 1.9.0.dev0"):
        _make_torch_grouped_expert_config()


def test_torch_grouped_expert_backend_accepts_te_grouped_linear(monkeypatch):
    monkeypatch.setattr(transformer_engine_extension, "TEColumnParallelGroupedLinear", object)

    config = _make_torch_grouped_expert_config()

    assert config.moe_expert_gemm_backend == "torch"
