# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import cuda_graphs
from megatron.core.transformer.module import (
    Float16Module,
    GraphableMegatronModule,
    MegatronModule,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

DEVICE_CAPABILITY = None
if torch.cuda.is_available():
    DEVICE_CAPABILITY = torch.cuda.get_device_capability()


class DummyModule(MegatronModule):
    # def __init__(self, config: TransformerConfig, share_embeddings_and_output_weights=True):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.linear = torch.nn.modules.Linear(in_features=2, out_features=1)

    def forward(self, x):
        return self.linear(x)


class DummyGraphableModule(GraphableMegatronModule):

    def forward(self, hidden_states):
        return hidden_states


class StubCudaGraph:

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return args


class TestMegatronModule:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.megatron_module = DummyModule(config=transformer_config).cuda()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_megatron_module(self):
        megatron_module = self.megatron_module
        assert megatron_module
        assert megatron_module.config.hidden_size == 12
        assert megatron_module.config.ffn_hidden_size == 48
        assert megatron_module.linear.weight.dtype == torch.float32

        x = torch.ones((2, 2)).cuda()
        assert megatron_module(x).dtype == torch.float32

        # TODO: test bad configs actually fail
        # failed_module = megatron_module
        # failed_module.fp16 = True
        # failed_module.bf16 = True


class TestGraphableMegatronModule:

    @staticmethod
    def _make_module():
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl="transformer_engine",
        )
        return DummyGraphableModule(config=transformer_config)

    def test_te_cuda_graph_replay_records_successful_replay(self):
        module = self._make_module()
        graph_0 = StubCudaGraph()
        graph_1 = StubCudaGraph()
        module.cuda_graphs = [graph_0, graph_1]
        module.cuda_graph_expected_hidden_state_shapes = [(4, 2, 12), (6, 2, 12)]
        module.current_microbatch = 3

        hidden_states = torch.ones((6, 2, 12))
        output = module._te_cuda_graph_replay(hidden_states)

        assert output[0] is hidden_states
        assert graph_0.calls == []
        assert len(graph_1.calls) == 1
        assert graph_1.calls[0][1]["is_first_microbatch"] is False
        assert module.cuda_graph_replay_count == 1

    def test_te_cuda_graph_replay_rejects_hidden_state_shape_mismatch(self):
        module = self._make_module()
        graph = StubCudaGraph()
        module.cuda_graphs = [graph]
        module.cuda_graph_expected_hidden_state_shapes = [(4, 2, 12)]

        with pytest.raises(
            RuntimeError,
            match=(
                r"CUDA graph 0 in DummyGraphableModule expected hidden-state shape "
                r"\(4, 2, 12\), but microbatch 0 provided \(6, 2, 12\)"
            ),
        ):
            module._te_cuda_graph_replay(torch.ones((6, 2, 12)))

        assert graph.calls == []
        assert module.cuda_graph_replay_count == 0

    def test_te_cuda_graph_replay_requires_shape_record(self):
        module = self._make_module()
        graph = StubCudaGraph()
        module.cuda_graphs = [graph]

        with pytest.raises(RuntimeError, match="Missing expected hidden-state shape"):
            module._te_cuda_graph_replay(torch.ones((4, 2, 12)))

        assert graph.calls == []
        assert module.cuda_graph_replay_count == 0

    def test_te_cuda_graph_deletion_clears_replay_witnesses(self, monkeypatch):
        module = self._make_module()
        module.cuda_graphs = [StubCudaGraph()]
        module.cuda_graph_manual_hooks = [(object(), ())]
        module.cuda_graph_expected_hidden_state_shapes = [(4, 2, 12)]
        module.cuda_graph_replay_count = 3
        helper = cuda_graphs.TECudaGraphHelper.__new__(cuda_graphs.TECudaGraphHelper)
        helper.callables_per_chunk = [[module]]
        helper._graphs_created = True
        monkeypatch.setattr(cuda_graphs, "is_te_min_version", lambda _: False)
        monkeypatch.setattr(cuda_graphs, "log_on_each_pipeline_stage", lambda **_: None)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

        helper.delete_cuda_graphs()

        assert module.cuda_graphs == []
        assert module.cuda_graph_manual_hooks == []
        assert module.cuda_graph_expected_hidden_state_shapes == []
        assert module.cuda_graph_replay_count == 0
        assert helper._graphs_created is False


class TestFloat16Module:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.megatron_module = DummyModule(config=self.transformer_config).cuda()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_fp16_module(self):
        transformer_config = self.transformer_config
        megatron_module = self.megatron_module
        transformer_config.fp16 = True
        fp16_module = Float16Module(config=transformer_config, module=megatron_module)

        assert fp16_module
        assert fp16_module.config.hidden_size == 12
        assert fp16_module.config.ffn_hidden_size == 48
        assert fp16_module.module.linear.weight.dtype == torch.float16

        x = torch.ones((2, 2)).cuda()
        # inputs are converted to fp16 then outputs are converted to fp32
        assert fp16_module(x).dtype == torch.float32

    pytest.mark.skipif(
        not DEVICE_CAPABILITY or DEVICE_CAPABILITY[0] < 8,
        reason='bfloat16 is not supported on this device',
    )

    def test_bf16_module(self):
        transformer_config = self.transformer_config
        megatron_module = self.megatron_module
        transformer_config.bf16 = True
        bf16_module = Float16Module(config=transformer_config, module=megatron_module)

        assert bf16_module
        assert bf16_module.config.hidden_size == 12
        assert bf16_module.config.ffn_hidden_size == 48
        assert bf16_module.module.linear.weight.dtype == torch.bfloat16

        x = torch.ones((2, 2)).cuda()
        # inputs are converted to bf16 then outputs are converted to fp32
        assert bf16_module(x).dtype == torch.float32
