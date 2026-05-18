"""Tests for comprehensive weight mapping between PyTorch and Tesseract formats."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from tesseract_cuda.network.weight_mapper import (
    load_weights_to_model, extract_weights_from_model,
    _extract_fc_weight, _load_fc_weight,
    _extract_recursive,
)
from tesseract_cuda.network.layers import (
    LSTMLayer, FullyConnectedLayer, SeriesLayer, ParallelLayer,
    ConvolveLayer, MaxpoolLayer, ReconfigLayer, InputLayer,
)
from tesseract_cuda.formats.network_ser import (
    NetworkLayer, WeightMatrix, LSTMModel,
    deserialize_network, serialize_network,
    TYPE_NAME_TO_ID,
)
from tesseract_cuda.formats.tfile import TFileReader, TFileWriter


class TestExtractFCWeight:
    def test_shape(self):
        linear = nn.Linear(5, 3)
        wm = _extract_fc_weight(linear)
        assert wm.dim1 == 3
        assert wm.dim2 == 6  # 5 + 1 bias

    def test_values_match(self):
        linear = nn.Linear(4, 2)
        nn.init.constant_(linear.weight, 1.0)
        nn.init.constant_(linear.bias, 2.0)
        wm = _extract_fc_weight(linear)
        arr = wm.get_weights_np()
        assert arr[0, :-1] == pytest.approx([1.0, 1.0, 1.0, 1.0])
        assert arr[0, -1] == pytest.approx(2.0)


class TestLoadFCWeight:
    def test_roundtrip_preserves_weights(self):
        linear = nn.Linear(6, 4)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.normal_(linear.bias)

        wm = _extract_fc_weight(linear)
        linear2 = nn.Linear(6, 4)
        _load_fc_weight(wm, linear2)

        assert torch.allclose(linear.weight, linear2.weight, atol=1e-6)
        assert torch.allclose(linear.bias, linear2.bias, atol=1e-6)


class TestExtractSeriesModel:
    def test_extract_series(self):
        model = SeriesLayer([
            InputLayer(4, 4),
            FullyConnectedLayer(4, 8, "tanh"),
            FullyConnectedLayer(8, 3, "softmax"),
        ])
        nl = _extract_recursive(model, [])
        assert nl.type_name == "Series"
        assert len(nl.children) == 3
        assert nl.children[0].type_name == "Input"
        assert nl.children[1].type_name == "Tanh"
        assert nl.children[2].type_name == "Softmax"

    def test_extract_weights_count(self):
        model = SeriesLayer([
            InputLayer(4, 4),
            FullyConnectedLayer(4, 8, "tanh"),
        ])
        nl = _extract_recursive(model, [])
        fc_child = nl.children[1]
        assert len(fc_child.weights) == 1
        assert fc_child.weights[0].dim1 == 8
        assert fc_child.weights[0].dim2 == 5  # 4+1


class TestLoadWeightsToModel:
    def test_load_fc_weights(self):
        model = SeriesLayer([
            InputLayer(4, 4),
            FullyConnectedLayer(4, 3, "softmax"),
        ])

        wm = WeightMatrix(dim1=3, dim2=5, weights=list(np.random.randn(15)))
        nl = NetworkLayer(
            type_id=TYPE_NAME_TO_ID["Series"], type_name="Series",
            training=0, needs_backprop=False, network_flags=0,
            ni=4, no=3, num_weights=15, name="Series",
            children=[
                NetworkLayer(type_id=1, type_name="Input", training=0,
                             needs_backprop=False, network_flags=0,
                             ni=4, no=4, num_weights=0, name="Input"),
                NetworkLayer(type_id=22, type_name="Softmax", training=0,
                             needs_backprop=False, network_flags=0,
                             ni=4, no=3, num_weights=15, name="Softmax",
                             weights=[wm]),
            ],
        )
        load_weights_to_model(nl, model)

        fc_layer = model.layers[1]
        expected_w = np.array(wm.weights, dtype=np.float64).reshape(3, 5)
        expected_weight = expected_w[:, :-1].astype(np.float32)
        assert torch.allclose(fc_layer.fc.weight.data,
                              torch.as_tensor(expected_weight), atol=1e-6)


class TestFullRoundtrip:
    def test_model_export_import(self):
        from tesseract_cuda.network.model import TessLSTMModel

        spec = "[1,0,0,1Lfx16O1c5]"
        model = TessLSTMModel.from_spec(spec, num_classes=5)
        model.null_char = 0

        original_w = model.network.layers[1].cell.gate_ci.weight.data.clone()

        export_bytes = model.export_lstm_component(
            training_iteration=42,
            sample_iteration=84,
        )

        r = TFileReader(export_bytes)
        from tesseract_cuda.formats.network_ser import deserialize_lstm_component
        parsed = deserialize_lstm_component(r)

        assert parsed.training_iteration == 42
        assert parsed.sample_iteration == 84

        model2 = TessLSTMModel.from_spec(spec, num_classes=5)
        model2.null_char = 0
        load_weights_to_model(parsed.network, model2.network)

        loaded_w = model2.network.layers[1].cell.gate_ci.weight.data
        assert torch.allclose(original_w, loaded_w, atol=1e-5)
