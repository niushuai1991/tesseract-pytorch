"""Tests for network serialization and weight mapping."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from tesseract_cuda.formats.tfile import TFileReader, TFileWriter
from tesseract_cuda.formats.network_ser import (
    WeightMatrix, NetworkLayer, LSTMModel,
    deserialize_network, serialize_network,
    deserialize_lstm_component, serialize_lstm_component,
    count_weights, WM_DOUBLE,
)
from tesseract_cuda.network.weight_mapper import (
    _extract_fc_weight, _load_fc_weight,
)


class TestWeightMatrix:
    def test_serialize_deserialize_roundtrip(self):
        wm = WeightMatrix(dim1=3, dim2=4, weights=[
            1.0, 2.0, 3.0, 0.1,
            4.0, 5.0, 6.0, 0.2,
            7.0, 8.0, 9.0, 0.3,
        ])

        w = TFileWriter()
        wm.serialize(w, training=False)
        data = w.get_bytes()

        r = TFileReader(data)
        wm2 = WeightMatrix.deserialize(r, training=False)

        assert wm2.dim1 == 3
        assert wm2.dim2 == 4
        assert len(wm2.weights) == 12
        assert wm2.weights == pytest.approx(wm.weights, abs=1e-10)
        assert wm2.int_mode == False

    def test_weight_matrix_properties(self):
        wm = WeightMatrix(dim1=5, dim2=10, weights=list(range(50)))
        assert wm.num_outputs == 5
        assert wm.num_inputs == 9  # last column is bias

    def test_get_weights_np(self):
        wm = WeightMatrix(dim1=2, dim2=3, weights=[1, 2, 3, 4, 5, 6])
        arr = wm.get_weights_np()
        assert arr.shape == (2, 3)
        assert arr[0, 2] == 3.0  # bias for first output
        assert arr[1, 0] == 4.0

    def test_serialize_with_adam(self):
        wm = WeightMatrix(dim1=2, dim2=2, weights=[1.0, 0.5, 2.0, 0.3], use_adam=True)

        w = TFileWriter()
        wm.serialize(w, training=True)  # training=True writes updates + adam_sq_sum
        data = w.get_bytes()

        r = TFileReader(data)
        wm2 = WeightMatrix.deserialize(r, training=True)
        assert wm2.use_adam == True


class TestNetworkLayerSerialization:
    def _make_fc_layer(self, ni=4, no=2) -> NetworkLayer:
        """Create a simple fully connected layer."""
        weights = list(np.random.randn(no, ni + 1).astype(np.float64).flatten())
        return NetworkLayer(
            type_id=22, type_name="Softmax", training=0,
            needs_backprop=False, network_flags=0,
            ni=ni, no=no, num_weights=no * (ni + 1),
            name="Output1", weights=[WeightMatrix(dim1=no, dim2=ni+1, weights=weights)],
        )

    def _make_lstm_layer(self, ni=4, ns=3) -> NetworkLayer:
        """Create a simple LSTM layer."""
        na = ni + ns
        weights = []
        for _ in range(4):  # CI, GI, GF1, GO
            w = list(np.random.randn(ns, na + 1).astype(np.float64).flatten())
            weights.append(WeightMatrix(dim1=ns, dim2=na + 1, weights=w))

        return NetworkLayer(
            type_id=14, type_name="LSTM", training=0,
            needs_backprop=False, network_flags=0,
            ni=ni, no=ns, num_weights=4 * ns * (na + 1),
            name="LSTM1", weights=weights, na=na, ns=ns,
        )

    def test_fc_roundtrip(self):
        layer = self._make_fc_layer()

        w = TFileWriter()
        serialize_network(layer, w, training=False)
        r = TFileReader(w.get_bytes())
        layer2 = deserialize_network(r)

        assert layer2.type_name == "Softmax"
        assert layer2.ni == 4
        assert layer2.no == 2
        assert len(layer2.weights) == 1
        assert layer2.weights[0].dim1 == 2
        assert layer2.weights[0].dim2 == 5
        assert layer2.weights[0].weights == pytest.approx(layer.weights[0].weights, abs=1e-10)

    def test_lstm_roundtrip(self):
        layer = self._make_lstm_layer()

        w = TFileWriter()
        serialize_network(layer, w, training=False)
        r = TFileReader(w.get_bytes())
        layer2 = deserialize_network(r)

        assert layer2.type_name == "LSTM"
        assert layer2.ni == 4
        assert layer2.no == 3
        assert layer2.na == 7
        assert len(layer2.weights) == 4
        for i in range(4):
            assert layer2.weights[i].weights == pytest.approx(
                layer.weights[i].weights, abs=1e-10)

    def test_series_roundtrip(self):
        fc1 = self._make_fc_layer(4, 8)
        fc2 = self._make_fc_layer(8, 2)

        series = NetworkLayer(
            type_id=9, type_name="Series", training=0,
            needs_backprop=False, network_flags=0,
            ni=4, no=2, num_weights=8*5 + 2*9,
            name="Series", children=[fc1, fc2],
        )

        w = TFileWriter()
        serialize_network(series, w, training=False)
        r = TFileReader(w.get_bytes())
        series2 = deserialize_network(r)

        assert series2.type_name == "Series"
        assert len(series2.children) == 2
        assert series2.children[0].type_name == "Softmax"
        assert series2.children[1].type_name == "Softmax"
        assert series2.children[0].ni == 4
        assert series2.children[1].ni == 8

    def test_convolve_roundtrip(self):
        layer = NetworkLayer(
            type_id=2, type_name="Convolve", training=0,
            needs_backprop=False, network_flags=0,
            ni=16, no=32, num_weights=0,
            name="Conv1", half_x=1, half_y=1,
        )

        w = TFileWriter()
        serialize_network(layer, w, training=False)
        r = TFileReader(w.get_bytes())
        layer2 = deserialize_network(r)

        assert layer2.type_name == "Convolve"
        assert layer2.half_x == 1
        assert layer2.half_y == 1

    def test_maxpool_roundtrip(self):
        layer = NetworkLayer(
            type_id=3, type_name="Maxpool", training=0,
            needs_backprop=False, network_flags=0,
            ni=32, no=288, num_weights=0,
            name="Maxpool1", x_scale=3, y_scale=3,
        )

        w = TFileWriter()
        serialize_network(layer, w, training=False)
        r = TFileReader(w.get_bytes())
        layer2 = deserialize_network(r)

        assert layer2.type_name == "Maxpool"
        assert layer2.x_scale == 3
        assert layer2.y_scale == 3

    def test_lstm_component_roundtrip(self):
        lstm = self._make_lstm_layer(4, 3)
        model = LSTMModel(
            network=lstm,
            network_str="[1,0,0,1Lfx3]O1c10",
            training_flags=192,
            training_iteration=5000,
            sample_iteration=6000,
            null_char=0,
            adam_beta=0.999,
            learning_rate=0.001,
            momentum=0.5,
        )

        w = TFileWriter()
        serialize_lstm_component(model, w, training=False)
        r = TFileReader(w.get_bytes())
        model2 = deserialize_lstm_component(r)

        assert model2.network_str == "[1,0,0,1Lfx3]O1c10"
        assert model2.training_flags == 192
        assert model2.training_iteration == 5000
        assert model2.sample_iteration == 6000
        assert model2.null_char == 0
        assert model2.learning_rate == pytest.approx(0.001)
        assert model2.momentum == pytest.approx(0.5)


class TestWeightMapper:
    def test_fc_weight_roundtrip(self):
        """Test nn.Linear <-> WeightMatrix conversion."""
        linear = nn.Linear(5, 3)
        wm = _extract_fc_weight(linear)

        assert wm.dim1 == 3
        assert wm.dim2 == 6  # 5 inputs + 1 bias

        # Create new linear and load
        linear2 = nn.Linear(5, 3)
        _load_fc_weight(wm, linear2)

        assert torch.allclose(linear.weight, linear2.weight)
        assert torch.allclose(linear.bias, linear2.bias)

    def test_weight_preservation(self):
        """Verify weight values survive round-trip (float32->float64->float32)."""
        linear = nn.Linear(3, 2)
        nn.init.constant_(linear.weight, 0.5)
        nn.init.constant_(linear.bias, 0.1)

        wm = _extract_fc_weight(linear)
        linear2 = nn.Linear(3, 2)
        _load_fc_weight(wm, linear2)

        # float32->float64->float32 has small precision loss
        assert torch.allclose(linear2.weight, torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]), atol=1e-4)
        assert torch.allclose(linear2.bias, torch.tensor([0.1, 0.1]), atol=1e-4)


class TestCountWeights:
    def test_count_simple_fc(self):
        wm = WeightMatrix(dim1=2, dim2=3, weights=[1.0]*6)
        layer = NetworkLayer(
            type_id=22, type_name="Softmax", training=0,
            needs_backprop=False, network_flags=0,
            ni=2, no=2, num_weights=6, name="FC",
            weights=[wm],
        )
        assert count_weights(layer) == 6

    def test_count_series(self):
        wm1 = WeightMatrix(dim1=2, dim2=3, weights=[1.0]*6)
        wm2 = WeightMatrix(dim1=4, dim2=5, weights=[1.0]*20)
        fc1 = NetworkLayer(type_id=22, type_name="Softmax", training=0,
                           needs_backprop=False, network_flags=0,
                           ni=2, no=2, num_weights=6, name="FC1", weights=[wm1])
        fc2 = NetworkLayer(type_id=22, type_name="Softmax", training=0,
                           needs_backprop=False, network_flags=0,
                           ni=4, no=4, num_weights=20, name="FC2", weights=[wm2])
        series = NetworkLayer(type_id=9, type_name="Series", training=0,
                              needs_backprop=False, network_flags=0,
                              ni=2, no=4, num_weights=26, name="S",
                              children=[fc1, fc2])
        assert count_weights(series) == 26
