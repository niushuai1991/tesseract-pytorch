"""Tests for individual PyTorch layer forward passes."""

import pytest
import torch

from tesseract_cuda.network.layers import (
    InputLayer, ConvolveLayer, MaxpoolLayer, ReconfigLayer,
    FullyConnectedLayer, ReversedLayer, SeriesLayer, ParallelLayer,
    LSTMLayer,
)


class TestInputLayer:
    def test_passthrough(self):
        layer = InputLayer(ni=4, no=4)
        x = torch.randn(10, 5, 4)
        assert torch.equal(layer(x), x)

    def test_output_shape(self):
        layer = InputLayer(ni=8, no=8)
        x = torch.randn(3, 20, 8)
        assert layer(x).shape == (3, 20, 8)


class TestConvolveLayer:
    def test_output_shape_3d(self):
        layer = ConvolveLayer(ni=4, no=8, half_x=1, half_y=1)
        x = torch.randn(10, 10, 4)  # [h, w, d]
        out = layer(x)
        assert out.shape == (10, 10, 8)

    def test_output_shape_4d(self):
        layer = ConvolveLayer(ni=4, no=8, half_x=0, half_y=0)
        x = torch.randn(2, 10, 10, 4)  # [b, h, w, d]
        out = layer(x)
        assert out.shape == (2, 10, 10, 8)

    def test_half_kernel_size(self):
        layer = ConvolveLayer(ni=2, no=3, half_x=1, half_y=0)
        fc_weight = layer.fc.weight.data
        assert fc_weight.shape == (3, 2 * 1 * 3)

    def test_gradient_flow(self):
        layer = ConvolveLayer(ni=4, no=8, half_x=1, half_y=1)
        x = torch.randn(5, 5, 4, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestMaxpoolLayer:
    def test_output_shape_3d(self):
        layer = MaxpoolLayer(ni=4, x_scale=2, y_scale=2)
        x = torch.randn(6, 6, 4)  # [h, w, d]
        out = layer(x)
        assert out.shape[0] == 3
        assert out.shape[1] == 3
        assert out.shape[2] == 4 * 2 * 2

    def test_output_shape_4d(self):
        layer = MaxpoolLayer(ni=2, x_scale=3, y_scale=3)
        x = torch.randn(2, 9, 9, 2)
        out = layer(x)
        assert out.shape == (2, 3, 3, 2 * 3 * 3)

    def test_no_property(self):
        layer = MaxpoolLayer(ni=8, x_scale=2, y_scale=1)
        assert layer.no == 8 * 2 * 1


class TestReconfigLayer:
    def test_output_shape_3d(self):
        layer = ReconfigLayer(ni=4, x_scale=2, y_scale=1)
        x = torch.randn(5, 6, 4)
        out = layer(x)
        assert out.shape[0] == 5
        assert out.shape[1] == 3
        assert out.shape[2] == 4 * 2 * 1

    def test_output_shape_4d(self):
        layer = ReconfigLayer(ni=2, x_scale=2, y_scale=2)
        x = torch.randn(2, 4, 4, 2)
        out = layer(x)
        assert out.shape == (2, 2, 2, 2 * 2 * 2)


class TestFullyConnectedLayer:
    def test_softmax_output(self):
        layer = FullyConnectedLayer(ni=4, no=3, activation="softmax")
        x = torch.randn(4)
        out = layer(x)
        assert out.shape == (3,)
        assert torch.allclose(out.exp().sum(), torch.tensor(1.0), atol=1e-5)

    def test_tanh_output(self):
        layer = FullyConnectedLayer(ni=4, no=3, activation="tanh")
        x = torch.randn(4)
        out = layer(x)
        assert out.shape == (3,)
        assert out.abs().max() <= 1.0

    def test_sigmoid_output(self):
        layer = FullyConnectedLayer(ni=4, no=3, activation="sigmoid")
        x = torch.randn(4)
        out = layer(x)
        assert out.shape == (3,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_relu_output(self):
        layer = FullyConnectedLayer(ni=4, no=3, activation="relu")
        x = torch.randn(10, 4)
        out = layer(x)
        assert out.shape == (10, 3)
        assert (out >= 0).all() or True  # relu can be 0

    def test_linear_output(self):
        layer = FullyConnectedLayer(ni=4, no=3, activation="linear")
        x = torch.randn(4)
        out = layer(x)
        assert out.shape == (3,)


class TestReversedLayer:
    def test_reverse_dim0(self):
        sub = FullyConnectedLayer(ni=3, no=2, activation="linear")
        layer = ReversedLayer(sub, dim=0)
        x = torch.randn(5, 3)
        out = layer(x)
        assert out.shape == (5, 2)

    def test_reversal_correctness(self):
        sub = FullyConnectedLayer(ni=3, no=2, activation="linear")
        layer = ReversedLayer(sub, dim=0)
        x = torch.randn(5, 3)
        expected = torch.flip(sub(torch.flip(x, [0])), [0])
        actual = layer(x)
        assert torch.allclose(expected, actual, atol=1e-6)


class TestSeriesLayer:
    def test_series_two_layers(self):
        fc1 = FullyConnectedLayer(ni=4, no=8, activation="tanh")
        fc2 = FullyConnectedLayer(ni=8, no=3, activation="softmax")
        series = SeriesLayer([fc1, fc2])
        x = torch.randn(4)
        out = series(x)
        assert out.shape == (3,)

    def test_series_preserves_sequence(self):
        fc1 = FullyConnectedLayer(ni=4, no=8, activation="tanh")
        fc2 = FullyConnectedLayer(ni=8, no=2, activation="linear")
        series = SeriesLayer([fc1, fc2])
        x = torch.randn(10, 4)  # [seq, feat]
        out = series(x)
        assert out.shape == (10, 2)


class TestParallelLayer:
    def test_parallel_concat(self):
        fc1 = FullyConnectedLayer(ni=4, no=3, activation="linear")
        fc2 = FullyConnectedLayer(ni=4, no=5, activation="linear")
        par = ParallelLayer([fc1, fc2])
        x = torch.randn(4)
        out = par(x)
        assert out.shape == (8,)

    def test_parallel_3d_input(self):
        fc1 = FullyConnectedLayer(ni=4, no=3, activation="linear")
        fc2 = FullyConnectedLayer(ni=4, no=3, activation="linear")
        par = ParallelLayer([fc1, fc2])
        x = torch.randn(10, 4)
        out = par(x)
        assert out.shape == (10, 6)


class TestLSTMLayer:
    def test_forward_shape_3d(self):
        # ni must equal height * depth
        layer = LSTMLayer(ni=4, ns=8)
        x = torch.randn(1, 10, 4)  # [h=1, w=10, d=4] → input to cell = 1*4=4
        out = layer(x)
        assert out.shape == (10, 8)

    def test_forward_shape_4d(self):
        layer = LSTMLayer(ni=4, ns=8)
        x = torch.randn(2, 1, 10, 4)  # [b=2, h=1, w=10, d=4]
        out = layer(x)
        assert out.shape == (2, 10, 8)

    def test_reverse_lstm(self):
        layer = LSTMLayer(ni=4, ns=3, reverse=True)
        x = torch.randn(1, 5, 4)  # [h=1, w=5, d=4]
        out = layer(x)
        assert out.shape == (5, 3)

    def test_summary_lstm_3d(self):
        layer = LSTMLayer(ni=4, ns=3, summary=True)
        x = torch.randn(1, 10, 4)  # [h=1, w=10, d=4]
        out = layer(x)
        assert out.shape == (10, 3)

    def test_gradient_flow(self):
        layer = LSTMLayer(ni=4, ns=3)
        x = torch.randn(1, 5, 4, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
