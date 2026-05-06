"""Tests for spec parser and model construction."""

import pytest
import torch

from tesseract_cuda.network.spec_parser import parse_network_spec, LayerDesc
from tesseract_cuda.network.model import TessLSTMModel
from tesseract_cuda.network.layers import (
    LSTMLayer, FullyConnectedLayer, SeriesLayer, ParallelLayer,
    MaxpoolLayer, ReconfigLayer, ConvolveLayer, ReversedLayer,
)
from tesseract_cuda.network.lstm_cell import TesseractLSTMCell


class TestSpecParser:
    def test_simple_lstm_spec(self):
        desc = parse_network_spec("[1,0,0,1Lfx128O1c1]")
        assert desc.type == "series"
        assert len(desc.children) == 3
        assert desc.children[0].type == "input"
        assert desc.children[1].type == "lstm"
        assert desc.children[1].num_states == 128
        assert desc.children[1].direction == "forward"
        assert desc.children[2].type == "output"
        assert desc.children[2].loss_type == "ctc"

    def test_complex_spec(self):
        spec = "[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]"
        desc = parse_network_spec(spec)
        assert desc.type == "series"
        children = desc.children
        assert children[0].type == "input"
        assert children[0].height == 36
        assert children[1].type == "conv"
        assert children[1].kernel_x == 3
        assert children[1].kernel_y == 3
        assert children[1].num_outputs == 16
        assert children[2].type == "maxpool"
        assert children[3].type == "lstm"
        assert children[3].direction == "forward"
        assert children[3].dim == "y"
        assert children[3].summary == True
        assert children[3].num_states == 64
        assert children[4].type == "lstm"
        assert children[4].direction == "forward"
        assert children[4].num_states == 96
        assert children[5].type == "lstm"
        assert children[5].direction == "reverse"
        assert children[6].type == "lstm"
        assert children[6].num_states == 512
        assert children[7].type == "output"

    def test_bidirectional_parallel(self):
        desc = parse_network_spec("(Lfx64Lrx64)")
        assert desc.type == "parallel"
        assert len(desc.children) == 2
        assert desc.children[0].direction == "forward"
        assert desc.children[1].direction == "reverse"

    def test_fc_layer(self):
        desc = parse_network_spec("[1,0,0,1Fc128O1c10]")
        fc = desc.children[1]
        assert fc.type == "fc"
        assert fc.num_outputs == 128

    def test_output_types(self):
        desc = parse_network_spec("[1,0,0,1O2s5]")
        out = desc.children[1]
        assert out.output_type == "heatmap"
        assert out.loss_type == "softmax"
        assert out.num_outputs == 5


class TestModelConstruction:
    def test_build_from_spec(self):
        spec = "[1,0,0,1Lfx32O1c10]"
        model = TessLSTMModel.from_spec(spec, num_classes=10)
        assert model.network is not None
        assert isinstance(model.network, SeriesLayer)
        assert model.network_str == spec

    def test_forward_pass_shape(self):
        spec = "[1,0,0,1Lfx32O1c10]"
        model = TessLSTMModel.from_spec(spec, num_classes=10)

        # Simulate a simple input: [height=1, width=20, depth=1]
        x = torch.randn(1, 20, 1)
        output = model(x)

        # Output should be [width, num_classes]
        assert output.dim() == 2
        assert output.shape[0] == 20  # width preserved
        assert output.shape[1] == 10  # num_classes

    def test_export_import_roundtrip(self):
        spec = "[1,0,0,1Lfx16O1c5]"
        model = TessLSTMModel.from_spec(spec, num_classes=5)
        model.null_char = 0

        export_bytes = model.export_lstm_component(
            training_iteration=100,
            sample_iteration=200,
        )

        # Parse the exported bytes
        from tesseract_cuda.formats.tfile import TFileReader
        from tesseract_cuda.formats.network_ser import deserialize_lstm_component

        r = TFileReader(export_bytes)
        parsed = deserialize_lstm_component(r)

        assert parsed.network_str == spec
        assert parsed.training_iteration == 100
        assert parsed.sample_iteration == 200
        assert parsed.null_char == 0

    def test_lstm_cell_forward(self):
        cell = TesseractLSTMCell(ni=4, ns=3)

        x = torch.randn(4)   # ni=4
        state = torch.zeros(3)  # ns=3
        prev_out = torch.zeros(3)  # ns=3

        output, new_state = cell(x, state, prev_out)
        assert output.shape == (3,)
        assert new_state.shape == (3,)
        assert not torch.allclose(output, torch.zeros(3))  # should produce non-zero output

    def test_lstm_cell_sequence(self):
        cell = TesseractLSTMCell(ni=4, ns=3)
        seq = torch.randn(10, 4)  # 10 timesteps, 4 features

        output = cell.forward_sequence(seq)
        assert output.shape == (10, 3)

    def test_lstm_cell_batched(self):
        cell = TesseractLSTMCell(ni=4, ns=3)
        batch = torch.randn(2, 10, 4)  # batch=2, 10 timesteps, 4 features

        output = cell.forward_sequence(batch)
        assert output.shape == (2, 10, 3)

    def test_summary_lstm(self):
        from tesseract_cuda.network.lstm_cell import TesseractLSTMSummaryCell

        cell = TesseractLSTMSummaryCell(ni=4, ns=3)
        seq = torch.randn(10, 4)

        output = cell.forward_sequence(seq)
        assert output.shape == (3,)  # single output, no batch
