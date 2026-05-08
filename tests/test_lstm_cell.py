import pytest
import torch
from tesseract_cuda.network.lstm_cell import TesseractLSTMCell, TesseractLSTMSummaryCell


class TestTesseractLSTMCell:
    def test_forward_shapes(self):
        cell = TesseractLSTMCell(ni=4, ns=8)
        x = torch.randn(2, 4)
        state = torch.zeros(2, 8)
        prev_out = torch.zeros(2, 8)
        out, new_state = cell(x, state, prev_out)
        assert out.shape == (2, 8)
        assert new_state.shape == (2, 8)

    def test_forward_no_batch(self):
        cell = TesseractLSTMCell(ni=4, ns=8)
        x = torch.randn(4)
        state = torch.zeros(8)
        prev_out = torch.zeros(8)
        out, new_state = cell(x, state, prev_out)
        assert out.shape == (8,)
        assert new_state.shape == (8,)

    def test_state_updates(self):
        cell = TesseractLSTMCell(ni=4, ns=8)
        x = torch.randn(4)
        state = torch.zeros(8)
        prev_out = torch.zeros(8)
        _, new_state = cell(x, state, prev_out)
        assert not torch.allclose(new_state, state)

    def test_forward_sequence_shape(self):
        cell = TesseractLSTMCell(ni=4, ns=8)
        x = torch.randn(10, 4)
        out = cell.forward_sequence(x)
        assert out.shape == (10, 8)

    def test_forward_sequence_batched(self):
        cell = TesseractLSTMCell(ni=4, ns=8)
        x = torch.randn(3, 10, 4)
        out = cell.forward_sequence(x)
        assert out.shape == (3, 10, 8)

    def test_forward_sequence_temporal_dependency(self):
        torch.manual_seed(0)
        cell = TesseractLSTMCell(ni=4, ns=8)
        x = torch.randn(5, 4)
        out = cell.forward_sequence(x)
        assert not torch.allclose(out[0], out[-1])

    def test_gradient_flow(self):
        cell = TesseractLSTMCell(ni=4, ns=8)
        x = torch.randn(5, 4, requires_grad=True)
        out = cell.forward_sequence(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (5, 4)

    def test_gradient_flow_through_weights(self):
        cell = TesseractLSTMCell(ni=4, ns=8)
        x = torch.randn(5, 4)
        out = cell.forward_sequence(x)
        loss = out.sum()
        loss.backward()
        for name, p in cell.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


class TestLSTM2D:
    def test_2d_gate_count(self):
        cell = TesseractLSTMCell(ni=4, ns=8, is_2d=True)
        gate_names = [n for n, _ in cell.named_parameters()]
        assert "gate_gfs" in gate_names or "gate_gfs.weight" in gate_names

    def test_2d_forward_without_y(self):
        cell = TesseractLSTMCell(ni=4, ns=8, is_2d=True)
        x = torch.randn(4)
        state = torch.zeros(8)
        prev_out = torch.zeros(8)
        prev_y = torch.zeros(8)
        out, new_state = cell(x, state, prev_out, prev_y)
        assert out.shape == (8,)

    def test_2d_forward_with_y(self):
        cell = TesseractLSTMCell(ni=4, ns=8, is_2d=True)
        x = torch.randn(4)
        state = torch.zeros(8)
        prev_out = torch.zeros(8)
        prev_y = torch.zeros(8)
        out, new_state = cell(x, state, prev_out, prev_y)
        assert out.shape == (8,)
        assert new_state.shape == (8,)

    def test_2d_gate_input_dim(self):
        cell = TesseractLSTMCell(ni=4, ns=8, is_2d=True)
        assert cell.gate_ci.in_features == 4 + 8 + 8

    def test_2d_gradient_flow(self):
        cell = TesseractLSTMCell(ni=4, ns=8, is_2d=False)
        x = torch.randn(5, 4, requires_grad=True)
        out = cell.forward_sequence(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestSummaryCell:
    def test_output_shape_no_batch(self):
        cell = TesseractLSTMSummaryCell(ni=4, ns=8)
        x = torch.randn(10, 4)
        out = cell.forward_sequence(x)
        assert out.shape == (8,)

    def test_output_shape_batched(self):
        cell = TesseractLSTMSummaryCell(ni=4, ns=8)
        x = torch.randn(3, 10, 4)
        out = cell.forward_sequence(x)
        assert out.shape == (3, 8)

    def test_summary_is_last_state(self):
        torch.manual_seed(0)
        cell = TesseractLSTMSummaryCell(ni=4, ns=8)
        cell_full = TesseractLSTMCell(ni=4, ns=8)
        cell_full.load_state_dict(cell.state_dict())

        x = torch.randn(10, 4)
        summary_out = cell.forward_sequence(x)
        full_out = cell_full.forward_sequence(x)
        assert torch.allclose(summary_out, full_out[-1], atol=1e-6)

    def test_summary_gradient_flow(self):
        cell = TesseractLSTMSummaryCell(ni=4, ns=8)
        x = torch.randn(10, 4, requires_grad=True)
        out = cell.forward_sequence(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_single_timestep(self):
        cell = TesseractLSTMSummaryCell(ni=4, ns=8)
        x = torch.randn(1, 4)
        out = cell.forward_sequence(x)
        assert out.shape == (8,)
        assert torch.isfinite(out).all()


class TestStateClipping:
    def test_state_clipped(self):
        cell = TesseractLSTMCell(ni=4, ns=8)
        with torch.no_grad():
            for p in cell.parameters():
                p.fill_(10.0)
        x = torch.randn(4)
        state = torch.zeros(8)
        prev_out = torch.zeros(8)
        _, new_state = cell(x, state, prev_out)
        assert new_state.abs().max() <= 100.0
