"""5-gate LSTM cell compatible with Tesseract's weight format."""

import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_CLIP = 100.0
ERR_CLIP = 1.0


class TesseractLSTMCell(nn.Module):
    """Custom 5-gate LSTM matching Tesseract's implementation."""

    def __init__(self, ni: int, ns: int, is_2d: bool = False):
        super().__init__()
        self.ni = ni
        self.ns = ns
        self.is_2d = is_2d

        na = ni + ns + (ns if is_2d else 0)

        self.gate_ci = nn.Linear(na, ns)   # Cell input (tanh)
        self.gate_gi = nn.Linear(na, ns)   # Input gate (sigmoid)
        self.gate_gf1 = nn.Linear(na, ns)  # Forget gate 1 (sigmoid)
        self.gate_go = nn.Linear(na, ns)   # Output gate (sigmoid)
        if is_2d:
            self.gate_gfs = nn.Linear(na, ns)

    def forward(self, x: torch.Tensor, state: torch.Tensor,
                prev_output: torch.Tensor,
                prev_output_y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_2d and prev_output_y is not None:
            source = torch.cat([x, prev_output, prev_output_y], dim=-1)
        else:
            source = torch.cat([x, prev_output], dim=-1)

        ci = torch.tanh(self.gate_ci(source))
        gi = torch.sigmoid(self.gate_gi(source))
        gf1 = torch.sigmoid(self.gate_gf1(source))
        go = torch.sigmoid(self.gate_go(source))

        new_state = gf1 * state + ci * gi
        new_state = torch.clamp(new_state, -STATE_CLIP, STATE_CLIP)

        output = torch.tanh(new_state) * go
        return output, new_state

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        has_batch = x_seq.dim() == 3
        if not has_batch:
            x_seq = x_seq.unsqueeze(0)

        batch, seq_len, _ = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        state = torch.zeros(batch, self.ns, device=device, dtype=dtype)
        prev_output = torch.zeros(batch, self.ns, device=device, dtype=dtype)
        outputs: list[torch.Tensor] = []

        for t in range(seq_len):
            prev_output, state = self.forward(x_seq[:, t], state, prev_output)
            outputs.append(prev_output)

        result = torch.stack(outputs, dim=1)
        if not has_batch:
            result = result.squeeze(0)
        return result


class TesseractLSTMSummaryCell(TesseractLSTMCell):
    """LSTM that only outputs at the final timestep."""

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        has_batch = x_seq.dim() == 3
        if not has_batch:
            x_seq = x_seq.unsqueeze(0)

        batch, seq_len, _ = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        state = torch.zeros(batch, self.ns, device=device, dtype=dtype)
        prev_output = torch.zeros(batch, self.ns, device=device, dtype=dtype)

        for t in range(seq_len):
            prev_output, state = self.forward(x_seq[:, t], state, prev_output)

        if not has_batch:
            return prev_output.squeeze(0)
        return prev_output
