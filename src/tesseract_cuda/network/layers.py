"""PyTorch layer implementations matching Tesseract's network types."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .lstm_cell import TesseractLSTMCell, TesseractLSTMSummaryCell


class InputLayer(nn.Module):
    """Pass-through input layer (image preprocessing handled externally)."""

    def __init__(self, ni: int, no: int):
        super().__init__()
        self.ni = ni
        self.no = no

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ConvolveLayer(nn.Module):
    """Tesseract's Convolve: sliding window neighborhood convolution."""

    def __init__(self, ni: int, no: int, half_x: int, half_y: int,
                 activation: str = "tanh"):
        super().__init__()
        self.ni = ni
        self.no = no
        self.half_x = half_x
        self.half_y = half_y
        self.activation = activation
        # Tesseract convolve is implemented as fully connected over the window
        kernel_h = 2 * half_y + 1
        kernel_w = 2 * half_x + 1
        self.fc = nn.Linear(ni * kernel_h * kernel_w, no)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, height, width, depth] or [height, width, depth]
        if x.dim() == 3:
            x = x.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False

        batch, height, width, depth = x.shape
        kh, kw = 2 * self.half_y + 1, 2 * self.half_x + 1

        # Pad spatially
        padded = F.pad(x, [0, 0, self.half_x, self.half_x,
                           self.half_y, self.half_y])

        # Extract patches
        patches = padded.unfold(1, kh, 1).unfold(2, kw, 1)
        # patches: [batch, h_out, w_out, depth, kh, kw]
        patches = patches.contiguous().view(batch, height, width, -1)

        # Apply FC to each position
        out = self.fc(patches)

        if unsqueezed:
            out = out.squeeze(0)
        return out


class MaxpoolLayer(nn.Module):
    """Tesseract's Maxpool: reduces spatial size by x_scale, y_scale."""

    def __init__(self, ni: int, x_scale: int, y_scale: int):
        super().__init__()
        self.ni = ni
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.no = ni * x_scale * y_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [height, width, depth] or [batch, h, w, d]
        if x.dim() == 3:
            x = x.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False

        batch, height, width, depth = x.shape
        ys, xs = self.y_scale, self.x_scale

        # Pad to multiples
        new_h = ((height + ys - 1) // ys) * ys
        new_w = ((width + xs - 1) // xs) * xs
        if new_h != height or new_w != width:
            x = F.pad(x, [0, 0, 0, new_w - width, 0, new_h - height])

        # Reshape and take max, then flatten blocks into depth
        x = x.view(batch, new_h // ys, ys, new_w // xs, xs, depth)
        x = x.max(dim=2).values
        x = x.max(dim=3).values
        # Expand depth by repeating
        x = x.unsqueeze(-1).expand(-1, -1, -1, -1, xs * ys)
        x = x.reshape(batch, new_h // ys, new_w // xs, depth * xs * ys)

        if unsqueezed:
            x = x.squeeze(0)
        return x


class ReconfigLayer(nn.Module):
    """Tesseract's Reconfig: scales time/y size, makes output deeper."""

    def __init__(self, ni: int, x_scale: int, y_scale: int):
        super().__init__()
        self.ni = ni
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.no = ni * x_scale * y_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False

        batch, height, width, depth = x.shape
        ys, xs = self.y_scale, self.x_scale

        # Pad to multiples
        new_h = ((height + ys - 1) // ys) * ys
        new_w = ((width + xs - 1) // xs) * xs
        if new_h != height or new_w != width:
            x = F.pad(x, [0, 0, 0, new_w - width, 0, new_h - height])

        # Reshape: concatenate blocks into depth dimension
        x = x.view(batch, new_h // ys, ys, new_w // xs, xs, depth)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch, new_h // ys, new_w // xs, depth * ys * xs)

        if unsqueezed:
            x = x.squeeze(0)
        return x


class FullyConnectedLayer(nn.Module):
    """Fully connected layer with various activation functions."""

    def __init__(self, ni: int, no: int, activation: str = "softmax"):
        super().__init__()
        self.ni = ni
        self.no = no
        self.activation = activation
        self.fc = nn.Linear(ni, no)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., depth]
        out = self.fc(x)
        if self.activation == "softmax":
            return F.log_softmax(out, dim=-1)
        elif self.activation == "tanh":
            return torch.tanh(out)
        elif self.activation == "sigmoid":
            return torch.sigmoid(out)
        elif self.activation == "relu":
            return F.relu(out)
        return out


class ReversedLayer(nn.Module):
    """Reverses input along a dimension, runs sub-network, reverses output."""

    def __init__(self, sub_net: nn.Module, dim: int = 0):
        super().__init__()
        self.sub_net = sub_net
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reversed_x = torch.flip(x, [self.dim])
        out = self.sub_net(reversed_x)
        return torch.flip(out, [self.dim])


class SeriesLayer(nn.Module):
    """Sequential execution of sub-layers."""

    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ParallelLayer(nn.Module):
    """Runs sub-networks in parallel, concatenates outputs along depth."""

    def __init__(self, nets: list[nn.Module]):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [net(x) for net in self.nets]
        return torch.cat(outputs, dim=-1)


class LSTMLayer(nn.Module):
    """Wrapper for TesseractLSTMCell handling spatial I/O."""

    def __init__(self, ni: int, ns: int, is_2d: bool = False,
                 summary: bool = False, reverse: bool = False):
        super().__init__()
        self.ni = ni
        self.ns = ns
        self.is_2d = is_2d
        self.summary = summary
        self.reverse = reverse

        if summary:
            self.cell = TesseractLSTMSummaryCell(ni, ns, is_2d)
        else:
            self.cell = TesseractLSTMCell(ni, ns, is_2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [height, width, depth] or [batch, h, w, d]
        # For 1D LSTM along x: treat width as sequence, flatten h*d as features
        if x.dim() == 3:
            height, width, depth = x.shape
            # Reshape to [width, height*depth] as sequence
            x = x.permute(1, 0, 2).reshape(width, height * depth)
            if self.reverse:
                x = torch.flip(x, [0])
            out = self.cell.forward_sequence(x)
            if self.reverse:
                out = torch.flip(out, [0])
            if self.summary:
                # Summary outputs single timestep, expand back
                out = out.unsqueeze(0).expand(width, -1)
                return out
            # out: [width, ns]
            return out
        else:
            batch, height, width, depth = x.shape
            x = x.permute(0, 1, 2, 3).reshape(batch, width, height * depth)
            if self.reverse:
                x = torch.flip(x, [1])
            out = self.cell.forward_sequence(x)
            if self.reverse:
                out = torch.flip(out, [1])
            if self.summary:
                out = out.unsqueeze(1).expand(-1, width, -1)
                return out
            return out
