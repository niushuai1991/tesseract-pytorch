"""Top-level TessLSTMModel that combines all components."""

import torch
import torch.nn as nn
from typing import Optional

from ..formats.tessdata import TessdataManager, TESSDATA_LSTM
from ..formats.network_ser import (
    deserialize_lstm_component, serialize_lstm_component, LSTMModel,
    NetworkLayer as NetworkLayerSer,
)
from ..formats.tfile import TFileReader, TFileWriter
from .spec_parser import parse_network_spec, LayerDesc
from .layers import (
    InputLayer, ConvolveLayer, MaxpoolLayer, ReconfigLayer,
    FullyConnectedLayer, ReversedLayer, XYTransposeLayer,
    SeriesLayer, ParallelLayer, LSTMLayer,
)
from .weight_mapper import load_weights_to_model, extract_weights_from_model


class TessLSTMModel(nn.Module):
    """Tesseract-compatible LSTM model for OCR training."""

    def __init__(self):
        super().__init__()
        self.network: Optional[nn.Module] = None
        self.network_str: str = ""
        self.null_char: int = 0
        self.training_flags: int = 0

    @classmethod
    def from_spec(cls, spec: str, num_classes: int) -> "TessLSTMModel":
        model = cls()
        desc = parse_network_spec(spec)
        model.network_str = spec
        model.network = _build_series(desc, num_classes)
        return model

    @classmethod
    def from_traineddata(cls, path: str) -> "TessLSTMModel":
        mgr = TessdataManager.from_file(path)
        lstm_data = mgr.get_component(TESSDATA_LSTM)
        if not lstm_data:
            raise ValueError(f"No LSTM component in {path}")

        reader = TFileReader(lstm_data)
        lstm_model = deserialize_lstm_component(reader)

        model = cls()
        model.network_str = lstm_model.network_str
        model.null_char = lstm_model.null_char
        model.training_flags = lstm_model.training_flags

        desc = parse_network_spec(lstm_model.network_str)
        model.network = _build_series(desc, lstm_model.network.no)
        load_weights_to_model(lstm_model.network, model.network)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def export_lstm_component(self, training_iteration: int = 0,
                              sample_iteration: int = 0) -> bytes:
        nl = extract_weights_from_model(self.network)
        nl.num_weights = _count_all_weights(nl)

        lstm_model = LSTMModel(
            network=nl,
            network_str=self.network_str,
            training_flags=self.training_flags,
            training_iteration=training_iteration,
            sample_iteration=sample_iteration,
            null_char=self.null_char,
            adam_beta=0.999,
            learning_rate=0.001,
            momentum=0.5,
        )

        writer = TFileWriter()
        serialize_lstm_component(lstm_model, writer, training=False)
        return writer.get_bytes()


def _build_series(desc: LayerDesc, num_classes: int) -> nn.Module:
    """Build a complete network from a parsed spec (always a series at top level)."""
    if desc.type == "series" and desc.children:
        return _build_series_children(desc.children, num_classes)
    elif desc.type == "parallel" and desc.children:
        return _build_parallel(desc.children, num_classes)
    raise ValueError(f"Expected series/parallel at top level, got {desc.type}")


def _build_series_children(children: list[LayerDesc], num_classes: int) -> SeriesLayer:
    """Build a SeriesLayer from children, tracking ni through the chain."""
    layers = []
    ni = 0

    for child in children:
        if child.type == "input":
            ni = child.depth
            layer = InputLayer(ni, ni)
            layers.append(layer)

        elif child.type == "conv":
            half_x = child.kernel_x // 2
            half_y = child.kernel_y // 2
            layer = ConvolveLayer(ni, child.num_outputs, half_x, half_y, child.activation)
            layers.append(layer)
            ni = child.num_outputs

        elif child.type == "maxpool":
            layer = MaxpoolLayer(ni, child.x_scale, child.y_scale)
            layers.append(layer)
            # Maxpool does NOT change depth (unlike Reconfig)

        elif child.type == "reconfig":
            layer = ReconfigLayer(ni, child.x_scale, child.y_scale)
            layers.append(layer)
            ni = ni * child.x_scale * child.y_scale

        elif child.type == "lstm":
            is_reverse = child.direction == "reverse"
            is_bidi = child.direction == "bidirectional"
            dim_y = child.dim == "y"

            if is_bidi:
                fwd = LSTMLayer(
                    ni=ni, ns=child.num_states,
                    summary=child.summary, reverse=False,
                )
                rev = LSTMLayer(
                    ni=ni, ns=child.num_states,
                    summary=child.summary, reverse=True,
                )
                layer = ParallelLayer([fwd, rev])
                ni = child.num_states * 2
            else:
                lstm = LSTMLayer(
                    ni=ni, ns=child.num_states,
                    summary=child.summary,
                    reverse=is_reverse,
                )
                if is_reverse:
                    lstm = ReversedLayer(lstm, dim="x")
                if dim_y:
                    lstm = XYTransposeLayer(lstm)
                layer = lstm
                ni = child.num_states
            layers.append(layer)

        elif child.type == "fc":
            layer = FullyConnectedLayer(ni, child.num_outputs, child.activation)
            layers.append(layer)
            ni = child.num_outputs

        elif child.type == "output":
            n = num_classes if num_classes > 0 else child.num_outputs
            if n <= 0:
                n = child.num_outputs
            act = "logistic" if child.loss_type == "logistic" else "softmax"
            layer = FullyConnectedLayer(ni, n, act)
            layers.append(layer)
            ni = n

        elif child.type == "series":
            layer = _build_series(child, num_classes)
            layers.append(layer)

        elif child.type == "parallel":
            layer = _build_parallel(child.children or [], num_classes, ni)
            layers.append(layer)
            # Parallel adds depths
            ni = sum(_get_output_size(n) for n in layer.nets)

        else:
            raise ValueError(f"Unknown layer type: {child.type}")

    return SeriesLayer(layers)


def _build_parallel(children: list[LayerDesc], num_classes: int, ni: int = 0) -> ParallelLayer:
    """Build a ParallelLayer from children."""
    nets = []
    for child in children:
        if child.type == "series" and child.children:
            nets.append(_build_series_children(child.children, num_classes))
        elif child.type == "lstm":
            lstm = LSTMLayer(
                ni=ni, ns=child.num_states,
                summary=child.summary,
                reverse=(child.direction == "reverse"),
            )
            if child.direction == "reverse":
                lstm = ReversedLayer(lstm, dim="x")
            if child.dim == "y":
                lstm = XYTransposeLayer(lstm)
            nets.append(lstm)
        else:
            raise ValueError(f"Unsupported parallel child: {child.type}")
    return ParallelLayer(nets)


def _get_output_size(module: nn.Module) -> int:
    """Get the output dimension of a module."""
    if isinstance(module, LSTMLayer):
        return module.ns
    if isinstance(module, FullyConnectedLayer):
        return module.no
    if isinstance(module, ConvolveLayer):
        return module.no
    if isinstance(module, MaxpoolLayer):
        return module.no
    if isinstance(module, ReconfigLayer):
        return module.no
    if isinstance(module, ReversedLayer):
        return _get_output_size(module.sub_net)
    if isinstance(module, XYTransposeLayer):
        return _get_output_size(module.sub_net)
    return 0


def _count_all_weights(nl: NetworkLayerSer) -> int:
    total = 0
    for wm in nl.weights:
        total += wm.dim1 * wm.dim2
    for child in nl.children:
        total += _count_all_weights(child)
    if nl.softmax:
        total += _count_all_weights(nl.softmax)
    return total
