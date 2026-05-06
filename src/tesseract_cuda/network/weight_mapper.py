"""Map weights between PyTorch model and Tesseract binary format."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from ..formats.network_ser import (
    NetworkLayer, WeightMatrix, LSTMModel,
    TYPE_NAME_TO_ID, NF_LAYER_SPECIFIC_LR,
    CI, GI, GF1, GO, GFS,
)
from .layers import (
    TesseractLSTMCell, LSTMLayer, FullyConnectedLayer,
    ConvolveLayer, MaxpoolLayer, ReconfigLayer,
    SeriesLayer, ParallelLayer, ReversedLayer, InputLayer,
)


def load_weights_to_model(network_layer: NetworkLayer, model: nn.Module) -> None:
    """Load weights from a deserialized Tesseract NetworkLayer into a PyTorch model.

    Recursively traverses the network tree and copies weights.
    """
    _load_recursive(network_layer, model)


def _load_recursive(nl: NetworkLayer, module: nn.Module) -> None:
    if isinstance(module, SeriesLayer):
        for i, child_module in enumerate(module.layers):
            if i < len(nl.children):
                _load_recursive(nl.children[i], child_module)

    elif isinstance(module, ParallelLayer):
        for i, child_module in enumerate(module.nets):
            if i < len(nl.children):
                _load_recursive(nl.children[i], child_module)

    elif isinstance(module, ReversedLayer):
        if nl.children:
            _load_recursive(nl.children[0], module.sub_net)

    elif isinstance(module, LSTMLayer):
        cell = module.cell
        _load_lstm_weights(nl, cell)

    elif isinstance(module, FullyConnectedLayer):
        if nl.weights:
            _load_fc_weight(nl.weights[0], module.fc)

    elif isinstance(module, ConvolveLayer):
        if nl.weights:
            _load_fc_weight(nl.weights[0], module.fc)


def _load_lstm_weights(nl: NetworkLayer, cell: TesseractLSTMCell) -> None:
    """Load LSTM gate weights from Tesseract format."""
    gates: list[nn.Linear] = [cell.gate_ci, cell.gate_gi, cell.gate_gf1, cell.gate_go]
    if cell.is_2d:
        gates.append(cell.gate_gfs)

    for i, gate in enumerate(gates):
        if i < len(nl.weights):
            _load_fc_weight(nl.weights[i], gate)

    # Handle internal softmax sub-network
    if nl.softmax is not None and hasattr(cell, 'softmax') and cell.softmax is not None:
        _load_recursive(nl.softmax, cell.softmax)


def _load_fc_weight(wm: WeightMatrix, linear: nn.Linear) -> None:
    """Load a Tesseract WeightMatrix into a PyTorch nn.Linear layer.

    Tesseract stores weights as [no, ni+1] (last column is bias).
    PyTorch nn.Linear stores weight as [no, ni] and bias as [no].
    """
    weights = np.array(wm.weights, dtype=np.float64).reshape(wm.dim1, wm.dim2)

    # All columns except last -> weight
    w = torch.as_tensor(weights[:, :-1].astype(np.float32))
    # Last column -> bias
    b = torch.as_tensor(weights[:, -1].astype(np.float32))

    linear.weight.data.copy_(w)
    linear.bias.data.copy_(b)


def extract_weights_from_model(model: nn.Module, spec_str: str = "") -> NetworkLayer:
    """Extract weights from a PyTorch model into a NetworkLayer tree.

    This is the inverse of load_weights_to_model.
    """
    return _extract_recursive(model)


def _extract_recursive(module: nn.Module) -> NetworkLayer:
    if isinstance(module, SeriesLayer):
        children = [_extract_recursive(m) for m in module.layers]
        ni = children[0].ni if children else 0
        no = children[-1].no if children else 0
        nw = sum(c.num_weights for c in children)
        return NetworkLayer(
            type_id=TYPE_NAME_TO_ID["Series"], type_name="Series",
            training=0, needs_backprop=False, network_flags=0,
            ni=ni, no=no, num_weights=nw, name="Series",
            children=children,
        )

    elif isinstance(module, ParallelLayer):
        children = [_extract_recursive(m) for m in module.nets]
        ni = children[0].ni if children else 0
        no = sum(c.no for c in children)
        nw = sum(c.num_weights for c in children)
        return NetworkLayer(
            type_id=TYPE_NAME_TO_ID["Parallel"], type_name="Parallel",
            training=0, needs_backprop=False, network_flags=0,
            ni=ni, no=no, num_weights=nw, name="Parallel",
            children=children,
        )

    elif isinstance(module, ReversedLayer):
        child = _extract_recursive(module.sub_net)
        return NetworkLayer(
            type_id=TYPE_NAME_TO_ID["RTLReversed"], type_name="RTLReversed",
            training=0, needs_backprop=False, network_flags=0,
            ni=child.ni, no=child.no, num_weights=child.num_weights,
            name="RTLReversed", children=[child],
        )

    elif isinstance(module, LSTMLayer):
        return _extract_lstm(module)

    elif isinstance(module, FullyConnectedLayer):
        wm = _extract_fc_weight(module.fc)
        type_name = "Softmax" if module.activation == "softmax" else "Tanh"
        return NetworkLayer(
            type_id=TYPE_NAME_TO_ID.get(type_name, 22), type_name=type_name,
            training=0, needs_backprop=False, network_flags=0,
            ni=module.ni, no=module.no,
            num_weights=wm.dim1 * wm.dim2, name=type_name,
            weights=[wm],
        )

    elif isinstance(module, ConvolveLayer):
        wm = _extract_fc_weight(module.fc)
        return NetworkLayer(
            type_id=TYPE_NAME_TO_ID["Convolve"], type_name="Convolve",
            training=0, needs_backprop=False, network_flags=0,
            ni=module.ni, no=module.no,
            num_weights=wm.dim1 * wm.dim2, name="Convolve",
            weights=[wm], half_x=module.half_x, half_y=module.half_y,
        )

    elif isinstance(module, MaxpoolLayer):
        return NetworkLayer(
            type_id=TYPE_NAME_TO_ID["Maxpool"], type_name="Maxpool",
            training=0, needs_backprop=False, network_flags=0,
            ni=module.ni, no=module.no, num_weights=0, name="Maxpool",
            x_scale=module.x_scale, y_scale=module.y_scale,
        )

    elif isinstance(module, ReconfigLayer):
        return NetworkLayer(
            type_id=TYPE_NAME_TO_ID["Reconfig"], type_name="Reconfig",
            training=0, needs_backprop=False, network_flags=0,
            ni=module.ni, no=module.no, num_weights=0, name="Reconfig",
            x_scale=module.x_scale, y_scale=module.y_scale,
        )

    elif isinstance(module, InputLayer):
        return NetworkLayer(
            type_id=TYPE_NAME_TO_ID["Input"], type_name="Input",
            training=0, needs_backprop=False, network_flags=0,
            ni=module.ni, no=module.no, num_weights=0, name="Input",
        )

    raise ValueError(f"Unknown module type: {type(module)}")


def _extract_lstm(module: LSTMLayer) -> NetworkLayer:
    cell = module.cell
    gates = [cell.gate_ci, cell.gate_gi, cell.gate_gf1, cell.gate_go]
    if cell.is_2d:
        gates.append(cell.gate_gfs)

    weights = [_extract_fc_weight(g) for g in gates]

    ns = cell.ns
    ni = cell.ni
    na = ni + ns + (ns if cell.is_2d else 0)

    type_name = "SummLSTM" if module.summary else "LSTM"

    return NetworkLayer(
        type_id=TYPE_NAME_TO_ID[type_name], type_name=type_name,
        training=0, needs_backprop=False, network_flags=0,
        ni=ni, no=ns, num_weights=sum(w.dim1 * w.dim2 for w in weights),
        name=type_name, weights=weights, na=na,
        is_2d=cell.is_2d, ns=ns,
    )


def _extract_fc_weight(linear: nn.Linear) -> WeightMatrix:
    """Extract weights from nn.Linear into Tesseract WeightMatrix format.

    PyTorch: weight [no, ni], bias [no]
    Tesseract: combined [no, ni+1] (last column = bias)
    """
    w = linear.weight.data.numpy()
    b = linear.bias.data.numpy().reshape(-1, 1)
    combined = np.concatenate([w, b], axis=1)
    return WeightMatrix(
        dim1=combined.shape[0],
        dim2=combined.shape[1],
        weights=combined.astype(np.float64).flatten().tolist(),
    )
