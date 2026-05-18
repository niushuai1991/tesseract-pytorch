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
    SeriesLayer, ParallelLayer, ReversedLayer, XYTransposeLayer, InputLayer,
)


def load_weights_to_model(network_layer: NetworkLayer, model: nn.Module) -> None:
    _load_recursive(network_layer, model)


def extract_weights_from_model(model: nn.Module,
                               preserved_convs: list | None = None) -> NetworkLayer:
    return _extract_recursive(model, preserved_convs or [])


def _load_recursive(nl: NetworkLayer, module: nn.Module) -> None:
    if isinstance(module, SeriesLayer):
        for i, child_module in enumerate(module.layers):
            if i >= len(nl.children):
                continue
            child_nl = nl.children[i]
            if isinstance(child_module, ConvolveLayer):
                if _try_load_convolve(child_nl, child_module):
                    continue
            _load_recursive(child_nl, child_module)

    elif isinstance(module, ParallelLayer):
        for i, child_module in enumerate(module.nets):
            if i < len(nl.children):
                _load_recursive(nl.children[i], child_module)

    elif isinstance(module, ReversedLayer):
        if nl.children:
            _load_recursive(nl.children[0], module.sub_net)

    elif isinstance(module, XYTransposeLayer):
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


def _try_load_convolve(nl: NetworkLayer, module: ConvolveLayer) -> bool:
    if nl.type_name == "Series" and len(nl.children) == 2:
        c0, c1 = nl.children
        if (c0.type_name == "Convolve"
                and c1.type_name in ("Tanh", "Sigmoid", "Relu", "Linear", "Logistic")
                and c1.weights):
            _load_fc_weight(c1.weights[0], module.fc)
            return True
    if nl.type_name == "Convolve" and nl.weights:
        _load_fc_weight(nl.weights[0], module.fc)
        return True
    return False


def _load_lstm_weights(nl: NetworkLayer, cell: TesseractLSTMCell) -> None:
    gates: list[nn.Linear] = [cell.gate_ci, cell.gate_gi, cell.gate_gf1, cell.gate_go]
    if cell.is_2d:
        gates.append(cell.gate_gfs)

    for i, gate in enumerate(gates):
        if i < len(nl.weights):
            _load_fc_weight(nl.weights[i], gate)

    if nl.softmax is not None and hasattr(cell, 'softmax') and cell.softmax is not None:
        _load_recursive(nl.softmax, cell.softmax)


def _load_fc_weight(wm: WeightMatrix, linear: nn.Linear) -> None:
    weights = np.array(wm.weights, dtype=np.float64).reshape(wm.dim1, wm.dim2)
    w = torch.as_tensor(weights[:, :-1].astype(np.float32))
    b = torch.as_tensor(weights[:, -1].astype(np.float32))
    linear.weight.data.copy_(w)
    linear.bias.data.copy_(b)


def _extract_recursive(module: nn.Module, preserved_convs: list) -> NetworkLayer:
    if isinstance(module, SeriesLayer):
        children = [_extract_recursive(m, preserved_convs) for m in module.layers]
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
        children = [_extract_recursive(m, preserved_convs) for m in module.nets]
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
        child = _extract_recursive(module.sub_net, preserved_convs)
        return NetworkLayer(
            type_id=TYPE_NAME_TO_ID["RTLReversed"], type_name="RTLReversed",
            training=0, needs_backprop=False, network_flags=0,
            ni=child.ni, no=child.no, num_weights=child.num_weights,
            name="RTLReversed", children=[child],
        )

    elif isinstance(module, XYTransposeLayer):
        child = _extract_recursive(module.sub_net, preserved_convs)
        return NetworkLayer(
            type_id=TYPE_NAME_TO_ID["XYTranspose"], type_name="XYTranspose",
            training=0, needs_backprop=False, network_flags=0,
            ni=child.ni, no=child.no, num_weights=child.num_weights,
            name="XYTranspose", children=[child],
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
        kernel_h = 2 * module.half_y + 1
        kernel_w = 2 * module.half_x + 1
        conv_no = module.ni * kernel_h * kernel_w
        conv_wm = None
        for pc in preserved_convs:
            if pc.type_name == "Convolve" and pc.ni == module.ni and pc.half_x == module.half_x:
                conv_wm = pc.weights[0] if pc.weights else None
                break
        if conv_wm is None:
            conv_wm = WeightMatrix(
                dim1=conv_no, dim2=conv_no + 1,
                weights=[0.0] * (conv_no * (conv_no + 1)),
            )
        conv_nl = NetworkLayer(
            type_id=TYPE_NAME_TO_ID["Convolve"], type_name="Convolve",
            training=0, needs_backprop=False, network_flags=0,
            ni=module.ni, no=conv_no,
            num_weights=conv_wm.dim1 * conv_wm.dim2, name="Convolve",
            weights=[conv_wm],
            half_x=module.half_x, half_y=module.half_y,
        )
        act_name = {"tanh": "Tanh", "relu": "Relu", "sigmoid": "Logistic"}.get(
            module.activation, "Tanh")
        act_nl = NetworkLayer(
            type_id=TYPE_NAME_TO_ID.get(act_name, 22), type_name=act_name,
            training=0, needs_backprop=False, network_flags=0,
            ni=conv_no, no=module.no,
            num_weights=wm.dim1 * wm.dim2, name=act_name,
            weights=[wm],
        )
        return NetworkLayer(
            type_id=TYPE_NAME_TO_ID["Series"], type_name="Series",
            training=0, needs_backprop=False, network_flags=0,
            ni=module.ni, no=module.no,
            num_weights=conv_nl.num_weights + act_nl.num_weights, name="Series",
            children=[conv_nl, act_nl],
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
    w = linear.weight.data.cpu().numpy()
    b = linear.bias.data.cpu().numpy().reshape(-1, 1)
    combined = np.concatenate([w, b], axis=1)
    return WeightMatrix(
        dim1=combined.shape[0],
        dim2=combined.shape[1],
        weights=combined.astype(np.float64).flatten().tolist(),
    )
