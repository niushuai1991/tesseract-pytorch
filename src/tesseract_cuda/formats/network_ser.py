"""Serialize and deserialize Tesseract LSTM network weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from .tfile import TFileReader, TFileWriter

# Network type names, synced with kTypeNames in network.cpp
TYPE_NAMES = [
    "Invalid", "Input",
    "Convolve", "Maxpool",
    "Parallel", "Replicated",
    "ParBidiLSTM", "DepParUDLSTM",
    "Par2dLSTM", "Series",
    "Reconfig", "RTLReversed",
    "TTBReversed", "XYTranspose",
    "LSTM", "SummLSTM",
    "Logistic", "LinLogistic",
    "LinTanh", "Tanh",
    "Relu", "Linear",
    "Softmax", "SoftmaxNoCTC",
    "LSTMSoftmax", "LSTMBinarySoftmax",
    "TensorFlow",
]

TYPE_NAME_TO_ID = {name: i for i, name in enumerate(TYPE_NAMES)}

# NetworkFlags
NF_LAYER_SPECIFIC_LR = 64
NF_ADAM = 128

# WeightMatrix mode flags
WM_INT8 = 1
WM_ADAM = 4
WM_DOUBLE = 128

# LSTM gate indices
CI, GI, GF1, GO, GFS = 0, 1, 2, 3, 4
WT_COUNT = 5


@dataclass
class WeightMatrix:
    dim1: int = 0
    dim2: int = 0
    weights: list[float] = field(default_factory=list)
    int_mode: bool = False
    use_adam: bool = False

    @property
    def num_outputs(self) -> int:
        return self.dim1

    @property
    def num_inputs(self) -> int:
        return self.dim2 - 1  # last column is bias

    def get_weights_np(self) -> "np.ndarray":
        import numpy as np
        return np.array(self.weights, dtype=np.float64).reshape(self.dim1, self.dim2)

    def set_weights_np(self, arr: "np.ndarray") -> None:
        import numpy as np
        self.dim1 = arr.shape[0]
        self.dim2 = arr.shape[1]
        self.weights = arr.astype(np.float64).flatten().tolist()

    @classmethod
    def deserialize(cls, reader: TFileReader, training: bool) -> "WeightMatrix":
        mode = reader.read_uint8()
        int_mode = (mode & WM_INT8) != 0
        use_adam = (mode & WM_ADAM) != 0
        has_double = (mode & WM_DOUBLE) != 0

        wm = cls(int_mode=int_mode, use_adam=use_adam)

        if int_mode:
            # Int8 quantized mode
            dim1, dim2, data_i = _read_2d_int8_array(reader)
            wm.dim1 = dim1
            wm.dim2 = dim2
            # Read scales
            num_scales = reader.read_uint32()
            scales = []
            for _ in range(num_scales):
                if has_double:
                    scales.append(reader.read_double())
                else:
                    scales.append(reader.read_float())
            # Skip shaped_w if SIMD
        else:
            # Float/double mode
            wm.dim1, wm.dim2, wm.weights = reader.read_2d_double_array()
            if training:
                # Read updates_
                _dim1, _dim2, _updates = reader.read_2d_double_array()
                if use_adam:
                    _dim1, _dim2, _sq_sum = reader.read_2d_double_array()

        return wm

    def serialize(self, writer: TFileWriter, training: bool = False) -> None:
        mode = WM_DOUBLE  # Always write double format
        if self.use_adam:
            mode |= WM_ADAM
        writer.write_uint8(mode)

        # Write weights as GENERIC_2D_ARRAY<double>
        writer.write_uint32(self.dim1)
        writer.write_uint32(self.dim2)
        writer.write_double(0.0)  # empty value
        writer.write_double_array(self.weights)

        if training and not self.int_mode:
            # Write updates_ (same shape, all zeros for export)
            writer.write_uint32(self.dim1)
            writer.write_uint32(self.dim2)
            writer.write_double(0.0)
            writer.write_double_array([0.0] * (self.dim1 * self.dim2))
            if self.use_adam:
                writer.write_uint32(self.dim1)
                writer.write_uint32(self.dim2)
                writer.write_double(0.0)
                writer.write_double_array([0.0] * (self.dim1 * self.dim2))


def _read_2d_int8_array(reader: TFileReader):
    dim1 = reader.read_uint32()
    dim2 = reader.read_uint32()
    _empty = reader.read_int8()
    total = dim1 * dim2
    if total == 0:
        return dim1, dim2, []
    data = list(reader.read_bytes(total))
    return dim1, dim2, data


@dataclass
class NetworkLayer:
    type_id: int
    type_name: str
    training: int  # 0=disabled, 1=enabled
    needs_backprop: bool
    network_flags: int
    ni: int
    no: int
    num_weights: int
    name: str
    # Layer-specific data
    children: list["NetworkLayer"] = field(default_factory=list)
    weights: list[WeightMatrix] = field(default_factory=list)
    # Convolve
    half_x: int = 0
    half_y: int = 0
    # Reconfig/Maxpool
    x_scale: int = 0
    y_scale: int = 0
    # LSTM
    na: int = 0
    is_2d: bool = False
    ns: int = 0
    nf: int = 0
    softmax: Optional["NetworkLayer"] = None
    # Input
    input_shape: Optional[tuple[int, int, int, int]] = None
    # Plumbing
    learning_rates: list[float] = field(default_factory=list)

    @property
    def is_plumbing(self) -> bool:
        return self.type_name in (
            "Series", "Parallel", "Replicated",
            "ParBidiLSTM", "DepParUDLSTM", "Par2dLSTM",
            "RTLReversed", "TTBReversed", "XYTranspose",
        )

    @property
    def is_lstm(self) -> bool:
        return self.type_name in ("LSTM", "SummLSTM", "LSTMSoftmax", "LSTMBinarySoftmax")

    @property
    def has_weights(self) -> bool:
        return len(self.weights) > 0 or self.is_lstm


def deserialize_network(reader: TFileReader) -> Optional[NetworkLayer]:
    """Read a complete network from binary data (recursively)."""
    # Read common header
    type_id = reader.read_int8()
    if type_id == 0:  # NT_NONE - new format with string type name
        type_name = reader.read_string()
        type_id = TYPE_NAME_TO_ID.get(type_name, 0)
    else:
        type_name = TYPE_NAMES[type_id] if type_id < len(TYPE_NAMES) else f"Unknown({type_id})"

    training_val = reader.read_int8()
    needs_backprop = reader.read_int8() != 0
    network_flags = reader.read_int32()
    ni = reader.read_int32()
    no = reader.read_int32()
    num_weights = reader.read_int32()
    name = reader.read_string()

    is_training = training_val == 1

    layer = NetworkLayer(
        type_id=type_id,
        type_name=type_name,
        training=training_val,
        needs_backprop=needs_backprop,
        network_flags=network_flags,
        ni=ni,
        no=no,
        num_weights=num_weights,
        name=name,
    )

    # Read layer-specific data
    if type_name in ("Series", "Parallel", "Replicated",
                     "ParBidiLSTM", "DepParUDLSTM", "Par2dLSTM",
                     "RTLReversed", "TTBReversed", "XYTranspose"):
        # Plumbing: uint32 stack_size + children + optional learning_rates
        stack_size = reader.read_uint32()
        for _ in range(stack_size):
            child = deserialize_network(reader)
            if child is None:
                return None
            layer.children.append(child)
        if network_flags & NF_LAYER_SPECIFIC_LR:
            count = reader.read_uint32()
            layer.learning_rates = [reader.read_float() for _ in range(count)]

    elif type_name in ("LSTM", "SummLSTM", "LSTMSoftmax", "LSTMBinarySoftmax"):
        layer.na = reader.read_int32()
        # Determine nf
        if type_name == "LSTMSoftmax":
            layer.nf = no
        elif type_name == "LSTMBinarySoftmax":
            import math
            layer.nf = math.ceil(math.log2(no)) if no > 1 else 1
        else:
            layer.nf = 0

        # Read 4 gates (CI, GI, GF1, GO)
        for w in range(4):
            wm = WeightMatrix.deserialize(reader, is_training)
            layer.weights.append(wm)
            if w == CI:
                layer.ns = wm.num_outputs
                layer.is_2d = (layer.na - layer.nf) == ni + 2 * layer.ns

        # Read GFS gate if 2D
        if layer.is_2d:
            wm = WeightMatrix.deserialize(reader, is_training)
            layer.weights.append(wm)

        # Read softmax sub-network if LSTMSoftmax/LSTMBinarySoftmax
        if type_name in ("LSTMSoftmax", "LSTMBinarySoftmax"):
            layer.softmax = deserialize_network(reader)
            if layer.softmax is None:
                return None

    elif type_name in ("Convolve",):
        layer.half_x = reader.read_int32()
        layer.half_y = reader.read_int32()

    elif type_name in ("Maxpool", "Reconfig"):
        layer.x_scale = reader.read_int32()
        layer.y_scale = reader.read_int32()

    elif type_name == "Input":
        # StaticShape: batch, height, width, depth
        batch = reader.read_int32()
        height = reader.read_int32()
        width = reader.read_int32()
        depth = reader.read_int32()
        layer.input_shape = (batch, height, width, depth)

    elif type_name in ("Softmax", "SoftmaxNoCTC", "Logistic", "LinLogistic",
                       "LinTanh", "Tanh", "Relu", "Linear",
                       "PosClip", "SymClip"):
        wm = WeightMatrix.deserialize(reader, is_training)
        layer.weights.append(wm)

    return layer


def serialize_network(layer: NetworkLayer, writer: TFileWriter,
                      training: bool = False) -> None:
    """Write a complete network to binary data (recursively)."""
    # Common header
    writer.write_int8(0)  # NT_NONE = new format
    writer.write_string(layer.type_name)
    writer.write_int8(layer.training)
    writer.write_int8(1 if layer.needs_backprop else 0)
    writer.write_int32(layer.network_flags)
    writer.write_int32(layer.ni)
    writer.write_int32(layer.no)
    writer.write_int32(layer.num_weights)
    writer.write_string(layer.name)

    # Layer-specific data
    if layer.is_plumbing:
        writer.write_uint32(len(layer.children))
        for child in layer.children:
            serialize_network(child, writer, training)
        if layer.network_flags & NF_LAYER_SPECIFIC_LR:
            writer.write_uint32(len(layer.learning_rates))
            for lr in layer.learning_rates:
                writer.write_float(lr)

    elif layer.is_lstm:
        writer.write_int32(layer.na)
        # Write gates: CI, GI, GF1, GO, [GFS]
        for wm in layer.weights:
            wm.serialize(writer, training)
        if layer.softmax is not None:
            serialize_network(layer.softmax, writer, training)

    elif layer.type_name == "Convolve":
        writer.write_int32(layer.half_x)
        writer.write_int32(layer.half_y)

    elif layer.type_name in ("Maxpool", "Reconfig"):
        writer.write_int32(layer.x_scale)
        writer.write_int32(layer.y_scale)

    elif layer.type_name == "Input":
        # Always write StaticShape (4 int32)
        if layer.input_shape:
            for v in layer.input_shape:
                writer.write_int32(v)
        else:
            # Default: batch=1, height=0, width=0, depth=ni
            writer.write_int32(1)
            writer.write_int32(0)
            writer.write_int32(0)
            writer.write_int32(layer.ni)

    elif layer.type_name in ("Softmax", "SoftmaxNoCTC", "Logistic", "LinLogistic",
                             "LinTanh", "Tanh", "Relu", "Linear"):
        for wm in layer.weights:
            wm.serialize(writer, training)


@dataclass
class LSTMModel:
    """Complete LSTM model as stored in TESSDATA_LSTM component."""
    network: NetworkLayer
    network_str: str
    training_flags: int
    training_iteration: int
    sample_iteration: int
    null_char: int
    adam_beta: float
    learning_rate: float
    momentum: float


def deserialize_lstm_component(reader: TFileReader) -> LSTMModel:
    """Read the complete LSTM component from TESSDATA_LSTM slot."""
    network = deserialize_network(reader)
    if network is None:
        raise ValueError("Failed to deserialize network")

    network_str = reader.read_string()
    training_flags = reader.read_int32()
    training_iteration = reader.read_int32()
    sample_iteration = reader.read_int32()
    null_char = reader.read_int32()
    adam_beta = reader.read_float()
    learning_rate = reader.read_float()
    momentum = reader.read_float()

    return LSTMModel(
        network=network,
        network_str=network_str,
        training_flags=training_flags,
        training_iteration=training_iteration,
        sample_iteration=sample_iteration,
        null_char=null_char,
        adam_beta=adam_beta,
        learning_rate=learning_rate,
        momentum=momentum,
    )


def serialize_lstm_component(model: LSTMModel, writer: TFileWriter,
                             training: bool = False) -> None:
    """Write the complete LSTM component for TESSDATA_LSTM slot."""
    serialize_network(model.network, writer, training)
    writer.write_string(model.network_str)
    writer.write_int32(model.training_flags)
    writer.write_int32(model.training_iteration)
    writer.write_int32(model.sample_iteration)
    writer.write_int32(model.null_char)
    writer.write_float(model.adam_beta)
    writer.write_float(model.learning_rate)
    writer.write_float(model.momentum)


def count_weights(layer: NetworkLayer) -> int:
    """Recursively count total weights in a network."""
    total = 0
    for wm in layer.weights:
        total += wm.dim1 * wm.dim2
    if layer.softmax:
        total += count_weights(layer.softmax)
    for child in layer.children:
        total += count_weights(child)
    return total
