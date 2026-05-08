"""Tests for real eng.traineddata file."""

import pytest

from tesseract_cuda.formats.tessdata import TessdataManager, TESSDATA_LSTM
from tesseract_cuda.formats.tfile import TFileReader, TFileWriter
from tesseract_cuda.formats.network_ser import (
    deserialize_network, serialize_network,
    deserialize_lstm_component, serialize_lstm_component,
    count_weights,
)
from tesseract_cuda.formats.unicharset import Unicharset
from tesseract_cuda.formats.recoder import Recoder
from tesseract_cuda.network.spec_parser import parse_network_spec
from tesseract_cuda.network.model import TessLSTMModel


ENG_TRAINEDDATA = "/tmp/eng.traineddata"


def test_read_eng_traineddata():
    """Test reading the eng.traineddata file."""
    mgr = TessdataManager.from_file(ENG_TRAINEDDATA)
    components = mgr.list_components()
    assert len(components) > 0
    assert any(idx == TESSDATA_LSTM for idx, _, _ in components)


def test_deserialize_lstm_component():
    """Test deserializing the LSTM component from eng.traineddata."""
    mgr = TessdataManager.from_file(ENG_TRAINEDDATA)
    lstm_data = mgr.get_component(TESSDATA_LSTM)
    assert len(lstm_data) > 0

    reader = TFileReader(lstm_data)
    model = deserialize_lstm_component(reader)
    assert model is not None
    assert model.network_str
    assert model.training_iteration >= 0
    assert model.null_char >= 0


def test_network_structure():
    """Test that the network structure is correctly deserialized."""
    mgr = TessdataManager.from_file(ENG_TRAINEDDATA)
    lstm_data = mgr.get_component(TESSDATA_LSTM)
    reader = TFileReader(lstm_data)
    model = deserialize_lstm_component(reader)

    network = model.network
    assert network is not None
    assert network.type_name == "Series"


def test_count_weights_real():
    """Test counting weights in the real network."""
    mgr = TessdataManager.from_file(ENG_TRAINEDDATA)
    lstm_data = mgr.get_component(TESSDATA_LSTM)
    reader = TFileReader(lstm_data)
    model = deserialize_lstm_component(reader)

    total = count_weights(model.network)
    assert total > 0


def test_roundtrip_lstm_component():
    """Test that serializing and deserializing preserves the data."""
    mgr = TessdataManager.from_file(ENG_TRAINEDDATA)
    lstm_data = mgr.get_component(TESSDATA_LSTM)

    reader = TFileReader(lstm_data)
    model1 = deserialize_lstm_component(reader)

    writer = TFileWriter()
    serialize_lstm_component(model1, writer, training=False)
    serialized = writer.get_bytes()

    reader2 = TFileReader(serialized)
    model2 = deserialize_lstm_component(reader2)

    assert model1.network_str == model2.network_str
    assert model1.training_flags == model2.training_flags
    assert model1.training_iteration == model2.training_iteration
    assert model1.null_char == model2.null_char
    assert count_weights(model1.network) == count_weights(model2.network)


def test_parse_network_spec():
    """Test parsing the network spec from eng.traineddata."""
    mgr = TessdataManager.from_file(ENG_TRAINEDDATA)
    lstm_data = mgr.get_component(TESSDATA_LSTM)
    reader = TFileReader(lstm_data)
    model = deserialize_lstm_component(reader)

    spec = model.network_str
    desc = parse_network_spec(spec)
    assert desc is not None
    assert desc.type == "series"


def test_unicharset_component():
    """Test reading the unicharset component."""
    from tesseract_cuda.formats.tessdata import TESSDATA_LSTM_UNICHARSET

    mgr = TessdataManager.from_file(ENG_TRAINEDDATA)
    unicharset_data = mgr.get_component(TESSDATA_LSTM_UNICHARSET)
    if unicharset_data:
        ucs = Unicharset.from_bytes(unicharset_data)
        assert ucs.size > 0


def test_recoder_component():
    """Test reading the recoder component."""
    from tesseract_cuda.formats.tessdata import TESSDATA_LSTM_RECODER

    mgr = TessdataManager.from_file(ENG_TRAINEDDATA)
    recoder_data = mgr.get_component(TESSDATA_LSTM_RECODER)
    if recoder_data:
        recoder = Recoder.from_bytes(recoder_data)
        assert recoder.num_codes > 0
        assert recoder.code_range > 0


def test_load_model_from_traineddata():
    """Test loading a PyTorch model from eng.traineddata."""
    model = TessLSTMModel.from_traineddata(ENG_TRAINEDDATA)
    assert model is not None
    assert model.network is not None
    assert model.network_str
    assert model.null_char >= 0


def test_model_forward_pass():
    """Test running a forward pass with the loaded model."""
    import torch

    model = TessLSTMModel.from_traineddata(ENG_TRAINEDDATA)

    # Real network: [1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]
    # Input is [height=1, width=N, depth=1] (spec says height=36 but
    # the network was built with ni=1 as the depth)
    width = 20
    x = torch.randn(1, width, 1)
    output = model(x)

    assert output.dim() == 3
    assert output.shape[0] == 1
    assert output.shape[2] == 111


def test_binary_stability():
    """Test that round-trip produces identical binary."""
    mgr = TessdataManager.from_file(ENG_TRAINEDDATA)
    lstm_data = mgr.get_component(TESSDATA_LSTM)

    reader = TFileReader(lstm_data)
    model1 = deserialize_lstm_component(reader)

    writer = TFileWriter()
    serialize_lstm_component(model1, writer, training=False)
    serialized1 = writer.get_bytes()

    # Deserialize again and serialize again
    reader2 = TFileReader(serialized1)
    model2 = deserialize_lstm_component(reader2)
    writer2 = TFileWriter()
    serialize_lstm_component(model2, writer2, training=False)
    serialized2 = writer2.get_bytes()

    # Binary should be identical after first round-trip
    assert serialized1 == serialized2
