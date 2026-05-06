"""Tests for TessdataManager traineddata container."""

import pytest
from tesseract_cuda.formats.tessdata import (
    TessdataManager, TESSDATA_LSTM, TESSDATA_LSTM_UNICHARSET,
    TESSDATA_LSTM_RECODER, TESSDATA_VERSION, TESSDATA_NUM_ENTRIES,
)
from tesseract_cuda.formats.tfile import TFileWriter


def _make_traineddata(entries: dict[int, bytes]) -> bytes:
    """Build a minimal traineddata binary with given components."""
    mgr = TessdataManager()
    for idx, data in entries.items():
        mgr.set_component(idx, data)
    return mgr.to_bytes()


class TestTessdataManager:
    def test_empty_container(self):
        mgr = TessdataManager()
        data = mgr.to_bytes()
        assert len(data) == 4 + 8 * TESSDATA_NUM_ENTRIES  # header only

        mgr2 = TessdataManager.from_bytes(data)
        assert not mgr2.has_component(TESSDATA_LSTM)

    def test_set_get_component(self):
        mgr = TessdataManager()
        mgr.set_component(TESSDATA_LSTM, b"lstm_data_here")
        mgr.set_component(TESSDATA_VERSION, b"5.0.0")

        assert mgr.get_component(TESSDATA_LSTM) == b"lstm_data_here"
        assert mgr.get_component(TESSDATA_VERSION) == b"5.0.0"
        assert mgr.get_component(TESSDATA_LSTM_UNICHARSET) == b""

    def test_round_trip(self):
        mgr = TessdataManager()
        mgr.set_component(TESSDATA_LSTM, b"\x00\x01\x02\x03" * 100)
        mgr.set_component(TESSDATA_LSTM_UNICHARSET, b"unicharset_data")
        mgr.set_component(TESSDATA_LSTM_RECODER, b"\xff\xfe\xfd")
        mgr.set_component(TESSDATA_VERSION, b"5.0.0-alpha")

        data = mgr.to_bytes()
        mgr2 = TessdataManager.from_bytes(data)

        assert mgr2.get_component(TESSDATA_LSTM) == b"\x00\x01\x02\x03" * 100
        assert mgr2.get_component(TESSDATA_LSTM_UNICHARSET) == b"unicharset_data"
        assert mgr2.get_component(TESSDATA_LSTM_RECODER) == b"\xff\xfe\xfd"
        assert mgr2.get_component(TESSDATA_VERSION) == b"5.0.0-alpha"

    def test_overwrite_component(self):
        mgr = TessdataManager()
        mgr.set_component(TESSDATA_LSTM, b"old_data")
        mgr.set_component(TESSDATA_LSTM, b"new_data")

        data = mgr.to_bytes()
        mgr2 = TessdataManager.from_bytes(data)
        assert mgr2.get_component(TESSDATA_LSTM) == b"new_data"

    def test_remove_component(self):
        mgr = TessdataManager()
        mgr.set_component(TESSDATA_LSTM, b"data")
        mgr.remove_component(TESSDATA_LSTM)
        assert not mgr.has_component(TESSDATA_LSTM)

    def test_list_components(self):
        mgr = TessdataManager()
        mgr.set_component(TESSDATA_LSTM, b"x" * 100)
        mgr.set_component(TESSDATA_VERSION, b"5.0.0")

        components = mgr.list_components()
        assert len(components) == 2
        idx_17 = [c for c in components if c[0] == TESSDATA_LSTM]
        assert len(idx_17) == 1
        assert idx_17[0][2] == 100  # size

    def test_header_format(self):
        """Verify the binary header is correctly structured."""
        mgr = TessdataManager()
        mgr.set_component(TESSDATA_LSTM, b"lstm")
        mgr.set_component(TESSDATA_VERSION, b"5.0")

        data = mgr.to_bytes()
        import struct

        num_entries = struct.unpack_from('<I', data, 0)[0]
        assert num_entries == TESSDATA_NUM_ENTRIES

        offset_table = []
        for i in range(TESSDATA_NUM_ENTRIES):
            off = struct.unpack_from('<q', data, 4 + i * 8)[0]
            offset_table.append(off)

        # LSTM (idx 17) and VERSION (idx 23) should have positive offsets
        assert offset_table[TESSDATA_LSTM] > 0
        assert offset_table[TESSDATA_VERSION] > 0
        # Others should be -1
        assert offset_table[0] == -1

    def test_multiple_round_trips(self):
        """Verify multiple serialize/deserialize cycles are identical."""
        mgr = TessdataManager()
        mgr.set_component(TESSDATA_LSTM, b"test_data_123")
        mgr.set_component(TESSDATA_VERSION, b"1.0")

        data1 = mgr.to_bytes()
        mgr2 = TessdataManager.from_bytes(data1)
        data2 = mgr2.to_bytes()

        assert data1 == data2
