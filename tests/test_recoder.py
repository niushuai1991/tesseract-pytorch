"""Tests for Recoder (UnicharCompress) binary format."""

import pytest

from tesseract_cuda.formats.recoder import Recoder
from tesseract_cuda.formats.tfile import TFileWriter


def _build_recoder_bytes(entries: list[tuple[bool, list[int]]]) -> bytes:
    writer = TFileWriter()
    writer.write_uint32(len(entries))
    for self_norm, codes in entries:
        writer.write_int8(1 if self_norm else 0)
        writer.write_int32(len(codes))
        for c in codes:
            writer.write_int32(c)
    return writer.get_bytes()


class TestRecoderDeserialize:
    def test_basic(self):
        data = _build_recoder_bytes([
            (True, [0]),
            (True, [1]),
            (True, [2]),
        ])
        r = Recoder.from_bytes(data)
        assert r.num_codes == 3
        assert r.code_range == 3

    def test_multi_code_entries(self):
        data = _build_recoder_bytes([
            (True, [0]),
            (True, [1, 2]),
        ])
        r = Recoder.from_bytes(data)
        assert r.num_codes == 2
        assert r.encode(0) == [0]
        assert r.encode(1) == [1, 2]

    def test_encode_out_of_range(self):
        data = _build_recoder_bytes([(True, [0])])
        r = Recoder.from_bytes(data)
        assert r.encode(-1) == []
        assert r.encode(5) == []

    def test_decode(self):
        data = _build_recoder_bytes([
            (True, [0]),
            (True, [1]),
            (True, [2]),
        ])
        r = Recoder.from_bytes(data)
        assert r.decode([0]) == 0
        assert r.decode([1]) == 1
        assert r.decode([2]) == 2

    def test_decode_not_found(self):
        data = _build_recoder_bytes([(True, [0])])
        r = Recoder.from_bytes(data)
        assert r.decode([99]) == -1

    def test_self_normalized_flag(self):
        data = _build_recoder_bytes([
            (True, [0]),
            (False, [1]),
        ])
        r = Recoder.from_bytes(data)
        assert r._encoder[0][0] is True
        assert r._encoder[1][0] is False


class TestRecoderRoundtrip:
    def test_roundtrip(self):
        entries = [
            (True, [0]),
            (True, [1, 2]),
            (False, [3]),
            (True, [4, 5, 6]),
        ]
        data = _build_recoder_bytes(entries)
        r1 = Recoder.from_bytes(data)
        out = r1.to_bytes()

        r2 = Recoder.from_bytes(out)
        assert r2.num_codes == r1.num_codes
        assert r2.code_range == r1.code_range
        for i in range(r1.num_codes):
            assert r1.encode(i) == r2.encode(i)
            assert r1._encoder[i][0] == r2._encoder[i][0]

    def test_binary_stable(self):
        entries = [
            (True, [0]),
            (False, [1]),
        ]
        data1 = _build_recoder_bytes(entries)
        r = Recoder.from_bytes(data1)
        data2 = r.to_bytes()
        assert data1 == data2


class TestRecoderCodeRange:
    def test_code_range_with_gap(self):
        data = _build_recoder_bytes([
            (True, [0]),
            (True, [5]),
            (True, [10]),
        ])
        r = Recoder.from_bytes(data)
        assert r.code_range == 11

    def test_code_range_single(self):
        data = _build_recoder_bytes([(True, [42])])
        r = Recoder.from_bytes(data)
        assert r.code_range == 43
