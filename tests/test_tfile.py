"""Tests for TFile binary serialization primitives."""

import struct
import pytest

from tesseract_cuda.formats.tfile import TFileReader, TFileWriter


class TestTFileWriter:
    def test_write_read_int8(self):
        w = TFileWriter()
        w.write_int8(-42)
        r = TFileReader(w.get_bytes())
        assert r.read_int8() == -42

    def test_write_read_uint8(self):
        w = TFileWriter()
        w.write_uint8(200)
        r = TFileReader(w.get_bytes())
        assert r.read_uint8() == 200

    def test_write_read_int32(self):
        w = TFileWriter()
        w.write_int32(-123456)
        r = TFileReader(w.get_bytes())
        assert r.read_int32() == -123456

    def test_write_read_uint32(self):
        w = TFileWriter()
        w.write_uint32(3000000000)
        r = TFileReader(w.get_bytes())
        assert r.read_uint32() == 3000000000

    def test_write_read_int64(self):
        w = TFileWriter()
        w.write_int64(-9876543210)
        r = TFileReader(w.get_bytes())
        assert r.read_int64() == -9876543210

    def test_write_read_float(self):
        w = TFileWriter()
        w.write_float(3.14)
        r = TFileReader(w.get_bytes())
        assert r.read_float() == pytest.approx(3.14, abs=1e-5)

    def test_write_read_double(self):
        w = TFileWriter()
        w.write_double(2.718281828)
        r = TFileReader(w.get_bytes())
        assert r.read_double() == pytest.approx(2.718281828, abs=1e-10)

    def test_write_read_string(self):
        w = TFileWriter()
        w.write_string("Hello, Tesseract!")
        r = TFileReader(w.get_bytes())
        assert r.read_string() == "Hello, Tesseract!"

    def test_write_read_empty_string(self):
        w = TFileWriter()
        w.write_string("")
        r = TFileReader(w.get_bytes())
        assert r.read_string() == ""

    def test_write_read_utf8_string(self):
        w = TFileWriter()
        w.write_string("中文テスト한국어")
        r = TFileReader(w.get_bytes())
        assert r.read_string() == "中文テスト한국어"

    def test_write_read_bytes_vector(self):
        w = TFileWriter()
        w.write_bytes_vector(b"\x00\x01\x02\xff")
        r = TFileReader(w.get_bytes())
        assert r.read_bytes_vector() == b"\x00\x01\x02\xff"

    def test_write_read_empty_bytes(self):
        w = TFileWriter()
        w.write_bytes_vector(b"")
        r = TFileReader(w.get_bytes())
        assert r.read_bytes_vector() == b""


class TestTFileReader:
    def test_read_mixed_types(self):
        w = TFileWriter()
        w.write_uint32(42)
        w.write_string("test")
        w.write_double(1.5)
        w.write_int8(-1)

        r = TFileReader(w.get_bytes())
        assert r.read_uint32() == 42
        assert r.read_string() == "test"
        assert r.read_double() == pytest.approx(1.5)
        assert r.read_int8() == -1

    def test_read_2d_double_array(self):
        w = TFileWriter()
        # dim1=2, dim2=3, empty=0.0, data=[1,2,3,4,5,6]
        w.write_2d_double_array(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        r = TFileReader(w.get_bytes())
        d1, d2, data = r.read_2d_double_array()
        assert d1 == 2
        assert d2 == 3
        assert data == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_read_int32_vector(self):
        w = TFileWriter()
        w.write_int32_vector([10, 20, 30])
        r = TFileReader(w.get_bytes())
        assert r.read_int32_vector() == [10, 20, 30]

    def test_skip(self):
        w = TFileWriter()
        w.write_int32(1)
        w.write_int32(2)
        w.write_int32(3)

        r = TFileReader(w.get_bytes())
        assert r.read_int32() == 1
        r.skip(4)  # skip second int32
        assert r.read_int32() == 3

    def test_remaining(self):
        w = TFileWriter()
        w.write_int32(42)
        r = TFileReader(w.get_bytes())
        assert r.remaining() == 4
        r.read_int32()
        assert r.remaining() == 0

    def test_eof_error(self):
        w = TFileWriter()
        w.write_int8(1)
        r = TFileReader(w.get_bytes())
        with pytest.raises(EOFError):
            r.read_int32()  # only 1 byte available


class TestRoundTrip:
    def test_full_round_trip(self):
        """Write mixed data, read it back, verify all values match."""
        w = TFileWriter()
        w.write_uint32(24)
        w.write_int64(100)
        w.write_int64(-1)
        w.write_int64(200)
        w.write_string("LSTM")
        w.write_double(0.001)
        w.write_float(0.5)
        w.write_int32(10000)
        w.write_int8(0)

        data = w.get_bytes()
        r = TFileReader(data)

        assert r.read_uint32() == 24
        assert r.read_int64() == 100
        assert r.read_int64() == -1
        assert r.read_int64() == 200
        assert r.read_string() == "LSTM"
        assert r.read_double() == pytest.approx(0.001)
        assert r.read_float() == pytest.approx(0.5, abs=1e-5)
        assert r.read_int32() == 10000
        assert r.read_int8() == 0
        assert r.remaining() == 0
