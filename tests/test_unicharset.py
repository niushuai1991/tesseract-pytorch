"""Tests for Unicharset text format parsing."""

import pytest

from tesseract_cuda.formats.unicharset import Unicharset


SAMPLE_UNICHARSET = """\
4
NULL 0 0 1  Common 0 0 0
a 1 0 1  Latin 0 0 0 a
b 2 0 1  Latin 0 0 0 b
c 4 0 1  Latin 0 0 0 c
"""


class TestUnicharsetParse:
    def test_parse_basic(self):
        ucs = Unicharset.from_text(SAMPLE_UNICHARSET)
        assert ucs.size == 4

    def test_id_to_unichar(self):
        ucs = Unicharset.from_text(SAMPLE_UNICHARSET)
        assert ucs.id_to_unichar(0) == "NULL"
        assert ucs.id_to_unichar(1) == "a"
        assert ucs.id_to_unichar(2) == "b"
        assert ucs.id_to_unichar(3) == "c"

    def test_unichar_to_id(self):
        ucs = Unicharset.from_text(SAMPLE_UNICHARSET)
        assert ucs.unichar_to_id("NULL") == 0
        assert ucs.unichar_to_id("a") == 1
        assert ucs.unichar_to_id("b") == 2

    def test_has_unichar(self):
        ucs = Unicharset.from_text(SAMPLE_UNICHARSET)
        assert ucs.has_unichar("a")
        assert ucs.has_unichar("NULL")
        assert not ucs.has_unichar("z")
        assert not ucs.has_unichar("")

    def test_encode_string(self):
        ucs = Unicharset.from_text(SAMPLE_UNICHARSET)
        ids = ucs.encode_string("abc")
        assert ids == [1, 2, 3]

    def test_encode_string_with_multi_char(self):
        ucs = Unicharset.from_text(SAMPLE_UNICHARSET)
        ids = ucs.encode_string("abc")
        assert ids == [1, 2, 3]

    def test_encode_string_unknown_char_skipped(self):
        ucs = Unicharset.from_text(SAMPLE_UNICHARSET)
        ids = ucs.encode_string("axb")
        assert ids == [1, 2]

    def test_encode_empty_string(self):
        ucs = Unicharset.from_text(SAMPLE_UNICHARSET)
        assert ucs.encode_string("") == []

    def test_from_bytes(self):
        ucs = Unicharset.from_bytes(SAMPLE_UNICHARSET.encode("utf-8"))
        assert ucs.size == 4
        assert ucs.has_unichar("a")

    def test_parse_empty(self):
        ucs = Unicharset.from_text("0\n")
        assert ucs.size == 0

    def test_parse_minimal(self):
        text = "1\nNULL 0 0 1 Common 0 0 0\n"
        ucs = Unicharset.from_text(text)
        assert ucs.size == 1
        assert ucs.id_to_unichar(0) == "NULL"


class TestUnicharsetProperties:
    def test_entry_properties(self):
        ucs = Unicharset.from_text(SAMPLE_UNICHARSET)
        assert ucs._entries[0].unichar == "NULL"
        assert ucs._entries[0].properties == 0
        assert ucs._entries[0].script == "Common"
        assert ucs._entries[1].script == "Latin"

    def test_other_case(self):
        ucs = Unicharset.from_text(SAMPLE_UNICHARSET)
        assert ucs._entries[0].other_case == 0

    def test_direction_and_mirror(self):
        ucs = Unicharset.from_text(SAMPLE_UNICHARSET)
        assert ucs._entries[1].direction == 0
        assert ucs._entries[1].mirror == 0

    def test_normed(self):
        ucs = Unicharset.from_text(SAMPLE_UNICHARSET)
        assert ucs._entries[1].normed == "a"
        assert ucs._entries[2].normed == "b"
