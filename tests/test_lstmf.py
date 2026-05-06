"""Tests for LSTMF training data format reading."""

import struct
import pytest

from tesseract_cuda.formats.tfile import TFileWriter
from tesseract_cuda.formats.lstmf import read_lstmf, ImageData


def _build_image_data_bytes(
    filename: str = "test.png",
    page_number: int = 0,
    image_data: bytes = b"\x89PNG\r\n",
    language: str = "eng",
    transcription: str = "hello",
    boxes: list[tuple[int, int, int, int]] | None = None,
    box_texts: list[str] | None = None,
    vertical_text: bool = False,
) -> bytes:
    if boxes is None:
        boxes = [(0, 0, 10, 20)]
    if box_texts is None:
        box_texts = ["h"]

    w = TFileWriter()
    w.write_string(filename)
    w.write_int32(page_number)
    w.write_bytes_vector(image_data)
    w.write_string(language)
    w.write_string(transcription)

    w.write_uint32(len(boxes))
    for x0, y0, x1, y1 in boxes:
        w.write_int32(x0)
        w.write_int32(y0)
        w.write_int32(x1)
        w.write_int32(y1)

    w.write_uint32(len(box_texts))
    for t in box_texts:
        w.write_string(t)

    w.write_int8(1 if vertical_text else 0)
    return w.get_bytes()


def _build_lstmf(pages: list[bytes | None]) -> bytes:
    w = TFileWriter()
    w.write_uint32(len(pages))
    for p in pages:
        if p is None:
            w.write_uint8(0)
        else:
            w.write_uint8(1)
            w.write_bytes(p)
    return w.get_bytes()


class TestReadLstmf:
    def test_single_page(self):
        page = _build_image_data_bytes(
            filename="img1.png",
            transcription="abc",
            image_data=b"\x89PNG\x00\x01\x02",
            boxes=[(0, 0, 10, 20), (10, 0, 20, 20)],
            box_texts=["a", "b"],
        )
        data = _build_lstmf([page])
        pages = read_lstmf(data)
        assert len(pages) == 1
        assert pages[0].imagefilename == "img1.png"
        assert pages[0].transcription == "abc"
        assert pages[0].image_data == b"\x89PNG\x00\x01\x02"
        assert pages[0].language == "eng"
        assert len(pages[0].boxes) == 2
        assert pages[0].boxes[0].x_min == 0
        assert pages[0].boxes[1].x_max == 20
        assert pages[0].box_texts == ["a", "b"]
        assert pages[0].vertical_text is False

    def test_multiple_pages(self):
        p1 = _build_image_data_bytes(filename="a.png", transcription="hello")
        p2 = _build_image_data_bytes(filename="b.png", transcription="world")
        data = _build_lstmf([p1, p2])
        pages = read_lstmf(data)
        assert len(pages) == 2
        assert pages[0].imagefilename == "a.png"
        assert pages[1].imagefilename == "b.png"

    def test_null_page_skipped(self):
        p1 = _build_image_data_bytes(filename="a.png")
        p2 = _build_image_data_bytes(filename="b.png")
        data = _build_lstmf([p1, None, p2])
        pages = read_lstmf(data)
        assert len(pages) == 2

    def test_empty_pages(self):
        data = _build_lstmf([])
        pages = read_lstmf(data)
        assert pages == []

    def test_vertical_text(self):
        page = _build_image_data_bytes(vertical_text=True)
        data = _build_lstmf([page])
        pages = read_lstmf(data)
        assert pages[0].vertical_text is True

    def test_page_number(self):
        page = _build_image_data_bytes(page_number=3)
        data = _build_lstmf([page])
        pages = read_lstmf(data)
        assert pages[0].page_number == 3

    def test_empty_boxes(self):
        page = _build_image_data_bytes(boxes=[], box_texts=[])
        data = _build_lstmf([page])
        pages = read_lstmf(data)
        assert pages[0].boxes == []
        assert pages[0].box_texts == []

    def test_large_image_data(self):
        big_data = bytes(range(256)) * 100
        page = _build_image_data_bytes(image_data=big_data)
        data = _build_lstmf([page])
        pages = read_lstmf(data)
        assert pages[0].image_data == big_data
