import pytest
import numpy as np
import torch
from io import BytesIO
from PIL import Image

from tesseract_cuda.formats.lstmf import ImageData, TBOX
from tesseract_cuda.formats.tfile import TFileWriter
from tesseract_cuda.formats.unicharset import Unicharset
from tesseract_cuda.formats.recoder import Recoder
from tesseract_cuda.training.dataset import LSTMFDataset, collate_fn


def _make_image_bytes(h=36, w=100):
    img = Image.new("L", (w, h), color=255)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_unicharset():
    return Unicharset.from_bytes(b"2\nNULL 0 0 0\na 1 0 0\n")


def _make_recoder():
    writer = TFileWriter()
    writer.write_uint32(2)
    writer.write_int8(0)
    writer.write_int32(1)
    writer.write_int32(0)
    writer.write_int8(0)
    writer.write_int32(1)
    writer.write_int32(1)
    return Recoder.from_bytes(writer.get_bytes())


def _write_image_data(writer: TFileWriter, page: ImageData):
    writer.write_string(page.imagefilename)
    writer.write_int32(page.page_number)
    writer.write_bytes_vector(page.image_data)
    writer.write_string(page.language)
    writer.write_string(page.transcription)
    writer.write_uint32(len(page.boxes))
    for box in page.boxes:
        writer.write_int32(box.x_min)
        writer.write_int32(box.y_min)
        writer.write_int32(box.x_max)
        writer.write_int32(box.y_max)
    writer.write_uint32(len(page.box_texts))
    for t in page.box_texts:
        writer.write_string(t)
    writer.write_int8(1 if page.vertical_text else 0)


def _write_lstmf(pages, path):
    non_null = [p for p in pages if p is not None]
    writer = TFileWriter()
    writer.write_uint32(len(pages))
    for p in pages:
        if p is None:
            writer.write_uint8(0)
        else:
            writer.write_uint8(1)
            _write_image_data(writer, p)
    with open(path, "wb") as f:
        f.write(writer.get_bytes())


class TestEncodeTranscription:
    def test_without_recoder(self):
        ds = LSTMFDataset.__new__(LSTMFDataset)
        ds.unicharset = _make_unicharset()
        ds.recoder = None
        ds.null_char_id = 0
        assert ds._encode_transcription("a") == [1]

    def test_unknown_char_skipped(self):
        ds = LSTMFDataset.__new__(LSTMFDataset)
        ds.unicharset = _make_unicharset()
        ds.recoder = None
        ds.null_char_id = 0
        assert ds._encode_transcription("xyz") == []

    def test_with_recoder(self):
        ds = LSTMFDataset.__new__(LSTMFDataset)
        ds.unicharset = _make_unicharset()
        ds.recoder = _make_recoder()
        ds.null_char_id = 0
        result = ds._encode_transcription("a")
        assert result == [1]

    def test_empty_string(self):
        ds = LSTMFDataset.__new__(LSTMFDataset)
        ds.unicharset = _make_unicharset()
        ds.recoder = None
        ds.null_char_id = 0
        assert ds._encode_transcription("") == []


class TestCollateFn:
    def test_single_item(self):
        img = torch.randn(36, 100)
        labels = torch.tensor([1, 2, 3])
        batch = [(img, labels, 100, 3)]
        imgs, labs, ilens, tlens = collate_fn(batch)
        assert imgs.shape == (1, 1, 36, 100)
        assert labs.shape == (3,)
        assert ilens.tolist() == [100]
        assert tlens.tolist() == [3]

    def test_padding(self):
        img1 = torch.randn(36, 100)
        img2 = torch.randn(36, 50)
        lab1 = torch.tensor([1, 2])
        lab2 = torch.tensor([3])
        batch = [(img1, lab1, 100, 2), (img2, lab2, 50, 1)]
        imgs, labs, ilens, tlens = collate_fn(batch)
        assert imgs.shape == (2, 1, 36, 100)
        assert labs.shape == (3,)
        assert ilens.tolist() == [100, 50]
        assert tlens.tolist() == [2, 1]

    def test_three_items(self):
        items = [
            (torch.randn(36, 80), torch.tensor([1]), 80, 1),
            (torch.randn(36, 60), torch.tensor([2, 3]), 60, 2),
            (torch.randn(36, 100), torch.tensor([4]), 100, 1),
        ]
        imgs, labs, ilens, tlens = collate_fn(items)
        assert imgs.shape == (3, 1, 36, 100)
        assert labs.tolist() == [1, 2, 3, 4]
        assert ilens.tolist() == [80, 60, 100]
        assert tlens.tolist() == [1, 2, 1]


def _make_page(h=36, w=100, text="a"):
    return ImageData(
        imagefilename="test.png",
        page_number=0,
        image_data=_make_image_bytes(h, w),
        language="eng",
        transcription=text,
        boxes=[],
        box_texts=[],
        vertical_text=False,
    )


class TestLSTMFDataset:
    def test_load_and_length(self, tmp_path):
        pages = [_make_page(text="a"), _make_page(text="a")]
        _write_lstmf(pages, str(tmp_path / "test.lstmf"))
        ds = LSTMFDataset([str(tmp_path / "test.lstmf")], _make_unicharset())
        assert len(ds) == 2

    def test_getitem_shapes(self, tmp_path):
        pages = [_make_page(36, 100, "a")]
        _write_lstmf(pages, str(tmp_path / "test.lstmf"))
        ds = LSTMFDataset([str(tmp_path / "test.lstmf")], _make_unicharset(), target_height=36)
        img, labels, w, lab_len = ds[0]
        assert img.shape == (36, 100)
        assert labels.dtype == torch.long
        assert w == 100
        assert lab_len >= 1

    def test_empty_transcription_filtered(self, tmp_path):
        pages = [_make_page(text=""), _make_page(text="a")]
        _write_lstmf(pages, str(tmp_path / "test.lstmf"))
        ds = LSTMFDataset([str(tmp_path / "test.lstmf")], _make_unicharset())
        assert len(ds) == 1

    def test_no_image_filtered(self, tmp_path):
        writer = TFileWriter()
        writer.write_uint32(2)
        writer.write_uint8(0)  # null page
        writer.write_uint8(1)
        page = _make_page(text="a")
        _write_image_data(writer, page)
        p = tmp_path / "test.lstmf"
        with open(str(p), "wb") as f:
            f.write(writer.get_bytes())
        ds = LSTMFDataset([str(p)], _make_unicharset())
        assert len(ds) == 1

    def test_resizing(self, tmp_path):
        pages = [_make_page(72, 200, "a")]
        _write_lstmf(pages, str(tmp_path / "test.lstmf"))
        ds = LSTMFDataset([str(tmp_path / "test.lstmf")], _make_unicharset(), target_height=36)
        img, _, w, _ = ds[0]
        assert img.shape[0] == 36
        assert w == 100

    def test_with_recoder(self, tmp_path):
        pages = [_make_page(text="a")]
        _write_lstmf(pages, str(tmp_path / "test.lstmf"))
        ds = LSTMFDataset([str(tmp_path / "test.lstmf")], _make_unicharset(), recoder=_make_recoder())
        _, labels, _, _ = ds[0]
        assert labels.shape[0] >= 1
