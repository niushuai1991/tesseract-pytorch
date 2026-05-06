"""Read Tesseract .lstmf training data files."""

from dataclasses import dataclass
from typing import Optional

from .tfile import TFileReader


@dataclass
class TBOX:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass
class ImageData:
    imagefilename: str
    page_number: int
    image_data: bytes
    language: str
    transcription: str
    boxes: list[TBOX]
    box_texts: list[str]
    vertical_text: bool


def read_lstmf(data: bytes) -> list[ImageData]:
    """Read all pages from .lstmf binary data."""
    reader = TFileReader(data)
    total_pages = reader.read_uint32()
    pages = []
    for _ in range(total_pages):
        non_null = reader.read_uint8()
        if not non_null:
            pages.append(None)
            continue
        pages.append(_read_image_data(reader))
    return [p for p in pages if p is not None]


def read_lstmf_file(path: str) -> list[ImageData]:
    """Read all pages from a .lstmf file."""
    with open(path, "rb") as f:
        data = f.read()
    return read_lstmf(data)


def iter_lstmf_pages(data: bytes):
    """Iterate over pages in .lstmf binary data without loading all at once."""
    reader = TFileReader(data)
    total_pages = reader.read_uint32()
    for _ in range(total_pages):
        non_null = reader.read_uint8()
        if not non_null:
            continue
        yield _read_image_data(reader)


def _read_image_data(reader: TFileReader) -> ImageData:
    imagefilename = reader.read_string()
    page_number = reader.read_int32()
    image_data = reader.read_bytes_vector()
    language = reader.read_string()
    transcription = reader.read_string()

    # boxes
    box_count = reader.read_uint32()
    boxes = []
    for _ in range(box_count):
        x_min = reader.read_int32()
        y_min = reader.read_int32()
        x_max = reader.read_int32()
        y_max = reader.read_int32()
        boxes.append(TBOX(x_min, y_min, x_max, y_max))

    # box_texts
    text_count = reader.read_uint32()
    box_texts = []
    for _ in range(text_count):
        box_texts.append(reader.read_string())

    vertical_text = reader.read_int8() != 0

    return ImageData(
        imagefilename=imagefilename,
        page_number=page_number,
        image_data=image_data,
        language=language,
        transcription=transcription,
        boxes=boxes,
        box_texts=box_texts,
        vertical_text=vertical_text,
    )
