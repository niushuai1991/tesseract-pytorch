import pytest
import torch
import numpy as np
from io import BytesIO
from PIL import Image

from tesseract_cuda.network.model import TessLSTMModel
from tesseract_cuda.training.trainer import LSTMTrainer
from tesseract_cuda.training.dataset import LSTMFDataset
from tesseract_cuda.formats.lstmf import ImageData
from tesseract_cuda.formats.tfile import TFileWriter
from tesseract_cuda.formats.unicharset import Unicharset


def _make_unicharset():
    return Unicharset.from_bytes(b"3\nNULL 0 0 0\na 1 0 0\nb 2 0 0\n")


def _make_image_bytes(h=36, w=100):
    img = Image.new("L", (w, h), color=128)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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


def _make_dataset(tmp_path):
    page = ImageData("test.png", 0, _make_image_bytes(h=1, w=10), "eng", "ab", [], [], False)
    writer = TFileWriter()
    writer.write_uint32(1)
    writer.write_uint8(1)
    _write_image_data(writer, page)
    p = tmp_path / "train.lstmf"
    with open(str(p), "wb") as f:
        f.write(writer.get_bytes())
    return LSTMFDataset([str(p)], _make_unicharset(), target_height=1)


class TestLSTMTrainer:
    def test_init(self):
        model = TessLSTMModel.from_spec("[1,0,0,1Lfx16O1c3]", num_classes=3)
        model.null_char = 0
        trainer = LSTMTrainer(model, device="cpu")
        assert trainer.training_iteration == 0
        assert trainer.sample_iteration == 0

    def test_optimizer_created(self):
        model = TessLSTMModel.from_spec("[1,0,0,1Lfx16O1c3]", num_classes=3)
        model.null_char = 0
        trainer = LSTMTrainer(model, device="cpu")
        assert trainer.optimizer is not None

    def test_criterion_blank(self):
        model = TessLSTMModel.from_spec("[1,0,0,1Lfx16O1c3]", num_classes=3)
        model.null_char = 0
        trainer = LSTMTrainer(model, device="cpu")
        assert trainer.criterion.blank == 0

    def test_train_one_iteration(self, tmp_path):
        model = TessLSTMModel.from_spec("[1,0,0,1Lfx16O1c3]", num_classes=3)
        model.null_char = 0
        trainer = LSTMTrainer(model, device="cpu", max_iterations=1)
        ds = _make_dataset(tmp_path)
        trainer.train(ds)
        assert trainer.training_iteration >= 1

    def test_train_multiple_iterations(self, tmp_path):
        model = TessLSTMModel.from_spec("[1,0,0,1Lfx16O1c3]", num_classes=3)
        model.null_char = 0
        torch.manual_seed(42)
        trainer = LSTMTrainer(model, device="cpu", max_iterations=5)
        ds = _make_dataset(tmp_path)
        trainer.train(ds)
        assert trainer.training_iteration == 5

    def test_model_on_device(self):
        model = TessLSTMModel.from_spec("[1,0,0,1Lfx16O1c3]", num_classes=3)
        model.null_char = 0
        trainer = LSTMTrainer(model, device="cpu")
        assert next(model.parameters()).device.type == "cpu"

    def test_checkpoint_save(self, tmp_path):
        model = TessLSTMModel.from_spec("[1,0,0,1Lfx16O1c3]", num_classes=3)
        model.null_char = 0
        out = str(tmp_path / "model")
        trainer = LSTMTrainer(model, device="cpu", max_iterations=1,
                              model_output=out, checkpoint_interval=1)
        ds = _make_dataset(tmp_path)
        trainer.train(ds)
        ckpt_path = f"{out}_checkpoint.pt"
        assert torch.load(ckpt_path, weights_only=False)["training_iteration"] >= 1
