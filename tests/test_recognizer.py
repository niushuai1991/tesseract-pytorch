import pytest
import numpy as np
import torch
from PIL import Image

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("tesseract_cuda"),
    reason="tesseract_cuda not available",
)

ENG_TRAINEDDATA = "/tmp/eng.traineddata"


@pytest.fixture
def recognizer():
    from tesseract_cuda.recognizer import TesseractRecognizer
    return TesseractRecognizer(ENG_TRAINEDDATA)


def test_recognizer_load(recognizer):
    assert recognizer.model is not None
    assert recognizer.unicharset is not None
    assert recognizer.recoder is not None
    assert recognizer.unicharset.size > 0
    assert recognizer.recoder.code_range > 0


def test_preprocess_shape(recognizer):
    img = Image.fromarray(np.full((48, 300), 128, dtype=np.uint8), mode='L')
    x = recognizer.preprocess_image(img)
    assert x.dim() == 3
    assert x.shape[0] == 36
    assert x.shape[2] == 1
    assert x.shape[1] > 0


def test_preprocess_normalization(recognizer):
    img = Image.fromarray(np.full((36, 100), 128, dtype=np.uint8), mode='L')
    x = recognizer.preprocess_image(img)
    assert x.dtype == torch.float32
    assert x.min() >= -1.0
    assert x.max() <= 1.0


def test_preprocess_rgb_input(recognizer):
    arr = np.zeros((36, 100, 3), dtype=np.uint8)
    arr[:, :, 0] = 128
    img = Image.fromarray(arr, mode='RGB')
    x = recognizer.preprocess_image(img)
    assert x.dim() == 3
    assert x.shape[2] == 1


def test_preprocess_rgba_input(recognizer):
    arr = np.zeros((36, 100, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    img = Image.fromarray(arr, mode='RGBA')
    x = recognizer.preprocess_image(img)
    assert x.dim() == 3
    assert x.shape[2] == 1


def test_preprocess_narrow_image(recognizer):
    img = Image.fromarray(np.full((100, 3), 128, dtype=np.uint8), mode='L')
    x = recognizer.preprocess_image(img, target_height=36)
    assert x.shape[1] >= 1


def test_preprocess_wide_image(recognizer):
    img = Image.fromarray(np.full((36, 2000), 128, dtype=np.uint8), mode='L')
    x = recognizer.preprocess_image(img)
    assert x.shape[1] == 2000


def test_ctc_decode():
    from tesseract_cuda.recognizer import ctc_best_path_decode
    assert ctc_best_path_decode([0, 0, 1, 1, 2, 2, 0, 0], blank=0) == [1, 2]
    assert ctc_best_path_decode([3, 3, 3, 0, 4, 4, 0, 5], blank=0) == [3, 4, 5]
    assert ctc_best_path_decode([0, 0, 0], blank=0) == []
    assert ctc_best_path_decode([1, 1, 2, 2, 1, 1], blank=0) == [1, 2, 1]


def test_ctc_decode_single_char():
    from tesseract_cuda.recognizer import ctc_best_path_decode
    assert ctc_best_path_decode([5], blank=0) == [5]
    assert ctc_best_path_decode([0], blank=0) == []


def test_ctc_decode_all_blank():
    from tesseract_cuda.recognizer import ctc_best_path_decode
    assert ctc_best_path_decode([0, 0, 0, 0], blank=0) == []


def test_ctc_decode_repeated_char():
    from tesseract_cuda.recognizer import ctc_best_path_decode
    assert ctc_best_path_decode([1, 1, 0, 1, 1], blank=0) == [1, 1]


def test_recognize_blank_image(recognizer):
    img = Image.fromarray(np.full((36, 200), 255, dtype=np.uint8), mode='L')
    text = recognizer.recognize(img)
    assert isinstance(text, str)


def test_recognize_black_image(recognizer):
    img = Image.fromarray(np.full((36, 200), 0, dtype=np.uint8), mode='L')
    text = recognizer.recognize(img)
    assert isinstance(text, str)


def test_recognize_random_image(recognizer):
    np.random.seed(42)
    img = Image.fromarray(np.random.randint(0, 256, (36, 200), dtype=np.uint8), mode='L')
    text = recognizer.recognize(img)
    assert isinstance(text, str)


def test_labels_to_text(recognizer):
    assert recognizer.labels_to_text([]) == ""
    assert recognizer.labels_to_text([recognizer.null_char]) == ""
    result = recognizer.labels_to_text([3, 4, 5])
    assert isinstance(result, str)


def test_labels_to_text_with_nulls(recognizer):
    labels = [3, recognizer.null_char, 4, recognizer.null_char, 5]
    result = recognizer.labels_to_text(labels)
    assert isinstance(result, str)


def test_labels_to_text_out_of_range(recognizer):
    result = recognizer.labels_to_text([9999])
    assert isinstance(result, str)


def test_labels_to_text_no_unicharset(recognizer):
    recognizer.unicharset = None
    assert recognizer.labels_to_text([3]) == ""


def test_labels_to_text_no_recoder(recognizer):
    recognizer.recoder = None
    assert recognizer.labels_to_text([3]) == ""


def test_recognize_tensor(recognizer):
    x = torch.randn(1, 50, 1)
    labels = recognizer.recognize_tensor(x)
    assert isinstance(labels, list)
    for l in labels:
        assert isinstance(l, int)
