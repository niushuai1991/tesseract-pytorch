import torch
import numpy as np
from PIL import Image
from .network.model import TessLSTMModel
from .formats.tessdata import TessdataManager, TESSDATA_LSTM, TESSDATA_LSTM_UNICHARSET, TESSDATA_LSTM_RECODER
from .formats.tfile import TFileReader
from .formats.unicharset import Unicharset
from .formats.recoder import Recoder


class TesseractRecognizer:
    def __init__(self, traineddata_path: str):
        mgr = TessdataManager.from_file(traineddata_path)
        self.model = TessLSTMModel.from_traineddata(traineddata_path)
        self.model.eval()

        unicharset_data = mgr.get_component(TESSDATA_LSTM_UNICHARSET)
        if unicharset_data:
            self.unicharset = Unicharset.from_bytes(unicharset_data)
        else:
            unicharset_data = mgr.get_component(TESSDATA_UNICHARSET)
            self.unicharset = Unicharset.from_bytes(unicharset_data) if unicharset_data else None

        recoder_data = mgr.get_component(TESSDATA_LSTM_RECODER)
        if recoder_data:
            self.recoder = Recoder.from_bytes(recoder_data)
        else:
            self.recoder = None

        self.null_char = self.model.null_char

    def preprocess_image(self, image: Image.Image, target_height: int = 36) -> torch.Tensor:
        if image.mode != "L":
            image = image.convert("L")

        orig_w, orig_h = image.size
        scale = target_height / orig_h
        new_w = max(1, int(orig_w * scale))
        image = image.resize((new_w, target_height), Image.LANCZOS)

        pixels = np.array(image, dtype=np.float32)

        black = float(np.percentile(pixels, 5))
        white = float(np.percentile(pixels, 95))
        contrast = max((white - black) / 2.0, 1.0)

        normalized = (pixels - black) / contrast - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)

        tensor = torch.from_numpy(normalized).float()
        tensor = tensor.unsqueeze(-1)
        return tensor

    def recognize_tensor(self, x: torch.Tensor) -> list[int]:
        with torch.no_grad():
            output = self.model(x)

        # output: [H, W, num_classes] — take the last row (after LSTM chain, H should be 1)
        if output.dim() == 3:
            output = output[0]  # [W, num_classes]
        elif output.dim() == 4:
            output = output[0, 0]  # [W, num_classes]

        probs = output.exp()
        best_path = probs.argmax(dim=-1)

        labels = ctc_best_path_decode(best_path.tolist(), self.null_char)
        return labels

    def labels_to_text(self, labels: list[int]) -> str:
        if not self.unicharset or not self.recoder:
            return ""

        chars = []
        for label in labels:
            if label == self.null_char:
                continue
            unichar_id = self.recoder.decode([label])
            if unichar_id >= 0 and unichar_id < self.unicharset.size:
                chars.append(self.unicharset.id_to_unichar(unichar_id))
        return "".join(chars)

    def recognize(self, image: Image.Image) -> str:
        x = self.preprocess_image(image)
        labels = self.recognize_tensor(x)
        return self.labels_to_text(labels)


def ctc_best_path_decode(labels: list[int], blank: int) -> list[int]:
    result = []
    prev = blank
    for label in labels:
        if label != blank and label != prev:
            result.append(label)
        prev = label
    return result
