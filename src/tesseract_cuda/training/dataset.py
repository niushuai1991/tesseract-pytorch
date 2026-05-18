"""PyTorch Dataset for Tesseract .lstmf training data."""

import numpy as np
import torch
from torch import from_numpy, tensor, cat, stack, long as torch_long
from torch.utils.data import Dataset
from typing import Optional
from io import BytesIO
from PIL import Image, Image as PILImage
import warnings

from ..formats.lstmf import read_lstmf_file
from ..formats.unicharset import Unicharset
from ..formats.recoder import Recoder


def compute_black_white(pixels: np.ndarray) -> tuple[float, float]:
    height, width = pixels.shape[:2]
    y = height // 2
    if pixels.ndim == 2:
        row = pixels[y]
    else:
        row = pixels[y, :, 0]

    mins: list[int] = []
    maxs: list[int] = []
    for x in range(1, width - 1):
        prev, curr, nxt = int(row[x - 1]), int(row[x]), int(row[x + 1])
        if (curr < prev and curr <= nxt) or (curr <= prev and curr < nxt):
            mins.append(curr)
        if (curr > prev and curr >= nxt) or (curr >= prev and curr > nxt):
            maxs.append(curr)

    if not mins:
        mins = [0]
    if not maxs:
        maxs = [255]

    mins.sort()
    maxs.sort()
    black = float(np.percentile(mins, 25))
    white = float(np.percentile(maxs, 75))
    return black, white


def tesseract_normalize(pixels: np.ndarray) -> np.ndarray:
    black, white = compute_black_white(pixels)
    contrast = max((white - black) / 2.0, 1.0)
    normalized = (pixels - black) / contrast - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


class LSTMFDataset(Dataset):
    """Dataset that reads .lstmf files and yields (image_tensor, labels) pairs."""

    def __init__(self, lstmf_files: list[str],
                 unicharset: Unicharset,
                 recoder: Optional[Recoder] = None,
                 target_height: int = 36,
                 null_char_id: int = 0):
        self.unicharset = unicharset
        self.recoder = recoder
        self.target_height = target_height
        self.null_char_id = null_char_id

        # Load all samples
        self.samples = []
        for path in lstmf_files:
            pages = read_lstmf_file(path)
            for page in pages:
                if page.transcription and page.image_data:
                    self.samples.append(page)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        page = self.samples[idx]

        image = Image.open(BytesIO(page.image_data)).convert("L")

        w, h = image.size
        if h != self.target_height:
            new_w = max(1, int(w * self.target_height / h))
            image = image.resize((new_w, self.target_height), PILImage.Resampling.LANCZOS)
            w, h = new_w, self.target_height

        arr = np.array(image, dtype=np.float32)
        arr = tesseract_normalize(arr)
        img_tensor = from_numpy(arr)

        labels = self._encode_transcription(page.transcription)

        labels_tensor = tensor(labels, dtype=torch_long)
        return img_tensor, labels_tensor, w, len(labels)

    def _encode_transcription(self, text: str) -> list[int]:
        if self.recoder:
            char_ids = self.unicharset.encode_string(text)
            labels = []
            for uid in char_ids:
                codes = self.recoder.encode(uid)
                if codes:
                    labels.extend(codes)
                else:
                    warnings.warn(
                        f"Cannot encode unichar id {uid} via recoder, skipping sample",
                        stacklevel=2,
                    )
                    return [self.null_char_id]
            return labels
        else:
            return self.unicharset.encode_string(text)


def collate_fn(batch):
    images, labels, widths, label_lengths = zip(*batch)

    max_w = max(widths)
    h = images[0].shape[0]
    padded_images = []
    for img, w in zip(images, widths):
        if w < max_w:
            pad = torch.rand(h, max_w - w) * 2.0 - 1.0
            padded_images.append(cat([img, pad], dim=1))
        else:
            padded_images.append(img)

    images_tensor = stack(padded_images).unsqueeze(1)

    labels_tensor = cat(labels)
    input_lengths = tensor(widths, dtype=torch_long)
    target_lengths = tensor(label_lengths, dtype=torch_long)

    return images_tensor, labels_tensor, input_lengths, target_lengths
