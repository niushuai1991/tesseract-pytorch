"""PyTorch Dataset for Tesseract .lstmf training data."""

import numpy as np
import torch
from torch import from_numpy, tensor, zeros, cat, stack, long as torch_long
from torch.utils.data import Dataset
from typing import Optional
from io import BytesIO
from PIL import Image, Image as PILImage

from ..formats.lstmf import read_lstmf_file
from ..formats.unicharset import Unicharset
from ..formats.recoder import Recoder


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

        # Decode image
        image = Image.open(BytesIO(page.image_data)).convert("L")

        # Scale to target height
        w, h = image.size
        if h != self.target_height:
            new_w = max(1, int(w * self.target_height / h))
            image = image.resize((new_w, self.target_height), PILImage.Resampling.BILINEAR)
            w, h = new_w, self.target_height

        # Convert to tensor, normalize to [-0.5, 0.5]
        arr = np.array(image, dtype=np.float32) / 255.0 - 0.5
        img_tensor = from_numpy(arr)  # [height, width]

        # Encode transcription to label sequence
        labels = self._encode_transcription(page.transcription)

        labels_tensor = tensor(labels, dtype=torch_long)
        return img_tensor, labels_tensor, w, len(labels)

    def _encode_transcription(self, text: str) -> list[int]:
        """Encode text to label IDs using unicharset and recoder."""
        if self.recoder:
            char_ids = self.unicharset.encode_string(text)
            labels = []
            for uid in char_ids:
                codes = self.recoder.encode(uid)
                if codes:
                    labels.extend(codes)
                else:
                    labels.append(self.null_char_id)
            return labels
        else:
            return self.unicharset.encode_string(text)


def collate_fn(batch):
    """Custom collate that pads variable-width images and variable-length labels."""
    images, labels, widths, label_lengths = zip(*batch)

    # Pad images to max width
    max_w = max(widths)
    h = images[0].shape[0]
    padded_images = []
    for img, w in zip(images, widths):
        if w < max_w:
            pad = zeros(h, max_w - w)
            padded_images.append(cat([img, pad], dim=1))
        else:
            padded_images.append(img)

    # Stack and add channel dim: [batch, 1, height, width]
    images_tensor = stack(padded_images).unsqueeze(1)

    # Concatenate labels
    labels_tensor = cat(labels)
    input_lengths = tensor(widths, dtype=torch_long)
    target_lengths = tensor(label_lengths, dtype=torch_long)

    return images_tensor, labels_tensor, input_lengths, target_lengths
