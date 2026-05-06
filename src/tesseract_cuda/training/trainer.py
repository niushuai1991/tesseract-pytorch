"""Training loop for Tesseract LSTM models with PyTorch."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import time
import os

from ..network.model import TessLSTMModel
from ..formats.tessdata import TessdataManager, TESSDATA_LSTM
from .dataset import LSTMFDataset, collate_fn


class LSTMTrainer:
    """Trainer for Tesseract LSTM models using PyTorch."""

    def __init__(
        self,
        model: TessLSTMModel,
        device: str = "cpu",
        learning_rate: float = 0.001,
        adam_beta1: float = 0.5,
        adam_beta2: float = 0.999,
        max_iterations: int = 0,
        target_error_rate: float = 0.01,
        checkpoint_interval: int = 10000,
        model_output: str = "output/checkpoint",
    ):
        self.model = model
        self.device = device
        self.max_iterations = max_iterations
        self.target_error_rate = target_error_rate
        self.checkpoint_interval = checkpoint_interval
        self.model_output = model_output

        self.model.to(device)

        # Adam optimizer matching Tesseract defaults
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
        )

        # CTC Loss
        self.criterion = nn.CTCLoss(blank=model.null_char, reduction="mean", zero_infinity=True)

        self.training_iteration = 0
        self.sample_iteration = 0
        self.best_error = float("inf")

    def train(
        self,
        train_dataset: LSTMFDataset,
        eval_files: Optional[list[str]] = None,
        starter_traineddata: Optional[str] = None,
    ) -> None:
        """Run the training loop."""
        loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )

        os.makedirs(os.path.dirname(self.model_output) or ".", exist_ok=True)

        start_time = time.time()
        running_loss = 0.0
        running_count = 0

        while True:
            for batch in loader:
                if self.max_iterations > 0 and self.training_iteration >= self.max_iterations:
                    self._print_progress(start_time, running_loss, running_count, final=True)
                    return

                images, labels, input_lengths, target_lengths = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward
                self.optimizer.zero_grad()
                output = self.model(images)

                # output shape: [batch, width, num_classes] or [width, num_classes]
                if output.dim() == 2:
                    output = output.unsqueeze(0)

                # For CTC: need [seq_len, batch, num_classes]
                log_probs = output.permute(1, 0, 2).log_softmax(dim=-1)

                # Compute CTC loss
                batch_size = log_probs.shape[1]
                loss = self.criterion(
                    log_probs,
                    labels,
                    input_lengths.to(self.device),
                    target_lengths.to(self.device),
                )

                # Backward
                if not loss.isinf() and not loss.isnan():
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    running_loss += loss.item()
                    running_count += 1

                self.training_iteration += 1
                self.sample_iteration += 1

                # Progress report
                if self.training_iteration % 100 == 0:
                    self._print_progress(start_time, running_loss, running_count)

                # Checkpoint
                if self.training_iteration % self.checkpoint_interval == 0:
                    self._save_checkpoint(starter_traineddata)

    def _print_progress(self, start_time, running_loss, running_count, final=False):
        if running_count == 0:
            return
        avg_loss = running_loss / running_count
        elapsed = time.time() - start_time
        status = "Finished" if final else ""
        print(
            f"At iteration {self.training_iteration}/{self.sample_iteration}, "
            f"Mean loss={avg_loss:.6f}, "
            f"elapsed={elapsed:.0f}s {status}"
        )

    def _save_checkpoint(self, starter_traineddata: Optional[str] = None):
        # Save PyTorch checkpoint
        ckpt_path = f"{self.model_output}_checkpoint.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "training_iteration": self.training_iteration,
            "sample_iteration": self.sample_iteration,
        }, ckpt_path)

        # Save Tesseract-format checkpoint
        if starter_traineddata:
            tessdata_path = f"{self.model_output}_{self.training_iteration}.traineddata"
            self._export_traineddata(starter_traineddata, tessdata_path)
            print(f"Saved checkpoint: {tessdata_path}")

    def _export_traineddata(self, starter_path: str, output_path: str):
        """Export current model to traineddata format."""
        from ..export.exporter import export_model
        export_model(self.model, starter_path, output_path,
                     self.training_iteration, self.sample_iteration)
