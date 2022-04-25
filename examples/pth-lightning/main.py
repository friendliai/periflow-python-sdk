# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MNIST autoencoder example.
To run: python autoencoder.py --trainer.max_epochs=50
"""
import argparse
import math
import re
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE
from pytorch_lightning.utilities.types import STEP_OUTPUT

from utils import _DATASETS_PATH
from mnist_datamodule import MNIST

import periflow_sdk as pf

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms


class LitAutoEncoder(pl.LightningModule):
    """
    >>> LitAutoEncoder()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitAutoEncoder(
      (encoder): ...
      (decoder): ...
    )
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.decoder = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 28 * 28))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _prepare_batch(self, batch):
        x, _ = batch
        return x.view(x.size(0), -1)

    def _common_step(self, batch, batch_idx, stage: str):
        x = self._prepare_batch(batch)
        loss = F.mse_loss(x, self(x))
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss


class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        dataset = MNIST(_DATASETS_PATH, train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = MNIST(_DATASETS_PATH, train=False, download=True, transform=transforms.ToTensor())
        self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])
        self.batch_size = batch_size

    @property
    def num_steps_per_epoch(self) -> int:
        return math.ceil(len(self.mnist_train) / self.batch_size)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class PeriFlowCallback(Callback):
    def on_train_batch_start(self,
                             trainer: pl.Trainer,
                             pl_module: pl.LightningModule,
                             batch: Any,
                             batch_idx: int,
                             unused: int = 0) -> None:
        pf.start_step()

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: STEP_OUTPUT,
                           batch: Any,
                           batch_idx: int,
                           unused: int = 0) -> None:
        loss = float(outputs['loss'])
        pf.metric({
            "iteration": trainer.global_step,
            "loss": loss,
        })
        pf.end_step()


class PeriFlowTrainer(Trainer):
    def save_checkpoint(self,
                        filepath: Union[str, Path],
                        weights_only: bool = False,
                        storage_options: Optional[Any] = None) -> None:
        super().save_checkpoint(filepath, weights_only=weights_only, storage_options=storage_options)
        pf.upload_checkpoint()


def main(args):
    if args.checkpoint_dir is not None:
        # When use PeriFlow with PyTorch Lightning, do not save the checkpoint twice (i.e., save_top_k > 0 && save_last = True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="checkpoint-{step:07d}",
            save_last=False,
            every_n_epochs=1,
            save_top_k=1,
        )
        pattern = re.compile(f"step=(\d+)")
        checkpoint_iter = None
        for ckpt_path in Path(args.checkpoint_dir).glob("**/*"):
            step = int(pattern.findall(ckpt_path.name)[0])
            if checkpoint_iter is None:
                checkpoint_iter = step
            else:
                checkpoint_tier = max(checkpoint_iter, step)

        if checkpoint_iter is not None:
            ckpt_path = checkpoint_callback.format_checkpoint_name(dict(step=checkpoint_iter))
        else:
            ckpt_path = None
    else:
        checkpoint_callback = Callback()
        ckpt_path = None

    periflow_callback = PeriFlowCallback()
    trainer = PeriFlowTrainer(
        max_epochs=args.num_epochs,
        callbacks=[periflow_callback, checkpoint_callback],
        enable_checkpointing=isinstance(checkpoint_callback, ModelCheckpoint),
    )

    model = LitAutoEncoder()
    datamodule = MyDataModule()
    pf.init(total_train_steps=args.num_epochs * datamodule.num_steps_per_epoch)

    trainer.fit(model=model,
                datamodule=datamodule,
                ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    main(parser.parse_args())
