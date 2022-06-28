import argparse
import re
from pathlib import Path
from typing import Any, Optional, Union

import periflow_sdk as pf
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import AutoTokenizer
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
    TextClassificationTransformer,
)


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
        pattern = re.compile(r"step=(\d+)")
        checkpoint_iter = None
        for ckpt_path in Path(args.checkpoint_dir).glob("**/*"):
            step = int(pattern.findall(ckpt_path.name)[0])
            if checkpoint_iter is None:
                checkpoint_iter = step
            else:
                checkpoint_iter = max(checkpoint_iter, step)

        if checkpoint_iter is not None:
            ckpt_path = checkpoint_callback.format_checkpoint_name(dict(step=checkpoint_iter))
        else:
            ckpt_path = None
    else:
        checkpoint_callback = Callback()
        ckpt_path = None

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased", cache_dir=args.cache_dir)
    dm = TextClassificationDataModule(
        batch_size=args.batch_size,
        dataset_name="glue",
        dataset_config_name="sst2",
        max_length=512,
        tokenizer=tokenizer,
        cache_dir=args.cache_dir,
        num_workers=4,
    )
    model = TextClassificationTransformer(
        pretrained_model_name_or_path="bert-base-uncased", num_labels=dm.num_classes,
        cache_dir=args.cache_dir
    )

    # PeriFlow
    periflow_callback = PeriFlowCallback()

    trainer = PeriFlowTrainer(
        accelerator="ddp", gpus=args.num_gpus, max_epochs=args.num_epochs,
        callbacks=[periflow_callback, checkpoint_callback],
        enable_checkpointing=isinstance(checkpoint_callback, ModelCheckpoint),
        progress_bar_refresh_rate=1,
    )

    pf.init(total_train_steps=100000)

    trainer.fit(model=model,
                datamodule=dm,
                ckpt_path=ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=8)
    main(parser.parse_args())
