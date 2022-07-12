from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers import AutoTokenizer
from lightning_transformers.task.nlp.language_modeling import (
    LanguageModelingDataModule,
    LanguageModelingTransformer
)
from torch import distributed as torch_ddp
from torch.distributed.elastic.multiprocessing.errors import record

import periflow_sdk as pf
import argparse
import os
from tqdm import tqdm


@record
def main(args: argparse.Namespace):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    torch_ddp.init_process_group(backend='nccl')

    # Detect saved checkpoints
    if args.load_from_checkpoint:
        if args.checkpoint_dir and os.path.isdir(args.checkpoint_dir):
            # Load from the latest checkpoint file
            print_once("### Finding checkpoint to initialize model")
            filenames = os.listdir(args.checkpoint_dir)
            if len(filenames) == 0:
                raise RuntimeError("Specified checkpoint directory is empty.")
            for epoch in reversed(range(args.num_epochs)):
                if (file := f"checkpoint-epoch={epoch:02d}.ckpt") in filenames:
                    break
            ckpt_path = os.path.join(args.checkpoint_dir, file)
        else:
            raise RuntimeError("Cannot find checkpoint directory.")
    else:
        ckpt_path = None

    # Prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if ckpt_path is None:
        model = LanguageModelingTransformer(
            args.model_name,
            load_weights=False,
            tokenizer=tokenizer,
        )
    else:
        model = LanguageModelingTransformer.load_from_checkpoint(
            ckpt_path,
            tokenizer=tokenizer,
        )

    # Prepare data module
    print_once("### Preparing data module")
    num_train_examples = 36718
    dm = LanguageModelingDataModule(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        batch_size=args.batch_size,
        train_val_split=0.2,
        max_length=512,
        cache_dir=args.cache_dir,
        num_workers=8,
    )

    # Prepare trainer callbacks
    periflow_callback = PeriflowCallback(log_interval=args.log_interval)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="checkpoint-{epoch:02d}",
        save_last=False,
        every_n_epochs=1,
        save_top_k=1,
    )
    trainer = PeriFlowTrainer(
        accelerator="gpu",
        gpus=args.num_gpus,
        max_epochs=args.num_epochs, 
        callbacks=[periflow_callback, checkpoint_callback],
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    # Initialize Periflow
    total_train_steps = num_train_examples * args.num_epochs / args.batch_size
    pf.init(total_train_steps=total_train_steps)

    # Train
    print_once("### Start training")
    trainer.fit(model, dm, ckpt_path=ckpt_path)


class PeriflowCallback(Callback):

    def __init__(self, log_interval):
        super().__init__()
        self.log_interval = log_interval

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, unused=0):
        pf.start_step()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        step = trainer.global_step
        loss = float(outputs['loss'])
        pf.metric({
            "iteration": step,
            "loss": loss,
        })
        pf.end_step()

        if step % self.log_interval == 0:
            if trainer.global_rank == 0:
                epoch = trainer.current_epoch
                tqdm.write(f"Epoch {epoch} step {step} - loss {loss:.4f}")


class PeriFlowTrainer(Trainer):
    def save_checkpoint(self,
                        filepath,
                        weights_only=False,
                        storage_options=None) -> None:
        super().save_checkpoint(filepath, weights_only=weights_only, storage_options=storage_options)
        pf.upload_checkpoint()


def print_once(msg):
    if not torch_ddp.is_initialized() or torch_ddp.get_rank() != 0:
        return
    print(msg, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-config-name", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--tokenizer-name", type=str, required=True)
    parser.add_argument("--load-from-checkpoint", action='store_true')
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    # Print program arguments
    print_once("#"*40)
    print_once("INPUT ARGUMENTS")
    for name, val in vars(args).items():
        print_once(f"{name} = {val}")
    print_once("#"*40)
    main(args)
