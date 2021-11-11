# Huggingface Transformers Trainer for Periflow

This is Trainer of [Huggingface Transformers (v4.10.1)](https://github.com/huggingface/transformers/tree/v4.10.1) modified to support Periflow.
Periflow will automatically load model parameters and training states (optimizer, LR scheduler, and scaler) if exists.
**Note that** you can still manually load checkpoints as in the original Trainer (by feeding `model_path`), but Periflow load and overwrite the loaded parameters and training states if there is any checkpoint Periflow can load.

## Usage

### Language Modeling Example

```bash
$ pip install -r requirements.txt
$ python -m torch.distributed.launch --nproc_per_node=8 run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size=1 \
    --output_dir /ckpt \
    --cache_dir /data
```

### Use Trainer with Huggingface Transformers Examples
```bash
$ git clone https://github.com/huggingface/transformers.git
$ cd transformers
$ git checkout tags/v4.10.1
# Overwrite the original Trainer
$ cp <SDK root>/src/periflow_sdk/examples/transformers/trainer.py src/transformers/
```

You can simply try this Trainer by running examples in Hugginface Transformers. [Here](https://huggingface.co/transformers/examples.html) for detail.
