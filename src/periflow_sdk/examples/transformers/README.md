# Huggingface Transformers Trainer for Periflow

This is modified Trainer of [Huggingface Transformers (v4.10.1)](https://github.com/huggingface/transformers/tree/v4.10.1) modified supporting Periflow.
Periflow will automatically load model parameters and training states (optimizer, LR scheduler, and scaler) if exists.
**Note that** you can still manually load checkpoints as in the original Trainer (by feeding `model_path`), but Periflow load and overwrite the loaded parameters and training states if there are checkpoints Periflow can load.

## Usage
```bash
$ git clone https://github.com/huggingface/transformers.git
$ cd transformers
$ git checkout tags/v4.10.1
# Overwrite the original Trainer
$ cp <SDK root>/src/periflow_sdk/examples/transformers/trainer.py src/transformers/
```

You can try this Trainer simply by running examples in Hugginface Transformers. [Here](https://huggingface.co/transformers/examples.html) for detail.
