# PeriFlow Python SDK

PeriFlow Python SDK for training machine learning models on [FriendliAI](https://friendli.ai) [PeriFlow](https://github.com/friendliai/periflow-cli).
PeriFlow SDK is compatible with both **local** mode and **cloud** mode.
When running your training script in your local machine, PeriFlow SDK runs in local mode.
On the other hand, PeriFlow SDK runs in cloud mode when running your training script by using PeriFlow.
If you want to use PeriFlow, please [contact](mailto:contant@friendli.ai?subject=[GitHub]%20PeriFlow%20Python%20SDK)
to us.

#### Contents

1. [Installation](#installation)
2. [Using PeriFlow SDK](#using-periflow-sdk)
3. [Examples](#examples)

### Installation

PeriFlow SDK is built to PyPI and can be installed as follows:
```sh
pip install periflow_sdk
```
Also, you can install from source by cloning this repository:
```sh
git clone https://github.com/friendliai/periflow-python-sdk.git
cd periflow-python-sdk
pip install .
```

### Using PeriFlow SDK

Basically, you can import and use our SDK as follows:
```python
import periflow_sdk as pf
```

PeriFlow SDK provides several functions to make your training code works with PeriFlow.

```python
pf.init(total_train_steps: int, local_log_name: Optional[str] = None) -> None
```

This function initializes PeriFlow.
All other functions of PeriFlow SDK should be called after initialization.

- `total_train_steps`: The number of total training steps of the training job
- `local_log_name`: local filename where `pf.metric` writes into

**Note**: `local_log_name` is meaningful to local mode

---

```python
pf.start_step() -> None
```
Mark that a single training step begins.

---

```python
pf.end_step() -> None
```
Mark that the single training step ends.

**Note**: we provide function `pf.train_step`, a contextmanager which wraps `start_step` and `end_step`. `train_step` can be used as follows:
```python
with pf.train_step():
    # your training step code
```

---

```python
pf.upload_checkpoint() -> None
```
Trigger uploading the checkpoint of the current step.

**Note**: This function does nothing in local mode  
**Note**: In distributed training (e.g., torch DDP), all ranks should call this function even if some of them have not actually saved the checkpoint.
For example, not
```python
import torch

...
if torch.distributed.get_rank() == 0:
    torch.save(state_dict, CKPT_PATH)
    pf.upload_checkpoint()
```
, but
```python
import torch

...
if torch.distributed.get_rank() == 0:
    torch.save(state_dict, CKPT_PATH)
pf.upload_checkpoint()
```

---

```python
pf.metric(msg: Dict[str, JSONValue]) -> None
```
Optional function which logs a key-value metric dict and sends it to PeriFlow.
However, this function does nothing in cloud mode.
In local mode, this function writes the given metric dict into the local file system.

### Examples

We provide simple [examples](./examples) in which PeriFlow SDK is applied.
Each example contains a template as `pf-template.yml`, which is needed when launching a training job with PeriFlow:
```sh
pf job run -f (cifar|huggingface|pth-lightning)/pf-template.yml -d (cifar|huggingface|pth-lightning)
```

#### Vanilla PyTorch Example

[CIFAR example](./examples/cifar/main.py) is an example training script that uses vanilla PyTorch and Torch Distributed Data Parallel (DDP).
To apply PeriFlow SDK, we first initialize PeriFlow as follows:
```python
pf.init(total_train_steps=total_steps)
```
Then we wrap calling the `train_step` with `pf.train_step`.
```python
with pf.train_step():
    loss, learning_rate = train_step(inputs=inputs,
                                     labels=labels,
                                     model=net,
                                     loss_function=loss_function,
                                     optimizer=optimizer,
                                     lr_scheduler=lr_scheduler)
    if not args.use_cpu:
        torch.cuda.synchronize()
    end_time = time.time()
```
Finally, we call `upload_checkpoint` after saving checkpoint.
```python
if args.save and step % args.save_interval == 0:
    if torch_ddp.get_rank() == 0:
        torch.save({"latest_step": step,
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict()},
                   os.path.join(args.save, "checkpoint.pt"))

    pf.upload_checkpoint()
```


#### HuggingFace Trainer Example

[HuggingFace example](./examples/huggingface/run_glue.py) is an example training script that uses HuggingFace Trainer class.
Instead of inheriting and writing a custom Trainer class, we inject `PeriFlowCallback` as follows:
```python
class PeriFlowCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        pf.start_step()

    def on_step_end(self, args, state, control, **kwargs):
        pf.end_step()

    def on_save(self, args, state, control, **kwargs):
        pf.upload_checkpoint()
```
Then, we create a Trainer class using our custom callback.
```python
pf.init(total_train_steps=training_args.max_steps)
callback = PeriFlowCallback()

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[callback],
)
```

#### PyTorch Lightning Trainer Example
[PyTorch Lightning example](./examples/pth-lightning/main.py) is an example training script that uses PyTorch Lightning Trainer class.
Similar to HuggingFace, we can create custom callback as follows:
```python
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
```
However, because PyTorch Lightning does not provide `on_checkpoint_save` callback, we write a simple `PeriFlowTrainer`.
```python
class PeriFlowTrainer(Trainer):
    def save_checkpoint(self,
                        filepath: Union[str, Path],
                        weights_only: bool = False,
                        storage_options: Optional[Any] = None) -> None:
        super().save_checkpoint(filepath, weights_only=weights_only, storage_options=storage_options)
        pf.upload_checkpoint()
```
With `PeriFlowCallback` and `PeriFlowTrainer`, we can start the training.
```python
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
```
