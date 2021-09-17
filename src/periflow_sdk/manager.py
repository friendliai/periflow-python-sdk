"""The periflow training manager module.
"""

import os
import atexit
from enum import Enum
from dataclasses import dataclass, asdict
import functools
import time
import sys
from threading import Thread
from typing import Callable, Dict
import logging

import torch
from periflow_sdk.checkpoint.checkpoint_func import sync_checkpoint_func
from periflow_sdk.checkpoint.state_provider import default_state_provider
from periflow_sdk.comm.ipc import IpcCommPurpose, CommResultStatus, get_default_ipc_channel
from periflow_sdk.comm.errors import IpcTimeoutException, IpcConnectionFailureException

periflow_logger = logging.getLogger("periflow")


def get_checkpoint_name(it: int) -> str:
    return 'iter_{:07d}/mp_rank_00_000/model_optim_rng.pt'.format(it)


class SaveType(str, Enum):
    PERIODIC = "PERIODIC"
    EMERGENCY = "EMERGENCY"


@dataclass
class TrainStepOutput:
    """ The base output class of a training step.
    Users are encouraged to add statistics to this class, so that Periflow can automatically log necessary data.
    """
    iteration: int


class TrainingManager:
    """ The training wrapper class for general PyTorch training code.
    """
    def __init__(self):

        self._is_local = os.environ.get("PERIFLOW_ENABLED") != "1"

        if self._is_local:
            periflow_logger.info("Periflow SDK is working in local mode.")
            self._stat_ipc_channel = None
            self._ack_ipc_channel = None
            self._emergency_save_ipc_channel = None
            self._local_rank = None
        else:
            periflow_logger.info("Periflow SDK is working in cloud mode.")


    def init(self,
             total_train_steps: int,
             model,
             optimizer = None,
             lr_scheduler = None,
             sampler = None,
             save_interval: int = 0,
             save_dir: str = None,
             checkpoint_save_fn: Callable[[Dict, str], None] = sync_checkpoint_func,
             state_dict_provider_fn: Callable[..., None] = default_state_provider,
             local_rank: int = 0):
        """ Initialize training manager and perform automatic recovery in case that periflow is deployed.

        Arguments:
            - total_train_steps: The number of total training steps.
            - save_interval: The interval step of checkpoint saving. 0 for no checkpointing.
            - checkpoint_save_fn: The function to save model checkpoint. The function should return the name of
                                  checkpoint file.
        """
        self._total_train_steps = total_train_steps
        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._sampler = sampler
        self._save_interval = save_interval
        self._save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        assert os.path.isdir(save_dir), "The save directory already exists and it is not a directory!"
        self._checkpoint_save_fn = checkpoint_save_fn
        self._state_dict_provider_fn = state_dict_provider_fn
        self._curr_step = 0
        self._emergency_save_step = None
        self._log_file = open(os.path.join(save_dir, "periflow_trainer.log"), "w")
        self._local_rank = local_rank

        if not self._is_local:
            self._stat_ipc_channel = get_default_ipc_channel(purpose=IpcCommPurpose.STAT,
                                                             local_rank=local_rank)
            self._ack_ipc_channel = get_default_ipc_channel(purpose=IpcCommPurpose.ACK,
                                                            local_rank=local_rank)
            self._emergency_save_ipc_channel = get_default_ipc_channel(purpose=IpcCommPurpose.EMERGENCY_SAVE,
                                                                       local_rank=local_rank)
            self._stat_ipc_channel.open()
            self._ack_ipc_channel.open()
            self._emergency_save_ipc_channel.open()

            # Start a thread waiting for emergency save request.
            self._wait_emergency_save_thread = Thread(target=self._wait_for_emergency_save_request, daemon=True)
            self._wait_emergency_save_thread.start()

        # Recover from the latest checkpoint.
        # First, we search the latest checkpoint.
        latest_iter = 0
        if save_dir:
            latest_ckpt_file_path = os.path.join(save_dir, "latest_checkpointed_iteration.txt")
            try:
                with open(latest_ckpt_file_path, "r") as latest_ckpt_file:
                    latest_iter = int(latest_ckpt_file.readline().strip())
            except FileNotFoundError:
                periflow_logger.info("Cannot find latest checkpointed iteration from the directory!")
        # There is a checkpoint that we need to recover from...
        if latest_iter > 0:
            ckpt_path = os.path.join(save_dir, get_checkpoint_name(latest_iter))
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # TODO: Recover rng states.
            assert hasattr(sampler, "set_processed_steps"), "Sampler should be one of 'ResumableRandomSampler' or 'ResumableSequentialSasmpler'!"
            sampler.set_processed_steps(latest_iter)

        # teardown will be called at exit of the program.
        atexit.register(self.teardown)
        return latest_iter


    def teardown(self):
        """ Clean up resources.
        Do nothing for local mode.
        """
        if not self._is_local:
            self._stat_ipc_channel.close()
            self._ack_ipc_channel.close()
            self._emergency_save_ipc_channel.close()
            self._wait_emergency_save_thread.join()

        self._log_file.close()


    def _wait_for_emergency_save_request(self):
        """ Wait for the emergency save request from the IPC channel.
        Do nothing for local mode.
        """
        try:
            msg = self._emergency_save_ipc_channel.read(timeout=None)
        except (IpcTimeoutException, IpcConnectionFailureException):
            pass
        else:
            self._emergency_save_step = msg['emergency_save_step']


    def ft_train_batch(self, train_batch_fn: Callable[..., TrainStepOutput]):
        """ Decorator function for training batch function to support automatic checkpoint save.
        """
        @functools.wraps(train_batch_fn)
        def wrapper(*args, **kwargs):
            assert 'iteration' in kwargs and 'model' in kwargs
            iteration = kwargs.get('iteration', None)
            model = kwargs.get('model', None)
            optimizer = kwargs.get('optimizer', None)
            lr_scheduler = kwargs.get('lr_scheduler', None)

            assert model is self._model, "Model in training function should be the same as the one used in init"
            assert optimizer is self._optimizer, "Optimizer in training function should be the same as the one used in init"
            assert lr_scheduler is self._lr_scheduler, "LR Scheduler in training function should be the same as the one used in init"

            start_time = time.time()
            step_output = train_batch_fn(*args, **kwargs)
            end_time = time.time()

            self._curr_step = step_output.iteration
            step_time = end_time - start_time

            is_save_step = self._curr_step % self._save_interval == 0

            if is_save_step or self._curr_step == self._emergency_save_step:
                checkpoint_path = os.path.join(self._save_dir, get_checkpoint_name(iteration))
                # Checkpointing is done only when the local rank is zero.
                if self._local_rank == 0:
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    state_dict = self._state_dict_provider_fn(iteration, model, optimizer, lr_scheduler)
                    self._checkpoint_save_fn(state_dict, checkpoint_path)
                if is_save_step:
                    save_type = SaveType.PERIODIC
                elif self._emergency_save_step is not None and self._curr_step == self._emergency_save_step:
                    save_type = SaveType.EMERGENCY
            else:
                checkpoint_path= None
                save_type = None

            if self._is_local:
                step_output_dict = asdict(step_output)
                step_output_dict["step_time"] = step_time
                self._log_file.write(str(step_output_dict) + "\n")
            else:
                try:
                    # Write training stat of the current rank to FTModule via IPC channel.
                    msg = {
                        "step": self._curr_step,
                        "saved": is_save_step,
                        "save_type": save_type,
                        "is_last_step": self._curr_step == self._total_train_steps,
                        "checkpoint_path": checkpoint_path,
                        "step_time": step_time
                    }
                    periflow_logger.debug(f"IPC WR || send training stat: {msg}")
                    self._stat_ipc_channel.write(msg)

                    # Wait for ack.
                    periflow_logger.debug("Wait for ACK.")
                    ack = self._ack_ipc_channel.read(timeout=None)
                    periflow_logger.debug("ACK received.")
                    if ack["status"] != CommResultStatus.SUCCESS:
                        raise RuntimeError(f"Invalid IPC message from FTModule: {ack}")

                    # If emergency save is done, terminate the training process.
                    if save_type is SaveType.EMERGENCY:
                        sys.exit()
                except IpcConnectionFailureException as ipc_connection_failure:
                    raise RuntimeError("IPC connection between training manager and FTModule is broken.") \
                         from ipc_connection_failure

        return wrapper


ft_train_manager = TrainingManager()
init = ft_train_manager.init
periflow_trainer = ft_train_manager.ft_train_batch
