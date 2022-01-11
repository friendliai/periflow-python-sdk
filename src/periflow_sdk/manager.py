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
from typing import Any, Callable, Dict, List
import logging

from periflow_sdk.checkpoint.checkpoint_func import sync_checkpoint_save, sync_checkpoint_load
from periflow_sdk.comm.ipc import IpcCommPurpose, CommResultStatus, get_default_ipc_channel
from periflow_sdk.comm.errors import IpcTimeoutException, IpcConnectionFailureException

periflow_logger = logging.getLogger("PF_SYS")


@dataclass
class DistOption:
    """Similar to PeriFlow.DistOption, but save rank (not degree)
    """
    local_rank: int
    pp_rank: int
    dp_rank: int
    mp_rank: int

    def checkpoint_name(self, iteration) -> str:
        directory = 'iter_{:07d}'.format(iteration)
        return os.path.join(directory,
                            'mp_rank_{:02d}_{:03d}'.format(self.mp_rank, self.pp_rank),
                            'model_optim_rng.pt')


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
             save_interval: int = 0,
             save_dir: Optional[str] = None,
             load_dir: Optional[str] = None,
             checkpoint_save_fn: Callable[..., None] = sync_checkpoint_save,
             checkpoint_load_fn: Callable[..., None] = sync_checkpoint_load,
             # distributed option
             local_rank: int = 0,
             pp_rank: int = 0,
             dp_rank: int = 0,
             mp_rank: int = 0,
             # state containers
             model = None,
             optimizer = None,
             lr_scheduler = None) -> int:
        """ Initialize training manager and perform automatic recovery in case that periflow is deployed.

        Arguments:
            - total_train_steps: The number of total training steps.
            - save_interval: The interval step of checkpoint saving. 0 for no checkpointing.
            - save_dir: checkpoint directory root for saving.
            - load_dir: checkpoint directory root for loading.
            - checkpoint_save_fn: The function to save model checkpoint.
            - checkpoint_load_fn: The function to load model checkpoint.
            - local_rank / pp_rank / dp_rank / mp_rank: distributed training config
            - model: DL Model (could be a single nn.Module, collection of separate models)
            - optimizer: Training optimizer (could be collectioN)
            - lr_scheduler: learning rate scheduler (could be collection)

        Returns:
            - latest training iteration
        """
        self._total_train_steps = total_train_steps

        self._save_interval = save_interval

        if save_interval > 0:
            assert save_dir is not None, "save directory should be specified"

        self._save_dir = save_dir
        self._load_dir = load_dir
        os.makedirs(self._save_dir, exist_ok=True)
        assert os.path.isdir(save_dir), "The save directory already exists and it is not a directory!"

        self._checkpoint_save_fn = checkpoint_save_fn
        self._checkpoint_load_fn = checkpoint_load_fn

        self._emergency_save_step = None
        self._log_file = open(os.path.join(save_dir, "periflow_trainer.log"), "w")

        self._dist_option = DistOption(
            local_rank=local_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            mp_rank=mp_rank)

        # State related objects
        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler

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
        if self._load_dir is not None:
            assert os.path.exists(self._load_dir), "The load directory does not exist!"
            assert os.path.isdir(self._load_dir), "The load directory exists but it is not a directory"
            latest_ckpt_file_path = os.path.join(self._load_dir, "latest_checkpointed_iteration.txt")
            try:
                with open(latest_ckpt_file_path, "r", encoding="utf-8") as f:
                    latest_iter = int(f.readline().strip())
            except FileNotFoundError:
                periflow_logger.error(f"Cannot find latest checkpointed iteration from the {self._load_dir}! "
                                      "Start from 0...")

        if latest_iter > 0:
            checkpoint_name = self._dist_option.checkpoint_name(latest_iter)
            self._checkpoint_load_fn(checkpoint_name,
                                     self._model,
                                     self._optimizer,
                                     self._lr_scheduler)

        self._cur_iter = latest_iter

        # teardown will be called at exit of the program.
        atexit.register(self.teardown)
        return latest_iter


    def recover_samplers(self, samplers: List):
        for sampler in samplers:
            assert hasattr(sampler, 'set_processed_steps'), "Samplers should have 'set_processed_steps()'"
            sampler.set_processed_steps(self._cur_iter)


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
            start_time = time.time()
            step_output = train_batch_fn(*args, **kwargs)
            end_time = time.time()
            self._cur_iter += 1

            step_time = end_time - start_time

            is_save_step = self._save_interval > 0 and self._cur_iter % self._save_interval == 0

            if is_save_step or self._cur_iter == self._emergency_save_step:
                checkpoint_path = os.path.join(self._save_dir, self._dist_option.checkpoint_name(self._cur_iter))
                self._checkpoint_save_fn(self._cur_iter,
                                         checkpoint_path,
                                         self._model,
                                         self._optimizer,
                                         self._lr_scheduler)
                if self._is_local and self._dist_option.local_rank == 0:
                    with open(os.path.join(self._save_dir, "latest_checkpointed_iteration.txt"),
                              "w", encoding="utf-8") as iter_log:
                        iter_log.write(str(self._cur_iter))
                        os.fsync(iter_lod)

                save_type = None
                if is_save_step:
                    save_type = SaveType.PERIODIC
                elif self._emergency_save_step is not None and self._cur_iter == self._emergency_save_step:
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
                        "step": self._cur_iter,
                        "saved": is_save_step,
                        "save_type": save_type,
                        "is_last_step": self._cur_iter == self._total_train_steps,
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

            return step_output
        return wrapper


ft_train_manager = TrainingManager()
