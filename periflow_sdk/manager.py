import os
import atexit
from enum import Enum
from dataclasses import dataclass, asdict
import functools
from periflow_sdk.checkpoint.checkpoint_func import sync_checkpoint_func
from periflow_sdk.checkpoint.state_provider import default_state_provider
import time
import sys
from threading import Thread
from typing import Callable, Dict
import logging

from .comm.ipc import IpcCommPurpose, CommResultStatus, get_default_ipc_channel
from .comm.errors import IpcTimeoutException, IpcConnectionFailureException

periflow_logger = logging.getLogger("periflow")

def get_checkpoint_name(iter: int):
    return 'iter_{:07d}'.format(iter)


class SaveType(str, Enum):
    PERIODIC = "PERIODIC"
    EMERGENCY = "EMERGENCY"


@dataclass
class TrainStepOutput:
    """ The base output class of a training step.
    Users are encouraged to add statistics to this class, so that Periflow can automatically log necessary data.
    """
    iteration: int
    step_time: float


class TrainingManager:
    """ The training wrapper class for general PyTorch training code.
    """
    def __init__(self):

        self._is_local = os.environ.get("PERYFLOW_ENABLED") != "1"

        if self._is_local:
            self._stat_ipc_channel = None
            self._ack_ipc_channel = None
            self._emergency_save_ipc_channel = None
        else:
            host_slot_list = list(map(int, os.environ["HOST_SLOT_LIST"].split(",")))
            node_rank = int(os.environ["NODE_RANK"])
            num_devices = host_slot_list[node_rank]
            rank = int(os.environ['RANK'])
            local_rank = rank % num_devices
            self._stat_ipc_channel = get_default_ipc_channel(purpose=IpcCommPurpose.STAT,
                                                             local_rank=local_rank)
            self._ack_ipc_channel = get_default_ipc_channel(purpose=IpcCommPurpose.ACK,
                                                            local_rank=local_rank)
            self._emergency_save_ipc_channel = get_default_ipc_channel(purpose=IpcCommPurpose.EMERGENCY_SAVE,
                                                                       local_rank=local_rank)


    def init(self,
             total_train_steps: int,
             save_interval: int = 0,
             save_dir: str = None,
             checkpoint_save_fn: Callable[[Dict, str], None] = sync_checkpoint_func,
             state_dict_provider_fn: Callable[..., None] = default_state_provider):
        """ Initialize training manager.

        Arguments:
            - total_train_steps: The number of total training steps.
            - save_interval: The interval step of checkpoint saving. 0 for no checkpointing.
            - checkpoint_save_fn: The function to save model checkpoint. The function should return the name of
                                  checkpoint file.
        """
        self._total_train_steps = total_train_steps
        self._save_interval = save_interval
        self._save_dir = save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        assert os.path.isdir(save_dir), "The save directory already exists and it is not a directory!"
        self._checkpoint_save_fn = checkpoint_save_fn
        self._state_dict_provider_fn = state_dict_provider_fn
        self._curr_step = 0
        self._emergency_save_step = None
        self._log_file = open(os.path.join(save_dir, "periflow_trainer.log"), "w")

        if not self._is_local:
            self._stat_ipc_channel.open()
            self._ack_ipc_channel.open()
            self._emergency_save_ipc_channel.open()

            # Start a thread waiting for emergency save request.
            self._wait_emergency_save_thread = Thread(target=self._wait_for_emergency_save_request, daemon=True)
            self._wait_emergency_save_thread.start()

        # teardown will be called at exit of the program.
        atexit.register(self.teardown)


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
            start_time = time.time()
            step_output = train_batch_fn(*args, **kwargs)
            end_time = time.time()

            self._curr_step = step_output.iteration
            step_time = end_time - start_time

            is_save_step = self._curr_step % self._save_interval == 0

            if is_save_step or self._curr_step == self._emergency_save_step:
                iteration = kwargs.get('iteration', None)
                model = kwargs.get('model', None)
                optimizer = kwargs.get('optimizer', None)
                lr_scheduler = kwargs.get('lr_scheduler', None)

                checkpoint_path = os.path.join(self._save_dir, get_checkpoint_name(iteration))
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

                    return step_output
                except IpcConnectionFailureException:
                    raise RuntimeError("IPC connection between training manager and FTModule is broken.")
                

        return wrapper


ft_train_manager = TrainingManager()
init = ft_train_manager.init
periflow_trainer = ft_train_manager.ft_train_batch