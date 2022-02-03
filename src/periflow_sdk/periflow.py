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
from typing import Callable, Dict, List
import logging
from pathlib import Path

import torch
from periflow_sdk.checkpoint.checkpoint_func import sync_checkpoint_func
from periflow_sdk.checkpoint.state_provider import default_state_provider
from periflow_sdk.comm.ipc import IpcCommPurpose, CommResultStatus, get_default_ipc_channel
from periflow_sdk.comm.errors import IpcTimeoutException, IpcConnectionFailureException

periflow_logger = logging.getLogger("periflow")


class SaveType(str, Enum):
    NORMAL = "NORMAL"
    EMERGENCY = "EMERGENCY"


class TrainingManager:
    """ The training wrapper class for general PyTorch training code.
    """
    def __init__(self):
        self._is_local = os.environ.get("PERIFLOW_ENABLED") != "1"
        self._total_train_steps = None
        self._cur_step = 0
        self._step_info_ipc_channel = None
        self._ack_ipc_channel = None
        self._emergency_save_ipc_channel = None
        self._metric_ipc_channel = None
        self._wait_emergency_save_thread = None
        self._local_rank = None
        self._is_step_started = False
        self._is_saved = False
        self._save_method = SaveType.NORMAL
        self._checkpoint_path = None
        self._step_start_time = 0
        self._local_log_path = Path(f"./periflow_trainer_{int(time.time())}.log")

    def init(self,
             total_train_steps: int,
             local_rank: int = 0):
        """ Initialize training manager.

        Arguments:
            - total_train_steps: The number of total training steps
            - local_rank: The local rank of this training process
        """
        self._total_train_steps = total_train_steps
        self._local_rank = local_rank

        if self._is_local:
            periflow_logger.debug("Periflow SDK is working in local mode.")
        else:
            periflow_logger.debug("Periflow SDK is working in cloud mode.")
            self._step_info_ipc_channel = get_default_ipc_channel(purpose=IpcCommPurpose.STEP_INFO,
                                                                  local_rank=local_rank)
            self._ack_ipc_channel = get_default_ipc_channel(purpose=IpcCommPurpose.ACK,
                                                            local_rank=local_rank)
            self._emergency_save_ipc_channel = get_default_ipc_channel(purpose=IpcCommPurpose.EMERGENCY_SAVE,
                                                                       local_rank=local_rank)
            self._metric_ipc_channel = get_default_ipc_channel(purpose=IpcCommPurpose.METRIC,
                                                               local_rank=local_rank)
            self._step_info_ipc_channel.open()
            self._ack_ipc_channel.open()
            self._emergency_save_ipc_channel.open()
            self._metric_ipc_channel.open()

            # Start a thread waiting for emergency save request.
            self._wait_emergency_save_thread = Thread(target=self._wait_for_emergency_save_request, daemon=True)
            self._wait_emergency_save_thread.start()

        # teardown will be called at exit of the program.
        atexit.register(self._teardown)

    def _teardown(self):
        """ Clean up resources.
        Do nothing for local mode.
        """
        if not self._is_local:
            self._step_info_ipc_channel.close()
            self._ack_ipc_channel.close()
            self._emergency_save_ipc_channel.close()
            self._metric_ipc_channel.close()
            self._wait_emergency_save_thread.join()

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

    def start_step(self):
        assert not self._is_step_started, "Existing steps must finish before calling start_step()!"
        self._step_start_time = time.time()
        self._cur_step += 1
        self._is_step_started = True

    def end_step(self):
        assert self._is_step_started, "Existing steps must start before calling end_step()!"
        step_time = time.time() - self._step_start_time
        try:
            msg = {
                "step": self._cur_step,
                "is_last_step": self._cur_step == self._total_train_steps,
                "step_time": step_time,
                "saved": self._is_saved,
                "save_type": self._save_method,
                "checkpoint_path": self._checkpoint_path
            }
            periflow_logger.debug(f"IPC WR || send training stat: {msg}")
            self._step_info_ipc_channel.write(msg)

            # Wait for ack.
            periflow_logger.debug("Wait for ACK.")
            ack = self._ack_ipc_channel.read(timeout=None)
            periflow_logger.debug("ACK received.")
            if ack["status"] != CommResultStatus.SUCCESS:
                raise RuntimeError(f"Invalid IPC message from FTModule: {ack}")

            # If emergency save is done, terminate the training process.
            if self._is_saved and self._save_method is SaveType.EMERGENCY:
                sys.exit()

        except IpcConnectionFailureException as ipc_connection_failure:
            raise RuntimeError("IPC connection between training manager and FTModule is broken.") \
                from ipc_connection_failure

    def is_emergency_save(self):
        return self._emergency_save_step == self._cur_step

    def _local_log(self, msg):
        mode = "a" if self._cur_step > 1 else "w"
        with self._local_log_path.open(mode=mode) as log_file:
            log_file.write(f"{msg}\n")

    def log(self, msg: Dict):
        if "step" not in msg:
            msg["step"] = self._cur_step
        if self._is_local:
            self._local_log(msg)
        else:
            self._metric_ipc_channel.write(msg)

    def save(self, obj, path: str, async_save: bool = False):
        assert not self._is_saved, "You cannot call `pf.save()` twice within a training step."
        assert self._is_step_started, "You can only call `pf.save()` within a training step scope."
        if async_save:
            raise NotImplementedError("Asynchronous checkpointing is not supported for now.")
        torch.save(obj, path)
        self._is_saved = True
        if self.is_emergency_save():
            self._save_method = SaveType.EMERGENCY
        else:
            self._save_method = SaveType.NORMAL
        self._checkpoint_path = path


manager = TrainingManager()
init = manager.init
save = manager.save
log = manager.log
is_emergency_save = manager.is_emergency_save
start_step = manager.start_step
end_step = manager.end_step
