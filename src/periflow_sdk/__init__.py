"""The periflow training manager module.
"""

import atexit
import copy
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from threading import Thread
from typing import Dict, Union, Any, Optional

import torch

from periflow_sdk.comm.errors import IpcTimeoutException, IpcConnectionFailureException
from periflow_sdk.comm.ipc import IpcCommPurpose, CommResultStatus, get_default_ipc_channel

periflow_logger = logging.getLogger("periflow")
CKPT_FILE_NAME = "model_optim_rng.pt"


class SaveType(str, Enum):
    NORMAL = "NORMAL"
    EMERGENCY = "EMERGENCY"


class TrainingManager:

    """ The training wrapper class for general PyTorch training code.
    """
    def __init__(self,
                 log_file_name: Optional[str] = None,
                 is_local: Optional[bool] = None,
                 teardown_at_exit: bool = True):
        if is_local is None:
            self._is_local = os.environ.get("PERIFLOW_ENABLED") != "1"
        else:
            self._is_local = is_local
        self._total_train_steps = None
        self._cur_step = 0
        self._step_info_ipc_channel = None
        self._ack_ipc_channel = None
        self._emergency_save_ipc_channel = None
        self._metric_ipc_channel = None
        self._wait_emergency_save_thread = None
        self._local_rank = None
        self._is_saved = False
        self._save_method = SaveType.NORMAL
        self._checkpoint_path = None
        self._step_start_time = None
        if self._is_local:
            if log_file_name is None:
                self._log_path = Path(f"./periflow_trainer_{int(time.time())}.log")
            else:
                self._log_path = Path(log_file_name)
        else:
            self._log_path = None
        self._has_locally_logged = False
        self._teardown_at_exit = teardown_at_exit
        self._emergency_save_step = -1

    @property
    def _is_step_started(self) -> bool:
        return self._step_start_time is not None

    def init(self,
             total_train_steps: int,
             processed_steps: int = 0,
             local_rank: int = 0) -> None:
        """ Initialize training manager.

        Arguments:
            - total_train_steps: The number of total training steps
            - processed_steps: How many training steps processed before init()?
            - local_rank: The local rank of this training process
        """
        self._total_train_steps = total_train_steps
        self._cur_step = processed_steps
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

        if self._teardown_at_exit:
            # teardown will be called at exit of the program.
            atexit.register(self._teardown)

    def _teardown(self) -> None:
        """ Clean up resources.
        """
        if not self._is_local:
            self._step_info_ipc_channel.close()
            self._ack_ipc_channel.close()
            self._emergency_save_ipc_channel.close()
            self._metric_ipc_channel.close()

            self._step_info_ipc_channel.remove()
            self._ack_ipc_channel.remove()
            self._emergency_save_ipc_channel.remove()
            self._metric_ipc_channel.remove()

    def _wait_for_emergency_save_request(self) -> None:
        """ Wait for the emergency save request from the IPC channel.
        Do nothing for local mode.
        """
        try:
            msg = self._emergency_save_ipc_channel.read(timeout=None)
        except (IpcTimeoutException, IpcConnectionFailureException):
            pass
        else:
            self._emergency_save_step = msg['emergency_save_step']

    def start_step(self) -> None:
        """
        Start a new training step.
        Returns: None
        """
        assert not self._is_step_started, "Existing steps must finish before calling start_step()!"
        self._step_start_time = time.monotonic()
        self._cur_step += 1

    def end_step(self) -> None:
        """
        Finish the current training step.
        Returns: None
        """
        assert self._is_step_started, "Existing steps must start before calling end_step()!"
        step_time = time.monotonic() - self._step_start_time
        if not self._is_local:
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
        self._step_start_time = None

    @contextmanager
    def train_step(self) -> None:
        """
        The context management wrapper for `start_step` and `end_step`.
        Returns: None
        """
        self.start_step()
        try:
            yield
        finally:
            self.end_step()

    def is_emergency_save(self) -> bool:
        """
        Informs whether emergency save should be handled this step or not.
        Returns: 'True' if emergency save is set, 'False' if not.
        """
        return self._emergency_save_step == self._cur_step

    def _local_log(self, msg):
        mode = "a" if self._has_locally_logged else "w"
        with self._log_path.open(mode=mode) as log_file:
            log_file.write(f"{json.dumps(msg)}\n")
        self._has_locally_logged = True

    def metric(self, msg: Dict) -> None:
        """
        Log a key-value metric dict and send it to Periflow. `step` info is added if not exists.
        Args:
            msg: A key-value dict containing user-defined metrics.

        Returns: None

        """
        new_msg = msg.copy()
        if "step" not in new_msg:
            new_msg["step"] = self._cur_step
        if self._is_local:
            self._local_log(new_msg)
        else:
            self._metric_ipc_channel.write(new_msg)

    def _get_cloud_path(self) -> Path:
        assert "CKPT_PATH" in os.environ, "Environment variable `CKPT_PATH` should be set in cloud mode!"
        assert "MP_DEGREE" in os.environ, "Environment variable `MP_DEGREE` should be set in cloud mode!"
        assert "PP_DEGREE" in os.environ, "Environment variable `PP_DEGREE` should be set in cloud mode!"
        mp_degree = os.environ.get("MP_DEGREE")
        pp_degree = os.environ.get("PP_DEGREE")
        path = Path(os.environ.get("CKPT_PATH")) / "iter_{:07d}/mp_rank_{:02d}_{:03d}".format(
            self._cur_step, int(mp_degree), int(pp_degree)) / CKPT_FILE_NAME
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def load(self, path: Union[os.PathLike, str]) -> Any:
        """
        Load the saved object from persistent storage. In local mode, this is same as `torch.save()`.
        In cloud mode, it ignores the 'path' and loads from the cloud path.
        Args:
            path: The path of target object. Ignored in cloud mode.

        Returns: Loaded object

        """
        if not self._is_local:
            path = self._get_cloud_path()
        return torch.load(path)

    def save(self, obj, path: Union[os.PathLike, str], async_save: bool = False) -> None:
        """
        Save the desired object to persistent storage.
        Args:
            obj: Object to be stored
            path: Path to be stored. Ignored in cluster mode.
            async_save: Save is done asynchronously if true. Currently not supported.

        Returns: None

        """
        assert not self._is_saved, "You cannot call `pf.save()` twice within a training step."
        assert self._is_step_started, "You can only call `pf.save()` within a training step scope."
        if async_save:
            raise NotImplementedError("Asynchronous checkpointing is not supported for now.")
        # Override path in cluster mode.
        if not self._is_local:
            path = self._get_cloud_path()
        torch.save(obj, path)
        self._is_saved = True
        if self.is_emergency_save():
            self._save_method = SaveType.EMERGENCY
        else:
            self._save_method = SaveType.NORMAL
        self._checkpoint_path = str(Path(path).resolve())


periflow = TrainingManager()
