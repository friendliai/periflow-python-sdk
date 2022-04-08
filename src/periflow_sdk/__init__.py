"""The periflow training manager module.
"""
import asyncio
import atexit
import copy
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Thread
from typing import Dict, Union, Any, Optional, BinaryIO, IO

import torch

from periflow_sdk.comm.errors import IpcConnectionError, IpcChannelNotOpenedError
from periflow_sdk.comm.ipc import (
    IpcCommPurpose,
    CommResultStatus,
    get_default_ipc_channel,
    IpcChannel
)
from periflow_sdk.errors import PeriFlowError, PeriFlowInternalError
from periflow_sdk.utils import (
    check_initialized,
    ensure_divisibility,
    DistributeConfig,
    SaveType
)


class TrainingManager:

    """ The training wrapper class for general PyTorch training code.
    """
    def __init__(self, teardown_at_exit: bool = True):
        self.has_initialized: bool = False

        self._is_local: bool = os.environ.get("PERIFLOW_ENABLED") != "1"

        self._cur_step: int = 0
        self._save_method: Optional[SaveType] = None
        self._step_start_time: Optional[float] = None
        self._dist_config: Optional[DistributeConfig] = None

        # IPC Channels
        self._ipc_channels: Dict[IpcCommPurpose, IpcChannel] = {}

        # Emergency save
        self._wait_emergency_save_thread: Optional[Thread] = None
        self._emergency_save_step: Optional[int] = None

        # Used only for local mode
        self._log_path: Optional[Path] = None

        if not self._is_local:

            # Environment variable check.
            required_env_vars = ["RANK",
                                 "WORLD_SIZE",
                                 "NODE_RANK",
                                 "NUM_NODES",
                                 "PROCESSED_ITERS"]

            for env_var in required_env_vars:
                if env_var not in os.environ:
                    raise PeriFlowInternalError(
                        f"Environment variable '{env_var}' should be set in cloud mode!."
                    )

            # Configure dist info
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])
            num_nodes = int(os.environ["NUM_NODES"])
            ensure_divisibility(world_size, num_nodes)
            devices_per_node = world_size // num_nodes
            local_rank = rank % devices_per_node
            self._cur_step = int(os.environ["PROCESSED_ITERS"])
            self._dist_config = DistributeConfig(local_rank=local_rank, rank=rank)

            # IPC Channels
            self._ipc_channels: Dict[IpcCommPurpose, IpcChannel] = {
                k: get_default_ipc_channel(purpose=k, local_rank=local_rank) for k in IpcCommPurpose
            }
            for ipc_channel in self._ipc_channels.values():
                ipc_channel.open()

            # Start a thread waiting for emergency save request.
            self._wait_emergency_save_thread = Thread(target=self._wait_for_emergency_save_request, daemon=True)
            self._wait_emergency_save_thread.start()

        if teardown_at_exit:
            # teardown will be called at exit of the program.
            atexit.register(self._teardown)

    @property
    def _is_saved(self) -> bool:
        return self._save_method is not None

    @property
    def _is_step_started(self) -> bool:
        return self._step_start_time is not None

    @property
    def _has_locally_logged(self) -> bool:
        if self._log_path is None:
            return False

        return self._log_path.exists()

    def _teardown(self) -> None:
        """ Clean up resources.
        """
        if not self._is_local:
            for ipc_channel in self._ipc_channels.values():
                ipc_channel.close()
                ipc_channel.remove()

    def _wait_for_emergency_save_request(self) -> None:
        """ Wait for the emergency save request from the IPC channel.
        Do nothing for local mode.
        """
        try:
            msg = asyncio.run(self._ipc_channels[IpcCommPurpose.EMERGENCY_SAVE].read())
        except (IpcConnectionError, IpcChannelNotOpenedError):
            pass
        else:
            self._emergency_save_step = msg['emergency_save_step']

    def _overwrite_path(self, path: Union[os.PathLike, str]) -> Path:
        path = Path(path)
        filename = path.name

        if not self._is_local and "CKPT_DIR" in os.environ:
            mp_rank = getattr(self._dist_config, 'mp_rank', 0)
            pp_rank = getattr(self._dist_config, 'pp_rank', 0)

            path = Path(os.environ["CKPT_DIR"]) / "iter_{:07d}/mp_rank_{:02d}_{:03d}/{}".format(
                self._cur_step, mp_rank, pp_rank, filename)

        return path

    def _local_log(self, msg):
        mode = "a" if self._has_locally_logged else "w"
        with self._log_path.open(mode=mode) as log_file:
            log_file.write(f"{json.dumps(msg)}\n")

    def init(self,
             total_train_steps: int,
             local_log_name: Optional[str] = None) -> None:
        """Initialize the training manager.

        Args:
            total_train_steps: The number of total training steps
            local_log_name: log file name for local mode

        Raises:
            PeriFlowError: when total_train_steps is not an integer
        """
        if self._is_local:
            if local_log_name is not None:
                self._log_path = Path(local_log_name)
            else:
                if torch.distributed.is_initialized():
                    # To prevent path overlap among processes, we add rank at the end of the log file name.
                    rank = torch.distributed.get_rank()
                    self._log_path = Path(f"./periflow_trainer_{int(time.time())}_{rank}.log")
                else:
                    self._log_path = Path(f"./periflow_trainer_{int(time.time())}.log")
        else:
            if not isinstance(total_train_steps, int):
                raise PeriFlowError(f'total_train_steps should be an integer, got {type(total_train_steps)}')

            asyncio.run(
                self._ipc_channels[IpcCommPurpose.LAST_STEP].write({
                    "step": total_train_steps
                }))

        self.has_initialized = True

    @check_initialized
    def start_step(self) -> None:
        """Start a new training step.

        Raises:
            PeriFlowError: when the previous step is not ended
        """
        if self._is_step_started:
            raise PeriFlowError(
                'Step already started. Maybe `end_step` is not called for the previous step?)')

        self._save_method = None
        if not self._is_local:
            self._cur_step += 1

        self._step_start_time = time.monotonic()

    @check_initialized
    def end_step(self) -> None:
        """Finish the current training step.

        Raises:
            PeriFlowError: when try to call `end_step` before calling `start_step`
            PeriFlowInternalError: IPC failed
        """
        if not self._is_step_started:
            raise PeriFlowError(
                'Cannot call `end_step` before calling `start_step`')

        step_time = time.monotonic() - self._step_start_time
        if not self._is_local:
            try:
                msg = {
                    "step": self._cur_step,
                    "step_time": step_time,
                    "saved": self._is_saved,
                    "save_type": self._save_method,
                }
                asyncio.run(self._ipc_channels[IpcCommPurpose.STEP_INFO].write(msg))

                # Wait for ack.
                ack = asyncio.run(self._ipc_channels[IpcCommPurpose.ACK].read())
                if ack["status"] != CommResultStatus.SUCCESS:
                    raise PeriFlowInternalError(f'Invalid IPC message from FTModule: {ack}')

                # If emergency save is done, terminate the training process.
                if self._save_method is SaveType.EMERGENCY:
                    sys.exit()

            except IpcConnectionError as e:
                raise PeriFlowInternalError('IPC connection between training manager and FTModule is broken.') from e

        self._step_start_time = None

    @contextmanager
    def train_step(self):
        """Context manager wrapper for `start_step` and `end_step`."""
        self.start_step()
        try:
            yield
        finally:
            self.end_step()

    @check_initialized
    def is_emergency_save(self) -> bool:
        """Check whether emergency save should be handled or not

        Returns: 'True' if emergency save step is equal to current step, 'False' if not.
        """
        return self._emergency_save_step == self._cur_step

    @check_initialized
    def metric(self, msg: Dict) -> None:
        """Log a key-value metric dict and send it to Periflow. `step` info is added if not exists.
        Args:
            msg: A key-value dict containing user-defined metrics.

        Returns: None
        """
        if self._is_local:
            self._local_log(msg)
        else:
            new_msg = msg.copy()

            new_msg["step"] = self._cur_step
            new_msg["rank"] = self._dist_config.rank
            new_msg["local_rank"] = self._dist_config.local_rank

            asyncio.run(self._ipc_channels[IpcCommPurpose.METRIC].write(new_msg))

    def is_checkpoint_exist(self, path: Union[os.PathLike, str]) -> bool:
        """Check whether the checkpoint file exists or not

        Args:
            path: Pathlike object which points target checkpoint file

        Returns:
            'True' if path exist 'False' if not
        """
        path = self._overwrite_path(path)
        return os.path.exists(path)

    def load(self, path: Union[os.PathLike, str, BinaryIO, IO[bytes]], *args, **kwargs) -> Any:
        """Wrapped method of `torch.load`

        Args:
            path: The path of target object. Ignored in cloud mode.
            args: other arguments that will be passed to `torch.load`
            kwargs: other keyword arguments that will be passed to `torch.load`

        Raises:
            PeriFlowError: when path is a file-like object

        Returns:
            Loaded object
        """
        if not isinstance(path, (str, Path)):
            raise PeriFlowError(
                f"Invalid argument {path}. pf.load does not support IO object. Use torch.load or pass PathLike object")

        path = self._overwrite_path(path)
        return torch.load(path, *args, **kwargs)

    @check_initialized
    def save(self,
             obj: Any,
             path: Union[os.PathLike, str, BinaryIO, IO[bytes]],
             *args,
             **kwargs):
        """Wrapped method of `torch.save`

        Args:
            obj: Object to be stored
            path: Path to be stored. Ignored in cluster mode.
            args: other arguments that will be passed to `torch.save`
            kwargs: other keyword arguments that will be passed to `torch.save`

        Raises:
            PeriFlowError: when path is a file-like object
        """
        if not isinstance(path, (str, Path)):
            raise PeriFlowError(
                f"Invalid argument {path}. pf.save does not support IO object. Use torch.save or pass PathLike object")

        if self._is_saved:
            raise PeriFlowError(
                "You cannot call `pf.save()` twice within a training step.")

        if not self._is_step_started:
            raise PeriFlowError(
                "You can only call `pf.save()` within a training step scope.")

        path = self._overwrite_path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(obj, path, *args, **kwargs)

        if self.is_emergency_save():
            self._save_method = SaveType.EMERGENCY
        else:
            self._save_method = SaveType.NORMAL


periflow = TrainingManager()
