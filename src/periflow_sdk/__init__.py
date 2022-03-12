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
from enum import Enum
from pathlib import Path
from threading import Thread
from typing import Dict, Union, Any, Optional

import torch

from periflow_sdk.comm.errors import IpcConnectionError
from periflow_sdk.comm.ipc import IpcCommPurpose, CommResultStatus, get_default_ipc_channel, IpcChannel
from periflow_sdk.utils import ensure_valid_parallelism_config, ensure_divisibility, DistributeConfig

periflow_logger = logging.getLogger("periflow")
periflow_logger.setLevel(os.environ.get("PERIFLOW_LOG_LEVEL", "INFO"))
CKPT_FILE_NAME = "model_optim_rng.pt"


class SaveType(str, Enum):
    NORMAL = "NORMAL"
    EMERGENCY = "EMERGENCY"


class TrainingManager:

    """ The training wrapper class for general PyTorch training code.
    """
    def __init__(self, teardown_at_exit: bool = True):
        self._is_local = os.environ.get("PERIFLOW_ENABLED") != "1"

        self._cur_step = -1
        self._is_saved = False
        self._save_method = SaveType.NORMAL
        self._checkpoint_path = None
        self._step_start_time = None
        self._has_initialized = False
        self._dist_config = None

        # IPC Channels
        self._ipc_channels: Dict[IpcCommPurpose, IpcChannel] = {}

        # Emergency save
        self._wait_emergency_save_thread = None
        self._emergency_save_step = None

        # Used only for local mode
        self._log_path = None
        self._has_locally_logged = False

        if not self._is_local:
            periflow_logger.debug("Periflow SDK is working in cloud mode.")

            # Environment variable check.
            required_env_vars = ["RANK",
                                 "WORLD_SIZE",
                                 "NODE_RANK",
                                 "NUM_NODES",
                                 "PROCESSED_ITERS"]

            for env_var in required_env_vars:
                assert env_var in os.environ, f"Environment variable '{env_var}' should be set in cloud mode!"

            # Configure dist info
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])
            num_nodes = int(os.environ["NUM_NODES"])
            ensure_divisibility(world_size, num_nodes)
            devices_per_node = world_size // num_nodes
            local_rank = rank % devices_per_node
            self._cur_step = int(os.environ["PROCESSED_ITERS"])
            self._dist_config = DistributeConfig(local_rank=local_rank, rank=rank)
            ensure_valid_parallelism_config(self._dist_config)

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
    def _is_step_started(self) -> bool:
        return self._step_start_time is not None

    def init(self,
             total_train_steps: int,
             local_log_name: Optional[str] = None) -> None:
        """ Initialize training manager.

        Arguments:
            - total_train_steps: The number of total training steps
            - local_rank: The local rank of this training process
        """
        if self._is_local:
            periflow_logger.debug("Periflow SDK is working in local mode.")
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
            asyncio.run(
                self._ipc_channels[IpcCommPurpose.LAST_STEP].write({
                    "step": total_train_steps
                }))

        self._has_initialized = True

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
        except IpcConnectionError:
            pass
        else:
            self._emergency_save_step = msg['emergency_save_step']

    def get_current_step(self) -> int:
        assert self._has_initialized, "get_current_step() must be called after init()!"
        return self._cur_step

    def start_step(self) -> None:
        """
        Start a new training step.
        Returns: None
        """
        assert self._has_initialized, "start_step() must be called after init()!"
        assert not self._is_step_started, "Existing steps must finish before calling start_step()!"
        self._step_start_time = time.monotonic()
        self._is_saved = False
        if not self._is_local:
            self._cur_step += 1

    def end_step(self) -> None:
        """
        Finish the current training step.
        Returns: None
        """
        assert self._has_initialized, "end_step() must be called after init()!"
        assert self._is_step_started, "Existing steps must start before calling end_step()!"
        step_time = time.monotonic() - self._step_start_time
        if not self._is_local:
            try:
                msg = {
                    "step": self._cur_step,
                    "step_time": step_time,
                    "saved": self._is_saved,
                    "save_type": self._save_method,
                    "checkpoint_path": self._checkpoint_path
                }
                periflow_logger.debug(f"IPC WR || send training stat: {msg}")
                asyncio.run(self._ipc_channels[IpcCommPurpose.STEP_INFO].write(msg))

                # Wait for ack.
                periflow_logger.debug("Wait for ACK.")
                ack = asyncio.run(self._ipc_channels[IpcCommPurpose.ACK].read())
                periflow_logger.debug("ACK received.")
                if ack["status"] != CommResultStatus.SUCCESS:
                    raise RuntimeError(f"Invalid IPC message from FTModule: {ack}")

                # If emergency save is done, terminate the training process.
                if self._is_saved and self._save_method is SaveType.EMERGENCY:
                    sys.exit()

            except IpcConnectionError as ipc_connection_failure:
                raise RuntimeError("IPC connection between training manager and FTModule is broken.") \
                    from ipc_connection_failure
        self._step_start_time = None

    @contextmanager
    def train_step(self):
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
        assert self._has_initialized, "is_emergency_save() must be called after init()!"
        return self._emergency_save_step == self._cur_step

    def _local_log(self, msg):
        assert self._log_path is not None
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
        assert self._has_initialized, "metric() must be called after init()!"
        new_msg = msg.copy()
        if not self._is_local:
            assert self._dist_config is not None
            new_msg["step"] = self._cur_step
            new_msg["rank"] = self._dist_config.rank
            new_msg["local_rank"] = self._dist_config.local_rank
        if self._is_local:
            self._local_log(new_msg)
        else:
            asyncio.run(self._ipc_channels[IpcCommPurpose.METRIC].write(new_msg))

    def _get_cloud_path(self) -> Path:
        assert self._dist_config is not None
        mp_rank = self._dist_config.mp_rank
        pp_rank = self._dist_config.pp_rank
        path = Path(os.environ["CKPT_DIR"]) / "iter_{:07d}/mp_rank_{:02d}_{:03d}".format(
            self._cur_step, mp_rank, pp_rank) / CKPT_FILE_NAME
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def is_checkpoint_exist(self, path: Union[os.PathLike, str]) -> bool:
        if not self._is_local and "CKPT_DIR" in os.environ:
            path = self._get_cloud_path()
        return os.path.exists(path)

    def load(self, path: Union[os.PathLike, str], *args, **kwargs) -> Any:
        """
        Load the saved object from persistent storage. In local mode, this is same as `torch.save()`.
        In cloud mode, it ignores the 'path' and loads from the cloud path.
        Args:
            path: The path of target object. Ignored in cloud mode.

        Returns: Loaded object

        """
        if not self._is_local and "CKPT_DIR" in os.environ:
            path = self._get_cloud_path()
        return torch.load(path, *args, **kwargs)

    def save(self, obj, path: Union[os.PathLike, str], async_save: bool = False) -> None:
        """
        Save the desired object to persistent storage.
        Args:
            obj: Object to be stored
            path: Path to be stored. Ignored in cluster mode.
            async_save: Save is done asynchronously if true. Currently not supported.

        Returns: None

        """
        assert self._has_initialized, "save() must be called after init()!"
        assert not self._is_saved, "You cannot call `pf.save()` twice within a training step."
        assert self._is_step_started, "You can only call `pf.save()` within a training step scope."
        if async_save:
            raise NotImplementedError("Asynchronous checkpointing is not supported for now.")
        # Override path in cluster mode.
        if not self._is_local and "CKPT_DIR" in os.environ:
            path = self._get_cloud_path()
        torch.save(obj, path)
        self._is_saved = True
        if self.is_emergency_save():
            self._save_method = SaveType.EMERGENCY
        else:
            self._save_method = SaveType.NORMAL
        self._checkpoint_path = str(Path(path).resolve())


periflow = TrainingManager()
