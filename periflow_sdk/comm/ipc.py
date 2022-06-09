# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

""" IPC utils
"""

import asyncio
import errno
import json
import os
import struct
from enum import Enum
from typing import Optional

from periflow_sdk.comm.errors import (
    IpcChannelIOError,
    IpcChannelNotOpenedError,
    IpcConnectionError,
)
from periflow_sdk.errors import PeriFlowInternalError


class CommResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class IpcCommPurpose(str, Enum):
    STEP_INFO = "STEP_INFO"
    METRIC = "METRIC"
    ACK = "ACK"
    EMERGENCY_SAVE = "EMERGENCY_SAVE"
    LAST_STEP = "LAST_STEP"
    CKPT = "CKPT"


class FifoBase:
    """Abstraction for FIFO
    """
    def __init__(self, fifoname: str):
        self._fifo = None
        self._fifoname = fifoname
        _try_mkfifo(fifoname)

    @property
    def mode(self):
        raise NotImplementedError

    def open(self):
        if not self._fifo:
            self._fifo = os.open(self._fifoname, self.mode)

    def close(self):
        if self._fifo:
            try:
                os.close(self._fifo)
            except BrokenPipeError:
                pass
            self._fifo = None

    def remove(self):
        if self._fifo is not None:
            raise IpcChannelIOError("You should close the file first with FifoBase.close().")

        if os.path.exists(self._fifoname):
            os.remove(self._fifoname)


class FifoReader(FifoBase):
    """ Abstraction for FIFO reader side
    """
    @property
    def mode(self):
        # Read-only and non-blocking
        return os.O_RDONLY | os.O_NONBLOCK

    async def read(self) -> bytes:
        """Implementation of reading msg from the named pipe
        """
        # NOTE (TB): not compatible with Window OS
        # https://docs.python.org/3.8/library/asyncio-platforms.html#asyncio-platform-support
        assert self._fifo is not None
        loop = asyncio.get_running_loop()
        msg = bytearray()
        msg_size = None

        def _read(future: asyncio.Future):
            nonlocal msg_size
            try:
                if msg_size is None:
                    # read size first
                    msg_size_bytes = os.read(self._fifo, 4)
                    msg_size = _decode_msg_size(msg_size_bytes)

                msg_chunk = os.read(self._fifo, msg_size)
            except OSError as exc:
                if exc.errno == errno.EAGAIN:
                    future.set_result(b"")
                else:
                    future.set_exception(IpcConnectionError(str(exc)))
            except Exception as exc:  # pylint: disable=broad-except
                future.set_exception(IpcConnectionError(str(exc)))
            else:
                future.set_result(msg_chunk)
            finally:
                loop.remove_reader(self._fifo)

        while msg_size is None or msg_size > 0:
            future = loop.create_future()
            loop.add_reader(self._fifo, _read, future)
            msg_chunk = await future
            msg += msg_chunk
            msg_size -= len(msg_chunk)

        return bytes(msg)


class FifoWriter(FifoBase):
    """ Abstraction for FIFO writer side
    """
    @property
    def mode(self):
        return os.O_WRONLY | os.O_NONBLOCK

    async def write(self, content: bytes):
        """Implementation of writing msg into named pipe
        """
        assert self._fifo is not None
        loop = asyncio.get_running_loop()
        written_size = 0
        msg = _create_msg(content)
        msg_size = len(msg)

        def _write(future: asyncio.Future, m: bytes):
            try:
                wrote = os.write(self._fifo, m)
            except OSError as exc:
                if exc.errno == errno.EAGAIN:
                    future.set_result(0)
                else:
                    future.set_exception(IpcConnectionError(str(exc)))
            except Exception as exc:  # pylint: disable=broad-except
                future.set_exception(IpcConnectionError(str(exc)))
            else:
                future.set_result(wrote)
            finally:
                loop.remove_writer(self._fifo)

        while written_size < msg_size:
            future = loop.create_future()
            loop.add_writer(self._fifo, _write, future, msg)
            chunk_size = await future
            written_size += chunk_size
            msg = msg[chunk_size:]


class IpcChannel:
    """IPC communication channel
    """
    def __init__(self,
                 fifoname: str,
                 local_rank: Optional[int] = None):
        self._local_rank = local_rank
        self._fifoname = fifoname
        self._reader = FifoReader(fifoname)
        self._writer = FifoWriter(fifoname)
        self._opened = False

    async def read(self) -> dict:
        if not self._opened:
            error_msg = "IPC channel is not open. Call open() or __enter__ first."
            raise IpcChannelNotOpenedError(error_msg)

        msg = await self._reader.read()
        return json.loads(msg.decode())

    async def write(self, msg: dict):
        if not self._opened:
            error_msg = "IPC channel is not open. Call open() or __enter__ first."
            raise IpcChannelNotOpenedError(error_msg)

        bytes_msg = json.dumps(msg).encode()
        await self._writer.write(bytes_msg)

    def open(self):
        if self._opened:
            raise IpcChannelIOError("IPC channel is already open.")

        # NOTE: FIFO opening order
        # A process can open a FIFO in nonblocking mode. In this case, opening for read-only succeeds even if no one
        # has opened on the write side yet and opening for write-only fails with ENXIO (no such device or address)
        # unless the other end has already been opened.
        self._reader.open()
        self._writer.open()
        self._opened = True

    def close(self):
        if not self._opened:
            msg = "IPC channel is not open."
            raise IpcChannelNotOpenedError(msg)

        self._reader.close()
        self._writer.close()
        self._opened = False

    def remove(self):
        if self._opened:
            raise IpcChannelIOError(f"IPC channel is not closed yet: {self._fifoname}")
        self._reader.remove()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def fifoname(self) -> str:
        return self._fifoname

    @property
    def local_rank(self) -> Optional[int]:
        return self._local_rank

    @property
    def closed(self) -> bool:
        return not self._opened


def _encode_msg_size(size: int) -> bytes:
    """ Return a bytes object encoding the size of message.
    """
    return struct.pack("<I", size)


def _decode_msg_size(size_bytes: bytes) -> int:
    """ Return a message size in the integer format.
    """
    return struct.unpack("<I", size_bytes)[0]


def _create_msg(content: bytes) -> bytes:
    """ Create a message with the following format:

    ┌----------------┬--------------------------------┐
    | size (4 bytes) |        content (N bytes)       |
    └----------------┴--------------------------------┘
    """
    size = len(content)
    return _encode_msg_size(size) + content


def _try_mkfifo(fifoname: str):
    """ Create a FIFO (a named pipe) named path.
    """
    try:
        os.mkfifo(fifoname)
    except FileExistsError:
        pass


def get_default_ipc_channel(purpose: IpcCommPurpose, local_rank: int) -> IpcChannel:
    """ Create and return a IPC channel by the purpose.
    """
    if purpose == IpcCommPurpose.STEP_INFO:
        fifoname = f"/tmp/periflow_step_info_ipc_fifo_{local_rank}"
    elif purpose == IpcCommPurpose.METRIC:
        fifoname = f"/tmp/periflow_metric_ipc_fifo_{local_rank}"
    elif purpose == IpcCommPurpose.ACK:
        fifoname = f"/tmp/periflow_ack_ipc_fifo_{local_rank}"
    elif purpose == IpcCommPurpose.EMERGENCY_SAVE:
        fifoname = f"/tmp/periflow_emergency_save_ipc_fifo_{local_rank}"
    elif purpose == IpcCommPurpose.LAST_STEP:
        fifoname = f"/tmp/periflow_last_step_ipc_fifo_{local_rank}"
    elif purpose == IpcCommPurpose.CKPT:
        fifoname = f"/tmp/periflow_ckpt_ipc_fifo_{local_rank}"
    else:
        raise ValueError(f"Invalid purpose ({purpose}) is provided")
    return IpcChannel(fifoname, local_rank)


def ipc_read(channel: IpcChannel) -> dict:
    try:
        return asyncio.run(channel.read())
    except (IpcConnectionError, IpcChannelNotOpenedError) as exc:
        raise PeriFlowInternalError("IPC connection between PeriFlow and TrainingManager is Broken.") from exc


def ipc_status_read(channel: IpcChannel) -> dict:
    msg = ipc_read(channel)
    if msg["status"] != CommResultStatus.SUCCESS:
        raise PeriFlowInternalError(f"ERROR from PeriFlow: {msg}")
    return msg


def ipc_write(channel: IpcChannel, msg: dict) -> None:
    try:
        asyncio.run(channel.write(msg))
    except (IpcConnectionError, IpcChannelNotOpenedError) as exc:
        raise PeriFlowInternalError("IPC connection between PeriFlow and TrainingManager is Broken.") from exc
