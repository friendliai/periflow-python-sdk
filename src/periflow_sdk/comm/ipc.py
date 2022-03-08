""" IPC utils
"""

import json
import os
import select
import struct
from enum import Enum
from typing import Optional
import logging

from periflow_sdk.comm.errors import (
    IpcChannelIOError,
    IpcChannelNotOpenedError,
    IpcTimeoutException,
    IpcConnectionFailureException,
)

periflow_logger = logging.getLogger("periflow")


class CommResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class IpcCommPurpose(str, Enum):
    STEP_INFO = "STEP_INFO"
    ACK = "ACK"
    EMERGENCY_SAVE = "EMERGENCY_SAVE"
    METRIC = "METRIC"
    LAST_STEP = "LAST_STEP"


class FifoBase:
    """ Abstraction for FIFO
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
        os.remove(self._fifoname)


class FifoReader(FifoBase):
    """ Abstraction for FIFO reader side
    """
    @property
    def mode(self):
        # Read-only and non-blocking
        return os.O_RDONLY | os.O_NONBLOCK

    def read(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """ Read a message from the FIFO.
        If timeout is given, it specifies the length of time in milliseconds which the system will wait for events
        before returning. If timeout is omitted, negative, or None, the call will block until there is an event for the
        poll object.
        """
        # Create a polling object to monitor the fifo for new message.
        poll = select.poll()
        poll.register(self._fifo, select.POLLIN)
        try:
            if (self._fifo, select.POLLIN) in poll.poll(timeout):
                msg = _get_message(self._fifo)
                return msg

            if self._fifo is not None:
                raise IpcTimeoutException

            # TODO(TB): raise error, not None
            return None
        finally:
            if self._fifo is not None:
                poll.unregister(self._fifo)


class FifoWriter(FifoBase):
    """ Abstraction for FIFO writer side
    """
    @property
    def mode(self):
        return os.O_WRONLY

    def write(self, content: bytes):
        """ Write a message to the FIFO.
        """
        msg = _create_msg(content)
        os.write(self._fifo, msg)


class IpcChannel:
    """ The IPC Channel for communication between processes
    """
    def __init__(self,
                 fifoname: str,
                 local_rank: Optional[int] = None):
        self._local_rank = local_rank
        self._fifoname = fifoname
        self._reader = FifoReader(fifoname)
        self._writer = FifoWriter(fifoname)
        self._opened = False

    def read(self, timeout: Optional[float] = None) -> dict:
        if not self._opened:
            msg = "IPC channel is not open. Call open() or __enter__ first."
            raise IpcChannelNotOpenedError(msg)

        msg = self._reader.read(timeout=timeout)
        if msg is None or msg == b"":
            raise IpcConnectionFailureException
        return json.loads(msg.decode())

    def write(self, msg: dict):
        if not self._opened:
            msg = "IPC channel is not open. Call open() or __enter__ first."
            raise IpcChannelNotOpenedError(msg)

        msg = json.dumps(msg).encode()
        try:
            self._writer.write(msg)
        except BrokenPipeError as broken_pipe_error:
            raise IpcConnectionFailureException from broken_pipe_error

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
    def local_rank(self) -> int:
        return self._local_rank


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


def _get_message(fifo: int) -> bytes:
    """ Get a message from the named pipe.
    """
    msg_size_bytes = os.read(fifo, 4)
    msg_size = _decode_msg_size(msg_size_bytes)
    msg_content = os.read(fifo, msg_size)

    return msg_content


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
    elif purpose == IpcCommPurpose.ACK:
        fifoname = f"/tmp/periflow_ack_ipc_fifo_{local_rank}"
    elif purpose == IpcCommPurpose.EMERGENCY_SAVE:
        fifoname = f"/tmp/periflow_emergency_save_ipc_fifo_{local_rank}"
    elif purpose == IpcCommPurpose.METRIC:
        fifoname = f"/tmp/periflow_metric_ipc_fifo_{local_rank}"
    elif purpose == IpcCommPurpose.LAST_STEP:
        fifoname = f"/tmp/periflow_last_step_ipc_fifo_{local_rank}"
    else:
        raise ValueError(f"Invalid purpose ({purpose}) is provided")
    return IpcChannel(fifoname, local_rank)
