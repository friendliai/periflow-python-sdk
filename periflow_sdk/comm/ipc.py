""" IPC utils
"""

import asyncio
from asyncio.events import get_running_loop
import json
import os
import select
import struct
from enum import Enum
from contextlib import suppress
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import Future, CancelledError
from typing import Literal, List, Optional

from .errors import (
    IpcChannelIOError,
    IpcChannelNotOpenedError,
    IpcTimeoutException,
    IpcConnectionFailureException,
)


class CommResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class IpcCommPurpose(str, Enum):
    STAT = "STAT"
    ACK = "ACK"
    EMERGENCY_SAVE = "EMERGENCY_SAVE"


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

    def read(self, timeout: int = None) -> Optional[bytes]:
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
    def __init__(self,
                 fifoname: str,
                 local_rank: Optional[int] = None):
        self._local_rank = local_rank
        self._fifoname = fifoname
        self._reader = FifoReader(fifoname)
        self._writer = FifoWriter(fifoname)
        self._opened = False

    def read(self, timeout: int = None) -> dict:
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
        except BrokenPipeError:
            raise IpcConnectionFailureException

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


class IpcChannelBundle:
    """ A set of IPC channels between FTModule and training managers
    """
    def __init__(self, purpose: IpcCommPurpose, num_devices: int):
        self._thread_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="IpcChannelThreadPool")
        self._ipc_channels = []
        for local_rank in range(num_devices):
            ipc_channel = get_default_ipc_channel(purpose, local_rank)
            self._ipc_channels.append(ipc_channel)

    async def read_all(self, timeout: Optional[int] = None) -> List[dict]:
        """ Read all messages from the FIFOs under management.
        The call of this method will wait for the read for the specified length of time in milliseconds.
        If timeout is omitted, negative, or None, the call will block until it reads all the messages from every FIFO.
        """
        periflow_logger.debug("now read all")
        return await self._execute_in_thread_pool("read", timeout)

    async def write_all(self, msg: dict):
        """ Write all messages to the FIFOs under management.
        """
        await self._execute_in_thread_pool("write", msg)

    async def open_all(self):
        """ Open all FIFOs under management.
        """
        await self._execute_in_thread_pool("open")

    async def close_all(self):
        """ Close all FIFOs under management.
        """
        await self._execute_in_thread_pool("close")

    async def remove_all(self):
        """ Remove all FIFOs under management.
        """
        await self._execute_in_thread_pool("remove")

    async def _execute_in_thread_pool(self, method_name: str, *args):
        """ A helper function for resolving futures.
        If one of the futures raises an exception, the other futures are cancelled together.

        Arguments:
            - method_name: A method name of IpcChannel class.

        Return:
            - A list of results.
        """
        futures = []
        for ipc_channel in self._ipc_channels:
            futures.append(
                asyncio.get_running_loop().run_in_executor(
                    self._thread_pool, getattr(ipc_channel, method_name), *args
                )
            )

        try:
            return await asyncio.gather(*futures)
        except Exception as exc:
            periflow_logger.error(f"Exception: ({exc}) occurs.")
            for future in futures:
                future.cancel()
            for future in futures:
                with suppress(CancelledError):
                    future.result()

    async def __aenter__(self):
        await self.open_all()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close_all()

    @property
    def fifonames(self) -> List[str]:
        return [ipc_channel.fifoname for ipc_channel in self._ipc_channels]


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
    if purpose == IpcCommPurpose.STAT:
        fifoname = f"/tmp/periflow_stat_ipc_fifo_{local_rank}"
    elif purpose == IpcCommPurpose.ACK:
        fifoname = f"/tmp/periflow_ack_ipc_fifo_{local_rank}"
    elif purpose == IpcCommPurpose.EMERGENCY_SAVE:
        fifoname = f"/tmp/periflow_emergency_save_ipc_fifo_{local_rank}"
    else:
        raise ValueError(f"Invalid purpose ({purpose}) is provided")
    return IpcChannel(fifoname, local_rank)
