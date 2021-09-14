"""Exceptions that can be raised during IPC
"""

class IpcException(Exception):
    pass


class IpcTimeoutException(IpcException):
    pass


class IpcConnectionFailureException(IpcException):
    pass


class IpcChannelNotOpenedError(IpcException):
    pass


class IpcChannelIOError(IpcException):
    pass
