# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Exceptions that can be raised during IPC
"""

class IpcException(Exception):
    pass


class IpcConnectionError(IpcException):
    pass


class IpcChannelNotOpenedError(IpcException):
    pass


class IpcChannelIOError(IpcException):
    pass
