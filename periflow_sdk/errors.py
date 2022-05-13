# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Wrapped errors for PeriFlow sdk
"""

class PeriFlowError(Exception):
    pass


class PeriFlowInternalError(PeriFlowError):
    def __init__(self, message=None):
        if message is None:
            message = "Please contact to FriendliAI"
        else:
            message = message.rstrip() + "\nPlease contant to FriendliAI"

        super().__init__(message)
