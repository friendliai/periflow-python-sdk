# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Helper functions
"""

import functools
from enum import Enum


class SaveType(str, Enum):
    NORMAL = "NORMAL"
    EMERGENCY = "EMERGENCY"


def check_initialized(func):
    @functools.wraps(func)
    def wrapper(training_manager, *args, **kwargs):
        assert training_manager.has_initialized, f'{func.__name__} must be called after pf.init()!'
        return func(training_manager, *args, **kwargs)
    return wrapper


def ensure_divisibility(numerator: int, denominator: int):
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"
