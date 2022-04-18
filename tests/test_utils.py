"""Test module for periflow_sdk/utils.py
"""
import pytest

from periflow_sdk.utils import ensure_divisibility


def test_ensure_divisibility():
    ensure_divisibility(10, 2)

    with pytest.raises(AssertionError):
        ensure_divisibility(9, 2)
