"""Test module for periflow_sdk/utils.py
"""
import pytest

from periflow_sdk.utils import ensure_divisibility, DistributeConfig


def test_ensure_divisibility():
    ensure_divisibility(10, 2)

    with pytest.raises(AssertionError):
        ensure_divisibility(9, 2)


def test_ensure_valid_parallelism_config_some_configs_not_provided(monkeypatch):
    monkeypatch.setenv("PP_DEGREE", "2")
    monkeypatch.setenv("MP_DEGREE", "2")
    monkeypatch.setenv("PARALLELISM_ORDER", "dp,pp,mp")
    monkeypatch.setenv("WORLD_SIZE", "8")

    with pytest.raises(RuntimeError):
        dist_config = DistributeConfig(local_rank=0, rank=0)


def test_ensure_valid_parallelism_config1(monkeypatch):
    """Single node example

    MP GROUPS: (0, 1), (2, 3), (4, 5), (6, 7)
    PP GROUPS: (0, 2), (1, 3), (4, 6), (5, 7)
    DP GROUPS: (0, 4), (1, 5), (2, 6), (3, 7)
    """
    monkeypatch.setenv("PP_DEGREE", "2")
    monkeypatch.setenv("DP_DEGREE", "2")
    monkeypatch.setenv("MP_DEGREE", "2")
    monkeypatch.setenv("PARALLELISM_ORDER", "dp,pp,mp")
    monkeypatch.setenv("WORLD_SIZE", "8")

    dist_config = DistributeConfig(local_rank=0, rank=0)
    assert dist_config.dp_rank == 0
    assert dist_config.pp_rank == 0
    assert dist_config.mp_rank == 0

    dist_config = DistributeConfig(local_rank=0, rank=1)
    assert dist_config.dp_rank == 0
    assert dist_config.pp_rank == 0
    assert dist_config.mp_rank == 1

    dist_config = DistributeConfig(local_rank=0, rank=2)
    assert dist_config.dp_rank == 0
    assert dist_config.pp_rank == 1
    assert dist_config.mp_rank == 0

    dist_config = DistributeConfig(local_rank=0, rank=3)
    assert dist_config.dp_rank == 0
    assert dist_config.pp_rank == 1
    assert dist_config.mp_rank == 1

    dist_config = DistributeConfig(local_rank=0, rank=4)
    assert dist_config.dp_rank == 1
    assert dist_config.pp_rank == 0
    assert dist_config.mp_rank == 0

    dist_config = DistributeConfig(local_rank=0, rank=5)
    assert dist_config.dp_rank == 1
    assert dist_config.pp_rank == 0
    assert dist_config.mp_rank == 1

    dist_config = DistributeConfig(local_rank=0, rank=6)
    assert dist_config.dp_rank == 1
    assert dist_config.pp_rank == 1
    assert dist_config.mp_rank == 0

    dist_config = DistributeConfig(local_rank=0, rank=7)
    assert dist_config.dp_rank == 1
    assert dist_config.pp_rank == 1
    assert dist_config.mp_rank == 1


def test_ensure_valid_parallelism_config2(monkeypatch):
    """Single node example, but parallel order change

    MP GROUPS: (0, 1), (2, 3), (4, 5), (6, 7)
    DP GROUPS: (0, 2), (1, 3), (4, 6), (5, 7)
    PP GROUPS: (0, 4), (1, 5), (2, 6), (3, 7)
    """
    monkeypatch.setenv("PP_DEGREE", "2")
    monkeypatch.setenv("DP_DEGREE", "2")
    monkeypatch.setenv("MP_DEGREE", "2")
    monkeypatch.setenv("PARALLELISM_ORDER", "pp,dp,mp")
    monkeypatch.setenv("WORLD_SIZE", "8")

    dist_config = DistributeConfig(local_rank=0, rank=0)
    assert dist_config.dp_rank == 0
    assert dist_config.pp_rank == 0
    assert dist_config.mp_rank == 0

    dist_config = DistributeConfig(local_rank=0, rank=1)
    assert dist_config.dp_rank == 0
    assert dist_config.pp_rank == 0
    assert dist_config.mp_rank == 1

    dist_config = DistributeConfig(local_rank=0, rank=2)
    assert dist_config.dp_rank == 1
    assert dist_config.pp_rank == 0
    assert dist_config.mp_rank == 0

    dist_config = DistributeConfig(local_rank=0, rank=3)
    assert dist_config.dp_rank == 1
    assert dist_config.pp_rank == 0
    assert dist_config.mp_rank == 1

    dist_config = DistributeConfig(local_rank=0, rank=4)
    assert dist_config.dp_rank == 0
    assert dist_config.pp_rank == 1
    assert dist_config.mp_rank == 0

    dist_config = DistributeConfig(local_rank=0, rank=5)
    assert dist_config.dp_rank == 0
    assert dist_config.pp_rank == 1
    assert dist_config.mp_rank == 1

    dist_config = DistributeConfig(local_rank=0, rank=6)
    assert dist_config.dp_rank == 1
    assert dist_config.pp_rank == 1
    assert dist_config.mp_rank == 0

    dist_config = DistributeConfig(local_rank=0, rank=7)
    assert dist_config.dp_rank == 1
    assert dist_config.pp_rank == 1
    assert dist_config.mp_rank == 1


def test_ensure_valid_parallelism_config3(monkeypatch):
    """Multi nodes example

    Node 1
    --------------------------------
    | 0, 1, 2, 3, 4, 5, 6, 7       |
    --------------------------------

    Node 2
    --------------------------------
    | 8, 9, 10, 11, 12, 13, 14, 15 |
    --------------------------------

    MP GROUPS: (0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)  # intra VMP communications
    PP GROUPS: (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15)  # intra PP communications
    DP GROUPS: (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15)  # inter DP communications
    """
    monkeypatch.setenv("PP_DEGREE", "2")
    monkeypatch.setenv("DP_DEGREE", "2")
    monkeypatch.setenv("MP_DEGREE", "4")
    monkeypatch.setenv("PARALLELISM_ORDER", "dp,pp,mp")
    monkeypatch.setenv("WORLD_SIZE", "16")

    dist_config = DistributeConfig(local_rank=0, rank=0)
    assert dist_config.dp_rank == 0
    assert dist_config.pp_rank == 0
    assert dist_config.mp_rank == 0

    dist_config = DistributeConfig(local_rank=0, rank=2)
    assert dist_config.dp_rank == 0
    assert dist_config.pp_rank == 0
    assert dist_config.mp_rank == 2

    dist_config = DistributeConfig(local_rank=0, rank=7)
    assert dist_config.dp_rank == 0
    assert dist_config.pp_rank == 1
    assert dist_config.mp_rank == 3

    dist_config = DistributeConfig(local_rank=0, rank=13)
    assert dist_config.dp_rank == 1
    assert dist_config.pp_rank == 1
    assert dist_config.mp_rank == 1

    dist_config = DistributeConfig(local_rank=0, rank=15)
    assert dist_config.dp_rank == 1
    assert dist_config.pp_rank == 1
    assert dist_config.mp_rank == 3
