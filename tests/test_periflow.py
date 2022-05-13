# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

""" Unit test module for periflow main
"""

import asyncio
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import pytest

from periflow_sdk.comm.ipc import get_default_ipc_channel, IpcCommPurpose, IpcChannel, CommResultStatus
from periflow_sdk.errors import PeriFlowError
from periflow_sdk.manager import TrainingManager
from periflow_sdk.utils import SaveType


@pytest.fixture
def local_manager(monkeypatch):
    monkeypatch.setenv("PERIFLOW_ENABLED", "0")
    manager = TrainingManager(teardown_at_exit=False)
    manager.init(total_train_steps=5, local_log_name="./temp_log.txt")
    return manager


@pytest.fixture
def cloud_manager(monkeypatch):
    monkeypatch.setenv("PERIFLOW_ENABLED", "1")
    monkeypatch.setenv("CKPT_DIR", "/workspace/ckpt")
    monkeypatch.setenv("RANK", str(4))
    monkeypatch.setenv("NODE_RANK", str(1))
    monkeypatch.setenv("NUM_NODES", str(4))
    monkeypatch.setenv("WORLD_SIZE", str(16))
    monkeypatch.setenv("PROCESSED_ITERS", str(0))
    manager = TrainingManager(teardown_at_exit=False)
    manager.init(total_train_steps=5)
    return manager


@pytest.fixture
def cloud_manager_v2(monkeypatch):
    monkeypatch.setenv("PERIFLOW_ENABLED", "1")
    monkeypatch.setenv("CKPT_DIR", "/workspace/ckpt")
    monkeypatch.setenv("RANK", str(5))
    monkeypatch.setenv("NODE_RANK", str(1))
    monkeypatch.setenv("NUM_NODES", str(4))
    monkeypatch.setenv("WORLD_SIZE", str(16))
    monkeypatch.setenv("PROCESSED_ITERS", str(4))
    manager = TrainingManager(teardown_at_exit=False)
    manager.init(total_train_steps=5)
    return manager


def _send_ack_on_receive(step_info_channel: IpcChannel, ack_channel: IpcChannel):
    msg = asyncio.run(step_info_channel.read())
    asyncio.run(ack_channel.write(msg={"status": CommResultStatus.SUCCESS}))
    return msg


def _valid_step_info(msg: Dict):
    return "step" in msg and "step_time" in msg


def _send_emergency_save(emergency_channel: IpcChannel, step: int):
    asyncio.run(emergency_channel.write({"emergency_save_step": step}))
    return True


def test_step(cloud_manager):
    server_step_channel = get_default_ipc_channel(purpose=IpcCommPurpose.STEP_INFO,
                                                  local_rank=0)
    server_ack_channel = get_default_ipc_channel(purpose=IpcCommPurpose.ACK,
                                                 local_rank=0)
    server_step_channel.open()
    server_ack_channel.open()

    for i in range(4):
        with ThreadPoolExecutor(max_workers=1) as executor:
            f = executor.submit(_send_ack_on_receive, server_step_channel, server_ack_channel)
            cloud_manager.start_step()
            time.sleep(0.1)
            cloud_manager.end_step()
            stat_info_msg = f.result()
            assert _valid_step_info(stat_info_msg)
            assert stat_info_msg["step"] == i + 1

    with ThreadPoolExecutor(max_workers=1) as executor:
        f = executor.submit(_send_ack_on_receive, server_step_channel, server_ack_channel)
        cloud_manager.start_step()
        time.sleep(0.1)
        cloud_manager.end_step()
        stat_info_msg = f.result()
        assert _valid_step_info(stat_info_msg)

    server_step_channel.close()
    server_ack_channel.close()
    cloud_manager._teardown()


def test_step_error(local_manager):
    local_manager.start_step()
    with pytest.raises(PeriFlowError) as e:
        local_manager.start_step()
    local_manager.end_step()
    with pytest.raises(PeriFlowError) as e:
        local_manager.end_step()


def test_step_multi_ranks(cloud_manager, cloud_manager_v2):
    server_step_channel = get_default_ipc_channel(purpose=IpcCommPurpose.STEP_INFO,
                                                  local_rank=0)
    server_ack_channel = get_default_ipc_channel(purpose=IpcCommPurpose.ACK,
                                                 local_rank=0)
    server_step_channel.open()
    server_ack_channel.open()

    server_step_channel_2 = get_default_ipc_channel(purpose=IpcCommPurpose.STEP_INFO,
                                                    local_rank=1)
    server_ack_channel_2 = get_default_ipc_channel(purpose=IpcCommPurpose.ACK,
                                                   local_rank=1)
    server_step_channel_2.open()
    server_ack_channel_2.open()

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert cloud_manager._cur_step == 0
        assert cloud_manager_v2._cur_step == 4
        cloud_manager.start_step()
        cloud_manager_v2.start_step()
        executor.submit(_send_ack_on_receive, server_step_channel, server_ack_channel)
        time.sleep(1)
        cloud_manager.end_step()
        assert not cloud_manager._is_step_started
        assert cloud_manager_v2._is_step_started
        executor.submit(_send_ack_on_receive, server_step_channel_2, server_ack_channel_2)
        time.sleep(1)
        cloud_manager_v2.end_step()
        assert not cloud_manager_v2._is_step_started

    server_step_channel.close()
    server_ack_channel.close()
    server_step_channel_2.close()
    server_ack_channel_2.close()

    cloud_manager._teardown()
    cloud_manager_v2._teardown()


def test_upload_checkpoint_before_end_step(cloud_manager):
    ckpt_ipc_channel = get_default_ipc_channel(purpose=IpcCommPurpose.CKPT,
                                               local_rank=0)
    server_ack_channel = get_default_ipc_channel(purpose=IpcCommPurpose.ACK,
                                                 local_rank=0)

    with ckpt_ipc_channel:
        cloud_manager.start_step()
        cloud_manager.upload_checkpoint()

        result = asyncio.run(ckpt_ipc_channel.read())
        assert result["step"] == 1
        assert result["save_type"] == SaveType.NORMAL


def test_upload_checkpoint_after_end_step(cloud_manager):
    ckpt_ipc_channel = get_default_ipc_channel(purpose=IpcCommPurpose.CKPT,
                                               local_rank=0)
    server_ack_channel = get_default_ipc_channel(purpose=IpcCommPurpose.ACK,
                                                 local_rank=0)

    with ckpt_ipc_channel, server_ack_channel:
        cloud_manager.start_step()
        asyncio.run(server_ack_channel.write({
            "status": CommResultStatus.SUCCESS.value
        }))
        cloud_manager.end_step()
        cloud_manager.upload_checkpoint()

        result = asyncio.run(ckpt_ipc_channel.read())
        assert result["step"] == 1
        assert result["save_type"] == SaveType.NORMAL


def test_cloud_metric(cloud_manager):
    metric_ipc_channel = get_default_ipc_channel(purpose=IpcCommPurpose.METRIC,
                                                 local_rank=0)
    metric_ipc_channel.open()
    cloud_manager.start_step()
    float_metric = {'some_metric': 1.5}
    cloud_manager.metric(float_metric)
    result = asyncio.run(metric_ipc_channel.read())
    assert "some_metric" in result and result.get("some_metric") == float_metric.get("some_metric")
    assert result.get("step") == 1
    assert result.get("rank") == 4
    assert result.get("local_rank") == 0
    string_metric = {'another_metric': "hello"}
    cloud_manager.metric(string_metric)
    result = asyncio.run(metric_ipc_channel.read())
    assert "another_metric" in result and result.get("another_metric") == string_metric.get("another_metric")
    assert result.get("step") == 1
    assert result.get("rank") == 4
    assert result.get("local_rank") == 0
    metric_ipc_channel.close()
    cloud_manager._teardown()


def test_local_metric(local_manager):
    local_manager.start_step()
    float_metric = {'some_metric': 1.5}
    string_metric = {'another_metric': "hi"}
    local_manager.metric(float_metric)
    local_manager.metric(string_metric)
    with open("./temp_log.txt", "r") as log_file:
        metric = json.loads(log_file.readline().strip())
        assert metric["some_metric"] == 1.5
        metric = json.loads(log_file.readline().strip())
        assert metric["another_metric"] == "hi"
    os.unlink("./temp_log.txt")


def test_emergency_save(cloud_manager):
    server_step_channel = get_default_ipc_channel(purpose=IpcCommPurpose.STEP_INFO,
                                                  local_rank=0)
    server_ack_channel = get_default_ipc_channel(purpose=IpcCommPurpose.ACK,
                                                 local_rank=0)
    server_emergency_channel = get_default_ipc_channel(purpose=IpcCommPurpose.EMERGENCY_SAVE,
                                                       local_rank=0)
    server_step_channel.open()
    server_ack_channel.open()
    server_emergency_channel.open()
    cloud_manager.start_step()

    with ThreadPoolExecutor(max_workers=1) as executor:
        f = executor.submit(_send_emergency_save, server_emergency_channel, 2)
        while True:
            try:
                f.result(timeout=100)
            except TimeoutError:
                pass
            else:
                break
    time.sleep(0.1)
    assert not cloud_manager.is_emergency_save()
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(_send_ack_on_receive, server_step_channel, server_ack_channel)
        time.sleep(0.1)
        cloud_manager.end_step()

    cloud_manager.start_step()
    time.sleep(0.1)
    assert cloud_manager.is_emergency_save()

    server_step_channel.close()
    server_ack_channel.close()
    server_emergency_channel.close()
    cloud_manager._teardown()
