""" Unit test module for periflow main
"""

import os
import time
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Dict

import pytest
import torch
from periflow_sdk import TrainingManager, SaveType
from periflow_sdk.comm.ipc import get_default_ipc_channel, IpcCommPurpose, IpcChannel, CommResultStatus

TOTAL_TRAIN_STEPS = 5
LOCAL_RANK = 0
ANOTHER_LOCAL_RANK = 1
LOG_FILE_NAME = "./temp_log_txt"
CKPT_PATH = "./ckpt.pt"


@pytest.fixture
def local_manager():
    manager = TrainingManager(log_file_name=LOG_FILE_NAME, is_local=True, teardown_at_exit=False)
    manager.init(TOTAL_TRAIN_STEPS, LOCAL_RANK)
    return manager


@pytest.fixture
def cloud_manager():
    manager = TrainingManager(is_local=False, teardown_at_exit=False)
    manager.init(TOTAL_TRAIN_STEPS, LOCAL_RANK)
    return manager


@pytest.fixture
def cloud_manager_v2():
    manager = TrainingManager(is_local=False, teardown_at_exit=False)
    manager.init(TOTAL_TRAIN_STEPS, ANOTHER_LOCAL_RANK)
    return manager


def _send_ack_on_receive(step_info_channel: IpcChannel, ack_channel: IpcChannel):
    msg = step_info_channel.read(timeout=1000)
    ack_channel.write(msg={"status": CommResultStatus.SUCCESS})
    return msg


def _valid_step_info(msg: Dict):
    return "step" in msg \
        and "is_last_step" in msg \
        and "step_time" in msg \
        and "saved" in msg \
        and "save_type" in msg \
        and "checkpoint_path" in msg


def test_step(cloud_manager):
    server_step_channel = get_default_ipc_channel(purpose=IpcCommPurpose.STEP_INFO,
                                                  local_rank=LOCAL_RANK)
    server_ack_channel = get_default_ipc_channel(purpose=IpcCommPurpose.ACK,
                                                 local_rank=LOCAL_RANK)
    server_step_channel.open()
    server_ack_channel.open()

    for i in range(TOTAL_TRAIN_STEPS - 1):
        with ThreadPoolExecutor(max_workers=1) as executor:
            f = executor.submit(_send_ack_on_receive, server_step_channel, server_ack_channel)
            cloud_manager.start_step()
            time.sleep(0.1)
            cloud_manager.end_step()
            stat_info_msg = f.result()
            assert _valid_step_info(stat_info_msg)
            assert stat_info_msg["step"] == i + 1
            assert not stat_info_msg["is_last_step"]

    with ThreadPoolExecutor(max_workers=1) as executor:
        f = executor.submit(_send_ack_on_receive, server_step_channel, server_ack_channel)
        cloud_manager.start_step()
        time.sleep(0.1)
        cloud_manager.end_step()
        stat_info_msg = f.result()
        assert _valid_step_info(stat_info_msg)
        assert stat_info_msg["is_last_step"]

    server_step_channel.close()
    server_ack_channel.close()
    cloud_manager._teardown()


def test_step_multi_ranks(cloud_manager, cloud_manager_v2):
    server_step_channel = get_default_ipc_channel(purpose=IpcCommPurpose.STEP_INFO,
                                                  local_rank=LOCAL_RANK)
    server_ack_channel = get_default_ipc_channel(purpose=IpcCommPurpose.ACK,
                                                 local_rank=LOCAL_RANK)
    server_step_channel.open()
    server_ack_channel.open()

    server_step_channel_2 = get_default_ipc_channel(purpose=IpcCommPurpose.STEP_INFO,
                                                    local_rank=ANOTHER_LOCAL_RANK)
    server_ack_channel_2 = get_default_ipc_channel(purpose=IpcCommPurpose.ACK,
                                                   local_rank=ANOTHER_LOCAL_RANK)
    server_step_channel_2.open()
    server_ack_channel_2.open()

    with ThreadPoolExecutor(max_workers=1) as executor:
        cloud_manager.start_step()
        cloud_manager_v2.start_step()
        executor.submit(_send_ack_on_receive, server_step_channel, server_ack_channel)
        time.sleep(0.1)
        cloud_manager.end_step()
        assert not cloud_manager._is_step_started
        assert cloud_manager_v2._is_step_started
        executor.submit(_send_ack_on_receive, server_step_channel_2, server_ack_channel_2)
        time.sleep(0.1)
        cloud_manager_v2.end_step()
        assert not cloud_manager_v2._is_step_started

    server_step_channel.close()
    server_ack_channel.close()
    server_step_channel_2.close()
    server_ack_channel_2.close()

    cloud_manager._teardown()
    cloud_manager_v2._teardown()


def test_save(cloud_manager):
    server_step_channel = get_default_ipc_channel(purpose=IpcCommPurpose.STEP_INFO,
                                                  local_rank=LOCAL_RANK)
    server_ack_channel = get_default_ipc_channel(purpose=IpcCommPurpose.ACK,
                                                 local_rank=LOCAL_RANK)
    server_step_channel.open()
    server_ack_channel.open()

    with ThreadPoolExecutor(max_workers=1) as executor:
        f = executor.submit(_send_ack_on_receive, server_step_channel, server_ack_channel)
        cloud_manager.start_step()
        time.sleep(0.1)
        obj = {"Hello": 1.0}
        cloud_manager.save(obj, CKPT_PATH)
        cloud_manager.end_step()
        stat_info_msg = f.result()
        assert _valid_step_info(stat_info_msg)
        assert stat_info_msg["saved"]
        assert stat_info_msg["save_type"] == SaveType.NORMAL
        assert stat_info_msg["checkpoint_path"] == str(Path(CKPT_PATH).resolve())

    read_obj = torch.load(CKPT_PATH)
    assert read_obj == obj

    server_step_channel.close()
    server_ack_channel.close()
    cloud_manager._teardown()
    os.unlink(CKPT_PATH)


def test_cloud_log(cloud_manager):
    metric_ipc_channel = get_default_ipc_channel(purpose=IpcCommPurpose.METRIC,
                                                 local_rank=LOCAL_RANK)
    metric_ipc_channel.open()
    cloud_manager.start_step()
    float_metric = {'some_metric': 1.5}
    cloud_manager.log(float_metric)
    result = metric_ipc_channel.read()
    assert "some_metric" in result and result.get("some_metric") == float_metric.get("some_metric")
    assert result.get("step") == 1
    string_metric = {'another_metric': "hello"}
    cloud_manager.log(string_metric)
    result = metric_ipc_channel.read()
    assert "another_metric" in result and result.get("another_metric") == string_metric.get("another_metric")
    assert result.get("step") == 1
    metric_ipc_channel.close()
    cloud_manager._teardown()


def test_local_log(local_manager):
    local_manager.start_step()
    float_metric = {'some_metric': 1.5}
    string_metric = {'another_metric': "hi"}
    local_manager.log(float_metric)
    local_manager.log(string_metric)
    with open(LOG_FILE_NAME, "r") as log_file:
        metric = json.loads(log_file.readline().strip())
        assert metric["some_metric"] == 1.5 and metric["step"] == 1
        metric = json.loads(log_file.readline().strip())
        assert metric["another_metric"] == "hi" and metric["step"] == 1
    local_manager._teardown()
    os.unlink(LOG_FILE_NAME)


def _send_emergency_save(emergency_channel: IpcChannel):
    emergency_channel.write({"emergency_save_step": 1})
    return True


def test_emergency_save(cloud_manager):
    server_emergency_channel = get_default_ipc_channel(purpose=IpcCommPurpose.EMERGENCY_SAVE,
                                                       local_rank=LOCAL_RANK)
    server_emergency_channel.open()
    cloud_manager.start_step()
    with ThreadPoolExecutor(max_workers=1) as executor:
        f = executor.submit(_send_emergency_save, server_emergency_channel)
        while True:
            try:
                f.result(timeout=100)
            except TimeoutError:
                pass
            else:
                break
    time.sleep(1)
    assert cloud_manager.is_emergency_save()
    server_emergency_channel.close()
    cloud_manager._teardown()
