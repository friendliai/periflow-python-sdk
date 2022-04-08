""" Unit test module for periflow main
"""
import asyncio
import io
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

import pytest
from periflow_sdk import TrainingManager, SaveType
from periflow_sdk.comm.ipc import get_default_ipc_channel, IpcCommPurpose, IpcChannel, CommResultStatus
from periflow_sdk.errors import PeriFlowError, PeriFlowInternalError

TOTAL_TRAIN_STEPS = 5
LOCAL_RANK = 0
ANOTHER_LOCAL_RANK = 1
LOG_FILE_NAME = "./temp_log_txt"
CKPT_PATH = "./ckpt.pt"
CLOUD_CKPT_DIR = "./cloud"
DP_DEGREE = 2
MP_DEGREE = 4
PP_DEGREE = 2
RANK = 4
PARALLELISM_ORDER = "dp,pp,mp"
NODE_RANK = 1
NUM_NODES = 4
WORLD_SIZE = 16


@pytest.fixture
def local_manager(monkeypatch):
    monkeypatch.setenv("PERIFLOW_ENABLED", "0")
    manager = TrainingManager(teardown_at_exit=False)
    manager.init(total_train_steps=TOTAL_TRAIN_STEPS, local_log_name=LOG_FILE_NAME)
    return manager


@pytest.fixture
def cloud_manager(monkeypatch):
    monkeypatch.setenv("PERIFLOW_ENABLED", "1")
    monkeypatch.setenv("CKPT_DIR", CLOUD_CKPT_DIR)
    monkeypatch.setenv("DP_DEGREE", str(DP_DEGREE))
    monkeypatch.setenv("MP_DEGREE", str(MP_DEGREE))
    monkeypatch.setenv("PP_DEGREE", str(PP_DEGREE))
    monkeypatch.setenv("PARALLELISM_ORDER", PARALLELISM_ORDER)
    monkeypatch.setenv("RANK", str(RANK))
    monkeypatch.setenv("NODE_RANK", str(NODE_RANK))
    monkeypatch.setenv("NUM_NODES", str(NUM_NODES))
    monkeypatch.setenv("WORLD_SIZE", str(WORLD_SIZE))
    monkeypatch.setenv("PROCESSED_ITERS", str(0))
    manager = TrainingManager(teardown_at_exit=False)
    manager.init(total_train_steps=TOTAL_TRAIN_STEPS)
    return manager


@pytest.fixture
def cloud_manager_v2(monkeypatch):
    monkeypatch.setenv("PERIFLOW_ENABLED", "1")
    monkeypatch.setenv("CKPT_DIR", CLOUD_CKPT_DIR)
    monkeypatch.setenv("DP_DEGREE", str(DP_DEGREE))
    monkeypatch.setenv("MP_DEGREE", str(MP_DEGREE))
    monkeypatch.setenv("PP_DEGREE", str(PP_DEGREE))
    monkeypatch.setenv("PARALLELISM_ORDER", PARALLELISM_ORDER)
    monkeypatch.setenv("RANK", str(RANK + 1))
    monkeypatch.setenv("NODE_RANK", str(NODE_RANK))
    monkeypatch.setenv("NUM_NODES", str(NUM_NODES))
    monkeypatch.setenv("WORLD_SIZE", str(WORLD_SIZE))
    monkeypatch.setenv("PROCESSED_ITERS", str(5))
    manager = TrainingManager(teardown_at_exit=False)
    manager.init(total_train_steps=TOTAL_TRAIN_STEPS)
    return manager


def _send_ack_on_receive(step_info_channel: IpcChannel, ack_channel: IpcChannel):
    msg = asyncio.run(step_info_channel.read())
    asyncio.run(ack_channel.write(msg={"status": CommResultStatus.SUCCESS}))
    return msg


def _valid_step_info(msg: Dict):
    return "step" in msg \
        and "step_time" in msg \
        and "saved" in msg \
        and "save_type" in msg


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

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert cloud_manager._cur_step == 0
        assert cloud_manager_v2._cur_step == 5
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


def test_local_save_load(local_manager):
    local_manager.start_step()
    obj = {"Hello": 1.0}
    local_manager.save(obj, CKPT_PATH)
    local_manager.end_step()

    read_obj = local_manager.load(CKPT_PATH)
    assert read_obj == obj
    local_manager._teardown()
    os.unlink(CKPT_PATH)


def test_cloud_save_load(cloud_manager, cloud_manager_v2):
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
        expected_ckpt_path = (Path(CLOUD_CKPT_DIR) /
                              "iter_{:07d}/mp_rank_{:02d}_{:03d}".format(
                                  1,
                                  cloud_manager._dist_config.mp_rank,
                                  cloud_manager._dist_config.pp_rank) /  # pylint: disable=protected-access
                              'ckpt.pt')

    read_obj = cloud_manager.load(expected_ckpt_path)
    assert read_obj == obj

    expected_ckpt_path.unlink()

    # Save once again with the same manager.
    with ThreadPoolExecutor(max_workers=1) as executor:
        f = executor.submit(_send_ack_on_receive, server_step_channel, server_ack_channel)
        cloud_manager.start_step()
        time.sleep(0.1)
        obj = {"Hello": 1.5}
        cloud_manager.save(obj, CKPT_PATH)
        cloud_manager.end_step()
        stat_info_msg = f.result()
        assert _valid_step_info(stat_info_msg)
        assert stat_info_msg["saved"]
        assert stat_info_msg["save_type"] == SaveType.NORMAL
        expected_ckpt_path = (Path(CLOUD_CKPT_DIR) /
                              "iter_{:07d}/mp_rank_{:02d}_{:03d}".format(
                                  2,
                                  cloud_manager._dist_config.mp_rank,
                                  cloud_manager._dist_config.pp_rank) /  # pylint: disable=protected-access
                              'ckpt.pt')

    read_obj = cloud_manager.load(expected_ckpt_path)
    assert read_obj == obj

    expected_ckpt_path.unlink()
    expected_ckpt_path.parent.rmdir()
    expected_ckpt_path.parent.parent.rmdir()

    server_step_channel.close()
    server_ack_channel.close()
    cloud_manager._teardown()

    # Start from existing step...
    server_step_channel = get_default_ipc_channel(purpose=IpcCommPurpose.STEP_INFO,
                                                  local_rank=ANOTHER_LOCAL_RANK)
    server_ack_channel = get_default_ipc_channel(purpose=IpcCommPurpose.ACK,
                                                 local_rank=ANOTHER_LOCAL_RANK)
    server_step_channel.open()
    server_ack_channel.open()
    with ThreadPoolExecutor(max_workers=1) as executor:
        f = executor.submit(_send_ack_on_receive, server_step_channel, server_ack_channel)
        cloud_manager_v2.start_step()
        time.sleep(0.1)
        obj = {"Hello": 2.0}
        cloud_manager_v2.save(obj, CKPT_PATH)
        cloud_manager_v2.end_step()
        stat_info_msg = f.result()
        assert _valid_step_info(stat_info_msg)
        assert stat_info_msg["saved"]
        assert stat_info_msg["save_type"] == SaveType.NORMAL
        expected_ckpt_path = (Path(CLOUD_CKPT_DIR) /
                              "iter_{:07d}/mp_rank_{:02d}_{:03d}".format(
                                  6,
                                  cloud_manager_v2._dist_config.mp_rank,
                                  cloud_manager_v2._dist_config.pp_rank) /  # pylint: disable=protected-access
                              'ckpt.pt')

    read_obj = cloud_manager_v2.load(expected_ckpt_path)
    assert read_obj == obj

    expected_ckpt_path.unlink()
    expected_ckpt_path.parent.rmdir()
    expected_ckpt_path.parent.parent.rmdir()

    server_step_channel.close()
    server_ack_channel.close()
    cloud_manager_v2._teardown()


def test_load_save_with_io(cloud_manager):
    file_like = io.BytesIO()
    with pytest.raises(PeriFlowError):
        cloud_manager.load(file_like)

    cloud_manager.start_step()
    obj = {"some value": 2.1}
    with pytest.raises(PeriFlowError):
        cloud_manager.save(file_like, obj)


def test_duplicate_save_error(local_manager):
    local_manager.start_step()
    obj = {"some value": 2.1}
    local_manager.save(obj, CKPT_PATH)
    Path(CKPT_PATH).unlink()
    with pytest.raises(PeriFlowError) as e:
        local_manager.save(obj, CKPT_PATH)
    local_manager.end_step()


def test_save_out_of_scope(local_manager):
    local_manager.start_step()
    obj = {"some value": 2.1}
    local_manager.end_step()
    with pytest.raises(PeriFlowError) as e:
        local_manager.save(obj, CKPT_PATH)


def test_cloud_metric(cloud_manager):
    metric_ipc_channel = get_default_ipc_channel(purpose=IpcCommPurpose.METRIC,
                                                 local_rank=LOCAL_RANK)
    metric_ipc_channel.open()
    cloud_manager.start_step()
    float_metric = {'some_metric': 1.5}
    cloud_manager.metric(float_metric)
    result = asyncio.run(metric_ipc_channel.read())
    assert "some_metric" in result and result.get("some_metric") == float_metric.get("some_metric")
    assert result.get("step") == 1
    assert result.get("rank") == RANK
    assert result.get("local_rank") == LOCAL_RANK
    string_metric = {'another_metric': "hello"}
    cloud_manager.metric(string_metric)
    result = asyncio.run(metric_ipc_channel.read())
    assert "another_metric" in result and result.get("another_metric") == string_metric.get("another_metric")
    assert result.get("step") == 1
    assert result.get("rank") == RANK
    assert result.get("local_rank") == LOCAL_RANK
    metric_ipc_channel.close()
    cloud_manager._teardown()


def test_local_metric(local_manager):
    local_manager.start_step()
    float_metric = {'some_metric': 1.5}
    string_metric = {'another_metric': "hi"}
    local_manager.metric(float_metric)
    local_manager.metric(string_metric)
    local_manager._teardown()
    with open(LOG_FILE_NAME, "r") as log_file:
        metric = json.loads(log_file.readline().strip())
        assert metric["some_metric"] == 1.5
        metric = json.loads(log_file.readline().strip())
        assert metric["another_metric"] == "hi"
    os.unlink(LOG_FILE_NAME)


def _send_emergency_save(emergency_channel: IpcChannel, step: int):
    asyncio.run(emergency_channel.write({"emergency_save_step": step}))
    return True


def test_emergency_save(cloud_manager):
    server_step_channel = get_default_ipc_channel(purpose=IpcCommPurpose.STEP_INFO,
                                                  local_rank=LOCAL_RANK)
    server_ack_channel = get_default_ipc_channel(purpose=IpcCommPurpose.ACK,
                                                 local_rank=LOCAL_RANK)
    server_emergency_channel = get_default_ipc_channel(purpose=IpcCommPurpose.EMERGENCY_SAVE,
                                                       local_rank=LOCAL_RANK)
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
