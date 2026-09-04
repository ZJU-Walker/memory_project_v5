"""Pure-function tests for the v5 YAM client (metadata guard and overlay readout)."""

import importlib.util
import pathlib
import sys
import types
from unittest import mock

import pytest


def _load_client():
    """Import the client with the hardware/network modules stubbed (cv2, openpi_client, tyro)."""
    stubs = {
        "cv2": types.ModuleType("cv2"),
        "tyro": types.ModuleType("tyro"),
        "openpi_client": types.ModuleType("openpi_client"),
        "openpi_client.action_chunk_broker": types.ModuleType("openpi_client.action_chunk_broker"),
        "openpi_client.image_tools": types.ModuleType("openpi_client.image_tools"),
        "openpi_client.websocket_client_policy": types.ModuleType("openpi_client.websocket_client_policy"),
    }
    stubs["openpi_client"].__path__ = []
    path = pathlib.Path(__file__).with_name("client_memory_v5.py")
    spec = importlib.util.spec_from_file_location("_client_memory_v5_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {**stubs, spec.name: module}):
        spec.loader.exec_module(module)
    return module


client = _load_client()


def _metadata(**overrides):
    base = {
        "config_name": "pi05_yam_mem_v5_stageB6a",
        "memory_architecture": "v32_layer8_dual_query",
        "memory_v5_sentence_bank": True,
        "action_horizon": 50,
        "rtc_enabled": True,
        "rtc_delay_semantics": "inclusive_max",
        "rtc_max_delay": 6,
        "memory_stride_frames": 15,
    }
    base.update(overrides)
    return base


def test_validate_v5_metadata_accepts_the_b6a_contract():
    client.validate_v5_metadata(_metadata(), client.Args())


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"memory_v5_sentence_bank": False}, "v5 sentence-bank"),
        ({"memory_stride_frames": 10}, "trained at 15"),
        ({"rtc_max_delay": 4}, "RTC maximum"),
        ({"rtc_enabled": False}, "RTC-trained"),
        ({"action_horizon": 30}, "action_horizon"),
    ],
)
def test_validate_v5_metadata_rejects_mismatches(override, match):
    with pytest.raises(ValueError, match=match):
        client.validate_v5_metadata(_metadata(**override), client.Args())


def test_validate_v5_metadata_rejects_a_non_training_prompt():
    with pytest.raises(ValueError, match="training prompt"):
        client.validate_v5_metadata(_metadata(), client.Args(prompt="find the bin with banana"))


def test_memory_readout_shows_sentence_confidence_and_bank():
    result = {
        "subtask": "inspect both bins: banana left, grey pepper box right",
        "subtask_confidence": 0.97,
        "memory": {"changed": True, "committed": True},
        "bank": ["open both lids", "inspect both bins: banana left, grey pepper box right"],
        "writes": 2,
    }
    seen, held = client.memory_readout(result)
    assert seen == "sees: inspect both bins: banana left, grey pepper box right(0.97)*  (* = committed now)"
    assert held == "bank[2]: open both lids | inspect both bins: banana left, grey pepper box right  commits 2"
    seen, held = client.memory_readout({"subtask": "open both lids"})
    assert seen == "sees: open both lids  (* = committed now)" and held == "bank[0]: -  commits 0"
