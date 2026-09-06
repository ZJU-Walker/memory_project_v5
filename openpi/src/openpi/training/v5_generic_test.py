"""v5 generic task data mode (DataConfig.memory_v5_generic_task): the neutral v3.5 fields transform,
the generic manifest loader, and the config validation. The sequence sampler and the full pipeline are
exercised by the training smoke run (integration is the real gate)."""

import dataclasses
import hashlib
import json
import pathlib

import numpy as np
import pytest

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


def _generic_config(tmp_path: pathlib.Path, manifest_sha: str, **overrides) -> _config.DataConfig:
    base = dict(
        subtask_from_task=True,
        memory_stride_frames=5,
        memory_subtask_vocab=("a", "b"),
        memory_episode_manifest_path=str(tmp_path / "manifest.json"),
        memory_episode_manifest_sha256=manifest_sha,
        memory_manifest_split="train",
        memory_manifest_split_seed=7,
        memory_v5_subtask_labels_path=str(tmp_path / "sidecar.json"),
        memory_v5_subtask_labels_sha256="0" * 64,
        memory_v5_generic_task=True,
    )
    base.update(overrides)
    return _config.DataConfig(**base)


def _write_manifest(tmp_path: pathlib.Path, episodes: list[dict], seed: int = 7) -> str:
    payload = {"schema_version": "openpi.v5.generic-manifest.v1", "split_seed": seed, "episodes": episodes}
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    (tmp_path / "manifest.json").write_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_generic_fields_transform_contract():
    t = _transforms.MemoryV5GenericFields(num_fact_slots=3, num_fact_targets=3)
    step_mask = np.array([True, True, True, False])
    waiting = np.array([False, True, True, True])
    out = t({"seq_step_mask": step_mask, "seq_waiting_mask": waiting, "episode_memory_cell": np.int32(2)})
    np.testing.assert_array_equal(out["seq_write_mask"], step_mask)
    np.testing.assert_array_equal(out["seq_decision_mask"], [False, True, True, False])
    np.testing.assert_array_equal(out["seq_read_state_valid"], step_mask)
    np.testing.assert_array_equal(out["seq_read_credit_reachable"], step_mask)
    np.testing.assert_array_equal(out["seq_use_pressure_mask"], [False, True, True, False])
    np.testing.assert_array_equal(out["seq_decay_gap_before"], np.zeros(4, dtype=np.int32))
    assert out["seq_memory_cell"] == 2 and out["seq_memory_cell"].dtype == np.int32
    assert out["seq_side_label"] == -1
    np.testing.assert_array_equal(out["seq_fact_labels"], [2, 2, 2])
    assert out["seq_fact_observable"].shape == (4, 3) and not out["seq_fact_observable"].any()
    assert "episode_memory_cell" not in out


def test_generic_fields_refuses_v35_window():
    t = _transforms.MemoryV5GenericFields(num_fact_slots=3, num_fact_targets=3)
    with pytest.raises(ValueError, match="v3.5"):
        t({"seq_step_mask": np.ones(2, bool), "seq_waiting_mask": np.zeros(2, bool), "_v35_enabled": np.bool_(True)})
    with pytest.raises(ValueError, match="after MemoryV34Labels"):
        t({"seq_step_mask": np.ones(2, bool)})


def test_generic_manifest_loader(tmp_path: pathlib.Path):
    episodes = [
        {"episode_index": 0, "stable_id": "d/demo1", "expected_num_frames": 10, "split": "train", "class": "x=2"},
        {"episode_index": 1, "stable_id": "d/demo2", "expected_num_frames": 12, "split": "development", "class": "x=1"},
        {"episode_index": 2, "stable_id": "d/demo3", "expected_num_frames": 9, "split": "train", "class": "x=1"},
        {"episode_index": 99, "stable_id": "d/demo4", "expected_num_frames": 9, "split": "train", "include": False},
    ]
    sha = _write_manifest(tmp_path, episodes)
    cfg = _generic_config(tmp_path, sha)
    info = _data_loader._load_v5_generic_manifest(cfg, num_episodes=3, episode_lengths=np.array([10, 12, 9]))
    assert info["stable_id"] == ("d/demo1", "d/demo2", "d/demo3")
    np.testing.assert_array_equal(info["sampling_allowed"], [True, False, True])
    assert info["manifest_split"] == ("train", "development", "train")
    np.testing.assert_array_equal(info["memory_cell"], [1, 0, 0])
    assert info["memory_cell_names"] == ("x=1", "x=2")
    # wrong bytes, wrong seed, wrong length all fail closed
    with pytest.raises(ValueError, match="SHA256"):
        _data_loader._load_v5_generic_manifest(
            dataclasses.replace(cfg, memory_episode_manifest_sha256="1" * 64),
            num_episodes=3,
            episode_lengths=np.array([10, 12, 9]),
        )
    with pytest.raises(ValueError, match="split_seed"):
        _data_loader._load_v5_generic_manifest(
            dataclasses.replace(cfg, memory_manifest_split_seed=8), num_episodes=3, episode_lengths=np.array([10, 12, 9])
        )
    with pytest.raises(ValueError, match="frames"):
        _data_loader._load_v5_generic_manifest(cfg, num_episodes=3, episode_lengths=np.array([10, 12, 8]))
    with pytest.raises(ValueError, match="covers"):
        _data_loader._load_v5_generic_manifest(cfg, num_episodes=4, episode_lengths=np.array([10, 12, 9, 1]))


def test_generic_config_validation(tmp_path: pathlib.Path):
    sha = "a" * 64
    _generic_config(tmp_path, sha)  # valid
    with pytest.raises(ValueError, match="v3.5"):
        _generic_config(tmp_path, sha, memory_v35_enabled=True)
    with pytest.raises(ValueError, match="sidecar"):
        _generic_config(tmp_path, sha, memory_v5_subtask_labels_sha256=None)
    with pytest.raises(ValueError, match="manifest"):
        _generic_config(tmp_path, sha, memory_episode_manifest_sha256=None)
    with pytest.raises(ValueError, match="fact sidecar"):
        _generic_config(tmp_path, sha, memory_v4_fact_labels_path="x", memory_v4_fact_labels_sha256="b" * 64)
    with pytest.raises(ValueError, match="waiting core"):
        _generic_config(tmp_path, sha, memory_waiting_max_speed=1e-3)
    with pytest.raises(ValueError, match="memory_subtask_vocab"):
        _generic_config(tmp_path, sha, memory_subtask_vocab=())


def test_beans_configs_wire_the_generic_pipeline():
    for name in ("pi05_yam_mem_v5_beansA", "pi05_yam_mem_v5_beansB"):
        cfg = _config.get_config(name)
        bc = cfg.data.base_config
        assert bc.memory_v5_generic_task and not bc.memory_v35_enabled
        assert bc.memory_stride_frames == 5 and bc.subtask_lookahead == 0
        assert len(bc.memory_subtask_vocab) == 11 == len(cfg.model.memory_v5_reference_tokens)
        assert set(bc.evidence_subtasks) < set(bc.memory_subtask_vocab)
        assert set(bc.memory_required_subtasks) < set(bc.memory_subtask_vocab)
        assert cfg.model.memory_v5_prefill_max == 10
        dc = cfg.data.create(cfg.assets_dirs, cfg.model)
        names = [type(t).__name__ for t in dc.data_transforms.inputs]
        assert names[:3] == ["BuildMemorySequence", "MemoryV34Labels", "MemoryV5GenericFields"]
        structure = dc.repack_transforms.inputs[0].structure
        assert "episode_memory_cell" in structure and "memory_window" not in structure
        assert "episode_fact_targets" not in structure
    assert _config.get_config("pi05_yam_mem_v5_beansA").model.memory_v5_oracle_writes
    assert not _config.get_config("pi05_yam_mem_v5_beansB").model.memory_v5_oracle_writes


def test_generic_fields_passes_single_frame_inference_item_through():
    t = _transforms.MemoryV5GenericFields(num_fact_slots=3, num_fact_targets=3)
    item = {"observation/state": [0.0] * 14, "prompt": "scoop the beans"}
    assert t(dict(item)) == item
