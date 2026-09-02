"""Contracts of the v5 detailed-subtask sentence sidecar builder."""

import hashlib
import json
import pathlib

import pytest

import v5_build_subtask_labels as builder


def _write_manifest(tmp_path: pathlib.Path, episodes: list[dict], *, frozen: bool = True) -> pathlib.Path:
    manifest = {
        "review_status": "frozen" if frozen else "draft",
        "dataset_version": "v36",
        "raw_root": ".",
        "episodes": episodes,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _episode(tmp_path: pathlib.Path, stable_id: str, obj: str, side: str, split: str, include: bool = True) -> dict:
    other = "left" if side == "right" else "right"
    segments = [
        {"task": "open both lids", "start": 0, "end": 9},
        {"task": "inspect both bins", "start": 10, "end": 19},
        {"task": "close both lids and reset arms", "start": 20, "end": 29},
        {"task": f"wait; target bin is {side}", "start": 30, "end": 39},
        {"task": f"open {side} bin", "start": 40, "end": 49},
    ]
    raw_dir = tmp_path / stable_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    label_bytes = json.dumps(segments).encode("utf-8")
    (raw_dir / "subtask_labels.json").write_bytes(label_bytes)
    del other
    return {
        "stable_id": stable_id,
        "raw_dir": stable_id,
        "label_file": "subtask_labels.json",
        "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "object": obj,
        "target_side": side,
        "split": split,
        "include": include,
        "expected_num_frames": 50,
    }


def test_sentences_follow_the_templates(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        [
            _episode(tmp_path, "part1/demo1", "banana", "right", "train"),
            _episode(tmp_path, "part1/demo2", "grey_pepper_box", "right", "development"),
            _episode(tmp_path, "part1/demo3", "banana", "left", "final_test", include=False),
        ],
    )
    payload = builder.build_subtask_labels(manifest, None)
    assert payload["schema_version"] == builder.SCHEMA_VERSION
    assert payload["num_episodes"] == 2
    demo1 = payload["episodes"]["part1/demo1"]
    assert demo1["object_sides"] == {"banana": "right", "grey_pepper_box": "left"}
    assert [s["sentence"] for s in demo1["segments"]] == [
        "open both lids",
        "inspect both bins: banana right, grey pepper box left",
        "close both lids and reset arms",
        "wait; target bin is right",
        "open right bin",
    ]
    demo2 = payload["episodes"]["part1/demo2"]
    assert demo2["segments"][1]["sentence"] == "inspect both bins: banana left, grey pepper box right"
    assert demo2["segments"][3]["sentence"] == "wait; target bin is right"
    # Segments tile the episode and the sidecar self-hashes canonically.
    assert demo1["segments"][0]["start"] == 0 and demo1["segments"][-1]["end"] == 49
    body = {k: v for k, v in payload.items() if k != "content_sha256"}
    canonical = json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == payload["content_sha256"]
    assert payload["source_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_builder_refuses_unfrozen_manifest_and_tampered_labels(tmp_path):
    episodes = [_episode(tmp_path, "part1/demo1", "banana", "left", "train")]
    manifest = _write_manifest(tmp_path, episodes, frozen=False)
    with pytest.raises(ValueError, match="not frozen"):
        builder.build_subtask_labels(manifest, None)
    manifest = _write_manifest(tmp_path, episodes)
    (tmp_path / "part1/demo1/subtask_labels.json").write_text("[]")
    with pytest.raises(ValueError, match="label_sha256"):
        builder.build_subtask_labels(manifest, None)


def test_builder_cross_checks_the_v4_fact_sidecar(tmp_path):
    episodes = [_episode(tmp_path, "part1/demo1", "banana", "left", "train")]
    manifest = _write_manifest(tmp_path, episodes)
    fact_path = tmp_path / "facts.json"
    fact_path.write_text(
        json.dumps(
            {
                "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "episodes": {"part1/demo1": {"fact_targets": [1, 0]}},  # banana right: disagrees
            }
        )
    )
    with pytest.raises(ValueError, match="disagree"):
        builder.build_subtask_labels(manifest, fact_path)
    fact_path.write_text(
        json.dumps(
            {
                "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "episodes": {"part1/demo1": {"fact_targets": [0, 1]}},
            }
        )
    )
    payload = builder.build_subtask_labels(manifest, fact_path)
    assert payload["source_v4_fact_labels_sha256"] == hashlib.sha256(fact_path.read_bytes()).hexdigest()
