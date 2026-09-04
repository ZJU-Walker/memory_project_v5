"""A5 history prefill (cluster_v5/README.md §8): the data-side sentence history, its tokenization
and the battery's handling of the prefilled bank content."""

import os
import sys

import numpy as np
import pytest

from openpi import transforms as _transforms
from openpi.models import tokenizer as _tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4_side_flip_eval as battery  # noqa: E402


def _subtasks(*, stride=2, lookahead=2, delay=1, prefill_max=6):
    # frames 0-3 "a", 4-7 "b", 8-11 "c", 12-15 "d"
    sentences = ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4
    ep_tasks = np.zeros(len(sentences), dtype=np.int64)
    return _transforms.MemorySequenceSubtasks(
        stride=stride,
        steps=2,
        lookahead=lookahead,
        episode_tasks=(ep_tasks,),
        tasks={0: "task"},
        episode_sentences=(sentences,),
        prefill_history=True,
        prefill_max=prefill_max,
        write_delay_steps=delay,
    )


def _item(frame):
    return {"episode_index": np.asarray(0), "frame_index": np.asarray(frame)}


def test_history_prefill_walks_the_stride_grid_back_from_the_window():
    out = _subtasks()(_item(10))
    # Pre-window steps at frames 2,4,6,8,10 (+lookahead 2 -> frames 4,6,8,10,12): a,b,b,c,c.
    # Delay 1: history a,b,b,c (distinct a@0, b@1, c@3; gaps 0,1,0), pending c.
    assert out["memory_v5_prefill"] == ["a", "b", "c", "", "", ""]
    np.testing.assert_array_equal(out["memory_v5_prefill_gaps"], [0, 1, 0, 0, 0, 0])
    assert out["memory_v5_pending"] == "c"
    assert out["subtask"] == ["d", "d"]  # the window's own lookahead-shifted labels (frames 12, 14)


def test_history_prefill_edge_cases():
    out0 = _subtasks()(_item(0))
    assert out0["memory_v5_prefill"] == [""] * 6 and out0["memory_v5_pending"] == ""
    out1 = _subtasks()(_item(1))  # still no full stride before the window
    assert out1["memory_v5_prefill"] == [""] * 6 and out1["memory_v5_pending"] == ""
    out2 = _subtasks()(_item(2))  # one pre-window step: pending only
    assert out2["memory_v5_prefill"] == [""] * 6 and out2["memory_v5_pending"] == "a"
    # Undelayed: the last produced sentence is part of the committed history.
    out_nodelay = _subtasks(delay=0)(_item(10))
    assert out_nodelay["memory_v5_prefill"] == ["a", "b", "c", "", "", ""]
    np.testing.assert_array_equal(out_nodelay["memory_v5_prefill_gaps"], [0, 1, 1, 0, 0, 0])
    assert out_nodelay["memory_v5_pending"] == ""
    # Buffer shorter than the history keeps the most recent sentences.
    out_short = _subtasks(prefill_max=2)(_item(10))
    assert out_short["memory_v5_prefill"] == ["b", "c"]
    np.testing.assert_array_equal(out_short["memory_v5_prefill_gaps"], [1, 0])


def test_tokenize_sentence_matches_the_causal_prefix():
    tok = _tokenizer.FASTSubtaskTokenizer(48)
    sentence = "Inspect both bins: banana left, grey pepper box right"
    tokens, mask = tok.tokenize_sentence(sentence, 48)
    _ctx, _cmask, causal, causal_mask, fast, *_ = tok.tokenize_split(
        "find the banana", np.zeros((14,), dtype=np.float32), sentence, np.zeros((1, 14), dtype=np.float32), 160
    )
    prefix = causal[causal_mask & ~fast]
    np.testing.assert_array_equal(tokens[mask], prefix)
    assert tokens[mask][-1] == 108  # trailing newline, as in the causal buffer
    empty_tokens, empty_mask = tok.tokenize_sentence("", 48)
    assert not empty_mask.any() and not empty_tokens.any()


def test_battery_flips_only_evidence_sentences_in_the_prefill():
    L, R, W = battery.LEFT_TOKEN, battery.RIGHT_TOKEN, battery.WAIT_TOKEN
    tokens = np.array([[[1, L, 2, 0], [W, 3, R, 0], [0, 0, 0, 0]]])
    mask = np.array([[[True, True, True, False], [True, True, True, False], [False] * 4]])
    swapped = battery.swap_side_tokens_in_sentences(tokens, mask)
    np.testing.assert_array_equal(swapped[0, 0], [1, R, 2, 0])  # evidence row flipped
    np.testing.assert_array_equal(swapped[0, 1], [W, 3, R, 0])  # waiting row untouched
    np.testing.assert_array_equal(swapped[0, 2], 0)

    class _Obs:
        memory_v5_prefill_tokens = tokens
        memory_v5_prefill_mask = mask
        memory_v5_pending_tokens = np.array([[5, 6, 0, 0]])
        memory_v5_pending_mask = np.array([[True, True, False, False]])

    np.testing.assert_array_equal(battery.prefill_holds_waiting_sentence(_Obs()), [True])
    _Obs.memory_v5_prefill_tokens = np.array([[[1, L, 2, 0], [4, 3, R, 0], [0, 0, 0, 0]]])
    np.testing.assert_array_equal(battery.prefill_holds_waiting_sentence(_Obs()), [False])
    _Obs.memory_v5_pending_tokens = np.array([[W, 6, 0, 0]])
    np.testing.assert_array_equal(battery.prefill_holds_waiting_sentence(_Obs()), [True])
    assert battery.prefill_holds_waiting_sentence(object()) is None


def test_first_step_summary_excludes_history_decided_windows():
    base = {"batch": 0, "included": True, "D_normal": 1.0, "D_reset": 1.0, "D_donor": 1.0, "donor_mismatched": False}
    records = [
        {**base, "row": 0, "decision_order": 0, "history_decided": False},
        {**base, "row": 1, "decision_order": 0, "history_decided": True},
        {**base, "row": 1, "decision_order": 1, "history_decided": True},
    ]
    first = battery.summarize(records, first_step_only=True)
    assert first["decision_steps"] == 1 and first["excluded_history_decided"] == 1
    every = battery.summarize(records)
    assert every["decision_steps"] == 3


def test_b5_pipeline_carries_the_prefill_keys_to_the_tokenizer():
    """The first B5 smoke died because two key whitelists (the repack structure and YamInputs)
    silently dropped the raw prefill strings before the tokenizer transform saw them."""
    from openpi.policies import yam_policy
    from openpi.training import config as _config

    cfg = _config.get_config("pi05_yam_mem_v5_stageB5")
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    repack = data_config.repack_transforms.inputs[0]
    for key in ("memory_v5_prefill", "memory_v5_prefill_gaps", "memory_v5_pending"):
        assert repack.structure[key] == key
    tokenize = [t for t in data_config.model_transforms.inputs if isinstance(t, _transforms.TokenizeMemorySubtaskInputs)]
    assert tokenize and tokenize[0].prefill_len == cfg.model.memory_v5_sentence_len
    yam = yam_policy.YamInputs(model_type=cfg.model.model_type)
    item = {
        "observation/state": np.zeros((2, 14), dtype=np.float32),
        "observation/image": np.zeros((2, 4, 4, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.zeros((2, 4, 4, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.zeros((2, 4, 4, 3), dtype=np.uint8),
        "actions": np.zeros((2, 4, 14), dtype=np.float32),
        "prompt": "find the banana",
        "memory_v5_prefill": ["open both lids", "", "", "", "", ""],
        "memory_v5_prefill_gaps": np.zeros((6,), dtype=np.int32),
        "memory_v5_pending": "inspect both bins: banana left, grey pepper box right",
    }
    out = yam(item)
    assert out["memory_v5_prefill"][0] == "open both lids"
    assert out["memory_v5_pending"].startswith("inspect")
    assert out["memory_v5_prefill_gaps"].shape == (6,)


def test_b5a_warm_start_loader_is_the_cast_audited_loader():
    """B from A5 (two stages, user 2026-09-03 20:20): every leaf of the A5 checkpoint is matched,
    nothing is fresh, and the bf16 base leaves are widened to f32 explicitly (the strict loader
    refuses them, as the B4 graft did)."""
    from openpi.training import config as _config
    from openpi.training import weight_loaders as wl

    cfg = _config.get_config("pi05_yam_mem_v5_stageB5a")
    loader = cfg.weight_loader
    assert isinstance(loader, wl.AuditedPartialCheckpointWeightLoader)
    assert loader.source_cast_dtype == "float32"
    assert loader.matched_allowlist == (".+",) and loader.fresh_init_allowlist == ()
    assert "pi05_yam_mem_v5_stageA5/v5_stageA5_20260903_r1/999/params" in loader.params_path
    assert cfg.model.memory_v5_oracle_writes is False and cfg.model.memory_v5_prefill_history
    assert cfg.lr_schedule.peak_lr < _config.get_config("pi05_yam_mem_v5_stageA5").lr_schedule.peak_lr
    with pytest.raises(wl.AuditedGraftError, match="lossless"):
        import dataclasses

        dataclasses.replace(loader, source_cast_dtype="bfloat16", manifest_output_path="/dev/null").load_with_manifest({})
