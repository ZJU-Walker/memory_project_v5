"""How much do the v5 semantic read QUERIES change over an episode?

The 8 query keys come from the layer-8 states of the instruction rows (state digits excluded).
Those rows attend to the images in blocks 0..8, so the query is instruction-SELECTED but may be
image-DEPENDENT. This probe measures, for a few held-out episodes, (a) the cosine between the
query keys at step t and step 0 (same instruction, different frames), (b) the cosine between the
queries with and without the images (image masks off), (c) across episodes with the same / a
different prompt. Bank content is irrelevant (queries do not depend on it); a blank bank is used.

  python scripts/v5_probe_query_drift.py --config-name pi05_yam_mem_v5_stageA4 --params <params> \
      --episodes 1 2 21 61 --output <json>
"""

import argparse
import json
import pathlib
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.shared import project_paths
from openpi.training import config as _config
from openpi.training import data_loader as data_loader_lib


def make_query_fn(model):
    @nnx.jit
    def queries(model, observation, state_token_mask):
        preprocessed = _model.preprocess_observation(None, observation, train=False)
        batch = preprocessed.state.shape[0]
        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(preprocessed)
        num_img = prefix_mask.shape[1] - model.max_token_len
        top_tokens = num_img // len(preprocessed.images)
        prepared = model._v32_prepare_memory_prefix(  # noqa: SLF001
            prefix_tokens,
            prefix_mask,
            prefix_ar,
            model.memory.init_state(batch),
            top_token_count=top_tokens,
            state_token_mask=state_token_mask,
            semantic_state=model.memory_semantic.init_state(batch),
        )
        return prepared["sem_queries"].astype(jnp.float32)

    return queries


def _cos(a, b):  # [..., d] x [..., d] -> [...]
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return np.sum(a * b, axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_yam_mem_v5_stageA4")
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=(1, 2, 21, 61))
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    t0 = time.time()
    cfg = _config.get_config(args.config_name)
    model = cfg.model.load(_model.restore_params(args.params, restore_type=np.ndarray))
    model.eval()
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = data_loader_lib.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    tds = data_loader_lib.TransformedDataset(
        dataset,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, *data_config.model_transforms.inputs],
    )
    manifest = json.loads(project_paths.project_path(project_paths.V35_FROZEN_MANIFEST).read_text())
    sidecar = json.loads(project_paths.project_path(project_paths.V5_SUBTASK_LABELS).read_text())
    episodes = sorted([e for e in manifest["episodes"] if e.get("include")], key=lambda e: e["episode_index"])
    stride = data_config.memory_stride_frames
    steps_per_window = cfg.model.memory_seq_steps
    query_fn = make_query_fn(model)
    print(f"setup {time.time() - t0:.0f}s", flush=True)

    per_episode = {}
    for ep_index in args.episodes:
        episode = episodes[ep_index]
        start = sum(int(e["expected_num_frames"]) for e in episodes[:ep_index])
        length = int(episode["expected_num_frames"])
        segments = sidecar["episodes"][episode["stable_id"]]["segments"]
        frame_sentence = np.empty(length, dtype=object)
        for seg in segments:
            frame_sentence[seg["start"] : seg["end"] + 1] = seg["sentence"]
        q_img, q_noimg, phases, frames = [], [], [], []
        window_start = 0
        while window_start < length:
            first_t = 0
            while True:
                fetch_start = window_start - first_t * stride
                try:
                    item = tds[start + fetch_start]
                    break
                except ValueError as err:
                    if "E anchor" not in str(err) or fetch_start - stride < 0:
                        raise
                    first_t += 1
            window_start = fetch_start
            seq_obs = _model.Observation.from_dict(jax.tree.map(lambda x: np.asarray(x)[None], item))
            step_mask = np.asarray(seq_obs.seq_step_mask)[0]
            for t in range(first_t, steps_per_window):
                frame = window_start + t * stride
                if not step_mask[t] or frame >= length:
                    break
                obs = _model.Observation(
                    images={k: jnp.asarray(v[:, t]) for k, v in seq_obs.images.items()},
                    image_masks={k: jnp.asarray(v[:, t]) for k, v in seq_obs.image_masks.items()},
                    state=jnp.asarray(seq_obs.state[:, t]),
                    tokenized_prompt=jnp.asarray(seq_obs.tokenized_prompt[:, t]),
                    tokenized_prompt_mask=jnp.asarray(seq_obs.tokenized_prompt_mask[:, t]),
                )
                blind = obs.replace(image_masks={k: jnp.zeros_like(v) for k, v in obs.image_masks.items()})
                stm = jnp.asarray(seq_obs.token_state_mask[:, t])
                q_img.append(np.asarray(query_fn(model, obs, stm))[0])
                q_noimg.append(np.asarray(query_fn(model, blind, stm))[0])
                phases.append(str(frame_sentence[min(frame, length - 1)]).split(";")[0].split(":")[0])
                frames.append(frame)
            window_start = window_start + steps_per_window * stride
        q_img = np.stack(q_img)  # [T, 8, 512]
        q_noimg = np.stack(q_noimg)
        drift0 = _cos(q_img, q_img[:1])  # [T, 8] vs step 0
        consecutive = _cos(q_img[1:], q_img[:-1])
        image_dep = _cos(q_img, q_noimg)
        phase_names = sorted(set(phases), key=phases.index)
        by_phase = {
            p: float(np.mean(drift0[[i for i, ph in enumerate(phases) if ph == p]])) for p in phase_names
        }
        per_episode[ep_index] = {
            "prompt": episode["prompt"],
            "side": episode["target_side"],
            "steps": int(len(frames)),
            "cos_to_step0_mean": float(np.mean(drift0[1:])),
            "cos_to_step0_min": float(np.min(drift0[1:])),
            "cos_consecutive_mean": float(np.mean(consecutive)),
            "cos_with_vs_without_images_mean": float(np.mean(image_dep)),
            "cos_with_vs_without_images_min": float(np.min(image_dep)),
            "cos_to_step0_by_phase": by_phase,
            "per_head_cos_to_step0_mean": np.mean(drift0[1:], axis=0).round(3).tolist(),
            "_q": q_img,
        }
        r = per_episode[ep_index]
        print(
            f"ep{ep_index:02d} {r['prompt']!r} side={r['side']} steps={r['steps']}: cos(q_t,q_0) mean={r['cos_to_step0_mean']:.3f} "
            f"min={r['cos_to_step0_min']:.3f} consecutive={r['cos_consecutive_mean']:.3f} | with/without images "
            f"mean={r['cos_with_vs_without_images_mean']:.3f} min={r['cos_with_vs_without_images_min']:.3f} | by phase "
            f"{ {k: round(v, 3) for k, v in by_phase.items()} }",
            flush=True,
        )
    # Across episodes at matched steps: same prompt vs different prompt.
    cross = {}
    keys = list(per_episode)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = per_episode[keys[i]], per_episode[keys[j]]
            n = min(a["steps"], b["steps"])
            same = a["prompt"] == b["prompt"]
            c = float(np.mean(_cos(a["_q"][:n], b["_q"][:n])))
            cross[f"ep{keys[i]:02d}-ep{keys[j]:02d}"] = {"same_prompt": same, "cos_matched_steps_mean": c}
            print(f"ep{keys[i]:02d} vs ep{keys[j]:02d} ({'same' if same else 'different'} prompt): cos at matched steps = {c:.3f}", flush=True)
    for r in per_episode.values():
        r.pop("_q")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"config": args.config_name, "params": str(args.params), "episodes": per_episode, "cross": cross}, indent=2))
    print(f"wrote {args.output} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
