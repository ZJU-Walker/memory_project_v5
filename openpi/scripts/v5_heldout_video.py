"""Held-out episode rollout video for a v5 checkpoint (cluster_v5/README.md §6).

Walks one episode at the training stride with the semantic bank carried across steps, greedily
decodes the subtask sentence at every step (memory read exactly as in training: layer-8 split,
both banks, blind memory rows), applies the v5 write rule to the decoded sentence ("self" mode:
write iff the sentence changed and its mean token probability >= memory_v5_write_conf; "oracle"
mode: write the label sentence whenever it changes, as in stage A/A2 training), and renders the
raw top-camera video with the ground-truth phase sentence, the training target, the decoded
sentence and the bank contents overlaid. H.264 via ffmpeg. Actions are not sampled.

  python scripts/v5_heldout_video.py --config-name pi05_yam_mem_v5_stageA2 --params <ckpt>/params \\
      --episode-index 2 --write-mode self --output-dir <dir>
"""

import argparse
import dataclasses
import json
import pathlib
import shutil
import subprocess
import time

import cv2
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece

import openpi.models.model as _model
import openpi.shared.project_paths as project_paths
import openpi.training.config as _config
import openpi.training.data_loader as data_loader_lib

PALIGEMMA_EOS_TOKEN = 1
STOP_TOKEN = 108  # "\n" — the trained sentence terminator (FASTSubtaskTokenizer.tokenize_split)


@dataclasses.dataclass
class StepRecord:
    step: int
    frame: int
    gt_now: str
    gt_target: str
    pred: str
    conf: float
    changed: bool
    written: bool
    decision: bool
    evidence: bool
    bank: list[str]
    sem_read_rms: float
    qk_cos_max: float


def _decode_text(sp, tokens):
    ids = [int(t) for t in tokens]
    while ids and ids[-1] in (STOP_TOKEN, PALIGEMMA_EOS_TOKEN, 0):
        ids.pop()
    return sp.decode(ids).strip()


def make_decode_fn(model, max_decode_steps: int):
    """One rollout step: read both banks (visual bank blank; injection follows the config), greedily
    decode the subtask sentence against the memory-extended cache, return tokens / per-token probs."""

    @nnx.jit
    def decode(model, observation, state_token_mask, sem_state):
        preprocessed = _model.preprocess_observation(None, observation, train=False)
        batch = preprocessed.state.shape[0]
        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(preprocessed)
        prefix_len = prefix_mask.shape[1]
        num_img = prefix_len - model.max_token_len
        top_tokens = num_img // len(preprocessed.images)
        mem_len = model._memory_token_total  # noqa: SLF001
        gen_base = prefix_len + mem_len
        prepared = model._v32_prepare_memory_prefix(  # noqa: SLF001
            prefix_tokens,
            prefix_mask,
            prefix_ar,
            model.memory.init_state(batch),
            top_token_count=top_tokens,
            state_token_mask=state_token_mask,
            semantic_state=sem_state,
        )
        kv_cache = prepared["cache"]
        final_prefix = prepared["final_prefix"]
        memory_valid = prepared["memory_valid"]
        causal_len = model.causal_token_len

        def logits_of(hidden_vec):
            return model.PaliGemma.llm(hidden_vec[:, None], method="decode")[:, 0].astype(jnp.float32)

        def pick(logits):
            probs = jax.nn.softmax(logits, axis=-1)
            token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            return token, jnp.take_along_axis(probs, token[:, None], axis=-1)[:, 0]

        token0, prob0 = pick(logits_of(model._v32_causal_seed(final_prefix, prefix_mask)[:, 0]))  # noqa: SLF001
        gen_tokens = jnp.zeros((batch, causal_len), dtype=jnp.int32)
        gen_mask = jnp.zeros((batch, causal_len), dtype=bool)
        gen_prob = jnp.zeros((batch, causal_len), dtype=jnp.float32)

        def record(tokens, mask, prob, done, token, p, index):
            tokens = tokens.at[:, index].set(jnp.where(done, tokens[:, index], token))
            mask = mask.at[:, index].set(~done)
            prob = prob.at[:, index].set(jnp.where(done, prob[:, index], p))
            return tokens, mask, prob, done | (token == STOP_TOKEN) | (token == PALIGEMMA_EOS_TOKEN)

        gen_tokens, gen_mask, gen_prob, done = record(
            gen_tokens, gen_mask, gen_prob, jnp.zeros(batch, dtype=bool), token0, prob0, 0
        )

        def cond(carry):
            return (carry[-1] < max_decode_steps) & ~jnp.all(carry[3])

        def step(carry):
            tokens, mask, prob, done, previous, cache, index = carry
            token_emb = model.PaliGemma.llm(previous[:, None], method="embed")
            (out, _), cache = model.PaliGemma.llm(
                [token_emb, None],
                mask=model._v32_step_mask(prefix_mask, index, memory_valid=memory_valid),  # noqa: SLF001
                positions=jnp.broadcast_to(gen_base + index - 1, (batch, 1)),
                kv_cache=cache,
                cache_position=gen_base + index - 1,
            )
            token, p = pick(logits_of(out[:, 0]))
            tokens, mask, prob, done = record(tokens, mask, prob, done, token, p, index)
            return tokens, mask, prob, done, token, cache, index + 1

        carry = (gen_tokens, gen_mask, gen_prob, done, token0, kv_cache, jnp.asarray(1, dtype=jnp.int32))
        gen_tokens, gen_mask, gen_prob, _, _, _, _ = jax.lax.while_loop(cond, step, carry)
        sem_rms = jnp.sqrt(jnp.mean(jnp.square(prepared["sem_retrieved"].astype(jnp.float32)), axis=(1, 2)))
        return gen_tokens, gen_mask, gen_prob, sem_rms, prepared.get("sem_queries")

    return decode


def make_write_fn(model):
    @nnx.jit
    def write(model, tokens, mask, sem_state, commit):
        encoded = model.v5_encode_sentence(tokens, mask)
        keys, values = model.v5_sentence_intent(encoded)
        new_state, aux = model.v5_semantic_write(sem_state, keys, values, commit)
        return new_state, aux["commit_applied"][:, 0], keys[:, 0]

    return write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_yam_mem_v5_stageA2")
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True, help="LeRobot episode index (manifest episode_index)")
    parser.add_argument("--write-mode", choices=("self", "oracle"), default="self")
    parser.add_argument(
        "--intervention",
        choices=("none", "flip_sides", "blank"),
        default="none",
        help="flip_sides: swap the side words (left<->right) in every sentence WRITTEN to the bank; "
        "blank: never commit (the semantic bank stays empty). Decode targets/overlays are unchanged.",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--max-decode-steps", type=int, default=24)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    sp = sentencepiece.SentencePieceProcessor(
        model_file=str(project_paths.project_path("v35/cache/openpi/big_vision/paligemma_tokenizer.model"))
    )
    manifest_path = project_paths.project_path(project_paths.V35_FROZEN_MANIFEST)
    manifest = json.loads(manifest_path.read_text())
    sidecar = json.loads(project_paths.project_path(project_paths.V5_SUBTASK_LABELS).read_text())
    episodes = sorted([e for e in manifest["episodes"] if e.get("include")], key=lambda e: e["episode_index"])
    episode = episodes[args.episode_index]
    assert episode["episode_index"] == args.episode_index
    start = sum(int(e["expected_num_frames"]) for e in episodes[: args.episode_index])
    length = int(episode["expected_num_frames"])
    segments = sidecar["episodes"][episode["stable_id"]]["segments"]
    frame_sentence = np.empty(length, dtype=object)
    for seg in segments:
        frame_sentence[seg["start"] : seg["end"] + 1] = seg["sentence"]
    raw_root = pathlib.Path(manifest["raw_root"])
    if not raw_root.is_absolute():
        raw_root = manifest_path.parent / raw_root
    video_path = (raw_root / episode["raw_dir"] / "top_camera_rgb.mp4").resolve()
    print(f"episode {args.episode_index} {episode['stable_id']} prompt={episode['prompt']!r} side={episode['target_side']} "
          f"frames={length} start={start} video={video_path} (setup {time.time() - t0:.0f}s)", flush=True)

    stride = data_config.memory_stride_frames
    steps_per_window = cfg.model.memory_seq_steps
    lookahead = data_config.subtask_lookahead
    sentence_len = cfg.model.memory_v5_sentence_len
    conf_threshold = cfg.model.memory_v5_write_conf
    decode = make_decode_fn(model, args.max_decode_steps)
    write = make_write_fn(model)

    sem_state = model.memory_semantic.init_state(1)
    prev_tokens = np.full((1, sentence_len), -1, dtype=np.int32)
    pending = (np.zeros((1, sentence_len), dtype=np.int32), np.zeros(sentence_len, dtype=bool), False)
    bank: list[str] = []
    bank_keys: list[np.ndarray] = []
    records: list[StepRecord] = []
    step_index = 0
    window_start = 0
    while window_start < length:
        # The training transform refuses a window whose decision steps have no evidence anchor
        # inside it (a D phase straddling a window boundary).  For the rollout we simply fetch
        # the window from an earlier frame so the anchor is included, and skip the steps already
        # processed; the memory state itself is carried step by step and is unaffected.
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
        if first_t:
            print(f"window at frame {window_start} fetched from frame {fetch_start} (skipping {first_t} steps)", flush=True)
        window_start = fetch_start
        batched = jax.tree.map(lambda x: np.asarray(x)[None], item)
        # The same conversion the training loader applies (uint8 images -> [-1, 1] float, field mapping).
        seq_obs = _model.Observation.from_dict(batched)
        step_mask = np.asarray(seq_obs.seq_step_mask)[0]
        decision_mask = np.asarray(seq_obs.seq_decision_mask)[0]
        write_mask = np.asarray(seq_obs.seq_write_mask)[0]
        causal = np.asarray(seq_obs.tokenized_causal)[0]
        causal_mask = np.asarray(seq_obs.tokenized_causal_mask)[0]
        causal_fast = np.asarray(seq_obs.causal_fast_mask)[0]
        next_start = window_start + steps_per_window * stride
        for t in range(first_t, steps_per_window):
            frame = window_start + t * stride
            if not step_mask[t] or frame >= length:
                break
            # One time slice, exactly as the training scan builds its per-step observation.
            observation = _model.Observation(
                images={k: jnp.asarray(v[:, t]) for k, v in seq_obs.images.items()},
                image_masks={k: jnp.asarray(v[:, t]) for k, v in seq_obs.image_masks.items()},
                state=jnp.asarray(seq_obs.state[:, t]),
                tokenized_prompt=jnp.asarray(seq_obs.tokenized_prompt[:, t]),
                tokenized_prompt_mask=jnp.asarray(seq_obs.tokenized_prompt_mask[:, t]),
            )
            state_token_mask = jnp.asarray(seq_obs.token_state_mask[:, t])
            gen_tokens, gen_mask, gen_prob, sem_rms, sem_queries = decode(model, observation, state_token_mask, sem_state)
            gen_tokens = np.asarray(gen_tokens)[0]
            gen_mask = np.asarray(gen_mask)[0]
            gen_prob = np.asarray(gen_prob)[0]
            pred = _decode_text(sp, gen_tokens[gen_mask])
            conf = float(gen_prob[gen_mask].mean()) if gen_mask.any() else 0.0
            # the sentence to (maybe) write
            if args.write_mode == "oracle":
                span = (causal_mask[t] & ~causal_fast[t])[:sentence_len]
                cur = np.where(span, causal[t][:sentence_len], 0).astype(np.int32)[None]
                confident = True
                # A3 training protocol: a waiting label is stored side-stripped ("wait\n").
                prefix = tuple(cfg.model.memory_v5_bank_waiting_prefix)
                if prefix and span[: len(prefix)].all() and tuple(cur[0, : len(prefix)].tolist()) == prefix:
                    tokens = tuple(cfg.model.memory_v5_bank_waiting_tokens)
                    cur = np.zeros((1, sentence_len), dtype=np.int32)
                    cur[0, : len(tokens)] = tokens
                    span = np.arange(sentence_len) < len(tokens)
            else:
                span = np.zeros(sentence_len, dtype=bool)
                n = int(min(gen_mask.sum(), sentence_len))
                span[:n] = True
                cur = np.zeros((1, sentence_len), dtype=np.int32)
                cur[0, :n] = gen_tokens[gen_mask][:n]
                confident = conf >= conf_threshold
            if getattr(cfg.model, "memory_v5_write_delay_steps", 0) == 1:
                # A4: write what was produced one step ago (nothing at the first step).
                produced = (cur.copy(), span.copy(), confident)
                cur, span, confident = pending
                pending = produced
            if args.intervention == "flip_sides":
                # PaliGemma ids: 2731 = " left", 1833 = " right" (the sidecar's side words).
                flipped = np.where(cur == 2731, 1833, np.where(cur == 1833, 2731, cur))
                cur = flipped.astype(np.int32)
            changed = bool(np.any(cur != prev_tokens)) and bool(span.any())
            commit = changed and confident and args.intervention != "blank"
            sem_state, applied, key = write(model, jnp.asarray(cur), jnp.asarray(span[None]), sem_state, jnp.asarray([commit]))
            applied = bool(np.asarray(applied)[0])
            prev_tokens = cur
            written_text = _decode_text(sp, cur[0][span]) if commit else ""
            if applied:
                bank.append(written_text)
                bank_keys.append(np.asarray(key)[0])
            qk = 0.0
            if sem_queries is not None and bank_keys:
                q = np.asarray(sem_queries)[0]
                qk = float(np.max(q @ np.stack(bank_keys).T))
            gt_now = str(frame_sentence[frame])
            gt_target = str(frame_sentence[min(frame + lookahead, length - 1)])
            records.append(
                StepRecord(step_index, frame, gt_now, gt_target, pred, conf, changed, applied, bool(decision_mask[t]),
                           bool(write_mask[t]), list(bank), float(np.asarray(sem_rms)[0]), qk)
            )
            flag = "W" if applied else " "
            d = "D" if decision_mask[t] else " "
            print(f"[{step_index:3d} f{frame:4d} {d}{flag}] pred={pred!r} conf={conf:.2f} | target={gt_target!r} | bank={len(bank)}", flush=True)
            step_index += 1
        window_start = next_start

    # ---- summary
    decisions = [r for r in records if r.decision]
    def side(s):
        return "left" if " left" in f" {s}" else "right" if " right" in f" {s}" else None
    first = decisions[0] if decisions else None
    summary = {
        "episode_index": args.episode_index,
        "stable_id": episode["stable_id"],
        "prompt": episode["prompt"],
        "target_side": episode["target_side"],
        "write_mode": args.write_mode,
        "intervention": args.intervention,
        "steps": len(records),
        "decision_steps": len(decisions),
        "decision_side_correct": sum(1 for r in decisions if side(r.pred) == episode["target_side"]),
        "first_decision_pred": first.pred if first else None,
        "first_decision_correct": (side(first.pred) == episode["target_side"]) if first else None,
        "evidence_pred_exact": sum(1 for r in records if r.evidence and r.pred == r.gt_target),
        "evidence_steps": sum(1 for r in records if r.evidence),
        "writes": sum(1 for r in records if r.written),
        "final_bank": bank,
        "records": [dataclasses.asdict(r) for r in records],
    }
    tag = f"ep{args.episode_index:02d}_{args.write_mode}" + ("" if args.intervention == "none" else f"_{args.intervention}")
    (args.output_dir / f"{tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"decision steps {summary['decision_side_correct']}/{summary['decision_steps']} name the true side; "
          f"first decision: {summary['first_decision_pred']!r} ({'OK' if summary['first_decision_correct'] else 'WRONG'}); "
          f"inspect sentence exact {summary['evidence_pred_exact']}/{summary['evidence_steps']}; writes={summary['writes']}", flush=True)

    # ---- render
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    band = 150
    out_path = args.output_dir / f"{tag}.mp4"
    ffmpeg = shutil.which("ffmpeg")
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height + band}",
         "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
         str(out_path)],
        stdin=subprocess.PIPE,
    )
    by_frame = {}
    for r in records:
        by_frame[r.frame] = r
    current = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_id = 0
    while True:
        ok, img = cap.read()
        if not ok or frame_id >= length:
            break
        if frame_id in by_frame:
            current = by_frame[frame_id]
        canvas = np.zeros((height + band, width, 3), dtype=np.uint8)
        canvas[:height] = img
        y = height + 22
        def put(text, color=(255, 255, 255), scale=0.55):
            nonlocal y
            cv2.putText(canvas, text[:110], (8, y), font, scale, color, 1, cv2.LINE_AA)
            y += 24
        put(f"{episode['stable_id']}  prompt: {episode['prompt']}  frame {frame_id}  [{args.write_mode} writes{'' if args.intervention == 'none' else ' / ' + args.intervention}]", (200, 200, 200), 0.5)
        put(f"GT phase : {frame_sentence[frame_id]}", (255, 255, 255))
        if current is not None:
            ok_pred = current.pred == current.gt_target
            put(f"PRED @{current.frame}: {current.pred}   (conf {current.conf:.2f})", (80, 220, 80) if ok_pred else (60, 60, 255))
            put(f"target   : {current.gt_target}" + ("   DECISION STEP" if current.decision else ""), (180, 180, 180))
            put(f"bank[{len(current.bank)}]: " + (" | ".join(current.bank[-3:]) if current.bank else "(empty)") + ("   <- WRITE" if current.written else ""), (255, 200, 80))
        if current is not None and current.decision and frame_id - current.frame < stride:
            cv2.rectangle(canvas, (2, 2), (width - 3, height - 3), (0, 200, 255), 3)
        proc.stdin.write(canvas.tobytes())
        frame_id += 1
    proc.stdin.close()
    proc.wait()
    cap.release()
    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit(f"ffmpeg failed ({proc.returncode})")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {frame_id} frames, H.264) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
