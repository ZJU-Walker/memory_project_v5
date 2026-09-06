"""Bean-scoop BANK-GEOMETRY probe (cluster_v5/README.md §8, 2026-09-05 18:20, user "will this be a sentence
direction issue?").

The tray probe showed that "dump vs done" ignores the bank entirely. Two mechanisms can make an old note
unreadable, and they call for different fixes:

  * DECAY      -- alpha_step multiplies the whole fast-weight matrix by (1 - alpha) every stride step, so a note
                  written 30 s ago is at 17-56 % strength (README §8 17:00);
  * INTERFERENCE -- the delta rule makes each new note exact at the cost of older notes whose KEY DIRECTION
                  overlaps it, and the go sentence shares "scoop" with every scoop note written after it.

This measures both, with no decoder involved: encode the v6 sentences with the checkpoint's own encoder, print the
key-key cosines, then for each episode replay the true note sequence into a fresh bank on the real stride-5 clock
(write on a sentence change, one analytic decay step otherwise) and, at every tray arrival, read the bank back with
the GO note's own key. Reported per tray step:

  go_recall     cosine(read_key(state, go_key), go_value)     1.0 = the go note is perfectly recoverable
  go_vs_recent  cosine(read_key(state, go_key), latest_value) how much the go query returns the NEWEST note instead
  recent_recall cosine(read_key(state, latest_key), latest_value)   control: the note the working decisions read

`--alphas` replays the same sequence under other decays: alpha 0 isolates interference (no fading at all), so
  go_recall high at alpha 0 but low at 0.01  -> decay is the binding constraint (the A6sd/B6sd retrain is the fix);
  go_recall low at alpha 0 too               -> the sentence DIRECTIONS collide and the fix is on the label side.
Report: v5/diagnostics/bank_geometry_<tag>/bank_geometry_eval.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np

SCHEMA_VERSION = "v5_bank_geometry_eval/1"


def steps_of(segments: list[dict], num_frames: int, stride: int) -> list[str]:
    """The label sentence at every stride-th frame (the training/rollout step grid)."""
    per_frame = [""] * num_frames
    for seg in segments:
        for f in range(int(seg["start"]), int(seg["end"]) + 1):
            per_frame[f] = seg["sentence"]
    return [per_frame[f] for f in range(0, num_frames, stride)]


def main(argv=None) -> None:
    import jax
    import jax.numpy as jnp

    from openpi.models import model as model_lib
    from openpi.training import config as config_lib
    from openpi.training import weight_loaders

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--config-name", default="pi05_yam_mem_v5_beansA6")
    parser.add_argument("--split", default="development")
    parser.add_argument("--alphas", default="0.01,0.001,0.0")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    out = args.output_dir
    report_path = out / "bank_geometry_eval.json"
    if report_path.exists():
        raise SystemExit(f"{report_path} already exists (create-only).")
    out.mkdir(parents=True, exist_ok=True)

    config = config_lib.get_config(args.config_name)
    base = config.data.base_config
    sidecar = json.loads(pathlib.Path(base.memory_v5_subtask_labels_path).read_text())
    manifest = json.loads(pathlib.Path(base.memory_episode_manifest_path).read_text())
    stride = int(base.memory_stride_frames)
    episodes = [e for e in manifest["episodes"] if e.get("split") == args.split]
    params = model_lib.restore_params(args.params, restore_type=np.ndarray)
    parameter_tree_sha256 = weight_loaders.parameter_tree_sha256(params)
    model = config.model.load(params)
    model.eval()
    sent_len = model.memory_v5_sentence_len

    # --- encode the vocabulary once ------------------------------------------------------------
    from openpi.models import tokenizer as tokenizer_lib

    tok = tokenizer_lib.PaligemmaTokenizer()
    sp = next(v for v in vars(tok).values() if hasattr(v, "encode"))
    vocab = list(sidecar["sentences"])
    rows, masks = [], []
    for s in vocab:
        ids = sp.encode(s.lower().strip().replace("_", " ") + "\n")
        if len(ids) > sent_len:
            raise SystemExit(f"sentence longer than memory_v5_sentence_len: {s!r}")
        rows.append(list(ids) + [0] * (sent_len - len(ids)))
        masks.append([True] * len(ids) + [False] * (sent_len - len(ids)))
    tokens = jnp.asarray(rows, dtype=jnp.int32)
    token_mask = jnp.asarray(masks, dtype=bool)
    encoded = model.v5_encode_sentence(tokens, token_mask)
    keys, values = model.v5_sentence_intent(encoded)  # [v, 1, dk], [v, 1, dv]
    keys_np = np.asarray(jax.device_get(keys))[:, 0, :]
    values_np = np.asarray(jax.device_get(values))[:, 0, :]
    key_cos = keys_np @ keys_np.T
    value_cos = values_np @ values_np.T
    enc_np = np.asarray(jax.device_get(encoded))
    enc_unit = enc_np / np.maximum(np.linalg.norm(enc_np, axis=-1, keepdims=True), 1e-9)
    enc_cos = enc_unit @ enc_unit.T
    off = ~np.eye(len(vocab), dtype=bool)
    for name, mat in (("sentence encoding", enc_cos), ("write KEY", key_cos), ("write VALUE", value_cos)):
        v = mat[off]
        print(f"{name} cosines between different sentences: min {v.min():+.3f} mean {v.mean():+.3f} max {v.max():+.3f}", flush=True)
    print("key-key cosines (rows = sentences):", flush=True)
    for i, s in enumerate(vocab):
        worst = sorted(((float(key_cos[i, j]), vocab[j]) for j in range(len(vocab)) if j != i), reverse=True)[:2]
        print(f"  {s[:38]:38s} nearest: " + ", ".join(f"{c:+.3f} {n[:26]}" for c, n in worst), flush=True)

    index_of = {s: i for i, s in enumerate(vocab)}
    alphas = [float(a) for a in args.alphas.split(",")]
    records: list[dict] = []

    for alpha in alphas:
        # The write/read path is parameter-free apart from the memory MLP itself; the decay enters only
        # through `bank.config.alpha_step` (delta_write_kv_multi / analytic_decay read it there), so the
        # same module is replayed under each alpha with its config swapped.
        bank = model.memory_semantic
        original_config = bank.config
        bank.config = dataclasses.replace(original_config, alpha_step=alpha)
        try:
            for ep in episodes:
                entry = sidecar["episodes"][ep["stable_id"]]
                seq = steps_of(entry["segments"], int(entry["num_frames"]), stride)
                state = bank.init_state(1)
                last = None
                go_slot = None  # (vocab index, step written)
                latest = None
                for t, sentence in enumerate(seq):
                    if sentence != last:
                        vi = index_of[sentence]
                        k = jnp.asarray(keys_np[vi][None, None, :])
                        v = jnp.asarray(values_np[vi][None, None, :])
                        state, _ = bank.delta_write_kv_multi(state, k, v, jnp.ones((1, 1), dtype=bool))
                        if sentence.startswith("yellow go"):
                            go_slot = (vi, t)
                        latest = (vi, t)
                        last = sentence
                    else:
                        state, _ = bank.analytic_decay(state, 1)
                    # tray arrival: this step's sentence is a dump/done and the previous was a dig
                    if t > 0 and go_slot is not None and (
                        ("dump and return" in sentence or sentence.startswith("done"))
                        and ("dig and carry" in seq[t - 1])
                    ) and sentence != seq[t - 1]:
                        gi, gt = go_slot
                        li, lt = latest
                        q_go = jnp.asarray(keys_np[gi][None, None, :])
                        q_last = jnp.asarray(keys_np[li][None, None, :])
                        def cos(a, b):
                            na, nb = np.linalg.norm(a), np.linalg.norm(b)
                            return float(a @ b / max(na * nb, 1e-9))
                        r_go = np.asarray(jax.device_get(bank.read_key(state, q_go)))[0, 0]
                        r_last = np.asarray(jax.device_get(bank.read_key(state, q_last)))[0, 0]
                        # Decisive readout: does the bank still report the go sentence's COUNT? Compare the
                        # retrieved vector against the three go variants (they differ only in the digit).
                        go_variants = [i for i, sv in enumerate(vocab) if sv.startswith("yellow go")]
                        gv_cos = {int(vocab[i].split("scoop ")[-1].split(" ")[0]): cos(r_go, values_np[i]) for i in go_variants}
                        gv_best = max(gv_cos, key=gv_cos.get)
                        true_x = int(vocab[gi].split("scoop ")[-1].split(" ")[0])
                        gv_sorted = sorted(gv_cos.values(), reverse=True)
                        records.append({
                            "go_count_readout": gv_best, "go_count_true": true_x,
                            "go_count_correct": bool(gv_best == true_x),
                            "go_count_margin": float(gv_sorted[0] - gv_sorted[1]),
                            "go_count_cos": {str(k): float(v) for k, v in gv_cos.items()},
                            "alpha": alpha, "stable_id": ep["stable_id"], "x": int(entry["x"]), "step": t,
                            "label": "done" if sentence.startswith("done") else "dump",
                            "go_age_steps": t - gt, "go_sentence": vocab[gi], "latest_sentence": vocab[li],
                            "writes_since_go": sum(1 for tt in range(gt + 1, t + 1) if seq[tt] != seq[tt - 1]),
                            "go_recall": cos(r_go, values_np[gi]),
                            "go_vs_recent": cos(r_go, values_np[li]),
                            "go_read_norm": float(np.linalg.norm(r_go)),
                            "recent_recall": cos(r_last, values_np[li]),
                            "key_cos_go_latest": float(keys_np[gi] @ keys_np[li]),
                        })
        finally:
            bank.config = original_config
        rows_a = [r for r in records if r["alpha"] == alpha]
        print(f"== alpha {alpha}: COUNT READOUT from the bank {sum(r['go_count_correct'] for r in rows_a)}/{len(rows_a)} "
              f"(margin {np.mean([r['go_count_margin'] for r in rows_a]):.4f})", flush=True)
        print(f"== alpha {alpha}: {len(rows_a)} tray steps  go_recall mean {np.mean([r['go_recall'] for r in rows_a]):.3f} "
              f"(min {min(r['go_recall'] for r in rows_a):.3f})  go_vs_recent {np.mean([r['go_vs_recent'] for r in rows_a]):+.3f}  "
              f"recent_recall {np.mean([r['recent_recall'] for r in rows_a]):.3f}", flush=True)
        for r in rows_a:
            print(f"   {r['stable_id'][-6:]} x={r['x']} step {r['step']:3d} {r['label']:4s} go_age {r['go_age_steps']:3d} "
                  f"writes_since_go {r['writes_since_go']} go_recall {r['go_recall']:.3f} go_vs_recent {r['go_vs_recent']:+.3f} "
                  f"recent {r['recent_recall']:.3f}", flush=True)

    report = {
        "schema_version": SCHEMA_VERSION, "config_name": args.config_name, "params": str(args.params),
        "parameter_tree_sha256": parameter_tree_sha256, "split": args.split, "stride_frames": stride,
        "alphas": alphas, "sentences": vocab, "key_cosines": key_cos.tolist(), "value_cosines": value_cos.tolist(), "encoding_cosines": enc_cos.tolist(), "records": records,
        "argv": sys.argv,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
