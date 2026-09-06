"""Bank-level COUNT-RECOVERY test (2026-09-05 20:45, user "make sure the sentence direction is the real issue rather
than guessing"): replay each dev episode's true note sequence into a fresh bank (stride-5 clock: write on a sentence
change, one analytic decay step otherwise) with the checkpoint's own encoder, and at every tray arrival read the GO
slot back and pick the nearest of the three go variants ("scoop 1/2/3 times") -> did the bank keep the COUNT?
Control: at the go step, read the newest LIGHT note back and pick the nearest "light off: 1/2/3" -> the count the
model demonstrably reads today. Runs under the config's write encoding, so the same parameters can be compared as
beansA6 (plain keys/values) vs beansA8 (slot keys + whitened values) -> Test 2 for the A8 design.
Report: <output-dir>/count_recovery_eval.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np

SCHEMA_VERSION = "v5_count_recovery_eval/1"


def steps_of(segments: list[dict], num_frames: int, stride: int) -> list[str]:
    per_frame = [""] * num_frames
    for seg in segments:
        for f in range(int(seg["start"]), int(seg["end"]) + 1):
            per_frame[f] = seg["sentence"]
    return [per_frame[f] for f in range(0, num_frames, stride)]


def count_of(sentence: str) -> int:
    for tok in sentence.replace(":", " ").replace(",", " ").split():
        if tok in ("1", "2", "3"):
            return int(tok)
    return -1


def main(argv=None) -> None:
    import jax
    import jax.numpy as jnp

    from openpi.models import model as model_lib
    from openpi.models import tokenizer as tokenizer_lib
    from openpi.training import config as config_lib
    from openpi.training import weight_loaders

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--config-name", default="pi05_yam_mem_v5_beansA8")
    parser.add_argument("--split", default="development")
    parser.add_argument("--alphas", default="0.01,0.0")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    out = args.output_dir; report_path = out / "count_recovery_eval.json"
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
    sha = weight_loaders.parameter_tree_sha256(params)
    model = config.model.load(params); model.eval()
    sent_len = model.memory_v5_sentence_len
    print(f"config={args.config_name} slot_keys={getattr(model, 'memory_v5_slot_keys', False)} whiten_values={getattr(model, 'memory_v5_whiten_values', False)}", flush=True)

    tok = tokenizer_lib.PaligemmaTokenizer(); sp = next(v for v in vars(tok).values() if hasattr(v, "encode"))
    vocab = list(sidecar["sentences"]); rows, masks = [], []
    for s in vocab:
        ids = sp.encode(s.lower().strip().replace("_", " ") + "\n")
        rows.append(list(ids) + [0] * (sent_len - len(ids))); masks.append([True] * len(ids) + [False] * (sent_len - len(ids)))
    tokens = jnp.asarray(rows, dtype=jnp.int32); token_mask = jnp.asarray(masks, dtype=bool)
    keys, values = model.v5_sentence_kv(tokens, token_mask)
    keys_np = np.asarray(jax.device_get(keys))[:, 0, :]; values_np = np.asarray(jax.device_get(values))[:, 0, :]
    kc = keys_np @ keys_np.T; vc = values_np @ values_np.T
    index_of = {s: i for i, s in enumerate(vocab)}
    go_idx = [i for i, s in enumerate(vocab) if s.startswith("yellow go")]
    off_idx = [i for i, s in enumerate(vocab) if s.startswith("light off")]
    def pair(a, b):  # diagnostic only; sentences absent from this vocabulary (e.g. the v4 "scoop k of x" set) print as nan
        if a not in index_of or b not in index_of: return float("nan"), float("nan")
        return float(kc[index_of[a], index_of[b]]), float(vc[index_of[a], index_of[b]])
    print("key/value cosines: go1-go2 %.3f/%.3f  go2-go3 %.3f/%.3f  go2-dig1 %.3f/%.3f  off1-off2 %.3f/%.3f  on1-off1 %.3f/%.3f  dig1-dig2 %.3f/%.3f  dig1-dump1 %.3f/%.3f" % (
        *pair("yellow go: pick up the scoop, scoop 1 time", "yellow go: pick up the scoop, scoop 2 times"),
        *pair("yellow go: pick up the scoop, scoop 2 times", "yellow go: pick up the scoop, scoop 3 times"),
        *pair("yellow go: pick up the scoop, scoop 2 times", "scoop 1: dig and carry"),
        *pair("light off: 1 green blink so far", "light off: 2 green blinks so far"),
        *pair("light on: 1 green blink so far", "light off: 1 green blink so far"),
        *pair("scoop 1: dig and carry", "scoop 2: dig and carry"), *pair("scoop 1: dig and carry", "scoop 1: dump and return")), flush=True)
    off = np.abs(kc - np.eye(len(vocab))); offv = np.abs(vc - np.eye(len(vocab)))
    print(f"mean |cos| off-diagonal: keys {off.sum() / (len(vocab) * (len(vocab) - 1)):.3f}  values {offv.sum() / (len(vocab) * (len(vocab) - 1)):.3f}", flush=True)

    def cos(a, b):
        return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))

    records = []
    for alpha in [float(a) for a in args.alphas.split(",")]:
        bank = model.memory_semantic; original = bank.config
        bank.config = dataclasses.replace(original, alpha_step=alpha)
        try:
            for ep in episodes:
                entry = sidecar["episodes"][ep["stable_id"]]
                seq = steps_of(entry["segments"], int(entry["num_frames"]), stride)
                state = bank.init_state(1); last = None; go_slot = None; light_slot = None
                for t, sentence in enumerate(seq):
                    if sentence != last:
                        vi = index_of[sentence]
                        state, _ = bank.delta_write_kv_multi(state, jnp.asarray(keys_np[vi][None, None]), jnp.asarray(values_np[vi][None, None]), jnp.ones((1, 1), dtype=bool))
                        if sentence.startswith("yellow go"): go_slot = (vi, t)
                        if sentence.startswith("light off"): light_slot = (vi, t)
                        last = sentence
                    else:
                        state, _ = bank.analytic_decay(state, 1)
                    is_go_step = sentence.startswith("yellow go") and t > 0 and not seq[t - 1].startswith("yellow go")
                    is_tray = t > 0 and go_slot is not None and ("dump and return" in sentence or sentence.startswith("done")) and "dig and carry" in seq[t - 1]
                    if is_go_step and light_slot is not None:
                        li, lt = light_slot
                        r = np.asarray(jax.device_get(bank.read_key(state, jnp.asarray(keys_np[li][None, None]))))[0, 0]
                        cands = {count_of(vocab[i]): cos(r, values_np[i]) for i in off_idx}
                        best = max(cands, key=cands.get); srt = sorted(cands.values(), reverse=True)
                        records.append({"alpha": alpha, "kind": "light_at_go", "stable_id": ep["stable_id"], "step": t, "true": count_of(vocab[li]), "readout": best, "correct": best == count_of(vocab[li]), "margin": srt[0] - srt[1], "age": t - lt})
                    if is_tray:
                        gi, gt = go_slot
                        r = np.asarray(jax.device_get(bank.read_key(state, jnp.asarray(keys_np[gi][None, None]))))[0, 0]
                        cands = {count_of(vocab[i]): cos(r, values_np[i]) for i in go_idx}
                        best = max(cands, key=cands.get); srt = sorted(cands.values(), reverse=True)
                        writes = sum(1 for tt in range(gt + 1, t + 1) if seq[tt] != seq[tt - 1])
                        records.append({"alpha": alpha, "kind": "go_at_tray", "stable_id": ep["stable_id"], "step": t, "true": int(entry["x"]), "readout": best, "correct": best == int(entry["x"]), "margin": srt[0] - srt[1], "age": t - gt, "writes_since_go": writes, "go_recall": cos(r, values_np[gi]), "label": "done" if sentence.startswith("done") else "dump"})
        finally:
            bank.config = original
        for kind in ("go_at_tray", "light_at_go"):
            rs = [r for r in records if r["alpha"] == alpha and r["kind"] == kind]
            if rs:
                print(f"== alpha {alpha} {kind}: count readout {sum(r['correct'] for r in rs)}/{len(rs)}  margin mean {np.mean([r['margin'] for r in rs]):.4f} min {min(r['margin'] for r in rs):.4f}" + (f"  go_recall {np.mean([r['go_recall'] for r in rs]):.3f}" if kind == 'go_at_tray' else ""), flush=True)
        for r in [r for r in records if r["alpha"] == alpha and r["kind"] == "go_at_tray"]:
            print(f"   {r['stable_id'][-6:]} x={r['true']} step {r['step']:3d} {r['label']:4s} age {r['age']:3d} writes {r['writes_since_go']} readout {r['readout']} {'OK ' if r['correct'] else 'BAD'} margin {r['margin']:+.4f} go_recall {r['go_recall']:.3f}", flush=True)
    report = {"schema_version": SCHEMA_VERSION, "config_name": args.config_name, "params": str(args.params), "parameter_tree_sha256": sha, "split": args.split, "sentences": vocab, "key_cosines": kc.tolist(), "value_cosines": vc.tolist(), "records": records, "argv": sys.argv}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n"); print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
