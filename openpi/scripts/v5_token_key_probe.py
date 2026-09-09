"""TOKEN-LEVEL CONTEXTUAL KEYS probe for the v5 fast-weight bank (2026-09-08, user: "make sure point 1 makes it
general and solves the issue").

Question: if every note is written token by token -- key = the CAUSAL contextual state that precedes the token,
value = the token itself, exactly the addressing a fast-weight language model has for free -- do the four task1
facts coexist in one bank and can each be read back by its entity, does beans counting still work (newest wins for
the same context), and does the old bin task still work? All measured bank-level with the checkpoint's own encoder,
no decoder, no training: replay each episode's true note sequence into a fresh delta-rule bank and read it back at
the decision points.

Variants (all parameter-free, nothing task-specific):
  token_causal_std   causal layer-8 token states, standardized over the vocabulary's token states, unit keys;
                     value = unit input embedding of the token                            <- the proposal
  token_bidir_std    same with the encoder's current BIDIRECTIONAL states (the key then contains the digit itself)
  token_causal_raw   causal states without standardization (98 % shared direction, expected to interfere)
  token_causal_proj  token_causal_std projected by a fixed random map to the real bank size (d_key x d_value)
  pooled_sentence    the pre-A8 mechanism: one pooled key/value per sentence (model.v5_sentence_intent), decoded by
                     oracle-key scoring over the candidate sentences (the "bank reports" measure of the A8 probe)
Also prints, exactly, what the A8 template rule (_v5_slot_templates, max_diff 2) does to the task1 vocabulary.

Reads (decode = nearest candidate token value by cosine; margin = top1 - top2):
  task1   after the 4 placements (before the closing sentence) and at the episode end: "<obj> in bin" -> digit
  beans   before the go sentence and at the episode end: "light on:" -> blink count x; at every tray arrival:
          "scoop" -> k and "scoop k of" -> x
  oldbin  after "close both lids and reset arms" and at the end: "inspect both bins: banana" -> left/right
Report: <output-dir>/token_key_probe.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

SCHEMA = "v5_token_key_probe/1"


def steps_between(segments: list[dict], stride: int) -> list[tuple[str, int]]:
    """(sentence, number of stride steps the sentence lasts) in order."""
    out = []
    for seg in segments:
        n = max(1, (int(seg["end"]) - int(seg["start"]) + 1) // stride)
        out.append((seg["sentence"], n))
    return out


class Bank:
    """Plain delta-rule fast-weight bank: read(q) = W^T q; write makes read(k) == v exactly."""

    def __init__(self, dk: int, dv: int, alpha: float):
        self.w = np.zeros((dk, dv), dtype=np.float32)
        self.alpha = alpha

    def write(self, k: np.ndarray, v: np.ndarray) -> None:
        pred = self.w.T @ k
        self.w += np.outer(k, v - pred)

    def decay(self, steps: int) -> None:
        if self.alpha > 0 and steps > 0:
            self.w *= (1.0 - self.alpha) ** steps

    def read(self, q: np.ndarray) -> np.ndarray:
        return self.w.T @ q


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def main(argv=None) -> None:
    import jax
    import jax.numpy as jnp

    from openpi.models import model as model_lib
    from openpi.models import tokenizer as tokenizer_lib
    from openpi.models.pi0 import _v5_slot_templates, make_attn_mask
    from openpi.training import config as config_lib

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--config-name", default="pi05_yam_mem_v5_beansB9")
    parser.add_argument("--task1-sidecar", type=pathlib.Path, required=True)
    parser.add_argument("--beans-sidecar", type=pathlib.Path, required=True)
    parser.add_argument("--oldbin-sidecar", type=pathlib.Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--alphas", default="0.0,0.01")
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "token_key_probe.json"

    config = config_lib.get_config(args.config_name)
    params = model_lib.restore_params(args.params, restore_type=np.ndarray)
    model = config.model.load(params)
    model.eval()
    d_key = int(config.model.memory_semantic.d_key)
    d_value = int(config.model.memory_semantic.d_value)
    print(f"model loaded; memory_layer {model.memory_layer}; bank d_key {d_key} d_value {d_value}", flush=True)

    tok = tokenizer_lib.PaligemmaTokenizer()
    sp = next(v for v in vars(tok).values() if hasattr(v, "encode"))

    def ids_of(s: str) -> list[int]:
        return list(sp.encode(s.lower().strip().replace("_", " ") + "\n"))

    def token_states(rows: list[list[int]], causal: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """states [n, L, emb], embeddings [n, L, emb], mask [n, L] for padded token rows."""
        length = max(len(r) for r in rows)
        tokens = jnp.asarray([r + [0] * (length - len(r)) for r in rows], dtype=jnp.int32)
        mask = jnp.asarray([[True] * len(r) + [False] * (length - len(r)) for r in rows], dtype=bool)
        depth = model.PaliGemma.llm.module.configs[0].depth
        emb = model.PaliGemma.llm(tokens, method="embed")
        cache = model._v32_empty_cache(tokens.shape[0], length, emb.dtype)
        mask_ar = jnp.ones(mask.shape, dtype=jnp.int32) if causal else jnp.zeros(mask.shape, dtype=jnp.int32)
        attn = model._pad_attention_columns(make_attn_mask(mask, mask_ar), length)
        positions = jnp.maximum(jnp.cumsum(mask.astype(jnp.int32), axis=1) - 1, 0)
        (hidden, _), _ = model.PaliGemma.llm(
            [emb, None], mask=attn, positions=positions, kv_cache=cache, cache_position=0,
            active_layers=jnp.arange(depth) <= model.memory_layer, apply_final_norm=False,
        )
        return (np.asarray(jax.device_get(hidden), dtype=np.float32), np.asarray(jax.device_get(emb), dtype=np.float32),
                np.asarray(jax.device_get(mask)))

    rng = np.random.default_rng(args.seed)
    vocabs = {
        "task1": json.loads(args.task1_sidecar.read_text()),
        "beans": json.loads(args.beans_sidecar.read_text()),
        "oldbin": json.loads(args.oldbin_sidecar.read_text()),
    }
    alphas = [float(a) for a in args.alphas.split(",")]
    report: dict = {"schema": SCHEMA, "params": str(args.params), "config": args.config_name, "results": {}}

    # ---- A8 template rule on the task1 vocabulary (exact, no model needed) --------------------------------
    t1_rows = tuple(tuple(ids_of(s)) for s in vocabs["task1"]["sentences"])
    keep = _v5_slot_templates(t1_rows, 2)
    templates: dict[str, list[str]] = {}
    for s, row, kp in zip(vocabs["task1"]["sentences"], t1_rows, keep, strict=True):
        key_tokens = tuple(t if k else -1 for t, k in zip(row, kp, strict=True))
        templates.setdefault(str(key_tokens), []).append(s)
    print(f"A8 template rule (max_diff 2) on the task1 vocabulary: {len(templates)} slots for {len(t1_rows)} sentences")
    for members in templates.values():
        print(f"  slot with {len(members):2d} sentences, e.g. {members[:3]}")
    report["a8_templates_task1"] = {"num_slots": len(templates), "slots": list(templates.values())}

    # ---- per vocabulary ------------------------------------------------------------------------------------
    for name, sidecar in vocabs.items():
        sentences = list(sidecar["sentences"])
        rows = [ids_of(s) for s in sentences]
        idx = {s: i for i, s in enumerate(sentences)}
        variants: dict[str, dict] = {}
        for vname, causal in (("token_causal", True), ("token_bidir", False)):
            states, emb, mask = token_states(rows, causal)
            variants[vname] = {"states": states, "emb": emb, "mask": mask}
        # standardization statistics over every valid token state of the vocabulary (per variant)
        for vname in list(variants):
            v = variants[vname]
            valid = v["states"][v["mask"]]
            v["mu"], v["sd"] = valid.mean(0), valid.std(0) + 1e-6
            valid_e = v["emb"][v["mask"]]
            v["emu"], v["esd"] = valid_e.mean(0), valid_e.std(0) + 1e-6
        # pooled (pre-A8) keys/values; the encoder standardizes against the checkpoint's reference rows,
        # which must fit the padded length, so pad to the model's sentence length
        length = int(model.memory_v5_sentence_len)
        if max(len(r) for r in rows) > length:
            raise SystemExit(f"{name}: a sentence is longer than memory_v5_sentence_len={length}")
        tokens = jnp.asarray([r + [0] * (length - len(r)) for r in rows], dtype=jnp.int32)
        tmask = jnp.asarray([[True] * len(r) + [False] * (length - len(r)) for r in rows], dtype=bool)
        pk, pv = model.v5_sentence_intent(model.v5_encode_sentence(tokens, tmask))
        pooled_k = np.asarray(jax.device_get(pk))[:, 0, :]
        pooled_v = np.asarray(jax.device_get(pv))[:, 0, :]
        emb_width = variants["token_causal"]["states"].shape[-1]
        proj_k = rng.standard_normal((emb_width, d_key)).astype(np.float32) / np.sqrt(d_key)
        proj_v = rng.standard_normal((emb_width, d_value)).astype(np.float32) / np.sqrt(d_value)

        def kv_pairs(si: int, variant: str) -> list[tuple[np.ndarray, np.ndarray]]:
            """Token-level (key, value) pairs of sentence si under a variant."""
            base = "token_bidir" if variant.startswith("token_bidir") else "token_causal"
            v = variants[base]
            n = int(v["mask"][si].sum())
            h, e = v["states"][si, :n], v["emb"][si, :n]
            if variant.endswith("_raw"):
                keys_src = h
            else:
                keys_src = (h - v["mu"]) / v["sd"]
            # key of token t = state that PRECEDES it (position t-1); the first token keys on its own state
            k = np.concatenate([keys_src[:1], keys_src[:-1]], axis=0)
            vals = (e - v["emu"]) / v["esd"] if variant.endswith("_vstd") else e
            if variant.endswith("_proj"):
                k, vals = k @ proj_k, vals @ proj_v
            return list(zip(unit(k), unit(vals), strict=True))

        def query_and_candidates(group: list[str], variant: str, which: int = 0, query_group: list[str] | None = None):
            """Query key for the `which`-th varying position of a sentence group, and the candidate
            (token id -> value vector) map for that position. `query_group` (optional) supplies the
            CONTEXT the question is asked from (e.g. the closing sentence) while the candidates and the
            expected token still come from `group` (the notes that were written)."""
            grp_rows = [rows[idx[s]] for s in group]
            L = min(len(r) for r in grp_rows)
            var_pos = [p for p in range(L) if len({r[p] for r in grp_rows}) > 1]
            if not var_pos:
                raise ValueError(f"no varying position in {group}")
            p = var_pos[which]
            if query_group is None:
                keys = [kv_pairs(idx[s], variant)[p][0] for s in group]
            else:
                q_rows = [rows[idx[s]] for s in query_group]
                Lq = min(len(r) for r in q_rows)
                q_var = [pp for pp in range(Lq) if len({r[pp] for r in q_rows}) > 1]
                pq = q_var[0]
                keys = [kv_pairs(idx[s], variant)[pq][0] for s in query_group]
            q = np.mean(keys, axis=0)
            spread = float(min(unit(k) @ unit(keys[0]) for k in keys))
            cands = {rows[idx[s]][p]: kv_pairs(idx[s], variant)[p][1] for s in group}
            return unit(q), cands, spread

        def decode(read: np.ndarray, cands: dict[int, np.ndarray]) -> tuple[int, float]:
            scores = sorted(((float(unit(read) @ vv), t) for t, vv in cands.items()), reverse=True)
            margin = scores[0][0] - (scores[1][0] if len(scores) > 1 else 0.0)
            return scores[0][1], margin

        def pooled_decode(bank: Bank, group: list[str]) -> str:
            best = None
            for s in group:
                r = bank.read(pooled_k[idx[s]])
                c = float(unit(r) @ pooled_v[idx[s]])
                if best is None or c > best[0]:
                    best = (c, s)
            return best[1]

        # ---- query specs per task -------------------------------------------------------------------
        def digit_token(s: str, group: list[str]) -> int:
            grp_rows = [rows[idx[g]] for g in group]
            L = min(len(r) for r in grp_rows)
            p = [p for p in range(L) if len({r[p] for r in grp_rows}) > 1][0]
            return rows[idx[s]][p]

        results = {}
        episodes = list(sidecar["episodes"].items())[: args.max_episodes]
        variant_names = ["token_causal_std", "token_causal_std_vstd", "token_bidir_std", "token_causal_raw",
                         "token_causal_proj", "pooled_sentence"]
        for alpha in alphas:
            for variant in variant_names:
                tallies: dict[str, dict] = {}

                def tally(point: str, correct: bool, margin: float) -> None:
                    t = tallies.setdefault(point, {"n": 0, "correct": 0, "margins": []})
                    t["n"] += 1
                    t["correct"] += int(correct)
                    t["margins"].append(margin)

                for _sid, entry in episodes:
                    seq = steps_between(entry["segments"], args.stride)
                    sentences_in_ep = [s for s, _ in seq]
                    if variant == "pooled_sentence":
                        bank = Bank(pooled_k.shape[1], pooled_v.shape[1], alpha)
                    else:
                        probe_dim = kv_pairs(0, variant)[0]
                        bank = Bank(probe_dim[0].shape[0], probe_dim[1].shape[0], alpha)

                    def write_sentence(s: str) -> None:
                        if variant == "pooled_sentence":
                            bank.write(pooled_k[idx[s]], pooled_v[idx[s]])
                        else:
                            for k, v in kv_pairs(idx[s], variant):
                                bank.write(k, v)

                    def ask(point: str, group: list[str], expected: str, which: int = 0,
                            query_group: list[str] | None = None) -> None:
                        if variant == "pooled_sentence":
                            if query_group is not None:
                                return  # no pooled analogue of a cross-context question
                            pred = pooled_decode(bank, group)
                            tally(point, pred == expected, 0.0)
                        else:
                            q, cands, _ = query_and_candidates(group, variant, which, query_group)
                            pred_tok, margin = decode(bank.read(q), cands)
                            tally(point, pred_tok == digit_token(expected, group), margin)

                    # replay: write each sentence when it starts, decay for its duration, ask at the defined points
                    for i, (s, n_steps) in enumerate(seq):
                        # ---- reads that happen BEFORE this sentence is written
                        if name == "task1" and s.startswith("all bins closed"):
                            for obj in ("banana", "spoon", "box", "tape"):
                                truth = next(x for x in sentences_in_ep if x.startswith(f"{obj} in bin"))
                                ask("after_4_placements", [f"{obj} in bin {k}" for k in (1, 2, 3)], truth)
                            # the REAL question: asked from the closing sentence's own context
                            target = s.split("closed, ")[1].split(" is in")[0]
                            truth = next(x for x in sentences_in_ep if x.startswith(f"{target} in bin"))
                            ask("closing_context_target_bin", [f"{target} in bin {k}" for k in (1, 2, 3)], truth,
                                query_group=[f"all bins closed, {target} is in bin {k}" for k in (1, 2, 3)])
                        if name == "beans" and s.startswith("yellow go"):
                            x = int(s.split("scoop ")[-1].split(" ")[0])
                            grp = [f"light on: {k} green blink{'s' if k > 1 else ''} so far" for k in (1, 2, 3)]
                            ask("before_go_light_count", grp, grp[x - 1])
                            go_grp = [f"yellow go: pick up the scoop, scoop {k} time{'s' if k > 1 else ''}" for k in (1, 2, 3)]
                            if all(g in idx for g in go_grp):
                                ask("go_context_light_count", grp, grp[x - 1], query_group=go_grp)
                        if name == "oldbin" and s.startswith("wait;"):
                            truth = next(x for x in sentences_in_ep if x.startswith("inspect both bins"))
                            grp = ["inspect both bins: banana left, grey pepper box right",
                                   "inspect both bins: banana right, grey pepper box left"]
                            ask("wait_context_banana_side", grp, truth, 0,
                                query_group=["wait; target bin is left", "wait; target bin is right"])
                        if name == "beans" and ("dump and return" in s or s.startswith("done")) and i > 0 and "dig and carry" in seq[i - 1][0]:
                            prev = seq[i - 1][0]  # "scoop k of x: dig and carry"
                            k = int(prev.split("scoop ")[1].split(" ")[0])
                            x = int(prev.split(" of ")[1].split(":")[0])
                            grp_k = [f"scoop {kk} of {x}: dig and carry" for kk in range(1, x + 1)]
                            if len(grp_k) > 1:
                                ask("tray_scoop_k", grp_k, prev)
                            grp_x = [f"scoop {k} of {xx}: dig and carry" for xx in range(max(k, 1), 4)]
                            if len(grp_x) > 1:
                                ask("tray_scoop_x", grp_x, prev)
                        if name == "oldbin" and s.startswith("wait;"):
                            truth = next(x for x in sentences_in_ep if x.startswith("inspect both bins"))
                            grp = ["inspect both bins: banana left, grey pepper box right",
                                   "inspect both bins: banana right, grey pepper box left"]
                            ask("after_close_reset_banana_side", grp, truth, 0)
                        write_sentence(s)
                        bank.decay(n_steps)
                    # ---- end-of-episode reads (every note written, including the decision)
                    if name == "task1":
                        for obj in ("banana", "spoon", "box", "tape"):
                            truth = next(x for x in sentences_in_ep if x.startswith(f"{obj} in bin"))
                            ask("episode_end", [f"{obj} in bin {k}" for k in (1, 2, 3)], truth)
                    if name == "beans":
                        go = next((x for x in sentences_in_ep if x.startswith("yellow go")), None)
                        if go:
                            x = int(go.split("scoop ")[-1].split(" ")[0])
                            grp = [f"light on: {k} green blink{'s' if k > 1 else ''} so far" for k in (1, 2, 3)]
                            ask("episode_end_light_count", grp, grp[x - 1])
                    if name == "oldbin":
                        truth = next(x for x in sentences_in_ep if x.startswith("inspect both bins"))
                        grp = ["inspect both bins: banana left, grey pepper box right",
                               "inspect both bins: banana right, grey pepper box left"]
                        ask("episode_end_banana_side", grp, truth, 0)
                summary = {p: {"n": t["n"], "acc": round(t["correct"] / max(t["n"], 1), 3),
                               "margin_mean": round(float(np.mean(t["margins"])), 3) if t["margins"] else None}
                           for p, t in tallies.items()}
                results[f"{variant}@alpha{alpha}"] = summary
                print(f"[{name}] {variant:18s} alpha {alpha:<5} " + " | ".join(
                    f"{p}: {v['acc']:.2f} (n={v['n']}, m={v['margin_mean']})" for p, v in summary.items()), flush=True)
        # causality sanity: query states derived from different digits must coincide under causal, not bidir
        if name == "task1":
            for variant in ("token_causal_std", "token_bidir_std"):
                _, _, spread = query_and_candidates([f"banana in bin {k}" for k in (1, 2, 3)], variant)
                print(f"[{name}] {variant}: min cosine between the 'banana in bin' query keys taken from bin 1/2/3 sentences = {spread:.4f}")
                results[f"query_key_consistency_{variant}"] = spread
        report["results"][name] = results

    report_path.write_text(json.dumps(report, indent=1) + "\n")
    print(f"report {report_path}")


if __name__ == "__main__":
    sys.exit(main())
