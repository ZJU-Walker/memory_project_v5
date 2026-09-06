"""Bean-scoop TRAY-DECISION probe (cluster_v5/README.md §8, 2026-09-05 17:10, user "in parallel").

Question: at the tray arrival ("scoop k: dump and return" if k < x, "done" if k == x; v6 sub-phase sentences) does
the model's choice follow the TARGET x it should remember from the go sentence, and does the bank's time decay
(alpha_step 0.01 per step, half-life ~69 steps) matter? Windows come from the training loader (oracle-mode config:
the in-window sentences are the labels, the history is prefilled with the true gaps). At every tray-arrival step
(a dump/done label right after a dig label) the step's sentence is replaced by the two candidates -- the dump
sentence with the step's k, and done -- and the lower teacher-forced decision CE wins:

  * normal : history as-is;
  * flip   : every count in the LIGHT and GO sentences of the history (prefill, pending, in-window) is cyclically
             shifted 1->2->3->1 (the scoop sentences keep their k). The content-consistent answer is
             done iff k == shifted x;
  * blank  : the history is emptied (what the model does with no memory at all).

`--alphas` runs the same parameters under other bank decays (0.0 = no decay: every note as fresh as when written),
so "natural" vs "no-decay" separates "the note has faded" from "the model never learned to read it".
Report: v5/diagnostics/tray_flip_<...>/tray_flip_eval.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import v5_count_flip_eval as cf  # noqa: E402

SCHEMA_VERSION = "v5_tray_flip_eval/2"  # 2: v4 "scoop k of x" sentences (x carried + flipped in scoop rows, variable-length candidates)
SCOOP, DONE, LIGHT, WAIT, GO = 17390, 7262, 2462, 9532, 22006
DIG_TOKEN, DUMP_TOKEN = 3441, 21430
DONE_ROW = (7262, 235269, 2507, 1706, 573, 65522, 578, 2203, 108)


OF_TOKEN = 576  # "of" in the v4 target-carry sentences "scoop k of x: ..." (2026-09-06)


def dump_row(k: int, x: int = -1) -> tuple[int, ...]:
    if x > 0:  # v4 target-carry layout: "scoop k of x: dump and return" (12 tokens)
        return (SCOOP, 715, cf.SPACE_TOKEN, cf.DIGIT_TOKENS[k], OF_TOKEN, cf.SPACE_TOKEN, cf.DIGIT_TOKENS[x], 235292, DUMP_TOKEN, 578, 2203, 108)
    return (SCOOP, 715, cf.SPACE_TOKEN, cf.DIGIT_TOKENS[k], 235292, DUMP_TOKEN, 578, 2203, 108)


def row_x(row: np.ndarray, mask: np.ndarray) -> int:
    """Target x carried in a v4 scoop row ("scoop k of x: ..."), -1 for the v6 layout."""
    if row[0] == SCOOP and mask[6] and row[4] == OF_TOKEN:
        return cf.DIGIT_TO_COUNT.get(int(row[6]), -1)
    return -1


def shift_scoop_x(row: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flip condition for v4 scoop rows: cycle the carried target x (index 6), keep k."""
    out = row.copy()
    if out[0] == SCOOP and mask[6] and out[4] == OF_TOKEN and int(out[6]) in cf.DIGIT_TO_COUNT:
        out[6] = cf.DIGIT_TOKENS[cf.CYCLE[cf.DIGIT_TO_COUNT[int(out[6])]]]
    return out


def row_kind(row: np.ndarray, mask: np.ndarray) -> str:
    if not mask.any():
        return "none"
    if row[0] == SCOOP:
        toks = set(int(v) for v in row[mask])
        return "dig" if DIG_TOKEN in toks else ("dump" if DUMP_TOKEN in toks else "scoop?")
    if row[0] == DONE:
        return "done"
    if row[0] == GO:
        return "go"
    if row[0] in (LIGHT, WAIT):
        return "light"
    return "other"


def row_k(row: np.ndarray) -> int:
    return cf.DIGIT_TO_COUNT.get(int(row[3]), -1)


def go_count(row: np.ndarray, mask: np.ndarray) -> int:
    return cf.count_of_step(row, mask)


def summarize(records: list[dict], cond: str, alpha: float, **filt) -> dict:
    rows = [r for r in records if r["alpha"] == alpha and all(r.get(k) == v for k, v in filt.items())]
    n = len(rows)
    if n == 0:
        return {"count": 0}
    out = {"count": n}
    if cond == "flip":
        out["follows_content_rate"] = float(np.mean([r["flip_pred"] == r["flip_content"] for r in rows]))
        out["keeps_true_rate"] = float(np.mean([r["flip_pred"] == r["label"] for r in rows]))
        out["margin_nats"] = float(np.mean([r["flip_margin"] for r in rows]))
    else:
        out["accuracy"] = float(np.mean([r[f"{cond}_pred"] == r["label"] for r in rows]))
        out["done_rate"] = float(np.mean([r[f"{cond}_pred"] == "done" for r in rows]))
        out["margin_nats"] = float(np.mean([r[f"{cond}_margin"] for r in rows]))
    return out


def main(argv=None) -> None:
    import jax

    from openpi.models import model as model_lib
    from openpi.shared import nnx_utils
    from openpi.training import config as config_lib
    from openpi.training import data_loader as data_loader_lib
    from openpi.training import weight_loaders

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--config-name", default="pi05_yam_mem_v5_beansA6", help="an ORACLE-write (stage-A) v6 config")
    parser.add_argument("--split", choices=("development", "train", "final_test"), default="train")
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--alphas", default="0.01,0.0", help="comma list of bank decays to run the same parameters under")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    report_path = output_dir / "tray_flip_eval.json"
    if report_path.exists():
        raise SystemExit(f"{report_path} already exists (create-only).")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config_lib.get_config(args.config_name)
    if not config.data.base_config.memory_v5_generic_task or not getattr(config.model, "memory_v5_oracle_writes", False):
        raise SystemExit("the tray probe needs an oracle-write v5 generic-task config (beansA6-style).")
    config = dataclasses.replace(
        config,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
        v4_graft_sources=(),
        data=dataclasses.replace(
            config.data,
            base_config=dataclasses.replace(config.data.base_config, memory_manifest_split=args.split),
        ),
    )
    params = model_lib.restore_params(args.params, restore_type=np.ndarray)
    parameter_tree_sha256 = weight_loaders.parameter_tree_sha256(params)
    alphas = [float(a) for a in args.alphas.split(",")]

    # The batches are drawn once (same seed) and reused for every alpha.
    loader = data_loader_lib.create_data_loader(
        config, sharding=None, shuffle=True, num_batches=args.batches, exact_resume=False
    )
    batches = list(loader)
    print(f"config={args.config_name} split={args.split} batches={len(batches)} alphas={alphas}", flush=True)

    records: list[dict] = []
    rng = jax.random.key(args.seed)
    for alpha in alphas:
        model_config = dataclasses.replace(
            config.model,
            memory=dataclasses.replace(config.model.memory, alpha_step=alpha),
            memory_semantic=dataclasses.replace(config.model.memory_semantic, alpha_step=alpha),
        )
        model = model_config.load(params)
        model.eval()
        sequence_loss = nnx_utils.module_jit(
            model._compute_sequence_loss_v32,  # noqa: SLF001
            static_argnames=("train", "v4_intervention"),
        )
        print(f"alpha={alpha}: model ready", flush=True)
        for index, (observation, actions) in enumerate(batches):
            causal = np.asarray(observation.tokenized_causal)
            causal_mask = np.asarray(observation.tokenized_causal_mask)
            fast_mask = np.asarray(observation.causal_fast_mask)
            text_mask = causal_mask & ~fast_mask
            batch, steps, width = causal.shape
            prefill_tokens = np.asarray(observation.memory_v5_prefill_tokens)
            prefill_mask = np.asarray(observation.memory_v5_prefill_mask)
            prefill_gaps = np.asarray(observation.memory_v5_prefill_gaps)
            pending_tokens = np.asarray(observation.memory_v5_pending_tokens)
            pending_mask = np.asarray(observation.memory_v5_pending_mask)

            # Tray-arrival steps and the remembered target x for each.
            tray: dict[int, list[tuple[int, int, int, str, int]]] = {}  # b -> [(t, k, x, label, go_age)]
            for b in range(batch):
                kinds = [row_kind(causal[b, t], text_mask[b, t]) for t in range(steps)]
                for t in range(1, steps):
                    if kinds[t] in ("dump", "done") and kinds[t - 1] == "dig":
                        k = row_k(causal[b, t - 1])
                        carried_x = row_x(causal[b, t - 1], text_mask[b, t - 1])
                        # x: the latest go sentence before t (in-window), else the last go row of the prefill.
                        x, age = -1, -1
                        for tp in range(t - 1, -1, -1):
                            if kinds[tp] == "go":
                                x, age = go_count(causal[b, tp], text_mask[b, tp]), t - tp
                                break
                        if x < 0:
                            for p in range(prefill_tokens.shape[1] - 1, -1, -1):
                                if row_kind(prefill_tokens[b, p], prefill_mask[b, p]) == "go":
                                    x = go_count(prefill_tokens[b, p], prefill_mask[b, p])
                                    age = int(np.sum(np.maximum(prefill_gaps[b, p:], 0))) + t
                                    break
                        if x < 0 and carried_x > 0:
                            x, age = carried_x, t - (t - 1)
                        if x < 0 or k < 0:
                            continue
                        span = int(text_mask[b, t].sum())
                        if span == 0 or not text_mask[b, t, :span].all():
                            continue
                        tray.setdefault(b, []).append((t, k, x, kinds[t], age, carried_x > 0))
            if not tray:
                print(f"alpha={alpha} batch {index + 1}/{len(batches)}: no tray steps", flush=True)
                continue

            def shift_history_rows(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
                out = tokens.copy()
                flat_t = out.reshape(-1, out.shape[-1]); flat_m = mask.reshape(-1, mask.shape[-1])
                for i in range(flat_t.shape[0]):
                    kind = row_kind(flat_t[i], flat_m[i])
                    if kind in ("go", "light"):
                        flat_t[i] = cf.shift_counts(flat_t[i], flat_m[i])
                    elif kind in ("dig", "dump"):
                        flat_t[i] = shift_scoop_x(flat_t[i], flat_m[i])
                return flat_t.reshape(tokens.shape)

            flip_causal = causal.copy()
            for b in range(batch):
                for t in range(steps):
                    kind = row_kind(causal[b, t], text_mask[b, t])
                    if kind in ("go", "light"):
                        flip_causal[b, t] = cf.shift_counts(causal[b, t], text_mask[b, t])
                    elif kind in ("dig", "dump"):
                        flip_causal[b, t] = shift_scoop_x(causal[b, t], text_mask[b, t])
            flip_fields = {
                "memory_v5_prefill_tokens": jax.numpy.asarray(shift_history_rows(prefill_tokens, prefill_mask)),
                "memory_v5_pending_tokens": jax.numpy.asarray(shift_history_rows(pending_tokens, pending_mask)),
            }
            blank_fields = {
                "memory_v5_prefill_mask": jax.numpy.asarray(np.zeros_like(prefill_mask)),
                "memory_v5_pending_mask": jax.numpy.asarray(np.zeros_like(pending_mask)),
            }

            def with_candidate(base: np.ndarray, candidate: str, flipped: bool):
                """Replace the tray step's sentence by the candidate; the candidates may differ in length (v4 dump row
                12 tokens vs done 9), so the row is rebuilt as candidate + the original action tail and both masks
                follow. In the flip condition the dump candidate carries the flipped x (consistent with the history)."""
                v = base.copy(); cm = causal_mask.copy(); fm = fast_mask.copy()
                for b, items in tray.items():
                    for t, k, x, label, age, carried in items:
                        xx = (cf.CYCLE[x] if flipped else x) if carried else -1
                        row = np.asarray(DONE_ROW if candidate == "done" else dump_row(k, xx), dtype=v.dtype)
                        span = int(text_mask[b, t].sum()); L = v.shape[-1]
                        tail_t = base[b, t, span:]; tail_c = causal_mask[b, t, span:]; tail_f = fast_mask[b, t, span:]
                        nt = np.concatenate([row, tail_t])[:L]; nc = np.concatenate([np.ones(len(row), bool), tail_c])[:L]
                        nf = np.concatenate([np.zeros(len(row), bool), tail_f])[:L]
                        v[b, t] = 0; cm[b, t] = False; fm[b, t] = False
                        v[b, t, : len(nt)] = nt; cm[b, t, : len(nc)] = nc; fm[b, t, : len(nf)] = nf
                return v, cm, fm

            conditions = {
                "normal": (observation, causal),
                "flip": (observation.replace(**flip_fields), flip_causal),
                "blank": (observation.replace(**blank_fields), causal),
            }
            step_rng = jax.random.fold_in(rng, index)
            ce: dict[tuple[str, str], np.ndarray] = {}
            active = None
            for cond, (obs, base_causal) in conditions.items():
                for candidate in ("dump", "done"):
                    ct, cc, cfm = with_candidate(base_causal, candidate, cond == "flip")
                    losses = sequence_loss(
                        step_rng,
                        obs.replace(tokenized_causal=jax.numpy.asarray(ct), tokenized_causal_mask=jax.numpy.asarray(cc),
                                    causal_fast_mask=jax.numpy.asarray(cfm)),
                        actions,
                        train=False,
                    )
                    ce[(cond, candidate)] = np.asarray(jax.device_get(losses["v4_decision_ce_steps"])).T  # [b, T]
                    if active is None:
                        active = np.asarray(jax.device_get(losses["v4_decision_active_steps"])).T > 0.5
            for b, items in tray.items():
                for t, k, x, label, age, carried in items:
                    if not active[b, t]:
                        continue
                    rec = {
                        "alpha": alpha, "batch": index, "row": b, "step": int(t), "k": k, "x": x, "label": label,
                        "go_age_steps": int(age), "flip_x": cf.CYCLE[x],
                        "flip_content": "done" if k == cf.CYCLE[x] else "dump",
                    }
                    for cond in conditions:
                        s_dump, s_done = float(ce[(cond, "dump")][b, t]), float(ce[(cond, "done")][b, t])
                        rec[f"{cond}_pred"] = "done" if s_done < s_dump else "dump"
                        rec[f"{cond}_margin"] = abs(s_dump - s_done)
                        rec[f"{cond}_ce"] = [s_dump, s_done]
                    records.append(rec)
            print(f"alpha={alpha} batch {index + 1}/{len(batches)} done ({sum(1 for r in records if r['alpha'] == alpha)} tray steps)", flush=True)
        del model, sequence_loss

    summaries = {}
    for alpha in alphas:
        key = f"alpha_{alpha}"
        summaries[key] = {
            "normal": summarize(records, "normal", alpha),
            "flip": summarize(records, "flip", alpha),
            "blank": summarize(records, "blank", alpha),
            "normal_by_label": {lab: summarize(records, "normal", alpha, label=lab) for lab in ("dump", "done")},
            "flip_by_label": {lab: summarize(records, "flip", alpha, label=lab) for lab in ("dump", "done")},
            "normal_by_k": {k: summarize(records, "normal", alpha, k=k) for k in (1, 2, 3)},
        }
        print(f"== alpha {alpha}", flush=True)
        for cond in ("normal", "flip", "blank"):
            print(f"  {cond:6s} {json.dumps(summaries[key][cond], sort_keys=True)}", flush=True)
        print(f"  by label: normal {json.dumps(summaries[key]['normal_by_label'], sort_keys=True)}", flush=True)
        print(f"            flip   {json.dumps(summaries[key]['flip_by_label'], sort_keys=True)}", flush=True)
    report = {
        "schema_version": SCHEMA_VERSION, "config_name": args.config_name, "params": str(args.params),
        "parameter_tree_sha256": parameter_tree_sha256, "split": args.split, "batches": args.batches,
        "batch_size": args.batch_size, "seed": args.seed, "alphas": alphas, "summaries": summaries,
        "records": records, "argv": sys.argv,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
