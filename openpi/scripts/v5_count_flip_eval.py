"""Bean-scoop count-flip battery (v5 generic task; cluster_v5/README.md §8, 2026-09-04).

Does the go sentence's count come from the BANK? On development windows whose decision steps
carry a go sentence (`yellow go: pick up the scoop, scoop x time(s)`), score the three count
variants of that sentence (x = 1, 2, 3; same token length: digit + time/times) by the model's
per-step decision CE, under three bank conditions:

  * normal : the history prefill and the in-window written sentences as labelled;
  * flip   : every count in the prefilled/pending sentences AND in the window's non-go sentences
             (the blink sentences written in oracle mode) is cyclically shifted, 1->2->3->1
             (digit + blink/blinks fixed). The content-consistent answer is the shifted count;
  * blank  : the prefill/pending rows are emptied (the bank starts blank at the window).

Predicted count = argmin CE over the three variants. "normal_count_accuracy" is the fraction of
go steps predicting the true count, "flip_follows_content_rate" the fraction predicting the
SHIFTED count under flip, "blank_count_accuracy" the fraction still right without history
(chance 1/3 unless the blink phase lies inside the window: `count_in_window`). Windows whose
blink sentences all lie in the prefill are the clean read test (`history_only`).

In self-write mode (beans-B) the in-window sentences are the model's own, so the flip reaches
only the prefill; read the flip number on the history_only subset there.
"""

# ruff: noqa: I001 - pyarrow must precede the openpi/JAX stack for this dataset (libarrow).
import pyarrow.parquet  # noqa: F401  isort: skip

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np

SCHEMA_VERSION = "v5_count_flip_eval/1"
SPACE_TOKEN = 235248
DIGIT_TOKENS = {1: 235274, 2: 235284, 3: 235304}
DIGIT_TO_COUNT = {v: k for k, v in DIGIT_TOKENS.items()}
BLINK_TOKENS = {1: 45741, 2: 212105, 3: 212105}  # " blink" / " blinks"
TIMES_TOKENS = {1: 1069, 2: 3023, 3: 3023}  # " time" / " times"
GO_PREFIX = (22006, 871)  # "yellow go"
CYCLE = {1: 2, 2: 3, 3: 1}


def find_count_positions(row: np.ndarray, mask: np.ndarray) -> list[int]:
    """Indices i where row[i] is the space token and row[i+1] a count digit (masked positions)."""
    out = []
    for i in range(len(row) - 1):
        if mask[i] and mask[i + 1] and row[i] == SPACE_TOKEN and int(row[i + 1]) in DIGIT_TO_COUNT:
            out.append(i)
    return out


def _fix_plural(row: np.ndarray, mask: np.ndarray, space_index: int, count: int) -> None:
    """After the digit at space_index+1: " time(s)" follows directly (go sentence), " blink(s)" one
    token later (after " green"). Fix whichever plural token is there in place."""
    for j in (space_index + 2, space_index + 3):
        if j < len(row) and mask[j]:
            if int(row[j]) in (45741, 212105):
                row[j] = BLINK_TOKENS[count]
                return
            if int(row[j]) in (1069, 3023):
                row[j] = TIMES_TOKENS[count]
                return


def set_count(row: np.ndarray, mask: np.ndarray, count: int) -> np.ndarray:
    """Return a copy of `row` with every count digit set to `count` and the plural tokens fixed."""
    row = row.copy()
    for i in find_count_positions(row, mask):
        row[i + 1] = DIGIT_TOKENS[count]
        _fix_plural(row, mask, i, count)
    return row


def shift_counts(row: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cyclically shift every count in the row (1->2->3->1) with the plural tokens fixed."""
    row = row.copy()
    for i in find_count_positions(row, mask):
        count = CYCLE[DIGIT_TO_COUNT[int(row[i + 1])]]
        row[i + 1] = DIGIT_TOKENS[count]
        _fix_plural(row, mask, i, count)
    return row


def shift_sentence_rows(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply shift_counts to every row of a [..., L] token array."""
    flat_t = tokens.reshape(-1, tokens.shape[-1])
    flat_m = mask.reshape(-1, mask.shape[-1])
    out = np.stack([shift_counts(flat_t[i], flat_m[i]) for i in range(flat_t.shape[0])])
    return out.reshape(tokens.shape)


def go_step_mask(causal: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    """[b, T] steps whose causal text starts with the go prefix."""
    return text_mask[..., 0] & text_mask[..., 1] & (causal[..., 0] == GO_PREFIX[0]) & (causal[..., 1] == GO_PREFIX[1])


def count_of_step(row: np.ndarray, mask: np.ndarray) -> int:
    positions = find_count_positions(row, mask)
    return DIGIT_TO_COUNT[int(row[positions[0] + 1])] if positions else -1


def summarize(records: list[dict], *, first_step_only: bool = False, history_only: bool = False) -> dict:
    rows = [r for r in records if (not first_step_only or r["decision_order"] == 0)]
    if history_only:
        rows = [r for r in rows if not r["count_in_window"]]
    n = len(rows)
    if n == 0:
        return {"count": 0}

    def rate(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    def margin(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    return {
        "count": n,
        "normal_count_accuracy": rate("normal_correct"),
        "flip_follows_content_rate": rate("flip_follows_content"),
        "flip_keeps_true_rate": rate("flip_keeps_true"),
        "blank_count_accuracy": rate("blank_correct"),
        "normal_margin_nats": margin("normal_margin"),
        "flip_margin_nats": margin("flip_margin"),
        "blank_margin_nats": margin("blank_margin"),
    }


def main(argv=None) -> None:
    import jax

    from openpi.models import model as model_lib
    from openpi.shared import nnx_utils
    from openpi.training import config as config_lib
    from openpi.training import data_loader as data_loader_lib
    from openpi.training import weight_loaders

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--config-name", default="pi05_yam_mem_v5_beansA")
    parser.add_argument("--split", choices=("development", "train", "final_test"), default="development")
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--write-retry", choices=("config", "on", "off"), default="config",
                        help="override memory_v5_prev_is_committed for the in-window own writes (B configs)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    report_path = output_dir / "count_flip_eval.json"
    if report_path.exists():
        raise SystemExit(f"{report_path} already exists (create-only).")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config_lib.get_config(args.config_name)
    if not config.data.base_config.memory_v5_generic_task:
        raise SystemExit("the count-flip battery needs a v5 generic-task config (beans).")
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
    model = config.model.load(params)
    if args.write_retry != "config":
        model.memory_v5_prev_is_committed = args.write_retry == "on"
        print(f"write rule override: memory_v5_prev_is_committed={model.memory_v5_prev_is_committed}", flush=True)
    model.eval()
    loader = data_loader_lib.create_data_loader(
        config, sharding=None, shuffle=True, num_batches=args.batches, exact_resume=False
    )
    sequence_loss = nnx_utils.module_jit(
        model._compute_sequence_loss_v32,  # noqa: SLF001
        static_argnames=("train", "v4_intervention"),
    )
    oracle = bool(getattr(config.model, "memory_v5_oracle_writes", False))
    print(f"config={args.config_name} oracle_writes={oracle} split={args.split}", flush=True)

    records: list[dict] = []
    rng = jax.random.key(args.seed)
    for index, (observation, actions) in enumerate(loader):
        causal = np.asarray(observation.tokenized_causal)
        causal_mask = np.asarray(observation.tokenized_causal_mask)
        fast_mask = np.asarray(observation.causal_fast_mask)
        text_mask = causal_mask & ~fast_mask
        batch, steps = causal.shape[:2]
        go_steps = go_step_mask(causal, text_mask)  # [b, T]
        if not go_steps.any():
            print(f"batch {index + 1}/{args.batches}: no go steps", flush=True)
            continue
        prefill_tokens = np.asarray(observation.memory_v5_prefill_tokens)
        prefill_mask = np.asarray(observation.memory_v5_prefill_mask)
        pending_tokens = np.asarray(observation.memory_v5_pending_tokens)
        pending_mask = np.asarray(observation.memory_v5_pending_mask)

        # The window's non-go sentences (blink counts written in oracle mode) under the flip.
        flip_causal = causal.copy()
        for b in range(batch):
            for t in range(steps):
                if text_mask[b, t].any() and not go_steps[b, t]:
                    flip_causal[b, t] = shift_counts(causal[b, t], text_mask[b, t])
        flip_fields = {
            "memory_v5_prefill_tokens": jax.numpy.asarray(shift_sentence_rows(prefill_tokens, prefill_mask)),
            "memory_v5_pending_tokens": jax.numpy.asarray(shift_sentence_rows(pending_tokens, pending_mask)),
        }
        blank_fields = {
            "memory_v5_prefill_mask": jax.numpy.asarray(np.zeros_like(prefill_mask)),
            "memory_v5_pending_mask": jax.numpy.asarray(np.zeros_like(pending_mask)),
        }
        # Does the window itself contain a blink-count sentence (then blank/flip is not a pure history test)?
        count_in_window = np.zeros(batch, dtype=bool)
        for b in range(batch):
            for t in range(steps):
                if text_mask[b, t].any() and not go_steps[b, t]:
                    if any(int(causal[b, t, i + 2]) in (45741, 212105) for i in find_count_positions(causal[b, t], text_mask[b, t]) if i + 2 < causal.shape[-1]):
                        count_in_window[b] = True
                        break

        def variants(base_causal: np.ndarray) -> dict[int, np.ndarray]:
            out = {}
            for count in (1, 2, 3):
                v = base_causal.copy()
                for b in range(batch):
                    for t in range(steps):
                        if go_steps[b, t]:
                            v[b, t] = set_count(base_causal[b, t], text_mask[b, t], count)
                out[count] = v
            return out

        conditions = {
            "normal": (observation, variants(causal)),
            "flip": (observation.replace(**flip_fields), variants(flip_causal)),
            "blank": (observation.replace(**blank_fields), variants(causal)),
        }
        step_rng = jax.random.fold_in(rng, index)
        ce: dict[tuple[str, int], np.ndarray] = {}
        active = None
        for cond, (obs, causal_variants) in conditions.items():
            for count, causal_variant in causal_variants.items():
                losses = sequence_loss(
                    step_rng, obs.replace(tokenized_causal=jax.numpy.asarray(causal_variant)), actions, train=False
                )
                ce[(cond, count)] = np.asarray(jax.device_get(losses["v4_decision_ce_steps"])).T  # [b, T]
                if active is None:
                    active = np.asarray(jax.device_get(losses["v4_decision_active_steps"])).T > 0.5
        for b in range(batch):
            order = 0
            for t in range(steps):
                if not (go_steps[b, t] and active[b, t]):
                    continue
                true_count = count_of_step(causal[b, t], text_mask[b, t])
                if true_count < 0:
                    continue
                shifted = CYCLE[true_count]

                def pick(cond: str) -> tuple[int, float]:
                    scores = {count: float(ce[(cond, count)][b, t]) for count in (1, 2, 3)}
                    best = min(scores, key=scores.get)
                    others = [scores[c] for c in (1, 2, 3) if c != best]
                    return best, min(others) - scores[best]

                normal_pred, normal_margin = pick("normal")
                flip_pred, flip_margin = pick("flip")
                blank_pred, blank_margin = pick("blank")
                records.append(
                    {
                        "batch": index,
                        "row": b,
                        "step": int(t),
                        "decision_order": order,
                        "true_count": true_count,
                        "shifted_count": shifted,
                        "count_in_window": bool(count_in_window[b]),
                        "normal_pred": normal_pred,
                        "flip_pred": flip_pred,
                        "blank_pred": blank_pred,
                        "normal_correct": normal_pred == true_count,
                        "flip_follows_content": flip_pred == shifted,
                        "flip_keeps_true": flip_pred == true_count,
                        "blank_correct": blank_pred == true_count,
                        "normal_margin": normal_margin,
                        "flip_margin": flip_margin,
                        "blank_margin": blank_margin,
                        "ce": {cond: [float(ce[(cond, c)][b, t]) for c in (1, 2, 3)] for cond in conditions},
                    }
                )
                order += 1
        print(f"batch {index + 1}/{args.batches} done ({len(records)} go steps so far)", flush=True)

    summary = summarize(records)
    summary_first = summarize(records, first_step_only=True)
    summary_history = summarize(records, first_step_only=True, history_only=True)
    for title, s in (("all go steps", summary), ("first go step per window", summary_first), ("first go step, history only", summary_history)):
        print(f"{title}: {json.dumps(s, sort_keys=True)}", flush=True)
    report = {
        "schema_version": SCHEMA_VERSION,
        "config_name": args.config_name,
        "params": str(args.params),
        "parameter_tree_sha256": parameter_tree_sha256,
        "split": args.split,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "oracle_writes": oracle,
        "prev_is_committed": bool(getattr(model, "memory_v5_prev_is_committed", False)),
        "summary": summary,
        "summary_first_go_step": summary_first,
        "summary_first_go_step_history_only": summary_history,
        "records": records,
        "argv": sys.argv,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
