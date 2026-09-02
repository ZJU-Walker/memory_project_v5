"""Derive the v5 detailed-subtask sentence sidecar from the frozen v36 manifest
(cluster_v5/README.md §4).

No relabeling and no new data: every sentence is a deterministic function of the manifest's
3x side-verified fields and of the hashed raw per-episode phase segments (`subtask_labels.json`,
authenticated against the manifest's `label_sha256`). The bin scene always holds exactly two
objects, one per bin, so the only phase whose sentence gains content is the inspection phase:

    "inspect both bins"  ->  "inspect both bins: banana {side}, grey pepper box {side}"

Every other phase keeps its canonical string (`open both lids`, `close both lids and reset
arms`, `wait; target bin is {side}`, `open {side} bin`) -- the waiting label already names the
target bin, which is where the memory read has to pay off, and the side-flip battery swaps that
single side token. Sentence per (episode, frame) is stored as run-length segments.

The output sidecar is create-only, self-hashed, pins the SHA-256 of the manifest it was derived
from, and (when the v4 fact sidecar is available) cross-checks the derived sides against it.
"""

import argparse
import hashlib
import json
import pathlib

SCHEMA_VERSION = "openpi.v5.subtask-labels.v1"
INSPECT_TASK = "inspect both bins"
OBJECT_NAMES = {"banana": "banana", "grey_pepper_box": "grey pepper box"}
OBJECT_ORDER = ("banana", "grey_pepper_box")
_OPPOSITE = {"left": "right", "right": "left"}
_SIDE_TARGET = {"left": 0, "right": 1}  # v4 fact-label target ids (left_bin, right_bin)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_dumps(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def object_sides(episode: dict) -> dict[str, str]:
    """{object: side} for one included manifest episode; the other object is opposite."""
    prompted = episode["object"]
    side = episode["target_side"]
    if prompted not in OBJECT_NAMES:
        raise ValueError(f"{episode['stable_id']}: unknown object {prompted!r}.")
    if side not in _OPPOSITE:
        raise ValueError(f"{episode['stable_id']}: unknown target_side {side!r}.")
    return {name: (side if name == prompted else _OPPOSITE[side]) for name in OBJECT_ORDER}


def inspect_sentence(sides: dict[str, str]) -> str:
    parts = ", ".join(f"{OBJECT_NAMES[name]} {sides[name]}" for name in OBJECT_ORDER)
    return f"{INSPECT_TASK}: {parts}"


def sentence_for(task: str, sides: dict[str, str]) -> str:
    return inspect_sentence(sides) if task == INSPECT_TASK else task


def resolve_label_path(manifest_path: pathlib.Path, manifest: dict, episode: dict) -> pathlib.Path:
    raw_root = pathlib.Path(manifest["raw_root"])
    if not raw_root.is_absolute():
        raw_root = manifest_path.parent / raw_root
    return (raw_root.resolve() / episode["raw_dir"] / episode["label_file"]).resolve()


def build_episode_segments(label_bytes: bytes, sides: dict[str, str], stable_id: str) -> list[dict]:
    segments = json.loads(label_bytes)
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{stable_id}: label file holds no segments.")
    out = []
    expected_start = 0
    for segment in segments:
        start, end, task = int(segment["start"]), int(segment["end"]), str(segment["task"])
        if start != expected_start or end < start:
            raise ValueError(f"{stable_id}: segments are not contiguous from frame 0 ({start}, {end}).")
        out.append({"start": start, "end": end, "task": task, "sentence": sentence_for(task, sides)})
        expected_start = end + 1
    return out


def build_subtask_labels(manifest_path: pathlib.Path, fact_labels_path: pathlib.Path | None) -> dict:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("review_status") != "frozen":
        raise ValueError(f"manifest {manifest_path} is not frozen (review_status={manifest.get('review_status')!r}).")
    fact_records = None
    fact_sha256 = None
    if fact_labels_path is not None and fact_labels_path.is_file():
        fact_bytes = fact_labels_path.read_bytes()
        fact_sha256 = _sha256_bytes(fact_bytes)
        fact_payload = json.loads(fact_bytes)
        if fact_payload.get("source_manifest_sha256") != _sha256_bytes(manifest_bytes):
            raise ValueError("the v4 fact sidecar was derived from a different manifest.")
        fact_records = fact_payload["episodes"]
    episodes = [e for e in manifest["episodes"] if e.get("include")]
    if not episodes:
        raise ValueError("the manifest contains no included episodes.")
    records = {}
    sentences = set()
    for episode in episodes:
        stable_id = episode["stable_id"]
        if stable_id in records:
            raise ValueError(f"duplicate stable_id {stable_id!r}.")
        sides = object_sides(episode)
        if fact_records is not None:
            expected = [_SIDE_TARGET[sides[name]] for name in OBJECT_ORDER]
            if list(fact_records[stable_id]["fact_targets"]) != expected:
                raise ValueError(f"{stable_id}: derived sides disagree with the v4 fact sidecar.")
        label_path = resolve_label_path(manifest_path, manifest, episode)
        label_bytes = label_path.read_bytes()
        if _sha256_bytes(label_bytes) != episode["label_sha256"]:
            raise ValueError(f"{stable_id}: raw label file {label_path} does not match the manifest label_sha256.")
        segments = build_episode_segments(label_bytes, sides, stable_id)
        if segments[-1]["end"] != int(episode["expected_num_frames"]) - 1:
            raise ValueError(f"{stable_id}: segments end at {segments[-1]['end']}, expected_num_frames says otherwise.")
        sentences.update(segment["sentence"] for segment in segments)
        records[stable_id] = {
            "split": episode["split"],
            "object_sides": sides,
            "label_sha256": episode["label_sha256"],
            "segments": segments,
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": manifest_path.name,
        "source_manifest_sha256": _sha256_bytes(manifest_bytes),
        "source_v4_fact_labels_sha256": fact_sha256,
        "dataset_version": manifest.get("dataset_version"),
        "templates": {
            INSPECT_TASK: "inspect both bins: banana {banana side}, grey pepper box {grey pepper box side}",
            "other phases": "canonical task string, unchanged",
        },
        "sentences": sorted(sentences),
        "num_episodes": len(records),
        "episodes": records,
    }
    body = _canonical_dumps(payload)
    payload["content_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return payload


def check_sentence_lengths(sentences: list[str], max_tokens: int, tokenizer_model: pathlib.Path | None) -> dict[str, int]:
    """Token count (including the trained "\\n" stop token) of every sentence under the real
    PaliGemma sentencepiece model; fails if any exceeds `max_tokens`."""
    import sentencepiece

    if tokenizer_model is None or not tokenizer_model.is_file():
        raise FileNotFoundError(f"PaliGemma tokenizer model not found: {tokenizer_model}")
    sp = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_model))
    lengths = {}
    for sentence in sentences:
        cleaned = sentence.lower().strip().replace("_", " ")
        lengths[sentence] = len(sp.encode(cleaned + "\n"))
    too_long = {k: v for k, v in lengths.items() if v > max_tokens}
    if too_long:
        raise ValueError(f"sentences exceed {max_tokens} tokens: {too_long}")
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("../data/0830_0831_episode_manifest_v36_frozen.json"))
    parser.add_argument("--fact-labels", type=pathlib.Path, default=pathlib.Path("../data/v4_fact_labels_0830_0831.json"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("../data/v5_subtask_labels_0830_0831.json"))
    parser.add_argument("--max-tokens", type=int, default=48, help="Pi0Config.memory_v5_sentence_len")
    parser.add_argument("--tokenizer-model", type=pathlib.Path, default=None, help="paligemma_tokenizer.model (token-length check)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing sidecar")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists (create-only; pass --force to rebuild).")
    payload = build_subtask_labels(args.manifest, args.fact_labels)
    if args.tokenizer_model is not None:
        lengths = check_sentence_lengths(payload["sentences"], args.max_tokens, args.tokenizer_model)
        for sentence, n in sorted(lengths.items(), key=lambda kv: -kv[1]):
            print(f"  {n:3d} tokens  {sentence}")
    args.output.write_text(_canonical_dumps(payload))
    per_split: dict[str, int] = {}
    for record in payload["episodes"].values():
        per_split[record["split"]] = per_split.get(record["split"], 0) + 1
    print(f"wrote {args.output} ({payload['num_episodes']} episodes, splits={per_split}, {len(payload['sentences'])} distinct sentences)")
    print(f"source manifest sha256: {payload['source_manifest_sha256']}")
    print(f"content sha256: {payload['content_sha256']}")
    print(f"file sha256: {_sha256_bytes(args.output.read_bytes())}")


if __name__ == "__main__":
    main()
