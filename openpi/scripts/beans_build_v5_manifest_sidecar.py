"""Build the v5 generic episode manifest and the sentence sidecar for the 0902 bean-scoop task.

Inputs (all existing, nothing relabelled here):
  * the converted LeRobot dataset (examples/yam/convert_yam_data_to_lerobot.py): meta/episode_sources.json
    (episode_index -> stable_id / raw_dir), meta/episodes.jsonl (lengths), meta/episode_prompts.json;
  * the per-demo labels written by the beans labeler (cluster_v5/BEANS_LABELS.md): <raw_dir>/subtask_labels.json
    (contiguous {task, start, end} segments; the task string IS the sentence) and the dataset-level
    subtask_labels_manifest.json (num_frames, x per demo).

Outputs (create-only, self-hashed; the config pins their SHA256):
  * <out_dir>/beans_episode_manifest_v1.json  -- schema openpi.v5.generic-manifest.v1
        episodes: episode_index, stable_id, raw_dir, expected_num_frames, include, split, class ("x=<n>"),
        prompt; split = deterministic per-class ranking by sha256(seed, stable_id): the first
        `--final-test-per-class` of each class are final_test, the next `--dev-per-class` development, the rest train.
  * <out_dir>/beans_v5_subtask_labels_v1.json   -- schema openpi.v5.subtask-labels.v1 (data_loader._load_v5_subtask_labels):
        sentences (sorted vocabulary), episodes{stable_id: {segments: [{start, end, sentence}], num_frames, x}},
        source_manifest_sha256 = SHA256 of the manifest file written above, content_sha256 self-hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re

SCHEMA_MANIFEST = "openpi.v5.generic-manifest.v1"
SCHEMA_SIDECAR = "openpi.v5.subtask-labels.v1"


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _rank(seed: int, stable_id: str) -> str:
    return hashlib.sha256(f"{seed}|{stable_id}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lerobot-dir", type=pathlib.Path, required=True, help="converted dataset root (has meta/)")
    parser.add_argument("--labels-manifest", type=pathlib.Path, required=True, nargs="+",
                        help="subtask_labels_manifest.json; repeat for a dataset built from several raw "
                             "collections (2026-09-05: the 0905 beans set spans 3 folders that each "
                             "contain a demo1, so entries are keyed <collection>/<demo> as well)")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=902)
    parser.add_argument("--final-test-per-class", type=int, default=2)
    parser.add_argument("--dev-per-class", type=int, default=2)
    parser.add_argument("--dataset-name", default="0902_bean_scoop")
    parser.add_argument("--label-filename", default="subtask_labels.json",
                        help="per-demo label file (v3: subtask_labels.json; light-state v4: subtask_labels_light.json)")
    parser.add_argument("--manifest-name", default="beans_episode_manifest_v1.json")
    parser.add_argument("--sidecar-name", default="beans_v5_subtask_labels_v1.json")
    parser.add_argument("--reuse-manifest", type=pathlib.Path, default=None,
                        help="existing episode manifest to pin instead of writing a new one (same split/dataset)")
    parser.add_argument("--raw-dir", type=pathlib.Path, default=None,
                        help="raw demo root (demo1..demoN); needed when the dataset has no meta/episode_sources.json "
                             "(single-source conversions): episodes were converted in natural demo order and are "
                             "re-identified by that order, verified by frame count and task set")
    args = parser.parse_args()

    meta = args.lerobot_dir / "meta"
    prompts = json.loads((meta / "episode_prompts.json").read_text())
    lengths = {}
    episode_tasks = {}
    for line in (meta / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            lengths[int(record["episode_index"])] = int(record["length"])
            episode_tasks[int(record["episode_index"])] = set(record.get("tasks", []))
    # Key every entry as "<collection>/<demo>" (the raw folder holding the manifest) and, when unambiguous,
    # also by the bare demo name so single-collection callers are unaffected.
    labels_manifest: dict[str, dict] = {}
    _bare_seen: set[str] = set()
    for _mpath in args.labels_manifest:
        _collection = _mpath.resolve().parent.name
        for _demo, _entry in json.loads(_mpath.read_text()).items():
            labels_manifest[f"{_collection}/{_demo}"] = _entry
            if _demo in _bare_seen:
                labels_manifest.pop(_demo, None)  # ambiguous across collections: qualified key only
            else:
                labels_manifest[_demo] = _entry
                _bare_seen.add(_demo)
    sources_path = meta / "episode_sources.json"
    if sources_path.exists():
        sources = json.loads(sources_path.read_text())
    else:
        if args.raw_dir is None:
            raise SystemExit(f"{sources_path} is missing; pass --raw-dir to re-identify episodes by natural demo order")

        def natural_key(path: pathlib.Path):
            match = re.search(r"(\d+)$", path.name)
            return (int(match.group(1)) if match else 10**9, path.name)

        demo_dirs = sorted((d for d in args.raw_dir.glob("demo*") if d.is_dir()), key=natural_key)
        demo_dirs = [d for d in demo_dirs if (d / "subtask_labels.json").exists()]
        if len(demo_dirs) != len(lengths):
            raise ValueError(f"{len(demo_dirs)} labelled demo folders vs {len(lengths)} converted episodes")
        sources = {}
        for episode_index, demo in enumerate(demo_dirs):
            entry = labels_manifest[demo.name]
            tasks = {segment["task"] for segment in entry["segments"]}
            same_vocab = args.label_filename == "subtask_labels.json"  # the dataset carries the v3 task strings
            if int(entry["num_frames"]) != lengths[episode_index] or (same_vocab and tasks != episode_tasks[episode_index]):
                raise ValueError(
                    f"episode {episode_index} does not match {demo.name} (frames {lengths[episode_index]} vs "
                    f"{entry['num_frames']}, tasks {sorted(episode_tasks[episode_index])} vs {sorted(tasks)})"
                )
            sources[str(episode_index)] = {"stable_id": f"{args.dataset_name}/{demo.name}", "raw_dir": str(demo)}
        print(f"re-identified {len(sources)} episodes by natural demo order (frame counts and task sets verified)")

    episodes = []
    sidecar_episodes: dict[str, dict] = {}
    vocabulary: set[str] = set()
    for index_str, source in sorted(sources.items(), key=lambda item: int(item[0])):
        episode_index = int(index_str)
        stable_id = str(source["stable_id"])
        raw_dir = pathlib.Path(source["raw_dir"])
        demo_name = raw_dir.name
        qualified = f"{raw_dir.parent.name}/{demo_name}"
        key = qualified if qualified in labels_manifest else demo_name
        if key not in labels_manifest:
            raise ValueError(f"{stable_id}: no entry ({qualified!r} or {demo_name!r}) in {args.labels_manifest}")
        entry = labels_manifest[key]
        num_frames = lengths[episode_index]
        if int(entry["num_frames"]) != num_frames:
            raise ValueError(f"{stable_id}: labels manifest has {entry['num_frames']} frames, dataset {num_frames}")
        segments = json.loads((raw_dir / args.label_filename).read_text())
        if [s["task"] for s in segments] != [s["task"] for s in entry["segments"]]:
            raise ValueError(f"{stable_id}: on-disk labels differ from the labels manifest")
        cursor = 0
        side_segments = []
        for segment in segments:
            start, end, sentence = int(segment["start"]), int(segment["end"]), str(segment["task"])
            if start != cursor or end < start:
                raise ValueError(f"{stable_id}: segments do not tile the episode at frame {cursor}")
            side_segments.append({"start": start, "end": end, "sentence": sentence})
            vocabulary.add(sentence)
            cursor = end + 1
        if cursor != num_frames:
            raise ValueError(f"{stable_id}: segments end at {cursor}, episode has {num_frames} frames")
        x = int(entry["x"])
        episodes.append(
            {
                "episode_index": episode_index,
                "stable_id": stable_id,
                "raw_dir": str(raw_dir),
                "expected_num_frames": num_frames,
                "include": True,
                "class": f"x={x}",
                "prompt": prompts[index_str],
            }
        )
        sidecar_episodes[stable_id] = {"num_frames": num_frames, "x": x, "segments": side_segments}

    # Deterministic stratified split.
    by_class: dict[str, list[dict]] = {}
    for episode in episodes:
        by_class.setdefault(episode["class"], []).append(episode)
    for class_name, members in by_class.items():
        members.sort(key=lambda e: _rank(args.seed, e["stable_id"]))
        for rank, episode in enumerate(members):
            if rank < args.final_test_per_class:
                episode["split"] = "final_test"
            elif rank < args.final_test_per_class + args.dev_per_class:
                episode["split"] = "development"
            else:
                episode["split"] = "train"
    episodes.sort(key=lambda e: e["episode_index"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.reuse_manifest or (args.out_dir / args.manifest_name)
    sidecar_path = args.out_dir / args.sidecar_name
    for path in ([sidecar_path] if args.reuse_manifest else [manifest_path, sidecar_path]):
        if path.exists():
            raise FileExistsError(f"{path} exists (create-only; delete it deliberately to rebuild)")
    manifest = {
        "schema_version": SCHEMA_MANIFEST,
        "dataset": args.dataset_name,
        "lerobot_dir": str(args.lerobot_dir),
        "split_seed": args.seed,
        "split_rule": (
            f"per class (target count): rank by sha256('{args.seed}|stable_id'); first "
            f"{args.final_test_per_class} final_test, next {args.dev_per_class} development, rest train"
        ),
        "episodes": episodes,
    }
    if args.reuse_manifest:
        manifest_text = manifest_path.read_text(encoding="utf-8")
        existing = json.loads(manifest_text)
        if [(e["episode_index"], e["stable_id"]) for e in existing["episodes"]] != [(e["episode_index"], e["stable_id"]) for e in episodes]:
            raise ValueError("the reused manifest does not list the same episodes in the same order")
        for e, old in zip(episodes, existing["episodes"]):
            e["split"] = old["split"]
    else:
        manifest_text = _canonical(manifest)
        manifest_path.write_text(manifest_text, encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()

    sidecar = {
        "schema_version": SCHEMA_SIDECAR,
        "dataset_version": args.dataset_name,
        "num_episodes": len(sidecar_episodes),
        "sentences": sorted(vocabulary),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_sha,
        "episodes": sidecar_episodes,
    }
    body = _canonical({k: v for k, v in sidecar.items()})
    sidecar["content_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    sidecar_text = _canonical(sidecar)
    sidecar_path.write_text(sidecar_text, encoding="utf-8")
    sidecar_sha = hashlib.sha256(sidecar_text.encode("utf-8")).hexdigest()

    counts = {}
    for episode in episodes:
        counts.setdefault(episode["split"], {}).setdefault(episode["class"], 0)
        counts[episode["split"]][episode["class"]] += 1
    print(f"manifest {manifest_path}\n  sha256 {manifest_sha}\n  splits {json.dumps(counts, sort_keys=True)}")
    print(f"sidecar  {sidecar_path}\n  sha256 {sidecar_sha}\n  {len(vocabulary)} sentences")
    for split in ("development", "final_test"):
        ids = [e["stable_id"] for e in episodes if e["split"] == split]
        print(f"  {split}: {ids}")


if __name__ == "__main__":
    main()
