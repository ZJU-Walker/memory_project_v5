"""v5 generic episode manifest + sentence sidecar for the 0908 task1 find-the-object bin task.

Same outputs as scripts/beans_build_v5_manifest_sidecar.py (schema openpi.v5.generic-manifest.v1 and
openpi.v5.subtask-labels.v1, self-hashed, pinned by SHA256 in the training config), built from
  * the converted LeRobot dataset (meta/episode_sources.json, meta/episodes.jsonl, meta/episode_prompts.json),
  * the converter manifest written by task1_build_episode_manifest.py (carries the group-aware split,
    the class = opened bin and the target of every episode),
  * the per-episode label file recorded in episode_sources.json (subtask_labels.json of the episode view).
Differences to the beans builder: the split is NOT recomputed here (it must stay group-aware: both prompts
of a raw demo share it), `class` is "bin=<k>", and each sidecar episode records target/opened_bin/group
instead of x.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

SCHEMA_MANIFEST = "openpi.v5.generic-manifest.v1"
SCHEMA_SIDECAR = "openpi.v5.subtask-labels.v1"


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lerobot-dir", type=pathlib.Path, required=True)
    parser.add_argument("--episode-manifest", type=pathlib.Path, required=True,
                        help="converter manifest (data/0908_task1_episode_manifest_v1.json)")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-name", default="0908_task1")
    parser.add_argument("--manifest-name", default="task1_episode_manifest_v1.json")
    parser.add_argument("--sidecar-name", default="task1_v5_subtask_labels_v1.json")
    args = parser.parse_args()

    meta = args.lerobot_dir / "meta"
    prompts = json.loads((meta / "episode_prompts.json").read_text())
    sources = json.loads((meta / "episode_sources.json").read_text())
    lengths: dict[int, int] = {}
    for line in (meta / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            lengths[int(record["episode_index"])] = int(record["length"])
    conv = json.loads(args.episode_manifest.read_text())
    conv_by_id = {e["stable_id"]: e for e in conv["episodes"] if e.get("include")}
    if len(conv_by_id) != len(sources) or len(sources) != len(lengths):
        raise ValueError(f"{len(conv_by_id)} manifest episodes vs {len(sources)} converted sources vs {len(lengths)} lengths")

    episodes = []
    sidecar_episodes: dict[str, dict] = {}
    vocabulary: set[str] = set()
    for index_str, source in sorted(sources.items(), key=lambda item: int(item[0])):
        episode_index = int(index_str)
        stable_id = str(source["stable_id"])
        spec = conv_by_id.get(stable_id)
        if spec is None:
            raise ValueError(f"{stable_id}: not in {args.episode_manifest}")
        raw_dir = pathlib.Path(source["raw_dir"])
        label_file = raw_dir / str(source.get("label_file") or "subtask_labels.json")
        num_frames = lengths[episode_index]
        if int(spec["expected_num_frames"]) != num_frames:
            raise ValueError(f"{stable_id}: manifest has {spec['expected_num_frames']} frames, dataset {num_frames}")
        if prompts[index_str] != spec["instruction"]:
            raise ValueError(f"{stable_id}: prompt {prompts[index_str]!r} != manifest {spec['instruction']!r}")
        segments = json.loads(label_file.read_text())
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
        episodes.append({
            "episode_index": episode_index,
            "stable_id": stable_id,
            "raw_dir": str(raw_dir),
            "expected_num_frames": num_frames,
            "include": True,
            "class": spec["class"],
            "split": spec["split"],
            "group": spec["group"],
            "target": spec["target"],
            "prompt": prompts[index_str],
        })
        sidecar_episodes[stable_id] = {
            "num_frames": num_frames,
            "target": spec["target"],
            "opened_bin": int(spec["opened_bin"]),
            "group": spec["group"],
            "segments": side_segments,
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / args.manifest_name
    sidecar_path = args.out_dir / args.sidecar_name
    for path in (manifest_path, sidecar_path):
        if path.exists():
            raise FileExistsError(f"{path} exists (create-only; delete it deliberately to rebuild)")
    manifest = {
        "schema_version": SCHEMA_MANIFEST,
        "dataset": args.dataset_name,
        "lerobot_dir": str(args.lerobot_dir),
        "split_seed": conv.get("split_rule"),
        "split_rule": f"copied from {args.episode_manifest.name}: {conv.get('split_rule')}",
        "episodes": episodes,
    }
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
    body = _canonical(dict(sidecar))
    sidecar["content_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    sidecar_text = _canonical(sidecar)
    sidecar_path.write_text(sidecar_text, encoding="utf-8")
    sidecar_sha = hashlib.sha256(sidecar_text.encode("utf-8")).hexdigest()

    counts: dict[str, dict[str, int]] = {}
    for episode in episodes:
        counts.setdefault(episode["split"], {}).setdefault(episode["class"], 0)
        counts[episode["split"]][episode["class"]] += 1
    print(f"manifest {manifest_path}\n  sha256 {manifest_sha}\n  splits {json.dumps(counts, sort_keys=True)}")
    print(f"sidecar  {sidecar_path}\n  sha256 {sidecar_sha}\n  {len(vocabulary)} sentences")
    for split in ("development", "final_test"):
        print(f"  {split}: {[e['stable_id'] for e in episodes if e['split'] == split]}")


if __name__ == "__main__":
    main()
