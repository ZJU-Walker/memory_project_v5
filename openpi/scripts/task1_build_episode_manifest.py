"""Episode view + converter manifest for the 0908 task1 collection (one converter episode per (demo, target)).

A demo whose opened bin holds two objects is a valid demonstration for two prompts ("find the banana",
"find the box"): identical video and actions, a different instruction and a different closing sentence
("all bins closed, <target> is in bin k"). examples/yam/convert_yam_data_to_lerobot.py wants one raw
directory per episode (duplicate raw_dir is rejected), so this script materialises an *episode view*:

    <view_dir>/<demo>_<target>/   hard links of every raw stream of <demo> (no extra space)
                                  + subtask_labels.json = <demo>/subtask_labels_task1_<target>.json (copy)
                                  + source.json (provenance)

and writes the schema-1 converter manifest with instruction "find the <target>", plus a group-aware
held-out split: both episodes of a raw demo share the split (no video leaks from train into dev), classes
are the opened bin, ranking is sha256(seed|demo); per class the first `--final-test-per-class` demos are
final_test, the next `--dev-per-class` development, the rest train, with the rule that each class's
development set holds at least one two-object demo (both prompts testable). The split is copied into the
v5 manifest later by task1_build_v5_manifest_sidecar.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil

STREAMS = (
    "top_camera_rgb.mp4", "left_camera_rgb.mp4", "right_camera_rgb.mp4",
    "left_joint_positions.npy", "right_joint_positions.npy", "left_joint_velocities.npy",
    "right_joint_velocities.npy", "left_gripper_position.npy", "right_gripper_position.npy",
    "left_control.npy", "right_control.npy", "metadata.json", "write_complete.flag",
)


def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}|{key}".encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True, help="raw collection (demo folders)")
    parser.add_argument("--labels-manifest", type=pathlib.Path, required=True,
                        help="subtask_labels_manifest_task1.json written by task1_build_subtask_labels.py --write")
    parser.add_argument("--view-dir", type=pathlib.Path, required=True)
    parser.add_argument("--manifest-out", type=pathlib.Path, required=True)
    parser.add_argument("--excluded", nargs="*", default=[], help="'<demo>=<reason>' entries recorded as excluded")
    parser.add_argument("--seed", type=int, default=908)
    parser.add_argument("--final-test-per-class", type=int, default=1)
    parser.add_argument("--dev-per-class", type=int, default=2)
    parser.add_argument("--dataset-name", default="0908_task1")
    parser.add_argument("--overwrite-view", action="store_true")
    parser.add_argument("--manifest-only", action="store_true",
                        help="do not touch the view (it must exist and match the label files); rewrite only the manifest, "
                             "e.g. to change the split while a conversion is reading the view")
    args = parser.parse_args()

    labels = json.loads(args.labels_manifest.read_text())
    demos = sorted(labels, key=lambda s: int(s[4:]))
    for d in demos:
        if not labels[d]["ok"]:
            raise SystemExit(f"{d} is not labelled (ok=false); fix it first")

    # ---- split (per raw demo) -------------------------------------------------------------------
    by_class: dict[int, list[str]] = {}
    for d in demos:
        by_class.setdefault(int(labels[d]["opened_bin"]), []).append(d)
    split: dict[str, str] = {}
    for k, members in sorted(by_class.items()):
        members = sorted(members, key=lambda d: _rank(args.seed, d))
        n_ft, n_dev = args.final_test_per_class, args.dev_per_class
        final_test = members[:n_ft]
        dev = members[n_ft : n_ft + n_dev]
        rest = members[n_ft + n_dev :]
        if not any(len(labels[d]["revealed"]) == 2 for d in dev):
            doubles = [d for d in rest if len(labels[d]["revealed"]) == 2]
            if doubles:  # swap the last single dev demo for the best-ranked double
                swapped_out = dev[-1]
                dev = dev[:-1] + [doubles[0]]
                rest = [d for d in rest if d != doubles[0]] + [swapped_out]
        for d in final_test:
            split[d] = "final_test"
        for d in dev:
            split[d] = "development"
        for d in rest:
            split[d] = "train"

    # ---- episode view --------------------------------------------------------------------------
    if args.manifest_only:
        if not args.view_dir.is_dir():
            raise SystemExit(f"--manifest-only needs the existing view {args.view_dir}")
    else:
        if args.view_dir.exists():
            if not args.overwrite_view:
                raise SystemExit(f"{args.view_dir} exists (pass --overwrite-view to rebuild it)")
            shutil.rmtree(args.view_dir)
        args.view_dir.mkdir(parents=True)
    episodes = []
    for d in demos:
        rec = labels[d]
        src = args.data_dir / d
        for target in rec["revealed"]:
            name = f"{d}_{target}"
            dst = args.view_dir / name
            label_src = src / f"subtask_labels_task1_{target}.json"
            segments = json.loads(label_src.read_text())
            if segments[-1]["end"] + 1 != int(rec["num_frames"]):
                raise SystemExit(f"{label_src}: labels do not tile {rec['num_frames']} frames")
            if args.manifest_only:
                if not dst.is_dir() or (dst / "subtask_labels.json").read_bytes() != label_src.read_bytes():
                    raise SystemExit(f"{dst}: missing or its subtask_labels.json differs from {label_src}")
            else:
                dst.mkdir()
                for stream in STREAMS:
                    if (src / stream).exists():
                        os.link(src / stream, dst / stream)
                shutil.copyfile(label_src, dst / "subtask_labels.json")
                (dst / "source.json").write_text(json.dumps({
                    "raw_demo": str(src), "target": target, "label_source": str(label_src),
                    "opened_bin": rec["opened_bin"], "bins": rec["bins"], "arm": rec["arm"],
                    "episode_counter": json.loads((src / "metadata.json").read_text()).get("episode_counter"),
                }, indent=1) + "\n")
            episodes.append({
                "stable_id": f"{args.dataset_name}/{name}",
                "raw_dir": name,
                "instruction": f"find the {target}",
                "include": True,
                "label_file": "subtask_labels.json",
                "expected_num_frames": int(rec["num_frames"]),
                "group": d,
                "target": target,
                "opened_bin": int(rec["opened_bin"]),
                "class": f"bin={rec['opened_bin']}",
                "split": split[d],
                "two_object_bin": len(rec["revealed"]) == 2,
            })
    for item in args.excluded:
        demo, _, reason = item.partition("=")
        episodes.append({"stable_id": f"{args.dataset_name}/{demo}", "raw_dir": demo, "include": False,
                         "exclude_reason": reason or "excluded"})

    vocab = sorted({s["task"] for d in demos for t in labels[d]["revealed"]
                    for s in json.loads((args.data_dir / d / f"subtask_labels_task1_{t}.json").read_text())})
    manifest = {
        "schema_version": 1,
        "created": "2026-09-08",
        "dataset_version": args.dataset_name,
        "raw_root": str(args.view_dir.resolve()),
        "note": ("0908 task1 find-the-object bins. One episode per (demo, revealed target); two-object bins give two "
                 "episodes sharing the video (hard-linked view). Labels: task1_build_subtask_labels.py (auto + "
                 "hand overrides), target-carry closing sentence. Split is per raw demo (group), class = opened bin."),
        "split_rule": (f"per opened bin: rank demos by sha256('{args.seed}|demo'); first {args.final_test_per_class} "
                       f"final_test, next {args.dev_per_class} development (>= 1 two-object demo per class), rest train; "
                       "both targets of a demo share the split"),
        "task_vocabulary": vocab,
        "expected": {
            "included_episodes": sum(1 for e in episodes if e.get("include")),
            "included_frames": sum(e["expected_num_frames"] for e in episodes if e.get("include")),
            "raw_demos": len(demos),
        },
        "episodes": episodes,
    }
    args.manifest_out.write_text(json.dumps(manifest, indent=1) + "\n")
    counts: dict[str, dict[str, int]] = {}
    for e in episodes:
        if e.get("include"):
            counts.setdefault(e["split"], {}).setdefault(e["class"], 0)
            counts[e["split"]][e["class"]] += 1
    print(f"view: {args.view_dir} ({sum(1 for e in episodes if e.get('include'))} episodes from {len(demos)} demos)")
    print(f"manifest: {args.manifest_out}\n  splits (episodes): {json.dumps(counts, sort_keys=True)}")
    for s in ("development", "final_test"):
        print(f"  {s}: {sorted({e['group'] for e in episodes if e.get('include') and e['split'] == s}, key=lambda x: int(x[4:]))}")
    print(f"  {len(vocab)} sentences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
