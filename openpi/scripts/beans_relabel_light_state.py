"""Bean-scoop labels v4 "light state" (user 2026-09-04 19:06 "do 2", after the A-model self-write video showed the
double count: a blink spans two stride-5 memory steps in 86/119 cases and the model cannot tell "still the same
blink" from "a new blink" when the bank says k and the light is on).

Fix: the waiting-phase sentences carry the light state, so the model's previous sentence tells it whether the
light was already on at the last step:
    wait for the light: no green blink yet                (light off, before the first blink)
    light on: k green blink(s) so far                     (LED on, k = blinks so far incl. this one)
    light off: k green blink(s) so far                    (LED off after blink k)
Every other sentence (go / scoop k / done) is unchanged. Source: the v3 labels (subtask_labels.json) for the
segment boundaries and counts, led_on.npy for the per-frame light state inside the waiting phase. Writes
subtask_labels_light.json next to each demo's labels and subtask_labels_manifest_light.json at the dataset root
(create-only). The converted LeRobot dataset keeps the v3 task strings; only the v5 sentence sidecar changes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np

BLINK_RE = re.compile(r"^wait for the light: (\d+) green blinks? so far$")
NO_BLINK = "wait for the light: no green blink yet"


def light_sentence(count: int, on: bool) -> str:
    plural = "blink" if count == 1 else "blinks"
    return f"light {'on' if on else 'off'}: {count} green {plural} so far"


def relabel(segments: list[dict], led_on: np.ndarray) -> list[dict]:
    out: list[dict] = []
    for segment in segments:
        task, start, end = str(segment["task"]), int(segment["start"]), int(segment["end"])
        match = BLINK_RE.match(task)
        if match is None:
            out.append({"task": task, "start": start, "end": end})
            continue
        count = int(match.group(1))
        state = led_on[start : end + 1] > 0.5
        if not state[0]:
            raise ValueError(f"blink segment {task!r} [{start},{end}] does not start with the LED on")
        cursor = start
        for frame in range(start + 1, end + 2):
            if frame == end + 1 or state[frame - start] != state[cursor - start]:
                out.append({"task": light_sentence(count, bool(state[cursor - start])), "start": cursor, "end": frame - 1})
                cursor = frame
    # merge consecutive identical sentences (defensive; the LED signal can only alternate)
    merged: list[dict] = []
    for segment in out:
        if merged and merged[-1]["task"] == segment["task"] and merged[-1]["end"] + 1 == segment["start"]:
            merged[-1]["end"] = segment["end"]
        else:
            merged.append(dict(segment))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("/iris/u/kewalk/memory_project/data/0902_bean_scoop"))
    parser.add_argument("--source-manifest", default="subtask_labels_manifest.json")
    parser.add_argument("--label-filename", default="subtask_labels_light.json")
    parser.add_argument("--manifest-filename", default="subtask_labels_manifest_light.json")
    args = parser.parse_args()
    source = json.loads((args.data_dir / args.source_manifest).read_text())
    manifest_path = args.data_dir / args.manifest_filename
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists (create-only)")
    manifest: dict[str, dict] = {}
    vocabulary: set[str] = set()
    on_runs_seen = 0
    for demo_name, entry in sorted(source.items(), key=lambda item: int(re.search(r"(\d+)$", item[0]).group(1))):
        demo = args.data_dir / demo_name
        segments = json.loads((demo / "subtask_labels.json").read_text())
        if [s["task"] for s in segments] != [s["task"] for s in entry["segments"]]:
            raise ValueError(f"{demo_name}: on-disk labels differ from {args.source_manifest}")
        led_on = np.load(demo / "led_on.npy").astype(np.float32)
        if len(led_on) != int(entry["num_frames"]):
            raise ValueError(f"{demo_name}: led_on has {len(led_on)} frames, labels {entry['num_frames']}")
        new_segments = relabel(segments, led_on)
        # checks: tiling, count of on-runs == x
        cursor = 0
        for s in new_segments:
            if s["start"] != cursor or s["end"] < s["start"]:
                raise ValueError(f"{demo_name}: segments do not tile at frame {cursor}")
            cursor = s["end"] + 1
            vocabulary.add(s["task"])
        if cursor != int(entry["num_frames"]):
            raise ValueError(f"{demo_name}: segments end at {cursor}")
        on_runs = sum(1 for s in new_segments if s["task"].startswith("light on:"))
        if on_runs != int(entry["x"]):
            raise ValueError(f"{demo_name}: {on_runs} light-on runs but x={entry['x']}")
        on_runs_seen += on_runs
        (demo / args.label_filename).write_text(json.dumps(new_segments, indent=2) + "\n")
        manifest[demo_name] = {"num_frames": int(entry["num_frames"]), "x": int(entry["x"]), "segments": new_segments}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {len(manifest)} demos ({on_runs_seen} light-on runs), {len(vocabulary)} sentences -> {manifest_path}")
    for s in sorted(vocabulary):
        print("  ", s)


if __name__ == "__main__":
    main()
