"""Bean-scoop labels v5 "visible LED" (2026-09-05 01:50, after the demo17 trace): the blink boundaries in led_on.npy
(the LED CONTROL signal) lead the camera by 0-2 frames (measured over all 60 demos: onset lag 1 frame in 73/100, 2 in
18, 0 in 9; offsets alike). With a 5-frame memory stride, ~1 in 5 blink onsets is sampled inside that gap, so the model
is shown a dark LED with an "on" label (training noise) and, in a rollout whose grid hits the onset (demo17: all three
blinks), it answers one step late, the delayed write misses the bank at the next step and the count is lost.

Fix: move every light on/off boundary of the tray-cut light-state labels (subtask_labels_v4tray.json) to the first
frame where the LEFT camera actually shows the change. The LED patch is located per demo as the largest green-channel
rise between 6 frames before and 6 frames after the first signal onset; a frame is "visibly on" when the patch's green
mean exceeds the midpoint of its min/max over the blink window (frames 0 .. go). Only the light sentences move; the
go / scoop / done boundaries are untouched. Each shift must be in [-1, 3] frames (else the demo is reported and the
script stops). Writes subtask_labels_v5vis.json per demo + subtask_labels_manifest_v5vis.json (create-only).
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

import cv2
import numpy as np

LIGHT_RE = re.compile(r"^light (on|off): (\d+) green blinks? so far$")
NO_BLINK = "wait for the light: no green blink yet"


def visible_led(demo: pathlib.Path, led: np.ndarray, go_frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(demo / "left_camera_rgb.mp4"))
    greens = []
    i = 0
    while i <= go_frame + 5:
        ok, frame = cap.read()
        if not ok:
            break
        greens.append(frame[:, :, 1].astype(np.int16))
        i += 1
    cap.release()
    first_on = int(np.flatnonzero(led)[0])
    a, b = greens[first_on - 6], greens[first_on + 6]
    diff = cv2.GaussianBlur((b - a).astype(np.float32), (9, 9), 0)
    y, x = np.unravel_index(int(np.argmax(diff)), diff.shape)
    g = np.array([fr[max(0, y - 6) : y + 7, max(0, x - 6) : x + 7].mean() for fr in greens])
    window = g[: go_frame + 1]
    threshold = (window.min() + window.max()) / 2.0
    vis = np.zeros(len(led), dtype=bool)
    vis[: len(g)] = g > threshold
    return vis


def relabel(segments: list[dict], led: np.ndarray, vis: np.ndarray, x: int) -> tuple[list[dict], list[int]]:
    light = [i for i, s in enumerate(segments) if LIGHT_RE.match(s["task"])]
    if not light:
        raise ValueError("no light sentences")
    first, last = light[0], light[-1]
    # the light block = segments[first-1 (no blink)] .. segments[last]; every boundary inside it moves
    # to the visible transition nearest AFTER the signal transition (search -1..+3 frames)
    shifts = []
    new = [dict(s) for s in segments]
    for i in range(first, last + 1):
        s = segments[i]
        signal_start = s["start"]
        want_on = LIGHT_RE.match(s["task"]).group(1) == "on"
        cand = None
        for f in range(signal_start - 1, signal_start + 4):
            if 0 <= f < len(vis) and vis[f] == want_on and (f == 0 or vis[f - 1] != want_on):
                cand = f
                break
        if cand is None:
            raise ValueError(f"no visible {'on' if want_on else 'off'} transition near frame {signal_start} ({s['task']})")
        shifts.append(cand - signal_start)
        new[i]["start"] = cand
        new[i - 1]["end"] = cand - 1
    for a, b in zip(new, new[1:]):
        if b["start"] != a["end"] + 1 or a["end"] < a["start"]:
            raise ValueError(f"segments do not tile after the shift: {a} -> {b}")
    if sum(1 for s in new if s["task"].startswith("light on:")) != x:
        raise ValueError("light-on segment count != x")
    return new, shifts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("/iris/u/kewalk/memory_project/data/0902_bean_scoop"))
    parser.add_argument("--source-manifest", default="subtask_labels_manifest_v4tray.json")
    parser.add_argument("--source-label-filename", default="subtask_labels_v4tray.json")
    parser.add_argument("--label-filename", default="subtask_labels_v5vis.json")
    parser.add_argument("--manifest-filename", default="subtask_labels_manifest_v5vis.json")
    args = parser.parse_args()
    source = json.loads((args.data_dir / args.source_manifest).read_text())
    manifest_path = args.data_dir / args.manifest_filename
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists (create-only)")
    manifest: dict[str, dict] = {}
    vocabulary: set[str] = set()
    all_shifts = collections.Counter()
    for demo_name, entry in sorted(source.items(), key=lambda item: int(re.search(r"(\d+)$", item[0]).group(1))):
        demo = args.data_dir / demo_name
        segments = json.loads((demo / args.source_label_filename).read_text())
        if [s["task"] for s in segments] != [s["task"] for s in entry["segments"]]:
            raise ValueError(f"{demo_name}: on-disk labels differ from {args.source_manifest}")
        led = np.load(demo / "led_on.npy").ravel() > 0.5
        go = np.load(demo / "go_on.npy").ravel() > 0.5
        go_frame = int(np.flatnonzero(go)[0])
        vis = visible_led(demo, led, go_frame)
        new_segments, shifts = relabel(segments, led, vis, int(entry["x"]))
        all_shifts.update(shifts)
        for s in new_segments:
            vocabulary.add(s["task"])
        (demo / args.label_filename).write_text(json.dumps(new_segments, indent=2) + "\n")
        manifest[demo_name] = {"num_frames": int(entry["num_frames"]), "x": int(entry["x"]), "segments": new_segments}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {len(manifest)} demos, {len(vocabulary)} sentences -> {manifest_path}")
    print("boundary shifts (visible - signal, frames):", dict(sorted(all_shifts.items())))


if __name__ == "__main__":
    main()
