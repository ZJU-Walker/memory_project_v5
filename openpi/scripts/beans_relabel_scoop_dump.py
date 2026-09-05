"""Bean-scoop labels v5 "dump-cut scoops" (user 2026-09-04 20:10 "Ok do it", after the A keep_499 oracle videos
showed the model copies the scoop counter from the bank and never detects the bowl-arrival boundary itself: with the
v3 cut, "previous sentence = scoop k, arm over the bowl" is the input both during the dig of scoop k and at the
arrival for scoop k+1 -- identical inputs, different targets, one such step per transition).

Change (a label-level lookahead for the scoop sentences only), --cut:
  delivery_start (default, "v4tray"): `scoop k+1` starts when the arm arrives OVER THE TRAY with scoop k (base joint
      j0 > 0.8), i.e. the dump of scoop k already belongs to `scoop k+1`. Measured over the 119 scoops: the arm is over
      the tray for 50-158 frames (median 83 = ~17 memory steps) and the return from the tray to the bowl takes only
      0-30 frames (median 9), so the tray is the only long, visually distinct state between two digs. "previous =
      scoop k, over the tray" -> `scoop k+1` (k < x) or stay `scoop x` (k == x, then `done` after the dump as before):
      the increment is a memory decision (k vs the count) made in a persistent state.
  delivery_end ("v3dump"): `scoop k+1` starts the frame after delivery k ends (j0 back below 0.6). Rejected: it moves
      the cut by only 0-30 frames (median 9), i.e. ~2 memory steps.
`scoop 1` still starts at the first bowl arrival, `done` still starts after the last dump, the blink/light and go
sentences are untouched.

Source: the light-state labels (subtask_labels_light.json, beans_relabel_light_state.py) for every non-scoop segment,
scripts/beans_build_subtask_labels.events() for the delivery/arrival frames. Writes subtask_labels_v3dump.json next to
each demo and subtask_labels_manifest_v3dump.json at the dataset root (create-only). Vocabulary unchanged (14).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("beans_build_subtask_labels", _HERE / "beans_build_subtask_labels.py")
_labeler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_labeler)

SCOOP_RE = re.compile(r"^scoop (\d+)$")


def relabel(segments: list[dict], ev: dict, cut: str = "delivery_start") -> list[dict]:
    x = int(ev["x"])
    scoops = [s for s in segments if SCOOP_RE.match(s["task"])]
    if [int(SCOOP_RE.match(s["task"]).group(1)) for s in scoops] != list(range(1, x + 1)):
        raise ValueError("scoop segments are not 1..x")
    if scoops[0]["start"] != ev["arrivals"][0] or scoops[-1]["end"] != ev["deliveries"][-1][1]:
        raise ValueError("scoop 1 start / last scoop end do not match the detected events")
    out = []
    for s in segments:
        m = SCOOP_RE.match(s["task"])
        if m is None:
            out.append(dict(s))
            continue
        k = int(m.group(1))
        if cut == "delivery_start":
            start = ev["arrivals"][0] if k == 1 else ev["deliveries"][k - 2][0]
            end = ev["deliveries"][k - 1][0] - 1 if k < x else ev["deliveries"][-1][1]
        else:
            start = ev["arrivals"][0] if k == 1 else ev["deliveries"][k - 2][1] + 1
            end = ev["deliveries"][k - 1][1]
        if not (start <= end):
            raise ValueError(f"scoop {k}: empty segment [{start}, {end}]")
        out.append({"task": s["task"], "start": start, "end": end})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("/iris/u/kewalk/memory_project/data/0902_bean_scoop"))
    parser.add_argument("--source-manifest", default="subtask_labels_manifest_light.json")
    parser.add_argument("--source-label-filename", default="subtask_labels_light.json")
    parser.add_argument("--cut", choices=("delivery_start", "delivery_end"), default="delivery_start")
    parser.add_argument("--label-filename", default="subtask_labels_v4tray.json")
    parser.add_argument("--manifest-filename", default="subtask_labels_manifest_v4tray.json")
    args = parser.parse_args()
    source = json.loads((args.data_dir / args.source_manifest).read_text())
    manifest_path = args.data_dir / args.manifest_filename
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists (create-only)")
    manifest: dict[str, dict] = {}
    vocabulary: set[str] = set()
    shifts = []
    for demo_name, entry in sorted(source.items(), key=lambda item: int(re.search(r"(\d+)$", item[0]).group(1))):
        demo = args.data_dir / demo_name
        segments = json.loads((demo / args.source_label_filename).read_text())
        if [s["task"] for s in segments] != [s["task"] for s in entry["segments"]]:
            raise ValueError(f"{demo_name}: on-disk labels differ from {args.source_manifest}")
        ev = _labeler.events(demo)
        if ev["n"] != int(entry["num_frames"]) or ev["x"] != int(entry["x"]):
            raise ValueError(f"{demo_name}: events n={ev['n']} x={ev['x']} vs labels {entry['num_frames']} x={entry['x']}")
        new_segments = relabel(segments, ev, args.cut)
        cursor = 0
        for s in new_segments:
            if s["start"] != cursor or s["end"] < s["start"]:
                raise ValueError(f"{demo_name}: segments do not tile at frame {cursor}")
            cursor = s["end"] + 1
            vocabulary.add(s["task"])
        if cursor != ev["n"]:
            raise ValueError(f"{demo_name}: segments end at {cursor}")
        for k in range(2, ev["x"] + 1):
            new_start = ev["deliveries"][k - 2][0] if args.cut == "delivery_start" else ev["deliveries"][k - 2][1] + 1
            shifts.append(ev["arrivals"][k - 1] - new_start)
        (demo / args.label_filename).write_text(json.dumps(new_segments, indent=2) + "\n")
        manifest[demo_name] = {"num_frames": ev["n"], "x": ev["x"], "segments": new_segments}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {len(manifest)} demos, {len(vocabulary)} sentences -> {manifest_path}")
    if shifts:
        shifts.sort()
        print(f"scoop k+1 now starts earlier by {shifts[0]}-{shifts[-1]} frames (median {shifts[len(shifts)//2]}) over {len(shifts)} transitions")


if __name__ == "__main__":
    main()
