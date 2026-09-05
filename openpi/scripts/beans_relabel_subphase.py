"""Bean-scoop labels v6 "sub-phase scoops" (user 2026-09-05 10:22 "lets do the subphase").

Why: with one sentence per scoop cycle the previous sentence is constant through the cycle, so the switch to the
next scoop has to be read from the observation alone, and every cut we tried collides: bowl cut (v3) = "prev scoop
k, over the bowl" while digging k AND when arriving for k+1; tray cut (v4) = "prev scoop k+1, over the tray" while
dumping k AND when arriving with k+1 (openpi/cluster_v5/docs/beans_scoop_analysis.html). The light-state fix
worked because the previous sentence carried the LED state; this does the same for the scoop cycle:

    scoop k: to the tray   from bowl arrival k (dig, carry) until tray arrival k - 1
    scoop k: to the bowl   from tray arrival k (dump, return) until bowl arrival k+1 - 1      (k < x)
    done, put down the scoop and return   from tray arrival x (the last dump, release, return home)

Every (previous sentence, arm position) pair now has one target; the count decision is made at the last tray
arrival: prev "scoop x: to the tray" + over the tray -> "done" instead of "scoop x: to the bowl". The go
sentence ends at bowl arrival 1 as before; the light sentences are the v5 visible-LED ones (unchanged).
Source: subtask_labels_v5vis.json + scripts/beans_build_subtask_labels.events(). Writes subtask_labels_v6sub.json per
demo and subtask_labels_manifest_v6sub.json (create-only). Vocabulary: 16 sentences (x <= 3).
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import pathlib
import re

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("beans_build_subtask_labels", _HERE / "beans_build_subtask_labels.py")
_labeler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_labeler)

SCOOP_RE = re.compile(r"^scoop (\d+)$")
DONE = "done, put down the scoop and return"


def relabel(segments: list[dict], ev: dict) -> list[dict]:
    x = int(ev["x"]); arr = ev["arrivals"]; dl = ev["deliveries"]; n = int(ev["n"])
    out = []
    for s in segments:
        if SCOOP_RE.match(s["task"]) or s["task"] == DONE:
            continue
        out.append(dict(s))
    # out now ends with the go segment (ends at bowl arrival 1 - 1)
    if not out or not out[-1]["task"].startswith("yellow go"):
        raise ValueError("expected the go segment before the scoops")
    if out[-1]["end"] != arr[0] - 1:
        raise ValueError(f"go segment ends at {out[-1]['end']}, bowl arrival 1 is {arr[0]}")
    for k in range(1, x + 1):
        a = arr[k - 1]; t = dl[k - 1][0]
        out.append({"task": f"scoop {k}: to the tray", "start": a, "end": t - 1})
        if k < x:
            out.append({"task": f"scoop {k}: to the bowl", "start": t, "end": arr[k] - 1})
        else:
            out.append({"task": DONE, "start": t, "end": n - 1})
    cursor = 0
    for s in out:
        if s["start"] != cursor or s["end"] < s["start"]:
            raise ValueError(f"segments do not tile at frame {cursor}: {s}")
        cursor = s["end"] + 1
    if cursor != n:
        raise ValueError(f"segments end at {cursor}, episode has {n} frames")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("/iris/u/kewalk/memory_project/data/0902_bean_scoop"))
    parser.add_argument("--source-manifest", default="subtask_labels_manifest_v5vis.json")
    parser.add_argument("--source-label-filename", default="subtask_labels_v5vis.json")
    parser.add_argument("--label-filename", default="subtask_labels_v6sub.json")
    parser.add_argument("--manifest-filename", default="subtask_labels_manifest_v6sub.json")
    args = parser.parse_args()
    source = json.loads((args.data_dir / args.source_manifest).read_text())
    manifest_path = args.data_dir / args.manifest_filename
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists (create-only)")
    manifest: dict[str, dict] = {}
    vocabulary: collections.Counter = collections.Counter()
    lengths: dict[str, list[int]] = collections.defaultdict(list)
    for demo_name, entry in sorted(source.items(), key=lambda item: int(re.search(r"(\d+)$", item[0]).group(1))):
        demo = args.data_dir / demo_name
        segments = json.loads((demo / args.source_label_filename).read_text())
        if [s["task"] for s in segments] != [s["task"] for s in entry["segments"]]:
            raise ValueError(f"{demo_name}: on-disk labels differ from {args.source_manifest}")
        ev = _labeler.events(demo)
        if ev["n"] != int(entry["num_frames"]) or ev["x"] != int(entry["x"]):
            raise ValueError(f"{demo_name}: events disagree with the labels manifest")
        new_segments = relabel(segments, ev)
        for s in new_segments:
            vocabulary[s["task"]] += 1
            lengths[s["task"]].append(s["end"] - s["start"] + 1)
        (demo / args.label_filename).write_text(json.dumps(new_segments, indent=2) + "\n")
        manifest[demo_name] = {"num_frames": ev["n"], "x": ev["x"], "segments": new_segments}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {len(manifest)} demos, {len(vocabulary)} sentences -> {manifest_path}")
    for s, c in sorted(vocabulary.items()):
        l = sorted(lengths[s]); print(f"  {c:3d} segments  frames min/med/max {l[0]}/{l[len(l)//2]}/{l[-1]}  {s}")


if __name__ == "__main__":
    main()
