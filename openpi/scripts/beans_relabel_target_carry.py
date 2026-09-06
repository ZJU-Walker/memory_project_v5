"""Bean-scoop labels v7 "target-carry scoops" (2026-09-05 13:40, after the A6 rollouts).

Why: with the v6 sub-phase sentences every write up to the failure is correct, yet at the tray arrival the model says
"done" one scoop early in all four multi-scoop development episodes (A6 demo11/14/17/21; the k=1 "dump" of demo14/17
was at confidence 0.90-0.91). That decision needs TWO bank entries (the target x from the go sentence and k from the
last scoop sentence) plus a comparison; the blink count, which works, needs one entry plus an increment. This keeps
the v6 boundaries and carries the target in every scoop sentence, so each transition is a single-read copy /
increment / compare of the previous sentence:

    yellow go: ..., scoop x times          (unchanged)
    scoop k of x: dig and carry            from bowl arrival k   (copy x, k = last dump + 1)
    scoop k of x: dump and return          from tray arrival k, k < x   (copy k and x)
    done, put down the scoop and return    from tray arrival x   (k == x in the previous sentence)

Source: subtask_labels_v6sub.json + subtask_labels_manifest_v6sub.json (x per demo). Writes subtask_labels_v7tgt.json
per demo and subtask_labels_manifest_v7tgt.json (create-only). Vocabulary: 20 sentences for x <= 3 (1 no-blink + 6 light + 3 go +
6 dig + 3 dump + done).
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

SCOOP_RE = re.compile(r"^scoop (\d+): (dig and carry|dump and return)$")


def relabel(segments: list[dict], x: int) -> list[dict]:
    out = []
    for s in segments:
        m = SCOOP_RE.match(s["task"])
        if m:
            k = int(m.group(1))
            if not 1 <= k <= x:
                raise ValueError(f"scoop {k} with x={x}")
            if m.group(2) == "dump and return" and k >= x:
                raise ValueError(f"dump {k} with x={x}")
            out.append({"task": f"scoop {k} of {x}: {m.group(2)}", "start": s["start"], "end": s["end"]})
        else:
            out.append(dict(s))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("/iris/u/kewalk/memory_project/data/0902_bean_scoop"))
    parser.add_argument("--source-manifest", default="subtask_labels_manifest_v6sub.json")
    parser.add_argument("--source-label-filename", default="subtask_labels_v6sub.json")
    parser.add_argument("--label-filename", default="subtask_labels_v7tgt.json")
    parser.add_argument("--manifest-filename", default="subtask_labels_manifest_v7tgt.json")
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
        if segments != entry["segments"]:
            raise ValueError(f"{demo_name}: on-disk labels differ from {args.source_manifest}")
        x = int(entry["x"])
        new_segments = relabel(segments, x)
        digs = [s for s in new_segments if s["task"].endswith("dig and carry")]
        if len(digs) != x or not new_segments[-1]["task"].startswith("done"):
            raise ValueError(f"{demo_name}: {len(digs)} digs for x={x} / last {new_segments[-1]['task']}")
        for s in new_segments:
            vocabulary[s["task"]] += 1
            lengths[s["task"]].append(s["end"] - s["start"] + 1)
        (demo / args.label_filename).write_text(json.dumps(new_segments, indent=2) + "\n")
        manifest[demo_name] = {"num_frames": int(entry["num_frames"]), "x": x, "segments": new_segments}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {len(manifest)} demos, {len(vocabulary)} sentences -> {manifest_path}")
    for s, c in sorted(vocabulary.items()):
        l = sorted(lengths[s]); print(f"  {c:3d} segments  frames min/med/max {l[0]}/{l[len(l)//2]}/{l[-1]}  {s}")


if __name__ == "__main__":
    main()
