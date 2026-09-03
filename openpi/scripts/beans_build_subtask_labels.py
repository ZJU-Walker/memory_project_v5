"""Build per-episode `subtask_labels.json` for 0902_bean_scoop from the logged signals + right-arm joints.

Definition: cluster_v5/BEANS_LABELS.md (v3). Same file format as the bins task: a list of
{"task", "start", "end"} segments tiling [0, num_frames-1]. `--dry-run` prints the segments and writes nothing.
"""
import argparse, glob, json, os, pathlib
import numpy as np

DATA = pathlib.Path("/iris/u/kewalk/memory_project/data/0902_bean_scoop")


def segs(mask):
    mask = np.asarray(mask).astype(bool); out = []; start = None
    for i, v in enumerate(mask):
        if v and start is None: start = i
        if not v and start is not None: out.append((start, i - 1)); start = None
    if start is not None: out.append((start, len(mask) - 1))
    return out


# "Over the white bowl" (the bean SOURCE; the glass tray is the destination) from the right arm's
# base joint j0: the arm dumps into the tray at j0 > 0.8 and digs in the bowl at j0 in [-0.1, 0.75].
# Arrival = the first frame in that band where j0 has stopped moving (|dj0/dt| < VMAX over DWELL
# consecutive frames) -- the band alone fires mid-swing, while the arm is still travelling.
# Verified frame-by-frame on 12 sampled arrivals across 6 episodes (scoop over the bowl in all).
BOWL_BAND = (-0.10, 0.75)
BOWL_DWELL = 10
BOWL_VMAX = 0.006


def bowl_arrival(j0: np.ndarray, start: int, stop: int) -> int | None:
    """First frame in [start, stop) where the scoop has settled over the white bowl."""
    n = len(j0)
    speed = np.zeros(n)
    speed[4 : n - 4] = np.abs(j0[8:] - j0[:-8]) / 8.0
    settled = (j0 >= BOWL_BAND[0]) & (j0 <= BOWL_BAND[1]) & (speed < BOWL_VMAX)
    for i in range(start, min(stop, n - BOWL_DWELL)):
        if settled[i : i + BOWL_DWELL].all():
            return i
    return None


def events(ep: pathlib.Path) -> dict:
    j0 = np.load(ep / "right_joint_positions.npy")[:, 0]
    g = np.load(ep / "right_gripper_position.npy")[:, 0]
    led = np.load(ep / "led_on.npy")[:, 0]
    go = np.load(ep / "go_on.npy")[:, 0]
    x = int(np.load(ep / "cue_num_blinks.npy").max())
    n = len(g)
    blinks = segs(led > 0.5)
    go_frame = segs(go > 0.5)[0][0]
    hold = [h for h in segs(g < 0.5) if h[1] - h[0] > 60]
    state, deliveries, cur = 0, [], None
    for i in range(n):
        if state == 0 and j0[i] > 0.8: state, cur = 1, [i, i]
        elif state == 1:
            if j0[i] < 0.6: state = 0; deliveries.append(tuple(cur)); cur = None
            else: cur[1] = i
    if cur: deliveries.append(tuple(cur))
    if len(blinks) != x or len(hold) != 1 or len(deliveries) != x:
        raise ValueError(f"{ep.name}: x={x} blinks={len(blinks)} hold={len(hold)} deliveries={len(deliveries)}")
    if not (blinks[-1][1] < go_frame < hold[0][0] < deliveries[0][0] and deliveries[-1][1] < hold[0][1]):
        raise ValueError(f"{ep.name}: event order violated")
    arrivals = [bowl_arrival(j0, hold[0][0], deliveries[0][0])]
    arrivals += [bowl_arrival(j0, deliveries[k - 1][1] + 1, deliveries[k][0]) for k in range(1, x)]
    if any(a is None for a in arrivals):
        raise ValueError(f"{ep.name}: no bowl arrival found for scoop {arrivals.index(None) + 1}")
    return dict(n=n, x=x, blinks=blinks, go=go_frame, pickup=hold[0][0], release=hold[0][1],
                deliveries=deliveries, arrivals=arrivals)


def plural(k: int) -> str:
    return "1 green blink so far" if k == 1 else f"{k} green blinks so far"


def build(ev: dict) -> list[dict]:
    x, n = ev["x"], ev["n"]
    out = []
    out.append({"task": "wait for the light: no green blink yet", "start": 0, "end": ev["blinks"][0][0] - 1})
    for k, (on, _off) in enumerate(ev["blinks"], start=1):
        end = ev["blinks"][k][0] - 1 if k < x else ev["go"] - 1
        out.append({"task": f"wait for the light: {plural(k)}", "start": on, "end": end})
    # v3 (user 2026-09-03 15:05): the go segment covers reaching for and grasping the scoop AND
    # carrying it to the bowl; each scoop segment starts when the scoop is over the white bowl.
    out.append({"task": f"yellow go: pick up the scoop, scoop {x} time{'' if x == 1 else 's'}",
                "start": ev["go"], "end": ev["arrivals"][0] - 1})
    for k in range(1, x + 1):
        # v2 (user 2026-09-03 14:28): progress only, x is NOT restated. The stop decision at the
        # last scoop therefore cannot be read off the current sentence -- it needs the go sentence
        # ("scoop x times") or the blink count, both many memory steps in the past. That is the
        # memory test; restating x here would make a previous-sentence-only model sufficient.
        # A scoop segment runs bowl arrival k -> bowl arrival k+1 (the last one -> its dump end),
        # so it contains one whole cycle: dig, carry, dump, return.
        end = ev["arrivals"][k] - 1 if k < x else ev["deliveries"][-1][1]
        out.append({"task": f"scoop {k}", "start": ev["arrivals"][k - 1], "end": end})
    out.append({"task": "done, put down the scoop and return", "start": ev["deliveries"][-1][1] + 1, "end": n - 1})
    # must tile
    assert out[0]["start"] == 0 and out[-1]["end"] == n - 1
    for a, b in zip(out, out[1:]):
        assert b["start"] == a["end"] + 1 and a["end"] >= a["start"], (a, b)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=pathlib.Path, default=DATA)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--episodes", nargs="*", default=None, help="demo names; default all")
    a = p.parse_args()
    eps = sorted(a.data.glob("demo*"), key=lambda q: int(q.name[4:]))
    if a.episodes: eps = [a.data / e for e in a.episodes]
    manifest = {}
    for ep in eps:
        ev = events(ep); labels = build(ev)
        manifest[ep.name] = {"num_frames": ev["n"], "x": ev["x"], "segments": labels}
        if a.dry_run:
            print(f"== {ep.name} frames={ev['n']} x={ev['x']}")
            for s in labels: print(f"   {s['start']:5d}-{s['end']:5d}  {s['task']}")
        else:
            (ep / "subtask_labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    if not a.dry_run:
        (a.data / "subtask_labels_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"wrote subtask_labels.json for {len(manifest)} episodes + subtask_labels_manifest.json")


if __name__ == "__main__":
    main()
