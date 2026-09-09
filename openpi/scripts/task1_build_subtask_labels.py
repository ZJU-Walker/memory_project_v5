"""Automatic subtask labels for the 0908 "task1" find-the-object bin task.

Task (data/0908_task1_new, 2026-09-08): a human puts four objects (banana, spoon, box, tape) into three
lidded bins left->right, one bin gets two objects, every lid is closed right after its bin is filled;
~3 s later the robot opens the bin that holds the asked object (left arm = bin 1, right arm = bins 2/3).

Inputs
  * the audited inspection table (data/0908_task1_inspection/0908_task1_inspection_v1.json): objects per
    bin, opened bin -- checked by eye for every demo;
  * the three videos of each demo (top camera for placements and lid states, wrist cameras as an
    independent lid-close witness for bins 1 and 3, which the human's head sometimes hides from the top);
  * optional manual overrides (--overrides JSON: {demo: {"placements": {"obj": frame}, "closes": {"1": frame}}}).

Sentences (base file, prompt independent)
  watching: no object placed yet                 [0, first placement)
  {obj} in bin {k}                               [placement_i, placement_{i+1})  (last one until lid 3 closes)
  all bins closed                                [close_3, robot_start)
  open bin {k}                                   [robot_start, end]
Per-target files replace the closing sentence by "all bins closed, {target} is in bin {k}" (target-carry,
the same trick as the beans "scoop k of x" labels): one file per object revealed in the opened bin.

Boundary rules (persistent, visually distinct states)
  * placement: the bin's content is compared with its final content just before the lid; a placement
    starts at the first unoccluded frame from which the coverage of the final content stays above the
    step for that object (persistently, >= 80 % of the later unoccluded frames). Which object arrived first
    in a two-object bin is decided pairwise (only between the two objects known to be there) from the
    colour of the blob present between the two steps;
  * lid close: the first frame of the lid-on state that never reverts before the robot moves (top camera;
    the wrist camera witness replaces it only when the top view was blocked in between);
  * robot phase: first joint motion (> 0.05 rad).

Outputs (--write)
  <demo>/subtask_labels_task1_base.json, <demo>/subtask_labels_task1_<target>.json,
  <data_dir>/subtask_labels_manifest_task1.json, audit montages under --audit-dir.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

OBJECTS = ("banana", "spoon", "box", "tape")
TABLE_TO_LABEL = {"banana": "banana", "spoon": "spoon", "bottle": "box", "ring": "tape"}
WATCH = "watching: no object placed yet"
ALL_CLOSED = "all bins closed"
KER3 = np.ones((3, 3), np.uint8)
OPEN, CLOSED, UNK = 1, 2, 0


def read_video(path: pathlib.Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"no frames in {path}")
    return np.stack(frames)


def white_mask(hsv: np.ndarray) -> np.ndarray:
    return (hsv[..., 1] < 50) & (hsv[..., 2] > 160)


def wood_mask(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    return (h >= 8) & (h <= 26) & (s >= 60) & (s <= 180) & (v >= 110)


def find_bins(frame0: np.ndarray) -> list[tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(frame0, cv2.COLOR_BGR2HSV)
    m = white_mask(hsv).astype(np.uint8)
    m[:140] = 0
    m[300:] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    comps = sorted((stats[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 1500), key=lambda s: -s[cv2.CC_STAT_AREA])[:3]
    boxes = sorted((tuple(int(v) for v in s[:4]) for s in comps), key=lambda b: b[0])
    if len(boxes) != 3:
        raise ValueError(f"found {len(boxes)} bins in frame 0")
    return boxes


def roi_of(box: tuple[int, int, int, int], margin: int = 5) -> tuple[int, int, int, int]:
    x, y, w, h = box
    return (x + margin, y + margin, x + w - margin, y + h - margin)


# ----------------------------------------------------------------------------- content masks


def content_mask(hsv_crop: np.ndarray, bg: np.ndarray) -> tuple[np.ndarray, bool, float]:
    """Non-white pixels inside the bin (objects, hands, lid), whether the frame is blocked (head: dark, or
    the bin mostly covered), and the wood/skin fraction (lid or hand present)."""
    dark = (hsv_crop[..., 2] < 70).mean()
    nonwhite = ~white_mask(hsv_crop)
    nw = nonwhite.astype(np.uint8)
    nw[bg] = 0
    nw = cv2.morphologyEx(nw, cv2.MORPH_OPEN, KER3)
    blocked = dark > 0.4 or nonwhite.mean() > 0.7
    return nw.astype(bool), blocked, float(wood_mask(hsv_crop).mean())


def sliding_median(masks: list[np.ndarray], valid: np.ndarray, half: int = 7) -> list[np.ndarray]:
    """Per-frame temporal median of the content masks over +-half valid frames: moving hands and the
    descending lid vanish, resting objects stay (the placement time is then read off the raw frames)."""
    out = []
    n = len(masks)
    for t in range(n):
        idx = [u for u in range(max(0, t - half), min(n, t + half + 1)) if valid[u]]
        if len(idx) < 3:
            out.append(np.zeros_like(masks[t]))
            continue
        out.append(np.median(np.stack([masks[u] for u in idx]).astype(np.uint8), axis=0) >= 0.5)
    return out


def first_sustained_closed(state: np.ndarray, limit: int, soft: np.ndarray | None = None) -> int:
    """First CLOSED frame with no later OPEN frame before `limit` and a mostly-closed 20-frame run.
    `soft` marks frames with the lid mostly on but a hand still resting on it; the close is moved back
    over a soft run that leads into the hard close (the top camera otherwise reports it ~1 s late)."""
    opens = np.where(state[:limit] == OPEN)[0]
    last_open = int(opens[-1]) if len(opens) else -1
    hard = -1
    for t in range(last_open + 1, max(limit - 5, last_open + 1)):
        if state[t] == CLOSED and (state[t : t + 20] == CLOSED).mean() >= 0.6:
            hard = t
            break
    if hard < 0 or soft is None:
        return hard
    t = hard
    while t - 1 > last_open and soft[t - 1]:
        t -= 1
    return t


def wrist_close(frames: np.ndarray, roi: tuple[int, int, int, int], limit: int) -> int:
    x0, y0, x1, y1 = roi
    st = np.zeros(len(frames), dtype=int)
    for t in range(len(frames)):
        hsv = cv2.cvtColor(frames[t][y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        wf = wood_mask(hsv).mean()
        wh = white_mask(hsv).mean()
        st[t] = CLOSED if wf > 0.5 else OPEN if wh > 0.3 else UNK
    return first_sustained_closed(st, limit)


def robot_start(demo: pathlib.Path) -> tuple[int, str]:
    left = np.load(demo / "left_joint_positions.npy")
    right = np.load(demo / "right_joint_positions.npy")

    def first_move(a: np.ndarray) -> int:
        d = np.abs(a - a[0]).max(axis=1)
        idx = np.where(d > 0.05)[0]
        return int(idx[0]) if len(idx) else -1

    fl, fr = first_move(left), first_move(right)
    if fl < 0 and fr < 0:
        raise ValueError(f"{demo.name}: neither arm moves")
    if fl >= 0 and fr >= 0:
        raise ValueError(f"{demo.name}: both arms move (left {fl}, right {fr})")
    return (fl, "left") if fl >= 0 else (fr, "right")


# ----------------------------------------------------------------------------- placements


def persistent_from(series: np.ndarray, valid: np.ndarray, start: int, end: int, thr: float) -> int:
    """First t in [start, end) with series >= thr at t, in >= 50 % of the next 8 valid frames and in
    >= 70 % of all valid frames of [t, end) (hands passing over the object are the missing 30 %)."""
    for t in range(start, end):
        if not valid[t] or series[t] < thr:
            continue
        near = valid[t : t + 8]
        if near.sum() and (series[t : t + 8][near] >= thr).mean() < 0.5:
            continue
        v = valid[t:end]
        if v.sum() and (series[t:end][v] >= thr).mean() >= 0.7:
            return t
    return -1


def blob_features(bgr_crop: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)[mask].astype(float)
    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)[mask].astype(float)
    return {"H": hsv[:, 0].mean(), "S": hsv[:, 1].mean(), "V": hsv[:, 2].mean(), "L": lab[:, 0].mean(),
            "a": lab[:, 1].mean(), "b": lab[:, 2].mean(), "area": float(mask.sum())}


def pick_object(feat: dict[str, float], candidates: list[str]) -> str:
    """Decide which of two known objects a blob is. Measured 2026-09-08 (blob means, top camera):
    box S~25 V~115 | tape H~7.5 S~83 V~160 | spoon H~12.7 S~102 V~170-200 | banana H~15 S~90 V~215-237."""
    a, b = candidates
    pair = {a, b}
    if pair == {"box", "tape"} or pair == {"box", "spoon"} or pair == {"box", "banana"}:
        other = (pair - {"box"}).pop()
        return "box" if feat["S"] < 60 else other
    if pair == {"banana", "spoon"}:
        return "banana" if feat["V"] >= 205 else "spoon"
    if pair == {"banana", "tape"}:
        return "banana" if feat["V"] >= 195 else "tape"
    if pair == {"spoon", "tape"}:
        return "tape" if feat["H"] < 10 or feat["a"] > 145 else "spoon"
    raise ValueError(candidates)


def detect_placements(top: np.ndarray, roi: tuple[int, int, int, int], objects: list[str], start: int, end: int,
                      flags: list[str], bin_id: int) -> dict[str, int]:
    """Placement frames for the objects of one bin from the content change inside its ROI."""
    x0, y0, x1, y1 = roi
    bg = ~white_mask(cv2.cvtColor(top[0][y0:y1, x0:x1], cv2.COLOR_BGR2HSV))
    bg = cv2.dilate(bg.astype(np.uint8), KER3).astype(bool)
    raw = []
    valid = np.zeros(end, dtype=bool)
    woodf = np.zeros(end)
    for t in range(end):
        hsv = cv2.cvtColor(top[t][y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        m, blocked, wf = content_mask(hsv, bg)
        raw.append(m)
        valid[t] = not blocked
        woodf[t] = wf
    masks = sliding_median(raw, valid, half=4)
    # reference = the resting content ~1 s before the lid: median over the unblocked frames of
    # [close-45, close-10] (a hand or the descending lid is the minority there; the last placement
    # precedes the lid by >= 28 frames in 90 % of the bins, the rest is flagged and done by hand)
    clean = [t for t in range(max(start, end - 45), max(start + 1, end - 10)) if valid[t]]
    if len(clean) < 5:
        flags.append(f"bin{bin_id}: no unblocked frames before the lid")
        return {o: -1 for o in objects}
    final = np.median(np.stack([raw[t] for t in clean]).astype(np.uint8), axis=0) >= 0.5
    final = cv2.morphologyEx(final.astype(np.uint8), cv2.MORPH_OPEN, KER3).astype(bool)
    final_area = int(final.sum())
    if final_area < 80:
        flags.append(f"bin{bin_id}: final content too small ({final_area} px)")
        return {o: -1 for o in objects}
    cov = np.array([(masks[t] & final).sum() / final_area for t in range(end)])
    cov_raw = np.array([(raw[t] & final).sum() / final_area for t in range(end)])

    def sharpen(p: int, level: float) -> int:
        """The median lags the arrival by a few frames: move back to the first raw frame of the run."""
        q = p
        while q - 1 >= start and valid[q - 1] and cov_raw[q - 1] >= level:
            q -= 1
        return q

    if len(objects) == 1:
        p = persistent_from(cov, valid, start, end, 0.6)
        if p < 0:
            flags.append(f"{objects[0]} placement not found in bin {bin_id}")
            return {objects[0]: -1}
        return {objects[0]: sharpen(p, 0.6)}
    # two objects: first step, plateau, second step relative to the plateau
    p1 = persistent_from(cov, valid, start, end, 0.1)
    if p1 < 0:
        flags.append(f"bin{bin_id}: first placement not found")
        return {o: -1 for o in objects}
    p1 = sharpen(p1, 0.1)
    plateau_idx = [t for t in range(p1 + 1, min(p1 + 8, end)) if valid[t]]
    c1 = float(np.median(cov_raw[plateau_idx])) if plateau_idx else float(cov_raw[p1])
    if c1 > 0.9:
        flags.append(f"bin{bin_id}: the two objects appear together at {p1} (plateau {c1:.2f}); order unknown")
        return {objects[0]: p1, objects[1]: p1}
    thr2 = c1 + 0.35 * (1.0 - c1)
    v_after = valid.copy()
    v_after[: p1 + 2] = False
    p2 = persistent_from(cov, v_after, p1 + 2, end, thr2)
    if p2 < 0:
        flags.append(f"bin{bin_id}: second placement not found (plateau {c1:.2f})")
        return {objects[0]: p1, objects[1]: -1}
    mid = [t for t in range(p1, p2) if valid[t] and 0.05 <= cov[t] <= thr2]
    if len(mid) < 3:
        flags.append(f"bin{bin_id}: the two objects appear together (p1 {p1}, p2 {p2}); order unknown")
        return {objects[0]: p1, objects[1]: p2}
    mid_mask = np.median(np.stack([masks[t] for t in mid]).astype(np.uint8), axis=0) >= 0.5
    mid_mask &= final
    if mid_mask.sum() < 40:
        flags.append(f"bin{bin_id}: first object blob too small to identify")
        return {objects[0]: p1, objects[1]: p2}
    ref = top[mid[len(mid) // 2]][y0:y1, x0:x1]
    first = pick_object(blob_features(ref, mid_mask), objects)
    second = objects[1] if first == objects[0] else objects[0]
    return {first: p1, second: sharpen(p2, thr2)}


# ----------------------------------------------------------------------------- per demo


def analyse(demo: pathlib.Path, entry: dict, overrides: dict) -> dict:
    top = read_video(demo / "top_camera_rgb.mp4")
    num_frames = len(top)
    meta_steps = int(json.loads((demo / "metadata.json").read_text())["num_steps"])
    if meta_steps != num_frames:
        raise ValueError(f"{demo.name}: metadata num_steps {meta_steps} != video frames {num_frames}")
    rs, arm = robot_start(demo)
    boxes = find_bins(top[0])
    rois = [roi_of(b) for b in boxes]
    flags: list[str] = []

    # lid states from the top camera
    state = np.zeros((num_frames, 3), dtype=int)
    soft = np.zeros((num_frames, 3), dtype=bool)
    dark = np.zeros((num_frames, 3), dtype=bool)
    for t in range(num_frames):
        hsv = cv2.cvtColor(top[t], cv2.COLOR_BGR2HSV)
        for k, (x0, y0, x1, y1) in enumerate(rois):
            hc = hsv[y0:y1, x0:x1]
            wh, wd = white_mask(hc).mean(), wood_mask(hc).mean()
            state[t, k] = OPEN if wh > 0.3 else CLOSED if wd > 0.6 else UNK
            soft[t, k] = wh < 0.05 and wd > 0.3
            dark[t, k] = (hc[..., 2] < 70).mean() > 0.4
    closes_top = [first_sustained_closed(state[:, k], rs, soft[:, k]) for k in range(3)]
    left = read_video(demo / "left_camera_rgb.mp4")
    right = read_video(demo / "right_camera_rgb.mp4")
    close1_wrist = wrist_close(left, (160, 120, 480, 360), rs)
    close3_wrist = wrist_close(right, (140, 110, 460, 350), rs)
    del left, right
    closes = list(closes_top)
    for k, w in ((0, close1_wrist), (2, close3_wrist)):
        top_c = closes[k]
        if w <= 0 or (top_c > 0 and top_c - w <= 12):
            continue
        if top_c < 0:
            flags.append(f"bin{k+1} close: top not found, wrist {w} -> using wrist")
            closes[k] = w
            continue
        between = state[w:top_c, k]
        blocked = (between != OPEN).all() and (((between == UNK) | dark[w:top_c, k]).mean() > 0.5)
        if blocked:
            flags.append(f"bin{k+1} close: top {top_c} vs wrist {w} (top view blocked) -> using wrist")
            closes[k] = w
    for k in range(3):
        ov = overrides.get("closes", {}).get(str(k + 1))
        if ov is not None:
            flags.append(f"bin{k+1} close overridden -> {ov}")
            closes[k] = int(ov)

    bins = {int(k): [TABLE_TO_LABEL[o] for o in v] for k, v in entry["bins"].items()}
    placements: dict[str, tuple[int, int]] = {}
    for k in (1, 2, 3):
        end = closes[k - 1] if closes[k - 1] > 0 else rs
        start = closes[k - 2] if k > 1 and closes[k - 2] > 0 else 0
        bin_flags: list[str] = []
        found = detect_placements(top, rois[k - 1], bins[k], start, end, bin_flags, k)
        n_over = 0
        for obj, p in found.items():
            ov = overrides.get("placements", {}).get(obj)
            if ov is not None:
                bin_flags.append(f"{obj} placement overridden {p} -> {ov}")
                p = int(ov)
                n_over += 1
            placements[obj] = (k, p)
        if n_over == len(bins[k]):
            # every placement of this bin was set by hand: the detector's doubts no longer apply
            bin_flags = [f for f in bin_flags if "overridden" in f]
        flags.extend(bin_flags)

    order = sorted(placements.items(), key=lambda kv: kv[1][1])
    ok = all(p >= 0 for _, (_, p) in order) and min(closes) > 0
    if ok:
        frames_p = [p for _, (_, p) in order]
        if frames_p != sorted(set(frames_p)):
            flags.append(f"placements not strictly increasing: {order}")
            ok = False
        for obj, (k, p) in order:
            lo = closes[k - 2] if k > 1 else 0
            if not (lo <= p < closes[k - 1]):
                flags.append(f"{obj} placement {p} outside bin {k} window [{lo}, {closes[k-1]})")
                ok = False
        if not (closes[0] < closes[1] < closes[2] < rs):
            flags.append(f"lid order broken: closes {closes}, robot {rs}")
            ok = False
        if any("order unknown" in f for f in flags):
            ok = False
    opened = int(entry["opened_bin"])
    if arm != ("left" if opened == 1 else "right"):
        flags.append(f"arm {arm} does not match opened bin {opened}")
        ok = False
    segments: list[dict] = []
    if ok:
        segments.append({"task": WATCH, "start": 0, "end": order[0][1][1] - 1})
        for i, (obj, (k, p)) in enumerate(order):
            nxt = order[i + 1][1][1] if i + 1 < len(order) else closes[2]
            segments.append({"task": f"{obj} in bin {k}", "start": p, "end": nxt - 1})
        segments.append({"task": ALL_CLOSED, "start": closes[2], "end": rs - 1})
        segments.append({"task": f"open bin {opened}", "start": rs, "end": num_frames - 1})
        cursor = 0
        for s in segments:
            if s["start"] != cursor or s["end"] < s["start"]:
                raise AssertionError(f"{demo.name}: segments do not tile at {cursor}: {segments}")
            cursor = s["end"] + 1
        assert cursor == num_frames
    return {
        "demo": demo.name,
        "num_frames": num_frames,
        "bins": {str(k): v for k, v in bins.items()},
        "opened_bin": opened,
        "revealed": [TABLE_TO_LABEL[o] for o in entry["revealed"]],
        "arm": arm,
        "robot_start": rs,
        "closes": closes,
        "closes_top": closes_top,
        "closes_wrist": {"1": close1_wrist, "3": close3_wrist},
        "placements": {o: {"bin": k, "frame": p} for o, (k, p) in placements.items()},
        "segments": segments,
        "ok": ok,
        "flags": flags,
        "_rois": rois,
        "_top": top,
    }


def audit_strip(res: dict) -> np.ndarray:
    """One row per demo: the three-bin strip at every boundary frame, sentence written on top."""
    top, rois = res["_top"], res["_rois"]
    ax0 = min(r[0] for r in rois) - 8
    ax1 = max(r[2] for r in rois) + 8
    ay0 = min(r[1] for r in rois) - 8
    ay1 = max(r[3] for r in rois) + 8
    # (frame, text, bin or None for the whole strip)
    if res["segments"]:
        bounds = []
        for s in res["segments"]:
            k = None
            for kk in (1, 2, 3):
                if s["task"].endswith(f"bin {kk}"):
                    k = kk
            bounds.append((s["start"], s["task"], k))
    else:
        bounds = [(v["frame"], f"{o}? b{v['bin']}", v["bin"]) for o, v in res["placements"].items() if v["frame"] >= 0]
        bounds += [(c, f"close{k+1}", None) for k, c in enumerate(res["closes"]) if c >= 0]
        bounds += [(res["robot_start"], "robot", None)]
        bounds.sort(key=lambda b: b[0])
    H = 2 * (ay1 - ay0)
    tiles = []
    for frame, text, k in bounds:
        t = min(max(frame, 0), len(top) - 1)
        if k is None:
            tile = cv2.resize(top[t][ay0:ay1, ax0:ax1], None, fx=1.0, fy=1.0)
            tile = cv2.resize(tile, (int(tile.shape[1] * H / tile.shape[0]), H))
        else:
            x0, y0, x1, y1 = rois[k - 1]
            tile = top[t][y0 - 8 : y1 + 8, x0 - 8 : x1 + 8]
            tile = cv2.resize(tile, (int(tile.shape[1] * H / tile.shape[0]), H))
        tile = tile.copy()
        cv2.putText(tile, text[:26], (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
        cv2.putText(tile, str(t), (3, tile.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        tiles.append(cv2.copyMakeBorder(tile, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(50, 50, 50)))
    strip = np.concatenate(tiles, axis=1)
    label = f"{res['demo']} {'OK' if res['ok'] else 'FLAGGED'}  {' | '.join(res['flags'])[:150]}"
    bar = np.zeros((16, strip.shape[1], 3), np.uint8)
    cv2.putText(bar, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255) if res["ok"] else (0, 0, 255), 1)
    return np.concatenate([bar, strip], axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--inspection", type=pathlib.Path, required=True, help="audited inspection table (json)")
    parser.add_argument("--overrides", type=pathlib.Path, default=None)
    parser.add_argument("--audit-dir", type=pathlib.Path, required=True)
    parser.add_argument("--demos", nargs="*", default=None)
    parser.add_argument("--write", action="store_true", help="write label files + labels manifest")
    args = parser.parse_args()

    table = {e["demo"]: e for e in json.loads(args.inspection.read_text())["episodes"]}
    overrides = json.loads(args.overrides.read_text()) if args.overrides and args.overrides.exists() else {}
    demos = args.demos or sorted(table, key=lambda s: int(s[4:]))
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    results = []
    strips = []
    for name in demos:
        demo = args.data_dir / name
        if not demo.is_dir():
            print(f"{name}: missing raw folder, skipped")
            continue
        res = analyse(demo, table[name], overrides.get(name, {}))
        strips.append(audit_strip(res))
        results.append(res)
        status = "OK " if res["ok"] else "FLAG"
        print(f"{status} {name} T={res['num_frames']} closes={res['closes']} robot={res['robot_start']} "
              f"placements={ {o: v['frame'] for o, v in res['placements'].items()} } {res['flags']}", flush=True)
        if args.write and res["ok"]:
            (demo / "subtask_labels_task1_base.json").write_text(json.dumps(res["segments"], indent=4) + "\n")
            for target in res["revealed"]:
                segs = json.loads(json.dumps(res["segments"]))
                for s in segs:
                    if s["task"] == ALL_CLOSED:
                        s["task"] = f"all bins closed, {target} is in bin {res['opened_bin']}"
                (demo / f"subtask_labels_task1_{target}.json").write_text(json.dumps(segs, indent=4) + "\n")
    if strips:
        w = max(s.shape[1] for s in strips)
        strips = [cv2.copyMakeBorder(s, 0, 4, 0, w - s.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)) for s in strips]
        for i in range(0, len(strips), 13):
            m = np.concatenate(strips[i : i + 13], axis=0)
            cv2.imwrite(str(args.audit_dir / f"labels_audit_{i // 13}.png"), m)
    n_ok = sum(r["ok"] for r in results)
    print(f"{n_ok}/{len(results)} demos labelled automatically; flagged: {[r['demo'] for r in results if not r['ok']]}")
    if args.write:
        manifest = {r["demo"]: {k: v for k, v in r.items() if not k.startswith("_")} for r in results}
        (args.data_dir / "subtask_labels_manifest_task1.json").write_text(json.dumps(manifest, indent=1) + "\n")
        vocab = sorted({s["task"] for r in results if r["ok"] for s in r["segments"]})
        print(f"labels manifest written; {len(vocab)} base sentences: {vocab}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
