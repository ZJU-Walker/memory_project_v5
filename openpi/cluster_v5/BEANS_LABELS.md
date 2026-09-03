# 0902_bean_scoop — subtask label definition (v1, for approval)

Data: `/iris/u/kewalk/memory_project/data/0902_bean_scoop/demo{1..60}` (30 Hz, 515–1517 frames, right arm scoops,
left arm idle). Per-frame signals logged by the collector: `led_on.npy` (green light on), `go_on.npy` (yellow go),
`cue_num_blinks.npy` (= x), `right_gripper_position.npy`, `right_joint_positions.npy`. `scoop_mark.npy` is empty in
every episode, so scoop events come from the arm: pickup/release = right gripper closes/opens (value < 0.5 = holding),
each delivery = right base joint j0 swings past 0.8 rad to the tray and back below 0.6.

| x (blinks = scoops) | episodes |
|---|---|
| 1 | 19 |
| 2 | 23 |
| 3 | 18 |

Event statistics (frames, 30 fps): blink 8–9 frames long, gaps 15–21; go at 78–164; scoop pickup 76–178 after go;
first delivery ~195 after pickup; one scoop cycle ~230 (delivery ~80 + return/dip ~145); release ~50 after the last
delivery; ~100 frames of return after release. Detected on all 60 episodes with no count mismatch.

## Sentences (one per frame; same file format as the bins task, `subtask_labels.json` = list of {task, start, end})

| # | frames | sentence |
|---|---|---|
| 1 | 0 … first blink onset − 1 | `wait for the light: no green blink yet` |
| 2 | onset of blink k … next onset − 1 (last one runs to go − 1) | `wait for the light: k green blink(s) so far` (k = 1..x) |
| 3 | go … gripper closes on the scoop − 1 | `yellow go: pick up the scoop, scoop x times` (`1 time` for x = 1) |
| 4 | pickup … end of delivery 1; then end of delivery k−1 + 1 … end of delivery k | `scoop k of x` (k = 1..x) |
| 5 | end of last delivery + 1 … last frame | `done: x of x scooped, put down the scoop and return` |

Rules: the count in (2) increments at blink ONSET (the frame the light turns on). (4) restates x at every scoop so the
target is re-read from memory each time; k is the model's own progress. (5) covers release and the return home.
Exact strings: singular "1 green blink so far", plural otherwise; digits, not words.

## Memory-relevant notes (not labels)
* Evidence = the blink steps; decision = the first step of sentence (3) (count must come from memory: the light is now
  yellow); progress steps = each transition in (4).
* A blink (8–9 frames) is shorter than the current memory stride (15 frames): a sampled frame lands inside a blink only
  ~55 % of the time, so this task needs stride ≤ 5 during the signal phase (or a frame stack per step) — a data-config
  change, not a label change.
* From the last blink to the last scoop is up to ~1300 frames; at 1 %/step decay the trace keeps 0.6 (stride 15) or
  0.2 (stride 5) — decay per task, or allow the model to refresh an unchanged count.
