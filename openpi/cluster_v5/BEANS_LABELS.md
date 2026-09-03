# 0902_bean_scoop — subtask label definition (v2, approved 2026-09-03 14:28)

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
| 4 | pickup … end of delivery 1; then end of delivery k−1 + 1 … end of delivery k | `scoop k` (k = 1..x) |
| 5 | end of last delivery + 1 … last frame | `done, put down the scoop and return` |

Rules: the count in (2) increments at blink ONSET (the frame the light turns on). (5) covers release and the return
home. Exact strings: singular "1 green blink so far", plural otherwise; digits, not words.

**x is stated ONCE (v2, user decision 2026-09-03 14:28).** After the blink phase the target count appears only in
sentence (3). The scoop sentences carry progress alone, so at the end of scoop k the model must decide between
`scoop k+1` and `done` from memory: the go sentence is ~20–30 memory steps back and the blink count further still.
v1 wrote `scoop k of x`, which put the answer in the current sentence — a model fed only its previous sentence would
then solve the task with no memory at all. Baselines to run against the bank on these labels: (a) images only, no
memory; (b) previous sentence fed back as text; (c) full transcript of previous sentences. (c) is the fair one to
beat — the bank is a fixed-size compression of it.

## Memory-relevant notes (not labels)
* Evidence = the blink steps; decision = the first step of sentence (3) (count must come from memory: the light is now
  yellow); progress steps = each transition in (4).
* **Stride vs. blinks** (measured over all 60 episodes; blink length 6–9 frames, median 9):

  | stride | blinks a sampled frame lands on | episodes where all blinks are seen |
  |---|---|---|
  | 3 | 100 % | 60 / 60 |
  | 5 | 100 % | 60 / 60 |
  | 8 | 97 % | 57 / 60 |
  | 15 (current) | 52 % | 21 / 60 |

  So the memory stride must be ≤ 5 for this task (or each step must see a short frame stack). Data-config change only.
* **Decay vs. the span that must be remembered.** From the go sentence to the "done" decision is 307–1166 frames
  (median 626). With the current 1 %/step decay:

  | stride | memory steps (median / worst) | trace kept (median / worst) |
  |---|---|---|
  | 3 | 209 / 389 | 0.12 / 0.02 |
  | 5 | 125 / 233 | 0.28 / 0.10 |
  | 15 | 42 / 78 | 0.66 / 0.46 |

  Stride 5 is the compromise: every blink is seen and 0.10 of the trace survives the worst case. Either lower
  `alpha_step` for this task, or let the model refresh an unchanged count periodically (a re-write of the same
  sentence costs almost nothing under the delta rule).

## Status
Labels **written 2026-09-03 14:31**: `subtask_labels.json` in each of the 60 demo folders +
`/iris/u/kewalk/memory_project/data/0902_bean_scoop/subtask_labels_manifest.json` (per episode: num_frames, x,
segments). Validation: segments tile [0, n-1] with no gaps in all 60; scoop-sentence count == blink count == x in all
60; disk files match the manifest. Vocabulary is 11 distinct sentences (3 blink counts + "no green blink yet",
3 go sentences, 3 scoop-progress sentences, 1 done).

