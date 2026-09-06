# Bean-scoop task with the v5 sentence memory — status, structure, problems, results (2026-09-05 18:20)

Repo: `/iris/u/kewalk/memory_project_v5` (branch `v5`, GitHub `ZJU-Walker/memory_project_v5` main). Running log: `openpi/cluster_v5/README.md` §8.
Checkpoints: `/iris/u/kewalk/memory_project_v5/v5/checkpoints/<config>/<exp>/`. Rollout videos and probes: `/iris/u/kewalk/memory_project_v5/v5/diagnostics/`.

## 1. The task

60 teleop demos (`/iris/u/kewalk/memory_project/data/0902_bean_scoop`, 30 fps, 900 frames ≈ 30 s each). A green LED blinks
x ∈ {1,2,3} times (each blink ~9 frames), then a yellow "go" light; the robot picks up the scoop and transfers x scoops
of beans from the bowl to the tray, then puts the scoop down. Prompt (constant): "scoop the beans into the tray as many
times as the green light blinked". Splits (manifest `beans_episode_manifest_v1.json`): train 48 (x=1/2/3: 15/19/14),
development 6 (demo11, 12, 14, 17, 21, 51), final_test 6. The policy runs at stride 5 frames = one "step" every 1/6 s.

The task needs two things from memory: (a) count the blinks while waiting, then announce the count at "go";
(b) remember that count for 20–30 s while scooping and stop after x scoops.

## 2. What the model is (structure)

π0.5 (PaliGemma 3B VLM + flow action expert) plus a **sentence-fed fast-weight semantic bank** (Titans-style):

* **Sentence stream.** At every step the model decodes one short "subtask sentence" (teacher-forced CE in training,
  greedy decode at rollout) from a fixed vocabulary (16 sentences in the current v6 labels, see §3).
* **Writing.** A sentence is written to the bank when it is *new* (differs from the last committed one) and *confident*
  (mean token probability ≥ 0.9). The sentence is encoded memory-blind (frozen Gemma layer-8 token states, standardized
  against the vocabulary, attention-pooled) into a unit key (512-d) and a unit value (2048-d); the bank is one
  fast-weight matrix updated by the **delta rule** (each new note becomes exact at write time; notes with overlapping
  keys are overwritten, which is how "light off: 2" replaces "light off: 1"). Write delay: 0 steps (A5/A6 recipe) or
  1 step (A4/B4 recipe). "Retry-until-committed" (B stages since 09-05): the change detector compares with the last
  *committed* sentence, so an under-confident sentence is retried on the next step instead of being dropped.
* **Decay.** The whole matrix is multiplied by (1 − alpha_step) every step, write or no write. `alpha_step = 0.01` is a
  fixed constant (half-life 69 steps ≈ 11.5 s). The training prefill replays the true history with the true gaps, so
  training and rollout see the same decay. (This constant is the subject of the current experiment, §5.)
* **Reading.** 8 learned read queries (conditioned on the instruction context, standardized) retrieve from the bank;
  the retrieval is RMS-normalized and injected into the VLM's layer-8 stream, so both the sentence decoder and the
  action expert see it. The visual bank of the v4 design is present but its injection is off.
* **Training recipe (two stages, always warm-started from the pour-task model `stageB6a keep_499`).**
  Stage A: the bank is fed the *label* sentences (oracle writes) with the history prefilled — the model learns to read.
  Stage B: A's weights, the bank is fed the model's *own* sentences (teacher-forced argmax, same write rule), half the
  learning rate — the model learns to live with its own notes. 500 updates each (300 for the current slow-decay pair),
  lr 5e-5 / 2.5e-5, batch 8 (4×H100) or 4 (2×H100). Loss = flow matching + sentence CE (no auxiliary memory losses).
* **Rollout / robot.** `scripts/v5_heldout_video.py` renders self-write rollouts (the model's own sentences drive the
  bank) on the dev episodes; `scripts/serve_yam_memory.py` + `examples/yam/client_memory_v5.py` (beans prompt,
  `--steps-between-inference 5`) serve the same loop on the robot.

## 3. Label versions (the sentences the model writes)

| version | sentences | change | result |
|---|---|---|---|
| v1 | wait: k blinks so far / go: scoop x times / scoop k / done (11) | first cut | count double-counted (a blink spans 2 steps) |
| v2 "light state" | light on: k / light off: k (14) | the sentence carries the LED state | count 4/6 → fixed the double count |
| v4 "tray cut" | scoop k+1 from tray arrival k | | count 5/6, scoops held in 2/4 |
| v5 "visible LED" | light boundaries on the first camera-visible frame | LED signal leads the camera 0–2 frames | count 5/6 (delay 1) / 6/6 (delay 0) |
| **v6 "sub-phase"** (current) | scoop k: dig and carry (bowl arrival k) / scoop k: dump and return (tray arrival k, k<x) / done (tray arrival x); 16 sentences | every (previous sentence, arm position) pair has one target | count 5–6/6; tray decision unreliable (§4.2) |
| v7 "target-carry" (prepared, NOT trained) | scoop k of x: dig and carry … (20 sentences) | carries x in every scoop note | set aside: refreshes the fact instead of fixing recall |

Design rule that emerged: the memory reliably supports "next sentence = f(newest note, current image)". Boundaries must
sit on sharp, persistent, non-recurring visual events, and the previous note must disambiguate same-vs-next event.

Label tooling: `scripts/beans_relabel_*.py`, sidecars `cluster_v5/beans/beans_v5_subtask_labels_<version>.json`, inspection
server `examples/yam/label_subtasks.py --beans-task --beans-v6 --label-file subtask_labels_v6sub.json --port 8765`
(iris-ws-18; `ssh -L 8765:localhost:8765 kewalk@iris-ws-18.stanford.edu`).

## 4. Problems faced

### 4.1 Blink counting (solved)
Double counting (v1) → light-state sentences (v2). Single-step blinks (demo17, LED on for one step): lost with a 1-step
write delay, kept with delay 0 in the A stage, but lost again after any B stage; irrelevant on the robot at 5-step
inference spacing. Camera lag → visible-LED boundaries (v5).

### 4.2 The scoop count (open): the "dump or done" decision at the tray
With v6 labels the scoop cycle is tracked correctly (dig → dump → next dig) but the decision "dump and return" (k<x) vs
"done" (k=x) at the tray is unreliable: A6 says "done" one scoop early in all four multi-scoop dev episodes; B6 gets
4 of 7 such decisions right (errors both ways, confidence 0.96–0.99). Measured cause (tray probe, §6.3): **the model
does not use the memory for this decision** — with an empty bank it is exactly as accurate as with the true history,
and flipping the remembered target does not change its answer. It decides from the visible tray (k−1 dumps already in
it), which predicts 69 % of training decisions. Why it never learned to read the target: the go note is 60–175 steps
old at the tray (strength 0.17–0.56 under the 1 %/step decay) and buried under fresh scoop notes that share most of
its words; every decision that works reads a note ≤ 14 steps old (strength ≥ 0.87). Making the note fresh at inference
alone does not help (the read was never learned) → retrain with slow decay (§5).

### 4.3 The B5 collapse (dead end, understood enough)
B5 = A5 (visible-LED labels, delay 0) + own-writes fine-tune: in rollouts it answers "no blink yet" at the first "off"
step with [no blink, light on: 1] in the bank, writes it, and every blink restarts the count. B5d1 (same with delay 1)
reproduces it exactly, so the delay is not the cause; A5's weights are the only starting point that collapses — B4 (same
labels, delay 1) and B6 (v6 labels, delay 0) are healthy. Not explained further; the B6 line is the base now.

### 4.4 Side findings
* The write gate's "confidence" is the mean over ~9 tokens; 8 of them are near-certain once the first is chosen, so a
  coin-flip first token still passes at 0.96. The gate cannot catch the tray error.
* Every B stage weakens bank reliance (count-flip first-go: A4 0.985 → B4 0.79; A3 1.00 → B3 0.73).
* Ops: never edit a running bash runner; pgrep patterns with the bracket trick; waiters must key on the copy-complete
  line, not on the folder (two evals started on half-copied checkpoints); placeholder trainings never keep checkpoints.

## 5. Running right now (18:20)

| where | what | why | ETA |
|---|---|---|---|
| 4×H100 job 17249058 | **A6sd → B6sd**: A6/B6 recipe with `alpha_step 0.001` (half-life ~115 s), 300 updates each | does a strong (undecayed) go note during training make the tray decision read the target? | A6sd ckpt 299 ≈ 18:35, B6sd ≈ 20:10, videos after each |
| H200 job 17267793 | **tray-decision probe** `scripts/v5_tray_flip_eval.py`: A6/B6 weights × train/dev windows × decay 0.01/none, true / flipped / empty history | measure whether the decision uses the remembered target | train split done (§6.3); dev split ≈ 18:45 |
| 2×H100 job 17267129 | B6 continuation toward 3000 updates (step ~920, saves every 250) | old keep-training rule | 1000 ≈ 18:35; 3000 ≈ 04:00 |

## 6. Results

### 6.1 Self-write rollout videos (dev set; `v5/diagnostics/videos_<exp>_keep_499/ep<idx>_self.mp4`; ep10=demo11 x2, ep11=demo12 x1, ep13=demo14 x3, ep16=demo17 x3, ep20=demo21 x2, ep50=demo51 x1)

| stage (labels, delay) | exp | count right | scoops |
|---|---|---|---|
| A r1 (v1, 1) | v5_beansA_20260904_r1 | 3/6 | — |
| A2 (v2 light, 1) | v5_beansA2_20260904_r1 | 4/6 | — |
| A3 (v4 tray, 1) | v5_beansA3_20260904_r1 | 5/6 | not held |
| B3 (own writes) | v5_beansB3_20260904_r1 | 5/6 | held in 2/4 (demo21 116/116); retry rule fixed demo11 |
| A4 (v5 visible LED, 1) | v5_beansA4_20260905_r1 | 5/6 | like A3 |
| B4 | v5_beansB4_20260905_r1 | 5/6 (demo17) | demo11 ✓, demo14 ✓ (1→2→3→done), demo17 ✓ w/ flicker, demo21 stuck at 1 |
| A5 (v5, delay 0) | v5_beansA5_20260905_r1 | **6/6** | not held |
| B5 | v5_beansB5_20260905_r1 | collapse (1 time everywhere) | — |
| B5d1 (B5 with delay 1) | v5_beansB5d1_20260905_r1 | collapse (identical) | — |
| A6 (v6 sub-phase, 0) | v5_beansA6_20260905_r1 | **6/6** | "done" one scoop early in all 4 multi-scoop episodes |
| **B6** | v5_beansB6_20260905_r1 | 5/6 (demo17) | demo11 115/119 fully correct; demo14 dump1 ✓ then done early; demo21 done early; demo12 spurious dump; demo51 ✓ |

B6 tray decisions with a correct bank: 4/7 (demo11 k1 ✓ k2 ✓, demo12 ✗, demo14 k1 ✓ k2 ✗, demo21 ✗, demo51 ✓).

### 6.2 Count-flip battery (first go step, history only: does the go count follow the bank?)

| stage | count from bank | follows a flipped history | keeps true under flip |
|---|---|---|---|
| A4 | 0.985 | 0.985 | 0.000 |
| B4 | 0.788 | 0.758 | 0.045 |
| A5 | 1.000 | 0.924 | 0.045 |
| B5 | 0.833 | 0.758 | 0.106 |
| A6 | 1.000 | 0.955 | 0.000 |

### 6.3 Tray-decision probe (train-split windows, 18 tray decisions, "dump" vs "done" by teacher-forced CE)

| weights | history | decay 0.01 | no decay |
|---|---|---|---|
| A6 | true | 72 % | 78 % |
| A6 | empty bank | 72 % | 72 % |
| A6 | target flipped | unchanged 72 % | unchanged 78 % |
| B6 | true | 89 % | 94 % |
| B6 | empty bank | 94 % | 94 % |
| B6 | target flipped | unchanged 89 % | unchanged 94 % |

Reading: neither model uses the memory at the tray; B6's 89–94 % on training windows comes from visual cues it has
seen, which is why it drops to 4/7 on dev. Reports: `v5/diagnostics/tray_flip_<A6|B6>_keep_499_<split>/tray_flip_eval.json`.

### 6.4 Ages and strengths of the notes each decision reads (B6, dev; strength = 0.99^age)

| decision | note read | age (steps) | strength |
|---|---|---|---|
| blink count, go count | newest light note | 10–14 | 0.87–0.90 |
| tray, scoop 1 | go note | 58–83 | 0.43–0.56 |
| tray, scoop 2 | go note | 121–126 | 0.28–0.30 |
| tray, scoop 3 | go note | 175 | 0.17 |

For a longer task: 1 min → 0.03, 2 min → 0.0007, 3 min → 0.00002 of a note's strength under alpha 0.01.

## 7. Judged checkpoints

* B6 (current best for the robot): `/iris/u/kewalk/memory_project_v5/v5/checkpoints/pi05_yam_mem_v5_beansB6/v5_beansB6_20260905_r1/keep_499` (config `pi05_yam_mem_v5_beansB6`); continuation saves `750`, `1000` (≈18:35), … in the same folder.
* B4 (visible-LED labels, delay 1): `.../pi05_yam_mem_v5_beansB4/v5_beansB4_20260905_r1/keep_499`.
* A6sd/B6sd (slow decay): `.../pi05_yam_mem_v5_beansA6sd/v5_beansA6sd_20260905_r1/keep_299`, `.../pi05_yam_mem_v5_beansB6sd/.../keep_299` (pending).

## 8. What decides the next step

* If A6sd/B6sd make the tray decision follow the target (probe + rollouts): the fix is the decay constant; deploy B6sd and
  lower alpha further for long tasks.
* If not: the target is recoverable from the bank but the read is not learned → oversample tray decisions / add a recall
  signal; or the go note is not recoverable at all (key overlap with the scoop notes) → change the go sentence so it
  shares no words with the scoop notes (a bank-level key-overlap measurement is prepared for this).
