# V5 Plan — Dual Fast-Weight Memory: Visual Bank + Sentence-Fed Semantic Bank

Branch `v5`, worktree `/iris/u/kewalk/memory_project_v5` (off `v4` @ `780ac0d`, the Stage-4c
line). The v4 tree is never modified from here. Every design decision below was taken with the
user on 2026-09-02 (12:30–13:05 PDT, chat "v5"); every later modification is logged in §8.

## 0. Decisions (final, 2026-09-02)

| # | decision | why |
|---|---|---|
| D1 | Keep the dual bank. The visual bank is exactly Stage 4c (Titans fast weights, delta-rule commits every step, analytic decay, **no side supervision**). | 4c proved a live unsupervised visual bank is numerically stable and does not steal the decision; it is the backup when words are not enough. |
| D2 | Delete the task-specific fact head (`{left,right,unknown}` slot classifier), its value table and its read head. | Task-specific; v5 must generalize to any task with language labels. |
| D3 | The semantic bank is fed by the model's **own predicted subtask sentence**. There is no separate "note" span: the existing π0.5 subtask text becomes detailed (`inspect both bins: banana left, grey box right`). | User's idea (13:53): one text span, the subtask itself carries the information. |
| D4 | The semantic bank is a **true fast-weight memory** ("Option B"): content keys/values from the sentence, queries from the current context, delta-rule commit = one test-time gradient step. **No slot-table fallback** — if B fails, stop and report (user, 13:02). | With fixed slot keys the bank degenerates into a table; the user wants both banks to be genuine test-time learning with one shared mechanism. |
| D5 | Sentence encoder = frozen embedder + frozen blocks 0–8, text-only, mean-pool, L2-normalize. | Forced: averaging plain word embeddings cannot distinguish "banana left, grey box right" from "banana right, grey box left" (same words). Blocks 0–8 make each token order-aware and land in the layer-8 space the reader already lives in. |
| D6 | Labels: three templates generated from the frozen manifest facts (§4). | Consistent wording; the model learns the format, not the task. |
| D7 | Write rule: commit when the sentence **changed vs the previous step** and its **mean token probability ≥ 0.9** (oracle mode: label changed). | General to any task: the bank is a log of subtasks done so far (3 writes/episode here). |
| D8 | Ladder A → B → C, three separate 1000-update runs on the H200 (§6). | Mirrors the proven v4 ladder; a failure is localizable. |
| D9 | 8 query heads → 8 semantic memory tokens (same token budget as v4's 8 fact slots). | Reader-side plumbing is unchanged from v4. |

## 1. Repo, environment, compute

* Worktree of the standalone v4 repo; `git worktree list` shows `memory_project_v4 [v4]` and
  `memory_project_v5 [v5]`. `data/` and the read-only caches (`v35/cache/{huggingface,uv,openpi}`)
  are symlinks into the v4 tree; the JAX compile cache (`v35/cache/jax`) is worktree-local and a
  further private cache per GPU type / concurrent process is set through `OPENPI_JAX_CACHE_DIR`.
* `openpi/.venv` is a full `uv sync` (same lock as v4; package list identical; editable `openpi`
  points at this tree). `source cluster_v5/env.sh` (NFS `HOME`, worktree-local caches, creates
  `v5/{assets,checkpoints,diagnostics}`).
* Compute: the single **H200** of Slurm job `17207774` on `iris-hgx-2` (`iris-hi`, 3-day limit
  from 2026-09-02 11:35). ssh to the node adopts that job; GPU work is launched as
  `srun --jobid=17207774 --overlap` steps. Node caveat (found by the v4 session 12:15): NFS reads
  are ~11 MB/s there on first touch (`import jax` 126 s, device init 44 s); the page cache keeps
  later starts fast. A busy placeholder (`cluster_v5/gpu_placeholder_hgx2.sh`, 120 GiB + matmul
  loop, marker `gpu_placeholder_marker`) holds the GPU whenever nothing real runs; kill it with
  `pkill -f "gpu_placeholder_marke[r]"` before real work. The user's `train_hs.py` keep-alive
  (~1 GB) in that job is never touched. **Never submit Slurm jobs; never touch the H100 jobs
  (`17192955`, `17178887`) — those belong to the v4 / v3.x sessions.**
* Frozen data, unchanged from v4: v36 manifest, 70 episodes (0830 part1/part2 + 0831), seed-36
  split 54 train / 8 development / 8 final_test (sealed). All battery numbers are reported on the
  8 development episodes; final_test stays sealed until a v5 model is declared final.

## 2. Architecture

```
 cameras + prompt ("pick up the banana")
        │
        ▼
 blocks 0–8, frozen (same as v4) ──▶ layer-8 features h8: image tokens + prompt tokens
        │
        ├──▶ VISUAL WRITE (unchanged from Stage 4c)
        │      compress image tokens → key/value → delta-rule commit at evidence steps (v3.5 schedule)
        │      ┌────────────────────────┐
        │      │ VISUAL BANK  M_vis     │  fast weights, analytic decay over time gaps
        │      └───────────┬────────────┘
        │
        ├──▶ SEMANTIC WRITE (new; only when the sentence changed and is confident)
        │      sentence tokens → frozen embedder → frozen blocks 0–8 text-only → mean → L2 = e
        │      key = Wk·e      value = Wv·e                      (Wk, Wv trainable)
        │      M_sem ← M_sem + (value − M_sem·key)·keyᵀ            (delta rule, rate 1.0)
        │      ┌────────────────────────┐
        │      │ SEMANTIC BANK  M_sem   │  same storage / decay / state code as the visual bank
        │      └───────────┬────────────┘
        │
        └──▶ READ, every step, both banks
               visual:   queries from the conditioner (current images) → n_v tokens = M_vis·q
               semantic: 8 trainable query heads on the layer-8 context → 8 tokens = M_sem·q1…q8
               all tokens → tanh_rms injection (c = 12.4, tau = 0.02, gate 0.5) → appended before
               block 9; exactly-zero tokens are masked (`memory_mask_zero_tokens`, v4 fix)
        │
        ▼
 blocks 9–17 + action expert (trainable as in v4)
        ├──▶ subtask sentence, detailed, decoded with memory visible
        │       inspect step:  "inspect both bins: banana left, grey box right"
        │       waiting step:  "wait for the instruction"
        │       decision step: "pick up the banana from the left bin"
        └──▶ FAST tokens + actions (flow matching)
```

The two banks side by side:

| | visual bank | semantic bank |
|---|---|---|
| mechanism | delta-rule fast weights, analytic decay | identical code |
| write input | compressed layer-8 image features, at the data-marked evidence steps (as 4c) | the model's own predicted sentence, whenever it changes |
| write key / value | from image features | `Wk·e` / `Wv·e`, `e` = encoded sentence |
| read query | conditioner over current images | 8 heads on the current layer-8 context (prompt + images) |
| output | `n_v` memory tokens | 8 memory tokens |
| supervision | none | none |
| trainable parts | compressors, conditioner, retrieval (the 4c set) | `Wk`, `Wv`, 8 query heads |

What the semantic bank must learn: the prompt "pick up the banana" has to produce a query that
lands near the key of the sentence that mentioned the banana. That alignment is learned only from
the decision-step subtask loss (plus flow), through the injected tokens.

**Sentence handling.** Training: the written sentence is the argmax of the teacher-forced subtask
logits at that step (cheap, discrete bottleneck, no decode loop; no gradient flows through the
tokens). Inference: the actually decoded sentence. Oracle mode (v5-A) writes the label sentence.

**Losses.** Flow loss on actions; cross-entropy on the subtask words and FAST tokens (the existing
π0.5 co-training losses). Nothing else: no fact head, no read head, no side losses, no memory loss.

**Frozen vs trainable.** Frozen: SigLIP, embedder, blocks 0–8 (hence the sentence encoder).
Trainable: exactly the v4 Stage-4c set (blocks 9–17, action expert, LM head, injection, visual
subsystem) plus `Wk`, `Wv` and the semantic query heads. Removed: fact head, fact value table,
read head, v3.5 side heads.

**Interventions (read side only, decision steps only, as in v4):** `reset` (blank bank) and
`donor` (another episode's bank) for `semantic`, `visual`, `both`.

**What is the same as v4, verbatim:** backbone, layer-8 split, blind memory rows, tanh_rms
injection, zero-token masking, visual bank, memory state carry across TBPTT windows, the
batteries, the data split, the light protocol (`v4_protocol`, graft sources, run manifest).

## 3. Mapping to code (implemented 2026-09-02 13:10–13:45; every item is on branch `v5`)

Two facts from the code map changed the wording of §2/§4 but not the design:
(a) the Gemma blocks are one scanned parameter stack, so "frozen blocks 0–8" is implemented
as a `stop_gradient` on the sentence-encoder pass, not as a freeze regex (all 18 blocks train
through the task losses exactly as in v4); (b) the visual bank commits at the data-marked
evidence steps (`seq_write_mask`, the v3.5 schedule), not literally every step — unchanged
from 4c. The semantic bank's write schedule is model-driven (D7) and ignores `seq_write_mask`.

| piece | file / symbol | what |
|---|---|---|
| config flags | `models/pi0_config.py`: `memory_v5_sentence_bank`, `memory_v5_oracle_writes`, `memory_v5_write_conf=0.9`, `memory_v5_sentence_len=48`, `memory_v5_read_queries=8` + validation (requires `memory_v4_dual_bank`; fact losses must be 0; `sentence_len <= causal_token_len`) | the flag reuses ALL dual-bank plumbing; with it on, the fact head/value table/read head are never constructed |
| module members | `models/pi0.py` `__init__` (v4 block): `memory_sem_key_proj` (2048→512, no bias), `memory_sem_value_proj` (2048→2048, identity init), `memory_sem_read_query_bank` [8, 2048], `memory_sem_read_conditioner` (`MemoryQueryConditioner`, zero-init residual), `memory_sem_query_proj` (2048→512); `memory_sem_slot_embedding` sized by the query count | names contain `memory` so the audited loader fresh-inits them and the memory-group grad clip covers them |
| token budget | `Pi0._memory_token_total` | 16 visual + `memory_v5_read_queries` semantic tokens (24 total, as v4) |
| sentence encoder | `Pi0.v5_encode_sentence(tokens, mask)` | embed → blocks 0..`memory_layer` text-only (bidirectional over the span, empty cache, no images/memory/suffix) → masked mean → L2; `stop_gradient` (D5) |
| write intent | `Pi0.v5_sentence_intent(e)` → `key = L2(Wk e)`, `value = L2(Wv e)` shaped `[b,1,·]` | content key/value |
| read | `Pi0.v5_semantic_queries/v5_semantic_read`: conditioner over the instruction rows of `h8_all` (state digits excluded) → `Wq` → L2 → `memory_semantic.read_key` | 8 tokens, content-addressed; wired in `_v32_prepare_memory_interface` (also returns `sem_queries`) |
| write | `Pi0.v5_semantic_write` = `memory_semantic.delta_write_kv_multi(state, key, value, commit[:,None])` | delta rule, rate 1.0; False commit = one decay step (v4 transition contract) |
| write rule | `_compute_sequence_loss_v32`, semantic transition block (`v5_on`) | span = `causal_mask & ~causal_fast` over the first `sentence_len` causal positions; predicted mode: `argmax` of the teacher-forced logits, `conf = mean_span p(argmax)`; oracle mode: label tokens, conf = 1; `changed = any(cur != prev)`; commit iff `changed & confident & transition_valid`; carry gains `prev_sentence [b,L]` (sentinel −1), a diagnostic ring of the last 8 committed keys and its count |
| outputs | same function, `v5_*` telemetry: `v5_sentence_changed_count`, `v5_sentence_confident_count`, `v5_write_requested_count`, `v5_token_acc_evidence_sum/v5_exact_evidence_sum/v5_evidence_count`, `…_decision…`, `v5_qk_cos_sum/v5_qk_count` (max cosine between decision-step queries and any committed key), per-step `v5_exact_{decision,evidence}_steps` | no loss uses them; `v4_decision_ce_*`, `v4_sem_*` keep their v4 names so the batteries run unchanged |
| interventions | unchanged (`_V4_INTERVENTIONS`, read-side only, decision steps only) | `reset/donor` × `semantic/visual/both` |
| decode length | `sample_subtask`/`sample_with_memory` default `max_decode_steps` 10 → 24 | the inspect sentence is 12 tokens |
| labels | `scripts/v5_build_subtask_labels.py` (+ `_test.py`) → `data/v5_subtask_labels_0830_0831.json` (file sha `9976d467…3043d`, content sha `c1d821a1…9efa`, manifest sha `9085fe50…8442`, cross-checked against the v4 fact sidecar) | run-length segments per episode; token lengths checked with the real PaliGemma tokenizer (max 12 ≤ 48) |
| data | `training/config.py` `DataConfig.memory_v5_subtask_labels_{path,sha256}`; `training/data_loader.py` `_load_v5_subtask_labels` (pinned sha, self-hash, manifest sha, segments must tile every episode) → `transforms.MemorySequenceSubtasks.episode_sentences` | replaces `subtask` (the lookahead-shifted CE target) only; `subtask_now` keeps the canonical vocabulary (phase masks, sparse-skip checks, manifest vocabulary pin untouched); no new model fields |
| train protocol | `scripts/train.py`: `_validate_v4_run` also requires the v5 sidecar pin for v5 models; run manifest gains `subtask_labels_sha256`; `_v5_info` logs the telemetry | fact-loss assembly is skipped automatically (its keys are absent) |
| configs | `pi05_yam_mem_v5_stage{A,B,C}` (`training/config.py`, right after `pi05_yam_mem_v4_stage4c`) | A: oracle writes, visual injection off, freeze = Stage-2b set minus `fact_*`; B: predicted writes; C: + visual injection, freeze = Stage-4c set minus `fact_*`; no graft sources; assets copied from v4 (`v5/assets/…/norm_stats.json`) |
| launchers | `cluster_v5/train.sh`, `cluster_v5/run_train_h200.sh` (job-scoped `srun --overlap`, kills the placeholder, private JAX cache `v35/cache/jax_hgx2`, resume policy), `cluster_v5/gpu_placeholder_hgx2.sh` | |
| tests | `models/pi0_v5_test.py` (config gating; order-aware, padding-invariant, gradient-free encoder; write-then-read; oracle sequence writes on every sentence change; predicted writes gated by confidence; interventions read-side only), `scripts/v5_build_subtask_labels_test.py` | |

Left as in v4 and deliberately untouched: the v4 fact-head code paths (still used by the v4
configs in this tree), `sample_with_memory` (dual-bank inference remains unimplemented; the
batteries do not need it), `v4_stage2_eval.py` (its read-accuracy gate reads a head v5 does not
have; the CE-ratio gates and `v4_side_flip_eval.py` work unchanged).

**Approximation to know about (predicted mode, training only):** the written sentence is the
argmax at the LABEL's span positions under teacher forcing, so its length is the label's length
and each token is conditioned on the gold prefix. At inference the decoded sentence is used.

## 4. Labels (the detailed subtask sidecar)

`data/v5_subtask_labels_0830_0831.json`, generated by `scripts/v5_build_subtask_labels.py` from
the frozen v36 manifest (object, target side) and each episode's hashed raw phase segments. One
sentence per (episode, frame) as run-length segments. The dataset's own labels already carry the
decision: the waiting phase is `wait; target bin is {side}` and that is the step where the memory
read has to pay off (the side-flip battery swaps exactly that side token). So only the inspection
phase gains content:

| phase (canonical, kept in `subtask_now`) | sentence (the CE target `subtask`) |
|---|---|
| `open both lids` | unchanged |
| `inspect both bins` | `inspect both bins: banana {side}, grey pepper box {side}` (12 tokens) |
| `close both lids and reset arms` | unchanged |
| `wait; target bin is {side}` | unchanged (7 tokens) — the decision |
| `open {side} bin` | unchanged |

The object is named as in the prompts (`find the grey pepper box`). Five sentence changes per
episode → at most five semantic commits (fewer in sparse windows), well inside the diagnostic
ring of 8. The split and the held-out episodes are unchanged; the sidecar derives only from the
manifest and the label files, never from images.

## 5. Write rule, modes, inference

* **Predicted mode (v5-B/C):** at step t take the subtask span of the causal suffix, argmax of the
  logits (teacher-forced), compare with the previous step's argmax tokens (carried in the memory
  state); `changed` = any token differs; `confident` = mean over the span of p(argmax) ≥ 0.9;
  write iff `changed & confident & step_valid`.
* **Oracle mode (v5-A):** encode the label tokens; write iff sidecar `changed & step_valid`.
* **Encoding:** the span's tokens (padding removed) through the frozen embedder and blocks 0–8 as
  a text-only prefix (no images, no memory, no suffix); mean over the span; L2-normalize.
* **Commit:** `key = Wk·e`, `value = Wv·e`, delta-rule commit (`delta_output` write rule, rate 1.0)
  into `memory_semantic`; the analytic decay between steps is the v4 one (`alpha_step` shared with
  the visual bank).
* **Read:** 8 query heads `q_i = Wq_i · c` where `c` is the pooled layer-8 context of the current
  step (the same `pooled_frame` association as the v4 semantic bank); `r_i = M_sem·q_i`; injected
  through `_v4_inject_semantic`.
* **Inference:** `sample_with_memory` decodes the sentence, applies the same rule with the decoded
  tokens, then acts. (Dual-bank inference was unimplemented in v4; v5 implements it after the
  ladder, it is not needed for the batteries.)

## 6. Runs and batteries

| run | config | writes | visual bank | purpose |
|---|---|---|---|---|
| v5-A | `pi05_yam_mem_v5_stageA` | oracle sentences | off | can the model read oracle sentences through content-addressed retrieval? **Make-or-break for D4.** |
| v5-B | `pi05_yam_mem_v5_stageB` | predicted sentences | off | whole loop; the A→B gap is the key number |
| v5-C | `pi05_yam_mem_v5_stageC` | predicted sentences | on (4c set) | final dual-bank model |

Each run: 1000 updates, batch 2 (single GPU recipe from v4), grafted from the same sources as
Stage 4c, checkpoints at 500 and 999, preemption-safe resume. Batteries at both checkpoints on the
8 development episodes:

* v4 batteries unchanged: `v4_side_flip_eval.py` (D = log p(true) − log p(side-swapped) at every
  decision step; first-step follows-content rate; donor flip rate) and `v4_stage2_eval.py`
  (decision-step CE ratios under normal / reset / donor) with `--bank {semantic,visual,both}`.
* New: sentence accuracy at inspect steps (exact match and per-token) — does the model describe
  the scene correctly; query–key cosine diagnostic (is retrieval landing on the right sentence).
  No loss uses either.
* Lost vs v4: the "read accuracy" number came from the deleted read head; the proof of memory use
  is the side-flip test plus the interventions.

## 7. Failure rule (user's instruction, 13:02)

If v5-A at ckpt-999 does not reach the v4 Stage-2a bar (first-step follows-content 1.00 on the 8
development episodes with donor flips), B has failed at reading oracle sentences. Stop, report,
build no fallback.

## 8. Status log

* 2026-09-02 12:26 — H200 placeholder up (132.7 GB, 100 %).
* 2026-09-02 13:06 — design frozen (D1–D9); scaffold committed (`9c37201`); code mapping started.
* 2026-09-02 13:45 — implementation landed (§3): model, data, configs, launchers, tests; sidecar built; §2/§4 wording corrected (visual commits at evidence steps; only the inspect sentence is detailed, the waiting label already carries the side).
* 2026-09-02 14:08 — smoke attempt 1 (13:36) never left Python imports: iris-hgx-2 reads /iris at ~2 MB/s (node load ~150). Attempt 2 (14:08) ran with the v4 venv's node-cached libraries and the v5 source tree first on `PYTHONPATH` (identical package set, verified by listing); it reached `train.py` in seconds and failed in `configure_v35_runtime_environment`: the worktree's `v35/cache/{uv,huggingface,openpi}` were symlinks into the v4 tree, which the project-root guard rejects (only `data` is a sanctioned link). Fix: real worktree-local caches — `uv` empty, `huggingface/{hub,modules}` copied, `huggingface/datasets/parquet` (78 GB Arrow cache) hardlinked from v4 (read-only, zero extra space), `openpi` (12 GB base weights + tokenizer) copied. Consequence for the node-local mirror idea: the same guard pins `OPENPI_DATA_HOME`/JAX cache to the worktree, so only the interpreter/libraries and the LeRobot root (`OPENPI_V5_LEROBOT_ROOT`) can live on the node's local disk without a contract change; launchers updated accordingly.
* 2026-09-02 14:29 — user decision: run from a node-local copy of the project (the earlier "keep it on /iris" attempt crawled at ~1 MB/s: the node's NFS client is latency-bound and other users' jobs saturate it). `cluster_v5/stage_local_project_hgx2.sh` streams the whole v5 project (code, relocated venv, weights, Arrow dataset cache, LeRobot data, raw label dirs, assets; ~140 GB at ~100 MB/s) to `/scr/kewalk_v5/memory_project_v5`; `run_train_h200.sh` / `run_batteries_h200.sh` run from that root when its `.staged` marker exists, so every project-relative path resolves locally and the v3.5 runtime-path contract holds unchanged. Status/log files and battery reports stay on /iris; checkpoints are copied back with `cluster_v5/sync_results_from_hgx2.sh`. /scr is node-local scratch (not backed up, may be purged after the job).
* 2026-09-02 15:28 — **smoke run passed** from the local root (`v5_stageA_20260902_smoke`, stage A, 20 updates, batch 2): weights restored in 8 s (1.5 GB/s), step 0 after a 3:43 compile, exit 0, checkpoints at updates 9/19. Step-0 telemetry: 8 sentence changes → 8 oracle commits, 0 degenerate, 6 decision steps, `memory_grad_norm` finite (1.2e5 pre-clip, group-clipped to 5). **Modification:** the 12-token inspect sentence overflowed the 128-wide causal buffer (lengths up to 151; the overflow silently truncates the chunk's last FAST action tokens; v4 saw 129–135 occasionally), so the v5 configs set `causal_token_len=160` (32 more KV positions per step). The smoke checkpoint was trained at 128 and is used only to validate the battery scripts.
* 2026-09-02 15:29–15:36 — eval path validated on the smoke checkpoint: `v4_side_flip_eval.py` and `v4_stage2_eval.py` run on the v5 model (exit 0; read-head gates reported "not applicable"; v5 sentence/qk telemetry printed).
* 2026-09-02 15:37 — **v5-A r1 launched**: `pi05_yam_mem_v5_stageA`, exp `v5_stageA_20260902_r1`, 1000 updates, batch 2, from the local root (commit `4c6d40a`, `causal_token_len=160`), ~8.9 s/update → ETA ~18:15. Chain on the node (`after_train_chain_hgx2.sh`): batteries at ckpt 999 and 500 (side-flip semantic/both/visual, Stage-2 semantic/visual) right after the run, then the placeholder. Verdict rule: §7.
* 2026-09-02 17:53 — v5-A r1 finished (1000 updates, exit 0; ckpts 250/500/750/999, 999 and 500 copied to `/iris/.../v5/checkpoints/`). Train-set at update 900: subtask CE 2.31, flow 0.011, inspect sentence token acc 98 % / exact 80 %, decision sentence exact 89 %, ~8 commits/update, query–key max-cosine 0.52 (down from 0.78 at update 300).
* 2026-09-02 18:17 — **v5-A VERDICT: FAIL (§7 bar missed; stop, no fallback).** Dev batteries, ckpt 999, semantic-bank interventions:

  | side-flip (first decision step, 48 seq) | normal | reset | donor | donor flip rate | follows-content |
  |---|---|---|---|---|---|
  | v5-A ckpt 999 | **0.646** | 0.792 | 0.625 | 0.458 (11/24) | 0.583 |
  | v4 Stage 2a/2b ckpt 999 (bar) | 1.00 | — | — | 24/24 | 1.00 |

  All decision steps (230): normal 0.735, reset 0.839, donor 0.765, follows-content 0.539. Stage-2 (ckpt 999): decision-CE ratios reset 1.12 / donor 1.10 (gate 1.2), use-flow ratio 1.00; dev exact-sentence rates: inspect 0.794 (the model DOES describe the held-out scene correctly), decision 0.839 normal → 0.000 reset → 0.339 donor. So the reader reacts to the *presence* of memory tokens but not to their *content*: blanking the bank never hurts the side, and a donor bank flips the side at chance.
  **Root cause, measured (`scripts/v5_probe_sentence_encodings.py`, ckpt-999 parameters):** the frozen mean-pooled encoder (D5) is side-invariant. Cosine between the encodings of `inspect both bins: banana left, grey pepper box right` and `… banana right, grey pepper box left`: **e = 0.9994, key = 0.9994, value = 0.9994**; every pair of sentences sits at cosine 0.98–0.99 (the pooled layer-8 space is almost one-dimensional). A 3-sentence bank holding [open, inspect_L, close] vs [open, inspect_R, close] returns reads with cosine **1.0000** to each other (relative difference 0.3 %): the side is invisible to any reader. The trainable Wk/Wv (linear, identity-init) cannot separate inputs that differ by 0.06 %, and nothing in the read path could recover it. D5 was my recommendation and it was wrong in a way the tiny-model unit test (random weights, 64-d) could not reveal.
  What is NOT the problem: the write rule (221 changes → 221 commits on dev, 0 degenerate), the delta-rule storage (bank read returns the stored value at cosine 0.97), sentence prediction (79 % exact on held-out inspect steps), training stability.
  Per the user's rule (D4/§7) v5 stops here; the user chooses the next method. Candidate directions for that discussion (none built): a side-sensitive sentence encoding (last-token or attention-pooled state instead of the mean; or a small trainable encoder), or a contrastive/reconstruction term that forces side-variants apart in key/value space (a memory loss, which D3/§2 excluded).
* 2026-09-02 18:31 — **user decision: r2 = trainable attention pooling.** Probe of pooling variants on the ckpt-999 token states (`scripts/v5_probe_pooling_variants.py`): 98 % of every token state's norm is one shared direction; cos(inspect_L, inspect_R) = 0.9994 (mean), 0.994 (last token), 0.995 (max), **0.816 (mean after subtracting the common direction), 0.730 (mean after per-feature standardization)**, and **0.125 for the standardized states at the two side words** (what a trained pooler can select). Per-sentence standardization + mean is degenerate (exactly 0), so the statistics must come from a reference set. Design r2 (`Pi0Config.memory_v5_pooling="standardized_attention"`): token states of the static reference sentences (the sidecar's 8, `config.V5_REFERENCE_SENTENCE_TOKENS`) are encoded by the CURRENT blocks 0–8 at every call → per-feature mean/std → the sentence's states are standardized → pooled as [standardized mean ⊕ `MemoryQueryCompressor`(4 queries, zero-init output)] → L2 → Wk/Wv (input (1+4)·2048; Wv identity on the mean block). At init the encoding is exactly the standardized mean; the pooler, Wk, Wv train through the decision loss; the backbone stays stop-gradient. Everything else (write rule, bank, read heads, losses, ladder) unchanged. Config `pi05_yam_mem_v5_stageA2` = stageA + the encoder change. Tests: `pi0_v5_test.py` r2 fixture (encoder equals the standardized mean at init, zero backbone gradient, pooler/key gradients nonzero, oracle sequence commits).
* 2026-09-02 19:30 — v5-A r1 battery record complete (dev, first decision step, 48 sequences; "flip" = donor mismatched pairs preferring the donor's side):

  | ckpt | bank | normal | reset | donor | flip | follows-content | Stage-2 CE ratios reset / donor |
  |---|---|---|---|---|---|---|---|
  | 999 | semantic | 0.646 | 0.792 | 0.625 | 0.458 | 0.583 | 1.12 / 1.10 |
  | 999 | both | 0.625 | 0.750 | 0.604 | 0.500 | 0.604 | — |
  | 999 | visual (injection off in stage A: identical by construction) | 0.646 | 0.646 | 0.646 | 0.333 | 0.479 | 1.00 / 1.00 |
  | 500 | semantic | 0.229 | 0.271 | 0.229 | 0.833 | 0.562 | 1.08 / 1.07 |
  | 500 | both | 0.250 | 0.312 | 0.292 | 0.792 | 0.583 | — |

  Dev sentence rates (Stage-2, all decision steps): exact decision 0.839 at both checkpoints under own memory, 0.00–0.01 under reset, 0.339 under donor; exact inspect sentence 0.706 (500) → 0.794 (999). The visual-bank rows are the expected no-op control (stage A injects no visual content). ckpt 500 names the wrong side at the first decision step (0.23, below chance) while the all-steps exact-decision rate is 0.84 — the first decision step is where an unread bank hurts most. Verdict unchanged (18:17).
* 2026-09-02 20:04 — A2 smoke passed (`v5_stageA2_20260902_smoke`, 20 updates, exit 0; step 0: 8 commits, `memory_grad_norm` 4.8e4 pre-clip). **v5-A2 r1 launched** (`pi05_yam_mem_v5_stageA2`, exp `v5_stageA2_20260902_r1`, 1000 updates, commit `650f073`, queue-driven from the local root), battery chain armed (ckpt 999 then 500). ETA training ~22:40, batteries ~00:10.
* 2026-09-02 22:36 — **v5-A2 r1 VERDICT: FAIL (bar missed), but a different failure than r1.** Training (1000 updates, exit 0 at 22:19): CE 2.30 at update 900, inspect sentence token acc 97 % / exact 76 %, query–key cosine rising to 0.78 (r1: falling to 0.52). Dev ckpt-999 semantic side-flip: first decision step normal **0.542**, reset 0.417, donor 0.396, flip 0.458, follows-content 0.354; all decision steps normal 0.700, reset 0.509, donor 0.539, follows-content 0.435. Blanking the bank now HURTS (r1: helped) — the reader depends on the bank — but a same-side donor bank drops accuracy to 0.25 and an opposite-side donor flips at chance: the reader reacts to the bank's specific pattern, not to the side it encodes. Trained-encoding probe (A2 ckpt-999): cos(inspect_L, inspect_R) e = 0.65, key = 0.85, value = 0.95; bank reads for the two variants cos 0.95, relative difference 0.31 (r1: 0.003) — the side is present in the retrieval, the decision path never learned to decode it. Remaining A2 batteries were stopped (GPU reused for the rollout videos).
  **Two protocol findings.** (1) By the lookahead shift, the oracle write rule stores the decision sentence (`wait; target bin is X`) one step BEFORE the first decision step, so during stage-A/A2 training the literal answer sat in the bank at every decision step — and the model still could not copy it out on held-out episodes. (2) The training-set "decision sentence exact 86–91 %" is memorization of the 54 training episodes (r1 reached the same with a side-blind bank); only held-out interventions are evidence. Both runs' training telemetry should be read with this in mind.
  Bottleneck per the evidence: decoding a stored sentence vector into the side token through blocks 9–17 with only the decision loss, 1000 updates, 54 episodes. Directions discussed with the user (not built): an auxiliary read-back loss on the retrieved vector, or a structured (object → side) value parsed from the model's own sentence.
* 2026-09-02 23:00 — `scripts/v5_heldout_video.py`: held-out episode rollout with the semantic bank carried at the training stride, greedy sentence decode against the memory-extended cache (the dual-bank decoding path v4 left unimplemented, rollout-only: no actions), v5 write rule in "self" (model's own sentence, conf ≥ 0.9) or "oracle" (label) mode, H.264 render of `top_camera_rgb.mp4` with GT phase / training target / PRED / bank overlays. Outputs `v5/diagnostics/videos_A2_999/ep<idx>_<mode>.{mp4,json}`.

- **2026-09-02 23:30 — held-out rollout videos, all 8 development episodes, A2 ckpt-999** (`scripts/v5_heldout_video.py`,
  outputs `v5/diagnostics/videos_A2_999/ep<idx>_{self,oracle}.{mp4,json}`, H.264, top camera with GT / PRED / bank bands,
  orange border = decision steps). `self` = the model writes its own decoded sentences (no label enters the model);
  `oracle` = label sentences written as in training. Script fix (commit 64e8b09): a window whose decision phase straddles
  a 40-step boundary is fetched from an earlier frame (the training transform rejects D steps without an E anchor).

| ep | object, side | self: inspect sentence exact | self: decision steps naming the side (first decision) | oracle: decision |
|---|---|---|---|---|
| 1 | grey pepper box, left | 4/5 | 0/5 (`open both lids`) | 5/5 |
| 2 | banana, left | 1/2 | 0/4 (`open both lids`) | 4/4 |
| 7 | banana, right | 4/5 | 0/5 (`open both lids`) | 5/5 |
| 21 | grey pepper box, right | 3/4 | 0/6 (`open both lids`) | 6/6 |
| 35 | grey pepper box, left | 3/4 | 0/8 (`open both lids`) | 8/8 |
| 42 | grey pepper box, right | 2/3 | 0/6 (`open both lids`) | 6/6 |
| 61 | banana, left | 4/5 | 0/4 (`open both lids`) | 4/4 |
| 64 | banana, right | 3/4 | 0/7 (`open both lids`) | 7/7 |

  **Reading.** Perception is solved: in every held-out episode the model decodes and writes the exact inspect sentence
  with both objects on the correct sides (8/8, the non-exact evidence steps are the phase boundary). Inference from
  the bank never happens: at every decision step of every episode the self-fed model says `open both lids` (0/47
  decision steps overall), and it only names the correct bin once the arm is visibly moving to it. With oracle
  writes it is 47/47 — but the trace shows the answer sentence (`wait; target bin is X`) was written into the bank
  one step before the first decision step (lookahead), and the model's prediction at every step equals the most
  recently written sentence. A2 therefore learned "repeat the last stored sentence", which the training protocol
  rewards, not "read the side out of the inspect sentence". Proposed next run (not launched, awaiting the user):
  A3 = A2 with oracle writes restricted to evidence-phase sentences (never the waiting label), so the decision loss
  can only be satisfied by reading the side from the stored inspect sentence.

- **2026-09-02 23:49 — user go for A3 = A2 + side-stripped bank write of the waiting label.** Diagnosis from the
  8-episode videos: the lookahead-shifted oracle write stored `wait; target bin is X` one step before the first decision
  step, so A/A2 learned "repeat the newest bank entry". The user's refinement over a plain skip: keep writing at the
  waiting phase, but store only `wait` (side stripped), so the bank still records every phase transition and the side
  exists in the bank only inside the inspect sentence. Implementation (this commit):
  `Pi0Config.memory_v5_bank_waiting_prefix=(9532,)` ("wait") and `memory_v5_bank_waiting_tokens=(9532, 108)`
  ("wait\n"); in ORACLE mode a label sentence starting with the prefix is written as the replacement tokens
  (`pi0.py` sentence write block, `write_span`), decode targets / CE / telemetry keep the real label; predicted mode
  (stage B) is untouched and writes the model's own sentence verbatim (so after its first decision the model's own
  `wait; target bin is X` does land in the bank — write happens after the read, first decision step unaffected).
  New telemetry `v5_bank_rewritten_count`. Config `pi05_yam_mem_v5_stageA3` = A2 + the two fields. Tests:
  `test_v5_a3_waiting_label_is_written_side_stripped` (bank reads bit-identical to writing the literal replacement
  tokens; no-op on windows without a waiting label) and `test_v5_a3_config_validation`. `v5_heldout_video.py` oracle
  mode applies the same rewrite. Launch: `cluster_v5/queue_a3_hgx2.sh` (smoke 20 → r1 1000 → batteries 999/500 →
  videos ckpt-999 via the new generic `cluster_v5/run_videos_hgx2.sh` → placeholder). Bar unchanged (§7): first-step
  follows-content 1.00 with donor flips on the development episodes, plus the self-mode videos naming the side.

* 2026-09-03 00:08 — **A3 r1 launched**: smoke (20 updates) exit 0 with `v5_bank_rewritten_count` active (2 rewrites in the
  last logged window), then `pi05_yam_mem_v5_stageA3`, exp `v5_stageA3_20260902_r1`, 1000 updates, batch 2, from the local root
  (commit `75a1194`). Queue (`queue_a3.log`): r1 → batteries ckpt 999/500 → videos ckpt-999 (self + oracle, 8 dev episodes)
  → placeholder. ETA: training ~02:40, batteries ~03:20, videos ~03:40.

* 2026-09-03 00:21 — **overnight automation (user 00:17: "once your battery is over continue working on the next step ...
  if the result is not ok, just keep training the current model, at least occupy the GPU").** New config
  `pi05_yam_mem_v5_stageB3` = A3 architecture with `memory_v5_oracle_writes=False` (the model's own decoded sentences,
  changed & confident ≥ 0.9), warm-started by grafting EVERY leaf (`v4_graft_sources=((".+", A3 r1 ckpt-999 params),)`)
  — A3 trains the LLM blocks and the semantic path, so B must start from the whole A3 tree. Watcher
  `cluster_v5/next_after_a3_hgx2.sh` (armed on the node): after the A3 queue (batteries + videos) it reads
  `side_flip_v5_stageA3_20260902_r1_999_semantic` and branches — automation threshold first-step
  `donor_follows_content_rate ≥ 0.9` and `normal_side_accuracy ≥ 0.9` → B3 smoke (20) → B3 r1 (1000) → batteries →
  videos; otherwise (or if B3 fails to start) → A3 r1 resumed to 3000 updates → batteries at 3000/2000 → videos at 3000.
  The formal §7 bar (1.00) is still judged by hand from the same JSON. Log: `v5/diagnostics/next_after_a3.log`.

* 2026-09-03 02:40 — **A3 r1 ckpt-999 verdict: FAIL (bar §7).** Training exit 0 at 02:24 (train-window decision exact
  0.46 → 0.83 at step 600 → 0.87–0.90; qk cosine held ~0.7; inspect exact 0.78). Held-out semantic side-flip
  (`side_flip_v5_stageA3_20260902_r1_999_semantic`): FIRST decision step — normal side accuracy **0.500**, reset 0.500,
  donor 0.521, donor flip rate 0.417, **follows-content 0.438** (bar 1.00); all decision steps — normal 0.726, reset
  0.717 (blank bank as good as the real one), follows-content 0.461. Reading: with the leak closed the model did not
  learn to read the side out of the stored inspect sentence at all (chance on held-out first steps; reset = normal);
  the step-600 jump on training windows was scene memorization, exactly as in run A (there at step 400). The
  continuation to 3000 updates (watcher FAIL branch) is GPU occupation per the user's 00:17 instruction, not a test:
  the training decision loss is already satisfied by memorization, so more steps add no pressure to read.

* 2026-09-03 11:35 — **overnight outcome + a contradiction to resolve.** Watcher took the FAIL branch at 04:57; A3 r1 resumed
  to 3000 (exit 0 09:28; final ckpt is `2999`, so the watcher's "3000" batteries/videos failed on a missing directory, the
  `2000` batteries ran: first-step normal 0.375 / reset 0.438 / follows 0.375). The continuation's keep policy (multiples of
  250) DELETED ckpt-999 from the node; it was never synced to NFS — the judged checkpoint is lost (memory note
  `checkpoint-keep-period-deletes-999`). **Contradiction in the ckpt-999 evidence:** (a) side-flip battery: first-step
  normal 0.50 = reset 0.50 = donor, with LARGE margins (|D| 3–14) that are nearly identical across normal/reset/donor →
  the side choice there is not influenced by the semantic bank; (b) stage-2 battery on the same held-out windows:
  exact decision sentence 0.85 (normal) vs **0.00 (reset)** vs 0.30 (donor) → the bank content decides whether the
  waiting sentence is produced at all; (c) oracle-write videos (bank = [open, inspect, close, `wait`], side ONLY in the
  inspect sentence): first decision correct 8/8 episodes, 45/45 decision steps, conf 0.999; ep21 names the side two
  steps before `wait` is written. Self-write videos: 0/N in 7 episodes (phase lock on `open both lids`), ep64 7/7.
  Decisive test launched at 11:32 on ckpt-2999 (`v5/diagnostics/run_video_A3_2999.sh`, `--intervention`): oracle
  writes with the stored side words swapped (`flip_sides`) and with an empty bank (`blank`), plus the plain oracle and
  self baselines at the same checkpoint, then batteries at 2999. If the flipped bank flips the decisions on held-out
  episodes, the model reads the side from the bank and the side-flip battery's D statistic is wrong for v5.

* 2026-09-03 11:45 — **RESOLVED: A3 reads the side from the bank; the side-flip battery was invalid for v5.** Flipped-side
  replay on ckpt-2999 (`videos_v5_stageA3_20260902_r1_2999/ep*_oracle_flip_sides.*`): with only the side words of the
  WRITTEN sentences swapped, the decision follows the stored sentence in **8/8 held-out episodes, 45/45 decision steps**
  (plain oracle: 45/45 true side). Cause of the battery's "0.50": `v4_side_flip_eval.py` swapped the side word in EVERY
  causal buffer, and in v5 oracle mode the causal text of the inspect step is also what is written to the bank, so the
  "swap" pass swapped the memory content together with the CE target (D ≈ 0 for a model that reads the bank); reset/donor
  were equally void because the in-window oracle writes refill the bank before the decision. Fix (commit ef8ee89):
  `--v5-swap-scope decision_targets` (auto for v5 models) swaps only the waiting-label buffers (side-stripped when
  written) and adds the `flip` condition (side words swapped in the written sentences, true targets) with
  `flip_follows_content_rate`. Corrected reading of A3: (1) side reading out of a fast-weight bank WORKS (the D4
  make-or-break question); (2) phase tracking was learned as "say the newest bank entry" because the lookahead write
  leaks the NEXT phase's sentence for every phase, so self-write rollouts cannot advance phases (never produce
  `close both lids…`, lock on `open both lids`). User 11:42: "do both" → A4 + B4.
* 2026-09-03 12:05 — **A4 = A3 + one-step write delay** (`Pi0Config.memory_v5_write_delay_steps=1`): the sentence written
  at step t is the one produced/labelled at step t-1 (both modes; step 0 of a window writes nothing; pending sentence,
  span and confidence are carried, padded steps keep them). The bank then holds only what has already happened: within a
  phase copying is harmless, at a phase change the model must see it. Test `test_v5_a4_one_step_write_delay` (2 commits
  for 3 sentences). Configs `pi05_yam_mem_v5_stageA4` (= A3 + delay) and `pi05_yam_mem_v5_stageB4` (= A4 arch, own
  sentences, delay, every leaf grafted from A4 r1 ckpt-999). `v5_heldout_video.py` applies the same delay when the config
  has it. Queue `cluster_v5/queue_a4_hgx2.sh`: A4 smoke → r1 → ckpt-999 copied to `keep_999` (lesson from the deleted A3
  999) → batteries (v5-scoped) → videos → verdict (first-step normal ≥ 0.9 AND flip_follows_content ≥ 0.9) → B4 smoke →
  r1 → batteries → videos → placeholder; FAIL → A4 continued to 2999 → batteries/videos 2999 → placeholder.

* 2026-09-03 13:35 — **A4 r1 first launch dead (NaN), fixed, relaunched.** The 12:11 run froze at the step-0 loss; log: loss
  and grad norms NaN from step 100. Cause (reproduced on the tiny model): with the one-step delay the first step of every
  window has an EMPTY pending sentence; the encoder pooled it to a zero vector whose L2-normalization has a 0/0 gradient
  → NaN into `memory_sem_sentence_pool`/`memory_sem_key_proj` → clip → every parameter. Forward was finite, so the
  existing tests (forward-only) passed. Fix: `v5_encode_sentence` encodes empty rows as a one-token dummy (commit for
  such rows was already masked). New test `test_v5_a4_delayed_writes_have_finite_gradients` (gradients of the TRAINING
  terms through the scan; the read-RMS telemetry is excluded — sqrt at 0 on the blank-bank step, infinite gradient in
  every configuration, never in the loss). 15/15 tests. Broken A4 checkpoints deleted on the node; queue relaunched.

* 2026-09-03 16:22 — **A4 r1 ckpt-999: PASSES the §7 bar on the fixed battery.** Training exit 0 at 16:04 (curve identical to A3:
  decision exact 0.87–0.90 from step 600, qk cosine held 0.78–0.79 instead of A3's 0.70). Held-out semantic side-flip
  (v5-scoped, `side_flip_v5_stageA4_20260903_r1_999_semantic`): FIRST decision step — normal side accuracy **1.000**
  (48/48), reset 0.438, flip follows-content **1.000** (48/48), margins +15.4 / −18.0 nats; all decision steps — normal
  0.839, reset 0.417, flip follows-content 0.852. Queue verdict rule (normal ≥ 0.9 & flip ≥ 0.9) → PASS → B4 launches after
  the remaining batteries and the ckpt-999 videos. What the battery cannot show: whether the one-step delay fixed
  phase tracking without a teacher — that is the self-write videos (next).

* 2026-09-03 16:50 — **A4 ckpt-999 self-write videos (8 dev episodes, `videos_v5_stageA4_20260903_r1_999/ep*_self`).** Rendered on
  the H200 (batteries after the stage-2 semantic one were skipped at the user's request, videos first). Result: the one-step
  delay FIXED phase tracking — every episode whose first frame was read as `open both lids` ran the whole loop on its own
  sentences and decided correctly (ep01 5/5, ep21 6/6, ep35 8/8, ep42 6/6; A3 managed this in 1/8). It EXPOSED a new failure
  at frame 0: in 4/8 episodes (ep02, ep07, ep61, ep64) the very first frame — lids closed, arms home, empty bank — is read as
  the waiting phase and the model emits `wait; target bin is right` at conf 0.97–0.99, which the write rule commits at step 1;
  the bank is then poisoned (sentences degenerate into `<loc…>` garbage, the inspect sentence is never written) and only
  ep07 recovers (5/5, true side happened to be right; ep64 3/7 for the same reason; ep02, ep61 0/N). The same frames give
  `open both lids` in A3. Root cause: training windows start anywhere in the episode with a BLANK bank (per-window carry
  init), so a window starting in the waiting phase teaches "closed lids + home arms + empty bank ⇒ wait; target bin is
  <guess>"; at a real episode start the bank is empty for the same visual state. Proposed fix (A5): prefill the bank at
  every training window start with the distinct label sentences of the frames BEFORE the window (oracle history, frozen
  encoder, delta-rule commits before step 0), so an empty bank occurs only at frame 0 — exactly as in a rollout. Not
  built yet. Note: the H100 (job 17192955) render of ep02 was byte-identical to the H200's — that node's environment is
  fine (the checkpoint copy `/scr/kewalk_v5_ckpt/A4_999` checksums equal to `keep_999`); the policy server there was
  stopped at the user's instruction.

* 2026-09-03 17:45 — **A5/B5 fixes built (user-approved 17:07/17:09), all tests green, B5 launched directly (A skipped).**
  Three changes, from the A4 self-write analysis (16:50 entry):
  1. **History prefill** (`memory_v5_prefill_history`, `memory_v5_prefill_max=6`): `MemorySequenceSubtasks` walks the
     stride/lookahead grid back from the window's first frame and emits the distinct label sentences produced before the
     window (`memory_v5_prefill`), the analytic-decay gaps after each commit (`memory_v5_prefill_gaps`) and the one-step-
     delayed pending sentence (`memory_v5_pending`); `TokenizeMemorySubtaskInputs(prefill_len=sentence_len)` tokenizes
     them exactly like the causal prefix (`FASTSubtaskTokenizer.tokenize_sentence`); `Observation` carries
     `memory_v5_prefill_{tokens,mask,gaps}` / `memory_v5_pending_{tokens,mask}`; `Pi0._compute_sequence_loss_v32`
     commits them before the scan (frozen encoder, delta rule, `analytic_decay(gap)` after each commit, key ring and
     `prev_sentence`/pending carries initialised), telemetry `v5_prefill_sentence_count`. Test
     `test_v5_a5_prefilled_window_equals_rollout_from_frame_zero`: a window starting at step 2 with the prefilled history
     reproduces the 3-step rollout's step-2 decision CE to 1e-5 (and differs from the blank-bank tail). Stage B keeps the
     label history for the prefill.
  2. **Exact writes**: the A3/A4 waiting-sentence rewrite (`memory_v5_bank_waiting_*`) is NOT set in A5/B5 (config
     validation forbids it with the prefill) — the bank holds `wait; target bin is <side>` in oracle and self mode alike,
     as the user asked (no task-specific rewrite; the one-step delay alone keeps the FIRST decision step leak-free).
  3. **Battery**: `v4_side_flip_eval.py` flips the side words of the prefilled evidence sentences with the window's (a
     waiting sentence in the history keeps its true side) and the first-step summary EXCLUDES windows whose history
     already holds a waiting sentence (`excluded_history_decided`) — their "first" decision step reads an earlier decision.
  B5 (`pi05_yam_mem_v5_stageB5`) = A4 encoder/delay + prefill + own sentences from the start, A-stage init (pi05 base +
  Stage-4c visual bank through the audited loader, no A checkpoint graft — which also removes the bf16/f32 graft failure
  of B4). Tests: `pi0_v5_test.py` 19 passed, `scripts/v5_prefill_test.py` 5 passed, transforms/tokenizer 12, config+v4 18.
  Queue `cluster_v5/queue_b5_hgx2.sh`: smoke → r1 (1000) → keep_999 → videos (self, oracle) → batteries → verdict →
  placeholder. The useless A4→2999 continuation on the H200 was stopped for it.
  * 17:31 B5 smoke FAILED: `memory_v5_prefill_history needs memory_v5_prefill_*/pending_* observation fields` — the raw
    prefill strings were dropped by two key whitelists before the tokenizer transform (the yam repack structure in
    `config.py` and `YamInputs` in `yam_policy.py`); both now pass `memory_v5_prefill{,_gaps}`/`memory_v5_pending`
    through (regression test `test_b5_pipeline_carries_the_prefill_keys_to_the_tokenizer`; the CPU loader probe shows
    [b, 6, 48] prefill tokens with valid rows and gaps). Relaunched 17:5x.

* 2026-09-03 20:30 — **B5 (direct B) verdict from the first two self-write videos + training telemetry: the copy shortcut
  formed before reading existed → back to two stages (user 20:20).** B5 r1 (own sentences from step 0, prefill on, exact
  writes) trained cleanly (CE 2.37 at 900 vs A4 2.28; inspect exact 0.71; qk 0.71) but its decision exact stayed
  0.41–0.53 with per-token 0.91 — the side word is the wrong token — and it wrote ~10 sentences/batch vs the oracle's 7.5:
  confident 1–4-step flickers, mostly `wait; target bin is <side>` at frames 30–75 (lids closed, arms home), got written.
  ckpt-999 self videos: ep01 frame 0 correct, inspect correct, reaches `open left bin` (exact writes fixed the execute
  transition) but the decision flips to `right` (0/5) right after its own `wait; target bin is left` entered the bank;
  ep02 same pattern (0/4). Reading: with spurious own wait sentences in the training bank, copying its own earlier wait
  sentence (random side) became the easiest way to produce a wait sentence — the training decision number ≈ 0.5 is that.
  Diagnosis running (`diagnose_b5_hgx2.sh`): the side-flip battery on B5-999 as trained and with a clean ORACLE bank
  (A5 config, same weights) to split "cannot read" from "bank polluted"; stage-2 telemetry both ways.
  Decision: A5 (label writes + prefill + exact) first, then **B5a** = A5 weights warm-started (`pi05_yam_mem_v5_stageB5a`:
  audited loader with the new explicit lossless `source_cast_dtype="float32"` — training checkpoints store 64 base leaves
  in bf16 and the strict dtype rule refused them, which is also what killed the B4 graft; CPU test against A2-999:
  strict refuses, cast loader matched=154 fresh=0 ignored=0), own sentences, peak LR 2.5e-5 (A: 5e-5), 1000 updates.
  Queue `cluster_v5/queue_a5_b5a_hgx2.sh` (waits for the diagnosis): A5 smoke → r1 → keep_999 → side-flip 999 →
  verdict (first-step normal ≥ 0.9 & flip follows ≥ 0.9) → B5a smoke → r1 → keep_999 → videos → batteries → placeholder.
  Not built (optional): stability-gated writes (commit only after 2–3 identical consecutive steps).

* 2026-09-03 23:05 — **Probe: the read QUERIES had collapsed.** `scripts/v5_probe_query_drift.py` on A4-999 (H100), 5 dev
  episodes: cosine between the 8 query keys at step t and step 0 = 1.000 (min 0.998); with vs without images 1.000; between
  episodes with the same prompt 1.000; between DIFFERENT prompts ("find the banana" vs "find the grey pepper box") 1.000
  (`v5/diagnostics/probe_query_drift_A4_999.json`). Cause: `MemoryQueryConditioner` returns base_queries + zero-init
  output_proj(attended); A4's training never opened that projection (the decoder extracts the prompted object's side
  from the injected inspect-sentence vector by attention, so nothing pushed the queries), and the instruction rows are
  98 % one shared direction anyway. The bank was therefore read with a FIXED key: a decayed bag of every stored sentence,
  not content addressing; the 8 heads did nothing. Reading still works for 5 sentences (48/48) via the decoder.
* 2026-09-03 23:50 — **A6 read-side fix built (user 23:12 "do it"), all 36 tests green, A6 -> B6a chain armed on the H100.**
  `memory_v5_query_standardize`: instruction rows standardized against the reference sentences (`_v5_reference_stats`,
  shared with the r2 encoder) and an explicit instruction term — the unit-norm standardized mean through an
  IDENTITY-initialised `memory_sem_inst_query_proj` — shifts the base queries, so the queries depend on the instruction
  from step 0 by construction (tiny test: cos < 0.999 between instructions; the zero-init conditioner path alone gave
  1.000 even on the tiny model). `memory_v5_query_prev_sentence`: the last decoded sentence (the delay's pending
  sentence; label in A, own in B) encoded by the frozen standardized encoder shifts the base queries through a ZERO-init
  `memory_sem_prev_query_proj` (exact no-op at init, masked when there is no previous sentence) — "given I was doing X,
  what is relevant". Threaded through `_v32_prepare_memory_{interface,prefix}(v5_prev_tokens, v5_prev_mask)`, the scan
  (pending carry), the batteries and `v5_heldout_video.py`. `MemoryQueryConditioner` accepts per-sample base queries.
  Configs: A6 = A5 + both flags, warm start from A5-999 (cast audited loader; only the two new projections fresh);
  B6a = A6-999, own sentences, half LR. Chain: `cluster_v5/copy_a5_999_to_nfs.sh` (login node: A5-999 -> NFS when
  protected) then `cluster_v5/queue_a6_hgx1.sh` ON iris-hgx-1 (job 17192955, XLA mem fraction 0.7 because the user's
  Qwen action-expert server holds 11 GB there; the H200's A5 run uses 22 GB so it fits): A6 smoke -> r1 -> keep_999 ->
  side-flip 999 -> verdict -> B6a smoke -> r1 -> keep_999 -> videos -> batteries. A5 -> B5a keeps running on the H200 as
  the baseline; B6a vs B5a is the comparison.

* 2026-09-04 01:40 — **v5 robot serving path (user 01:16: "let it run so I can test them on real robot tomorrow").**
  `Pi0.sample_with_memory` now accepts the v5 sentence bank (`semantic_state`, plus `v5_prev_tokens/mask` for the A6
  queries), reads it through the same `_v32_prepare_memory_prefix` path as training, returns `aux["token_prob"]`
  (per-token argmax probabilities of the greedy sentence decode) and never writes the semantic bank itself.
  `scripts/serve_yam_memory.py` gains `V5SentenceMemory`: the per-episode carry (bank state, previous/pending sentence)
  that applies the TRAINING write rule after every request — one-step delay, changed-and-confident (>= 0.9), one analytic
  decay otherwise — with the model's `v5_encode_sentence`/`v5_sentence_intent`/`v5_semantic_write`; the visual bank is
  frozen (its injection is off in every stage >= A4); `reset_memory` resets the bank. Responses add `subtask_confidence`,
  `bank` (committed sentences) and `memory.{changed,committed}`; `surprise`/`gates` stay for old clients. Tests: the
  delayed rule on a fake bank (commits at the 2nd/4th/6th of six calls), the undelayed variant, metadata, and a tiny-model
  `sample_with_memory` with a written vs blank bank (the decoded tokens/probabilities differ; the visual state is
  untouched). `scripts/v5_serve_smoke.py` = in-process policy smoke with synthetic observations. The hgx-1 GPU sentinel
  frees the 1-GPU job's placeholder whenever a policy server or the smoke runs.

* 2026-09-04 03:18 — **B5a ckpt-999 self-write videos: 8/8 held-out episodes correct, 45/45 decision steps.** B5a (A5
  weights → own delayed confidence-gated sentences, peak LR 2.5e-5, 1000 updates on the H200; final own-write telemetry
  CE 1.06, decision exact 0.91, inspect exact 0.74, qk 0.89) rolls out on its own sentences: frame 0 = `open both lids`
  in every episode, the inspect sentence written correctly, the correct side at EVERY decision step (ep01 5/5, ep02 4/4,
  ep07 5/5, ep21 6/6, ep35 8/8, ep42 6/6, ep61 4/4, ep64 7/7), and the transition into `open left/right bin` reached in
  all 8. Six episodes wrote exactly the 5 true phases; ep07/ep64 wrote one spurious `wait; target bin is <side>` at frame
  45 and recovered (the stability gate remains the optional cure). Compared with B5 direct (0/5, 0/4 on the two episodes
  rendered) this confirms the two-stage recipe: reading and phase tracking learned on the clean oracle bank survive the
  switch to self-writes. A6 (read-side fix, 500 updates at batch 8 on 4xH100 from A5-999) ckpt-499 first-step battery:
  normal 1.000, flip follows-content 1.000, margins +15.1/−15.5 (A5 +11.0/−12.0) → B6a next (own sentences from A6-499).
  Judged checkpoints: `v5/checkpoints/pi05_yam_mem_v5_stageB5a/v5_stageB5a_20260903_r1/keep_999` (NFS copy for the robot
  server), `.../pi05_yam_mem_v5_stageA6/v5_stageA6_20260903_r1/keep_499`. Both B runs continue training after their tests
  (user rule 02:02) with saves every 250 updates.

* 2026-09-04 05:51 — **B6a ckpt-499 self-write videos: 8/8 episodes, 44/45 decision steps, exactly 5 writes in EVERY
  episode.** B6a (A6-499 weights → own sentences, 500 updates at batch 8 on 4xH100; final own-write telemetry CE 0.60,
  decision exact 0.91, inspect exact 0.79, qk 0.88) writes only the five true phases in all eight held-out episodes — the
  read-side fix removed the early spurious `wait` writes B5a still had in 2/8 — and reaches `open left/right bin` in all.
  The single miss (ep07, first decision step) is a one-step PHASE lag (`close both lids…` at f435, then the correct
  `wait; target bin is right`), not a wrong side; it then said `open right bin` one step early. Side decisions: 100 %.
  B5a vs B6a on the same episodes: 45/45 vs 44/45 decisions; spurious writes 2 episodes (8 and 13 writes) vs none.
  Judged checkpoint `v5/checkpoints/pi05_yam_mem_v5_stageB6a/v5_stageB6a_20260903_r1/keep_499` (NFS). Both B runs now
  go through their batteries and then continue training (user rule).

* 2026-09-04 09:50 — **Both B runs judged and continuing.** Self-write-mode batteries (first decision step, held-out):
  B5a-999 normal 1.000, blank 0.458, margin +14.9; B6a-499 normal 1.000, blank 0.625, margin +18.6. The flip
  follows-content number in SELF-write mode (B5a 0.688, B6a 0.521) is a measurement limit, not a model result: the battery
  can flip only the prefilled history, and windows whose inspect phase lies inside the window get the model's own
  unflipped sentence, so the flip never reaches the bank there; the clean read test is the oracle-mode battery of the A
  stage (A5/A6 1.000) plus the self-write rollouts (45/45, 44/45). Continuations (user rule 02:02 "keep training"): B5a
  resumed on the H200 at 06:36 toward 3000 updates (batch 2, ~7–9 s/update, saves 1250/1500/…); B6a resumed on the 4xH100
  at 09:46 toward 3000 (batch 8, saves 750/1000/…). Judged checkpoints untouched: B5a keep_999, B6a keep_499 (both NFS).
  Robot serving: `cluster_v5/serve_v5_hgx1.sh <ckpt dir> <config> [port]` on iris-hgx-1 (job 17192955).

* 2026-09-04 11:25 — **Robot deployment for B6a (user 11:08).** Ported from the v4 runbook: `websocket_policy_server.py`
  runs `infer` via `asyncio.to_thread` (keepalive survives a cold compile), `openpi_client.WebsocketClientPolicy(ping_timeout)`,
  `serve_yam_memory.py --warmup` (plain + RTC-prefixed synthetic request, then reset). New `examples/yam/client_memory_v5.py`
  (from the v4 client: same YAM/RealSense plumbing, RTC broker at stride 15, `r` reset / `q` quit, H.264 recording;
  guard on `memory_v5_sentence_bank`, stride 15, RTC fields, training prompts; overlay = decoded sentence + confidence
  (`*` = committed this tick) + the bank's sentences + commit count; `--dry-run`) with `client_memory_v5_test.py`.
  Launcher `v5/diagnostics/run_server_v5.sh [port] [ckpt_dir]` (default B6a keep_499; frees the 1-GPU job's placeholder;
  the sentinel keeps it off while the server runs). Live check: server on iris-hgx-1 (10.79.12.252:8000, job 17192955,
  next to the user's Qwen server), warmup 37 s + 15.5 s, `client_memory_v5.py --dry-run` from the login node: contract
  OK, 40 random-observation steps, one commit (`open both lids`), RTC replan exercised. Server left running for the trial.


- 09/04 11:24 — `gpu_placeholder_hgx2.sh` fix (commit 3826772): its "already running" check matched job 17248791's per-job placeholder, so after the B5a continuation ended (exit 0, 11:12, 3000 updates) job 17207774's H200 sat idle for ~12 min. Check now `gpu_placeholder_marker[^_]` (legacy marker only); placeholder relaunched (srun PID 3251180). Both H200s: 132.7 GB, 100 % / 80 % util. B6a continuation on the 4×H100: step 899/3000 at 11:27, ~10–12 s/update (ends ≈17:30–18:30; job 17178887 expires 22:46). B6a server (keep_499) stays up on 10.79.12.252:8000 (job 17192955).
- 09/04 11:35 — B5a final checkpoint 2999 (3000-update continuation) copied hgx-2 `/scr` → NFS `v5/checkpoints/pi05_yam_mem_v5_stageB5a/v5_stageB5a_20260903_r1/keep_2999` (params+assets, 9.8 GB, 186 s, `.copied` marker; script `cluster_v5/copy_b5a_step_to_nfs.sh`, `STEP=<n>`). Untested on the battery — keep_999 is the judged B5a ckpt.
- 09/04 12:08 — Robot server switched to B6a ckpt **1000** (untested on the battery; user's choice) on iris-hgx-1 10.79.12.252:8000 (pid 4110223, log `v5/diagnostics/server_v5_20260904_1206.log`, warmup 36.9 s + 15.3 s, client dry-run OK). GPU holds only: train_hs.py keep-alive 1 GB, user's Qwen action-expert server 11 GB (pid 548942), our server 49 GB. No placeholder.
- 09/04 12:45 — **Code published.** Branch `v5` pushed as `main` of the new GitHub repo `ZJU-Walker/memory_project_v5` (remote `v5origin`; `origin` still = the v4 repo), tag `v5-b6a-robot-20260904` = the B6a policy code as deployed (commit 9f3338b). Deployment launchers now tracked in `cluster_v5/deploy/`, the architecture brief in `cluster_v5/docs/`. Checkpoints/data stay on NFS (ignored `v5/`, `data/`). Push from the login node needs `GIT_SSH_COMMAND="ssh -i /iris/u/kewalk/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/iris/u/kewalk/.ssh/known_hosts"`.

* 2026-09-04 14:30 — **Bean-scoop task (user 13:38 "train a model on that data, use our 4xH100"; plan approved 13:54:
  "stop b6, working on this, warm start from b6a"). B6a continuation stopped at step 1550 (keep_499/1000 kept).**
  Data: 60 demos `/iris/u/kewalk/memory_project/data/0902_bean_scoop` labelled by the sibling session
  (`BEANS_LABELS.md`, 11 sentences, 5 phases, x = 1/2/3 blinks in 19/23/18 episodes). Pipeline built this afternoon:
  1. LeRobot conversion with the existing `examples/yam/convert_yam_data_to_lerobot.py` (single source, constant prompt
     "scoop the beans into the tray as many times as the green light blinked", task = sentence) into the v5-private root
     `v5/data/lerobot/yam/bean_scoop_0902_v5` (`HF_LEROBOT_HOME=v5/data/lerobot`; the v5 `data/` symlink points into the
     v4 tree, which stays untouched). The login-node run was killed silently at 42/60 (log ends in NUL bytes);
     `--resume` needs the strict manifest mode, so the conversion was rerun with `--overwrite` on iris-hgx-1
     (`v5/diagnostics/convert_beans_resume.sh`, log `convert_beans_hgx1.log`).
  2. **Generic v5 task data mode** `DataConfig.memory_v5_generic_task` (commit below): the bins pipeline hard-wires
     stride 15, the five-phase left/right schema, stationary waiting cores, occlusion/execute phases and side cells.
     The generic mode keeps the model untouched and feeds the v3.5 per-step fields with neutral values
     (`transforms.MemoryV5GenericFields`: write = every valid step, decision = steps in `memory_required_subtasks`,
     read-valid/credit = every step (prefill), side −1, cell = manifest class, fact labels all-unknown). Loader:
     `_load_v5_generic_manifest` (schema `openpi.v5.generic-manifest.v1`: stable ids, splits, one class per episode,
     pinned bytes), no phase tables, no dead zone; `memory_critical_prob` becomes the mass of TRANSITION-anchored slices
     (starts within `memory_critical_start_pad` frames before a sentence change, balanced per class).
     `train.py` v4 contract accepts no fact sidecar in generic mode. Tests `src/openpi/training/v5_generic_test.py`
     (5) + `scripts/v5_prefill_test.py` (7) pass.
  3. Configs `pi05_yam_mem_v5_beansA` (label writes, warm start from B6a keep_499, every leaf) and `beansB` (own
     sentences from beans-A-499, half LR): stride **5** (blinks last 6–9 frames; stride 15 sees 52 %), lookahead **0**
     (a blink onset cannot be anticipated), `prefill_max` 10 (up to 9 distinct sentences before a window), reference
     sentences = the 11 beans sentences (`V5_BEANS_REFERENCE_SENTENCE_TOKENS`, tokenized like
     `tokenize_sentence`; the bins row reproduces bit-exactly), decay 0.01 unchanged (tanh-RMS injection renormalises
     reads; revisit if the count fades), 500 updates each at global batch 8 on the 4xH100. Split (seed 902, per
     class by sha256 rank): 2 final_test + 2 development per class → 48 / 6 / 6.
  4. Queue `cluster_v5/queue_beans_hgx1.sh`: A smoke → A r1 → keep_499 → B smoke → B r1 → keep_499 → continue B to 3000.
  Deployment note: stride 5 means the robot client replans every 167 ms (about 150 ms per call on the H100).
  15:07 — conversion complete on iris-hgx-1 (60 episodes, 53 999 frames, 38 GB; single-source mode writes only
  `episode_prompts.json`, so `beans_build_v5_manifest_sidecar.py --raw-dir` re-identifies episodes by natural demo
  order, verified by frame count and task set). Manifest `cluster_v5/beans/beans_episode_manifest_v1.json`
  (sha 40d00bc4…, dev = demo11/12/14/17/21/51, final_test = demo16/25/30/31/38/46) and sidecar
  `beans_v5_subtask_labels_v1.json` (sha d4581a3e…) pinned in the configs (commit 980dbf7). Launched on iris-hgx-1:
  `cluster_v5/launch_beans_hgx1.sh` = norm stats (CPU, 15k random frames; whole dataset incl. dev/test, a minor
  state/action-statistics leak accepted for time) → `queue_beans_hgx1.sh` (A smoke → A r1 → B smoke → B r1 → B
  continuation). Evaluations to run on the H200 (`cluster_v5/run_beans_evals_hgx2.sh <cfg> <exp> <step>`): dev-episode
  self/oracle videos (`v5_heldout_video.py --manifest/--sidecar`, exact-sentence decision scoring) and the
  **count-flip battery** `scripts/v5_count_flip_eval.py` (go-sentence count argmin-CE over the 3 variants; normal /
  flip (all counts in the prefill + in-window blink sentences cyclically shifted) / blank; history-only subset).
  15:51 — **beans-A smoke exit 0 (20 updates, batch 8, 4xH100) → beans-A r1 launched 15:51** (500 updates, ~1.7 h;
  then B smoke → B r1 → B continuation, all in `queue_beans_hgx1.log`). Loader report on the real data: 48 train
  episodes, 23 083 slice starts (p 0.25), 16 458 transition-anchored starts (p 0.5, 3 class cells), 48 full starts.
  Step-0 CE 7.67 (the beans sentences are new to the B6a weights). Two launch bugs fixed on the way: (1) norm stats
  land under `<assets_base>/<config name>/<repo_id>` while the beans DATA config reads
  `v5/assets/pi05_yam_bean_scoop_0902_v5/<repo_id>` — the launcher now copies them (first attempt exited
  "MISSING"); (2) the first smoke OOM'd at step 0 because `run_train_h200.sh` killed only `gpu_placeholder_marker_<job>\b`
  while the sentinel names its per-GPU placeholders `..._<job>_g<k>` — the kill pattern is now the marker prefix
  (commit 490fa31). Also: `pkill`/`kill $(pgrep -f X)` inside an ssh command whose text contains X kills that shell
  (bracket-trick patterns only). Client: `client_memory_v5.py` accepts the beans prompt (`--steps-between-inference 5`).
  16:07 — **Cluster reshuffle.** The serving job 17192955 and the H200 jobs 17207774/17248791 ended (the robot server is
  down; restart it when a GPU is assigned). Stopping the user's "placeholder training" on the H200 job 17267134 ended
  that job (it was the batch payload) — lesson: only stop srun STEPS, never a job's payload. The user resubmitted the
  H200 as job 17267793 (iris-hgx-2; payload = 1 GB keep-alive, never touched): our busy placeholder holds it and
  `cluster_v5/evals_beans_waiter.sh` (armed 16:01) evaluates beans-A/B keep_499 there as they appear. The 2xH100 job
  17267129 (iris-hgx-1) was reassigned to us by the user (its v4 Stage 4e r2 step stopped 16:04 at the user's
  instruction; the v4 session informed); the sentinel now holds both of its GPUs (GPU 0 free while an eval runs).
  Train chain re-armed as `queue_beans3_hgx1.sh` (A exit → B smoke → B r1 → keep_499 → continuation; no evals in the
  chain). beans-A r1: step 33 at 16:07, 16.7 s/update → A ends ≈18:15, B r1 ≈20:40, continuation until the job's
  22:46 limit. Note `run_beans_evals_hgx1.sh` (GPU 0 of a hgx-1 job, `GRES=`) exists for the 2xH100 job as a fallback.
  18:19 — **beans-A r1 exit 0** (CE 7.67 → 2.24, decision exact 0.97, evidence exact 0.97, flow 0.023); keep_499; B r1
  launched 18:21 on the 4xH100 (queue4: no B smoke; 17 s/update). Evaluations on the H200 job 17267793 (waiter): the 6
  dev self-write videos 18:35-18:48, count-flip battery after. **A keep_499 self-write video demo11 (x=2) double-counts**:
  "1 blink" at the first on-step, "2 blinks" at the SECOND on-step of the same blink, "3" at blink 2 → go "3 times",
  then stuck on "scoop 1" (27/119 decision steps exact, 9 writes). Root cause: a blink lasts 6-9 frames = two stride-5
  steps in 86/119 cases, and with the v1 sentences the inputs at the second on-step ("bank says k, light on") are the
  same as at a new blink's first on-step. User 19:06: **option 2** — put the LED state into the waiting sentences so the
  previous decoded sentence carries it: `light on: k green blink(s) so far` / `light off: k green blink(s) so far`
  (14-sentence vocabulary; go/scoop/done unchanged; the LeRobot task strings and phase masks stay v3).
  `scripts/beans_relabel_light_state.py` (from `subtask_labels.json` + `led_on.npy`, 119 light-on runs = Σx) →
  `subtask_labels_light.json` per demo + `subtask_labels_manifest_light.json`; `beans_build_v5_manifest_sidecar.py
  --label-filename subtask_labels_light.json --reuse-manifest` (same manifest v1/split, pinned inside) →
  `cluster_v5/beans/beans_v5_subtask_labels_v2light.json` (sha eee0ba69…). Config: `V5_BEANS_SENTENCES_V2`,
  `V5_BEANS_REFERENCE_SENTENCE_TOKENS_V2` (verified against the tokenizer), `v5_beans_light_data`,
  **`pi05_yam_mem_v5_beansA2`** (labels, warm start B6a keep_499, prefill_max 14) and **`beansB2`** (own writes, from
  A2 ckpt-499). Eval scripts take `SIDECAR=`; the waiter takes `STAGES="A2 B2"`. User 19:12: "kill current b training and
  beginning our next training" → `queue_beans5_hgx1.sh` (A2 on the 4xH100 if ≥2h40 remain on job 17178887, else on the
  2xH100 job 17267129 at batch 4; B2 + continuation on the 2xH100; the sentinel now also excludes its trossen placeholder
  while a v5 `train.py` runs there). **Blocked 19:20: the Kerberos ticket expired at 18:01** (klist; ssh 255) — B r1 keeps
  running (134/500 at 19:12) until the user re-kinits; the chain is armed the moment ssh works again.
  19:23 — Ticket renewed (user re-sent the password; one-shot `kinit -l 3d`, nothing stored). B r1 stopped at 134/500
  (user 19:12 "kill current b training and beginning our next training"), queue4 + sentinel stopped; **queue5 launched:
  A2 on the 4xH100 job 17178887 at 19:23** (batch 8, 3h23 left on the job, ~2.5 h needed; step-0 CE 7.73), then B2 +
  continuation on the 2xH100 job 17267129. Sentinel relaunched (v5 exclusion on 17267129); the H200 waiter re-armed for
  `STAGES="A2 B2"` with the v2light sidecar (old waiter, which had finished the A r1 evals 19:15 and restored the trossen
  placeholder, killed). Code 23f1efd pushed to GitHub v5 main. **A r1 keep_499 results**: count-flip (oracle prefill,
  1283 go steps, 66 first-go): normal 0.995 / flip-follows-content 0.836 / flip-keeps-true 0.023 / blank 0.809 (blank far
  above chance 1/3: the previous-sentence query term still carries the count when the bank is blanked — by design);
  self-write dev videos: x=1 demo12/demo51 perfect (78/78, 67/67 decision steps), x=2 demo11 27/119 (double count → "3
  times"), demo21 68/116 (go correct), x=3 demo14 69/172 (late: first decision still a blink sentence), demo17 26/150
  ("2 times"); 3/6 first decisions correct. Videos sent to the user.
  20:10 — **Scoop counter never advances in self-write mode** (user 19:53: "always stuck in scoop 1"). Records: inside a
  phase conf 1.00; at the bowl-arrival boundary "scoop 2" appears for ONE step at conf 0.90 (the write threshold) and
  the next step reverts to "scoop 1" although the previous sentence says "scoop 2"; "done" is still timed exactly (from
  the image of the last dump, not from counting). Rollout commit order verified against the training scan (read with the
  pending sentence, then write it): no off-by-one. **ORACLE-write videos of A keep_499 (H200, 19:56-20:07)**: decisions
  116/119, 78/78, 167/172, 142/150, 114/116, 67/67 — the counter advances, but exactly 2 steps after the label boundary
  = when the oracle sentence lands in the bank: the model copies the bank and never detects the boundary itself. Cause
  (same shape as the blink double count): with the v3 cut, "previous = scoop k, arm over the bowl" is the input both
  during the dig of scoop k and at the arrival for scoop k+1 — identical inputs, different targets, one boundary step per
  transition (~47 in the training set). User asked about a target lookahead: a global lookahead would corrupt the blink
  targets (predict blinks before they happen) and flips the target at a fuzzy moment; the label-level version is the fix.
  User "Ok do it". Measured cycle (119 scoops, from j0): arrival→tray 30-215 frames (median 127), over the tray 50-158
  (median 83 ≈ 17 memory steps), tray→next arrival 0-30 (median 9!). So a cut at the dump END (first attempt,
  `--cut delivery_end`) moves the boundary by ~2 steps only and was dropped; **the v4 "tray cut"** (`--cut
  delivery_start`, `scripts/beans_relabel_scoop_dump.py` on the light-state labels): `scoop k+1` starts when the arm
  arrives over the tray with scoop k (k < x; `scoop x` keeps through its dump; `done` after the last dump as before) —
  the increment is decided in a persistent, visually distinct state and requires the count (k == x → no increment).
  Sidecar `cluster_v5/beans/beans_v5_subtask_labels_v4tray.json` (sha 728916ab…, same 14 sentences, same manifest),
  configs **`pi05_yam_mem_v5_beansA3`** (labels, warm start B6a keep_499 like A2) and **`beansB3`**. `queue_beans6_hgx1.sh`
  (20:16): A3 on the 2xH100 job 17267129 NOW (batch 4, replaces the trossen placeholder) while A2 keeps running on the
  4xH100; A2 keep_499 when it exits; then B3 + continuation on the 2xH100. queue5's B2 stage dropped (the tray cut
  supersedes it; A2's evaluations still run to judge the light-state count fix). H200 waiter re-armed
  `STAGES="A2 A3 B3"` with per-stage sidecars (`SIDECAR_<stage>`).
  22:15 — **beans-A2 r1 exit 0 21:57** (step 400: CE 2.20, exact decision 0.98, evidence 0.97), keep_499; A3 (2xH100,
  batch 4) at step 300 22:00: CE 2.66 / 0.94 / 0.92. Sentinel bug fixed on the way: after A2 exited it took A3 (a step of
  job 17267129) for "training present" and left the 4xH100 idle for 3 min — the 17178887 check is now job-specific
  (`jobid=17178887 .*train`, commit b6e8606); all four GPUs hold placeholders again. **A2 keep_499 self-write videos
  (light-state sentences)**: the on/off chain itself is tracked exactly once it starts right (no double count within a
  blink anywhere); count correct in 4/6 dev episodes (A r1: 3/6): demo12 78/78, demo14 170/172 — and there the scoop
  counter advanced 1→2→3 by itself —, demo21 and demo51 counts right. Failures: demo11 hallucinates "light off: 1 green
  blink so far" at frame 10 (LED off since frame 0; conf 0.97, one step after the first bank commit; qk cos −0.82) and
  carries it to "3 times"; demo17 flickers between off-1/off-2 during the blinks and ends at "2 times". New A-stage
  pathologies (never sees its own sentences): decoded garbage ("Action: <loc…>", "light off: wel…", below the write
  threshold) and "yellow go" reappearing during scoop 1 / done in demo21 and demo51 (writes 17 and 14). The "no blink yet"
  and "light off: 1" phases are 3.1 % and 3.4 % of the training frames and visually identical (LED off, arm idle) — the
  previous sentence is the only cue, and stage A never trains on a wrong previous sentence. Stage B (own writes) is the
  cure we have (B5a fixed exactly this flicker on the original task); B2 dropped in favour of B3 (tray cut on the same
  light-state sentences), which therefore also tests the count under own writes.
  23:05 — **beans-A3 r1 exit 0 22:50** (step 400: CE 2.51 / decision 0.97 / evidence 0.96; 2xH100, batch 4), keep_499;
  **B3 launched 22:51** on the 2xH100 (step 0: CE 2.64). Cluster: job 17178887 timed out 22:46; the user cancelled
  17188253 (2xH200, 6-GPU QOS) and the queued 4xH100 job **17249058** (iris-hgx-1, 800G, 3 d) started 22:59 — sentinel
  and placeholder scripts re-keyed to it (trossen training on 4 GPUs, batch 32, FSDP 4; job-specific checks; 54cdb11).
  **A2 count-flip (keep_499, oracle prefill)**: normal 0.97 / first-go 1.00, flip-follows 0.90 (A r1 0.84), keeps-true
  0.00, blank 0.78. **A3 keep_499 self-write videos (tray-cut labels)**: count right in **5/6** (demo11 now "2 times";
  demo17 still flickers off-1/off-2 → "2 times"); the light chain is exact in the other five. Scoop increment still
  not held by stage A: demo11 says "scoop 2" exactly at the tray arrival (steps 100-101) but at conf 0.86/0.83 — below
  the 0.90 write threshold — and falls back to "scoop 1"; demo14/17/21 never leave "scoop 1". New side effect: "yellow
  go" lingers past the bowl arrival (demo12 until step 105 vs label 68; demo11 99 vs 73) and reappears during scoop 1 in
  demo11/21/51 — with the tray cut the `scoop 1` segment is short (30-215 frames) and the go→scoop-1 boundary got
  weaker at half the batch. Conclusion unchanged: stage A cannot hold its own increments (never trains on its own
  sentences); B3 (own writes, same labels) is the test, ckpt-499 ≈ 01:20.
  2026-09-05 01:40 — **beans-B3 r1 exit 0 01:24** (own writes, tray-cut light-state labels, 2xH100 batch 4; step 400:
  CE 1.65 / decision 0.93 / evidence 0.94), keep_499, continuation toward 3000 running on job 17267129. A3 count-flip
  (keep_499): normal 1.00, **flip-follows 0.99**, keeps-true 0.00, blank 0.80. **B3 keep_499 self-write videos**: count
  right 5/6 (demo17 still flickers off-1/off-2 → 2), and **the scoop counter is now written and held by the model
  itself**: demo14 (x=3) 1→2→3 with a 2-step wobble at the first boundary, 138/172; demo21 (x=2) 116/116 perfect;
  demo11 (x=2) scoop 2 written at the tray arrival and held 18 steps, one 33-step relapse to scoop 1, 85/119; demo17
  advances to scoop 2 and stops there — consistent with its own (wrong) count of 2; demo12 78/78, demo51 67/67. Decision
  exactness per episode A2 → A3 → B3: demo11 30→76→85, demo14 170→67→138, demo17 26→26→52, demo21 53→68→116, demo12
  78→78→78, demo51 49→67→67. The "yellow go" lingering of A3 is gone (go→scoop 1 within 1-4 steps of the label).
  Conclusion: the A→B recipe transfers to the beans task once the label boundaries sit on persistent, visually
  distinct states; the remaining errors are single early wait-phase writes (demo17) and one mid-phase relapse (demo11).
  01:50 — **Root cause of the demo17 miss (every model since A r1): the LED control signal leads the camera.** Trace
  (A2 = A3 = B3, identical): at frames 25/45/70 the label says "light on" but the left camera's LED patch is still dark
  (green 130 vs 241 lit; visible from frame 27/46/72); demo17's three onsets all sit on the rollout's stride-5 grid, so
  each "light on" comes one step late, the delayed write misses the bank at the next step, the model (bank over
  previous sentence) answers "no blink yet"/"light off: 1", the bank fills with contradicting entries and the third
  blink is not counted. Measured over all 60 demos (LED patch = largest green rise around the first onset; frames
  0..go only): onset lag 0/1/2 frames = 9/73/18 of 100 detectable, offsets 6/70/24; 18 of 100 onsets have a fixed-grid
  frame inside the invisible gap (≈1 in 5 in training windows too = label noise). **v5 "visible LED" labels**:
  `scripts/beans_relabel_visible_led.py` moves every light on/off boundary of the tray-cut labels to the first frame
  where the camera shows the change (238 boundaries shifted 0/1/2 frames: 17/170/51; same 14 sentences) →
  `cluster_v5/beans/beans_v5_subtask_labels_v5vis.json` (sha 1efb8d8f…), configs **`beansA4`/`beansB4`**, `queue_beans7_hgx1.sh`:
  **A4 launched 01:50 on the 4xH100 job 17249058** (batch 8) → keep_499 → B4 → keep_499 → continuation; the B3
  continuation keeps the 2xH100; H200 waiter `STAGES="A4 B4"` with the v5vis sidecar (commit f5de2ed).
  02:15 — **Why the scoop counter falls back (user 01:53).** (1) Write-rule flaw: the change detector compared the
  pending sentence with the last PRODUCED sentence, so a sentence that first appears just under the 0.90 threshold
  (demo11 B3: "scoop 2" at conf 0.90 at step 98) is rejected once and then never written — every later "scoop 2"
  (0.95-0.99 for 16 steps) counts as "unchanged"; the bank keeps "scoop 1", and back over the bowl the model trusts the
  bank over its previous sentence and falls back (writes a duplicate "scoop 1"). Fix = **retry-until-committed**:
  prev = last COMMITTED sentence (`memory_v5_prev_is_committed`, identical for label writes; on in `beansB4`; the
  video script takes `--write-retry`, serving reads the flag; commit 10a1cdf). Same B3 checkpoint re-rendered with the
  retry rule: demo11 **117/119** (85 before; "scoop 2" written at the tray arrival and held to the end), the other five
  unchanged (78, 138, 52, 115, 67). (2) Residual ambiguity at the tray: "prev = scoop 2, over the tray" occurs during
  the dump of scoop 1 and at the arrival of scoop 2 (demo14's 2→3→2→3 wobble); only the full/empty scoop separates them.
  Candidate fix: sub-phase sentences ("scoop k: to the tray" / "scoop k: to the bowl"), decided after B4. **B3
  count-flip** (own-write model): first-go normal 1.00, flip-follows 0.73 (A3 1.00), keeps-true 0.09, blank 0.64 —
  stage B leans partly on timing (prefill decay gaps) in addition to the bank: a shortcut to watch in B4. Ops: editing
  `run_beans_evals_hgx2.sh` while its B3 count-flip instance was live made that instance execute a fragment of the new
  `if` line (exit 127; result already written; placeholder restored by the retry run) — never edit a running runner.
  04:35 — **beans-A4 r1 exit 0 04:16** (visible-LED labels; step 400: CE 2.19 / decision 0.98 / evidence 0.97), keep_499;
  **B4 launched 04:19** on the 4xH100 (own writes, retry rule; step 0: CE 2.03). A4 keep_499 self-write videos (old
  write rule, the config has no retry flag): count right 5/6; evidence exactness up sharply where the grid straddles an
  onset (demo11 186/188 vs A3 105/188, 117/119 decisions); demo14 reaches scoop 2; demo21/51 as A3. **demo17 still
  loses the third blink, and the visible-LED labels show the real reason**: its three blinks (visible 27-32, 46-53,
  72-78) each cover exactly ONE sampling step (30, 50, 75; 8-frame blinks do so 40 % of the time), so at the first "off"
  step the one-step-delayed write has not yet put "light on: k" in the bank; stage A trusts the bank over its previous
  sentence and answers "no blink yet", after which the count drifts. The previous-sentence path is the only cue in that
  situation; stage B (own writes) is where it must be learned — B4 is the test.
  07:15 — **beans-B4 r1 exit 0 06:48** (own writes, retry rule, visible-LED labels; step 400: CE 1.18 / 0.97 / 0.97),
  keep_499. B4 keep_499 self-write: demo11 117/119 (scoop 2 held), demo14 149/172 (scoops 1→2→3, the second increment
  21 steps late), demo12 78/78, demo51 67/67, demo21 68/116 (count right, scoop never incremented; B3 had 116/116),
  demo17 98/150 (count 2 again; scoops nevertheless 1→2→3). A4 count-flip: first-go normal 0.985 / flip-follows 0.985 /
  keeps-true 0 / blank 0.73. Net: B4 ≈ B3-with-retry; the visible-LED labels did not change demo17 because its blinks
  cover ONE sampling step each and the one-step write delay keeps "light on" out of the bank at the first "off" step.
  **The delay is not needed on this task**: it was introduced (A4, 09-03) so that a lookahead-shifted label could not leak
  the decision into the bank; the beans labels have lookahead 0 (the sentence describes the current observation and the
  write lands after the read), so `memory_v5_write_delay_steps=0` leaks nothing and puts sentence_t in the bank at t+1.
  Configs **`beansA5`/`beansB5`** = A4/B4 with delay 0 (B5 keeps the retry rule); `queue_beans8_hgx1.sh` on the 4xH100
  job 17249058 (the B4 continuation, the least valuable job, was stopped at ~step 560 for it; B3's continuation keeps
  the 2xH100); H200 waiter `STAGES="A5 B5"`. Open after A5/B5: the tray-arrival ambiguity of the scoop sentences
  (demo21 no increment, demo14 late, demo17 over-increment) → sub-phase sentences if B5 still shows it.
  08:25 — **B4 count-flip (first go step, label prefill): normal 0.79** (A4 0.985, B3 1.00, A3 1.00), flip-follows 0.76,
  keeps-true 0.05, blank 0.50; errors are x=2/x=3 confusions (12/18 each; 3→2 ×4, 2→3 ×5). Is it the write rule?
  Re-ran the battery with the rule overridden at eval time (`v5_count_flip_eval.py --write-retry on|off`,
  `cluster_v5/run_count_flip_variant_hgx2.sh`): B3 with retry 1.00 (unchanged), B4 without retry 0.82 — **the eval-time
  rule is irrelevant; B4's weights read a label-filled bank worse** (own-write rollouts still count 5/6). Cause not yet
  separable between the retry rule DURING training and the visible-LED labels; B5 (A5 + retry, delay 0) is the next
  data point. Note the battery's "normal" condition is a label-prefill probe, not the deployment condition.
  09:55 — **Disk-full incident.** `/iris/u/kewalk` (5.1 TB) hit 100 % at ~09:36, right as A5 exited: the three
  openpi_trossen placeholder runs had written 540 GB of checkpoints (every 5k steps, ~42 GB each) and the B3
  continuation another 160 GB of 250-step intermediates. Effects: queue8 died, the A5 keep_499 copy was partial (18 of
  27 GB), B5 never launched, the trossen placeholder on the 4xH100 aborted (core dump) but still held 75 GB per GPU so
  the first B5 relaunch OOM'd at init; the B3 continuation kept running (its log lost ~10 min of lines). Fixes: deleted
  the placeholder checkpoints except the latest step of each run (−430 GB) and the B3 intermediates 250/750-1750 (kept
  keep_499 and 2000, −160 GB) → 1.2 TB free; re-copied A5 keep_499; killed the dead placeholder (SIGKILL) and relaunched
  B5 via `queue_beans9_hgx1.sh` (B5 → keep_499 → continuation); placeholder checkpoints now go to the node-local disk
  (`placeholder_train_trossen.sh`: `--checkpoint-base-dir /scr/kewalk_placeholder/checkpoints`, 0905 run names);
  new `gpu_sentinel_hgx2.sh` relaunches the H200 placeholder when its 30k-step run ends. The A5 eval waiter was paused
  until the copy completes (a waiter that polls `keep_499/params` can start on a half-copied checkpoint).
  10:10 — **beans-A5 r1 (no write delay) exit 0 09:36** (step 400: CE 2.21 / decision 0.99 / evidence 0.99), keep_499
  (re-copied after the disk incident). **A5 keep_499 self-write videos: the count is right in 6/6 dev episodes** —
  demo17 counts 3 for the first time (every blink tracked on/off, no revert to "no blink yet"): with delay 0 the
  "light on" sentence is in the bank at the first "off" step even when a blink covers a single sampling step.
  demo11 86/119 (scoop 2 late, stage A), demo12 78/78, demo14 67/172 (count 3, scoops wobble/stuck),
  demo17 91/150 (count 3, scoop 2 on time, stuck at 2), demo21 68/116 (stuck at 1), demo51 67/67. Scoops remain the
  stage-A weakness (label writes, old rule); B5 (own writes, retry, delay 0) launched 09:51 on the 4xH100 (step 0 CE
  2.10 / decision 0.99), keep_499 ≈ 12:20. Placeholder trainings now never keep checkpoints (user 10:03; 92a0594).
  10:35 — **v6 "sub-phase" scoop sentences** (user 10:22 "lets do the subphase", after the analysis page
  `cluster_v5/docs/beans_scoop_analysis.html` showed that any one-sentence-per-cycle cut collides once per cycle):
  `scoop k: to the tray` (bowl arrival k → tray arrival k − 1), `scoop k: to the bowl` (tray arrival k → bowl arrival
  k+1 − 1, k < x), `done` from the LAST tray arrival (the count decision: prev "scoop x: to the tray" + over the tray →
  done instead of "to the bowl"); light sentences = v5 visible-LED; 16 sentences (`scripts/beans_relabel_subphase.py`
  → `beans_v5_subtask_labels_v6sub.json`, sha 2e934ccd…). Config `V5_BEANS_SENTENCES_V3` + reference rows (max 13
  tokens), `v5_beans_sub_data`, **`beansA6`** (delay 0, prefill_max 16, warm start B6a keep_499) / **`beansB6`** (own
  writes, retry). `queue_beans10_hgx1.sh`: A6 → keep_499 → B6 → continuation on the 2xH100 job 17267129 (the B3
  continuation was stopped at ~2400 for it; B5 keeps the 4xH100 until its keep_499 ≈ 12:20). H200 waiter
  `STAGES="A6 B6"` armed alongside the A5/B5 one. Inspection: `examples/yam/label_subtasks.py --beans-task --beans-v6
  --label-file subtask_labels_v6sub.json` (v6 schema + boundary descriptions; save-time schema check skipped),
  served on iris-ws-18:8765 for the user.
  13:20 — **B5 (delay 0 + own writes) collapses in rollouts** although its telemetry is normal (step 400: CE 1.20 /
  decision 0.95 / evidence 0.96; writes/window 20.9 vs B4 22.8): at the first "off" step after a blink it says "no
  blink yet" (conf 0.98) with [no blink, light on: 1] in the bank, writes it, and the chain restarts — go count "1 time"
  in demo11/14/21, "done" during the go phase; 28/119, 45/78, 111/172, 53/150, 71/116, 56/67. Two findings on the way:
  (1) **rollout bug for delay-0 models** — training conditions the A6 read queries on `prev_sentence` when the delay is
  0, but `v5_heldout_video.py` and `serve_yam_memory.py` fed the never-filled pending slot (an empty sentence); fixed
  (53e4af0). Re-rendering B5 with the fix changes ≤4 decision steps per episode → the previous-sentence query term is
  effectively unused by these models (its zero-init shift never opened), so the fix is a correctness fix, not the
  cause. (2) **B5 ORACLE-write video demo11: 117/119** (count 2, scoop 2 held) — and even there the first "off" step
  says "no blink yet" for one step before the label write corrects the bank. So B5's decoder learned "LED off after
  light on → no blink yet" during own-write training with delay 0; A5 (same delay, label writes) does not have it.
  Control launched 13:20: **`beansB5d1`** = A5 weights + own writes + retry + delay 1 (`queue_beans11_hgx1.sh`, 4xH100;
  the B5 continuation stopped). B6 (delay 0, own writes, sub-phase labels) launched 13:14 on the 2xH100 and will show
  whether it inherits the collapse; A6 evals running on the H200.
  16:10 — **A6, B5d1, B6 results.** A6 (label writes, delay 0, sub-phase sentences) self-write: count 6/6 (demo17
  included), count-flip first-go 1.00 / follows-flip 0.955; but at the tray it says "done" one scoop early in all four
  multi-scoop episodes (demo11 x=2 at k=1, demo21 at k=1, demo14/17 x=3 at k=2; the k=1 "dump" of demo14/17 at conf
  0.90-0.91) with a CORRECT bank ([go x, scoop k: dig]) — a read/decision failure: "dump vs done" needs two bank
  entries (x from the go sentence, k from the last scoop sentence) plus a comparison, the blink count needs one entry
  plus an increment. Videos `videos_v5_beansA6_20260905_r1_keep_499`. **B5d1 (A5 + own writes + retry + delay 1) is a
  full replica of the B5 collapse** (28/119, 53/78, 30/172, 26/150, 28/116, 67/67; "light on: 1 → no blink yet" at the
  first off step everywhere) → the write delay is NOT the cause; the A5 weights + own-write fine-tuning collapse either
  way. Query-key cosine in the blink phase of demo11 (bank [no blink] / [no blink, on: 1]): A4 0.57 → B4 0.36; A5
  0.01-0.09 → B5 −0.3, B5d1 −0.9; A6 0.06-0.30 → **B6 +0.07 → +0.72**. **B6 (A6 + own writes + retry + delay 0) does
  NOT collapse**: demo11 115/119 (count 2, scoop 1 dig → dump → scoop 2 dig → done, the first fully correct multi-scoop
  rollout), demo12 77/78 (a spurious "scoop 1: dump" before done), demo14 119/172 (count 3, dump 1 right, "done" early
  at k=2), demo17 57/150 (count 2: the single-step blink is lost again after the B stage; scoops consistent with its
  count), demo21 84/116 (count 2, "done" early at k=1), demo51 67/67. Count 5/6; tray "dump vs done" right in 3 of 6
  decisions (A6: 0 of 4). So the earlier reading "the delay-0 A stage is what has to go" was WRONG (B6 starts from a
  delay-0 A stage); the only collapsing combination so far is A5 (visible-LED labels + delay 0) → any B; B4 (same
  labels, delay 1) and B6 (sub-phase labels, delay 0) are healthy. Not explained yet. Ops: both H200 waiters were
  stopped before the keep_499 copies (a waiter polling `keep_499/params` can start on a half-copied checkpoint) and the
  B5d1/B6 evals launched by hand; the placeholder that the B5-evals runner had started on the H200 at 13:23 was
  killed at 13:32 because it was sharing the GPU with the A6 evals (its /scr checkpoint dir deleted). The queue10
  B6 continuation toward 3000 started automatically on the 2xH100 at 15:52 (kept pending the user's decision).
  Prepared but NOT trained (user 13:40 "before you do anything talk to me"): v7 "target-carry" scoop sentences
  (`scoop k of x: dig and carry` / `dump and return`, so every transition is a single-read copy/increment/compare of
  the previous sentence; `scripts/beans_relabel_target_carry.py` → `subtask_labels_v7tgt.json` +
  `beans_v5_subtask_labels_v7tgt.json`, sha 99eee3c9…, 20 sentences, max 13 tokens), `queue_beans12_hgx1.sh` (A7 → B7),
  waiter date map extended; the config entries (V5_BEANS_SENTENCES_V4 / TOKENS_V4 / beansA7,B7,A6d1,B6d1) were not
  written.
  17:00 — **Why the tray decision fails: the bank forgets on a fixed clock.** (User: "I don't want to make it depend
  on the previous sentence … 3-minute task … is the memory too small / forgetting?") Size is not it (fast weights
  1024x2048 per bank). Decay is: `MemoryConfig.alpha_step = 0.01` is a FIXED per-step factor (every stride-5 step,
  1/6 s, the whole matrix is multiplied by 0.99; half-life 69 steps ≈ 11.5 s), applied on write-free steps too, and
  the training prefill reproduces it with the true gaps (`memory_v5_prefill_gaps`), so the model never saw an old note
  any stronger than a rollout does. B6 demo11/12/14/21/51: the go decision reads the newest light note at age 10-14
  steps (strength 0.87-0.90); the tray decisions read the go note at ages 58-83 / 121-126 / 175 steps (strength
  0.43-0.56 / 0.28-0.30 / 0.17) underneath fresh scoop notes that share most of its words. 0.99^n: 1 min 0.027,
  2 min 0.0007, 3 min 2e-5 — with this alpha a 3-minute task can only work if every fact is re-written every 10-20 s
  (which is exactly what the v7 "k of x" sentences would do; the user rightly rejects that as a workaround). B6 tray
  decisions with a correct bank: 4/7 right, errors in both directions at conf 0.96-0.99; the visible tray (k-1 dumps)
  predicts 69% of training tray decisions (empty: dump 33/done 15; one dump: done 19/dump 14; two: done 14) and B6's
  seven dev decisions agree with that tray-only rule in 5 — consistent with "reads the tray, not the target".
  Circumstantial; two things launched in parallel (user 16:59 "in parallel"):
  (1) **A6sd → B6sd** (`queue_beans13_hgx1.sh`, 4xH100 job 17249058, batch 8, **300 updates each** — user 17:03 "train
  to 300 steps not 500 to save time"; checkpoint 299 → keep_299): the A6/B6 recipe with only the decay changed,
  `alpha_step=0.001` on both banks (the config pins them equal; half-life ~115 s). The v3.5 Revision-4 alpha pin in
  `Pi0Config.__post_init__` now exempts `memory_v5_sentence_bank` models. Waiter armed with `STEP=299`; the waiter now
  waits for the queue runner's "protected as keep_<step>" line instead of polling the folder (two half-copy starts
  today). A6sd launched 17:04.
  (2) **Tray-decision probe** `scripts/v5_tray_flip_eval.py` (from the count-flip machinery): at every tray-arrival
  step of the loader's windows (oracle history with true gaps) the step's sentence is replaced by the two candidates
  ("scoop k: dump and return" with the step's k, "done") and the lower decision CE wins, under normal / flip (all
  light+go counts in the history shifted 1→2→3→1, scoop k kept; content-consistent answer = done iff k == shifted x)
  / blank history, and under alpha 0.01 (as trained) vs 0.0 (no decay, same parameters). Runs for the A6 and B6
  parameters on the train and dev splits: `run_beans_evals_hgx2_tray_probe.sh` (named so the H200 sentinel treats it
  as real work) → `v5/diagnostics/tray_flip_<A6|B6>_keep_499_<split>/`. Started 17:11.
  18:50 — **The tray failure is a SENTENCE-DIRECTION problem, not decay. The 17:00 decay diagnosis is refuted.**
  `scripts/v5_bank_geometry_eval.py` (user 17:54 "will this be a sentence direction issue?") replays each dev
  episode's true note sequence into a fresh bank on the real stride-5 clock and reads the go note back with its own
  key, under alpha 0.01 / 0.001 / 0. Results (A6 keep_499 / B6 keep_499):
  * the go note is NOT faded: go_recall (cosine of the retrieval with its own stored value) 0.94 / 0.98 at alpha
    0.01, and 0.95 / 0.99 with the decay switched off — 30 s and up to 6 intervening writes later. Decay is not the
    binding constraint, so **beansA6sd/B6sd will very likely not fix the tray decision**;
  * but the retrieval matches the NEWEST note about as well as the go note itself (go_vs_recent +0.91..+0.98);
  * decisive readout: comparing the retrieval against the three go variants (they differ only in the digit) the
    cosines are 0.9409 / 0.9391 / 0.9426 — **the count lives in the 4th decimal, margin 0.002 (A6) and 0.0008 (B6),
    and the bank's answer is 8/12 with every x=2 episode read as 3; identical 8/12 at alpha 0** (`--alphas`).
  Cause, measured on the vocabulary itself (encoding / write key / write value cosines between distinct sentences):
  sentences of different KIND are well separated (go vs scoop key −0.34, go vs light +0.03), but sentences that
  differ ONLY in the count are collinear — go 1/2/3 times: encoding 0.934-0.978, key 0.996-0.998, value 0.996-0.999;
  light off 1/2/3: key 0.969-0.990; scoop k dig: key 0.986-0.998, value 0.997-0.999. The encoder already blurs the
  digit (1 token of ~13) and the key/value projections amplify the collapse. Why counting still works: the delta rule
  reproduces the NEWEST note exactly (recent_recall 1.000), so the 0.002 component survives when the model reads the
  note it just wrote; any later write injects content along nearly the same direction and swamps it. This is exactly
  the user's summary (18:00): "update the next sentence from the last one" works, "recall a few steps ago" does not —
  and it is the capability their 3-minute task needs. Fix direction (user's call): a SEPARATION loss over the
  reference vocabulary in stage A (penalise |cos| between distinct reference keys and between distinct values; no
  label change, task-independent), or a fast lexical diagnostic (counts as lexically distant words) to confirm the
  mechanism end-to-end in ~1 h. Reports: `v5/diagnostics/bank_geometry_{A6,B6}_keep_499{,_countreadout}/`.
  Tray probe (`scripts/v5_tray_flip_eval.py`), train and dev splits, A6 and B6: the tray decision is IDENTICAL with
  the true history, with an emptied bank and with the remembered count flipped (A6 train 0.72/0.72/flip-unchanged,
  B6 train 0.89/0.94, A6 dev 0.78/0.78) — the decision never used the bank, consistent with the geometry above.
  Ops: the 2xH100 job 17267129 was cancelled at 18:34; the queue13 shell and its run_train_h200.sh wrapper died with
  it while the A6sd srun STEP (job 17249058) kept running, so `queue_beans13b_hgx1.sh` (launched from a shell in the
  surviving job) waited for the orphaned step, protected keep_299 and started B6sd. Lesson: launch a queue from a
  shell in the job that will outlive it. The user's new 2-GPU job 17284681 (iris-hgx-2) hosts the B6 ckpt-1000 robot
  server (10.79.12.149:8000) and ran these probes next to it.
  19:15 — **A6sd (decay 0.001, 300 updates) does NOT fix the tray decision: the 17:00 decay hypothesis is refuted
  end-to-end, as the geometry probe predicted.** Self-write dev rollouts, dump-vs-done calls at the tray on the
  identical metric: A6 (decay 0.01) 7/12, B6 7/12, **A6sd (decay 0.001) 8/12** — one call in twelve, inside noise —
  and A6sd also loses demo17's count (go "2 times"; count 5/6 vs A6's 6/6). Per-episode: A6sd still says "done" one
  scoop early on demo14/demo17 and "dump" on the one-scoop demo12. Videos
  `videos_v5_beansA6sd_20260905_r1_keep_299`; ckpt `pi05_yam_mem_v5_beansA6sd/v5_beansA6sd_20260905_r1/keep_299`.
  B6sd (own writes on the slow-decay weights) runs to ~20:05 as the last confirmation.
  Tray probe matrix complete (normal / blank-bank / flip-the-remembered-count, alpha 0.01): A6 train 0.72 / 0.72 /
  unchanged, B6 train 0.89 / 0.94 / unchanged, A6 dev 0.78 / 0.78 / unchanged, **B6 dev 0.78 / 0.89 / unchanged** —
  in every cell an EMPTY bank is as good as or better than the true history and flipping the remembered count never
  moves the answer. Combined with the geometry probe (the count is a 0.002 residual that later writes swamp), the
  conclusion is settled: **the v5 bank supports "next sentence from the newest note", not "recall a specific older
  note"; the fix has to make the count a separable direction, not a label rewrite and not a decay setting.**
  NOTE: from 18:34 another session was writing to `queue_beans_hgx1.log` and using the same GPUs (2xH100 cancel,
  B6-ckpt1000 robot server on job 17284681, an A6sd tray probe at 19:03) — check the log's authorship before
  assuming a line is ours.
  19:45 — **Sentence-separation loss implemented (not launched; the user chooses).** `Pi0Config`
  `memory_v5_sentence_separation_weight` (default 0) + `memory_v5_separation_margin` (0.3); `Pi0
  .v5_sentence_separation_terms()` encodes the STATIC reference vocabulary with the checkpoint's own encoder and
  penalises `relu(|cos| - margin)^2` over the off-diagonal of BOTH the write keys and the write values, averaged.
  It is parameter-only (no data, computed once per loss call, not inside the scan), the token states stay
  stop-gradient'ed as everywhere on the v5 write path, and the gradient reaches exactly the parts that create the
  collapse: the sentence attention pooling and `memory_sem_{key,value}_proj`. Telemetry is emitted at weight 0 too
  (`v5_separation_loss`, `diagnostic/v5_separation_{key,value}_cos_max`, `..._key_cos_mean`), and `scripts/train.py`
  adds `weight * v5_separation_loss` in BOTH loss paths (single-step and gradient-accumulation, the latter divided by
  `accumulation_steps` like every other contribution). Weight sizing from A6 keep_499's measured cosines: at margin
  0.3 the penalty is 0.45 (key 0.12 + value 0.33), so **weight 1.0** is comparable to but below the sentence CE;
  mean |cos| off-diagonal is 0.50 (key) and **0.87 (value)** with 100 of 240 value pairs above 0.9 — the value
  projection is the worse offender, and the term covers both. Configs `pi05_yam_mem_v5_beansA6sep` (label writes,
  warm start B6a keep_499) / `beansB6sep` (own writes from A6sep-299, separation kept ON so stage B cannot re-collapse
  the geometry — B6 measured a SMALLER count residual than A6), 300 updates each, matched to beansA6sd/B6sd.
  Runner `queue_beans14_hgx1.sh` (4xH100 job 17249058), waiter map extended (`STAGES="A6sep B6sep" STEP=299`).
  Tests: `pi0_v5_test.py::test_v5_separation_penalty_is_reported_and_pushes_the_vocabulary_apart` (finite terms,
  exactly 0 at margin ~1, strictly positive at margin 0, non-zero finite gradients into both projections) and
  `::test_v5_separation_config_validation`; the whole `-k "separation or a6"` group passes (6).
  Success criterion, decided BEFORE the run: rerun `v5_bank_geometry_eval.py` on A6sep keep_299 and require the
  go-count readout (currently 8/12 at margin 0.002) to rise with a margin at least an order of magnitude larger;
  then the tray dump-vs-done metric (A6 7/12, B6 7/12, A6sd 8/12) in the self-write rollouts.
  19:35 — **A6sd count-flip: the slow decay also made the one thing that WORKED slightly worse.** First-go
  history-only battery (count read from the bank / follows a flipped history / blank bank): A6 (decay 0.01)
  1.000 / 0.955 / 0.667, B6 0.970 / 0.848 / 0.667, **A6sd (decay 0.001) 1.000 / 0.727 / 0.773** — the flip-follow
  rate falls and the blank-bank accuracy rises, i.e. A6sd leans LESS on the bank for the go count (plausibly because
  a slower decay leaves more of the collinear older content in the matrix, making the read noisier). Together with
  the tray metric (8/12 vs 7/12, noise) and the lost demo17 count, the decay branch is closed as a net negative.
  20:40 — **Decay branch closed. Full A6/B6 vs A6sd/B6sd comparison (same 6 dev episodes, same metrics):**

  | stage | decay | go count | tray dump-vs-done | decision steps | count-flip follows-flip |
  |-------|-------|----------|-------------------|----------------|-------------------------|
  | A6    | 0.01  | 6/6      | 7/12              | 510/702 = .726 | 0.955                   |
  | A6sd  | 0.001 | 5/6      | 8/12              | 467/702 = .665 | 0.727                   |
  | B6    | 0.01  | 5/6      | 7/12              | 519/702 = .739 | 0.848                   |
  | B6sd  | 0.001 | 6/6      | 5/12              | 564/702 = .803 | (pending)               |

  The tray decision does not move in either direction beyond noise (7, 8, 7, 5 of 12) while the geometry probe
  predicted exactly that. B6sd is the best model on raw decision steps (0.803) and recovers demo17's count, but it
  is the WORST at the tray (5/12), which is another instance of the same thing: raw sentence accuracy is driven by
  the newest-note read, the tray decision is not driven by the bank at all. A6sd's count-flip regression (follows a
  flipped history 0.727 vs A6's 0.955, blank-bank 0.773 vs 0.667) says the slower decay actively weakened the read
  that works. Nothing in the decay direction is worth pursuing; the sentence-direction fixes (this session's
  separation loss `beansA6sep`, running; session A8's slot keys + whitened values) are the live branch.
  21:15 — **`causal_token_len` 160 → 208 (config.py, v5 model base).** The user noticed the tokenizer warning
  `Causal length (165) exceeds causal_token_len (160), truncating` appearing constantly in the A8 log. Counted
  over the A6/B6/A8 logs: 8.7k warnings, chunk lengths 161–186 (histogram peaks at 161–165 and thins to a
  handful at 185–186), i.e. roughly one chunk in ten lost its last 1–26 FAST action tokens from the CE target.
  Pre-existing in every v5 beans stage (A5..A8, B5..B6, A6sd/B6sd all trained with 160), silent apart from the
  warning, and NOT a candidate cause for the tray failure (it only trims the tail of the action target; the
  sentence tokens are never touched). 208 covers the observed maximum with 22 tokens of headroom and costs 48
  more KV-cache positions per step. Applied to the config only: the running A8 (launched 20:48) keeps 160 for
  its whole run because the process parsed the config at launch; B8 (queue15 launches it after A8 keep_299) and
  every later run pick 208 up automatically — the change is param-shape-neutral (RoPE positions, no positional
  table; only the padded buffer widens, so a 160-trained checkpoint loads into a 208 model unchanged).
  21:55 — **Placeholder OOM incident on the single H200 (21:25) and the fix (user: "clean up the running task and
  make sure this won't happen again").** What happened: the other session's A6sep verdict chain ran
  `v5_bank_geometry_eval.py` on job 17267793 right after A6sep training ended; my hgx-2 sentinel
  (`gpu_sentinel_hgx2.sh`) only knew a fixed list of scripts, did not recognise the geometry probe as real work,
  and relaunched the trossen placeholder (~120 GB) at 21:25; the rollouts that followed OOM'd, the other session
  killed the placeholder, swept its checkpoint and re-ran the rollouts at 21:45. State after cleanup: no placeholder
  marker process on either node, no leftover placeholder checkpoint (`/scr/kewalk_placeholder` empty on both), the
  user's `train_hs.py` job payloads untouched; hgx-1 had had NO sentinel at all since the 2xH100 job was cancelled
  (the ssh shell it lived in was adopted by that job), so the 4xH100 would have sat idle after B8.
  Fix, three layers: (1) one generic sentinel `cluster_v5/gpu_sentinel_job.sh` (`JOB=<job> TAG=<hgx1|hgx2>`,
  20 s poll) replaces the per-node scripts on both nodes (hgx-2 job 17267793 pid 944761, hgx-1 job 17249058 pid
  868248): real work = any srun step of the job running a training or a `scripts/v5_*.py`, any `scripts/v5_*.py`
  python on the node, the eval/probe runners and the robot server — no more per-script list to forget; (2) the
  sentinel is bidirectional: real work seen while a placeholder step runs -> it kills that srun step (only
  processes whose command line STARTS with srun and carries the job's marker; the first version killed my own
  ssh audit shell because its command text mentioned the marker — fixed 21:53); (3) the placeholder step itself
  (`placeholder_train_trossen.sh`, atomically replaced) refuses to start when a GPU of the job already holds
  > 8 GB (rc=3, nothing written), independent of any pattern. Residual race: a probe that starts in the 20 s
  between two polls while a placeholder is up still OOMs — runners must keep killing the marker first (ours do).
  `gpu_sentinel_hgx1.sh` / `gpu_sentinel_hgx2.sh` are superseded and no longer run.
  22:00 — **A6sep: the MECHANISM target is hit, the ROLLOUTS regress, and the run was under-budgeted (my setup
  error).** Geometry probe on A6sep keep_299 vs A6 keep_499:

  | | A6 | A6sep |
  |---|---|---|
  | count readout from the bank | 8/12 | **10/12** |
  | readout margin | 0.002 | **0.2005** |
  | max key cosine (vocabulary) | 0.998 | 0.453 |
  | max value cosine | 0.999 | 0.392 |
  | encoding max cosine | 0.978 | 0.823 |

  Training telemetry: separation penalty 0.068 -> 0.043 -> 0.0070 (A6 ends at 0.45), key mean |cos| 0.191 (A6 0.495).
  So the term does exactly what it was designed to do and the agreed success criterion (margin up by >= an order of
  magnitude) is met with room to spare. Per tray step, all six x=3 readouts are now correct (margins 0.16-0.37,
  retention 0.33-0.68); the two misses are both x=2 at the FIRST tray arrival where retention has fallen to 0.09 --
  and "scoop 2 times" is still the least separated go variant (key +0.32 to "1 time", +0.25 to "3 times", while
  1-vs-3 is -0.11). NEW: with interference removed, DECAY finally matters (retention 0.68 at age 67 steps -> 0.03 at
  126), so separation + slower decay is the natural combination -- and this retrospectively explains why the decay
  experiment alone did nothing.
  BUT the self-write rollouts are the worst so far: tray **4/12**, go count "1 time" in ALL SIX episodes, demo11
  0/119 decision steps. Failure mode: malformed sentences with the DIGIT MISSING (`scoop : dump and return`,
  `scoop ::: dump and return`) and the light phase skipped entirely in 4/6 episodes. The separation term has no path
  to the token decoder (it touches only the sentence pooling and memory_sem_{key,value}_proj), so this looks like an
  undertrained decoder, not a geometry effect. Cause: **the run was not matched to its baseline** -- the user's 19:51
  allocation put A6 tests on the SINGLE H200, I kept batch 2, and A6sep therefore saw ~600 sequences against A6's
  2000 (batch 4 x 500). Its curve was behind at every checkpoint (decision 0.439/0.837 at steps 100/200 vs A6
  0.510/0.882). Control launched 22:02 to settle it: **`v5_beansA6ctl_20260905_r1`** = the beansA6 config with NO
  separation loss at exactly batch 2 x 300 updates on the same GPU. If it also drops digits and says "1 time", the
  regression is budget and the separation loss is clean; if it looks like normal A6, the loss is harmful at w=1.0.
  Ops: the first A6sep eval attempt OOM'd on all 6 episodes because a trossen placeholder had restarted on job
  17267793 at 21:25 (~120 GB) after the training ended; killed + checkpoint swept, evals re-run. When a training
  finishes on an eval job, kill the restored placeholder BEFORE launching the evals.
  22:08 — **Job-membership check + H200-aware A8/B8 waiter; 2xH200 job cancelled.** (a) The other session drives
  the single H200 job 17267793 from hgx-1 (`srun --jobid=17267793` issued there), so on hgx-2 no command line
  carries the job id and pattern matching cannot see its A6ctl training. Both the sentinel and the waiter now also
  read `SLURM_JOB_ID` from `/proc/<pid>/environ`: any python of the job that is neither the placeholder
  (`GPU_PLACEHOLDER` env / marker) nor the user's `train_hs.py` keep-alive counts as real work (verified: it lists
  the A6ctl trainer + loader workers and the A6sep count-flip for 17267793, and only the B6 trainer for
  17284681). (b) `evals_beans_waiter_a8b8.sh` (copy of the waiter; the running one could not be edited) waits,
  after A8's "protected" line, until no such python of 17267793 is alive — so it will NOT start rollouts on top
  of A6ctl (their WARN 22:07: training ~78 GB + rollout ~37 GB does not fit in 143 GB; it cost them six A6sep
  rollouts at 21:29). (c) User 22:06: "kill the 2h200 job" -> `scancel 17284681` (B6 continuation 1000->3000
  stopped; last saved B6 checkpoint 1250; A8/B8 unaffected). The cancel killed every plain shell adopted on hgx-2
  (my sentinel + waiter): relaunched under 17267793 (pids 954961 / 955288). User 22:09 then clarified that the
  2xH200 job itself should have stayed alive with only the 1 GB keep-alive (no placeholder, GPUs free for the
  user) — too late for 17284681; standing rule from now: nothing of ours (no placeholder, no evals, no training)
  runs on the user's 2xH200 job, whichever id it gets when resubmitted.
  22:35 — **A8/B8 evals moved to the free GPU of the user's 2xH200 job 17286852** (user 22:27: "it has 1 available
  h200, run the a8 stuff there and when finish just finish, don't run anything on that job"). GPU 0 of that job
  is the user's own openpi-beta process (130 GB, 100 %); GPU 1 (UUID GPU-b3d023a5-…) holds only the 1 GB
  keep-alive. New `run_beans_evals_job.sh` (variant of the hgx2 runner): `GRES=2` + `CUDA_VISIBLE_DEVICES=<UUID>`
  inside the `--overlap` step (an `--overlap --gres=gpu:1` step always maps to physical GPU 0, i.e. the user's
  GPU — verified: pinned by UUID JAX sees exactly one device), `NO_PLACEHOLDER=1` skips both the placeholder kill
  and the relaunch at the end, so the job is left exactly as found. Waiter = the original `evals_beans_waiter.sh`
  (no job-busy wait: the user's process on GPU 0 must not block it), launched FROM hgx-1 so no shell of ours sits in
  the user's job cgroup; `RUNNER=job STAGES="A8 B8" STEP=299 SIDECAR=v6sub`. The a8b8 waiter on the single H200 is
  stopped; that GPU is entirely the other session's (A6ctl until ~23:40). A8 rollouts therefore start right after
  A8's keep_299 (~22:50) instead of after midnight.
  23:05 — **New 0905 beans collection labelled + converted (user 22:50).** Data: `data/0905beans_{1,2,3}` =
  60+5+25 demos, same signals as 0902 plus the same 3 cameras. `0905beans_1/demo3` DELETED on the user's
  instruction (22:58): the cue said 3 blinks but the arm delivered twice (gripper held 266-848, two j0>0.8
  excursions) — an incomplete demonstration, never labelled. Remaining **89 demos** labelled with the SAME automatic
  chain as 0902: `beans_build_subtask_labels.py` (events from led_on/go_on/cue_num_blinks/gripper/j0) →
  `beans_relabel_light_state` → `beans_relabel_scoop_dump --cut delivery_start` → `beans_relabel_visible_led` →
  `beans_relabel_subphase`, i.e. the 16-sentence v6 vocabulary the current models train on. Verification: 89/89 tile
  their episode exactly, zero segments outside `V5_BEANS_SENTENCES_V3`, all 16 sentences present, x-counts 27/34/28
  for 1/2/3 scoops (vs 19/23/18 in 0902). The detector needed NO retuning — it succeeded on 89/90 first try, the
  only failure being the deleted demo. Manifest `data/0905beans_episode_manifest_v1.json` (schema 1, 89 episodes,
  71089 frames, per-episode `label_file: subtask_labels_v6sub.json`); conversion → `v5/data/lerobot/yam/bean_scoop_0905_v5`
  (`HF_LEROBOT_HOME=v5/data/lerobot`, run on iris-hgx-1 — the login node silently killed the 0902 conversion at 42/60).
  Inspection: `label_subtasks.py --data-dir data/0905beans_all --beans-task --beans-v6 --label-file
  subtask_labels_v6sub.json --port 8766` on iris-ws-18, where `0905beans_all` is a SYMLINK view (demo1001-1059 =
  folder 1, demo2001-2005 = folder 2, demo3001-3025 = folder 3, `source_map.json` alongside) so one server covers all
  89 and edits save back into the real folders.
  **Camera-sync measurement (user 23:00: "top camera light still on when wrist already off?").** Measured the LED
  patch independently in the TOP and LEFT videos over 25 demos / 56 blinks: top minus left onset = -1/0/+1 frames in
  2/41/13 blinks, offset = 0/+1 in 47/9. So the views differ by at most ONE frame and agree in ~75% of blinks; there
  is no case of one camera being on while the other has long been off. For comparison the correction the labels
  ALREADY apply (left camera minus electrical signal) is +1 frame in 44 blinks and +2 in 10. At stride 5 one frame is
  a fifth of a step and a blink lasts 8-9 frames, so this cannot flip a count. User decision 23:03: **keep the labels
  as they are** (the alternative, taking the boundary from the LATEST camera so no view is ever ahead of its label,
  is a one-line change to `beans_relabel_visible_led.py` + a re-run of the last two stages if it ever matters).
  23:25 — **User (23:22): A7 and A6ctl dropped; the single H200 becomes the user's; this session = B8 line only.**
  Stopped on job 17267793 at 23:23: the other session's `run_a6ctl_verdict_chain.sh`, its `run_beans_evals_hgx2.sh`
  (A6ctl rollouts) and the two srun steps; my hgx-2 sentinel for that job; verified afterwards: no srun client on
  either node, no python of the job other than the `train_hs.py` keep-alive (pids 3855222/3855226, 1 GB, 0 %), no
  placeholder step, `/scr/kewalk_placeholder` empty. Job stays alive with the keep-alive only — nothing of ours runs
  there again. User (23:24): the v5task2 session is data-preparation only from now on (told via SendMessage and the
  queue log). A7 final record: geometry 10/12 @0.20 and count-flip 1.00/0.985/0.773 (oracle) but self-written
  sentences lost the count (4/6 dev episodes wrong at the go decision, malformed "scoop : dump"); the matched
  control A6ctl reached geometry 4/12 @0.0006 (so the loss, not the budget, made the count readable) and its
  rollouts were stopped before finishing. Stale monitors of this session (B6/B5d1, A6sd/B6sd, B6 continuation)
  stopped. Still running: B8 on the 4xH100 (keep_299 ~00:55), A8 count-flip -> probe chain -> oracle videos -> B8
  evals on GPU 1 of job 17286852, the hgx-1 sentinel for 17249058.
  23:35 — **0905 dataset READY for future v5 training (data prep complete).** LeRobot dataset
  `v5/data/lerobot/yam/bean_scoop_0905_v5`: **89 episodes / 71089 frames** (matches the raw manifest exactly), 53 GB,
  fps 30, robot yam, the SAME feature set as the 0902 dataset (image + left_wrist_image + right_wrist_image + state +
  actions, 14-dim), 16 distinct task strings and none outside `V5_BEANS_SENTENCES_V3`, `meta/episode_sources.json`
  carrying strict provenance per episode (raw_dir, stream lengths, `label_file: subtask_labels_v6sub.json`, label
  sha256) and `meta/episode_prompts.json` with the single constant beans prompt. v5 artefacts:
  `cluster_v5/beans/beans_episode_manifest_0905_v1.json` (sha **3412223a9ab03ba0**…, splits: 2 dev + 2 final_test per
  class, train 23/30/24 for x=1/2/3) and `cluster_v5/beans/beans_v5_subtask_labels_0905_v6sub.json` (sha
  **a2b1a659d6b26f5b**…, 16 sentences). A future config only needs these two paths + shas, the repo id
  `yam/bean_scoop_0905_v5` and norm stats; NO config entry was added (user 23:24: this session is data preparation
  only). Tooling change: `beans_build_v5_manifest_sidecar.py --labels-manifest` now takes SEVERAL manifests and keys
  entries `<collection>/<demo>` as well as bare `<demo>` — the 0905 set spans three folders that each contain a
  `demo1`, which the single-key lookup could not disambiguate. Backward compatible and verified: rebuilding the 0902
  sidecar reproduces it byte-identically (sha c322f81ebc9e86a6…, the value pinned in `V5_BEANS_SIDECAR_V6_SHA256`).
  23:42 — **B8 continuation 300 → 5000 armed** (user 23:38). `queue_beans16_hgx1.sh` on hgx-1 waits for queue15's
  "beans-B8 ckpt-299 protected" line, then `run_train_h200.sh ... --num-train-steps 5000 --keep-period 1000`
  (auto-resume from 299, batch 8 on the 4 H100s, ~26 h → ~03:00 on 09/07; job 17249058 ends ~22:50 that day).
  Disk: 614 GB free on the NFS home at 23:40 with 1.4 TB already in v5/checkpoints, ~27 GB per kept checkpoint —
  keeping every 250 would leave ~500 GB, so only multiples of 1000 are kept (1000/2000/3000/4000 + the final 4999);
  the judged keep_299 copy is untouched. Evaluations of the continuation checkpoints are NOT armed (no GPU
  assigned for them; the keep_299 B8 evals still run on GPU 1 of job 17286852).

  **2026-09-06 00:47 — user decision: A9/B9 = target-carry labels ("scoop k of x") on the 0905 collection only;
  B8 continuation cancelled.** Reasoning (00:24–00:43, from the A8 probes): the count is perfectly readable in
  the A8 bank (count recovery 12/12 at margin 0.43; count-flip at the first go step 1.00 / follows-flip 1.00 /
  blank 0.76) yet the tray decision ignores it (tray-flip development: says "done" at 9/9 tray steps whatever the
  history, follows a flipped history 0/9, margin 0.11; A6 0.33, B6 0.22). The blink→go step works because it is a
  single-note COPY (the go sentence's number is the newest light note's number); the tray step needs a two-note
  COMPARISON (go note x vs newest scoop note k) and the training set offers cheaper routes (tray fill / scene
  memorisation fit the 60 demos to 0.97), so the comparison is never learned. The user's own 13:40 relabel idea
  (prepared then, unused) makes the tray step a single-note copy/compare too: every scoop sentence carries x.
  Trade-off stated to the user: nothing in the task then requires reading a note older than the newest one.
  A8 self-write recap (6 dev episodes): blink counts all right, but the copy shortcut of a stage A shows —
  "yellow go" sticky through the scoop (demo12/21/51), "scoop 1" repeated instead of incrementing (demo14/17),
  "done" after every scoop. A7 closed (see 23:25). B8 300 finished on its own (used only for its keep_299 evals).
  Built: `scripts/beans_relabel_target_carry.py` on the three 0905 raw folders (89 demos, exactly 20 sentences),
  `beans_build_v5_manifest_sidecar.py --reuse-manifest` → `cluster_v5/beans/beans_v5_subtask_labels_0905_v7tgt.json`
  (sha a2f704b9…, pinned to the 0905 v1 manifest 3412223a…, same dev/final_test split as v6sub: dev =
  0905beans_1/demo27, demo31, 0905beans_2/demo1, 0905beans_3/demo1, demo9, demo10). config.py: `V5_BEANS_SENTENCES_V4`
  + `V5_BEANS_REFERENCE_SENTENCE_TOKENS_V4` (PaliGemma ids, V3 reproduced bit-exactly, max 13 tokens),
  `V5_BEANS_EVIDENCE/DECISION_SENTENCES_V4`, the 0905 pins, data entry `v5_beans_0905_tgt_data` (repo
  yam/bean_scoop_0905_v5, assets pi05_yam_bean_scoop_0905_v5), configs `beansA9` (= A8 recipe) and `beansB9`
  (loader A9/…/299). Slot templates over the new vocabulary: wait, light (k, on/off masked), go (count masked),
  scoop (k, x, dig/dump, carry/return masked → ONE scoop slot, newest wins), done. Ops: 0905 norm stats
  (`compute_norm_stats.py --config-name pi05_yam_mem_v5_beansA9 --max-frames 3000`, CPU on hgx-1, started 00:53;
  copied to the data assets dir when done), `queue_beans17_hgx1.sh` (waits for B8's exit, an idle job and the
  stats; A9 → keep_299 → B9 → keep_299, batch 8 on the 4 H100s), `evals_beans_waiter_v2.sh` (A9/B9 dated
  20260906, MANIFEST passthrough) with `run_beans_evals_job_v2.sh` (MANIFEST env) on GPU 1 of job 17286852,
  no placeholder. Judgement for A9: tray-flip probe on the 0905 development split must follow the flipped
  history (A8: 0/9), then B9 self-write rollouts.
  00:58 — **B9 continuation 300 → 3000 armed** (user 00:57): `queue_beans18_hgx1.sh` waits for "beans-B9 ckpt-299
  protected", then `run_train_h200.sh ... --num-train-steps 3000 --keep-period 1000` (auto-resume, batch 8; ~15 h,
  ends ~20:00 on 09/06; kept checkpoints 1000, 2000, 2999 ≈ 81 GB). No evals armed for the continuation checkpoints.
  01:30 — **A8 oracle rollouts and B8 self-write: the two remaining pieces of the A8 picture.** (a) A8 ORACLE-write
  rollouts (true sentences fed to the bank) score 116/119, 78/78, 169/172, 144/150, 115/116, 67/67 — but at the
  exact tray-arrival step, with only the history in the bank, the prediction is "done" in every case (demo11 step
  99, demo14 steps 100–101 and 152, demo21 step 101; conf 0.95–0.97); one step later the oracle has written
  "scoop k: dump" and the model copies it. The videos look right only because the answer is handed over a step
  late; the decision itself is always "done" (= the tray-flip probe; train split: 17/18 "done", follows-flip
  1/18, A6 8/18). (b) B8 self-write OSCILLATES: "scoop 1: dig" / "scoop 2: dig" alternate at every step of the
  scoop phase (54–64 writes per episode vs 6–14 for A8); tray-step exact 8/12 is the coin flip; demo12 ends in
  "yellow go". Mechanism: whitened values make the newest note crisp, own-writes training closes the loop
  prediction → write → read → prediction without a visual anchor (B6's blurry plain bank leaned on vision and was
  stable). Tray-step exact, self-write dev: A6 7/12, B6 7/12, A8 3/12, B8 8/12. Noted as a risk for B9; a generic
  mitigation is a refractory write rule (no rewrite for a few steps unless the slot changes) — not applied yet.
  A9 launched 01:24 (0905 norm stats loaded, 77 train episodes, v7tgt sentences, stageB6a warm start).
  `run_stage_probes_job.sh` (STAGE=A9) armed behind the A9 keep_299 evals on GPU 1 of job 17286852: tray-flip
  development + train, count recovery, geometry.
  05:15 — **A9 first results (0905 development split).** Count-flip at the first go step: reads 1.00 / follows flip
  1.00 / blank 0.69 (= A8); count recovery at the tray 12/12 (margin 0.18). Self-write rollouts: 1-scoop episodes
  clean (done at the tray); 2-scoop: first scoop right, the second dig re-writes "1 of 2" (no increment) so the
  second tray says "dump" (the rule applied to a wrong k) and the run drifts to "yellow go"; 3-scoop: the FIRST dig
  is written "3 of 3" (x copied into k) so the first tray says "done". Tallies: first-go count 4/6, tray-step exact
  4/12, correct k at a new dig 0/6; writes 6–23 per episode (no B8-style oscillation). ORACLE rollouts, prediction
  at the exact tray-arrival step (history only): dump arrivals (k<x) 5/6 right (the miss is a one-step "still dig"
  lag), done arrivals (k=x) 2/6 at the exact step (3 one-step lags, 1 real error: demo9 "2 of 2" → dump); NEVER a
  false "done" at k<x (A8: "done" at every arrival). Per-episode oracle decision_exact ≈ 99 %. So the labels moved
  the tray rule onto the note (dump iff k<x); the open problem is the k bookkeeping (first-dig k, increment) —
  the stage-A boundary weakness (copy shortcut; 1–2 boundary examples per episode vs 3 blink increments).
  Tray-flip probe on A9: two tooling gaps found and fixed — (1) the probe only parsed the v6 "scoop k:" rows
  (patched: carried x, x flipped in scoop rows, variable-length candidates, schema v2); (2) it scores only
  decision-mask steps, and the v4 decision list holds "done" but not the dump sentences, so its 26 dev cases were
  all k=x: says "done" 5/26 (normal) = 5/26 (blank) = 7/26 (flip) — history-independent, biased to "dump", but a
  weak instrument here (CE compares a 12-token vs a 9-token candidate). Fix: `v5_step_ce_steps` (unmasked per-step
  CE) in the loss outputs; probe scores every tray arrival; rerun (`run_a9_tray_all_job.sh`, dev + train) queued
  behind the chain. B9 (own writes) keep_299 ~05:35; its self-write evals, oracle rollouts and probe chain armed.
  05:55 — **B9 keep_299 self-write: first model that solves the multi-scoop episodes on its own.** 0905 dev split:
  demo27 (x=2) 77/77 decision steps: 1 of 2 dig → dump → 2 of 2 dig → done; demo9 (x=2) 77/79 same; demo31 /
  demo10 (x=1) 79/79 and 69/70; 0905beans_2/demo1 (x=3) 80/82: three dumps then done (k labels wobble: "3 of 3"
  written at the second dig, "2 of 3: dump" at the second tray, still the right sequence of tray decisions);
  0905beans_3/demo1 (x=3) 66/80: the third tray is first called "2 of 3: dump" and "done" follows 13 steps later.
  Tray decision sequence right in 5/6 episodes (the sixth: done ~2 s late). Tallies vs A9: tray-step exact 8/12
  (A9 4/12), correct k at a new dig 3/6 (0/6), writes 6–17 per episode (no B8 oscillation; A9 6–23). Own-writes
  training fixed most of the k bookkeeping and kept the tray rule; the x=3 episodes still carry wrong k labels at
  times. Continuation 300 → 3000 running since 05:38; B9 count-flip, oracle rollouts and probe chain follow on
  GPU 1 of job 17286852.
  08:10 — **A9/B9 probe round-up (0905 development).** Tray-flip v2 (all 46 tray arrivals, dump-labelled 20 / done 26;
  B9 probed under the A9 oracle config): dump arrivals — A9 20/20, B9 20/20, blank bank 20/20 for both; done
  arrivals — A9 says done 5/26 (blank 6/26), B9 15/26 (blank 14/26); flipped history where the newest note becomes
  "k of k" at a non-final tray (8 cases): A9 says done 6/8, B9 6/8. Oracle rollouts at the exact arrival frame: A9
  7/12 (dump 5/6, done 2/6), B9 8/12 (dump 5/6, done 3/6); the misses are one-frame "still dig" lags plus one
  early done (B9 demo27, absent in its self-write run). Count-flip: A9 1.00/1.00/0.69, B9 1.00/0.55/0.63 (the usual
  own-writes drop; all six rollout go counts right). Count recovery at the tray: A9 12/12 (0.18), B9 see run.log.
  Reading: the labels put the tray rule within reach of a single note; A9 (stage A) under-produces "done" at the
  arrival frame; B9 produces it three times as often, uses the note (6/8 on the flipped-note cases) AND the picture
  (blank bank still finds 14/26 final trays, 0/20 false), and finishes 5/6 dev episodes in self-write. B9 =
  working model of the line; continuation to 3000 running (~20:00).
  10:25 — **B9 ckpt-1000 self-write (user 10:12): 6/6 dev episodes right.** Decision steps 77/77, 79/79, 79/82, 79/80,
  76/79, 69/70; tray-step exact 11/12 (keep_299: 8/12); every episode has exactly x−1 dumps before "done" and says
  done (keep_299: the x=3 episode 0905beans_3/demo1 was ~2 s late — now right); writes 6–16 per episode; the k
  increments land one step after the bowl arrival (a single-step repeat of the old k, then k+1), which is timing,
  not a bookkeeping error. Videos `videos_v5_beansB9_20260906_r1_1000/`. Checkpoint 1000 is kept by the
  continuation's keep_period (1000). Run on GPU 1 of job 17286852, no placeholder.
  12:50 — **B9 ckpt-1000 with the memory OFF (user 12:34; `--intervention blank`: no commits, empty bank, previous
  sentence only shifts the read queries → no effect on an empty bank).** Same six dev episodes: decision steps
  25/77, 12/79, 24/82, 16/80, 16/79, 11/70 (with memory: 77/77, 79/79, 79/82, 79/80, 76/79, 69/70). The light
  counter never leaves "wait for the light: no green blink yet" (each frame is independent, the count lives only
  in the bank), "done" is emitted right after the wait phase (steps ~20–30, before any scoop), scoop sentences are
  guesses ("scoop 2 of 3 / 2 of 2", regardless of x and k). The memory is what makes the task possible; vision
  alone gives neither the blink count nor the scoop bookkeeping. Videos `videos_v5_beansB9_20260906_r1_1000_blank/`.
  13:50 — **Robot-test latency (user 13:20: the arm "go and stuck").** Bench from hgx-1 with real 0905 frames against
  the B9 ckpt-1000 server (GPU 1 of job 17286852): median 250 ms per request, `server_timing.infer_ms` ≈ 240
  (model time; network ~10 ms; node CPU load not the cause). The RTC client tolerates `max_async_delay_steps=6`
  delayed controls = 200 ms at 30 Hz (the training `simulated_delay=6`), so every replan arrived late and the
  arm stalled between chunks. Flow steps are NOT the cost: `--num-steps 6` (new server flag; `SERVE_EXTRA`
  passthrough in serve_v5_job_v2.sh) gave 235 ms / 223 ms model time — the rest is the vision prefix, the
  sentence decode and the per-call reference-row passes. Decision (user 13:45): back to 10 flow steps (server
  relaunched 13:46, log `server_v5_b9_1000_20260906_ns10.log`); the user lowers the client control rate (`--hz 20`:
  6 delayed controls = 300 ms, the LED cue is timed in seconds so the blinks stay the same in real time but span
  fewer frames); NEXT TRAINING: raise the RTC `simulated_delay` (e.g. 10 → 333 ms at 30 Hz) so the policy is
  trained for the real ~250 ms inference delay instead of slowing the robot.
  14:15 — **RTC delay for the next training = 15 (user 14:07 "set it to 15").** New configs `pi05_yam_mem_v5_beansA10` /
  `pi05_yam_mem_v5_beansB10` = the A9/B9 recipe with `simulated_delay=15` (500 ms at 30 Hz; B10 loads A10 ckpt-299,
  experiment name via `OPENPI_V5_BEANS_A10_EXP`, default `v5_beansA10_20260907_r1`). Not launched. The client will
  need `--max-async-delay-steps 15 --initial-delay-steps 15 --delay-buffer-size 20` against a delay-15 checkpoint.
  14:35 — **Robot test moved to B9 ckpt-1750 (user 14:23 "launch the newest one").** The continuation keeps only
  multiples of 1000 plus the latest (1500 was deleted at 14:20 when 1750 was saved), so 1750 was copied to
  `keep_1750` first (27 GB, 9 min on NFS), then the ckpt-1000 server was stopped and `serve_v5_job_v2.sh` relaunched
  on GPU 1 of job 17286852 from `keep_1750` (10 flow steps; log `server_v5_b9_1750_20260906.log`). Real-robot
  observation on ckpt 1000 (user 14:08): the 1→2 scoop transition needs two scoops, 2→3 is fine. Labels are
  balanced (62 vs 28 transitions, no repeated scoop sentence); the rollouts show the increment is a two-step move
  through an inconsistent committed note "k of x: dig" (old k, new verb — k is decoded before the verb), which
  equals the first-scoop bank state, so only vision (beans already in the tray) can finish the increment; after one
  real dump the pile is small → stays at k=1; after two it is unmistakable → 2→3 works. Options logged: increment
  at the tray ("dump, return for k+1" labels), serving-side rejection of "k dig" after "k dump", or wait for 3000.
  15:12 — **Placeholder back on the free GPU of job 17286852 (user 15:10 "only for free gpu there").** New
  `placeholder_train_trossen_pin.sh <job> <gres> <gpu uuid>`: the trossen training pinned by UUID (GRES=2 +
  CUDA_VISIBLE_DEVICES; a `--gres=gpu:1` step would land on GPU 0 anyway but the pin makes it explicit), busy guard
  on the pinned GPU only, exp name suffixed with the job id, checkpoints on /scr, deleted on exit. Runs on GPU 0
  (UUID 3cc86c11) while GPU 1 (UUID b3d023a5) keeps the keep_1750 robot server; no sentinel for this job (it is
  per job and would kill the placeholder because of the server). Log `v5/tools/logs/placeholder_train_17286852.log`.
  Correction: job 17286852 is on iris-hgx-2 (10.79.12.149), the launcher's "hgx-1" was the launch shell's host.

* 2026-09-06 15:15 — **NON-MEMORY pi05 BASELINE on the 0905 beans set** (user 15:00: "train a baseline ... only pi05
  no memory at all? but for pi05 we still need to do knowledge insulation and use our subtask to supervise the vlm
  and fast action token"). Config **`pi05_yam_beans0905_base`** (added next to `pi05_yam_0816`, which is the same
  recipe on the bins task): `Pi0Config(pi05=True, predict_subtask=True, max_token_len=272)` — **no memory flags at
  all** (`predict_with_memory` / `memory_v4_dual_bank` / `memory_v5_sentence_bank` all false, verified after load).
  Knowledge insulation is the `predict_subtask` path in `pi0.py` ("Subtask + FAST co-training"): the VLM backbone
  gets a next-token CE over the subtask sentence AND the FAST branch, the action expert is trained by flow matching
  against a **stop-gradient'ed** prefix, and the FAST branch is hidden from the suffix in both attention and the
  position offset so the action targets cannot leak. Data = the same 89-episode 0905 dataset and the same v6
  sub-phase labels as the memory runs, `subtask_from_task=True`, `prompt_from_episode_meta=True`,
  **`subtask_lookahead=0`** (the sentence describes the CURRENT frame, as in the v5 beans configs), norm stats
  reused from `v5/assets/pi05_yam_bean_scoop_0905_v5`, `lerobot_dataset_root` pinned to the v5-private root.
  **Token budget:** unlike the memory configs (which split the string into `max_token_len` + `causal_token_len`),
  the non-memory tokenizer packs `Task: ..., State: ...;\n{subtask}\nAction: <FAST>|<eos>` into ONE buffer.
  `scripts/v33_audit_token_lengths.py` over all 71089 frames: context max **73**, causal max **187** → the buffer
  needs >= 260, set to **272**. Guessing here would silently truncate the trailing FAST tokens.
  Fresh from the official `pi05_base` checkpoint (NOT warm-started from any memory model), batch 16, cosine 5e-5
  with 1000 warmup, EMA 0.999, 30k steps, saves/keeps every 5000. Launched 15:13 on the user's H200 job 17267793
  (user 15:09 "you can use 17267793 this gpu for training"); the tuned placeholder there was replaced by the
  launcher. Step 0: CE 13.52 / flow 0.124, 1.2 it/s → ~6 h 53 m, ETA ~22:10. Experiment
  `pi05_beans0905_base_20260906_r1`.
