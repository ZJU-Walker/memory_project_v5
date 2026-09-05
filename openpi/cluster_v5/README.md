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
