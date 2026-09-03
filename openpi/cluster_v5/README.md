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
