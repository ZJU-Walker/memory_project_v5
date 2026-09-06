#!/usr/bin/env bash
# Generic stage probe chain (2026-09-06 01:30, from run_a8_probes_job.sh): STAGE=A9 JOB=... GRES=2 GPU=<uuid>. No placeholder ever (user
# 22:27: the free H200 of job 17286852, "when finish just finish"). Waits for the A8 keep_299 rollouts + count-flip
# of the evals waiter to finish on the same GPU, then runs sequentially:
#   1. tray-flip probe (scripts/v5_tray_flip_eval.py) on development and train, alphas 0.01/0.0 -> does the tray
#      dump-vs-done decision follow a flipped/blanked history now that the count is readable from the bank?
#   2. count recovery (scripts/v5_probe_count_recovery.py) on the A8 weights under the A8 encoding -> bank readout
#      margin of the go/light counts (A6 weights: plain 8/12 @0.002, A8 encoding 12/12 @0.33).
#   3. bank geometry (scripts/v5_bank_geometry_eval.py, plain encoder directions) for the record.
#   JOB=17286852 GRES=2 GPU=<uuid> setsid nohup bash cluster_v5/run_a8_probes_job.sh > /dev/null 2>&1 < /dev/null &
set -u
JOB="${JOB:?}"; GRES="${GRES:-1}"; GPU="${GPU:-0}"
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v5; cd "$root/openpi" || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk XLA_PYTHON_CLIENT_PREALLOCATE=false
diag=$root/v5/diagnostics; log=$diag/queue_beans_hgx1.log
STAGE="${STAGE:?A9|B9|...}"; EXP_DATE="${EXP_DATE:-20260906}"
exp=v5_beans${STAGE}_${EXP_DATE}_r1; cfg=pi05_yam_mem_v5_beans$STAGE; ck=$root/v5/checkpoints/$cfg/$exp/keep_299/params
# The tray-flip probe needs an ORACLE-write config: a B stage is probed under its A config (B9 -> A9), as B6 was under A6.
pcfg=$cfg; case "$STAGE" in B*) pcfg=pi05_yam_mem_v5_beansA${STAGE#B};; esac
echo "$STAGE probe chain armed on $(hostname) job $JOB gpu $GPU: waits for the $STAGE keep_299 evals $(date '+%m/%d %H:%M')" >> $log
if [ -z "${NOWAIT:-}" ]; then  # NOWAIT=1 (user 03:43 "do the probe first"): start immediately, sharing the GPU with the rollouts
  until grep -q "^count-flip exit=" $diag/videos_${exp}_keep_299/status.log 2>/dev/null; do sleep 60; done
  sleep 20
fi
step() {  # tag script outdir extra-args...
  local tag=$1 script=$2 out=$3; shift 3; mkdir -p "$out"
  echo "$tag started $(date +%H:%M)" >> "$out/status.log"
  srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:"$GRES" env CUDA_VISIBLE_DEVICES="$GPU" \
    .venv/bin/python "$script" --config-name "$( [ -n "${PCFG:-}" ] && echo $pcfg || echo $cfg )" --params "$ck" --output-dir "$out" "$@" > "$out/run.log" 2>&1
  local rc=$?; echo "$tag exit=$rc $(date +%H:%M)" >> "$out/status.log"; echo "$STAGE probes: $tag exit=$rc $(date '+%m/%d %H:%M')" >> $log
}
PCFG=1 step "tray probe ${STAGE}_keep_299 development" scripts/v5_tray_flip_eval.py $diag/tray_flip_${STAGE}_keep_299_development --split development --alphas 0.01,0.0 --batches 24
step "count recovery $STAGE params $STAGE encoding" scripts/v5_probe_count_recovery.py $diag/count_recovery_${STAGE}params_${STAGE}enc --split development --alphas 0.01,0.0
PCFG=1 step "tray probe ${STAGE}_keep_299 train" scripts/v5_tray_flip_eval.py $diag/tray_flip_${STAGE}_keep_299_train --split train --alphas 0.01,0.0 --batches 24
step "bank geometry ${STAGE}_keep_299" scripts/v5_bank_geometry_eval.py $diag/bank_geometry_${STAGE}_keep_299 --split development --alphas 0.01,0.001,0.0
echo "$STAGE probe chain done on job $JOB; nothing else launched $(date '+%m/%d %H:%M')" >> $log
