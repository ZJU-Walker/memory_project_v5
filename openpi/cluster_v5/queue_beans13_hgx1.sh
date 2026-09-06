#!/usr/bin/env bash
# Run ON iris-hgx-1 (2026-09-05 17:05, user "in parallel"): the A6/B6 recipe with ONLY the bank decay changed
# (alpha_step 0.001 instead of 0.01; README §8 17:00: the tray "dump vs done" decision reads the go sentence at
# 0.17-0.56 strength, the decisions that work read notes at 0.87-0.90). A6sd (label writes, warm start B6a keep_499)
# -> keep_299 -> B6sd (own writes, retry, delay 0) -> keep_299, on the user's 4xH100 job 17249058 (batch 8;
# 300 updates each, user 17:03 "train to 300 steps not 500 to save time").
# Evaluations: evals_beans_waiter.sh STAGES="A6sd B6sd" STEP=299 SIDECAR=<v6sub> on the H200 job 17267793.
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/iris/u/kewalk/memory_project_v5/v5/checkpoints
acfg=pi05_yam_mem_v5_beansA6sd; aexp=v5_beansA6sd_20260905_r1
bcfg=pi05_yam_mem_v5_beansB6sd; bexp=v5_beansB6sd_20260905_r1
log=$diag/queue_beans_hgx1.log
echo "queue13 (A6sd -> B6sd: A6/B6 recipe with bank decay 0.001, 4xH100 job 17249058) armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C /iris/u/kewalk/memory_project_v5 rev-parse --short HEAD)" >> $log
JOB=17249058 GPUS=4 BATCH=8 bash $cv5/run_train_h200.sh $acfg $aexp
code=$(grep "^exit=" $diag/train_${aexp}_status.log | tail -1); echo "beans-A6sd r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-A6sd r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$acfg/$aexp/299 $ckroot/$acfg/$aexp/keep_299 && echo "beans-A6sd ckpt-299 protected as keep_299" >> $log || { echo "beans-A6sd keep_299 COPY FAILED (stop)" >> $log; exit 1; }
JOB=17249058 GPUS=4 BATCH=8 bash $cv5/run_train_h200.sh $bcfg $bexp
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B6sd r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-B6sd r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$bcfg/$bexp/299 $ckroot/$bcfg/$bexp/keep_299 && echo "beans-B6sd ckpt-299 protected as keep_299" >> $log || echo "beans-B6sd keep_299 COPY FAILED" >> $log
