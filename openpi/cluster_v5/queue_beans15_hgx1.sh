#!/usr/bin/env bash
# Run ON iris-hgx-1 (2026-09-05 20:50, user "use 4 h100 run in parallel to accelerate"): A8 -> B8 = the A6/B6 recipe with
# SLOT KEYS + WHITENED VALUES (README §8 20:40), 300 updates each, batch 8 on the user's 4xH100 job 17249058. Waits for
# the job's eval/probe steps to drain (B6sd count-flip, the count-recovery test), then keep_299 -> B8 -> keep_299.
# Evaluations: evals_beans_waiter.sh STAGES="A8 B8" STEP=299 on the H200 job 17267793.
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/iris/u/kewalk/memory_project_v5/v5/checkpoints
acfg=pi05_yam_mem_v5_beansA8; aexp=v5_beansA8_20260905_r1
bcfg=pi05_yam_mem_v5_beansB8; bexp=v5_beansB8_20260905_r1
log=$diag/queue_beans_hgx1.log
echo "queue15 (A8 -> B8: slot keys + whitened values, 4xH100 job 17249058, waits for the job's evals/probes) armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C /iris/u/kewalk/memory_project_v5 rev-parse --short HEAD)" >> $log
while pgrep -f "jobid=17249058 .*(v5_heldout_vide[o]|v5_count_flip_eva[l]|v5_probe_[a-z_]*\.py)" >/dev/null; do sleep 30; done
echo "queue15: job 17249058 evals drained; launching A8 $(date '+%m/%d %H:%M')" >> $log
JOB=17249058 GPUS=4 BATCH=8 bash $cv5/run_train_h200.sh $acfg $aexp
code=$(grep "^exit=" $diag/train_${aexp}_status.log | tail -1); echo "beans-A8 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-A8 r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$acfg/$aexp/299 $ckroot/$acfg/$aexp/keep_299 && echo "beans-A8 ckpt-299 protected as keep_299" >> $log || { echo "beans-A8 keep_299 COPY FAILED (stop)" >> $log; exit 1; }
JOB=17249058 GPUS=4 BATCH=8 bash $cv5/run_train_h200.sh $bcfg $bexp
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B8 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-B8 r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$bcfg/$bexp/299 $ckroot/$bcfg/$bexp/keep_299 && echo "beans-B8 ckpt-299 protected as keep_299" >> $log || echo "beans-B8 keep_299 COPY FAILED" >> $log
