#!/usr/bin/env bash
# Run ON iris-hgx-1 (2026-09-05 10:40, sub-phase scoop sentences): A6 on the 2xH100 (the B3 continuation was stopped at ~2400 for it) (label writes, v5vis sidecar, warm start B6a keep_499) on
# the user's 4xH100 job 17249058 (batch 8; replaces its trossen placeholder through the marker), then keep_499 -> B4
# (own writes) on the same job -> keep_499 -> continuation toward 3000. The B3 continuation keeps the 2xH100 job.
# Evaluations: evals_beans_waiter.sh STAGES="A6 B6" SIDECAR=<v5vis> on the H200 job 17267793.
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/iris/u/kewalk/memory_project_v5/v5/checkpoints
acfg=pi05_yam_mem_v5_beansA6; aexp=v5_beansA6_20260905_r1
bcfg=pi05_yam_mem_v5_beansB6; bexp=v5_beansB6_20260905_r1
log=$diag/queue_beans_hgx1.log
echo "queue10 (A6 -> B6 -> continuation on the 2xH100 job 17267129; evals on the H200) armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C /iris/u/kewalk/memory_project_v5 rev-parse --short HEAD)" >> $log
JOB=17267129 GPUS=2 BATCH=4 bash $cv5/run_train_h200.sh $acfg $aexp
code=$(grep "^exit=" $diag/train_${aexp}_status.log | tail -1); echo "beans-A6 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-A6 r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$acfg/$aexp/499 $ckroot/$acfg/$aexp/keep_499 && echo "beans-A6 ckpt-499 protected as keep_499" >> $log
JOB=17267129 GPUS=2 BATCH=4 bash $cv5/run_train_h200.sh $bcfg $bexp
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B6 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-B6 r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$bcfg/$bexp/499 $ckroot/$bcfg/$bexp/keep_499 && echo "beans-B6 ckpt-499 protected as keep_499" >> $log
echo "continuing beans-B6 toward 3000 updates on job 17267129 (user rule: keep training)" >> $log
JOB=17267129 GPUS=2 BATCH=4 bash $cv5/run_train_h200.sh $bcfg $bexp --num-train-steps 3000
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B6 continuation $code $(date '+%m/%d %H:%M')" >> $log
