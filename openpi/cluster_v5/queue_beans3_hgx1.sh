#!/usr/bin/env bash
# Run ON iris-hgx-1. Training chain only (beans-A r1 already running since 15:51 on the 4xH100 job):
#   wait for beans-A r1 exit -> keep_499 -> B smoke -> B r1 -> keep_499 -> B continuation toward 3000.
# The evaluations run separately on the 2xH100 job 17267129 (cluster_v5/evals_beans_j17267129.sh).
export HOME=/iris/u/kewalk
export JOB=17178887
export GPUS=4
export BATCH=8
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/iris/u/kewalk/memory_project_v5/v5/checkpoints
acfg=pi05_yam_mem_v5_beansA; aexp=v5_beansA_20260904_r1
bcfg=pi05_yam_mem_v5_beansB; bexp=v5_beansB_20260904_r1; bsmoke=v5_beansB_20260904_smoke
log=$diag/queue_beans_hgx1.log
echo "queue3 (A exit -> B smoke -> B r1 -> continuation; evals on job 17267129) armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C /iris/u/kewalk/memory_project_v5 rev-parse --short HEAD)" >> $log
until grep -q "^exit=" $diag/train_${aexp}_status.log 2>/dev/null; do sleep 60; done
code=$(grep "^exit=" $diag/train_${aexp}_status.log | tail -1); echo "beans-A r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-A r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$acfg/$aexp/499 $ckroot/$acfg/$aexp/keep_499 && echo "beans-A ckpt-499 protected as keep_499" >> $log
bash $cv5/run_train_h200.sh $bcfg $bsmoke --num-train-steps 20 --save-interval 10 --keep-period 10
code=$(grep "^exit=" $diag/train_${bsmoke}_status.log | tail -1); echo "beans-B smoke $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-B smoke failed (stop)" >> $log; exit 1; fi
bash $cv5/run_train_h200.sh $bcfg $bexp
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-B r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$bcfg/$bexp/499 $ckroot/$bcfg/$bexp/keep_499 && echo "beans-B ckpt-499 protected as keep_499" >> $log
echo "continuing beans-B toward 3000 updates (user rule: keep training)" >> $log
bash $cv5/run_train_h200.sh $bcfg $bexp --num-train-steps 3000
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B continuation $code $(date '+%m/%d %H:%M')" >> $log
