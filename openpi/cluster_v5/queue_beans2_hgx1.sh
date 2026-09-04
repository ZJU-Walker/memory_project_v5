#!/usr/bin/env bash
# Run ON iris-hgx-1. Replacement for the tail of queue_beans_hgx1.sh once beans-A r1 is already training (15:51):
# the H200 job 17267134 died at 15:55, so the evaluations run on GPU 0 of the 4xH100 job between the stages.
#   wait for beans-A r1 exit -> keep_499 -> A evals (self videos + count-flip, GPU 0) -> B smoke -> B r1 -> keep_499
#   -> B evals -> B continuation toward 3000 (user rule: keep training).
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
echo "queue2 (A exit -> A evals on GPU 0 -> B) armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C /iris/u/kewalk/memory_project_v5 rev-parse --short HEAD)" >> $log
until grep -q "^exit=" $diag/train_${aexp}_status.log 2>/dev/null; do sleep 60; done
code=$(grep "^exit=" $diag/train_${aexp}_status.log | tail -1); echo "beans-A r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-A r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$acfg/$aexp/499 $ckroot/$acfg/$aexp/keep_499 && echo "beans-A ckpt-499 protected as keep_499" >> $log
MODES=self BATCHES=24 bash $cv5/run_beans_evals_hgx1.sh $acfg $aexp 499 >> $diag/chain_${aexp}.log 2>&1
echo "beans-A evals done $(date '+%m/%d %H:%M'): $(tail -1 $diag/videos_${aexp}_499/status.log)" >> $log
bash $cv5/run_train_h200.sh $bcfg $bsmoke --num-train-steps 20 --save-interval 10 --keep-period 10
code=$(grep "^exit=" $diag/train_${bsmoke}_status.log | tail -1); echo "beans-B smoke $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-B smoke failed (stop)" >> $log; exit 1; fi
bash $cv5/run_train_h200.sh $bcfg $bexp
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-B r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$bcfg/$bexp/499 $ckroot/$bcfg/$bexp/keep_499 && echo "beans-B ckpt-499 protected as keep_499" >> $log
MODES=self BATCHES=24 bash $cv5/run_beans_evals_hgx1.sh $bcfg $bexp 499 >> $diag/chain_${bexp}.log 2>&1
echo "beans-B evals done $(date '+%m/%d %H:%M'): $(tail -1 $diag/videos_${bexp}_499/status.log)" >> $log
echo "continuing beans-B toward 3000 updates (user rule: keep training)" >> $log
bash $cv5/run_train_h200.sh $bcfg $bexp --num-train-steps 3000
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B continuation $code $(date '+%m/%d %H:%M')" >> $log
