#!/usr/bin/env bash
# Run ON iris-hgx-1 (user 2026-09-04 13:54 "stop b6, working on this"). Bean-scoop task, v5 generic data mode.
#   beans-A smoke (20 updates) -> beans-A r1 (500 updates, batch 8, label writes, warm start from B6a keep_499) -> keep_499
#   -> beans-B smoke -> beans-B r1 (own sentences, from beans-A-499, half LR) -> keep_499 -> continue B (user rule
#   2026-09-04 02:02 "keep training") toward 3000 updates while the batteries/videos are built separately.
# 4xH100 job 17178887 ends 2026-09-04 22:46: A ~1.7 h + B ~1.7 h fits when launched by ~17:30.
export HOME=/iris/u/kewalk
export JOB=17178887
export GPUS=4
export BATCH=8
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/iris/u/kewalk/memory_project_v5/v5/checkpoints
acfg=pi05_yam_mem_v5_beansA; aexp=v5_beansA_20260904_r1; asmoke=v5_beansA_20260904_smoke
bcfg=pi05_yam_mem_v5_beansB; bexp=v5_beansB_20260904_r1; bsmoke=v5_beansB_20260904_smoke
log=$diag/queue_beans_hgx1.log
echo "queue beans A->B armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C /iris/u/kewalk/memory_project_v5 rev-parse --short HEAD)" >> $log
bash $cv5/run_train_h200.sh $acfg $asmoke --num-train-steps 20 --save-interval 10 --keep-period 10
code=$(grep "^exit=" $diag/train_${asmoke}_status.log | tail -1); echo "beans-A smoke $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-A smoke failed (stop)" >> $log; exit 1; fi
bash $cv5/run_train_h200.sh $acfg $aexp
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
echo "beans A/B done; continuing B toward 3000 updates (user rule: keep training)" >> $log
bash $cv5/run_train_h200.sh $bcfg $bexp --num-train-steps 3000
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B continuation $code $(date '+%m/%d %H:%M')" >> $log
