#!/usr/bin/env bash
# Run ON iris-hgx-1 (user 20:10 "Ok do it": tray-cut scoop labels). Replaces queue5's B2 stage:
#   now:            A3 (label writes, v4tray sidecar, warm start B6a keep_499) on the 2xH100 job 17267129 (batch 4)
#   in parallel:    A2 keeps running on the 4xH100 (queue5's child; its keep_499 is protected here when it exits)
#   after A3:       keep_499 -> B3 (own writes) on the 2xH100 -> keep_499 -> continuation toward 3000.
# Evaluations: evals_beans_waiter.sh STAGES="A2 A3 B3" (SIDECAR_A2=v2light, SIDECAR_A3/B3=v4tray) on the H200.
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/iris/u/kewalk/memory_project_v5/v5/checkpoints
a2cfg=pi05_yam_mem_v5_beansA2; a2exp=v5_beansA2_20260904_r1
acfg=pi05_yam_mem_v5_beansA3; aexp=v5_beansA3_20260904_r1
bcfg=pi05_yam_mem_v5_beansB3; bexp=v5_beansB3_20260904_r1
log=$diag/queue_beans_hgx1.log
echo "queue6 (A3 on 2xH100 now; A2 keep_499 when it exits; then B3 + continuation on 2xH100) armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C /iris/u/kewalk/memory_project_v5 rev-parse --short HEAD)" >> $log
( until grep -q "^exit=" $diag/train_${a2exp}_status.log 2>/dev/null; do sleep 60; done
  code=$(grep "^exit=" $diag/train_${a2exp}_status.log | tail -1); echo "beans-A2 r1 $code $(date '+%m/%d %H:%M')" >> $log
  if echo "$code" | grep -q "exit=0"; then cp -r $ckroot/$a2cfg/$a2exp/499 $ckroot/$a2cfg/$a2exp/keep_499 && echo "beans-A2 ckpt-499 protected as keep_499" >> $log; fi ) &
JOB=17267129 GPUS=2 BATCH=4 bash $cv5/run_train_h200.sh $acfg $aexp
code=$(grep "^exit=" $diag/train_${aexp}_status.log | tail -1); echo "beans-A3 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-A3 r1 failed (stop)" >> $log; wait; exit 1; fi
cp -r $ckroot/$acfg/$aexp/499 $ckroot/$acfg/$aexp/keep_499 && echo "beans-A3 ckpt-499 protected as keep_499" >> $log
JOB=17267129 GPUS=2 BATCH=4 bash $cv5/run_train_h200.sh $bcfg $bexp
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B3 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-B3 r1 failed (stop)" >> $log; wait; exit 1; fi
cp -r $ckroot/$bcfg/$bexp/499 $ckroot/$bcfg/$bexp/keep_499 && echo "beans-B3 ckpt-499 protected as keep_499" >> $log
echo "continuing beans-B3 toward 3000 updates on job 17267129 (user rule: keep training)" >> $log
JOB=17267129 GPUS=2 BATCH=4 bash $cv5/run_train_h200.sh $bcfg $bexp --num-train-steps 3000
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B3 continuation $code $(date '+%m/%d %H:%M')" >> $log
wait
