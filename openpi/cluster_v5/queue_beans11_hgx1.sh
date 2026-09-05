#!/usr/bin/env bash
# Run ON iris-hgx-1 (2026-09-05 13:20): control for the B5 rollout collapse. B5d1 = A5 weights + own writes + retry
# rule + the one-step write delay, on the 4xH100 job 17249058 (the B5 continuation was stopped for it) -> keep_499.
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/iris/u/kewalk/memory_project_v5/v5/checkpoints
bcfg=pi05_yam_mem_v5_beansB5d1; bexp=v5_beansB5d1_20260905_r1
log=$diag/queue_beans_hgx1.log
echo "queue11 (B5d1 control: delay 1 + own writes, 4xH100 job 17249058) armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C /iris/u/kewalk/memory_project_v5 rev-parse --short HEAD)" >> $log
JOB=17249058 GPUS=4 BATCH=8 bash $cv5/run_train_h200.sh $bcfg $bexp
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B5d1 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-B5d1 r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$bcfg/$bexp/499 $ckroot/$bcfg/$bexp/keep_499 && echo "beans-B5d1 ckpt-499 protected as keep_499" >> $log || echo "beans-B5d1 keep_499 COPY FAILED" >> $log
