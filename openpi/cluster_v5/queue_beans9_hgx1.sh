#!/usr/bin/env bash
# Run ON iris-hgx-1 (2026-09-05 09:55): the home filesystem hit 100 % right after A5 exited (09:36); queue8 died,
# the A5 keep_499 copy was partial (re-copied by hand) and B5 never launched. This queue = queue8 minus A5:
#   B5 (own writes, delay 0, retry) on the 4xH100 job 17249058 -> keep_499 -> continuation toward 3000.
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/iris/u/kewalk/memory_project_v5/v5/checkpoints
bcfg=pi05_yam_mem_v5_beansB5; bexp=v5_beansB5_20260905_r1
log=$diag/queue_beans_hgx1.log
echo "queue9 (B5 -> continuation on the 4xH100 job 17249058 after the disk-full incident) armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C /iris/u/kewalk/memory_project_v5 rev-parse --short HEAD)" >> $log
JOB=17249058 GPUS=4 BATCH=8 bash $cv5/run_train_h200.sh $bcfg $bexp
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B5 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-B5 r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$bcfg/$bexp/499 $ckroot/$bcfg/$bexp/keep_499 && echo "beans-B5 ckpt-499 protected as keep_499" >> $log || echo "beans-B5 keep_499 COPY FAILED" >> $log
echo "continuing beans-B5 toward 3000 updates on job 17249058 (user rule: keep training)" >> $log
JOB=17249058 GPUS=4 BATCH=8 bash $cv5/run_train_h200.sh $bcfg $bexp --num-train-steps 3000
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B5 continuation $code $(date '+%m/%d %H:%M')" >> $log
