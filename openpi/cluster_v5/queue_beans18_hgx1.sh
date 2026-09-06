#!/usr/bin/env bash
# B9 continuation 300 -> 3000 updates on the 4xH100 job 17249058 (user 2026-09-06 00:57: "for b9 continue to 3000 once
# finished"). Waits for queue17's "beans-B9 ckpt-299 protected as keep_299" line, then resumes from 299 with the same
# batch (8 over 4 GPUs) through run_train_h200.sh (auto --resume). Checkpoints every 250, only multiples of 1000 kept
# (--keep-period 1000): 1000, 2000 + the final 2999 (~27 GB each); keep_299 (a copy) is untouched.
# ~2700 updates x 20 s = ~15 h -> ends ~20:00 on 09/06 (job 17249058 ends ~22:50 on 09/07).
# Run ON iris-hgx-1:  setsid nohup bash cluster_v5/queue_beans18_hgx1.sh > /dev/null 2>&1 < /dev/null &
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v5; diag=$root/v5/diagnostics; cv5=$root/openpi/cluster_v5
log=$diag/queue_beans_hgx1.log
cfg=pi05_yam_mem_v5_beansB9; exp=v5_beansB9_20260906_r1
echo "queue18 (B9 continuation 300 -> 3000, keep every 1000, 4xH100 job 17249058) armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C $root rev-parse --short HEAD)" >> $log
until grep -q "beans-B9 ckpt-299 protected as keep_299" $log; do sleep 60; done
sleep 30
echo "beans-B9 continuation 300 -> 3000 starting $(date '+%m/%d %H:%M')" >> $log
JOB=17249058 GPUS=4 BATCH=8 bash $cv5/run_train_h200.sh $cfg $exp --num-train-steps 3000 --keep-period 1000
code=$(grep "^exit=" $diag/train_${exp}_status.log | tail -1); echo "beans-B9 continuation (4xH100) $code $(date '+%m/%d %H:%M')" >> $log
