#!/usr/bin/env bash
# B8 continuation 300 -> 5000 updates on the 4xH100 job 17249058 (user 2026-09-05 23:38: "after you finish 300 b8 keep
# training till 5000 steps"). Waits for queue15's "beans-B8 ckpt-299 protected as keep_299" line (B8 exit=0 and the
# judged checkpoint copied), then resumes from 299 with the same batch (8 over 4 GPUs) through run_train_h200.sh
# (auto --resume: a numeric checkpoint exists). Checkpoints: every 250 as before, but only multiples of 1000 are
# KEPT (--keep-period 1000): the NFS home had 614 GB free at 23:40 and each kept checkpoint costs ~27 GB, so keeping
# every 250 would leave ~500 GB behind; kept = 1000, 2000, 3000, 4000 + the final 4999 (~135 GB). keep_299 (a copy)
# is untouched; the raw 250/299 dirs may be garbage-collected by orbax. ~4700 updates x 20 s = ~26 h -> ~03:00 09/07.
# Run ON iris-hgx-1:  setsid nohup bash cluster_v5/queue_beans16_hgx1.sh > /dev/null 2>&1 < /dev/null &
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v5; diag=$root/v5/diagnostics; cv5=$root/openpi/cluster_v5
log=$diag/queue_beans_hgx1.log
cfg=pi05_yam_mem_v5_beansB8; exp=v5_beansB8_20260905_r1
echo "queue16 (B8 continuation 300 -> 5000, keep every 1000, 4xH100 job 17249058) armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C $root rev-parse --short HEAD)" >> $log
until grep -q "beans-B8 ckpt-299 protected as keep_299" $log; do sleep 60; done
sleep 30
echo "beans-B8 continuation 300 -> 5000 starting $(date '+%m/%d %H:%M')" >> $log
JOB=17249058 GPUS=4 BATCH=8 bash $cv5/run_train_h200.sh $cfg $exp --num-train-steps 5000 --keep-period 1000
code=$(grep "^exit=" $diag/train_${exp}_status.log | tail -1); echo "beans-B8 continuation (4xH100) $code $(date '+%m/%d %H:%M')" >> $log
