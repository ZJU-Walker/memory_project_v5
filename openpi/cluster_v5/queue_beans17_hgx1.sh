#!/usr/bin/env bash
# A9 -> B9 on the 4xH100 job 17249058 (user 2026-09-06 00:47: target-carry labels "scoop k of x", 0905 collection only,
# A9/B9 naming; B8 continuation cancelled). A9 = A8 recipe (slot keys + whitened values, oracle writes, delay 0),
# B9 = own writes from A9 keep_299; 300 updates each, batch 8 (4 GPUs). Waits for the B8 run to exit and for any
# eval/probe python of the job to drain, and for the 0905 norm stats to be in place. Evaluations: the A9/B9 waiter on
# GPU 1 of job 17286852 (evals_beans_waiter_v2.sh, MANIFEST=0905 v1, SIDECAR=0905 v7tgt).
# Run ON iris-hgx-1:  setsid nohup bash cluster_v5/queue_beans17_hgx1.sh > /dev/null 2>&1 < /dev/null &
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v5; diag=$root/v5/diagnostics; cv5=$root/openpi/cluster_v5
ckroot=$root/v5/checkpoints; log=$diag/queue_beans_hgx1.log
acfg=pi05_yam_mem_v5_beansA9; aexp=v5_beansA9_20260906_r1
bcfg=pi05_yam_mem_v5_beansB9; bexp=v5_beansB9_20260906_r1
stats=$root/v5/assets/pi05_yam_bean_scoop_0905_v5/yam/bean_scoop_0905_v5/norm_stats.json
echo "queue17 (A9 -> B9: target-carry labels, 0905 data, 4xH100 job 17249058) armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C $root rev-parse --short HEAD)" >> $log
until grep -q "^exit=" $diag/train_v5_beansB8_20260905_r1_status.log 2>/dev/null; do sleep 30; done
while pgrep -f "jobid=17249058 .*(v5_heldout_vide[o]|v5_count_flip_eva[l]|v5_probe_[a-z_]*\.py|scripts/train\.p[y])" | grep -v "gpu_placeholder_marke[r]" | grep -q .; do sleep 30; done
until [ -e "$stats" ]; do sleep 30; done
echo "queue17: B8 exited, job idle, 0905 norm stats present -> A9 starts $(date '+%m/%d %H:%M')" >> $log
JOB=17249058 GPUS=4 BATCH=8 bash $cv5/run_train_h200.sh $acfg $aexp
code=$(grep "^exit=" $diag/train_${aexp}_status.log | tail -1); echo "beans-A9 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-A9 r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$acfg/$aexp/299 $ckroot/$acfg/$aexp/keep_299 && echo "beans-A9 ckpt-299 protected as keep_299" >> $log || { echo "beans-A9 keep_299 COPY FAILED (stop)" >> $log; exit 1; }
JOB=17249058 GPUS=4 BATCH=8 bash $cv5/run_train_h200.sh $bcfg $bexp
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B9 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-B9 r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$bcfg/$bexp/299 $ckroot/$bcfg/$bexp/keep_299 && echo "beans-B9 ckpt-299 protected as keep_299" >> $log || echo "beans-B9 keep_299 COPY FAILED" >> $log
