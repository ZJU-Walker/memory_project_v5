#!/usr/bin/env bash
# Resume of queue_beans13 (2026-09-05 18:38). The 2xH100 job 17267129 was cancelled at 18:34; the queue13 shell and the
# run_train_h200.sh wrapper lived in that job and died with it, but the A6sd srun STEP (job 17249058) kept running.
# This waits for that step to finish, protects ckpt-299 and runs B6sd. Launch it from a shell that lands in a job that
# will outlive it (iris-hgx-1 now lands in 17249058, ends 09-07).
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/iris/u/kewalk/memory_project_v5/v5/checkpoints
acfg=pi05_yam_mem_v5_beansA6sd; aexp=v5_beansA6sd_20260905_r1
bcfg=pi05_yam_mem_v5_beansB6sd; bexp=v5_beansB6sd_20260905_r1
log=$diag/queue_beans_hgx1.log
echo "queue13b (resume: wait for the orphaned A6sd step, then B6sd) armed on $(hostname) $(date '+%m/%d %H:%M')" >> $log
while pgrep -f "train.py pi05_yam_mem_v5_beansA6s[d]" >/dev/null; do sleep 30; done
if [ ! -e "$ckroot/$acfg/$aexp/299/params" ]; then echo "beans-A6sd: no ckpt-299 after the step ended (stop)" >> $log; exit 1; fi
echo "beans-A6sd step ended; ckpt-299 present $(date '+%m/%d %H:%M')" >> $log
cp -r $ckroot/$acfg/$aexp/299 $ckroot/$acfg/$aexp/keep_299 && echo "beans-A6sd ckpt-299 protected as keep_299" >> $log || { echo "beans-A6sd keep_299 COPY FAILED (stop)" >> $log; exit 1; }
JOB=17249058 GPUS=4 BATCH=8 bash $cv5/run_train_h200.sh $bcfg $bexp
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B6sd r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-B6sd r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$bcfg/$bexp/299 $ckroot/$bcfg/$bexp/keep_299 && echo "beans-B6sd ckpt-299 protected as keep_299" >> $log || echo "beans-B6sd keep_299 COPY FAILED" >> $log
