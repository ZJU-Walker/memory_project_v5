#!/usr/bin/env bash
# Run ON iris-hgx-1 (2026-09-05 13:55, user 13:29 "switch back to a4 and b4 but train on the new label"): the A4/B4
# NOTE 2026-09-05 17:20: prepared only; the beansA7/B7 config entries were never written (the v7 target-carry idea was set
# aside: it refreshes the target in every sentence, which hides the bank decay instead of fixing it; README §8 17:00).
# recipe (one-step write delay, retry rule in B) on the v7 TARGET-CARRY scoop sentences ("scoop k of x: dig and
# carry" / "dump and return"; scripts/beans_relabel_target_carry.py) on the user's 4xH100 job 17249058 (batch 8, the
# B5d1 control was stopped for it; B6 keeps the 2xH100 as the delay-0 own-writes probe). A7 (label writes, warm start
# B6a keep_499) -> keep_499 -> B7 (own writes) -> keep_499. Evaluations: evals_beans_waiter.sh STAGES="A7 B7" on the H200.
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/iris/u/kewalk/memory_project_v5/v5/checkpoints
acfg=pi05_yam_mem_v5_beansA7; aexp=v5_beansA7_20260905_r1
bcfg=pi05_yam_mem_v5_beansB7; bexp=v5_beansB7_20260905_r1
log=$diag/queue_beans_hgx1.log
echo "queue12 (A7 -> B7: A4/B4 recipe on the v7 target-carry sentences, 4xH100 job 17249058) armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C /iris/u/kewalk/memory_project_v5 rev-parse --short HEAD)" >> $log
JOB=17249058 GPUS=4 BATCH=8 bash $cv5/run_train_h200.sh $acfg $aexp
code=$(grep "^exit=" $diag/train_${aexp}_status.log | tail -1); echo "beans-A7 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-A7 r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$acfg/$aexp/499 $ckroot/$acfg/$aexp/keep_499 && echo "beans-A7 ckpt-499 protected as keep_499" >> $log || { echo "beans-A7 keep_499 COPY FAILED (stop)" >> $log; exit 1; }
JOB=17249058 GPUS=4 BATCH=8 bash $cv5/run_train_h200.sh $bcfg $bexp
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B7 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-B7 r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$bcfg/$bexp/499 $ckroot/$bcfg/$bexp/keep_499 && echo "beans-B7 ckpt-499 protected as keep_499" >> $log || echo "beans-B7 keep_499 COPY FAILED" >> $log
