#!/usr/bin/env bash
# Run ON iris-hgx-1 (shell in job 17249058, ends 09-07 -- the 18:34 lesson), targeting the H200 pair of job 17284681
# whose GPUs went idle when the robot server stopped at 19:48. A6sep only; B6sep follows on whichever job is free.
# queue13 shell while its training step kept running). 2026-09-05 19:40: the SENTENCE-SEPARATION fix for the measured
# cause of the tray failure (README §8 18:50: sentences differing only in a count are written at cosine 0.996-0.999,
# so an old note's count is a ~0.002 residual). A6sep (label writes + separation loss, warm start B6a keep_499)
# -> keep_299 -> B6sep (own writes, retry, separation loss kept on) -> keep_299. 300 updates each on the 4xH100 job
# 17249058 (batch 8). Evaluations: evals_beans_waiter.sh STAGES="A6sep B6sep" STEP=299 SIDECAR=<v6sub> on the H200.
# Matched baselines on the same labels and step budget: beansA6sd/B6sd (300, decay only) and beansA6/B6 (500).
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/iris/u/kewalk/memory_project_v5/v5/checkpoints
acfg=pi05_yam_mem_v5_beansA6sep; aexp=v5_beansA6sep_20260905_r1
bcfg=pi05_yam_mem_v5_beansB6sep; bexp=v5_beansB6sep_20260905_r1
log=$diag/queue_beans_hgx1.log
echo "queue14a (A6sep: sentence-separation loss w=1.0 margin 0.3, H200 pair job 17284681) armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C /iris/u/kewalk/memory_project_v5 rev-parse --short HEAD)" >> $log
JOB=17284681 GPUS=2 BATCH=4 bash $cv5/run_train_h200.sh $acfg $aexp
code=$(grep "^exit=" $diag/train_${aexp}_status.log | tail -1); echo "beans-A6sep r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-A6sep r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$acfg/$aexp/299 $ckroot/$acfg/$aexp/keep_299 && echo "beans-A6sep ckpt-299 protected as keep_299" >> $log || { echo "beans-A6sep keep_299 COPY FAILED (stop)" >> $log; exit 1; }
echo "beans-A6sep done; B6sep NOT started by this runner (queue_beans14_hgx1.sh does A6sep->B6sep on the 4xH100)" >> $log
