#!/usr/bin/env bash
# Run ON iris-hgx-1 (user 19:12 "kill current b training and beginning our next training"; 19:06 "do 2" =
# light-state sentences). A2 (label writes, v2 light-state sidecar) on the 4xH100 job 17178887 (ends 22:46),
# then B2 (own writes) on the 2xH100 job 17267129 (3-day job; the trossen placeholder training there is killed
# by run_train_h200.sh through its marker and restored by the sentinel afterwards) -> keep_499 -> continuation 3000.
# Evaluations: evals_beans_waiter.sh STAGES="A2 B2" SIDECAR=<v2light> on the H200 job 17267793.
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/iris/u/kewalk/memory_project_v5/v5/checkpoints
acfg=pi05_yam_mem_v5_beansA2; aexp=v5_beansA2_20260904_r1
bcfg=pi05_yam_mem_v5_beansB2; bexp=v5_beansB2_20260904_r1
log=$diag/queue_beans_hgx1.log
echo "queue5 (A2 on 4xH100 -> B2 on 2xH100 -> continuation; evals on the H200) armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C /iris/u/kewalk/memory_project_v5 rev-parse --short HEAD)" >> $log
# A2 needs ~2.5 h at global batch 8 on the 4xH100 (A r1: 500 updates 15:51-18:19, 16.7 s/update). If job 17178887 has
# less than 2h40 left when this runs (Kerberos outage 19:20 delayed the start), A2 runs on the 2xH100 job instead (batch 4).
left=$(squeue -h -j 17178887 -o %L 2>/dev/null); left_min=0
if [ -n "$left" ]; then
  d=0; t=$left; case "$t" in *-*) d=${t%%-*}; t=${t#*-};; esac
  IFS=: read -r h m s <<< "$t"; [ -z "$s" ] && { s=$m; m=$h; h=0; }; left_min=$((d*1440 + 10#$h*60 + 10#$m))
fi
if [ "$left_min" -ge 160 ]; then ajob=17178887; agpus=4; abatch=8; else ajob=17267129; agpus=2; abatch=4; fi
echo "A2 on job $ajob ($agpus GPUs, batch $abatch); job 17178887 has $left ($left_min min) left" >> $log
JOB=$ajob GPUS=$agpus BATCH=$abatch bash $cv5/run_train_h200.sh $acfg $aexp
code=$(grep "^exit=" $diag/train_${aexp}_status.log | tail -1); echo "beans-A2 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-A2 r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$acfg/$aexp/499 $ckroot/$acfg/$aexp/keep_499 && echo "beans-A2 ckpt-499 protected as keep_499" >> $log
JOB=17267129 GPUS=2 BATCH=4 bash $cv5/run_train_h200.sh $bcfg $bexp
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B2 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "beans-B2 r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$bcfg/$bexp/499 $ckroot/$bcfg/$bexp/keep_499 && echo "beans-B2 ckpt-499 protected as keep_499" >> $log
echo "continuing beans-B2 toward 3000 updates on job 17267129 (user rule: keep training)" >> $log
JOB=17267129 GPUS=2 BATCH=4 bash $cv5/run_train_h200.sh $bcfg $bexp --num-train-steps 3000
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "beans-B2 continuation $code $(date '+%m/%d %H:%M')" >> $log
