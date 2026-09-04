#!/usr/bin/env bash
# User 2026-09-04 00:28: "keep training B5/6 regardless of the battery result". The running A6 chain stops on a
# FAIL verdict; this companion (run ON iris-hgx-1) waits for that line and then runs the B6a part itself.
# If the chain passed (its own "B6a smoke" line appears) there is nothing to do.
export HOME=/iris/u/kewalk JOB=17178887 GPUS=4 BATCH=8
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/iris/u/kewalk/memory_project_v5/v5/checkpoints
bcfg=pi05_yam_mem_v5_stageB6a; bexp=v5_stageB6a_20260903_r1; bsmoke=v5_stageB6a_20260903_smoke
log=$diag/queue_a6_hgx1.log
until grep -qE "A6 failed the bar|B6a smoke|A6 (smoke|r1) failed" $log 2>/dev/null; do sleep 60; done
if ! grep -q "A6 failed the bar" $log; then echo "B6a regardless: not needed ($(date '+%m/%d %H:%M'))" >> $log; exit 0; fi
while pgrep -f "v4_side_flip_eva[l]|v4_stage2_eva[l]" >/dev/null; do sleep 15; done
echo "B6a regardless of the A6 verdict (user rule) $(date '+%m/%d %H:%M')" >> $log
bash $cv5/run_train_h200.sh $bcfg $bsmoke --num-train-steps 20 --save-interval 10 --keep-period 10
code=$(grep "^exit=" $diag/train_${bsmoke}_status.log | tail -1); echo "B6a smoke $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "B6a smoke failed (stop)" >> $log; exit 1; fi
bash $cv5/run_train_h200.sh $bcfg $bexp
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "B6a r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "B6a r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$bcfg/$bexp/499 $ckroot/$bcfg/$bexp/keep_499 && echo "B6a ckpt-499 protected as keep_499" >> $log
bash $cv5/run_videos_hgx2.sh $bcfg $bexp 499 >> $diag/chain_${bexp}.log 2>&1
echo "B6a videos done $(date '+%m/%d %H:%M')" >> $log
STEPS="499 250" bash $cv5/run_batteries_h200.sh $bcfg $bexp >> $diag/chain_${bexp}.log 2>&1
echo "B6a batteries done $(date '+%m/%d %H:%M')" >> $log
