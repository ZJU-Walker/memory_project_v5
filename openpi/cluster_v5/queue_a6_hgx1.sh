#!/usr/bin/env bash
# Run ON iris-hgx-1 (user 2026-09-03 23:12 "do it"). NFS worktree root (no staged copy on this node).
#   wait for the A5-999 params on NFS -> A6 smoke -> A6 r1 (500 updates, batch 8, warm start from A5-999) -> keep_499
#   -> side-flip battery (semantic, 499) -> verdict -> B6a smoke -> B6a r1 (own sentences, from A6-499) -> keep_499
#   -> videos (self, oracle) -> batteries.
export HOME=/iris/u/kewalk
# 23:30: the user's Qwen action-expert TRAINING in the 4xH100 job 17178887 was stopped at their instruction; A6/B6a
# train there with FSDP over the 4 GPUs (GPUS=4, same global batch 2); batteries/videos take one GPU of that job.
export JOB=17178887
export GPUS=4
# The trainer requires global batch % devices == 0 (the 23:33 smoke died on batch 2 / 4 GPUs): one window per GPU,
# global batch 8 (2 windows/GPU) x 500 updates = 4000 windows, twice A5's, in about half the wall time.
export BATCH=8
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/iris/u/kewalk/memory_project_v5/v5/checkpoints
a6cfg=pi05_yam_mem_v5_stageA6; a6exp=v5_stageA6_20260903_r1; a6smoke=v5_stageA6_20260903_smoke
bcfg=pi05_yam_mem_v5_stageB6a; bexp=v5_stageB6a_20260903_r1; bsmoke=v5_stageB6a_20260903_smoke
log=$diag/queue_a6_hgx1.log
echo "queue A6->B6a armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C /iris/u/kewalk/memory_project_v5 rev-parse --short HEAD)" >> $log
until [ -e $ckroot/pi05_yam_mem_v5_stageA5/v5_stageA5_20260903_r1/999/.copied ]; do sleep 60; done
echo "A5-999 params on NFS at $(date '+%m/%d %H:%M'); A6 smoke" >> $log
bash $cv5/run_train_h200.sh $a6cfg $a6smoke --num-train-steps 20 --save-interval 10 --keep-period 10
code=$(grep "^exit=" $diag/train_${a6smoke}_status.log | tail -1); echo "A6 smoke $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "A6 smoke failed (stop)" >> $log; exit 1; fi
bash $cv5/run_train_h200.sh $a6cfg $a6exp
code=$(grep "^exit=" $diag/train_${a6exp}_status.log | tail -1); echo "A6 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "A6 r1 failed (stop)" >> $log; exit 1; fi
cp -r $ckroot/$a6cfg/$a6exp/499 $ckroot/$a6cfg/$a6exp/keep_499 && echo "A6 ckpt-499 protected as keep_499" >> $log
STEPS="499" bash $cv5/run_batteries_h200.sh $a6cfg $a6exp >> $diag/chain_${a6exp}.log 2>&1
sf=$diag/side_flip_${a6exp}_499_semantic/side_flip_eval.json
verdict=$(python3 - "$sf" <<'PY'
import json, sys
try:
    s = json.load(open(sys.argv[1]))["summary_first_decision_step"]
    acc, fol, reset = s["normal_side_accuracy"], s.get("flip_follows_content_rate", -1.0), s["reset_side_accuracy"]
    print(("PASS" if acc >= 0.9 and fol >= 0.9 else "FAIL") + f" first-step normal_acc={acc:.3f} flip_follows_content={fol:.3f} reset_acc={reset:.3f} excluded_history_decided={s.get('excluded_history_decided')}")
except Exception as e:
    print(f"FAIL (battery unreadable: {e})")
PY
)
echo "A6 ckpt-499 verdict: $verdict $(date '+%m/%d %H:%M')" >> $log
if ! echo "$verdict" | grep -q "^PASS"; then echo "A6 failed the bar (stop)" >> $log; exit 1; fi
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
