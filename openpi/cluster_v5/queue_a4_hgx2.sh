#!/usr/bin/env bash
# Run ON the node (user 2026-09-03 11:42: "do both"): after the A3-2999 fixed battery frees the GPU,
#   A4 smoke (20) -> A4 r1 (1000) -> protect ckpt-999 (copy to keep_999) -> batteries (v5-scoped side-flip)
#   -> videos ckpt-999 (self + oracle) -> verdict from the ckpt-999 semantic side-flip FIRST-STEP summary:
#      PASS = normal_side_accuracy >= 0.9 AND flip_follows_content_rate >= 0.9
#        -> B4 smoke -> B4 r1 (1000, own sentences, warm start from A4-999) -> batteries -> videos -> placeholder
#      FAIL (or B4 fails to start) -> continue A4 r1 to 2999 -> batteries 2999 -> videos 2999 -> placeholder
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/scr/kewalk_v5/memory_project_v5/v5/checkpoints
a4cfg=pi05_yam_mem_v5_stageA4; a4exp=v5_stageA4_20260903_r1; a4smoke=v5_stageA4_20260903_smoke
b4cfg=pi05_yam_mem_v5_stageB4; b4exp=v5_stageB4_20260903_r1; b4smoke=v5_stageB4_20260903_smoke
log=$diag/queue_a4.log
echo "queue A4 armed $(date '+%m/%d %H:%M')" >> $log
while pgrep -f "v4_side_flip_eva[l]" >/dev/null; do sleep 30; done
echo "GPU free of the A3-2999 battery at $(date '+%m/%d %H:%M'); A4 smoke" >> $log
bash $cv5/run_train_h200.sh $a4cfg $a4smoke --num-train-steps 20 --save-interval 10 --keep-period 10
code=$(grep "^exit=" $diag/train_${a4smoke}_status.log | tail -1); echo "A4 smoke $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "A4 smoke failed; placeholder" >> $log; bash $cv5/gpu_placeholder_hgx2.sh >> $log 2>&1; exit 1; fi
bash $cv5/run_train_h200.sh $a4cfg $a4exp
code=$(grep "^exit=" $diag/train_${a4exp}_status.log | tail -1); echo "A4 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "A4 r1 failed; placeholder" >> $log; bash $cv5/gpu_placeholder_hgx2.sh >> $log 2>&1; exit 1; fi
cp -r $ckroot/$a4cfg/$a4exp/999 $ckroot/$a4cfg/$a4exp/keep_999 && echo "ckpt-999 protected as keep_999" >> $log
bash $cv5/run_batteries_h200.sh $a4cfg $a4exp >> $diag/chain_${a4exp}.log 2>&1
bash $cv5/run_videos_hgx2.sh $a4cfg $a4exp 999 >> $diag/chain_${a4exp}.log 2>&1
sf=$diag/side_flip_${a4exp}_999_semantic/side_flip_eval.json
verdict=$(python3 - "$sf" <<'PY'
import json, sys
try:
    s = json.load(open(sys.argv[1]))["summary_first_decision_step"]
    acc, fol, reset = s["normal_side_accuracy"], s.get("flip_follows_content_rate", -1.0), s["reset_side_accuracy"]
    print(("PASS" if acc >= 0.9 and fol >= 0.9 else "FAIL") + f" first-step normal_acc={acc:.3f} flip_follows_content={fol:.3f} reset_acc={reset:.3f}")
except Exception as e:
    print(f"FAIL (battery unreadable: {e})")
PY
)
echo "A4 ckpt-999 verdict: $verdict $(date '+%m/%d %H:%M')" >> $log
continue_a4() {
  echo "continuing A4 r1 to 2999 $(date '+%m/%d %H:%M')" >> $log
  bash $cv5/run_train_h200.sh $a4cfg $a4exp --num-train-steps 3000
  echo "A4 continuation $(grep '^exit=' $diag/train_${a4exp}_status.log | tail -1) $(date '+%m/%d %H:%M')" >> $log
  STEPS="2999" bash $cv5/run_batteries_h200.sh $a4cfg $a4exp >> $diag/chain_${a4exp}_cont.log 2>&1
  bash $cv5/run_videos_hgx2.sh $a4cfg $a4exp 2999 >> $diag/chain_${a4exp}_cont.log 2>&1
  echo "A4 continuation done; placeholder restored $(date '+%m/%d %H:%M')" >> $log
}
if echo "$verdict" | grep -q "^PASS"; then
  bash $cv5/run_train_h200.sh $b4cfg $b4smoke --num-train-steps 20 --save-interval 10 --keep-period 10
  code=$(grep "^exit=" $diag/train_${b4smoke}_status.log | tail -1); echo "B4 smoke $code $(date '+%m/%d %H:%M')" >> $log
  if echo "$code" | grep -q "exit=0"; then
    bash $cv5/run_train_h200.sh $b4cfg $b4exp
    code=$(grep "^exit=" $diag/train_${b4exp}_status.log | tail -1); echo "B4 r1 $code $(date '+%m/%d %H:%M')" >> $log
    if echo "$code" | grep -q "exit=0"; then
      cp -r $ckroot/$b4cfg/$b4exp/999 $ckroot/$b4cfg/$b4exp/keep_999
      bash $cv5/run_batteries_h200.sh $b4cfg $b4exp >> $diag/chain_${b4exp}.log 2>&1
      bash $cv5/run_videos_hgx2.sh $b4cfg $b4exp 999 >> $diag/chain_${b4exp}.log 2>&1
      echo "B4 batteries+videos done; placeholder restored $(date '+%m/%d %H:%M')" >> $log
    else echo "B4 r1 failed -> A4 continuation" >> $log; continue_a4; fi
  else echo "B4 smoke failed -> A4 continuation" >> $log; continue_a4; fi
else
  continue_a4
fi
