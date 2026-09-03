#!/usr/bin/env bash
# Run ON the node (user instruction 2026-09-03 00:17: "once your battery is over continue working on
# the next step ... if the result is not ok, just keep training the current model, at least occupy
# the GPU"). Waits for the A3 queue (batteries + videos) to finish, reads the ckpt-999 semantic
# side-flip summary and branches:
#   PASS (first-step donor_follows_content_rate >= 0.9 AND normal_side_accuracy >= 0.9)
#        -> B3 smoke (20) -> B3 r1 (1000, warm start from A3-999) -> batteries -> videos -> placeholder
#   FAIL (or B3 smoke fails)
#        -> continue A3 r1 to 3000 updates (resume) -> batteries at 3000/2000 -> videos at 3000 -> placeholder
# The formal bar (README §7, follows-content 1.00) is still judged by hand; this threshold only
# decides which run keeps the GPU busy overnight.
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
a3cfg=pi05_yam_mem_v5_stageA3; a3exp=v5_stageA3_20260902_r1
b3cfg=pi05_yam_mem_v5_stageB3; b3exp=v5_stageB3_20260903_r1; b3smoke=v5_stageB3_20260903_smoke
log=$diag/next_after_a3.log
echo "watcher armed $(date '+%m/%d %H:%M')" >> $log
while ! grep -q "queue: videos done" $diag/queue_a3.log 2>/dev/null; do
  if grep -q "queue: smoke failed\|queue: training failed" $diag/queue_a3.log 2>/dev/null; then
    echo "A3 queue failed before batteries; nothing to decide (placeholder holds the GPU)" >> $log; exit 1
  fi
  sleep 120
done
sf=$diag/side_flip_${a3exp}_999_semantic/side_flip_eval.json
verdict=$(python3 - "$sf" <<'PY'
import json, sys
try:
    s = json.load(open(sys.argv[1]))["summary_first_decision_step"]
    fc, acc, flip = s["donor_follows_content_rate"], s["normal_side_accuracy"], s["donor_flip_rate_mismatched"]
    print(("PASS" if fc >= 0.9 and acc >= 0.9 else "FAIL") + f" follows_content={fc:.3f} normal_acc={acc:.3f} donor_flip={flip:.3f}")
except Exception as e:  # missing/unreadable battery -> keep the current model training
    print(f"FAIL (battery unreadable: {e})")
PY
)
echo "A3 ckpt-999 verdict: $verdict at $(date '+%m/%d %H:%M')" >> $log
continue_a3() {
  echo "continuing A3 r1 to 3000 updates at $(date '+%m/%d %H:%M')" >> $log
  bash $cv5/run_train_h200.sh $a3cfg $a3exp --num-train-steps 3000
  echo "A3 continuation $(grep '^exit=' $diag/train_${a3exp}_status.log | tail -1) at $(date '+%m/%d %H:%M')" >> $log
  STEPS="3000 2000" bash $cv5/run_batteries_h200.sh $a3cfg $a3exp >> $diag/chain_${a3exp}_cont.log 2>&1
  bash $cv5/run_videos_hgx2.sh $a3cfg $a3exp 3000 >> $diag/chain_${a3exp}_cont.log 2>&1
  echo "A3 continuation batteries+videos done at $(date '+%m/%d %H:%M'); placeholder restored" >> $log
}
if echo "$verdict" | grep -q "^PASS"; then
  echo "launching B3 smoke at $(date '+%m/%d %H:%M')" >> $log
  bash $cv5/run_train_h200.sh $b3cfg $b3smoke --num-train-steps 20 --save-interval 10 --keep-period 10
  code=$(grep "^exit=" $diag/train_${b3smoke}_status.log | tail -1)
  echo "B3 smoke $code at $(date '+%m/%d %H:%M')" >> $log
  if echo "$code" | grep -q "exit=0"; then
    echo "launching B3 r1 at $(date '+%m/%d %H:%M')" >> $log
    bash $cv5/run_train_h200.sh $b3cfg $b3exp
    code=$(grep "^exit=" $diag/train_${b3exp}_status.log | tail -1)
    echo "B3 r1 $code at $(date '+%m/%d %H:%M')" >> $log
    if echo "$code" | grep -q "exit=0"; then
      bash $cv5/run_batteries_h200.sh $b3cfg $b3exp >> $diag/chain_${b3exp}.log 2>&1
      bash $cv5/run_videos_hgx2.sh $b3cfg $b3exp 999 >> $diag/chain_${b3exp}.log 2>&1
      echo "B3 batteries+videos done at $(date '+%m/%d %H:%M'); placeholder restored" >> $log
    else
      echo "B3 r1 failed; falling back to the A3 continuation" >> $log; continue_a3
    fi
  else
    echo "B3 smoke failed; falling back to the A3 continuation" >> $log; continue_a3
  fi
else
  continue_a3
fi
