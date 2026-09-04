#!/usr/bin/env bash
# Run ON the node (user 2026-09-03 17:09: "skip A ... directly start training and testing B"):
#   B5 smoke (20) -> B5 r1 (1000, own delayed confidence-gated sentences + oracle history prefill, init = A-stage init)
#   -> protect ckpt-999 -> videos ckpt-999 (self first, then oracle) -> batteries (v5-scoped, ckpt 999 then 500) -> placeholder.
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/scr/kewalk_v5/memory_project_v5/v5/checkpoints
cfg=pi05_yam_mem_v5_stageB5; exp=v5_stageB5_20260903_r1; smoke=v5_stageB5_20260903_smoke
log=$diag/queue_b5.log
echo "queue B5 armed $(date '+%m/%d %H:%M') code=$(cd /scr/kewalk_v5/memory_project_v5 && git rev-parse --short HEAD 2>/dev/null)" >> $log
bash $cv5/run_train_h200.sh $cfg $smoke --num-train-steps 20 --save-interval 10 --keep-period 10
code=$(grep "^exit=" $diag/train_${smoke}_status.log | tail -1); echo "B5 smoke $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "B5 smoke failed; placeholder" >> $log; bash $cv5/gpu_placeholder_hgx2.sh >> $log 2>&1; exit 1; fi
bash $cv5/run_train_h200.sh $cfg $exp
code=$(grep "^exit=" $diag/train_${exp}_status.log | tail -1); echo "B5 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "B5 r1 failed; placeholder" >> $log; bash $cv5/gpu_placeholder_hgx2.sh >> $log 2>&1; exit 1; fi
cp -r $ckroot/$cfg/$exp/999 $ckroot/$cfg/$exp/keep_999 && echo "ckpt-999 protected as keep_999" >> $log
bash $cv5/run_videos_hgx2.sh $cfg $exp 999 >> $diag/chain_${exp}.log 2>&1
echo "B5 videos done $(date '+%m/%d %H:%M')" >> $log
bash $cv5/run_batteries_h200.sh $cfg $exp >> $diag/chain_${exp}.log 2>&1
sf=$diag/side_flip_${exp}_999_semantic/side_flip_eval.json
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
echo "B5 ckpt-999 verdict: $verdict $(date '+%m/%d %H:%M')" >> $log
echo "B5 batteries done; placeholder restored $(date '+%m/%d %H:%M')" >> $log
