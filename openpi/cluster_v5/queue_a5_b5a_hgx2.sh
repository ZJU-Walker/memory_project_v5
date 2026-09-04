#!/usr/bin/env bash
# Run ON the node (user 2026-09-03 20:20: "go back to two stages"). Waits for the B5 diagnosis to free the GPU, then:
#   A5 smoke (20) -> A5 r1 (1000, label writes + prefill + exact sentences) -> keep_999 -> side-flip battery (semantic, 999)
#   -> verdict on the FIRST-step summary (PASS = normal >= 0.9 AND flip follows-content >= 0.9)
#      PASS -> B5a smoke -> B5a r1 (1000, own sentences, warm start from A5-999 via the cast audited loader, half LR)
#              -> keep_999 -> videos 999 (self, oracle) -> batteries -> placeholder
#      FAIL -> placeholder (stop; report)
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/scr/kewalk_v5/memory_project_v5/v5/checkpoints
a5cfg=pi05_yam_mem_v5_stageA5; a5exp=v5_stageA5_20260903_r1; a5smoke=v5_stageA5_20260903_smoke
bcfg=pi05_yam_mem_v5_stageB5a; bexp=v5_stageB5a_20260903_r1; bsmoke=v5_stageB5a_20260903_smoke
log=$diag/queue_a5_b5a.log
echo "queue A5->B5a armed $(date '+%m/%d %H:%M') code=$(cd /scr/kewalk_v5/memory_project_v5 && git rev-parse --short HEAD 2>/dev/null)" >> $log
until grep -q "diagnose done" $diag/diagnose_b5_status.log 2>/dev/null; do sleep 30; done
while pgrep -f "v4_side_flip_eva[l]|v4_stage2_eva[l]" >/dev/null; do sleep 15; done
echo "GPU free of the B5 diagnosis at $(date '+%m/%d %H:%M'); A5 smoke" >> $log
bash $cv5/run_train_h200.sh $a5cfg $a5smoke --num-train-steps 20 --save-interval 10 --keep-period 10
code=$(grep "^exit=" $diag/train_${a5smoke}_status.log | tail -1); echo "A5 smoke $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "A5 smoke failed; placeholder" >> $log; bash $cv5/gpu_placeholder_hgx2.sh >> $log 2>&1; exit 1; fi
bash $cv5/run_train_h200.sh $a5cfg $a5exp
code=$(grep "^exit=" $diag/train_${a5exp}_status.log | tail -1); echo "A5 r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "A5 r1 failed; placeholder" >> $log; bash $cv5/gpu_placeholder_hgx2.sh >> $log 2>&1; exit 1; fi
cp -r $ckroot/$a5cfg/$a5exp/999 $ckroot/$a5cfg/$a5exp/keep_999 && echo "A5 ckpt-999 protected as keep_999" >> $log
STEPS="999" bash $cv5/run_batteries_h200.sh $a5cfg $a5exp >> $diag/chain_${a5exp}.log 2>&1
sf=$diag/side_flip_${a5exp}_999_semantic/side_flip_eval.json
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
echo "A5 ckpt-999 verdict: $verdict $(date '+%m/%d %H:%M')" >> $log
if ! echo "$verdict" | grep -q "^PASS"; then echo "A5 failed the bar; placeholder (stop)" >> $log; bash $cv5/gpu_placeholder_hgx2.sh >> $log 2>&1; exit 1; fi
bash $cv5/run_train_h200.sh $bcfg $bsmoke --num-train-steps 20 --save-interval 10 --keep-period 10
code=$(grep "^exit=" $diag/train_${bsmoke}_status.log | tail -1); echo "B5a smoke $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "B5a smoke failed; placeholder" >> $log; bash $cv5/gpu_placeholder_hgx2.sh >> $log 2>&1; exit 1; fi
bash $cv5/run_train_h200.sh $bcfg $bexp
code=$(grep "^exit=" $diag/train_${bexp}_status.log | tail -1); echo "B5a r1 $code $(date '+%m/%d %H:%M')" >> $log
if ! echo "$code" | grep -q "exit=0"; then echo "B5a r1 failed; placeholder" >> $log; bash $cv5/gpu_placeholder_hgx2.sh >> $log 2>&1; exit 1; fi
cp -r $ckroot/$bcfg/$bexp/999 $ckroot/$bcfg/$bexp/keep_999 && echo "B5a ckpt-999 protected as keep_999" >> $log
bash $cv5/run_videos_hgx2.sh $bcfg $bexp 999 >> $diag/chain_${bexp}.log 2>&1
echo "B5a videos done $(date '+%m/%d %H:%M')" >> $log
bash $cv5/run_batteries_h200.sh $bcfg $bexp >> $diag/chain_${bexp}.log 2>&1
echo "B5a batteries done; placeholder restored $(date '+%m/%d %H:%M')" >> $log
