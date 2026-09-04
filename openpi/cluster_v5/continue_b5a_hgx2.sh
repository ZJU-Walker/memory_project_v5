#!/usr/bin/env bash
# User 2026-09-04 02:02: once B5a's tests are done, keep TRAINING it (the judged ckpt-999 stays as keep_999; the
# continuation resumes from the experiment dir and saves every 250 updates: 1250, 1500, ...). Run ON iris-hgx-2.
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
until grep -q "B5a batteries done\|B5a r1 failed\|B5a smoke failed" $diag/queue_a5_b5a.log 2>/dev/null; do sleep 60; done
grep -q "B5a batteries done" $diag/queue_a5_b5a.log || { echo "B5a chain failed; no continuation $(date '+%m/%d %H:%M')" >> $diag/queue_a5_b5a.log; exit 1; }
echo "B5a continuation to 3000 updates (user rule: keep training) $(date '+%m/%d %H:%M')" >> $diag/queue_a5_b5a.log
bash $cv5/run_train_h200.sh pi05_yam_mem_v5_stageB5a v5_stageB5a_20260903_r1 --num-train-steps 3000
echo "B5a continuation $(grep '^exit=' $diag/train_v5_stageB5a_20260903_r1_status.log | tail -1) $(date '+%m/%d %H:%M'); placeholder" >> $diag/queue_a5_b5a.log
bash $cv5/gpu_placeholder_hgx2.sh >> $diag/queue_a5_b5a.log 2>&1
