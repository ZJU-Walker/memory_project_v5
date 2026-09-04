#!/usr/bin/env bash
# User 2026-09-04 02:02: once B6a's tests are done, keep TRAINING it on the 4 H100s (judged ckpt-499 stays as
# keep_499; the continuation resumes and saves every 250 updates: 750, 1000, ...). Run ON iris-hgx-1.
export HOME=/iris/u/kewalk JOB=17178887 GPUS=4 BATCH=8
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
until grep -q "B6a batteries done\|B6a r1 failed\|B6a smoke failed" $diag/queue_a6_hgx1.log 2>/dev/null; do sleep 60; done
grep -q "B6a batteries done" $diag/queue_a6_hgx1.log || { echo "B6a chain failed; no continuation $(date '+%m/%d %H:%M')" >> $diag/queue_a6_hgx1.log; exit 1; }
echo "B6a continuation to 3000 updates (user rule: keep training) $(date '+%m/%d %H:%M')" >> $diag/queue_a6_hgx1.log
bash $cv5/run_train_h200.sh pi05_yam_mem_v5_stageB6a v5_stageB6a_20260903_r1 --num-train-steps 3000
echo "B6a continuation $(grep '^exit=' $diag/train_v5_stageB6a_20260903_r1_status.log | tail -1) $(date '+%m/%d %H:%M')" >> $diag/queue_a6_hgx1.log
