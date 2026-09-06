#!/usr/bin/env bash
# A8 keep_299 ORACLE-write rollouts (ground-truth sentences fed to the bank) on the pinned GPU of a job, after the A8
# probe chain (2026-09-05 23:15): separates "does the decoder use the bank count at the tray" from "can the model
# write its own sentences" (the self-write rollouts showed the copy shortcut: sticky 'yellow go', 'scoop 1' repeated).
#   JOB=17286852 GRES=2 GPU=<uuid> setsid nohup bash cluster_v5/run_a8_oracle_videos_job.sh > /dev/null 2>&1 < /dev/null &
set -u
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics; log=$diag/queue_beans_hgx1.log
echo "A8 oracle-video runner armed on $(hostname) job $JOB: waits for the A8 probe chain $(date '+%m/%d %H:%M')" >> $log
until grep -q "^A8 probe chain done" $log; do sleep 60; done
cd /iris/u/kewalk/memory_project_v5/openpi || exit 2
MODES=oracle OUT_SUFFIX=_oracle SKIP_COUNT_FLIP=1 NO_PLACEHOLDER=1 SIDECAR=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5/beans/beans_v5_subtask_labels_v6sub.json \
  bash cluster_v5/run_beans_evals_job.sh pi05_yam_mem_v5_beansA8 v5_beansA8_20260905_r1 keep_299 >> $diag/chain_v5_beansA8_20260905_r1_oracle.log 2>&1
echo "A8 oracle videos: $(tail -1 $diag/videos_v5_beansA8_20260905_r1_keep_299_oracle/status.log) $(date '+%m/%d %H:%M')" >> $log
