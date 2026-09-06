#!/usr/bin/env bash
# B9 keep_299 ORACLE-write rollouts after the B9 self-write evals (count-flip line), same pinned GPU, no placeholder.
#   JOB=17286852 GRES=2 GPU=<uuid> setsid nohup bash cluster_v5/run_b9_oracle_videos_job.sh > /dev/null 2>&1 < /dev/null &
set -u
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics; log=$diag/queue_beans_hgx1.log
echo "B9 oracle-video runner armed on $(hostname) job $JOB: waits for the B9 keep_299 evals $(date '+%m/%d %H:%M')" >> $log
until grep -q "^count-flip exit=" $diag/videos_v5_beansB9_20260906_r1_keep_299/status.log 2>/dev/null; do sleep 60; done
cd /iris/u/kewalk/memory_project_v5/openpi || exit 2
MODES=oracle OUT_SUFFIX=_oracle SKIP_COUNT_FLIP=1 NO_PLACEHOLDER=1 MANIFEST=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5/beans/beans_episode_manifest_0905_v1.json SIDECAR=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5/beans/beans_v5_subtask_labels_0905_v7tgt.json \
  bash cluster_v5/run_beans_evals_job_v2.sh pi05_yam_mem_v5_beansB9 v5_beansB9_20260906_r1 keep_299 >> $diag/chain_v5_beansB9_20260906_r1_oracle.log 2>&1
echo "B9 oracle videos: $(tail -1 $diag/videos_v5_beansB9_20260906_r1_keep_299_oracle/status.log) $(date '+%m/%d %H:%M')" >> $log
