#!/usr/bin/env bash
# Sentinel for the H200 job 17267793 on iris-hgx-2 (2026-09-05 09:50): the trossen placeholder there finishes its
# 30k-step run and exits; relaunch it (cluster_v5/gpu_placeholder_job.sh -> placeholder_train_trossen.sh) whenever no
# placeholder marker and no v5 eval python is running. Run ON iris-hgx-2:  setsid nohup bash cluster_v5/gpu_sentinel_hgx2.sh &
export HOME=/iris/u/kewalk
LOGDIR=/iris/u/kewalk/memory_project_v5/v5/tools/logs; mkdir -p $LOGDIR
echo "$(date '+%m/%d %H:%M') sentinel up on $(hostname)" >> $LOGDIR/sentinel_hgx2.log
while true; do
  if ! pgrep -f "v5_heldout_vide[o]|v5_count_flip_eva[l]|run_beans_evals_hgx[2]|run_count_flip_varian[t]|v5_probe_[a-z_]*\.py|serve_yam_memor[y]" >/dev/null \
     && ! pgrep -f "gpu_placeholder_marker_1726779[3]" >/dev/null; then
    JOB=17267793 bash /iris/u/kewalk/memory_project_v5/openpi/cluster_v5/gpu_placeholder_job.sh >> $LOGDIR/sentinel_hgx2.log 2>&1
  fi
  sleep 60
done
