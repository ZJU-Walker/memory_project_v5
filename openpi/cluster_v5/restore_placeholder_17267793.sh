#!/usr/bin/env bash
# Re-hold job 17267793's H200 after the baseline training was stopped (user 2026-09-06 19:49).
# Same tuned profile the user asked for on 09-05: steady 80-90% on both memory and utilisation.
export HOME=/iris/u/kewalk
cd /iris/u/kewalk/memory_project_v5/openpi || exit 2
JOB=17267793 GPU=0 MEM_FRAC=0.85 DUTY=0.92 \
  setsid nohup bash cluster_v5/gpu_placeholder_tuned.sh >> /iris/u/kewalk/memory_project_v5/v5/tools/logs/placeholder_tuned_17267793.log 2>&1 &
sleep 3
echo "placeholder relaunched on job 17267793"
