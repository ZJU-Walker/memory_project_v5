#!/usr/bin/env bash
export HOME=/iris/u/kewalk
cd /iris/u/kewalk/memory_project_v5/openpi || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export PYTHONPATH=/iris/u/kewalk/memory_project_v5/openpi/src:/iris/u/kewalk/memory_project_v5/openpi/scripts
export XLA_PYTHON_CLIENT_PREALLOCATE=false
out=/iris/u/kewalk/memory_project_v5/v5/diagnostics/probe_query_drift_A4_999
srun --jobid=17192955 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  /iris/u/kewalk/memory_project_v4/openpi/.venv/bin/python scripts/v5_probe_query_drift.py --config-name pi05_yam_mem_v5_stageA4 \
    --params /scr/kewalk_v5_ckpt/A4_999/params --episodes 1 2 21 61 7 --output $out.json > $out.log 2>&1
echo "exit=$? $(date +%H:%M)" >> $out.log
