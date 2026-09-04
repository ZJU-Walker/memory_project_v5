#!/usr/bin/env bash
export HOME=/iris/u/kewalk
cd /iris/u/kewalk/memory_project_v5/openpi || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
ph=$(pgrep -f "gpu_placeholder_marker_17192955" || true); [ -n "$ph" ] && { kill $ph 2>/dev/null; sleep 8; }
log=/iris/u/kewalk/memory_project_v5/v5/diagnostics/serve_smoke_A5_999.log
srun --jobid=17192955 --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 \
  .venv/bin/python scripts/v5_serve_smoke.py --dir /iris/u/kewalk/memory_project_v5/v5/checkpoints/pi05_yam_mem_v5_stageA5/v5_stageA5_20260903_r1/999 --config pi05_yam_mem_v5_stageA5 > $log 2>&1
echo "exit=$? $(date +%H:%M)" >> $log
