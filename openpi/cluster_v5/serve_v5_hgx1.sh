#!/usr/bin/env bash
# Robot policy server for a v5 checkpoint on the 1-GPU job 17192955 of iris-hgx-1 (shared with the user's Qwen server).
#   bash cluster_v5/serve_v5_hgx1.sh <checkpoint dir with params/ and assets/> <config-name> [port=8000]
# Kills this job's busy placeholder first (the sentinel keeps it off while this script runs), then serves.
# Example: bash cluster_v5/serve_v5_hgx1.sh /iris/u/kewalk/memory_project_v5/v5/checkpoints/pi05_yam_mem_v5_stageB5a/v5_stageB5a_20260903_r1/keep_999 pi05_yam_mem_v5_stageB5a 8000
export HOME=/iris/u/kewalk
ck="$1"; cfg="$2"; port="${3:-8000}"
cd /iris/u/kewalk/memory_project_v5/openpi || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
ph=$(pgrep -f "gpu_placeholder_marker_17192955" || true)
[ -n "$ph" ] && { kill $ph 2>/dev/null; sleep 8; echo "killed 1-GPU-job placeholder: $ph"; }
log=/iris/u/kewalk/memory_project_v5/v5/diagnostics/serve_v5_$(date +%m%d_%H%M).log
echo "serving $cfg from $ck on port $port; log $log"
srun --jobid=17192955 --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 \
  .venv/bin/python scripts/serve_yam_memory.py --dir "$ck" --config "$cfg" --port "$port" 2>&1 | tee "$log"
