#!/usr/bin/env bash
# Start the v5 robot policy server on the GPU box (inside the SLURM job that owns the GPU: ssh iris-hgx-1 lands in
# job 17192955, shared with the user's Qwen action-expert server ~11 GB). Default: B6a judged checkpoint keep_499.
# Warms up the JIT (plain + RTC-prefixed request, ~2-3 min) before accepting connections, then serves on --port.
#   [CONFIG=pi05_yam_mem_v5_stageB6a] bash /iris/u/kewalk/memory_project_v5/v5/diagnostics/run_server_v5.sh [port] [ckpt_dir]
# B5a: CONFIG=pi05_yam_mem_v5_stageB5a bash run_server_v5.sh 8000 /iris/u/kewalk/memory_project_v5/v5/checkpoints/pi05_yam_mem_v5_stageB5a/v5_stageB5a_20260903_r1/keep_999
cd /iris/u/kewalk/memory_project_v5/openpi || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk PYTHONPATH=scripts
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
port=${1:-8000}
ck=${2:-/iris/u/kewalk/memory_project_v5/v5/checkpoints/pi05_yam_mem_v5_stageB6a/v5_stageB6a_20260903_r1/keep_499}
config=${CONFIG:-pi05_yam_mem_v5_stageB6a}
# Free this job's busy placeholder (the sentinel keeps it off while the server runs).
ph=$(pgrep -f "gpu_placeholder_marker_17192955" || true)
[ -n "$ph" ] && { kill $ph 2>/dev/null; sleep 8; echo "killed placeholder pids: $ph"; }
log=/iris/u/kewalk/memory_project_v5/v5/diagnostics/server_v5_$(date +%Y%m%d_%H%M).log
echo "serving $ck ($config) on $(hostname) ($(hostname -I | awk '{print $1}')) port $port; log $log"
exec .venv/bin/python -u scripts/serve_yam_memory.py --dir "$ck" --config "$config" --port "$port" --warmup 2>&1 | tee "$log"
