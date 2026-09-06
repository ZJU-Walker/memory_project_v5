#!/usr/bin/env bash
# Robot policy server for a v5 checkpoint on ONE GPU of any Slurm job of ours (2026-09-05 18:40; the 1-GPU serving
# job 17192955 of serve_v5_hgx1.sh ended on 09-04). Runs inside the job with `srun --overlap`; the GPU sentinels treat
# `serve_yam_memory` as real work. Warms up the JIT (plain + RTC-prefixed request) before accepting connections.
#   JOB=17284681 [GPU=0] [LOG=...] bash cluster_v5/serve_v5_job.sh <checkpoint dir with params/ and assets/> <config-name> [port=8000]
export HOME=/iris/u/kewalk
ck="$1"; cfg="$2"; port="${3:-8000}"; JOB="${JOB:?set JOB}"; GPU="${GPU:-0}"
cd /iris/u/kewalk/memory_project_v5/openpi || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
ph=$(pgrep -f "gpu_placeholder_marker_${JOB}" || true)
[ -n "$ph" ] && { kill $ph 2>/dev/null; sleep 8; echo "killed job $JOB placeholder: $ph"; }
log="${LOG:-/iris/u/kewalk/memory_project_v5/v5/diagnostics/server_v5_$(date +%Y%m%d_%H%M).log}"
echo "serving $cfg from $ck on $(hostname) ($(hostname -I | awk '{print $1}')) port $port (job $JOB gpu $GPU); log $log" | tee -a "$log"
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 \
  env CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 PYTHONPATH=scripts \
  .venv/bin/python -u scripts/serve_yam_memory.py --dir "$ck" --config "$cfg" --port "$port" --warmup 2>&1 | tee -a "$log"
