#!/usr/bin/env bash
# One count-flip battery variant on the H200 of job $JOB (run ON iris-hgx-2), e.g. the write-rule override:
#   JOB=17267793 bash cluster_v5/run_count_flip_variant_hgx2.sh <config> <exp> <step> <suffix> [extra count-flip args]
# Kills the job's placeholder first and restores it after. Output: v5/diagnostics/count_flip_<exp>_<step><suffix>/
set -u
config="$1"; exp="$2"; step="$3"; suffix="$4"; shift 4; JOB="${JOB:-17267793}"
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v5
cd "$root/openpi" || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk XLA_PYTHON_CLIENT_PREALLOCATE=false
ck="$root/v5/checkpoints/$config/$exp/$step/params"
cf="$root/v5/diagnostics/count_flip_${exp}_${step}${suffix}"; mkdir -p "$cf"
ph=$(pgrep -f "gpu_placeholder_marker_${JOB}" || true); [ -n "$ph" ] && { kill $ph 2>/dev/null; sleep 5; }
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 env CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/v5_count_flip_eval.py --config-name "$config" --params "$ck" --split development \
    --batches 24 --output-dir "$cf" "$@" > "$cf/run.log" 2>&1
echo "count-flip variant $suffix exit=$? $(date +%H:%M)" >> "$cf/status.log"
JOB=$JOB bash cluster_v5/gpu_placeholder_job.sh >> "$cf/status.log" 2>&1
