#!/usr/bin/env bash
# Bank-level count-recovery test (scripts/v5_probe_count_recovery.py; the hgx-1 sentinel treats v5_probe_*.py as real
# work) on one GPU of the 4xH100 job 17249058: the A6 parameters under the plain encoding (beansA6 config) and under
# the A8 encoding (slot keys + whitened values, beansA8 config) -> Test 2 of README §8 20:45.
set -u
JOB="${JOB:-17249058}"
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v5; cd "$root/openpi" || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk XLA_PYTHON_CLIENT_PREALLOCATE=false
ck=$root/v5/checkpoints/pi05_yam_mem_v5_beansA6/v5_beansA6_20260905_r1/keep_499/params
for cfg in pi05_yam_mem_v5_beansA6 pi05_yam_mem_v5_beansA8; do
  tag=${cfg##*beans}
  out="$root/v5/diagnostics/count_recovery_A6params_${tag}enc"; mkdir -p "$out"
  echo "count recovery A6 params under the $tag encoding started $(date +%H:%M)" >> "$out/status.log"
  srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 env CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python scripts/v5_probe_count_recovery.py --config-name "$cfg" --params "$ck" --split development \
      --alphas 0.01,0.0 --output-dir "$out" > "$out/run.log" 2>&1
  echo "count recovery exit=$? $(date +%H:%M)" >> "$out/status.log"
done
