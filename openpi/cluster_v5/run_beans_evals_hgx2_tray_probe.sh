#!/usr/bin/env bash
# Tray-decision probe chain on the H200 job 17267793 (2026-09-05 17:15; scripts/v5_tray_flip_eval.py). Named
# run_beans_evals_hgx2_* on purpose: gpu_sentinel_hgx2.sh treats that pattern as real work and will not start the
# placeholder next to it. Kills the job's placeholder first; restores it at the end unless a v5 eval is running.
#   JOB=17267793 bash cluster_v5/run_beans_evals_hgx2_tray_probe.sh
set -u
JOB="${JOB:-17267793}"
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v5; cd "$root/openpi" || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk XLA_PYTHON_CLIENT_PREALLOCATE=false
ph=$(pgrep -f "gpu_placeholder_marker_${JOB}" || true); [ -n "$ph" ] && { kill $ph 2>/dev/null; sleep 8; }
ck=$root/v5/checkpoints
run() {  # tag config params split alphas
  local out="$root/v5/diagnostics/tray_flip_$1_$4"; mkdir -p "$out"
  echo "tray probe $1 split=$4 alphas=$5 started $(date +%H:%M)" >> "$out/status.log"
  srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 env CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python scripts/v5_tray_flip_eval.py --config-name "$2" --params "$3" --split "$4" --alphas "$5" \
      --batches 24 --output-dir "$out" > "$out/run.log" 2>&1
  echo "tray probe exit=$? $(date +%H:%M)" >> "$out/status.log"
}
run A6_keep_499 pi05_yam_mem_v5_beansA6 $ck/pi05_yam_mem_v5_beansA6/v5_beansA6_20260905_r1/keep_499/params train 0.01,0.0
run B6_keep_499 pi05_yam_mem_v5_beansA6 $ck/pi05_yam_mem_v5_beansB6/v5_beansB6_20260905_r1/keep_499/params train 0.01,0.0
run A6_keep_499 pi05_yam_mem_v5_beansA6 $ck/pi05_yam_mem_v5_beansA6/v5_beansA6_20260905_r1/keep_499/params development 0.01,0.0
run B6_keep_499 pi05_yam_mem_v5_beansA6 $ck/pi05_yam_mem_v5_beansB6/v5_beansB6_20260905_r1/keep_499/params development 0.01,0.0
if ! pgrep -f "v5_heldout_vide[o]|v5_count_flip_eva[l]" >/dev/null; then JOB=$JOB bash cluster_v5/gpu_placeholder_job.sh >> "$root/v5/diagnostics/tray_flip_chain.log" 2>&1; fi
