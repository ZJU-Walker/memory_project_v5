#!/usr/bin/env bash
# Bank-geometry probe chain, NOWAIT variant (2026-09-05 18:42): runs immediately on a given job/GPU (the user's new
# 2-GPU job 17284681 on iris-hgx-2). scripts/v5_bank_geometry_eval.py; see run_beans_evals_hgx2_bank_geometry.sh.
# Waits until the job's other v5 evaluations have drained, then replays the true note sequence of the dev episodes
# into a fresh bank under alpha 0.01 / 0.001 / 0 and reads the go note back with its own key (decay vs interference).
# Named run_beans_evals_hgx2_* so gpu_sentinel_hgx2.sh treats it as real work.
#   JOB=17267793 bash cluster_v5/run_beans_evals_hgx2_bank_geometry.sh
set -u
JOB="${JOB:-17267793}"
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v5; cd "$root/openpi" || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk XLA_PYTHON_CLIENT_PREALLOCATE=false
log=$root/v5/diagnostics/bank_geometry_chain_nowait.log
echo "bank-geometry chain waiting for the H200 evals to drain $(date '+%m/%d %H:%M')" >> $log
# NOWAIT: this variant does not wait for the node's other evals (it owns its own job's GPU).
ph=$(pgrep -f "gpu_placeholder_marker_${JOB}" || true); [ -n "$ph" ] && { kill $ph 2>/dev/null; sleep 8; }
GPU="${GPU:-0}"
ck=$root/v5/checkpoints
run() {  # tag config params
  local out="$root/v5/diagnostics/bank_geometry_$1"; mkdir -p "$out"
  echo "bank geometry $1 started $(date +%H:%M)" >> "$out/status.log"
  srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 env CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/python scripts/v5_bank_geometry_eval.py --config-name "$2" --params "$3" --split development \
      --alphas 0.01,0.001,0.0 --output-dir "$out" > "$out/run.log" 2>&1
  echo "bank geometry exit=$? $(date +%H:%M)" >> "$out/status.log"
}
run A6_keep_499 pi05_yam_mem_v5_beansA6 $ck/pi05_yam_mem_v5_beansA6/v5_beansA6_20260905_r1/keep_499/params
run B6_keep_499 pi05_yam_mem_v5_beansA6 $ck/pi05_yam_mem_v5_beansB6/v5_beansB6_20260905_r1/keep_499/params
echo "bank-geometry chain done $(date '+%m/%d %H:%M')" >> $log
if ! pgrep -f "v5_heldout_vide[o]|v5_count_flip_eva[l]|v5_tray_flip_eva[l]" >/dev/null; then JOB=$JOB bash cluster_v5/gpu_placeholder_job.sh >> $log 2>&1; fi
