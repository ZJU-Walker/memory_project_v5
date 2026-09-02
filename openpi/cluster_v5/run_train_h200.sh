#!/usr/bin/env bash
# v5 training on the single H200 of Slurm job $JOB (default 17207774, iris-hgx-2), preemption-safe.
#   [JOB=17207774] [BATCH=2] [ACCUM=1] cluster_v5/run_train_h200.sh <config-name> <exp-name> [extra train.py args]
# Run ON the node (ssh iris-hgx-2 adopts the job): the payload is an --overlap step of the job so it
# is scoped to it. Kills the busy placeholder first (never the user's train_hs.py keep-alive).
# Recipe = v4's single-GPU recipe (global batch 2, no accumulation) for comparability; the H200's
# 143 GB would allow more, but batch >= 6 on one device crashed on the H100 (CUDA illegal address).
# Resume policy (as v4): numeric checkpoint in the experiment dir -> --resume; dir without one
# (crash before the first save) -> --overwrite; otherwise fresh.
# Private JAX compile cache for this GPU type: v35/cache/jax_hgx2 (the node's NFS is slow on first
# touch; the cache is worktree-local and never shared with another running process).
# Writes v5/diagnostics/train_<exp>_status.log; training log v5/diagnostics/train_<exp>.log (appended).
set -u
config="$1"; exp="$2"; shift 2
batch="${BATCH:-2}"; accum="${ACCUM:-1}"; JOB="${JOB:-17207774}"
nfs_root=/iris/u/kewalk/memory_project_v5
local_root=/scr/kewalk_v5/memory_project_v5
# Run from the node-local project copy when it is staged (cluster_v5/stage_local_project_hgx2.sh):
# every project-relative path (caches, data, checkpoints) then resolves on the local disk.
# Status/log files stay on NFS so they can be read from any host.
if [ -e "$local_root/.staged" ] && [ -x "$local_root/openpi/.venv/bin/python" ]; then root="$local_root"; else root="$nfs_root"; fi
cd "$root/openpi" || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk
diag="$nfs_root/v5/diagnostics"
ckdir="$root/v5/checkpoints/$config/$exp"
mode=fresh; extra=()
if [ -d "$ckdir" ]; then
  if ls "$ckdir" 2>/dev/null | grep -qE '^[0-9]+$'; then mode=resume; extra=(--resume); else mode=overwrite; extra=(--overwrite); fi
fi
ph=$(pgrep -f "gpu_placeholder_marke[r]" || true)
[ -n "$ph" ] && { kill $ph 2>/dev/null; sleep 5; echo "killed placeholder pids: $ph"; }
job=$(grep -oE 'job_[0-9]+' /proc/self/cgroup | sort -u | tr '\n' ' ')
echo "launch $(date +%m/%d\ %H:%M) host=$(hostname) job=$job step-of=$JOB config=$config exp=$exp batch=$batch accum=$accum mode=$mode root=$root extra=$*" >> "$diag/train_${exp}_status.log"
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  env CUDA_VISIBLE_DEVICES=0 \
  cluster_v5/train.sh "$config" --exp-name "$exp" --batch-size "$batch" --gradient-accumulation-steps "$accum" --fsdp-devices 1 \
  --no-wandb-enabled "${extra[@]}" "$@" >> "$diag/train_${exp}.log" 2>&1
echo "exit=$? $(date +%m/%d\ %H:%M)" >> "$diag/train_${exp}_status.log"
