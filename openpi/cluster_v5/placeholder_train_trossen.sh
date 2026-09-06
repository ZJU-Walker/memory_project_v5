#!/usr/bin/env bash
# Busy placeholder = a REAL training from the user's openpi_trossen repo (user 2026-09-04 16:22: "use my
# openpi_trossen repo to keep training"). Runs as an --overlap step of Slurm job $1 on $2 GPUs, so the GPU shows a
# genuine pi0.5 training (memory + utilisation). Killable by our launchers/sentinel through the marker string that
# the srun/env command line carries: gpu_placeholder_marker_<job> (killing the srun cancels the whole step).
#   bash cluster_v5/placeholder_train_trossen.sh <job> <gpus>
#   job 17267793 (H200, 1 GPU): resumes the user's run pi05_pack_with_human_full_0904_h200 (batch 32).
#   job 17267129 (2xH100):      exp pi05_pack_with_human_full_0904_2h100, batch 16, FSDP over 2 GPUs (fresh, then resume).
# wandb is disabled for these steps (no network dependency); checkpoints go to the user's usual
# openpi_trossen/checkpoints/pi05_trossen_pack_with_human_full/<exp>.
set -u
job="$1"; gpus="$2"
export HOME=/iris/u/kewalk
repo=/iris/u/kewalk/openpi_trossen
config=pi05_trossen_pack_with_human_full
# 2026-09-05 09:50: the 0904 runs wrote 540 GB of checkpoints to the NFS home and filled it (5.1 TB, 100 %). The
# placeholders now checkpoint on the NODE-LOCAL disk (/scr, wiped with the node; nothing to keep) as 0905 runs.
case "$gpus" in
  1) exp=pi05_pack_with_human_full_0905_h200; batch=32 ;;
  2) exp=pi05_pack_with_human_full_0905_2h100; batch=16 ;;
  4) exp=pi05_pack_with_human_full_0905_4h100; batch=32 ;;
  *) echo "unsupported gpu count $gpus"; exit 2 ;;
esac
ckbase=/scr/kewalk_placeholder/checkpoints; mkdir -p "$ckbase" 2>/dev/null || ckbase=/tmp/kewalk_placeholder/checkpoints; mkdir -p "$ckbase"
ckdir=$ckbase/$config/$exp
mode=--overwrite  # never resumed: placeholder checkpoints are always deleted
visible=$(seq -s, 0 $((gpus - 1)))
log=/iris/u/kewalk/memory_project_v5/v5/tools/logs/placeholder_train_${job}.log
echo "$(date '+%m/%d %H:%M') placeholder training start job=$job gpus=$gpus exp=$exp batch=$batch $mode" >> "$log"
cd "$repo" || exit 2
# 2026-09-05 10:03 (user: "future placeholder ckpt just delete, always delete"): no periodic saves (save interval
# beyond the run length) and the run's checkpoint dir is removed as soon as the step ends, whatever the exit code.
trap 'rm -rf "$ckdir"' EXIT
# 2026-09-05 21:55 (user: "make sure this won't happen again"): the step refuses to start when a GPU of the job already
# holds > 8 GB (a v5 rollout/probe/training the sentinel's pattern did not know about); rc=3, nothing written.
srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:"$gpus" \
  env GPU_PLACEHOLDER="gpu_placeholder_marker_${job}" CUDA_VISIBLE_DEVICES="$visible" \
      HF_LEROBOT_HOME=/iris/projects/humanoid/trossen_data XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
      WANDB_DIR=/iris/u/kewalk/openpi_trossen/wandb WANDB_MODE=offline HOME=/iris/u/kewalk \
  bash -c 'used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
           if [ "${used:-0}" -gt 8000 ]; then echo "$(date "+%m/%d %H:%M:%S") placeholder REFUSED: a GPU of this job already holds ${used} MiB (real work running)"; exit 3; fi
           exec "$0" "$@"' \
  "$repo/.venv/bin/python" scripts/train.py "$config" --exp-name "$exp" --batch-size "$batch" --fsdp-devices "$gpus" \
      --checkpoint-base-dir "$ckbase" --save-interval 100000000 --no-wandb-enabled "$mode" >> "$log" 2>&1
rc=$?; rm -rf "$ckdir"; echo "$(date '+%m/%d %H:%M') placeholder training ended rc=$rc; checkpoints deleted ($ckdir)" >> "$log"; exit $rc
