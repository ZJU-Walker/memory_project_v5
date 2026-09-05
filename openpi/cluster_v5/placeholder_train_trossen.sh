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
case "$gpus" in
  1) exp=pi05_pack_with_human_full_0904_h200; batch=32 ;;
  2) exp=pi05_pack_with_human_full_0904_2h100; batch=16 ;;
  4) exp=pi05_pack_with_human_full_0904_4h100; batch=32 ;;   # 23:00: the user's new 4xH100 job 17249058
  *) echo "unsupported gpu count $gpus"; exit 2 ;;
esac
ckdir=$repo/checkpoints/$config/$exp
if ls "$ckdir" 2>/dev/null | grep -qE '^[0-9]+$'; then mode=--resume; else mode=--overwrite; fi
visible=$(seq -s, 0 $((gpus - 1)))
log=/iris/u/kewalk/memory_project_v5/v5/tools/logs/placeholder_train_${job}.log
echo "$(date '+%m/%d %H:%M') placeholder training start job=$job gpus=$gpus exp=$exp batch=$batch $mode" >> "$log"
cd "$repo" || exit 2
exec srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:"$gpus" \
  env GPU_PLACEHOLDER="gpu_placeholder_marker_${job}" CUDA_VISIBLE_DEVICES="$visible" \
      HF_LEROBOT_HOME=/iris/projects/humanoid/trossen_data XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
      WANDB_DIR=/iris/u/kewalk/openpi_trossen/wandb WANDB_MODE=offline HOME=/iris/u/kewalk \
  "$repo/.venv/bin/python" scripts/train.py "$config" --exp-name "$exp" --batch-size "$batch" --fsdp-devices "$gpus" \
      --no-wandb-enabled "$mode" >> "$log" 2>&1
