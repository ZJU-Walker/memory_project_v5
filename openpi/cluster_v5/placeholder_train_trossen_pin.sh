#!/usr/bin/env bash
# Busy placeholder (the user's openpi_trossen pi0.5 training, see placeholder_train_trossen.sh) pinned to ONE GPU of a
# SHARED job (2026-09-06 15:10, user: placeholder on job 17286852 "only for free gpu there" while GPU 1 serves the
# robot). An --overlap --gres=gpu:1 step always lands on physical GPU 0, so the step requests all GRES of the job and
# pins with CUDA_VISIBLE_DEVICES=<uuid>; the busy guard checks ONLY the pinned GPU (rc=3 if it holds > 8 GB).
# Marker gpu_placeholder_marker_<job> (kill the srun line). No sentinel for such a job: the sentinel is per job and
# would kill this step because of the server on the other GPU.
#   bash cluster_v5/placeholder_train_trossen_pin.sh <job> <gres of the job> <gpu uuid> [cpus=8]
set -u
job="$1"; gres="$2"; gpu="$3"; cpus="${4:-8}"
export HOME=/iris/u/kewalk
repo=/iris/u/kewalk/openpi_trossen
config=pi05_trossen_pack_with_human_full
exp=pi05_pack_with_human_full_0905_h200_j${job}; batch=32
ckbase=/scr/kewalk_placeholder/checkpoints; mkdir -p "$ckbase" 2>/dev/null || ckbase=/tmp/kewalk_placeholder/checkpoints; mkdir -p "$ckbase"
ckdir=$ckbase/$config/$exp
log=/iris/u/kewalk/memory_project_v5/v5/tools/logs/placeholder_train_${job}.log
echo "$(date '+%m/%d %H:%M') placeholder training start job=$job gres=$gres gpu=$gpu exp=$exp batch=$batch --overwrite" >> "$log"
cd "$repo" || exit 2
trap 'rm -rf "$ckdir"' EXIT
srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --gres=gpu:"$gres" \
  env GPU_PLACEHOLDER="gpu_placeholder_marker_${job}" CUDA_VISIBLE_DEVICES="$gpu" PIN_GPU="$gpu" \
      HF_LEROBOT_HOME=/iris/projects/humanoid/trossen_data XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
      WANDB_DIR=/iris/u/kewalk/openpi_trossen/wandb WANDB_MODE=offline HOME=/iris/u/kewalk \
  bash -c 'used=$(nvidia-smi -i "$PIN_GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
           if [ "${used:-0}" -gt 8000 ]; then echo "$(date "+%m/%d %H:%M:%S") placeholder REFUSED: GPU $PIN_GPU already holds ${used} MiB (real work running)"; exit 3; fi
           exec "$0" "$@"' \
  "$repo/.venv/bin/python" scripts/train.py "$config" --exp-name "$exp" --batch-size "$batch" --fsdp-devices 1 \
      --checkpoint-base-dir "$ckbase" --save-interval 100000000 --no-wandb-enabled --overwrite >> "$log" 2>&1
rc=$?; rm -rf "$ckdir"; echo "$(date '+%m/%d %H:%M') placeholder training ended rc=$rc; checkpoints deleted ($ckdir)" >> "$log"; exit $rc
