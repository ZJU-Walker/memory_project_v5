#!/usr/bin/env bash
# Held-out rollout videos (scripts/v5_heldout_video.py) for one checkpoint: the 8 development
# episodes in self-write mode first, then oracle-write mode. Run ON the node.
#   [JOB=17207774] [MODES="self oracle"] cluster_v5/run_videos_hgx2.sh <config-name> <exp-name> <step>
# Output: v5/diagnostics/videos_<exp>_<step>/ep<idx>_<mode>.{mp4,json,_run.log} + status.log (NFS).
# Kills the busy placeholder first; restores it when done.
set -u
config="$1"; exp="$2"; step="$3"; JOB="${JOB:-17207774}"; modes="${MODES:-self oracle}"
export HOME=/iris/u/kewalk
nfs_root=/iris/u/kewalk/memory_project_v5
local_root=/scr/kewalk_v5/memory_project_v5
if [ -e "$local_root/.staged" ] && [ -x "$local_root/openpi/.venv/bin/python" ]; then root="$local_root"; else root="$nfs_root"; fi
cd "$root/openpi" || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
ck="$root/v5/checkpoints/$config/$exp/$step/params"
out="$nfs_root/v5/diagnostics/videos_${exp}_${step}"; mkdir -p "$out"
ph=$(pgrep -f "gpu_placeholder_marke[r]" || true); [ -n "$ph" ] && { kill $ph 2>/dev/null; sleep 5; }
echo "videos $exp ckpt-$step started $(date +%H:%M)" >> "$out/status.log"
for mode in $modes; do
  for ep in 1 2 7 21 35 42 61 64; do
    tag=$(printf 'ep%02d_%s' "$ep" "$mode")
    [ -e "$out/$tag.mp4" ] && continue
    srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 env CUDA_VISIBLE_DEVICES=0 \
      .venv/bin/python scripts/v5_heldout_video.py --config-name "$config" --params "$ck" --episode-index "$ep" \
        --write-mode "$mode" --output-dir "$out" > "$out/${tag}_run.log" 2>&1
    echo "$tag exit=$? $(date +%H:%M)" >> "$out/status.log"
  done
done
echo "all videos done $(date +%H:%M)" >> "$out/status.log"
bash cluster_v5/gpu_placeholder_hgx2.sh >> "$out/status.log" 2>&1
