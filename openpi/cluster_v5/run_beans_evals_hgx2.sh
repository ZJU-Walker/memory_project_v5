#!/usr/bin/env bash
# Bean-scoop evaluations for one checkpoint on the H200 of job $JOB (iris-hgx-2), NFS root (the beans dataset
# lives under the NFS worktree's v5/data). Run ON the node.
#   [JOB=17207774] [MODES="self oracle"] [BATCHES=24] cluster_v5/run_beans_evals_hgx2.sh <config-name> <exp-name> <step>
# 1. self-write (and oracle) rollout videos of the 6 development episodes (scripts/v5_heldout_video.py --manifest/--sidecar)
#    -> v5/diagnostics/videos_<exp>_<step>/ep<idx>_<mode>.{mp4,json}
# 2. count-flip battery (scripts/v5_count_flip_eval.py) -> v5/diagnostics/count_flip_<exp>_<step>/count_flip_eval.json
# Kills the job's busy placeholder first; restores it when done.
set -u
config="$1"; exp="$2"; step="$3"; JOB="${JOB:-17207774}"; modes="${MODES:-self oracle}"; batches="${BATCHES:-24}"
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v5
cd "$root/openpi" || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk XLA_PYTHON_CLIENT_PREALLOCATE=false
ck="$root/v5/checkpoints/$config/$exp/$step/params"
manifest="$root/openpi/cluster_v5/beans/beans_episode_manifest_v1.json"
sidecar="${SIDECAR:-$root/openpi/cluster_v5/beans/beans_v5_subtask_labels_v1.json}"  # A2/B2: SIDECAR=.../beans_v5_subtask_labels_v2light.json
out="$root/v5/diagnostics/videos_${exp}_${step}"; mkdir -p "$out"
cf="$root/v5/diagnostics/count_flip_${exp}_${step}"
ph=$(pgrep -f "gpu_placeholder_marker[^_]" || true); [ "$JOB" != "17207774" ] && ph=$(pgrep -f "gpu_placeholder_marker_${JOB}" || true)
[ -n "$ph" ] && { kill $ph 2>/dev/null; sleep 5; }
dev=$(.venv/bin/python -c "
import json; m=json.load(open('$manifest'))
print(' '.join(str(e['episode_index']) for e in sorted(m['episodes'], key=lambda e: e['episode_index']) if e.get('split')=='development'))")
echo "beans evals $exp ckpt-$step started $(date +%H:%M) dev episodes: $dev" >> "$out/status.log"
for mode in $modes; do
  for ep in $dev; do
    tag=$(printf 'ep%02d_%s' "$ep" "$mode")
    [ -e "$out/$tag.mp4" ] && continue
    srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 env CUDA_VISIBLE_DEVICES=0 \
      .venv/bin/python scripts/v5_heldout_video.py --config-name "$config" --params "$ck" --episode-index "$ep" \
        --write-mode "$mode" --output-dir "$out" --manifest "$manifest" --sidecar "$sidecar" > "$out/${tag}_run.log" 2>&1
    echo "$tag exit=$? $(date +%H:%M)" >> "$out/status.log"
  done
done
echo "all videos done $(date +%H:%M)" >> "$out/status.log"
if [ ! -e "$cf/count_flip_eval.json" ]; then
  mkdir -p "$cf"
  srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 env CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python scripts/v5_count_flip_eval.py --config-name "$config" --params "$ck" --split development \
      --batches "$batches" --output-dir "$cf" > "$cf/run.log" 2>&1
  echo "count-flip exit=$? $(date +%H:%M)" >> "$out/status.log"
fi
if [ "$JOB" = "17207774" ]; then bash cluster_v5/gpu_placeholder_hgx2.sh >> "$out/status.log" 2>&1; else JOB=$JOB bash cluster_v5/gpu_placeholder_job.sh >> "$out/status.log" 2>&1; fi
