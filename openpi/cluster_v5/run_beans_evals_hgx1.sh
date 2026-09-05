#!/usr/bin/env bash
# Bean-scoop evaluations for one checkpoint on GPU 0 of the 4xH100 job $JOB (iris-hgx-1), NFS root. Run ON the node
# between trainings (the sentinel keeps placeholders on GPUs 1-3 and leaves GPU 0 alone while these scripts run).
#   [JOB=17178887] [MODES="self oracle"] [BATCHES=24] cluster_v5/run_beans_evals_hgx1.sh <config-name> <exp-name> <step>
# 1. rollout videos of the 6 development episodes (scripts/v5_heldout_video.py --manifest/--sidecar)
#    -> v5/diagnostics/videos_<exp>_<step>/ep<idx>_<mode>.{mp4,json}
# 2. count-flip battery (scripts/v5_count_flip_eval.py) -> v5/diagnostics/count_flip_<exp>_<step>/count_flip_eval.json
set -u
config="$1"; exp="$2"; step="$3"; JOB="${JOB:-17178887}"; GRES="${GRES:-4}"; modes="${MODES:-self oracle}"; batches="${BATCHES:-24}"
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
# Free GPU 0 of the job (the sentinel re-fills GPUs 1-3 only while an eval python runs).
ph=$(pgrep -f "gpu_placeholder_marker_${JOB}_g0" || true); [ -n "$ph" ] && { kill $ph 2>/dev/null; sleep 5; }
dev=$(.venv/bin/python -c "
import json; m=json.load(open('$manifest'))
print(' '.join(str(e['episode_index']) for e in sorted(m['episodes'], key=lambda e: e['episode_index']) if e.get('split')=='development'))")
echo "beans evals $exp ckpt-$step started $(date +%H:%M) on GPU 0 of $JOB; dev episodes: $dev" >> "$out/status.log"
run() {  # one --overlap step on GPU 0 of the job
  srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:"$GRES" env CUDA_VISIBLE_DEVICES=0 "$@"
}
for mode in $modes; do
  for ep in $dev; do
    tag=$(printf 'ep%02d_%s' "$ep" "$mode")
    [ -e "$out/$tag.mp4" ] && continue
    run .venv/bin/python scripts/v5_heldout_video.py --config-name "$config" --params "$ck" --episode-index "$ep" \
        --write-mode "$mode" --output-dir "$out" --manifest "$manifest" --sidecar "$sidecar" > "$out/${tag}_run.log" 2>&1
    echo "$tag exit=$? $(date +%H:%M)" >> "$out/status.log"
  done
done
echo "all videos done $(date +%H:%M)" >> "$out/status.log"
if [ ! -e "$cf/count_flip_eval.json" ]; then
  mkdir -p "$cf"
  run .venv/bin/python scripts/v5_count_flip_eval.py --config-name "$config" --params "$ck" --split development \
      --batches "$batches" --output-dir "$cf" > "$cf/run.log" 2>&1
  echo "count-flip exit=$? $(date +%H:%M)" >> "$out/status.log"
fi
echo "evals done $(date +%H:%M)" >> "$out/status.log"
