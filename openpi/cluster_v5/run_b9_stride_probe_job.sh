#!/usr/bin/env bash
# Tick-rate probe (2026-09-06 21:00): self-write rollouts of the dev episodes with the memory stride overridden
# (STRIDE=8 frames = 250 ms ticks, what the robot client sees at --hz 20 with a replan every 5 controls; training
# stride is 5 = 167 ms). Tests whether the slower tick rate breaks the blink count ("light on: 2" at the first blink).
#   JOB=17286852 GRES=2 GPU=<uuid> STEPS="keep_2750 1000" EPS="25 29 59" STRIDE=8 bash cluster_v5/run_b9_stride_probe_job.sh
set -u
JOB="${JOB:?set JOB}"; GRES="${GRES:-2}"; GPU="${GPU:?set GPU}"; steps="${STEPS:-keep_2750 1000}"; eps="${EPS:-25 29 59}"; stride="${STRIDE:-8}"
config=pi05_yam_mem_v5_beansB9; exp=v5_beansB9_20260906_r1
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v5
cd "$root/openpi" || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.35}"
manifest=$root/openpi/cluster_v5/beans/beans_episode_manifest_0905_v1.json
sidecar=$root/openpi/cluster_v5/beans/beans_v5_subtask_labels_0905_v7tgt.json
for step in $steps; do
  ck="$root/v5/checkpoints/$config/$exp/$step/params"
  out="$root/v5/diagnostics/probe_stride${stride}_${exp}_${step}"; mkdir -p "$out"
  echo "stride-$stride probe $exp ckpt-$step started $(date +%H:%M) episodes: $eps" >> "$out/status.log"
  for ep in $eps; do
    tag=$(printf 'ep%02d_self_stride%d' "$ep" "$stride")
    [ -e "$out/$tag.json" ] && continue
    srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:"$GRES" env CUDA_VISIBLE_DEVICES="$GPU" \
      .venv/bin/python scripts/v5_heldout_video.py --config-name "$config" --params "$ck" --episode-index "$ep" \
        --write-mode self --stride "$stride" --output-dir "$out" --manifest "$manifest" --sidecar "$sidecar" > "$out/${tag}_run.log" 2>&1
    echo "$tag exit=$? $(date +%H:%M)" >> "$out/status.log"
  done
done
echo "stride probe done $(date +%H:%M)" >> "$out/status.log"
