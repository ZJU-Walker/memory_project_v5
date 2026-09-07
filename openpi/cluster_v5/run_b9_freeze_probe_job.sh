#!/usr/bin/env bash
# "LED never on" probe (2026-09-06 20:35, user: ckpt 2750 counts blinks on the real robot with the LED dark).
# Self-write rollouts of v5_heldout_video.py with --intervention freeze: the model sees the FIRST frame of the
# episode (LED off, arm still) at every step; only the memory advances. A counter that still writes
# "light on: k green blink(s)" runs on the demos' timing prior (first blink at frame 28 +- 8, period 25 +- 4), not
# on the LED. One GPU of a shared job, pinned by UUID; bounded memory so the policy servers on the card survive.
#   JOB=17286852 GRES=2 GPU=<uuid> STEPS="1000 keep_1750 keep_2750" EPS="25 59" bash cluster_v5/run_b9_freeze_probe_job.sh
set -u
JOB="${JOB:?set JOB}"; GRES="${GRES:-2}"; GPU="${GPU:?set GPU}"; steps="${STEPS:-1000 keep_1750 keep_2750}"; eps="${EPS:-25 59}"
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
  out="$root/v5/diagnostics/probe_freeze_${exp}_${step}"; mkdir -p "$out"
  echo "freeze probe $exp ckpt-$step started $(date +%H:%M) episodes: $eps" >> "$out/status.log"
  for ep in $eps; do
    tag=$(printf 'ep%02d_self_freeze' "$ep")
    [ -e "$out/$tag.json" ] && continue
    srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:"$GRES" env CUDA_VISIBLE_DEVICES="$GPU" \
      .venv/bin/python scripts/v5_heldout_video.py --config-name "$config" --params "$ck" --episode-index "$ep" \
        --write-mode self --intervention freeze --output-dir "$out" --manifest "$manifest" --sidecar "$sidecar" > "$out/${tag}_run.log" 2>&1
    echo "$tag exit=$? $(date +%H:%M)" >> "$out/status.log"
  done
done
echo "freeze probe done $(date +%H:%M)" >> "$out/status.log"
