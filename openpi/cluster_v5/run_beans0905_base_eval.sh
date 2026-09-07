#!/usr/bin/env bash
# Score the NON-memory pi05 baseline (pi05_yam_beans0905_base) on the same six development
# episodes B9 is judged on. Usage, from a shell on a node of the job:
#   [JOB=17267793] [STEP=30000] [EXP=pi05_beans0905_base_v7rtc_20260906_r1] cluster_v5/run_beans0905_base_eval.sh
#
# Why this script and not run_beans_evals_job_v2.sh: v5_heldout_video.py is memory-only (it calls
# _v32_prepare_memory_prefix / memory_semantic / memory_stride_frames, none of which exist without a
# bank). The non-memory evaluator is scripts/eval_yam_subtask_raw.py, which reads the RAW demo folder
# and runs Pi0.sample_subtask_and_actions per frame -> subtask-overlay mp4 + predicted-vs-teleop joint
# plots, into scripts/eval_results/. It needs --prompt because it skips the LeRobot pipeline, so
# InjectPromptFromEpisode never runs and this config has no default_prompt.
set -u
JOB="${JOB:-17267793}"
STEP="${STEP:-30000}"
EXP="${EXP:-pi05_beans0905_base_v7rtc_20260906_r1}"
CONFIG=pi05_yam_beans0905_base
GRES="${GRES:-1}"; GPU="${GPU:-0}"
# stride 15 matches the cadence of the v5 memory videos; --gt-labels auto reads each demo's own
# subtask_labels_v7tgt.json, verified frame-for-frame identical to the SHA-pinned sidecar.
STRIDE="${STRIDE:-15}"; BATCH="${BATCH:-8}"
root=/iris/u/kewalk/memory_project_v5
raw=/iris/u/kewalk/memory_project/data/0905beans_all
prompt='scoop the beans into the tray as many times as the green light blinked'
# The six manifest split=development episodes, mapped through data/0905beans_all/source_map.json:
#   ep25 0905beans_1/demo27  ep29 0905beans_1/demo31  ep59 0905beans_2/demo1
#   ep64 0905beans_3/demo1   ep72 0905beans_3/demo9   ep73 0905beans_3/demo10
DEMOS="${DEMOS:-demo1027 demo1031 demo2001 demo3001 demo3009 demo3010}"

cd "$root/openpi" || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk
ck="$root/v5/checkpoints/$CONFIG/$EXP/$STEP"
[ -d "$ck/params" ] || { echo "no checkpoint at $ck/params"; exit 2; }
out="$root/v5/diagnostics/base_eval_${EXP}_${STEP}"; mkdir -p "$out"

# A restored placeholder holds ~120 GB and will OOM the eval (the A6sep lesson). Kill only THIS job's.
if [ -z "${NO_PLACEHOLDER:-}" ]; then
  ph=$(pgrep -f "gpu_placeholder_marker_${JOB}" || true)
  [ -n "$ph" ] && { kill $ph 2>/dev/null; sleep 5; echo "killed placeholder pids: $ph" >> "$out/status.log"; }
fi

echo "base eval $EXP ckpt-$STEP started $(date '+%m/%d %H:%M') demos: $DEMOS" >> "$out/status.log"
for demo in $DEMOS; do
  [ -d "$raw/$demo" ] || { echo "$demo MISSING $(date +%H:%M)" >> "$out/status.log"; continue; }
  srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:"$GRES" \
    env CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/eval_yam_subtask_raw.py \
      --config "$CONFIG" --ckpt-dir "$ck" --raw-demo "$raw/$demo" --prompt "$prompt" \
      --gt-labels auto --stride "$STRIDE" --batch-size "$BATCH" \
      ${EXTRA_ARGS:-} > "$out/${demo}_run.log" 2>&1
  echo "$demo exit=$? $(date +%H:%M)" >> "$out/status.log"
  grep -hE "^(pred|gt) +timeline:|^exact-match subtask:" "$out/${demo}_run.log" >> "$out/timelines.txt" 2>/dev/null
done
echo "base eval done $(date '+%m/%d %H:%M'); artifacts in scripts/eval_results/" >> "$out/status.log"
