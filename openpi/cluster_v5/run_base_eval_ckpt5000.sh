#!/usr/bin/env bash
# One-off: score the non-memory baseline ckpt-5000 on a dev demo, with GT vs predicted subtask
# overlaid. Runs as an --overlap step of job 17286852 pinned to GPU 1 (the keep_1750 server card,
# which the user cleared at 17:16 while not testing the robot). PREALLOCATE=false + MEM_FRACTION=0.4
# so the server keeps its ~67 GB. GPU 0 of that job is the user's own train_qwen.py: never touch it.
set -u
DEMO="${DEMO:-demo1027}"
# The script default is 10 greedy steps, but the v7tgt sentences run to 13 PaliGemma tokens,
# so 10 truncates every long one ("scoop 1 of 2: dig and" instead of "... dig and carry") and
# scores it as a miss. 16 leaves headroom above the longest sentence.
MAXDEC="${MAXDEC:-16}"
STEP="${STEP:-5000}"
EXP=pi05_beans0905_base_v7rtc_20260906_r1
root=/iris/u/kewalk/memory_project_v5
out="$root/v5/diagnostics/base_eval_ckpt${STEP}"; mkdir -p "$out"
cd "$root/openpi" || exit 2
export HOME=/iris/u/kewalk
source cluster_v5/env.sh >/dev/null 2>&1
srun --jobid=17286852 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:2 \
  env CUDA_VISIBLE_DEVICES=GPU-b3d023a5-d124-a509-7f5a-9fb34083371a \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  .venv/bin/python scripts/eval_yam_subtask_raw.py \
    --config pi05_yam_beans0905_base \
    --ckpt-dir "$root/v5/checkpoints/pi05_yam_beans0905_base/$EXP/$STEP" \
    --raw-demo /iris/u/kewalk/memory_project/data/0905beans_all/"$DEMO" \
    --prompt 'scoop the beans into the tray as many times as the green light blinked' \
    --gt-labels auto --stride 15 --batch-size 8 --max-decode-steps "$MAXDEC" > "$out/$DEMO.log" 2>&1
echo "$DEMO exit=$? $(date '+%m/%d %H:%M')" >> "$out/status.log"
