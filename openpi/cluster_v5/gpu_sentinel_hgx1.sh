#!/usr/bin/env bash
# Standing rule (user 2026-09-04 00:05): every idle GPU of ours on iris-hgx-1 is held busy (memory + util 70-100 %).
# Sentinel loop, run ON the node:  nohup bash cluster_v5/gpu_sentinel_hgx1.sh &
#   job 17178887 (4xH100): while a v5 training runs (launcher or train.py present) -> no placeholders;
#     otherwise GPUs 1-3 always get one; GPU 0 only when no battery/video/probe python is running
#     (those steps take the job's first GPU).
#   job 17192955 (1xH100, shared with the user's Qwen action-expert server ~11 GB) -> one placeholder (HOLD 52 GiB)
#     whenever no v5 python runs in that job.
# Each placeholder is an --overlap step of its job pinned to one GPU by CUDA_VISIBLE_DEVICES, marked
# gpu_placeholder_marker_<job>_g<k> (the train launcher kills every gpu_placeholder_marke[r] before it starts).
export HOME=/iris/u/kewalk
VENV=/iris/u/kewalk/memory_project_v4/openpi/.venv
LOGDIR=/iris/u/kewalk/memory_project_v5/v5/tools/logs; mkdir -p $LOGDIR
launch() {  # job gpu_index gres_count hold_gb
  local job=$1 g=$2 gres=$3 hold=$4 marker="gpu_placeholder_marker_${1}_g${2}"
  pgrep -f "$marker" >/dev/null && return
  nohup setsid srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:"$gres" \
    env CUDA_VISIBLE_DEVICES="$g" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 HOLD_GB="$hold" \
    "$VENV/bin/python" -c "
import time, os  # $marker
import jax, jax.numpy as jnp
n = int(os.environ.get('HOLD_GB', '64')) // 4
hold = jnp.zeros((n, 1024, 1024, 1024), dtype=jnp.float32)
a = jnp.ones((12288, 12288), dtype=jnp.float32)
step = jax.jit(lambda x: x @ x / 12288.0)
jax.block_until_ready(hold)
print('busy placeholder up on', jax.devices(), 'holding %d GiB' % (4 * n), flush=True)
while True:
    for _ in range(50):
        a = step(a)
    jax.block_until_ready(a)
    time.sleep(0.03)
" > "$LOGDIR/placeholder_${job}_g${g}.log" 2>&1 &
  echo "$(date '+%m/%d %H:%M') launched $marker (hold ${hold} GiB) pid=$!" >> $LOGDIR/sentinel_hgx1.log
}
echo "$(date '+%m/%d %H:%M') sentinel up on $(hostname)" >> $LOGDIR/sentinel_hgx1.log
while true; do
  if pgrep -f "run_train_h200.s[h]|cluster_v5/train.s[h]|train.py pi05_yam_mem_v[5]" >/dev/null; then
    ph=$(pgrep -f "gpu_placeholder_marker_1717888[7]" || true)
    [ -n "$ph" ] && { kill $ph 2>/dev/null; echo "$(date '+%m/%d %H:%M') training present: killed 4-GPU-job placeholders $ph" >> $LOGDIR/sentinel_hgx1.log; }
  else
    for g in 1 2 3; do launch 17178887 $g 4 64; done
    if ! pgrep -f "v4_side_flip_eva[l]|v4_stage2_eva[l]|v5_heldout_vide[o]|v5_count_flip_eva[l]|run_beans_evals_hgx[1]|v5_probe_[a-z_]*\.py|v5_probe_query_drif[t]" >/dev/null; then
      launch 17178887 0 4 64
    fi
  fi
  # 2026-09-04 15:58 (user: the 2xH100 job 17267129 was "also placeholder training", ours to use): GPU 1 always
  # holds a busy placeholder; GPU 0 only while no eval python runs (the beans evaluations take GPU 0). The job's
  # batch payload is the user's 1 GB keep-alive (train_hs.sh) and is never touched. The 1-GPU serving job 17192955
  # is gone.
  if ! pgrep -f "train.py pi05_yam_mem_v[4]" >/dev/null; then  # only once the v4 step there is gone
    if ! pgrep -f "v4_side_flip_eva[l]|v4_stage2_eva[l]|v5_heldout_vide[o]|v5_count_flip_eva[l]|run_beans_evals_hgx[1]|v5_probe_[a-z_]*\.py" >/dev/null; then
      launch 17267129 0 2 64
    fi
    launch 17267129 1 2 64
  fi
  sleep 60
done
