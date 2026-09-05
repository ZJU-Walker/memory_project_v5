#!/usr/bin/env bash
# Standing rule (user 2026-09-04 00:05): every idle GPU of ours on iris-hgx-1 is held busy (memory + util 70-100 %).
# Sentinel loop, run ON the node:  nohup bash cluster_v5/gpu_sentinel_hgx1.sh &
#   job 17249058 (4xH100, from 23:00; 17178887 before): the openpi_trossen training on all 4 GPUs whenever no v5
#     train step of that job and no eval python runs (see the loop).
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
  # 23:00 (job 17178887 timed out 22:46; the user's 4xH100 job 17249058 replaced it): like the 2xH100 job, the
  # placeholder is the REAL openpi_trossen training on all 4 GPUs (marker gpu_placeholder_marker_17249058), launched
  # only while no v5 train step of that job and no eval python runs (run_train_h200.sh / the eval runner kill it first).
  if ! pgrep -f "jobid=17249058 .*(cluster_v5/train.s[h]|train.py)|v5_heldout_vide[o]|v5_count_flip_eva[l]|run_beans_evals_hgx[1]|v5_probe_[a-z_]*\.py" >/dev/null \
     && ! pgrep -f "gpu_placeholder_marker_1724905[8]" >/dev/null; then
    JOB=17249058 bash /iris/u/kewalk/memory_project_v5/openpi/cluster_v5/gpu_placeholder_job.sh >> $LOGDIR/sentinel_hgx1.log 2>&1
  fi
  # 2026-09-04 15:58 (user: the 2xH100 job 17267129 was "also placeholder training", ours to use): GPU 1 always
  # holds a busy placeholder; GPU 0 only while no eval python runs (the beans evaluations take GPU 0). The job's
  # batch payload is the user's 1 GB keep-alive (train_hs.sh) and is never touched. The 1-GPU serving job 17192955
  # is gone.
  # 16:22 (user: "use my openpi_trossen repo to keep training"): the placeholder on job 17267129 is a REAL 2-GPU
  # pi0.5 training from openpi_trossen (cluster_v5/placeholder_train_trossen.sh, marker gpu_placeholder_marker_17267129);
  # it is launched only while no v4 step and no eval python is running there (the eval runner kills it first).
  # 19:20: beans B2 trains on this job too (queue_beans5) -> a v5 train.py there also excludes the placeholder.
  # 23:02: job-specific (a v5 training on the 4xH100 job must not suppress this job's placeholder).
  if ! pgrep -f "jobid=17267129 .*(cluster_v5/train.s[h]|train.py)|train.py pi05_yam_mem_v[4]|v4_side_flip_eva[l]|v4_stage2_eva[l]|v5_heldout_vide[o]|v5_count_flip_eva[l]|run_beans_evals_hgx[1]|v5_probe_[a-z_]*\.py" >/dev/null \
     && ! pgrep -f "gpu_placeholder_marker_1726712[9]" >/dev/null; then
    JOB=17267129 bash /iris/u/kewalk/memory_project_v5/openpi/cluster_v5/gpu_placeholder_job.sh >> $LOGDIR/sentinel_hgx1.log 2>&1
  fi
  sleep 60
done
