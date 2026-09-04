#!/usr/bin/env bash
# Busy placeholder for ONE GPU of an arbitrary Slurm job of ours (user 2026-09-04 10:17: a second H200 job on
# iris-hgx-2, 17248791). Marker gpu_placeholder_marker_<JOB> so launchers only kill the placeholder of THEIR job.
#   JOB=17248791 [HOLD_GB=120] [GPU=0] bash cluster_v5/gpu_placeholder_job.sh     (run ON the node)
export HOME=/iris/u/kewalk
JOB="${JOB:?set JOB}"; HOLD_GB="${HOLD_GB:-120}"; GPU="${GPU:-0}"
VENV="${VENV:-/iris/u/kewalk/memory_project_v4/openpi/.venv}"
LOGDIR=/iris/u/kewalk/memory_project_v5/v5/tools/logs; mkdir -p $LOGDIR
marker="gpu_placeholder_marker_${JOB}"
if pgrep -f "$marker" >/dev/null; then echo "placeholder already running for job $JOB: $(pgrep -f "$marker" | tr '\n' ' ')"; exit 0; fi
# 2026-09-04 16:22 (user: "use my openpi_trossen repo to keep training"): on the jobs below the placeholder is a
# REAL pi0.5 training from openpi_trossen (cluster_v5/placeholder_train_trossen.sh), same marker string.
case "$JOB" in
  17267793) gpus=1 ;;   # H200: resumes pi05_pack_with_human_full_0904_h200
  17267129) gpus=2 ;;   # 2xH100: pi05_pack_with_human_full_0904_2h100, batch 16, FSDP 2
  *) gpus="" ;;
esac
if [ -n "$gpus" ]; then
  cd /iris/u/kewalk/memory_project_v5/openpi || exit 2
  setsid nohup bash cluster_v5/placeholder_train_trossen.sh "$JOB" "$gpus" > /dev/null 2>&1 < /dev/null &
  echo "placeholder training (openpi_trossen) for job $JOB on $gpus GPU(s) launched; log=$LOGDIR/placeholder_train_${JOB}.log"
  exit 0
fi
nohup setsid srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
  env CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 HOLD_GB="$HOLD_GB" \
  "$VENV/bin/python" -c "
import time, os  # $marker
import jax, jax.numpy as jnp
n = int(os.environ.get('HOLD_GB', '120')) // 4
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
" > "$LOGDIR/placeholder_${JOB}.log" 2>&1 &
echo "placeholder for job $JOB srun PID=$! (hold ${HOLD_GB} GiB) log=$LOGDIR/placeholder_${JOB}.log"
