#!/usr/bin/env bash
# v5 busy placeholder for the single H200 of Slurm job $JOB (default 17207774, iris-hgx-2).
# Holds ~HOLD_GB GiB resident + a 12288^2 fp32 matmul loop, as an --overlap step of the job.
# Run ON the node: ssh iris-hgx-2 'bash /iris/u/kewalk/memory_project_v5/openpi/cluster_v5/gpu_placeholder_hgx2.sh'
# Kill before real work: pkill -f "gpu_placeholder_marke[r]"  (never kill the user's train_hs.py).
export HOME=/iris/u/kewalk
JOB="${JOB:-17207774}"
HOLD_GB="${HOLD_GB:-120}"          # H200 = 143 GB; leave room for the user's 1 GB keep-alive + matmul buffers
VENV="${VENV:-/iris/u/kewalk/memory_project_v4/openpi/.venv}"   # v4 venv: already page-cached on hgx-2
LOG=/iris/u/kewalk/memory_project_v5/v5/tools/logs/placeholder_hgx2.log
# Only THIS job's placeholder counts (marker without a job suffix); other jobs use gpu_placeholder_job.sh.
if pgrep -f "gpu_placeholder_marker[^_]" >/dev/null; then echo "placeholder already running: $(pgrep -f 'gpu_placeholder_marker[^_]' | tr '\n' ' ')"; exit 0; fi
nohup setsid srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
  env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  "$VENV/bin/python" -c "
import time, os  # gpu_placeholder_marker
t0 = time.time()
import jax, jax.numpy as jnp
print('jax', jax.__version__, jax.devices(), 'import+init %.0fs' % (time.time() - t0), flush=True)
n = int(os.environ.get('HOLD_GB', '120')) // 4
hold = jnp.zeros((n, 1024, 1024, 1024), dtype=jnp.float32)  # n x 4 GiB resident
a = jnp.ones((12288, 12288), dtype=jnp.float32)
step = jax.jit(lambda x: x @ x / 12288.0)
jax.block_until_ready(hold)
print('busy placeholder up: holding %d GiB' % (4 * n), flush=True)
while True:
    for _ in range(50):
        a = step(a)
    jax.block_until_ready(a)
    time.sleep(0.03)
" > "$LOG" 2>&1 &
echo "H200 placeholder srun PID=$! (job $JOB, hold ${HOLD_GB} GiB) log=$LOG"
