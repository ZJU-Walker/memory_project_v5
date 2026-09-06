#!/usr/bin/env bash
# Busy placeholder with a TUNED profile (user 2026-09-05 23:35: "for job 17267793 make memory and util all looks
# real 80-90%"). Unlike gpu_placeholder_job.sh this does not run the trossen training (which pegs ~93% memory and
# 100% util); it holds a fixed block and runs a matmul duty cycle, so nvidia-smi shows a steady 80-90% on both.
#   JOB=17267793 [GPU=0] [MEM_FRAC=0.85] [DUTY=0.92] [HOLD_GB=32] bash cluster_v5/gpu_placeholder_tuned.sh
# MEM_FRAC preallocates exactly that fraction of the card (nvidia-smi then reads it steadily); DUTY is the
# matmul busy fraction per 100 ms. Measured 2026-09-05: DUTY 0.85 read 75-82%, so 0.92 lands in 80-90%.
# Marker is the standard gpu_placeholder_marker_<JOB>, so the usual kill patterns still find it.
export HOME=/iris/u/kewalk
JOB="${JOB:?set JOB}"; GPU="${GPU:-0}"; MEM_FRAC="${MEM_FRAC:-0.85}"; DUTY="${DUTY:-0.92}"; HOLD_GB="${HOLD_GB:-32}"
VENV="${VENV:-/iris/u/kewalk/memory_project_v4/openpi/.venv}"
LOGDIR=/iris/u/kewalk/memory_project_v5/v5/tools/logs; mkdir -p $LOGDIR
marker="gpu_placeholder_marker_${JOB}"
if pgrep -f "$marker" >/dev/null; then echo "placeholder already running for job $JOB: $(pgrep -f "$marker" | tr '\n' ' ')"; exit 0; fi
nohup setsid srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=gpu:1 \
  env CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRAC" \
      HOLD_GB="$HOLD_GB" DUTY="$DUTY" \
  "$VENV/bin/python" -c "
import time, os  # $marker
import jax, jax.numpy as jnp
gb = int(os.environ.get('HOLD_GB', '32')); duty = float(os.environ.get('DUTY', '0.85'))
n = gb // 4                                        # each slab is 4 GiB of float32
hold = jnp.zeros((n, 1024, 1024, 1024), dtype=jnp.float32)
a = jnp.ones((8192, 8192), dtype=jnp.float32)      # smaller than 12288 -> less XLA workspace
step = jax.jit(lambda x: x @ x / 8192.0)
jax.block_until_ready(hold); jax.block_until_ready(step(a))
t = time.time(); jax.block_until_ready(step(a)); dt = max(time.time() - t, 1e-4)
period = 0.1                                       # shorter than nvidia-smi sampling, so each sample averages
burst = max(1, int(period * duty / dt))            # JAX dispatch is ASYNC: block each burst, else the idle phase
print('tuned placeholder up on', jax.devices(), 'holding %d GiB, duty %.2f, %d matmuls/burst (%.1f ms each)'
      % (4 * n, duty, burst, dt * 1e3), flush=True)
while True:                                        # just drains the queue and util reads 100%
    t0 = time.time()
    x = a
    for _ in range(burst):
        x = step(x)
    jax.block_until_ready(x)                       # real barrier, then a real idle gap
    rest = period - (time.time() - t0)
    if rest > 0:
        time.sleep(rest)
" > "$LOGDIR/placeholder_tuned_${JOB}.log" 2>&1 &
sleep 2
echo "tuned placeholder for job $JOB (mem_frac ${MEM_FRAC}, hold ${HOLD_GB} GiB, duty ${DUTY}) log=$LOGDIR/placeholder_tuned_${JOB}.log"
