#!/usr/bin/env bash
# Robot policy server for the NON-MEMORY pi05 baseline. Sibling of serve_v5_job_v2.sh, but runs
# scripts/serve_yam_subtask.py (no bank, no memory prefix, no RTC action-prefix on the wire) instead
# of serve_yam_memory.py.
#   JOB=17286852 GRES=2 GPU=GPU-b3d0... NO_PLACEHOLDER=1 [PORT=8000] [MAXDEC=16] \
#     bash cluster_v5/serve_base_job.sh <ckpt dir with params/ and assets/> <config-name>
# MAXDEC matters: the script default of 10 truncates the v7 target-carry sentences, which reach 13
# PaliGemma tokens ("scoop 1 of 2: dig and" instead of "... dig and carry").
export HOME=/iris/u/kewalk
ck="$1"; cfg="$2"; port="${PORT:-8000}"; JOB="${JOB:?set JOB}"; GPU="${GPU:-0}"; GRES="${GRES:-1}"
MAXDEC="${MAXDEC:-16}"
cd /iris/u/kewalk/memory_project_v5/openpi || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
[ -d "$ck/params" ] || { echo "no params/ under $ck"; exit 2; }
if [ -z "${NO_PLACEHOLDER:-}" ]; then
  ph=$(pgrep -f "^srun .*gpu_placeholder_marker_${JOB}" || true)
  [ -n "$ph" ] && { kill $ph 2>/dev/null; sleep 8; echo "killed job $JOB placeholder: $ph"; }
fi
log="${LOG:-/iris/u/kewalk/memory_project_v5/v5/diagnostics/server_base_$(date +%Y%m%d_%H%M).log}"
echo "serving $cfg from $ck on $(hostname) ($(hostname -I | awk '{print $1}')) port $port (job $JOB gpu $GPU maxdec $MAXDEC); log $log" | tee -a "$log"
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:"$GRES" \
  env CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 PYTHONPATH=scripts \
  .venv/bin/python -u scripts/serve_yam_subtask.py --dir "$ck" --config "$cfg" --port "$port" \
    --max-decode-steps "$MAXDEC" ${SERVE_EXTRA:-} 2>&1 | tee -a "$log"
