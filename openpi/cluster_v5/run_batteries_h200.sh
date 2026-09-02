#!/usr/bin/env bash
# v5 batteries for one run on the H200 of Slurm job $JOB (default 17207774), evaluation only.
#   [JOB=17207774] [STEPS="999 500"] cluster_v5/run_batteries_h200.sh <config-name> <exp-name>
# Per checkpoint: side-flip v2 with semantic / visual / both interventions, Stage-2 battery
# with semantic and visual interventions (its read-accuracy gates are marked not applicable
# for v5). Output dirs mirror the v4 diagnostics naming so the comparison tables line up.
# Run ON the node. Kills the busy placeholder first; restores it when done.
set -u
config="$1"; exp="$2"; JOB="${JOB:-17207774}"; steps="${STEPS:-999 500}"
nfs_root=/iris/u/kewalk/memory_project_v5
local_root=/scr/kewalk_v5/memory_project_v5
if [ -e "$local_root/.staged" ] && [ -x "$local_root/openpi/.venv/bin/python" ]; then root="$local_root"; else root="$nfs_root"; fi
cd "$root/openpi" || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk PYTHONPATH=scripts XLA_PYTHON_CLIENT_PREALLOCATE=false
ck="$root/v5/checkpoints/$config/$exp"
diag="$nfs_root/v5/diagnostics"
status="$diag/batteries_${exp}_status.log"
ph=$(pgrep -f "gpu_placeholder_marke[r]" || true); [ -n "$ph" ] && { kill $ph 2>/dev/null; sleep 5; }
echo "batteries $exp started on $(hostname) at $(date +%H:%M) job=$(grep -oE 'job_[0-9]+' /proc/self/cgroup | sort -u | tr '\n' ' ')" >> "$status"
run() {  # script tag step bank
  local out="$diag/${2}_${exp}_${3}_${4}"
  [ -e "$out/${5}" ] && return
  srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
    env CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python "scripts/$1" --config-name "$config" --params "$ck/$3/params" --batches 12 --batch-size 4 \
      --bank "$4" --output-dir "$out" > "${out}_run.log" 2>&1
  echo "$2 ckpt-$3 bank=$4 exit=$? at $(date +%H:%M)" >> "$status"
}
for step in $steps; do
  run v4_side_flip_eval.py side_flip $step semantic side_flip_eval.json
  run v4_stage2_eval.py stage2_eval $step semantic stage2_eval.json
  run v4_side_flip_eval.py side_flip $step both side_flip_eval.json
  run v4_side_flip_eval.py side_flip $step visual side_flip_eval.json
  run v4_stage2_eval.py stage2_eval $step visual stage2_eval.json
done
touch "$diag/batteries_${exp}.done"
bash cluster_v5/gpu_placeholder_hgx2.sh >> "$status" 2>&1
