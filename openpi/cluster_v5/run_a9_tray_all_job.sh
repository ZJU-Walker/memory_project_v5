#!/usr/bin/env bash
# A9 tray-flip probe v2 on ALL tray arrivals (dump- and done-labelled; needs the v5_step_ce_steps output, 2026-09-06
# 05:02), development then train, on the pinned GPU of a job, after the A9 probe chain and the A9 oracle rollouts.
#   JOB=17286852 GRES=2 GPU=<uuid> setsid nohup bash cluster_v5/run_a9_tray_all_job.sh > /dev/null 2>&1 < /dev/null &
set -u
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v5; cd "$root/openpi" || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk XLA_PYTHON_CLIENT_PREALLOCATE=false
diag=$root/v5/diagnostics; log=$diag/queue_beans_hgx1.log
ck=$root/v5/checkpoints/pi05_yam_mem_v5_beansA9/v5_beansA9_20260906_r1/keep_299/params; cfg=pi05_yam_mem_v5_beansA9
echo "A9 tray-flip v2 (all tray arrivals) armed on $(hostname) job $JOB: waits for the A9 probe chain + oracle rollouts $(date '+%m/%d %H:%M')" >> $log
until grep -q "^A9 probe chain done" $log && grep -q "all videos done" $diag/videos_v5_beansA9_20260906_r1_keep_299_oracle/status.log 2>/dev/null; do sleep 60; done
for split in development train; do
  out=$diag/tray_flip_A9_keep_299_${split}_all; mkdir -p "$out"
  echo "tray probe v2 A9 $split started $(date +%H:%M)" >> "$out/status.log"
  srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:"$GRES" env CUDA_VISIBLE_DEVICES="$GPU" \
    .venv/bin/python scripts/v5_tray_flip_eval.py --config-name "$cfg" --params "$ck" --split $split --alphas 0.01,0.0 --batches 24 --output-dir "$out" > "$out/run.log" 2>&1
  rc=$?; echo "tray probe v2 exit=$rc $(date +%H:%M)" >> "$out/status.log"; echo "A9 tray-flip v2 $split exit=$rc $(date '+%m/%d %H:%M')" >> $log
done
echo "A9 tray-flip v2 done; nothing else launched on job $JOB $(date '+%m/%d %H:%M')" >> $log
