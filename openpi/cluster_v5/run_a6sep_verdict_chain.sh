#!/usr/bin/env bash
# Verdict chain for A6sep (sentence-separation loss), 2026-09-05 20:45. Waits for queue14b's "protected as keep_299"
# line, then runs the DECISIVE test first -- scripts/v5_bank_geometry_eval.py, whose go-count readout (A6: 8/12 at
# margin 0.002) is the agreed success criterion -- and only then the self-write rollouts + count-flip battery.
# Runs on the single H200 job 17267793 (user 19:51: all A6 tests go there). Launch from a shell in a job that
# OUTLIVES it (iris-hgx-1 lands in 17249058, ends 09-07).
set -u
JOB="${JOB:-17267793}"
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v5; cd "$root/openpi" || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk XLA_PYTHON_CLIENT_PREALLOCATE=false
log=$root/v5/diagnostics/queue_beans_hgx1.log
ck=$root/v5/checkpoints/pi05_yam_mem_v5_beansA6sep/v5_beansA6sep_20260905_r1/keep_299/params
echo "A6sep verdict chain armed on $(hostname) $(date '+%m/%d %H:%M') (geometry probe -> rollouts, job $JOB)" >> $log
until grep -q "beans-A6sep ckpt-299 protected as keep_299" $log; do sleep 60; done
out=$root/v5/diagnostics/bank_geometry_A6sep_keep_299; mkdir -p $out
echo "A6sep geometry probe started $(date +%H:%M)" >> $out/status.log
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 env CUDA_VISIBLE_DEVICES=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
  .venv/bin/python scripts/v5_bank_geometry_eval.py --config-name pi05_yam_mem_v5_beansA6sep --params "$ck" \
    --split development --alphas 0.01 --output-dir "$out" > "$out/run.log" 2>&1
echo "A6sep geometry probe exit=$? $(date +%H:%M)" >> $out/status.log
echo "A6sep geometry: $(grep -E 'COUNT READOUT' $out/run.log | tail -1) $(date '+%m/%d %H:%M')" >> $log
JOB=$JOB MODES=self BATCHES=24 SIDECAR=$root/openpi/cluster_v5/beans/beans_v5_subtask_labels_v6sub.json \
  bash cluster_v5/run_beans_evals_hgx2.sh pi05_yam_mem_v5_beansA6sep v5_beansA6sep_20260905_r1 keep_299 \
  >> $root/v5/diagnostics/chain_v5_beansA6sep_20260905_r1.log 2>&1
echo "A6sep evals: $(tail -1 $root/v5/diagnostics/videos_v5_beansA6sep_20260905_r1_keep_299/status.log) $(date '+%m/%d %H:%M')" >> $log
