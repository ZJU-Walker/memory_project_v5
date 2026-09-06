#!/usr/bin/env bash
# Generic GPU sentinel for ONE Slurm job of ours (2026-09-05 21:55; replaces gpu_sentinel_hgx1.sh / gpu_sentinel_hgx2.sh /
# gpu_sentinel_hgx2b.sh). Standing rule: every idle GPU of ours runs the openpi_trossen placeholder; real work always wins.
#   JOB=17267793 TAG=hgx2 setsid nohup bash cluster_v5/gpu_sentinel_job.sh > /dev/null 2>&1 < /dev/null &   (run ON the node)
# Every 20 s:
#   * real work running and a placeholder of this job running  -> kill the placeholder (marker gpu_placeholder_marker_<JOB>;
#     never the user's train_hs.py payload, which carries no marker);
#   * no real work and no placeholder                          -> launch it (cluster_v5/gpu_placeholder_job.sh, whose
#     trossen step refuses to start if a GPU of the job already holds > 8 GB: second guard, independent of this list).
# Real work = any srun step of THIS job running a v5 training or a scripts/v5_*.py script, any scripts/v5_*.py python on
# the node (conservative), the eval/probe runners that stay alive between their steps, and the robot server. The
# 21:25 incident: the old hgx-2 sentinel matched a fixed list of scripts, missed v5_bank_geometry_eval.py, and
# relaunched the placeholder during the A6sep geometry probe; the rollouts that followed OOM'd.
export HOME=/iris/u/kewalk
JOB="${JOB:?set JOB}"; TAG="${TAG:?set TAG (hgx1|hgx2)}"
LOGDIR=/iris/u/kewalk/memory_project_v5/v5/tools/logs; mkdir -p $LOGDIR; LOG=$LOGDIR/sentinel_${TAG}.log
MARKER="gpu_placeholder_marker_${JOB}"
REAL="jobid=${JOB} .*(trai[n]|scripts/v5_[a-z0-9_]*\.p[y])|scripts/v5_[a-z0-9_]*\.p[y] |v5_heldout_vide[o]|v5_count_flip_eva[l]|v5_bank_geometry_eva[l]|v5_tray_flip_eva[l]|v5_probe_[a-z_]*\.py|run_beans_evals_hgx[12]|run_count_flip_varian[t]|run_count_recover[y]|run_[a-z0-9]*_verdict_chai[n]|serve_yam_memor[y]"
# A python of THIS job that is neither the placeholder (env GPU_PLACEHOLDER / marker in its command) nor the user's
# train_hs.py keep-alive = real work, whatever its command line looks like. Needed because an srun step launched
# from ANOTHER node (the other session drives job 17267793 from hgx-1) leaves no "--jobid=" text on this node.
job_busy() {
  local p
  for p in $(pgrep -f "pytho[n]"); do
    grep -qz "^SLURM_JOB_ID=${JOB}$" /proc/$p/environ 2>/dev/null || continue
    grep -qz "^GPU_PLACEHOLDER=" /proc/$p/environ 2>/dev/null && continue
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -qE "train_hs\.py|gpu_placeholder_marker" && continue
    return 0
  done
  return 1
}
echo "$(date '+%m/%d %H:%M') sentinel (generic, bidirectional, job-env check) up for job $JOB on $(hostname) pid=$$" >> $LOG
while true; do
  busy=0; pgrep -af "$REAL" | grep -v "$MARKER" | grep -q . && busy=1
  [ $busy = 0 ] && job_busy && busy=1
  # The placeholder is the srun step whose command line carries the marker (env GPU_PLACEHOLDER=<marker> for the
  # trossen training, "# <marker>" for the legacy busy loop). Kill ONLY srun processes: a shell whose command text
  # merely mentions the marker (an ssh audit, a launcher) must never be killed (21:52: the first version killed its
  # own ssh shell that way). Killing the srun cancels the step; placeholder_train_trossen.sh then deletes the ckpt dir.
  ph=$(pgrep -f "^srun .*$MARKER" | tr '\n' ' ')
  any=$(pgrep -f "$MARKER" | tr '\n' ' ')
  if [ $busy = 1 ] && [ -n "$ph" ]; then
    echo "$(date '+%m/%d %H:%M:%S') real work detected while the placeholder runs -> killing placeholder srun pids $ph" >> $LOG
    kill $ph 2>/dev/null
  elif [ $busy = 0 ] && [ -z "$any" ]; then
    echo "$(date '+%m/%d %H:%M:%S') job $JOB idle -> launching placeholder" >> $LOG
    JOB=$JOB bash /iris/u/kewalk/memory_project_v5/openpi/cluster_v5/gpu_placeholder_job.sh >> $LOG 2>&1
  fi
  sleep 20
done
