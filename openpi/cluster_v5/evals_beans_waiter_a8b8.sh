#!/usr/bin/env bash
# (a8b8 copy, 22:05: waits for the H200 to be free of trainings; the running original cannot be edited)
# Run ON the node that owns the eval GPU. Waits for each judged beans checkpoint and evaluates it there:
#   beans-A keep_499 -> self videos + count-flip (oracle-mode config) ; beans-B keep_499 -> the same.
#   H200 job on iris-hgx-2:  JOB=17267793 RUNNER=hgx2 bash cluster_v5/evals_beans_waiter.sh
#   2xH100 job on iris-hgx-1: JOB=17267129 GRES=2 RUNNER=hgx1 bash cluster_v5/evals_beans_waiter.sh
export HOME=/iris/u/kewalk
JOB="${JOB:?}"; RUNNER="${RUNNER:-hgx2}"; export JOB GRES="${GRES:-1}"
# STAGES="A2 B2" SIDECAR=<v2light sidecar> for the light-state models (19:20).
STAGES="${STAGES:-A B}"; default_sidecar="${SIDECAR:-}"; STEP="${STEP:-499}"  # STEP=299 for the 300-update A6sd/B6sd
# per-stage override SIDECAR_<stage> (20:15: A2 = v2light, A3/B3 = v4tray).
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/iris/u/kewalk/memory_project_v5/v5/checkpoints
log=$diag/queue_beans_hgx1.log
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
echo "evals waiter armed on $(hostname) job $JOB ($RUNNER) stages=$STAGES $(date '+%m/%d %H:%M')" >> $log
for stage in $STAGES; do
  cfg=pi05_yam_mem_v5_beans$stage; exp=v5_beans${stage}_$( case "$stage" in A4|B4|A5|B5|B5d1|A6|B6|A7|B7|A6d1|B6d1|A6sd|B6sd|A6sep|B6sep|A8|B8) echo 20260905;; *) echo 20260904;; esac )_r1
  v="SIDECAR_$stage"; sc="${!v:-$default_sidecar}"; if [ -n "$sc" ]; then export SIDECAR="$sc"; else unset SIDECAR; fi
  # 2026-09-05 16:45: wait for the queue runner's "protected" line (written after `cp -r` returns), not for the
  # folder: polling keep_499/params started evals on half-copied checkpoints twice today.
  until grep -q "beans-$stage ckpt-$STEP protected as keep_$STEP" $log; do sleep 60; done
  sleep 30  # let the copy settle
  # 2026-09-05 22:05: the single H200 is shared with the other session's trainings (A6sep, then the A6ctl control,
  # batch 2 x 300 updates, ~90 min each). Never start rollouts on top of a training step of this job: wait until
  # no v5 train.py srun step of the job is alive (the trossen placeholder's train.py is excluded by its marker).
  while job_busy; do sleep 60; done
  echo "beans-$stage keep_$STEP present; evals start on job $JOB $(date '+%m/%d %H:%M')" >> $log
  MODES=self BATCHES=24 bash $cv5/run_beans_evals_${RUNNER}.sh $cfg $exp keep_$STEP >> $diag/chain_${exp}.log 2>&1
  echo "beans-$stage evals: $(tail -1 $diag/videos_${exp}_keep_$STEP/status.log) $(date '+%m/%d %H:%M')" >> $log
done
