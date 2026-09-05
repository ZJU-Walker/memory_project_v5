#!/usr/bin/env bash
# Run ON the node that owns the eval GPU. Waits for each judged beans checkpoint and evaluates it there:
#   beans-A keep_499 -> self videos + count-flip (oracle-mode config) ; beans-B keep_499 -> the same.
#   H200 job on iris-hgx-2:  JOB=17267793 RUNNER=hgx2 bash cluster_v5/evals_beans_waiter.sh
#   2xH100 job on iris-hgx-1: JOB=17267129 GRES=2 RUNNER=hgx1 bash cluster_v5/evals_beans_waiter.sh
export HOME=/iris/u/kewalk
JOB="${JOB:?}"; RUNNER="${RUNNER:-hgx2}"; export JOB GRES="${GRES:-1}"
# STAGES="A2 B2" SIDECAR=<v2light sidecar> for the light-state models (19:20).
STAGES="${STAGES:-A B}"; default_sidecar="${SIDECAR:-}"
# per-stage override SIDECAR_<stage> (20:15: A2 = v2light, A3/B3 = v4tray).
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
ckroot=/iris/u/kewalk/memory_project_v5/v5/checkpoints
log=$diag/queue_beans_hgx1.log
echo "evals waiter armed on $(hostname) job $JOB ($RUNNER) stages=$STAGES $(date '+%m/%d %H:%M')" >> $log
for stage in $STAGES; do
  cfg=pi05_yam_mem_v5_beans$stage; exp=v5_beans${stage}_$( case "$stage" in A4|B4|A5|B5) echo 20260905;; *) echo 20260904;; esac )_r1
  v="SIDECAR_$stage"; sc="${!v:-$default_sidecar}"; if [ -n "$sc" ]; then export SIDECAR="$sc"; else unset SIDECAR; fi
  until [ -e $ckroot/$cfg/$exp/keep_499/params ]; do sleep 60; done
  sleep 30  # let the copy settle
  echo "beans-$stage keep_499 present; evals start on job $JOB $(date '+%m/%d %H:%M')" >> $log
  MODES=self BATCHES=24 bash $cv5/run_beans_evals_${RUNNER}.sh $cfg $exp keep_499 >> $diag/chain_${exp}.log 2>&1
  echo "beans-$stage evals: $(tail -1 $diag/videos_${exp}_keep_499/status.log) $(date '+%m/%d %H:%M')" >> $log
done
