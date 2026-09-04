#!/usr/bin/env bash
# Run ON iris-hgx-1: beans norm stats (CPU, random 15k-frame subset of the converted dataset), then the
# beans A -> B queue (cluster_v5/queue_beans_hgx1.sh: smoke -> A -> B -> continuation) on the 4xH100 job.
#   setsid nohup bash cluster_v5/launch_beans_hgx1.sh > v5/diagnostics/launch_beans_hgx1.out 2>&1 &
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v5
cd "$root/openpi" || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk
log=$root/v5/diagnostics/queue_beans_hgx1.log
stats=$root/v5/assets/pi05_yam_bean_scoop_0902_v5/yam/bean_scoop_0902_v5/norm_stats.json
if [ ! -e "$stats" ]; then
  echo "norm stats start $(date '+%m/%d %H:%M')" >> $log
  JAX_PLATFORMS=cpu .venv/bin/python scripts/compute_norm_stats.py --config-name pi05_yam_mem_v5_beansA --max-frames 15000 \
    > $root/v5/diagnostics/norm_stats_beans.log 2>&1
  echo "norm stats exit=$? $(date '+%m/%d %H:%M') $( [ -e "$stats" ] && echo present || echo MISSING)" >> $log
  [ -e "$stats" ] || exit 1
fi
exec bash cluster_v5/queue_beans_hgx1.sh
