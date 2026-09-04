#!/usr/bin/env bash
# Run ON iris-hgx-1: beans norm stats (CPU, random 3k-frame subset of the converted dataset), then the
# beans A -> B queue (cluster_v5/queue_beans_hgx1.sh: smoke -> A -> B -> continuation) on the 4xH100 job.
#   setsid nohup bash cluster_v5/launch_beans_hgx1.sh > v5/diagnostics/launch_beans_hgx1.out 2>&1 &
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v5
cd "$root/openpi" || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk
log=$root/v5/diagnostics/queue_beans_hgx1.log
# compute_norm_stats.py writes under <assets_base_dir>/<config name>/<repo_id>; the beans DATA config reads its
# stats from AssetsConfig(assets_dir=v5/assets/pi05_yam_bean_scoop_0902_v5)/<repo_id>, shared by beansA and beansB.
written=$root/v5/assets/pi05_yam_mem_v5_beansA/yam/bean_scoop_0902_v5/norm_stats.json
stats=$root/v5/assets/pi05_yam_bean_scoop_0902_v5/yam/bean_scoop_0902_v5/norm_stats.json
if [ ! -e "$stats" ]; then
  if [ ! -e "$written" ]; then
    echo "norm stats start $(date '+%m/%d %H:%M')" >> $log
    JAX_PLATFORMS=cpu .venv/bin/python scripts/compute_norm_stats.py --config-name pi05_yam_mem_v5_beansA --max-frames 3000 \
      > $root/v5/diagnostics/norm_stats_beans.log 2>&1
    echo "norm stats exit=$? $(date '+%m/%d %H:%M') $( [ -e "$written" ] && echo written || echo MISSING)" >> $log
    [ -e "$written" ] || exit 1
  fi
  mkdir -p "$(dirname "$stats")" && cp "$(dirname "$written")"/* "$(dirname "$stats")"/ && echo "norm stats copied to the data assets dir" >> $log
fi
exec bash cluster_v5/queue_beans_hgx1.sh
