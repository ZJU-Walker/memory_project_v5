#!/bin/bash
# LeRobot conversion of the 0908 task1 collection (episode view, 71 episodes). Run on a compute node
# (iris-hgx-1: the login node killed the 0902 conversion, the workstation OOMs), e.g.
#   ssh iris-hgx-1 'nohup setsid bash /iris/u/kewalk/memory_project_v5/openpi/cluster_v5/task1/convert_task1_hgx1.sh > <log> 2>&1 &'
set -euo pipefail
export HOME=/iris/u/kewalk
export PYTHONDONTWRITEBYTECODE=1
export HF_LEROBOT_HOME=/iris/u/kewalk/memory_project_v5/v5/data/lerobot
cd /iris/u/kewalk/memory_project_v5/openpi
find examples/yam src/openpi -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "[$(date)] host $(hostname) start conversion"
.venv/bin/python examples/yam/convert_yam_data_to_lerobot.py \
  --episode-manifest /iris/u/kewalk/memory_project/data/0908_task1_episode_manifest_v1.json \
  --repo-name yam/task1_find_0908_v5 "$@"
echo "[$(date)] conversion finished rc=$?"
touch /iris/u/kewalk/memory_project/data/0908_task1_inspection/logs/convert_task1.done
