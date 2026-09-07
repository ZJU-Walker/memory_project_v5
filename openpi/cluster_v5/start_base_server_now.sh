#!/usr/bin/env bash
# Detach the baseline policy server so it outlives the launching ssh session.
export HOME=/iris/u/kewalk
CK=/iris/u/kewalk/memory_project_v5/v5/checkpoints/pi05_yam_beans0905_base/pi05_beans0905_base_v7rtc_20260906_r1/${STEP:-5000}
LOG=/iris/u/kewalk/memory_project_v5/v5/diagnostics/server_base_ckpt${STEP:-5000}.log
cd /iris/u/kewalk/memory_project_v5/openpi || exit 2
JOB=17286852 GRES=2 GPU=GPU-b3d023a5-d124-a509-7f5a-9fb34083371a NO_PLACEHOLDER=1 PORT=${PORT:-8000} MAXDEC=16 LOG="$LOG" \
  setsid nohup bash cluster_v5/serve_base_job.sh "$CK" pi05_yam_beans0905_base >> "$LOG" 2>&1 &
sleep 3
echo "launched; log $LOG"
