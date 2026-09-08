#!/usr/bin/env bash
# Occupy the single H200 of job 17329416 (user 2026-09-08 13:34: "run sth on the job 17329416 to
# occupy the gpu make util higher"). Tuned profile: steady ~85% memory and ~90% utilisation.
# NOTE: job 17267793 is the user's and must NEVER get a placeholder; this is a different job.
export HOME=/iris/u/kewalk
cd /iris/u/kewalk/memory_project_v5/openpi || exit 2
JOB=17329416 \
GPU=GPU-dcaeae50-e4f3-ee9f-02ab-936c7c2b36ba \
MEM_FRAC="${MEM_FRAC:-0.85}" DUTY="${DUTY:-0.92}" HOLD_GB="${HOLD_GB:-32}" \
  bash cluster_v5/gpu_placeholder_tuned.sh
