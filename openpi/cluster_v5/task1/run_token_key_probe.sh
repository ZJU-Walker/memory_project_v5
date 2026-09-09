#!/bin/bash
# Bank-level probe of token-level contextual keys (scripts/v5_token_key_probe.py) on the B9-2000 encoder.
# Light GPU use (one 3B forward over ~60 sentences); run on the spare memory of a node GPU, no placeholder needed.
set -euo pipefail
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
cd /iris/u/kewalk/memory_project_v5/openpi
find scripts src/openpi -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "[$(date)] host $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-all}"
.venv/bin/python scripts/v5_token_key_probe.py \
  --params /iris/u/kewalk/memory_project_v5/v5/checkpoints/pi05_yam_mem_v5_beansB9/v5_beansB9_20260906_r1/2000/params \
  --config-name pi05_yam_mem_v5_beansB9 \
  --task1-sidecar cluster_v5/task1/task1_v5_subtask_labels_v1.json \
  --beans-sidecar cluster_v5/beans/beans_v5_subtask_labels_0905_v7tgt.json \
  --oldbin-sidecar /iris/u/kewalk/memory_project_v5/data/v5_subtask_labels_0830_0831.json \
  --output-dir /iris/u/kewalk/memory_project_v5/v5/diagnostics/token_key_probe_B9_2000 "$@"
echo "[$(date)] probe finished rc=$?"
