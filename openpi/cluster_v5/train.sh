#!/usr/bin/env bash
# Generic v5 launcher (v5 light protocol = v4's; see cluster_v5/README.md §1, §6):
#   cluster_v5/train.sh <config-name> --exp-name <exp> [extra train.py args]
# Stage recipes: pi05_yam_mem_v5_stageA (oracle sentence writes, visual bank off),
# pi05_yam_mem_v5_stageB (predicted sentence writes), pi05_yam_mem_v5_stageC (+ visual bank).
set -euo pipefail

v5_launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=env.sh
source "${v5_launcher_dir}/env.sh"
if [[ -n "${OPENPI_V5_LEROBOT_ROOT:-}" ]]; then
  echo "v5: LeRobot dataset root override -> ${OPENPI_V5_LEROBOT_ROOT}" >&2
fi

if [[ $# -lt 1 ]]; then
  echo "usage: cluster_v5/train.sh <config-name> --exp-name <exp> [args]" >&2
  exit 2
fi
v5_config="$1"; shift

v5_python="${V5_PYTHON:-${MEMORY_PROJECT_ROOT}/openpi/.venv/bin/python}"
if [[ ! -x "${v5_python}" ]]; then
  echo "v5 Python is not executable: ${v5_python}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
cd -- "${MEMORY_PROJECT_ROOT}/openpi"
exec "${v5_python}" scripts/train.py "${v5_config}" "$@"
