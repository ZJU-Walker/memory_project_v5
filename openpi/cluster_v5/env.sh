#!/usr/bin/env bash

# v5 runtime environment (cluster_v5/README.md §1). Reuses the portable v3.5 env contract
# verbatim: every mutable cache path derives from THIS checkout's root, so sourcing it from the
# v5 worktree yields worktree-local caches (fresh JAX cache -- never shared with a concurrent
# JAX process, which corrupts it). Read-only caches (OpenPI assets, HF, uv) are symlinked to the
# v4 tree's copies; `data` is the sanctioned shared link. v5 run artifacts live under `v5/`.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source cluster_v5/env.sh instead of executing it" >&2
  exit 2
fi

v5_env_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# The login HOME on the iris cluster is a tiny-quota AFS directory; anything that defaults a
# cache to ~ dies with "Disk quota exceeded". Every v5 launcher runs with the NFS home.
export HOME=/iris/u/kewalk
# shellcheck source=../cluster_v35/env.sh
source "${v5_env_dir}/../cluster_v35/env.sh"

mkdir -p \
  "${MEMORY_PROJECT_ROOT}/v5/assets" \
  "${MEMORY_PROJECT_ROOT}/v5/checkpoints" \
  "${MEMORY_PROJECT_ROOT}/v5/diagnostics"

unset v5_env_dir
