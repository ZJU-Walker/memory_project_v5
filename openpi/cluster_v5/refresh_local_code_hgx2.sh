#!/usr/bin/env bash
# Re-stream only the CODE of the v5 project (openpi/ without .venv, i2rt, openpi_test_mem, .git
# pointer, README, .gitignore) to the node-local copy, so the copy matches the branch after a
# commit. Seconds. Run FROM a fast-NFS host.   bash cluster_v5/refresh_local_code_hgx2.sh
set -u
export HOME=/iris/u/kewalk
export KRB5CCNAME=FILE:/tmp/krb5cc_24706_claude
src=/iris/u/kewalk/memory_project_v5; dst=/scr/kewalk_v5/memory_project_v5
t0=$(date +%s)
tar -C "$src" --exclude='./openpi/.venv' --exclude='./v35' --exclude='./v5' --exclude='./data' --exclude='./.claude' -cf - . \
  | ssh -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR iris-hgx-2 "tar -C $dst -xf - && cd $dst && git rev-parse --short HEAD 2>/dev/null" 2> >(grep -v "pubkey\|Could not create\|known_hosts" >&2)
echo "code refreshed (exit=${PIPESTATUS[1]}) in $(( $(date +%s) - t0 ))s; branch HEAD=$(git -C $src rev-parse --short HEAD)"
