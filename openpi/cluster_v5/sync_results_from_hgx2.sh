#!/usr/bin/env bash
# Copy a run's checkpoints (and its run manifest) from the node-local project copy back to /iris.
#   bash cluster_v5/sync_results_from_hgx2.sh <config-name> <exp-name> [step ...]
# Without steps: every numeric checkpoint dir + v4_run_manifest.json. Run FROM a fast-NFS host.
set -u
export HOME=/iris/u/kewalk
export KRB5CCNAME=FILE:/tmp/krb5cc_24706_claude
config="$1"; exp="$2"; shift 2
local_dir=/scr/kewalk_v5/memory_project_v5/v5/checkpoints/$config/$exp
nfs_dir=/iris/u/kewalk/memory_project_v5/v5/checkpoints/$config/$exp
S() { ssh -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR iris-hgx-2 "$@" 2> >(grep -v "pubkey\|Could not create\|known_hosts" >&2); }
mkdir -p "$nfs_dir"
if [ $# -eq 0 ]; then
  items=$(S "cd $local_dir && ls -d [0-9]* v4_run_manifest.json 2>/dev/null | tr '\n' ' '")
else
  items="$* v4_run_manifest.json"
fi
echo "sync $exp: $items"
t0=$(date +%s)
S "cd $local_dir && tar -cf - $items" | tar -C "$nfs_dir" -xf -
echo "exit=${PIPESTATUS[0]} in $(( $(date +%s) - t0 ))s -> $nfs_dir"; ls "$nfs_dir"
