#!/usr/bin/env bash
# Mirror everything the v5 runs read at start-up / per batch onto iris-hgx-2's LOCAL disk
# (/scr, 42 TB RAID). The node's NFS client reads /iris at ~2 MB/s under its load (~150), so a
# cold venv import alone took >20 min there and the 39 GB video dataset would take hours/epoch.
# Run FROM a fast-NFS host (iris-ws-18). Small-file trees (the venv) are latency-bound on NFS
# (~30 files/s per stream), so they go as PARALLEL tar streams; big-file trees as one stream.
#   bash cluster_v5/mirror_to_hgx2_scr.sh            (re-runnable; tar overwrites)
# Layout on the node (used by run_train_h200.sh / run_batteries_h200.sh through env overrides):
#   /scr/kewalk_v5/python        uv-managed CPython 3.11.14 (the venv's `home`)
#   /scr/kewalk_v5/venv          copy of memory_project_v5/openpi/.venv, relocated to that python
#   /scr/kewalk_v5/openpi_cache  OPENPI_DATA_HOME (pi05_base params)
#   /scr/kewalk_v5/lerobot       OPENPI_V5_LEROBOT_ROOT (yam/bin_memory_0830_0831_v36_subtask)
set -u
export HOME=/iris/u/kewalk
SSH="bash /iris/u/kewalk/memory_project_v5/openpi/cluster_v5/ssh_hgx2.sh"
export SSH
dst=/scr/kewalk_v5; export dst
log=/iris/u/kewalk/memory_project_v5/v5/diagnostics/mirror_hgx2.log
echo "mirror start $(date +%H:%M:%S)" >> "$log"
$SSH "mkdir -p $dst/python $dst/venv $dst/openpi_cache/openpi-assets/checkpoints $dst/lerobot/yam $dst/jax_cache" >> "$log" 2>&1
stream() {  # src-dir dst-subdir  (one tar stream, whole tree)
  local t0=$(date +%s)
  tar -C "$1" -cf - . | $SSH "mkdir -p $dst/$2 && tar -C $dst/$2 -xf -" >> "$log" 2>&1
  echo "tar $2 exit=${PIPESTATUS[1]} in $(( $(date +%s) - t0 ))s" >> "$log"
}
parallel_stream() {  # src-dir dst-subdir  (top-level entries in batches of 12, 16 streams)
  local t0=$(date +%s)
  export PSRC="$1" PDST="$dst/$2"
  $SSH "mkdir -p $PDST" >> "$log" 2>&1
  ( cd "$PSRC" && ls -A ) | xargs -d '\n' -P 16 -n 12 bash -c 'tar -C "$PSRC" -cf - "$@" | $SSH "tar -C $PDST -xf -"' _ >> "$log" 2>&1
  echo "parallel tar $2 exit=$? in $(( $(date +%s) - t0 ))s" >> "$log"
}
venv=/iris/u/kewalk/memory_project_v5/openpi/.venv
stream /iris/u/kewalk/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/ python/
# venv: everything except site-packages as one stream; site-packages in parallel.
( cd "$venv" && tar -cf - --exclude='./lib/python3.11/site-packages' . ) | $SSH "tar -C $dst/venv -xf -" >> "$log" 2>&1
echo "tar venv-skeleton exit=${PIPESTATUS[1]}" >> "$log"
parallel_stream "$venv/lib/python3.11/site-packages" venv/lib/python3.11/site-packages
# Relocate the venv: interpreter symlinks + pyvenv.cfg home -> the local python.
$SSH "cd $dst/venv && sed -i 's#^home = .*#home = $dst/python/bin#' pyvenv.cfg && for b in python python3 python3.11; do ln -sfn $dst/python/bin/python3.11 bin/\$b; done && bin/python -c 'import sys, jax, openpi; print(\"relocated venv ok\", sys.executable, jax.__version__, openpi.__file__)'" >> "$log" 2>&1
stream /iris/u/kewalk/memory_project_v4/v35/cache/openpi/openpi-assets/checkpoints/pi05_base/ openpi_cache/openpi-assets/checkpoints/pi05_base/
parallel_stream /iris/u/kewalk/memory_project_v4/data/lerobot/yam/bin_memory_0830_0831_v36_subtask lerobot/yam/bin_memory_0830_0831_v36_subtask
$SSH "du -sh $dst/* 2>/dev/null; df -h /scr | tail -1; find $dst/venv -type f | wc -l" >> "$log" 2>&1
echo "mirror done $(date +%H:%M:%S)" >> "$log"
touch /iris/u/kewalk/memory_project_v5/v5/diagnostics/mirror_hgx2.done
