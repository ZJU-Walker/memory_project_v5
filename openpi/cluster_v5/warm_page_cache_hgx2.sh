#!/usr/bin/env bash
# Pre-read, on the node, every file a v5 run touches so it lands in the node's page cache
# (no local copy, nothing written): iris-hgx-2's NFS client is latency-bound (~1 MB/s per
# stream), so 32 parallel readers raise the aggregate rate. Order: libraries, weights, Arrow
# dataset cache, source. Re-runnable; cached files cost nothing the second time.
#   ssh iris-hgx-2 'nohup setsid nice -n 19 bash .../warm_page_cache_hgx2.sh > /dev/null 2>&1 &'
export HOME=/iris/u/kewalk
log=/iris/u/kewalk/memory_project_v5/v5/diagnostics/warm_page_cache_hgx2.log
warm() {  # label dir   (streaming: files are read as find lists them; no up-front du/stat walk)
  local t0=$(date +%s)
  local n
  n=$(find "$2" -type f -print0 2>/dev/null | tee >(xargs -0 -P 32 -n 8 cat > /dev/null 2>/dev/null) | tr -cd '\0' | wc -c)
  wait
  echo "$1: $n files in $(( $(date +%s) - t0 ))s $(date +%H:%M:%S)" >> "$log"
}
echo "warm start $(date +%H:%M:%S) on $(hostname)" >> "$log"
warm venv-v4-libs   /iris/u/kewalk/memory_project_v4/openpi/.venv/lib/python3.11/site-packages
warm v5-src         /iris/u/kewalk/memory_project_v5/openpi/src
warm weights        /iris/u/kewalk/memory_project_v5/v35/cache/openpi
warm hf-hub         /iris/u/kewalk/memory_project_v5/v35/cache/huggingface/hub
warm dataset-meta   /iris/u/kewalk/memory_project_v4/data/lerobot/yam/bin_memory_0830_0831_v36_subtask/meta
warm arrow-cache    /iris/u/kewalk/memory_project_v5/v35/cache/huggingface/datasets/parquet
warm dataset-parquet /iris/u/kewalk/memory_project_v4/data/lerobot/yam/bin_memory_0830_0831_v36_subtask/data
echo "warm done $(date +%H:%M:%S)" >> "$log"
