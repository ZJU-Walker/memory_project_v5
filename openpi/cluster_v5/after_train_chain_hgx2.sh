#!/usr/bin/env bash
# Run ON the node after launching a training run: wait for its status "exit=", then run the
# batteries on ckpt 999/500 (cluster_v5/run_batteries_h200.sh) and restore the placeholder.
#   nohup setsid bash cluster_v5/after_train_chain_hgx2.sh <config> <exp> &
export HOME=/iris/u/kewalk
config="$1"; exp="$2"
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
status="$diag/train_${exp}_status.log"
while ! grep -q "^exit=" "$status" 2>/dev/null; do sleep 60; done
code=$(grep "^exit=" "$status" | tail -1)
echo "chain: training finished ($code) at $(date +%H:%M); starting batteries" >> "$diag/chain_${exp}.log"
if echo "$code" | grep -q "exit=0"; then
  bash /iris/u/kewalk/memory_project_v5/openpi/cluster_v5/run_batteries_h200.sh "$config" "$exp" >> "$diag/chain_${exp}.log" 2>&1
else
  echo "chain: training did not exit 0; batteries skipped; placeholder restored" >> "$diag/chain_${exp}.log"
  bash /iris/u/kewalk/memory_project_v5/openpi/cluster_v5/gpu_placeholder_hgx2.sh >> "$diag/chain_${exp}.log" 2>&1
fi
echo "chain: done at $(date +%H:%M)" >> "$diag/chain_${exp}.log"
