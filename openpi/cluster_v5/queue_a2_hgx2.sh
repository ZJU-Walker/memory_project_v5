#!/usr/bin/env bash
# Run ON the node: wait until the r1 battery chain is done AND the r2 code refresh marker exists,
# then run the A2 smoke (20 updates); if it exits 0, launch A2 r1 (1000 updates) with its
# post-training battery chain. Everything logs to /iris diagnostics.
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
qlog=$diag/queue_a2.log
echo "queue armed $(date +%H:%M)" >> $qlog
while ! grep -q "chain: done" $diag/chain_v5_stageA_20260902_r1.log 2>/dev/null || [ ! -e $diag/r2_code_ready ]; do sleep 60; done
echo "queue: chain done + r2 code ready at $(date +%H:%M); launching A2 smoke" >> $qlog
bash $cv5/run_train_h200.sh pi05_yam_mem_v5_stageA2 v5_stageA2_20260902_smoke --num-train-steps 20 --save-interval 10 --keep-period 10
code=$(grep "^exit=" $diag/train_v5_stageA2_20260902_smoke_status.log | tail -1)
echo "queue: smoke $code at $(date +%H:%M)" >> $qlog
if echo "$code" | grep -q "exit=0"; then
  nohup setsid bash $cv5/after_train_chain_hgx2.sh pi05_yam_mem_v5_stageA2 v5_stageA2_20260902_r1 > /dev/null 2>&1 &
  echo "queue: launching A2 r1 at $(date +%H:%M)" >> $qlog
  bash $cv5/run_train_h200.sh pi05_yam_mem_v5_stageA2 v5_stageA2_20260902_r1
  echo "queue: A2 r1 $(grep '^exit=' $diag/train_v5_stageA2_20260902_r1_status.log | tail -1) at $(date +%H:%M)" >> $qlog
else
  echo "queue: smoke failed; restoring placeholder" >> $qlog
  bash $cv5/gpu_placeholder_hgx2.sh >> $qlog 2>&1
fi
