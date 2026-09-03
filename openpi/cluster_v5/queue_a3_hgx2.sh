#!/usr/bin/env bash
# Run ON the node: A3 smoke (20 updates) -> A3 r1 (1000 updates) -> batteries (ckpt 999, 500)
# -> held-out videos (ckpt 999, self + oracle) -> placeholder. Sequential; logs to /iris diagnostics.
#   nohup setsid bash cluster_v5/queue_a3_hgx2.sh > /dev/null 2>&1 &
export HOME=/iris/u/kewalk
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
cv5=/iris/u/kewalk/memory_project_v5/openpi/cluster_v5
cfg=pi05_yam_mem_v5_stageA3; exp=v5_stageA3_20260902_r1; smoke=v5_stageA3_20260902_smoke
qlog=$diag/queue_a3.log
echo "queue A3 armed $(date +%H:%M)" >> $qlog
bash $cv5/run_train_h200.sh $cfg $smoke --num-train-steps 20 --save-interval 10 --keep-period 10
code=$(grep "^exit=" $diag/train_${smoke}_status.log | tail -1)
echo "queue: smoke $code at $(date +%H:%M)" >> $qlog
if ! echo "$code" | grep -q "exit=0"; then
  echo "queue: smoke failed; restoring placeholder" >> $qlog
  bash $cv5/gpu_placeholder_hgx2.sh >> $qlog 2>&1; exit 1
fi
echo "queue: launching A3 r1 at $(date +%H:%M)" >> $qlog
bash $cv5/run_train_h200.sh $cfg $exp
code=$(grep "^exit=" $diag/train_${exp}_status.log | tail -1)
echo "queue: A3 r1 $code at $(date +%H:%M)" >> $qlog
if ! echo "$code" | grep -q "exit=0"; then
  echo "queue: training failed; restoring placeholder" >> $qlog
  bash $cv5/gpu_placeholder_hgx2.sh >> $qlog 2>&1; exit 1
fi
bash $cv5/run_batteries_h200.sh $cfg $exp >> $diag/chain_${exp}.log 2>&1
echo "queue: batteries done at $(date +%H:%M)" >> $qlog
bash $cv5/run_videos_hgx2.sh $cfg $exp 999 >> $diag/chain_${exp}.log 2>&1
echo "queue: videos done at $(date +%H:%M); placeholder restored" >> $qlog
