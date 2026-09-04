# B6a robot deployment (2026-09-04)

Judged policy: config `pi05_yam_mem_v5_stageB6a`, checkpoint `v5/checkpoints/pi05_yam_mem_v5_stageB6a/v5_stageB6a_20260903_r1/keep_499`
(NFS, not in git; 8/8 held-out self-write episodes, 44/45 decisions, 5 writes per episode). Later steps (750, 1000, ...) come
from the continuation run and are untested on the battery.

Server (GPU box, inside the SLURM job that owns the GPU):

    bash openpi/cluster_v5/deploy/run_server_v5.sh 8000 [ckpt_dir]      # CONFIG=... overrides the train config

Client (robot computer, from the openpi directory):

    python examples/yam/client_memory_v5.py --host <server ip> --port 8000 --dry-run
    python examples/yam/client_memory_v5.py --host <server ip> --port 8000 --prompt "find the banana"

`r` resets the bank between trials, `q` quits. The client needs `packages/openpi-client/.../websocket_client_policy.py`
(ping timeout) from this tree. Full history: `openpi/cluster_v5/README.md` §8. Architecture brief: `../docs/v5_architecture_brief.html`.
