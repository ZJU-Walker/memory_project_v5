"""In-process serving smoke for a v5 checkpoint: build the policy exactly as serve_yam_memory.py does,
feed it a few synthetic robot observations (random images, zero state, the task prompt), and print the
decoded sentence, confidence, bank and latency per call. Run on a GPU:

  python scripts/v5_serve_smoke.py --dir <ckpt dir with params/ and assets/> --config pi05_yam_mem_v5_stageB5a
"""

import argparse
import logging
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import serve_yam_memory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", default="find the banana")
    parser.add_argument("--calls", type=int, default=6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    policy = serve_yam_memory.create_policy(serve_yam_memory.Args(dir=args.dir, config=args.config))
    rng = np.random.default_rng(0)
    print("reset ->", policy.infer({"reset_memory": True}), flush=True)
    for i in range(args.calls):
        obs = {
            "observation/image": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            "observation/left_wrist_image": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            "observation/right_wrist_image": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            "observation/state": np.zeros((14,), dtype=np.float32),
            "prompt": args.prompt,
        }
        t0 = time.time()
        out = policy.infer(obs)
        dt = time.time() - t0
        print(
            f"call {i}: {dt * 1000:.0f} ms | subtask={out['subtask']!r} conf={out.get('subtask_confidence', float('nan')):.2f} "
            f"| committed={out.get('memory', {}).get('committed')} bank={out.get('bank')} | actions {np.asarray(out['actions']).shape}",
            flush=True,
        )
    print("reset ->", policy.infer({"reset_memory": True}), flush=True)


if __name__ == "__main__":
    main()
