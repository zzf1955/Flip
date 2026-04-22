"""Multi-GPU launcher for sam2_precompute.py.

Spawns N workers across M GPUs, each handling a shard of the full segment
list. Multiple workers can share a GPU (SAM2 tiny uses ~1.8GB, 4090 has 24GB).

Usage:
  # 1 worker per GPU (4 total)
  python -m src.pipeline.batch_sam2_precompute \
      --gpus 0 1 2 3 --task all --resume

  # 2 workers per GPU (8 total)
  python -m src.pipeline.batch_sam2_precompute \
      --gpus 0 1 2 3 --workers-per-gpu 2 --task all --resume
"""

import argparse
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(
        description="Multi-GPU launcher for SAM2 mask precompute")
    ap.add_argument("--gpus", nargs="+", type=int, required=True,
                    help="GPU indices, e.g. 0 1 2 3")
    ap.add_argument("--workers-per-gpu", type=int, default=1,
                    help="number of parallel workers per GPU (default: 1)")
    ap.add_argument("--task", default="all",
                    help="task filter (passed to sam2_precompute)")
    ap.add_argument("--sam2-model", default="tiny",
                    choices=["tiny", "small", "base", "large"])
    ap.add_argument("--prompt-interval", type=int, default=30)
    ap.add_argument("--output", default=None,
                    help="output root (default: training_data/sam2_mask)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-segments", type=int, default=0)
    args = ap.parse_args()

    num_shards = len(args.gpus) * args.workers_per_gpu
    log_dir = os.path.join(
        args.output or os.path.join(BASE_DIR, "training_data", "sam2_mask"),
        "_logs")
    os.makedirs(log_dir, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("LD_PRELOAD",
                   "/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8")
    env.setdefault("no_proxy", "localhost,127.0.0.1")

    procs = []
    shard_idx = 0
    for gpu in args.gpus:
        for w in range(args.workers_per_gpu):
            cmd = [
                sys.executable, "-m", "src.pipeline.sam2_precompute",
                "--task", args.task,
                "--sam2-model", args.sam2_model,
                "--prompt-interval", str(args.prompt_interval),
                "--device", "cuda:0",
                "--shard-index", str(shard_idx),
                "--num-shards", str(num_shards),
            ]
            if args.output:
                cmd += ["--output", args.output]
            if args.resume:
                cmd += ["--resume"]
            if args.max_segments > 0:
                cmd += ["--max-segments", str(args.max_segments)]

            sub_env = dict(env)
            sub_env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            log_path = os.path.join(log_dir, f"shard{shard_idx}_gpu{gpu}.log")
            log_fh = open(log_path, "w")
            log_fh.write(f"$ CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}\n\n")
            log_fh.flush()

            proc = subprocess.Popen(
                cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                env=sub_env, cwd=BASE_DIR)
            procs.append({
                "proc": proc, "gpu": gpu, "shard": shard_idx,
                "log_path": log_path, "log_fh": log_fh, "t0": time.time(),
            })
            print(f"[launch] gpu={gpu} shard={shard_idx}/{num_shards} "
                  f"-> {log_path}", flush=True)
            shard_idx += 1

    print(f"\n{num_shards} workers on {len(args.gpus)} GPUs "
          f"({args.workers_per_gpu}/gpu). Waiting...\n")

    while procs:
        time.sleep(5)
        for item in list(procs):
            rc = item["proc"].poll()
            if rc is None:
                continue
            item["log_fh"].close()
            elapsed = time.time() - item["t0"]
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"[done] gpu={item['gpu']} shard={item['shard']} "
                  f"{status} {elapsed:.0f}s ({elapsed/3600:.1f}h)", flush=True)
            procs.remove(item)

        if procs:
            elapsed_total = time.time() - min(p["t0"] for p in procs)
            alive = len(procs)
            print(f"  ... {alive} workers still running "
                  f"({elapsed_total:.0f}s elapsed)", flush=True)

    print("\nAll shards complete.")


if __name__ == "__main__":
    main()
