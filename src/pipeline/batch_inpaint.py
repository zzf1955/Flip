"""Multi-GPU batch launcher for sam2_inpaint_pipeline.py.

Dispatches N tasks across M GPUs: each task runs as a subprocess bound to one GPU;
a free GPU immediately picks up the next queued task. Output is per-task logs so
progress bars from different tasks don't interleave.

Usage example:
  python -m src.pipeline.batch_inpaint \
    --gpus 1 2 3 \
    --camera-params output/camera_estimation/5point/best_params.json \
    --output-root output/inpaint/v3 \
    --start 0 --duration 5 \
    --tasks \
      G1_WBT_Inspire_Pickup_Pillow_MainCamOnly \
      G1_WBT_Inspire_Put_Clothes_Into_Basket \
      G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly \
      G1_WBT_Inspire_Collect_Clothes_MainCamOnly
"""

import argparse
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_cmd(task, gpu, args):
    out_dir = os.path.join(args.output_root, task)
    # Subprocess is pinned via CUDA_VISIBLE_DEVICES (set in main loop), so
    # inside the process the target GPU is always cuda:0.
    cmd = [
        sys.executable, "-m", "src.pipeline.sam2_inpaint",
        "--task", task,
        "--episode", str(args.episode),
        "--start", str(args.start),
        "--duration", str(args.duration),
        "--inpaint-method", args.inpaint_method,
        "--sam2-model", args.sam2_model,
        "--output-dir", out_dir,
        "--device", "cuda:0",
    ]
    if args.camera_params:
        cmd += ["--camera-params", args.camera_params]
    cmd += [
        "--bbox-margin", str(args.bbox_margin),
        "--min-visible-area", str(args.min_visible_area),
    ]
    return cmd, out_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--gpus", nargs="+", type=int, required=True,
                   help="GPU indices to use, e.g. 1 2 3")
    p.add_argument("--camera-params", type=str, default=None)
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--duration", type=float, default=0)
    p.add_argument("--sam2-model", default="small",
                   choices=["tiny", "small", "base", "large"])
    p.add_argument("--inpaint-method", default="propainter",
                   choices=["lama", "propainter"])
    p.add_argument("--bbox-margin", type=int, default=0)
    p.add_argument("--min-visible-area", type=int, default=50)
    p.add_argument("--log-root", type=str, default=None,
                   help="Directory for per-task stdout logs. "
                        "Default: <output-root>/_logs")
    args = p.parse_args()

    log_root = args.log_root or os.path.join(args.output_root, "_logs")
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(args.output_root, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("LD_PRELOAD",
                   "/home/leadtek/miniconda3/envs/flip/lib/libjpeg.so.8")
    env.setdefault("no_proxy", "localhost,127.0.0.1")

    # free_gpus: pool of currently available GPU indices
    # running: list of (proc, gpu, task, log_path, start_time)
    free_gpus = list(args.gpus)
    queue = list(args.tasks)
    running = []
    results = {}  # task -> (rc, elapsed, log_path)

    def launch_one():
        if not queue or not free_gpus:
            return False
        task = queue.pop(0)
        gpu = free_gpus.pop(0)
        cmd, out_dir = build_cmd(task, gpu, args)
        log_path = os.path.join(log_root, f"{task}.log")
        log_fh = open(log_path, "w")
        # Pin each subprocess to exactly one physical GPU. This prevents
        # PyTorch from touching cuda:0 (occupied by ComfyUI) during init.
        sub_env = dict(env)
        sub_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_fh.write(f"$ CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}\n")
        log_fh.flush()
        proc = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT,
            env=sub_env, cwd=BASE_DIR)
        running.append({
            "proc": proc, "gpu": gpu, "task": task,
            "log_path": log_path, "log_fh": log_fh, "t0": time.time(),
        })
        print(f"[launch] gpu={gpu} task={task} -> {log_path}", flush=True)
        return True

    # Prime the pool
    while launch_one():
        pass

    # Event loop: wait on any child, free GPU, launch next
    while running:
        time.sleep(2)
        for item in list(running):
            rc = item["proc"].poll()
            if rc is None:
                continue
            item["log_fh"].close()
            elapsed = time.time() - item["t0"]
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"[done]  gpu={item['gpu']} task={item['task']} "
                  f"{status} {elapsed:.0f}s", flush=True)
            results[item["task"]] = (rc, elapsed, item["log_path"])
            free_gpus.append(item["gpu"])
            running.remove(item)
            launch_one()

    # Summary
    n_ok = sum(1 for rc, _, _ in results.values() if rc == 0)
    print(f"\n==== Summary: {n_ok}/{len(results)} OK ====")
    for task in args.tasks:
        if task not in results:
            print(f"  {task}: (not run)")
            continue
        rc, el, log = results[task]
        print(f"  {task}: {'OK' if rc==0 else f'FAIL({rc})'} {el:.0f}s  log={log}")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
