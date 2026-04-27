"""
Batch Seedance generation using overlay (inpaint + SMPLH) videos as input.

Instead of feeding raw robot videos to Seedance, this pipeline uses the
human mesh overlay videos (from segment_pipeline Phase D) as input, asking
Seedance to enhance the synthetic CG human into a photorealistic one.

Input:   training_data/overlay/4s/<task>/ep<N>/seg<M>_human.mp4
Default output:
         training_data/seedance_advance/4s/<task>/ep<N>/seg<M>_human.mp4

For prompt-ablation or overlay-guided experiment outputs, pass --output-root,
for example training_data/seedance_overlay/4s. The directory layout under the
output root stays identical to training_data/seedance_direct/4s.

Usage:
  ARK_API_KEY="ark-xxx" python -m src.pipeline.seedance_advance \
    --task all --resume --workers 3

  python -m src.pipeline.seedance_advance \
    --task Inspire_Pickup_Pillow_MainCamOnly --episode 0 --resume

  python -m src.pipeline.seedance_advance \
    --task all --output-root training_data/seedance_overlay/4s --resume

  python -m src.pipeline.seedance_advance --task all --dry-run

Environment:
  ARK_API_KEY   - Volcengine Ark API key (required unless --dry-run)
  https_proxy   - proxy for catbox upload (default http://127.0.0.1:20171)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import MAIN_ROOT
from src.pipeline.seedance_gen import (
    FFMPEG, FFPROBE, MIN_PIXELS,
    MODEL_STANDARD, MODEL_FAST,
    get_video_info, upload_catbox, create_task, poll_task,
    download, resize_video,
)

_TRAINING_DATA = os.path.join(MAIN_ROOT, "training_data")
INPUT_ROOT = os.path.join(_TRAINING_DATA, "overlay", "4s")
OUTPUT_ROOT = os.path.join(_TRAINING_DATA, "seedance_advance", "4s")

DEFAULT_PROMPT_ADVANCE = (
    "将视频中的合成人体替换为真实的真人。真人为亚洲男性，穿白色短袖T恤，黑色长裤，灰色拖鞋。"
    "双手裸露，没有手套。皮肤有自然的漫反射和高光。"
    "人物动作、姿态和轨迹与原视频完全一致。"
    "背景、地面、家具保持不变。"
)


def discover_tasks() -> list[str]:
    """Return sorted list of task short names that have overlay dirs."""
    if not os.path.isdir(INPUT_ROOT):
        return []
    return sorted(
        d for d in os.listdir(INPUT_ROOT)
        if os.path.isdir(os.path.join(INPUT_ROOT, d))
    )


def collect_overlay_videos(task_short: str,
                           episodes: list[int] | None = None) -> list[str]:
    """Collect overlay human video paths for a task."""
    task_dir = os.path.join(INPUT_ROOT, task_short)
    if not os.path.isdir(task_dir):
        return []

    videos = []
    for root, _, files in os.walk(task_dir):
        for f in files:
            if not f.endswith("_human.mp4"):
                continue
            path = os.path.join(root, f)
            if episodes is not None:
                rel = os.path.relpath(root, task_dir)
                ep_part = rel.split(os.sep)[0]
                try:
                    ep_idx = int(ep_part.replace("ep", ""))
                except ValueError:
                    continue
                if ep_idx not in episodes:
                    continue
            videos.append(path)

    videos.sort()
    return videos


def upscale_for_api(path: str) -> str:
    """Upscale to meet Seedance min pixel requirement (800x600 for 4:3)."""
    info = get_video_info(path)
    pixels = info["width"] * info["height"]
    if pixels >= MIN_PIXELS:
        return path

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    subprocess.check_call([
        FFMPEG, "-y", "-i", path,
        "-vf", "scale=800:600",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        tmp.name,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp.name


_print_lock = threading.Lock()


def _log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def process_one(idx: int, total: int, video_path: str,
                output_path: str, api_key: str, prompt: str,
                model: str, resolution: str) -> dict:
    """Process a single overlay video through Seedance API."""
    tag = f"[{idx+1}/{total}]"
    rel = os.path.relpath(video_path, INPUT_ROOT)
    _log(f"{tag} start: {rel}")

    t0 = time.time()
    upscaled = upscale_for_api(video_path)
    is_tmp = upscaled != video_path

    try:
        url = upload_catbox(upscaled)
        task_id = create_task(api_key, url, prompt, model,
                              resolution, ratio="4:3", duration=4)
        result = poll_task(api_key, task_id, interval=10, timeout=600)
        video_url = result["content"]["video_url"]

        raw_path = output_path + ".raw.mp4"
        download(video_url, raw_path)
        resize_video(raw_path, output_path, 640, 480)
        os.remove(raw_path)

        elapsed = time.time() - t0
        tokens = result["usage"]["total_tokens"]
        _log(f"{tag} done: {elapsed:.0f}s, tokens={tokens}, -> {output_path}")

        return {
            "input": video_path,
            "output": output_path,
            "task_id": task_id,
            "tokens": tokens,
            "seed": result.get("seed"),
            "elapsed": round(elapsed, 1),
            "status": "ok",
        }

    except Exception as e:
        elapsed = time.time() - t0
        _log(f"{tag} FAILED ({elapsed:.0f}s): {e}")
        return {
            "input": video_path,
            "output": output_path,
            "elapsed": round(elapsed, 1),
            "status": "failed",
            "error": str(e),
        }

    finally:
        if is_tmp and os.path.exists(upscaled):
            os.remove(upscaled)


def main():
    p = argparse.ArgumentParser(
        description="Batch Seedance overlay->enhanced human generation")
    p.add_argument("--task", required=True,
                   help="task short name, comma-separated, or 'all'")
    p.add_argument("--episode", type=str, default=None,
                   help="episode indices, e.g. '0' or '0,1,2' (default: all)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT_ADVANCE)
    p.add_argument("--fast", action="store_true",
                   help="use fast model (default: standard)")
    p.add_argument("--resolution", default="480p",
                   choices=["480p", "720p", "1080p", "2k"])
    p.add_argument("--output-root", default=OUTPUT_ROOT,
                   help="output root, defaults to training_data/seedance_advance/4s")
    p.add_argument("--api-key", default=None,
                   help="Ark API key (or set ARK_API_KEY env)")
    p.add_argument("--workers", type=int, default=3,
                   help="number of concurrent API requests (default: 3)")
    p.add_argument("--resume", action="store_true",
                   help="skip videos whose output already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="show work list without calling API")
    args = p.parse_args()

    api_key = args.api_key or os.environ.get("ARK_API_KEY")
    if not api_key and not args.dry_run:
        p.error("provide --api-key or set ARK_API_KEY env variable")

    if args.task == "all":
        tasks = discover_tasks()
    else:
        tasks = [t.strip() for t in args.task.split(",")]

    episodes = None
    if args.episode is not None:
        episodes = [int(e.strip()) for e in args.episode.split(",")]

    model = MODEL_FAST if args.fast else MODEL_STANDARD

    output_root = args.output_root
    if not os.path.isabs(output_root):
        output_root = os.path.join(os.getcwd(), output_root)

    all_videos = []
    for task in tasks:
        videos = collect_overlay_videos(task, episodes)
        for vpath in videos:
            rel = os.path.relpath(vpath, os.path.join(INPUT_ROOT, task))
            out_path = os.path.join(output_root, task, rel)
            all_videos.append((task, vpath, out_path))

    if not all_videos:
        print("no overlay videos found")
        return

    work = []
    skipped = []
    for task, vpath, out_path in all_videos:
        if args.resume and os.path.isfile(out_path):
            skipped.append((task, vpath, out_path))
        else:
            work.append((task, vpath, out_path))

    ep_str = ",".join(str(e) for e in episodes) if episodes else "all"
    print(f"Seedance Advance Batch Generation")
    print(f"  input:    {INPUT_ROOT}")
    print(f"  output:   {output_root}")
    print(f"  tasks:    {', '.join(tasks)}")
    print(f"  episodes: {ep_str}")
    print(f"  model:    {model}")
    print(f"  workers:  {args.workers}")
    print(f"  total:    {len(all_videos)} "
          f"(skip {len(skipped)}, generate {len(work)})")
    print()

    if skipped:
        print(f"--- SKIP ({len(skipped)}) ---")
        for task, vpath, _ in skipped:
            print(f"  {task}/{os.path.relpath(vpath, os.path.join(INPUT_ROOT, task))}")

    print(f"\n--- GENERATE ({len(work)}) ---")
    for i, (task, vpath, out_path) in enumerate(work):
        print(f"  {i+1:3d}. {task}/{os.path.relpath(vpath, os.path.join(INPUT_ROOT, task))}")

    if args.dry_run:
        print(f"\n[dry-run] would generate {len(work)} videos")
        return

    if not work:
        print("\nnothing to generate")
        return

    for _, _, out_path in work:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total = len(work)
    t_total = time.time()

    results = [{"input": vp, "output": op, "status": "skipped"}
               for _, vp, op in skipped]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_one, i, total, task_vp_op[1], task_vp_op[2],
                api_key, args.prompt, model, args.resolution,
            ): task_vp_op
            for i, task_vp_op in enumerate(work)
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    elapsed = time.time() - t_total
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_fail = sum(1 for r in results if r["status"] == "failed")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    total_tokens = sum(r.get("tokens", 0) for r in results)

    print(f"\n{'='*60}")
    print(f"Summary: {n_ok} ok, {n_fail} failed, {n_skip} skipped "
          f"/ {len(results)} total")
    print(f"  tokens: {total_tokens}, time: {elapsed:.0f}s")

    log_path = os.path.join(output_root, "batch_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "tasks": tasks,
            "episodes": episodes,
            "model": model,
            "resolution": args.resolution,
            "prompt": args.prompt,
            "total": len(results),
            "ok": n_ok,
            "failed": n_fail,
            "skipped": n_skip,
            "total_tokens": total_tokens,
            "elapsed_sec": round(elapsed, 1),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  log: {log_path}")


if __name__ == "__main__":
    main()
