"""Seedance robot-arm to human-hand smoke pipeline for H2R HDF5 videos.

This entry is intentionally separate from ``seedance_batch.py`` because the
existing G1 Seedance path assumes 4:3 segment mp4 files and a full-body human
prompt.  H2R camera videos are 16:9 HDF5 streams and currently need a small
robot-arm/hand edit smoke before any Wan cache or training work.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import h5py
import numpy as np

from src.core.config import MAIN_ROOT
from src.pipeline.seedance_gen import (
    MIN_PIXELS,
    MODEL_FAST,
    MODEL_STANDARD,
    ark_request,
    create_task,
    download,
    get_video_info,
    upload_catbox,
)


DEFAULT_PROMPT = (
    "视频编辑任务：把画面中原有的机器人机械臂、黑色两指夹爪和白色机械外壳完全擦除，"
    "并在完全相同的位置替换成一只真实裸露的人类手和前臂。"
    "手掌和手指必须覆盖原夹爪位置，前臂从原机械臂进入画面的同一边缘伸入，"
    "沿用原机械臂的运动轨迹、朝向、抓取点、接触关系和遮挡关系。"
    "保持桌面、背景、方块、杯子、托盘、杆、相机视角、光照、阴影和所有物体位置完全不变。"
    "禁止在夹爪旁边、桌面空白处或画面边缘额外生成手；不要出现第二只手、袖子、手套、"
    "完整人物、头、脸、身体、机器人夹爪、机械零件、金属或塑料残留。"
)

DEFAULT_SAMPLES = (
    "grab_both_cubes_v1:0:0",
    "grab_cup_v1:0:0",
    "roll:0:0",
)
DEFAULT_DATA_ROOT = Path(MAIN_ROOT) / "data" / "h2r" / "v1" / "data"
DEFAULT_OUTPUT_ROOT = Path(MAIN_ROOT) / "tmp" / "h2r_seedance_edit_smoke"
DEFAULT_FPS = 30
DEFAULT_NUM_FRAMES = 120


@dataclass(frozen=True)
class SampleSpec:
    task: str
    episode: int
    start_frame: int

    @property
    def sample_id(self) -> str:
        return f"{self.task}_ep{self.episode:06d}_f{self.start_frame:06d}"


def parse_hxw(value: str) -> tuple[int, int]:
    parts = value.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise ValueError(f"size must be HxW, got {value!r}")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"size dimensions must be positive, got {value!r}")
    if height % 2 or width % 2:
        raise ValueError(f"H.264 yuv420p output needs even H/W, got {value!r}")
    return height, width


def parse_sample(value: str) -> SampleSpec:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"sample must be task:episode:start_frame, got {value!r}")
    task = parts[0].strip()
    if not task:
        raise ValueError(f"sample task is empty: {value!r}")
    return SampleSpec(task=task, episode=int(parts[1]), start_frame=int(parts[2]))


def load_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            raise ValueError(f"Invalid .env line without '=': {path}:{line_no}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid .env empty key: {path}:{line_no}")
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        values[key] = value
    return values


def hdf5_path(data_root: Path, sample: SampleSpec) -> Path:
    return data_root / sample.task / f"episode_{sample.episode}.hdf5"


def camera_to_uint8(frames: np.ndarray, *, source: str) -> tuple[np.ndarray, dict]:
    info = {
        "dtype": str(frames.dtype),
        "min": int(frames.min()),
        "max": int(frames.max()),
    }
    if frames.dtype == np.uint8:
        return frames, info
    if np.issubdtype(frames.dtype, np.integer):
        if frames.min() < 0 or frames.max() > 255:
            raise ValueError(
                f"{source} cannot be cast to uint8 without rescaling: "
                f"dtype={frames.dtype} min={frames.min()} max={frames.max()}"
            )
        return frames.astype(np.uint8), info
    raise ValueError(f"{source} unsupported dtype: {frames.dtype}")


def aspect_fill_resize_crop(
    frames_rgb: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, dict]:
    src_h, src_w = frames_rgb.shape[1:3]
    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(target_w, int(round(src_w * scale)))
    new_h = max(target_h, int(round(src_h * scale)))
    interpolation = cv2.INTER_AREA if new_w <= src_w and new_h <= src_h else cv2.INTER_LINEAR
    resized = np.empty((frames_rgb.shape[0], new_h, new_w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames_rgb):
        resized[i] = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    out = resized[:, top:top + target_h, left:left + target_w]
    if out.shape[1:3] != (target_h, target_w):
        raise RuntimeError(
            f"bad crop shape {out.shape[1:3]} for target {(target_h, target_w)}"
        )
    return out, {
        "mode": "aspect_fill_center_crop",
        "source_h": src_h,
        "source_w": src_w,
        "target_h": target_h,
        "target_w": target_w,
        "scale": scale,
        "resized_h": new_h,
        "resized_w": new_w,
        "crop_top": top,
        "crop_left": left,
        "crop_bottom_exclusive": top + target_h,
        "crop_right_exclusive": left + target_w,
        "source_aspect": src_w / src_h,
        "target_aspect": target_w / target_h,
    }


def write_video_rgb(frames_rgb: np.ndarray, out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3] frames, got {frames_rgb.shape}")
    height, width = frames_rgb.shape[1:3]
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = int(width)
    stream.height = int(height)
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}
    try:
        for frame_rgb in frames_rgb:
            frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def read_h2r_camera_clip(
    h5_path: Path,
    camera: str,
    start_frame: int,
    num_frames: int,
) -> tuple[np.ndarray, dict]:
    dataset_path = f"cam_data/{camera}"
    with h5py.File(h5_path, "r") as handle:
        if dataset_path not in handle:
            raise FileNotFoundError(f"HDF5 missing {dataset_path}: {h5_path}")
        ds = handle[dataset_path]
        end = start_frame + num_frames
        if start_frame < 0 or end > ds.shape[0]:
            raise ValueError(
                f"clip range [{start_frame}, {end}) exceeds {dataset_path} "
                f"frame count {ds.shape[0]} in {h5_path}"
            )
        raw = ds[start_frame:end]
    return camera_to_uint8(raw, source=f"{h5_path}:{dataset_path}")


def prepare_input_video(
    sample: SampleSpec,
    args: argparse.Namespace,
) -> dict:
    h5_path = hdf5_path(args.data_root, sample)
    frames, dtype_info = read_h2r_camera_clip(
        h5_path, args.camera, sample.start_frame, args.num_frames)
    if frames.shape[1:3] != (240, 426):
        raise ValueError(
            f"expected H2R source shape 240x426, got {frames.shape[1:3]} "
            f"for {h5_path}:{args.camera}"
        )
    input_frames, geom = aspect_fill_resize_crop(
        frames, args.api_size[0], args.api_size[1])
    sample_dir = args.output_root / sample.sample_id
    input_path = sample_dir / "input" / f"{sample.sample_id}_{args.camera}_seedance_ref_864x480.mp4"
    write_video_rgb(input_frames, input_path, args.fps)
    info = get_video_info(str(input_path))
    validate_seedance_input(info, expected_ratio=args.ratio)
    return {
        "sample_id": sample.sample_id,
        "task": sample.task,
        "episode": sample.episode,
        "start_frame": sample.start_frame,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "camera": args.camera,
        "hdf5_path": str(h5_path),
        "input_path": str(input_path),
        "dtype_info": dtype_info,
        "geometry": geom,
        "input_info": info,
        "source_frame_indices": list(range(sample.start_frame, sample.start_frame + args.num_frames)),
    }


def validate_seedance_input(info: dict, *, expected_ratio: str) -> None:
    w, h = int(info["width"]), int(info["height"])
    pixels = w * h
    if pixels < MIN_PIXELS:
        raise ValueError(f"Seedance input {w}x{h}={pixels} px < {MIN_PIXELS}")
    if expected_ratio == "16:9":
        target = 16 / 9
    else:
        raise ValueError(f"unsupported expected ratio for H2R Seedance: {expected_ratio}")
    ratio = w / h
    if abs(ratio - target) > 0.03:
        raise ValueError(
            f"Seedance input ratio {w}:{h} ({ratio:.4f}) does not match {expected_ratio}"
        )
    duration = float(info["duration"])
    if duration < 2.0 or duration > 15.0:
        raise ValueError(f"Seedance input duration {duration:.3f}s is outside 2-15s")


_print_lock = threading.Lock()


def log(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


def poll_task_quiet(api_key: str, task_id: str, *, interval: float, timeout: float, tag: str) -> dict:
    t0 = time.time()
    last_status = ""
    while True:
        resp = ark_request("GET", f"contents/generations/tasks/{task_id}", api_key)
        status = str(resp.get("status", "unknown"))
        elapsed = time.time() - t0
        if status != last_status:
            log(f"{tag} poll: {status} ({elapsed:.0f}s)")
            last_status = status
        if status == "succeeded":
            return resp
        if status in {"failed", "cancelled", "expired"}:
            raise RuntimeError(f"task {status}: {json.dumps(resp, ensure_ascii=False)}")
        if elapsed > timeout:
            raise TimeoutError(f"task still {status} after {timeout}s: {task_id}")
        time.sleep(interval)


def resample_frames(frames: np.ndarray, target_frames: int) -> tuple[np.ndarray, list[int]]:
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")
    if len(frames) == target_frames:
        return frames, list(range(target_frames))
    indices = np.linspace(0, len(frames) - 1, target_frames).round().astype(int)
    return frames[indices], indices.tolist()


def resize_seedance_output(
    src: Path,
    dst: Path,
    size_hxw: tuple[int, int],
    fps: int,
    num_frames: int,
) -> dict:
    frames = []
    container = av.open(str(src))
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    if not frames:
        raise ValueError(f"Seedance output has no frames: {src}")
    arr = np.stack(frames, axis=0)
    resized, geom = aspect_fill_resize_crop(arr, size_hxw[0], size_hxw[1])
    resampled, indices = resample_frames(resized, num_frames)
    write_video_rgb(resampled, dst, fps=fps)
    geom["raw_frame_count"] = int(len(arr))
    geom["final_frame_count"] = int(num_frames)
    geom["resample_indices"] = indices
    return geom


def process_one(
    index: int,
    total: int,
    record: dict,
    args: argparse.Namespace,
    api_key: str,
) -> dict:
    tag = f"[{index + 1}/{total}] {record['sample_id']}"
    input_path = Path(record["input_path"])
    sample_dir = args.output_root / record["sample_id"]
    raw_path = sample_dir / "seedance_raw" / f"{record['sample_id']}_human_hand_raw.mp4"
    final_path = sample_dir / "final" / f"{record['sample_id']}_human_hand_{args.final_size[0]}x{args.final_size[1]}_hxw.mp4"
    log(f"{tag} upload: {input_path}")
    t0 = time.time()
    try:
        public_url = upload_catbox(str(input_path))
        task_id = create_task(
            api_key,
            public_url,
            args.prompt,
            args.model,
            args.resolution,
            args.ratio,
            args.duration,
        )
        log(f"{tag} task: {task_id}")
        result = poll_task_quiet(
            api_key,
            task_id,
            interval=args.poll_interval,
            timeout=args.poll_timeout,
            tag=tag,
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        download(result["content"]["video_url"], str(raw_path))
        raw_info = get_video_info(str(raw_path))
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_geom = resize_seedance_output(
            raw_path, final_path, args.final_size, args.fps, args.num_frames)
        final_info = get_video_info(str(final_path))
        elapsed = time.time() - t0
        log(f"{tag} done: {elapsed:.0f}s -> {raw_path}")
        out = dict(record)
        out.update({
            "status": "ok",
            "task_id": task_id,
            "tokens": result.get("usage", {}).get("total_tokens", 0),
            "seed": result.get("seed"),
            "prompt": args.prompt,
            "raw_output_path": str(raw_path),
            "raw_output_info": raw_info,
            "final_output_path": str(final_path),
            "final_output_info": final_info,
            "final_geometry": final_geom,
            "elapsed_sec": round(elapsed, 1),
        })
        return out
    except Exception as exc:
        elapsed = time.time() - t0
        log(f"{tag} FAILED after {elapsed:.0f}s: {exc}")
        out = dict(record)
        out.update({
            "status": "failed",
            "error": str(exc),
            "prompt": args.prompt,
            "elapsed_sec": round(elapsed, 1),
        })
        return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and run 16:9 H2R Seedance robot-arm to human-hand smoke"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample", action="append", default=[],
                        help="task:episode:start_frame; repeatable")
    parser.add_argument("--camera", default="robot_camera")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--api-size", default="480x864",
                        help="Seedance reference input HxW; default 480x864")
    parser.add_argument("--final-size", default="256x488",
                        help="final local review output HxW; default follows user request")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--resolution", default="480p",
                        choices=["480p", "720p", "1080p", "2k"])
    parser.add_argument("--ratio", default="16:9", choices=["16:9"])
    parser.add_argument("--duration", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--env-file", type=Path, default=Path(MAIN_ROOT) / ".env")
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--poll-timeout", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="prepare inputs and plan only; do not call Seedance")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    args.api_size = parse_hxw(args.api_size)
    args.final_size = parse_hxw(args.final_size)
    args.data_root = args.data_root if args.data_root.is_absolute() else Path(MAIN_ROOT) / args.data_root
    args.output_root = args.output_root if args.output_root.is_absolute() else Path(MAIN_ROOT) / args.output_root
    args.model = MODEL_FAST if args.fast else MODEL_STANDARD
    args.env_file = args.env_file if args.env_file.is_absolute() else Path(MAIN_ROOT) / args.env_file

    samples = [parse_sample(value) for value in (args.sample or DEFAULT_SAMPLES)]
    if len(samples) != 3:
        raise ValueError(
            f"this smoke is intended for exactly 3 robot videos; got {len(samples)}"
        )

    print("H2R Seedance robot-arm -> human-hand smoke")
    print(f"  samples:     {[sample.sample_id for sample in samples]}")
    print(f"  camera:      {args.camera}")
    print(f"  frames/fps:  {args.num_frames} @ {args.fps}fps")
    print(f"  api size:    {args.api_size[0]}x{args.api_size[1]} HxW")
    print(f"  final size:  {args.final_size[0]}x{args.final_size[1]} HxW")
    print(f"  model:       {args.model}")
    print(f"  workers:     {args.workers}")
    print(f"  dry_run:     {args.dry_run}")

    prepared = [prepare_input_video(sample, args) for sample in samples]
    write_jsonl(args.output_root / "prepared_inputs.jsonl", prepared)

    if args.dry_run:
        plan = {
            "status": "dry_run",
            "model": args.model,
            "resolution": args.resolution,
            "ratio": args.ratio,
            "duration": args.duration,
            "workers": args.workers,
            "fps": args.fps,
            "num_frames": args.num_frames,
            "api_size_hxw": list(args.api_size),
            "final_size_hxw": list(args.final_size),
            "prompt": args.prompt,
            "prepared_count": len(prepared),
        }
        (args.output_root / "run_plan.json").write_text(
            json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"prepared inputs: {args.output_root / 'prepared_inputs.jsonl'}")
        print(f"run plan:        {args.output_root / 'run_plan.json'}")
        return

    env_values = load_env_file(args.env_file)
    api_key = args.api_key or os.environ.get("ARK_API_KEY") or env_values.get("ARK_API_KEY", "")
    if not api_key:
        raise ValueError(f"provide --api-key, set ARK_API_KEY, or add it to {args.env_file}")

    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(process_one, index, len(prepared), record, args, api_key)
            for index, record in enumerate(prepared)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["sample_id"])
    write_jsonl(args.output_root / "seedance_results.jsonl", results)
    summary = {
        "status": "complete",
        "ok": sum(1 for row in results if row["status"] == "ok"),
        "failed": sum(1 for row in results if row["status"] == "failed"),
        "elapsed_sec": round(time.time() - t0, 1),
        "model": args.model,
        "resolution": args.resolution,
        "ratio": args.ratio,
        "duration": args.duration,
        "workers": args.workers,
        "prompt": args.prompt,
    }
    (args.output_root / "seedance_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
