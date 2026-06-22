#!/usr/bin/env python3
"""Materialize canonical human2robot data from the legacy data/h2r/v1 tree.

The legacy HDF5 camera arrays are uint8 [T,H,W,3] but their first/last color
channels are reversed for normal RGB consumers. This script copies the HDF5
files into a new canonical root, rewrites camera datasets as RGB by swapping
B/R in the destination copy, and regenerates MP4 videos from those RGB arrays.
It never modifies the legacy source tree.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import av
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import MAIN_ROOT


DEFAULT_SOURCE_ROOT = Path(MAIN_ROOT) / "data" / "h2r" / "v1"
DEFAULT_OUTPUT_ROOT = Path(MAIN_ROOT) / "data" / "human2robot"
DEFAULT_CAMERAS = ("robot_camera", "human_camera")
MIGRATION_VERSION = "human2robot_bgr_to_rgb_v1"


@dataclass(frozen=True)
class EpisodeJob:
    source_hdf5: Path
    rel_hdf5: Path


@dataclass(frozen=True)
class EpisodeResult:
    rel_hdf5: str
    frame_count: int
    copied_hdf5: bool
    written_videos: int
    skipped_videos: int


def parse_csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError(f"Expected comma-separated value, got {value!r}")
    return items


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(MAIN_ROOT) / path
    return path.resolve()


def ensure_not_same_tree(source_root: Path, output_root: Path) -> None:
    if source_root == output_root:
        raise ValueError(f"Output root must differ from source root: {source_root}")
    if output_root.is_relative_to(source_root):
        raise ValueError(
            f"Output root must not be inside legacy source root: {output_root}"
        )


def discover_jobs(source_root: Path, tasks: tuple[str, ...] | None) -> list[EpisodeJob]:
    source_data = source_root / "data"
    if not source_data.is_dir():
        raise NotADirectoryError(f"Legacy source data root does not exist: {source_data}")
    search_roots = [source_data] if tasks is None else [source_data / task for task in tasks]
    for search_root in search_roots:
        if not search_root.is_dir():
            raise NotADirectoryError(f"Requested task data dir does not exist: {search_root}")

    jobs = [
        EpisodeJob(source_hdf5=path, rel_hdf5=path.relative_to(source_data))
        for search_root in search_roots
        for path in sorted(search_root.rglob("episode_*.hdf5"))
    ]
    jobs.sort(key=lambda item: item.rel_hdf5.as_posix())
    if not jobs:
        raise ValueError(f"No episode_*.hdf5 files found under {source_data}")
    return jobs


def validate_camera_dataset(
    path: Path,
    camera: str,
    dataset: h5py.Dataset,
    *,
    require_uint8: bool = True,
) -> int:
    if dataset.ndim != 4 or dataset.shape[-1] != 3:
        raise ValueError(
            f"{path}:cam_data/{camera} must be [T,H,W,3], got {dataset.shape}"
        )
    if require_uint8 and dataset.dtype != np.uint8:
        raise ValueError(
            f"{path}:cam_data/{camera} must be uint8 RGB, got {dataset.dtype}"
        )
    if not require_uint8 and not np.issubdtype(dataset.dtype, np.integer):
        raise ValueError(
            f"{path}:cam_data/{camera} must use an integer image dtype, got {dataset.dtype}"
        )
    frame_count, height, width, _ = dataset.shape
    if frame_count <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            f"{path}:cam_data/{camera} has invalid shape {dataset.shape}"
        )
    return int(frame_count)


def stamp_hdf5_metadata(path: Path, cameras: tuple[str, ...]) -> int:
    with h5py.File(path, "r+") as handle:
        handle.attrs["dataset_name"] = "human2robot"
        handle.attrs["canonical_root"] = "data/human2robot"
        handle.attrs["legacy_source_root"] = "data/h2r/v1"
        handle.attrs["camera_color_space"] = "RGB"
        handle.attrs["migration_version"] = MIGRATION_VERSION
        handle.attrs["legacy_camera_channel_transform"] = "BGR_TO_RGB"

        frame_count: int | None = None
        for camera in cameras:
            dataset_path = f"cam_data/{camera}"
            if dataset_path not in handle:
                raise FileNotFoundError(f"HDF5 missing {dataset_path}: {path}")
            ds = handle[dataset_path]
            current_count = validate_camera_dataset(path, camera, ds)
            if frame_count is None:
                frame_count = current_count
            elif frame_count != current_count:
                raise ValueError(
                    f"Camera frame-count mismatch in {path}: "
                    f"expected {frame_count}, got {camera}={current_count}"
                )
            ds.attrs["color_space"] = "RGB"
            ds.attrs["channel_order"] = "RGB"
            ds.attrs["legacy_channel_order"] = "BGR"
            ds.attrs["migration_transform"] = "BGR_TO_RGB"
    if frame_count is None:
        raise ValueError(f"No camera datasets stamped in {path}")
    return frame_count


def hdf5_complete(path: Path, cameras: tuple[str, ...]) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("migration_version") != MIGRATION_VERSION:
            return False
        for camera in cameras:
            dataset_path = f"cam_data/{camera}"
            if dataset_path not in handle:
                return False
            ds = handle[dataset_path]
            if ds.attrs.get("migration_transform") != "BGR_TO_RGB":
                return False
            validate_camera_dataset(path, camera, ds)
    return True


def convert_camera_datasets_to_uint8_rgb(
    source_path: Path,
    dest_path: Path,
    cameras: tuple[str, ...],
    chunk_frames: int,
) -> int:
    if chunk_frames <= 0:
        raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")
    with h5py.File(source_path, "r") as source, h5py.File(dest_path, "r+") as dest:
        frame_count: int | None = None
        for camera in cameras:
            dataset_path = f"cam_data/{camera}"
            if dataset_path not in source:
                raise FileNotFoundError(f"HDF5 missing {dataset_path}: {source_path}")
            src_ds = source[dataset_path]
            current_count = validate_camera_dataset(
                source_path, camera, src_ds, require_uint8=False,
            )
            if frame_count is None:
                frame_count = current_count
            elif frame_count != current_count:
                raise ValueError(
                    f"Camera frame-count mismatch in {source_path}: "
                    f"expected {frame_count}, got {camera}={current_count}"
                )
            src_min = int(src_ds[: min(5, current_count)].min())
            src_max = int(src_ds[: min(5, current_count)].max())
            if src_min < 0 or src_max > 255:
                raise ValueError(
                    f"{source_path}:{dataset_path} cannot be cast to uint8 "
                    f"without rescaling: dtype={src_ds.dtype} min={src_min} max={src_max}"
                )
            parent_name, leaf_name = dataset_path.split("/", 1)
            if dataset_path in dest:
                del dest[dataset_path]
            if parent_name not in dest:
                dest.create_group(parent_name)
            chunks = src_ds.chunks
            if chunks is not None:
                chunks = (
                    min(int(chunks[0]), int(src_ds.shape[0])),
                    int(chunks[1]),
                    int(chunks[2]),
                    int(chunks[3]),
                )
            dest_ds = dest[parent_name].create_dataset(
                leaf_name,
                shape=src_ds.shape,
                dtype=np.uint8,
                chunks=chunks,
                compression=src_ds.compression,
                compression_opts=src_ds.compression_opts,
                shuffle=src_ds.shuffle,
                fletcher32=src_ds.fletcher32,
            )
            for attr_key, attr_value in src_ds.attrs.items():
                dest_ds.attrs[attr_key] = attr_value
            for start in range(0, current_count, chunk_frames):
                end = min(start + chunk_frames, current_count)
                block = src_ds[start:end]
                dest_ds[start:end] = block[..., ::-1].astype(np.uint8, copy=False)
    if frame_count is None:
        raise ValueError(f"No camera datasets converted in {source_path}")
    return frame_count


def copy_hdf5(
    source: Path,
    dest: Path,
    cameras: tuple[str, ...],
    overwrite: bool,
    chunk_frames: int,
) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        if not hdf5_complete(dest, cameras):
            raise ValueError(
                f"Existing output HDF5 is not a completed {MIGRATION_VERSION} file: {dest}. "
                "Rerun with --overwrite after removing or replacing partial output."
            )
        return False
    tmp = dest.with_name(f".{dest.name}.{time.time_ns()}.tmp")
    try:
        shutil.copy2(source, tmp)
        convert_camera_datasets_to_uint8_rgb(source, tmp, cameras, chunk_frames)
        stamp_hdf5_metadata(tmp, cameras)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return True


def probe_video(path: Path) -> tuple[int, int, int]:
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        frames = 0
        width = int(stream.codec_context.width or stream.width)
        height = int(stream.codec_context.height or stream.height)
        for _frame in container.decode(stream):
            frames += 1
    finally:
        container.close()
    if frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid video probe for {path}: {frames} frames {width}x{height}")
    return frames, height, width


def video_complete(path: Path, expected_frames: int, expected_hw: tuple[int, int]) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    frames, height, width = probe_video(path)
    return frames == expected_frames and (height, width) == expected_hw


def write_camera_video(
    hdf5_path: Path,
    camera: str,
    out_path: Path,
    fps: int,
    crf: int,
    preset: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.{time.time_ns()}.tmp.mp4")
    try:
        with h5py.File(hdf5_path, "r") as handle:
            dataset_path = f"cam_data/{camera}"
            if dataset_path not in handle:
                raise FileNotFoundError(f"HDF5 missing {dataset_path}: {hdf5_path}")
            ds = handle[dataset_path]
            validate_camera_dataset(hdf5_path, camera, ds)
            frame_count, height, width, _ = ds.shape

            container = av.open(str(tmp), mode="w")
            stream = container.add_stream("libx264", rate=fps)
            stream.width = int(width)
            stream.height = int(height)
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(crf), "preset": preset}
            try:
                for frame_index in range(int(frame_count)):
                    frame_rgb = np.asarray(ds[frame_index], dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            finally:
                container.close()
        tmp.replace(out_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def submit_episode(
    executor: ProcessPoolExecutor,
    job: EpisodeJob,
    source_root: Path,
    output_root: Path,
    cameras: tuple[str, ...],
    args: argparse.Namespace,
) -> Future:
    return executor.submit(
        process_episode,
        job,
        source_root,
        output_root,
        cameras,
        args.fps,
        args.crf,
        args.preset,
        args.overwrite,
        args.chunk_frames,
    )


def process_episode(
    job: EpisodeJob,
    source_root: Path,
    output_root: Path,
    cameras: tuple[str, ...],
    fps: int,
    crf: int,
    preset: str,
    overwrite: bool,
    chunk_frames: int,
) -> EpisodeResult:
    dest_hdf5 = output_root / "data" / job.rel_hdf5
    copied = copy_hdf5(
        job.source_hdf5,
        dest_hdf5,
        cameras,
        overwrite=overwrite,
        chunk_frames=chunk_frames,
    )
    frame_count = stamp_hdf5_metadata(dest_hdf5, cameras)

    task_rel = job.rel_hdf5.parent
    episode_name = job.rel_hdf5.stem
    video_dir = output_root / "video" / task_rel / episode_name

    written = 0
    skipped = 0
    with h5py.File(dest_hdf5, "r") as handle:
        first_camera = cameras[0]
        height, width = handle[f"cam_data/{first_camera}"].shape[1:3]
    for camera in cameras:
        out_path = video_dir / f"{camera}.mp4"
        if not overwrite and video_complete(out_path, frame_count, (height, width)):
            skipped += 1
            continue
        write_camera_video(dest_hdf5, camera, out_path, fps=fps, crf=crf, preset=preset)
        written += 1

    return EpisodeResult(
        rel_hdf5=job.rel_hdf5.as_posix(),
        frame_count=frame_count,
        copied_hdf5=copied,
        written_videos=written,
        skipped_videos=skipped,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy legacy H2R HDF5 data into data/human2robot and regenerate RGB MP4 videos."
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=64,
        help="camera frames per HDF5 BGR->RGB rewrite chunk",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not (0 <= args.crf <= 51):
        raise ValueError("--crf must be in [0, 51]")

    source_root = resolve_path(args.source_root)
    output_root = resolve_path(args.output_root)
    ensure_not_same_tree(source_root, output_root)
    cameras = parse_csv(args.cameras)
    tasks = None if args.tasks == "all" else parse_csv(args.tasks)
    jobs = discover_jobs(source_root, tasks)

    print(f"source:  {source_root}")
    print(f"output:  {output_root}")
    print(f"episodes: {len(jobs)}")
    print(f"cameras:  {', '.join(cameras)}")
    print(f"workers:  {args.workers}")
    if args.dry_run:
        for job in jobs[:10]:
            print(f"  {job.rel_hdf5}")
        if len(jobs) > 10:
            print(f"  ... {len(jobs) - 10} more")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    copied_hdf5 = 0
    written_videos = 0
    skipped_videos = 0
    start = time.time()
    executor = ProcessPoolExecutor(max_workers=args.workers)
    pending_jobs = iter(jobs)
    futures: dict[Future, EpisodeJob] = {}
    try:
        for _ in range(min(args.workers, len(jobs))):
            job = next(pending_jobs)
            futures[submit_episode(executor, job, source_root, output_root, cameras, args)] = job

        while futures:
            for future in as_completed(futures):
                futures.pop(future)
                result = future.result()
                next_job = next(pending_jobs, None)
                if next_job is not None:
                    futures[
                        submit_episode(
                            executor,
                            next_job,
                            source_root,
                            output_root,
                            cameras,
                            args,
                        )
                    ] = next_job
                break
            completed += 1
            copied_hdf5 += int(result.copied_hdf5)
            written_videos += result.written_videos
            skipped_videos += result.skipped_videos
            if completed == 1 or completed % 25 == 0 or completed == len(jobs):
                elapsed = time.time() - start
                print(
                    f"[{completed}/{len(jobs)}] {result.rel_hdf5} "
                    f"frames={result.frame_count} copied_hdf5={copied_hdf5} "
                    f"written_videos={written_videos} skipped_videos={skipped_videos} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    except Exception:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True, cancel_futures=False)

    print(
        f"done episodes={completed} copied_hdf5={copied_hdf5} "
        f"written_videos={written_videos} skipped_videos={skipped_videos}"
    )


if __name__ == "__main__":
    main()
