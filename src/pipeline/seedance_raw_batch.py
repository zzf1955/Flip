"""CSV-driven Seedance raw-output batch generation.

This entry point is for the WBT 4s Seedance expansion lists. It preserves the
Seedance API output exactly as returned by the service and writes it directly
under training_data/seedance_raw/4s with the same task/episode/clip layout used
by training_data/seedance_direct/4s.

Prepared 4s inputs are temporary files only. No resized final videos and no
``*.raw.mp4`` side files are written.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config import MAIN_ROOT
from src.pipeline.seedance_gen import (
    DEFAULT_PROMPT,
    FFMPEG,
    MIN_PIXELS,
    MODEL_FAST,
    MODEL_STANDARD,
    create_task,
    download,
    get_video_info,
    poll_task,
    upload_catbox,
    validate_input,
)

sys.stdout.reconfigure(line_buffering=True)


ROOT = Path(MAIN_ROOT).resolve()
RAW_OUTPUT_ROOT = (ROOT / "training_data" / "seedance_raw" / "4s").resolve()

EXPECTED_WORKER_ERRORS = (
    RuntimeError,
    TimeoutError,
    subprocess.CalledProcessError,
    subprocess.TimeoutExpired,
    OSError,
    ValueError,
    KeyError,
    urllib.error.URLError,
)

_PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class Job:
    """One Seedance API job selected from a CSV row."""

    row_index: int
    sequence: int
    row: dict[str, str]
    output_path: Path


def log(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def row_label(row: dict[str, str]) -> str:
    task = row.get("task", "")
    source_id = row.get("source_id", "")
    if task and source_id.startswith(f"{task}/"):
        return source_id
    if task and source_id:
        return f"{task}/{source_id}"
    return task or source_id


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y"}


def resolve_workspace_path(value: str, *, field_name: str) -> Path:
    if not value:
        raise ValueError(f"missing required CSV field: {field_name}")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def require_raw_output_path(row: dict[str, str]) -> Path:
    output = resolve_workspace_path(
        row.get("planned_seedance_output", ""),
        field_name="planned_seedance_output",
    )
    if output.suffix.lower() != ".mp4":
        raise ValueError(f"planned output is not an mp4: {output}")
    if not output.is_relative_to(RAW_OUTPUT_ROOT):
        raise ValueError(
            "planned output must be under "
            f"{RAW_OUTPUT_ROOT}; got {output}"
        )
    return output


def load_generation_rows(csv_path: Path) -> list[tuple[int, dict[str, str]]]:
    rows: list[tuple[int, dict[str, str]]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader, start=2):
            if truthy(row.get("needs_seedance_api")):
                rows.append((row_index, row))
    return rows


def run_ffmpeg(cmd: list[str]) -> None:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr[-4000:]
        raise RuntimeError(
            "ffmpeg failed with return code "
            f"{proc.returncode}: {' '.join(cmd)}\n{stderr}"
        )


def float_field(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    if value == "":
        raise ValueError(f"missing required CSV field: {name}")
    return float(value)


def transcode_to_api_input(source: Path, output: Path) -> Path:
    """Transcode a 4s 4:3 source to Seedance-valid 800x600 temporary input."""
    run_ffmpeg([
        FFMPEG,
        "-y",
        "-i",
        str(source),
        "-vf",
        "fps=30,scale=800:600:flags=lanczos,setsar=1",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ])
    validate_input(get_video_info(str(output)))
    return output


def extract_window_to_api_input(row: dict[str, str], tmp_dir: Path) -> Path:
    """Extract the row's exact sequential 4s WBT window to a temp API input."""
    source = resolve_workspace_path(row.get("input_video", ""), field_name="input_video")
    if not source.is_file():
        raise FileNotFoundError(f"input video not found: {source}")

    start = float_field(row, "clip_start_sec")
    end = float_field(row, "clip_end_sec")
    duration = end - start
    if duration <= 0:
        raise ValueError(
            f"invalid clip window for row {row.get('source_id', '')}: "
            f"{start:.3f}s -> {end:.3f}s"
        )

    output = tmp_dir / "api_input.mp4"
    trim = (
        f"trim=start={start:.6f}:duration={duration:.6f},"
        "setpts=PTS-STARTPTS,fps=30,"
        "scale=800:600:flags=lanczos,setsar=1"
    )
    run_ffmpeg([
        FFMPEG,
        "-y",
        "-i",
        str(source),
        "-vf",
        trim,
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ])
    validate_input(get_video_info(str(output)))
    return output


def prepare_api_input(row: dict[str, str], tmp_dir: Path) -> Path:
    """Return a Seedance-valid input path, using only temporary files if needed."""
    if truthy(row.get("needs_clip_extract")):
        return extract_window_to_api_input(row, tmp_dir)

    source = resolve_workspace_path(row.get("input_video", ""), field_name="input_video")
    if not source.is_file():
        raise FileNotFoundError(f"input video not found: {source}")

    info = get_video_info(str(source))
    if info["width"] * info["height"] >= MIN_PIXELS:
        validate_input(info)
        return source

    return transcode_to_api_input(source, tmp_dir / "api_input.mp4")


def download_original(video_url: str, output_path: Path) -> dict[str, Any]:
    """Download Seedance raw output atomically, leaving only the final mp4."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".seedance_download_",
        dir=str(output_path.parent),
    ) as download_dir:
        tmp_output = Path(download_dir) / output_path.name
        download(video_url, str(tmp_output))
        output_info = get_video_info(str(tmp_output))
        tmp_output.replace(output_path)
    return output_info


def process_one(
    idx: int,
    total: int,
    job: Job,
    *,
    api_key: str,
    prompt: str,
    model: str,
    resolution: str,
    poll_interval: float,
    poll_timeout: float,
) -> dict[str, Any]:
    tag = f"[{idx + 1}/{total}]"
    row = job.row
    task = row.get("task", "")
    source_id = row.get("source_id", "")
    rel_output = job.output_path.relative_to(ROOT)
    log(f"{tag} start: {row_label(row)} -> {rel_output}")

    started_at = time.time()
    with tempfile.TemporaryDirectory(prefix="seedance_raw_api_") as tmp_name:
        api_input = prepare_api_input(row, Path(tmp_name))
        api_input_info = get_video_info(str(api_input))

        public_url = upload_catbox(str(api_input))
        task_id = create_task(
            api_key,
            public_url,
            prompt,
            model,
            resolution,
            ratio="4:3",
            duration=4,
        )
        result = poll_task(
            api_key,
            task_id,
            interval=poll_interval,
            timeout=poll_timeout,
        )
        video_url = result["content"]["video_url"]
        output_info = download_original(video_url, job.output_path)

    elapsed = time.time() - started_at
    usage = result.get("usage", {})
    total_tokens = usage.get("total_tokens", 0)
    log(f"{tag} done: {elapsed:.0f}s, tokens={total_tokens}, -> {rel_output}")
    return {
        "status": "ok",
        "row_index": job.row_index,
        "task": task,
        "source_id": source_id,
        "input_video": row.get("input_video", ""),
        "output": str(rel_output),
        "task_id": task_id,
        "tokens": total_tokens,
        "seed": result.get("seed"),
        "elapsed_sec": round(elapsed, 1),
        "api_input_info": api_input_info,
        "output_info": output_info,
    }


def build_jobs(rows: list[tuple[int, dict[str, str]]], args: argparse.Namespace) -> tuple[list[Job], list[dict[str, Any]]]:
    jobs: list[Job] = []
    skipped: list[dict[str, Any]] = []
    for row_index, row in rows:
        output_path = require_raw_output_path(row)
        if output_path.exists():
            if args.resume:
                skipped.append({
                    "status": "skipped",
                    "row_index": row_index,
                    "task": row.get("task", ""),
                    "source_id": row.get("source_id", ""),
                    "output": str(output_path.relative_to(ROOT)),
                })
                continue
            if not args.overwrite:
                raise FileExistsError(
                    f"output already exists; use --resume or --overwrite: "
                    f"{output_path}"
                )

        jobs.append(Job(
            row_index=row_index,
            sequence=len(jobs) + 1,
            row=row,
            output_path=output_path,
        ))

    if args.limit is not None:
        jobs = jobs[:args.limit]
    return jobs, skipped


def print_work_list(jobs: list[Job], skipped: list[dict[str, Any]], csv_path: Path, args: argparse.Namespace) -> None:
    print("Seedance Raw CSV Batch")
    print(f"  list:        {csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path}")
    print(f"  output root: {RAW_OUTPUT_ROOT.relative_to(ROOT)}")
    print(f"  model:       {MODEL_FAST if args.fast else MODEL_STANDARD}")
    print(f"  resolution:  {args.resolution}")
    print(f"  workers:     {args.workers}")
    print(f"  skip:        {len(skipped)}")
    print(f"  generate:    {len(jobs)}")
    print()
    if jobs:
        print("--- GENERATE ---")
        for idx, job in enumerate(jobs, start=1):
            row = job.row
            output = job.output_path.relative_to(ROOT)
            print(f"  {idx:3d}. {row_label(row)} -> {output}")


def write_log_if_requested(path_value: str | None, payload: dict[str, Any]) -> None:
    if not path_value:
        return
    path = resolve_workspace_path(path_value, field_name="log")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"  log: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Seedance for CSV rows and keep raw API outputs only."
    )
    parser.add_argument(
        "--list",
        required=True,
        help="CSV list with needs_seedance_api/planned_seedance_output fields",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use fast model (default: standard)",
    )
    parser.add_argument(
        "--resolution",
        default="480p",
        choices=["480p", "720p", "1080p", "2k"],
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Ark API key (or set ARK_API_KEY env)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="number of concurrent API requests",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N non-skipped rows",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip rows whose raw output already exists",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing raw output files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print selected rows without extracting clips or calling API",
    )
    parser.add_argument("--poll-interval", type=float, default=10)
    parser.add_argument("--poll-timeout", type=float, default=600)
    parser.add_argument(
        "--log",
        default=None,
        help="optional JSON summary path; omitted by default to avoid extra files",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")

    csv_path = resolve_workspace_path(args.list, field_name="list")
    if not csv_path.is_file():
        parser.error(f"list not found: {csv_path}")

    rows = load_generation_rows(csv_path)
    jobs, skipped = build_jobs(rows, args)
    print_work_list(jobs, skipped, csv_path, args)

    if args.dry_run:
        print(f"\n[dry-run] would generate {len(jobs)} raw Seedance videos")
        return 0
    if not jobs:
        print("nothing to generate")
        return 0

    api_key = args.api_key or os.environ.get("ARK_API_KEY")
    if not api_key:
        parser.error("provide --api-key or set ARK_API_KEY env variable")

    model = MODEL_FAST if args.fast else MODEL_STANDARD
    total = len(jobs)
    started_at = time.time()
    results: list[dict[str, Any]] = list(skipped)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_one,
                idx,
                total,
                job,
                api_key=api_key,
                prompt=args.prompt,
                model=model,
                resolution=args.resolution,
                poll_interval=args.poll_interval,
                poll_timeout=args.poll_timeout,
            ): job
            for idx, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except EXPECTED_WORKER_ERRORS as exc:
                rel_output = job.output_path.relative_to(ROOT)
                log(f"[{job.sequence}/{total}] FAILED: {exc}")
                results.append({
                    "status": "failed",
                    "row_index": job.row_index,
                    "task": job.row.get("task", ""),
                    "source_id": job.row.get("source_id", ""),
                    "output": str(rel_output),
                    "error": str(exc),
                })

    elapsed = time.time() - started_at
    ok = sum(1 for result in results if result["status"] == "ok")
    failed = sum(1 for result in results if result["status"] == "failed")
    skipped_count = sum(1 for result in results if result["status"] == "skipped")
    total_tokens = sum(int(result.get("tokens", 0)) for result in results)

    payload = {
        "list": str(csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path),
        "output_root": str(RAW_OUTPUT_ROOT.relative_to(ROOT)),
        "model": model,
        "resolution": args.resolution,
        "total": len(results),
        "ok": ok,
        "failed": failed,
        "skipped": skipped_count,
        "tokens": total_tokens,
        "elapsed_sec": round(elapsed, 1),
        "results": results,
    }

    print()
    print(
        f"Summary: {ok} ok, {failed} failed, {skipped_count} skipped "
        f"/ {len(results)} total"
    )
    print(f"  tokens: {total_tokens}, time: {elapsed:.0f}s")
    write_log_if_requested(args.log, payload)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
