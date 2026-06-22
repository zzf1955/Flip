#!/usr/bin/env python3
"""Download Unitree WBT datasets into the local LeRobot layout.

The three MainCamOnly Inspire repositories on Hugging Face contain an extra
``G1_WB_Dex5_*`` directory. Local FLIP data keeps all tasks under
``data/unitree_G1_WBT/<repo_name>/data|meta|videos``, so this downloader strips
that remote prefix while preserving the rest of each LeRobot repository.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_ROOT = Path(os.environ.get("FLIP_MAIN_ROOT", "/disk_n/zzf/flip"))
DEFAULT_DATA_ROOT = DEFAULT_MAIN_ROOT / "data" / "unitree_G1_WBT"
DEFAULT_LOG_ROOT = DEFAULT_MAIN_ROOT / "training_data" / "log" / "unitree_wbt_download"


def _default_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", "/disk_n/zzf/.cache/huggingface")) / "hub"


DEFAULT_CACHE_DIR = _default_cache_dir()


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    group: str
    strip_prefix: str = ""

    @property
    def name(self) -> str:
        return self.repo_id.split("/", 1)[1]


@dataclass(frozen=True)
class FileJob:
    spec: DatasetSpec
    remote_path: str
    target_path: Path
    size: int | None


@dataclass
class FileResult:
    status: str
    repo: str
    remote_path: str
    target_path: str
    size: int | None
    method: str = ""
    seconds: float = 0.0


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "unitreerobotics/G1_WBT_Inspire_Collect_Clothes_MainCamOnly",
        "inspire_dex5",
        "G1_WB_Dex5_Collect_Clothes",
    ),
    DatasetSpec(
        "unitreerobotics/G1_WBT_Inspire_Pickup_Pillow_MainCamOnly",
        "inspire_dex5",
        "G1_WB_Dex5_Pickup_Pillow",
    ),
    DatasetSpec(
        "unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly",
        "inspire_dex5",
        "G1_WB_Dex5_Put_Clothes_into_Washing_Machine",
    ),
    DatasetSpec("unitreerobotics/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher", "brainco"),
    DatasetSpec("unitreerobotics/G1_WBT_Brainco_Pickup_Pillow", "brainco"),
    DatasetSpec("unitreerobotics/G1_WBT_Brainco_Make_The_Bed", "brainco"),
    DatasetSpec("unitreerobotics/G1_WBT_Brainco_Pick_Up_Medicine", "brainco"),
    DatasetSpec("unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine", "inspire_flat"),
    DatasetSpec("unitreerobotics/G1_WBT_Inspire_Put_Clothes_Into_Basket", "inspire_flat"),
    DatasetSpec("unitreerobotics/G1_WBT_Inspire_Put_Drinks_Into_Fridge", "inspire_flat"),
    DatasetSpec("unitreerobotics/G1_WBT_Inspire_Put_Vegetables_Into_Basket", "inspire_flat"),
    DatasetSpec("unitreerobotics/G1_WBT_Inspire_Pick_Up_Drinks", "inspire_flat"),
    DatasetSpec("unitreerobotics/G1_WBT_Inspire_Clean_The_Living_Room", "inspire_flat"),
)

GROUPS = ("brainco", "inspire_dex5", "inspire_flat")


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def _format_bytes(size: int | None) -> str:
    if size is None:
        return "unknown"
    return f"{size / 1e9:.2f} GB ({size / (1024 ** 3):.2f} GiB)"


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_path(value: str | Path, *, base: Path = DEFAULT_MAIN_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def _target_relative_path(spec: DatasetSpec, remote_path: str) -> Path:
    prefix = spec.strip_prefix.strip("/")
    if prefix and remote_path.startswith(prefix + "/"):
        return Path(remote_path[len(prefix) + 1 :])
    return Path(remote_path)


def _target_complete(path: Path, expected_size: int | None, force_download: bool) -> bool:
    if force_download or not path.exists() or not path.is_file():
        return False
    if expected_size is None:
        return True
    return path.stat().st_size == expected_size


def _selected_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    requested_groups = [item.lower().replace("-", "_") for item in _csv(args.groups)]
    if not requested_groups:
        raise ValueError("--groups must not be empty")
    if "all" in requested_groups:
        requested_groups = list(GROUPS)
    invalid_groups = sorted(set(requested_groups) - set(GROUPS))
    if invalid_groups:
        raise ValueError(f"unknown --groups entries: {', '.join(invalid_groups)}")

    selected = [spec for spec in DATASETS if spec.group in requested_groups]
    requested_repos = set()
    for item in args.repo:
        for token in _csv(item):
            requested_repos.add(token if "/" in token else f"unitreerobotics/{token}")
    if requested_repos:
        selected = [spec for spec in selected if spec.repo_id in requested_repos]
        missing = sorted(requested_repos - {spec.repo_id for spec in DATASETS})
        if missing:
            raise ValueError(f"unknown --repo entries: {', '.join(missing)}")
    if not selected:
        raise ValueError("selection is empty")
    return selected


def _build_jobs(
    *,
    api: HfApi,
    specs: list[DatasetSpec],
    data_root: Path,
    revision: str | None,
    force_download: bool,
) -> tuple[list[FileJob], list[FileResult], dict[str, dict[str, int]]]:
    jobs: list[FileJob] = []
    skipped: list[FileResult] = []
    repo_stats: dict[str, dict[str, int]] = {}

    for spec in specs:
        info = api.repo_info(
            spec.repo_id,
            repo_type="dataset",
            revision=revision,
            files_metadata=True,
        )
        total_bytes = 0
        missing_bytes = 0
        total_files = 0
        missing_files = 0
        for sibling in info.siblings:
            remote_path = sibling.rfilename
            if not remote_path or remote_path.endswith("/"):
                continue
            total_files += 1
            size = sibling.size
            if size is not None:
                total_bytes += size
            target_path = data_root / spec.name / _target_relative_path(spec, remote_path)
            if _target_complete(target_path, size, force_download):
                skipped.append(
                    FileResult(
                        status="skipped",
                        repo=spec.repo_id,
                        remote_path=remote_path,
                        target_path=str(target_path),
                        size=size,
                    )
                )
                continue
            jobs.append(
                FileJob(
                    spec=spec,
                    remote_path=remote_path,
                    target_path=target_path,
                    size=size,
                )
            )
            missing_files += 1
            if size is not None:
                missing_bytes += size
        repo_stats[spec.repo_id] = {
            "total_files": total_files,
            "total_bytes": total_bytes,
            "missing_files": missing_files,
            "missing_bytes": missing_bytes,
        }
    return jobs, skipped, repo_stats


def _install_from_cache(cache_path: Path, target_path: Path, expected_size: int | None) -> str:
    cache_path = cache_path.resolve(strict=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(
        f"{target_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    if tmp_path.exists():
        tmp_path.unlink()

    method = "hardlink"
    try:
        os.link(cache_path, tmp_path)
    except OSError:
        method = "copy"
        shutil.copy2(cache_path, tmp_path)

    if expected_size is not None:
        actual_size = tmp_path.stat().st_size
        if actual_size != expected_size:
            tmp_path.unlink()
            raise RuntimeError(
                f"downloaded size mismatch for {target_path}: "
                f"expected {expected_size}, got {actual_size}"
            )
    os.replace(tmp_path, target_path)
    return method


def _download_one(
    job: FileJob,
    *,
    cache_dir: Path,
    revision: str | None,
    endpoint: str | None,
    token: str | None,
    etag_timeout: float,
    force_download: bool,
    local_files_only: bool,
    max_retries: int,
    retry_sleep: float,
) -> FileResult:
    start = time.monotonic()
    if _target_complete(job.target_path, job.size, force_download):
        return FileResult(
            status="skipped",
            repo=job.spec.repo_id,
            remote_path=job.remote_path,
            target_path=str(job.target_path),
            size=job.size,
            seconds=time.monotonic() - start,
        )

    last_error: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            cache_path = Path(
                hf_hub_download(
                    repo_id=job.spec.repo_id,
                    filename=job.remote_path,
                    repo_type="dataset",
                    revision=revision,
                    cache_dir=cache_dir,
                    endpoint=endpoint,
                    token=token,
                    etag_timeout=etag_timeout,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            )
            method = _install_from_cache(cache_path, job.target_path, job.size)
            return FileResult(
                status="downloaded",
                repo=job.spec.repo_id,
                remote_path=job.remote_path,
                target_path=str(job.target_path),
                size=job.size,
                method=method,
                seconds=time.monotonic() - start,
            )
        except Exception as exc:  # final failure is re-raised after bounded retries
            last_error = exc
            if attempt == max_retries:
                raise
            print(
                f"[retry {attempt}/{max_retries}] {job.spec.repo_id} "
                f"{job.remote_path}: {exc}",
                flush=True,
            )
            time.sleep(retry_sleep)
    raise RuntimeError(f"unreachable retry state: {last_error}")


class FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pid": os.getpid(), "started_at": datetime.now().isoformat(timespec="seconds")}
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                existing = self._read_existing()
                pid = existing.get("pid")
                if isinstance(pid, int) and self._pid_alive(pid):
                    raise RuntimeError(f"download lock is held by pid {pid}: {self.path}")
                self.path.unlink()
                continue
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, sort_keys=True)
                fh.write("\n")
            self.acquired = True
            return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.acquired and self.path.exists():
            self.path.unlink()

    def _read_existing(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _write_jsonl(path: Path, rows: list[FileResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row.__dict__, sort_keys=True) + "\n")


def _print_plan(
    *,
    specs: list[DatasetSpec],
    data_root: Path,
    cache_dir: Path,
    repo_stats: dict[str, dict[str, int]],
    jobs: list[FileJob],
    skipped: list[FileResult],
) -> None:
    total_bytes = sum(item["total_bytes"] for item in repo_stats.values())
    missing_bytes = sum(item["missing_bytes"] for item in repo_stats.values())
    print(f"Data root: {data_root}", flush=True)
    print(f"HF cache:  {cache_dir}", flush=True)
    print(f"Repos:     {len(specs)}", flush=True)
    print(
        f"Remote total: {_format_bytes(total_bytes)} across "
        f"{sum(item['total_files'] for item in repo_stats.values())} files",
        flush=True,
    )
    print(
        f"To fetch:     {_format_bytes(missing_bytes)} across {len(jobs)} files "
        f"({len(skipped)} already complete)",
        flush=True,
    )
    for spec in specs:
        stats = repo_stats[spec.repo_id]
        print(
            f"  - {spec.group:13s} {spec.name}: "
            f"{_format_bytes(stats['missing_bytes'])} missing / "
            f"{_format_bytes(stats['total_bytes'])} total, "
            f"{stats['missing_files']}/{stats['total_files']} files",
            flush=True,
        )


def _background_command(argv: list[str], log_dir: Path) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve())]
    command.extend(arg for arg in argv if arg != "--background")
    if "--log-dir" not in argv:
        command.extend(["--log-dir", str(log_dir)])
    return command


def _maybe_start_background(args: argparse.Namespace, argv: list[str]) -> bool:
    if not args.background:
        return False
    log_dir = _resolve_path(args.log_dir, base=DEFAULT_MAIN_ROOT) if args.log_dir else DEFAULT_LOG_ROOT / _now_tag()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "download.log"
    command = _background_command(argv, log_dir)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("ab", buffering=0) as log_fh:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            env=env,
            start_new_session=True,
        )
    _write_json(
        log_dir / "background.json",
        {
            "pid": process.pid,
            "command": command,
            "log_path": str(log_path),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(f"Started background Unitree WBT download: pid={process.pid}")
    print(f"Log: {log_path}")
    return True


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Brainco plus Inspire Dex5/flat Unitree WBT datasets into "
            "data/unitree_G1_WBT. Dex1 is intentionally excluded."
        )
    )
    parser.add_argument(
        "--groups",
        default="all",
        help="comma-separated subset: all, brainco, inspire_dex5, inspire_flat",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="optional repo id or repo name filter; can be repeated or comma-separated",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", ""))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--workers", type=int, default=8,
                        help="global concurrent file downloads")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=20.0)
    parser.add_argument("--etag-timeout", type=float, default=30.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--background", action="store_true",
                        help="spawn a detached child and write logs under --log-dir")
    parser.add_argument("--no-lock", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be positive")

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    data_root = _resolve_path(args.data_root, base=DEFAULT_MAIN_ROOT)
    cache_dir = _resolve_path(args.cache_dir, base=DEFAULT_MAIN_ROOT)
    log_dir = _resolve_path(args.log_dir, base=DEFAULT_MAIN_ROOT) if args.log_dir else DEFAULT_LOG_ROOT / _now_tag()
    endpoint = args.endpoint or None
    token = args.token or None

    specs = _selected_specs(args)
    api = HfApi(endpoint=endpoint, token=token)
    jobs, skipped, repo_stats = _build_jobs(
        api=api,
        specs=specs,
        data_root=data_root,
        revision=args.revision,
        force_download=args.force_download,
    )
    _print_plan(
        specs=specs,
        data_root=data_root,
        cache_dir=cache_dir,
        repo_stats=repo_stats,
        jobs=jobs,
        skipped=skipped,
    )

    _write_json(
        log_dir / "plan.json",
        {
            "data_root": str(data_root),
            "cache_dir": str(cache_dir),
            "endpoint": endpoint,
            "groups": args.groups,
            "repos": [spec.repo_id for spec in specs],
            "repo_stats": repo_stats,
            "dry_run": args.dry_run,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    if args.dry_run:
        print(f"Dry run complete. Plan written to {log_dir / 'plan.json'}", flush=True)
        return

    data_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if not jobs:
        _write_jsonl(log_dir / "files.jsonl", skipped)
        print("All selected files are already complete.", flush=True)
        return

    lock_context = FileLock(data_root / ".unitree_wbt_download.lock")
    if args.no_lock:
        lock_context = None

    results: list[FileResult] = list(skipped)
    failures: list[dict[str, str]] = []
    start = time.monotonic()

    def execute() -> None:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {
                executor.submit(
                    _download_one,
                    job,
                    cache_dir=cache_dir,
                    revision=args.revision,
                    endpoint=endpoint,
                    token=token,
                    etag_timeout=args.etag_timeout,
                    force_download=args.force_download,
                    local_files_only=args.local_files_only,
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                ): job
                for job in jobs
            }
            completed = 0
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    print(
                        f"[{completed}/{len(jobs)}] {result.status} "
                        f"{job.spec.name}/{job.remote_path} -> "
                        f"{Path(result.target_path).relative_to(data_root)} "
                        f"{_format_bytes(result.size)} {result.method}",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "repo": job.spec.repo_id,
                            "remote_path": job.remote_path,
                            "target_path": str(job.target_path),
                            "error": repr(exc),
                        }
                    )
                    print(
                        f"[{completed}/{len(jobs)}] FAILED "
                        f"{job.spec.repo_id}/{job.remote_path}: {exc}",
                        flush=True,
                    )

    if lock_context is None:
        execute()
    else:
        with lock_context:
            execute()

    elapsed = time.monotonic() - start
    _write_jsonl(log_dir / "files.jsonl", results)
    _write_json(
        log_dir / "summary.json",
        {
            "downloaded": sum(1 for item in results if item.status == "downloaded"),
            "skipped": sum(1 for item in results if item.status == "skipped"),
            "failed": len(failures),
            "failures": failures,
            "seconds": elapsed,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(f"Summary written to {log_dir / 'summary.json'}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; see {log_dir / 'summary.json'}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if _maybe_start_background(args, argv):
        return
    run(args)


if __name__ == "__main__":
    main()
