"""H2R HDF5 Diffusion Policy behavior cloning.

The downloaded HumanAndRobot / H2R data is not in LeRobot parquet layout.  This
entry reads the HDF5 layout directly:

- ``data/<task>/episode_<id>.hdf5`` contains ``cam_data/robot_camera``,
  ``qpos``, ``qvel``, ``end_position``, ``gripper_state`` and ``action``.
- ``video/<task>/episode_<id>/robot_camera.mp4`` is checked for alignment, but
  training reads frames from HDF5 to avoid a separate conversion step.

The policy is action-only diffusion BC: clean video/state history conditions a
denoiser over noisy future action chunks.  It does not predict future video.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATA_ROOT = Path("/disk_n/zzf/flip/data/h2r/v1")
DEFAULT_OUTPUT_DIR = Path("tmp/h2r_diffusion_policy_t068")
DEFAULT_TASKS = "grab_cup_v1,grab_cube2_v1,push_box_random_v1"
DEFAULT_STATE_KEYS = "qpos,qvel,end_position,gripper_state"
DEFAULT_G1_TASKS = (
    "Inspire_Collect_Clothes_MainCamOnly,"
    "Inspire_Pickup_Pillow_MainCamOnly,"
    "Inspire_Put_Clothes_into_Washing_Machine"
)
DEFAULT_G1_PAIR_ROOT = Path("training_data/pair/identity_r2r/2s61f30_slide")
DEFAULT_G1_RAW_ROOT = Path("data/unitree_G1_WBT")
DEFAULT_G1_SEGMENT_ROOT = Path("training_data/segment")
DEFAULT_G1_STATE_KEYS = "observation.state.robot_q_current,observation.state.hand_state"
DEFAULT_G1_ACTION_KEYS = "action.robot_q_desired,action.hand_cmd"


@dataclass(frozen=True)
class EpisodeInfo:
    task: str
    episode: int
    hdf5_path: str
    video_path: str
    length: int


@dataclass(frozen=True)
class SampleInfo:
    task: str
    episode: int
    hdf5_path: str
    obs_start: int
    action_start: int


@dataclass(frozen=True)
class G1ClipInfo:
    task: str
    episode: int
    episode_name: str
    seg: str
    source_id: str
    source_segment_id: str
    video_path: str
    joints_path: str
    action_root: str
    source_frame_indices: tuple[int, ...]
    clip_idx: int


@dataclass(frozen=True)
class G1SampleInfo:
    task: str
    episode: int
    source_id: str
    video_path: str
    joints_path: str
    action_root: str
    source_frame_indices: tuple[int, ...]
    obs_start: int
    action_start: int


@dataclass(frozen=True)
class NormStats:
    action_mean: list[float]
    action_std: list[float]
    state_mean: list[float]
    state_std: list[float]


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"expected comma-separated values, got {value!r}")
    return items


def parse_resize(value: str) -> tuple[int, int]:
    parts = value.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise ValueError(f"--resize must be formatted as HxW, got {value!r}")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"--resize dimensions must be positive, got {value!r}")
    return height, width


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty csv: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad jsonl at {path}:{line_no}: {exc}") from exc
    return rows


def resolve_record_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def episode_id(path: Path) -> int:
    if not path.stem.startswith("episode_"):
        raise ValueError(f"unexpected H2R episode filename: {path}")
    return int(path.stem.removeprefix("episode_"))


def parse_g1_episode(value: object) -> int:
    text = str(value)
    if text.startswith("ep"):
        return int(text.removeprefix("ep"))
    return int(text)


def g1_action_root(raw_root: Path, task: str) -> Path:
    prefixed = raw_root / (task if task.startswith("G1_WBT_") else f"G1_WBT_{task}")
    if prefixed.is_dir():
        return prefixed
    direct = raw_root / task
    if direct.is_dir():
        return direct
    raise FileNotFoundError(f"G1 action data root not found for task={task}: {prefixed}")


def concat_vector(row: object, keys: list[str], *, context: str) -> np.ndarray:
    parts: list[np.ndarray] = []
    for key in keys:
        value = np.asarray(row[key], dtype=np.float32).reshape(-1)
        if value.size == 0:
            raise ValueError(f"empty vector for {key} at {context}")
        parts.append(value)
    return np.concatenate(parts, axis=0).astype(np.float32)


def read_state_matrix(f: h5py.File, state_keys: list[str]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for key in state_keys:
        if key not in f:
            raise ValueError(f"H2R HDF5 missing state key {key!r}: {f.filename}")
        value = np.asarray(f[key], dtype=np.float32)
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2:
            raise ValueError(f"H2R state key {key!r} must be 1D or 2D, got {value.shape}")
        parts.append(value)
    return np.concatenate(parts, axis=1)


def discover_episodes(args: argparse.Namespace) -> list[EpisodeInfo]:
    data_dir = args.data_root / "data"
    video_dir = args.data_root / "video"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"H2R data dir not found: {data_dir}")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"H2R video dir not found: {video_dir}")

    tasks = sorted(path.name for path in data_dir.iterdir() if path.is_dir()) if args.tasks == "all" else parse_csv(args.tasks)
    state_keys = parse_csv(args.state_keys)
    episodes: list[EpisodeInfo] = []
    for task in tasks:
        task_data = data_dir / task
        task_video = video_dir / task
        if not task_data.is_dir():
            raise FileNotFoundError(f"H2R task data dir not found: {task_data}")
        if not task_video.is_dir():
            raise FileNotFoundError(f"H2R task video dir not found: {task_video}")
        hdf5_paths = sorted(task_data.glob("episode_*.hdf5"), key=episode_id)
        if args.max_episodes_per_task > 0:
            hdf5_paths = hdf5_paths[: args.max_episodes_per_task]
        for hdf5_path in hdf5_paths:
            ep = episode_id(hdf5_path)
            video_path = task_video / f"episode_{ep}" / f"{args.camera_key}.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(f"H2R camera video not found: {video_path}")
            with h5py.File(hdf5_path, "r") as f:
                required = [f"cam_data/{args.camera_key}", args.action_key, *state_keys]
                missing = [key for key in required if key not in f]
                if missing:
                    raise ValueError(f"H2R episode missing keys {missing}: {hdf5_path}")
                length = int(f[args.action_key].shape[0])
                for key in required:
                    if int(f[key].shape[0]) != length:
                        raise ValueError(
                            f"H2R key length mismatch for {key}: {f[key].shape[0]} vs "
                            f"{length} in {hdf5_path}"
                        )
            episodes.append(EpisodeInfo(task, ep, str(hdf5_path), str(video_path), length))
    if not episodes:
        raise ValueError(f"no H2R episodes found under {args.data_root}")
    return episodes


def split_episodes(episodes: list[EpisodeInfo], train_ratio: float, seed: int) -> tuple[list[EpisodeInfo], list[EpisodeInfo]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
    rng = random.Random(seed)
    by_task: dict[str, list[EpisodeInfo]] = {}
    for episode in episodes:
        by_task.setdefault(episode.task, []).append(episode)
    train: list[EpisodeInfo] = []
    val: list[EpisodeInfo] = []
    for task, group in sorted(by_task.items()):
        ordered = list(group)
        rng.shuffle(ordered)
        if len(ordered) < 2:
            raise ValueError(f"episode split needs at least two episodes for task={task}")
        n_train = max(1, min(len(ordered) - 1, int(round(len(ordered) * train_ratio))))
        train.extend(ordered[:n_train])
        val.extend(ordered[n_train:])
    return train, val


def build_samples(
    episodes: list[EpisodeInfo],
    obs_horizon: int,
    pred_horizon: int,
    frame_stride: int,
    max_samples: int,
    seed: int,
) -> list[SampleInfo]:
    if obs_horizon <= 0 or pred_horizon <= 0:
        raise ValueError("obs_horizon and pred_horizon must be positive")
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")
    samples: list[SampleInfo] = []
    for episode in episodes:
        if episode.length < obs_horizon + pred_horizon:
            continue
        for action_start in range(obs_horizon - 1, episode.length - pred_horizon + 1, frame_stride):
            samples.append(
                SampleInfo(
                    task=episode.task,
                    episode=episode.episode,
                    hdf5_path=episode.hdf5_path,
                    obs_start=action_start - obs_horizon + 1,
                    action_start=action_start,
                )
            )
    random.Random(seed).shuffle(samples)
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError("no H2R samples built; check horizons, stride and max sample limits")
    return samples


def compute_norm(episodes: list[EpisodeInfo], args: argparse.Namespace) -> NormStats:
    state_keys = parse_csv(args.state_keys)
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    remaining = args.norm_max_frames
    ordered = list(episodes)
    random.Random(args.seed).shuffle(ordered)
    for episode in ordered:
        with h5py.File(episode.hdf5_path, "r") as f:
            action = np.asarray(f[args.action_key], dtype=np.float32)
            state = read_state_matrix(f, state_keys)
        if remaining > 0 and action.shape[0] > remaining:
            idx = np.linspace(0, action.shape[0] - 1, remaining, dtype=np.int64)
            action = action[idx]
            state = state[idx]
        actions.append(action)
        states.append(state)
        if remaining > 0:
            remaining -= int(action.shape[0])
            if remaining <= 0:
                break
    action_arr = np.concatenate(actions, axis=0)
    state_arr = np.concatenate(states, axis=0)
    return NormStats(
        action_mean=[float(v) for v in action_arr.mean(axis=0).tolist()],
        action_std=[float(v) for v in np.maximum(action_arr.std(axis=0), 1e-6).tolist()],
        state_mean=[float(v) for v in state_arr.mean(axis=0).tolist()],
        state_std=[float(v) for v in np.maximum(state_arr.std(axis=0), 1e-6).tolist()],
    )


class H2RDataset(Dataset):
    def __init__(self, samples: list[SampleInfo], stats: NormStats, args: argparse.Namespace) -> None:
        self.samples = list(samples)
        self.stats = stats
        self.camera_key = args.camera_key
        self.action_key = args.action_key
        self.state_keys = parse_csv(args.state_keys)
        self.obs_horizon = int(args.obs_horizon)
        self.pred_horizon = int(args.pred_horizon)
        self.resize = parse_resize(args.resize)
        self.action_mean = np.asarray(stats.action_mean, dtype=np.float32)
        self.action_std = np.asarray(stats.action_std, dtype=np.float32)
        self.state_mean = np.asarray(stats.state_mean, dtype=np.float32)
        self.state_std = np.asarray(stats.state_std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        with h5py.File(sample.hdf5_path, "r") as f:
            obs_slice = slice(sample.obs_start, sample.obs_start + self.obs_horizon)
            act_slice = slice(sample.action_start, sample.action_start + self.pred_horizon)
            frames = np.asarray(f[f"cam_data/{self.camera_key}"][obs_slice], dtype=np.float32)
            state = read_state_matrix(f, self.state_keys)[obs_slice]
            action = np.asarray(f[self.action_key][act_slice], dtype=np.float32)
        if frames.shape[0] != self.obs_horizon:
            raise ValueError(f"bad frame horizon for {sample}: {frames.shape}")
        if action.shape[0] != self.pred_horizon:
            raise ValueError(f"bad action horizon for {sample}: {action.shape}")

        height, width = self.resize
        resized = np.empty((self.obs_horizon, height, width, 3), dtype=np.float32)
        for i, frame in enumerate(frames):
            resized[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        video = np.transpose(resized / 127.5 - 1.0, (0, 3, 1, 2)).astype(np.float32)
        state = ((state - self.state_mean) / self.state_std).astype(np.float32)
        action = ((action - self.action_mean) / self.action_std).astype(np.float32)
        return {
            "video": torch.from_numpy(video),
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "task": sample.task,
            "episode": sample.episode,
            "action_start": sample.action_start,
        }


def g1_tasks_from_args(args: argparse.Namespace) -> list[str]:
    if args.tasks == "all":
        if not args.g1_pair_root.is_dir():
            raise FileNotFoundError(f"G1 pair root not found: {args.g1_pair_root}")
        tasks = sorted(path.name for path in args.g1_pair_root.iterdir() if path.is_dir())
    elif args.tasks == DEFAULT_TASKS:
        tasks = parse_csv(DEFAULT_G1_TASKS)
    else:
        tasks = parse_csv(args.tasks)
    if not tasks:
        raise ValueError("no G1 tasks selected")
    return tasks


def discover_g1_clips(args: argparse.Namespace) -> list[G1ClipInfo]:
    if not args.g1_pair_root.is_dir():
        raise FileNotFoundError(f"G1 pair root not found: {args.g1_pair_root}")
    if not args.g1_segment_root.is_dir():
        raise FileNotFoundError(f"G1 segment root not found: {args.g1_segment_root}")
    if not args.g1_raw_root.is_dir():
        raise FileNotFoundError(f"G1 raw root not found: {args.g1_raw_root}")

    clips: list[G1ClipInfo] = []
    for task in g1_tasks_from_args(args):
        task_dir = args.g1_pair_root / task
        manifest_path = task_dir / "manifest.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"G1 pair manifest not found: {manifest_path}")
        action_root = g1_action_root(args.g1_raw_root, task)
        task_count = 0
        for row in read_jsonl(manifest_path):
            if args.g1_max_clips_per_task > 0 and task_count >= args.g1_max_clips_per_task:
                break
            if args.g1_video_field not in row:
                raise ValueError(f"G1 manifest missing video field {args.g1_video_field!r}: {manifest_path}")
            if "source_frame_indices" not in row:
                raise ValueError(f"G1 manifest missing source_frame_indices: {manifest_path}")
            source_indices = tuple(int(v) for v in row["source_frame_indices"])
            if len(source_indices) != int(row.get("num_frames", len(source_indices))):
                raise ValueError(
                    f"G1 source_frame_indices length mismatch for {row.get('source_id')}: "
                    f"{len(source_indices)} vs num_frames={row.get('num_frames')}"
                )
            if len(source_indices) < args.obs_horizon + args.pred_horizon:
                continue

            source_segment_id = str(row.get("source_segment_id", ""))
            parts = source_segment_id.split("/")
            row_task = str(row.get("task", row.get("robot_task", task)))
            episode_name = str(row.get("episode", parts[1] if len(parts) >= 2 else ""))
            seg = str(row.get("seg", parts[2] if len(parts) >= 3 else ""))
            if row_task != task:
                raise ValueError(f"G1 manifest task mismatch: dir={task} row={row_task} source_id={row.get('source_id')}")
            if not episode_name or not seg:
                raise ValueError(f"G1 manifest cannot resolve episode/seg for source_id={row.get('source_id')}")
            episode = parse_g1_episode(episode_name)
            video_path = resolve_record_path(task_dir, str(row[args.g1_video_field]))
            if not video_path.is_file():
                raise FileNotFoundError(f"G1 video not found: {video_path}")
            joints_path = args.g1_segment_root / task / episode_name / f"{seg}_joints.parquet"
            if not joints_path.is_file():
                raise FileNotFoundError(f"G1 joints parquet not found: {joints_path}")
            clips.append(
                G1ClipInfo(
                    task=task,
                    episode=episode,
                    episode_name=episode_name,
                    seg=seg,
                    source_id=str(row.get("source_id", f"{task}/{episode_name}/{seg}_clip{task_count:06d}")),
                    source_segment_id=source_segment_id,
                    video_path=str(video_path),
                    joints_path=str(joints_path),
                    action_root=str(action_root),
                    source_frame_indices=source_indices,
                    clip_idx=int(row.get("clip_idx", task_count)),
                )
            )
            task_count += 1
    if not clips:
        raise ValueError(f"no G1 clips found under {args.g1_pair_root}")
    clips.sort(key=lambda item: (item.task, item.episode, item.seg, item.clip_idx, item.source_id))
    return clips


def split_g1_clips(
    clips: list[G1ClipInfo],
    train_ratio: float,
    seed: int,
    split_by: str,
) -> tuple[list[G1ClipInfo], list[G1ClipInfo]]:
    if split_by not in {"episode", "source_segment", "clip"}:
        raise ValueError(f"g1_split_by must be episode, source_segment or clip, got {split_by!r}")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
    rng = random.Random(seed)
    train: list[G1ClipInfo] = []
    val: list[G1ClipInfo] = []
    by_task: dict[str, list[G1ClipInfo]] = {}
    for clip in clips:
        by_task.setdefault(clip.task, []).append(clip)

    for task, group in sorted(by_task.items()):
        buckets: dict[object, list[G1ClipInfo]] = {}
        for clip in group:
            if split_by == "episode":
                key: object = clip.episode
            elif split_by == "source_segment":
                key = clip.source_segment_id
            else:
                key = clip.source_id
            buckets.setdefault(key, []).append(clip)
        keys = sorted(buckets)
        rng.shuffle(keys)
        if len(keys) < 2:
            raise ValueError(f"G1 split needs at least two {split_by} buckets for task={task}")
        n_train = max(1, min(len(keys) - 1, int(round(len(keys) * train_ratio))))
        train_keys = set(keys[:n_train])
        for key in keys:
            (train if key in train_keys else val).extend(buckets[key])
    train.sort(key=lambda item: (item.task, item.episode, item.seg, item.clip_idx, item.source_id))
    val.sort(key=lambda item: (item.task, item.episode, item.seg, item.clip_idx, item.source_id))
    return train, val


def build_g1_samples(
    clips: list[G1ClipInfo],
    obs_horizon: int,
    pred_horizon: int,
    frame_stride: int,
    max_samples: int,
    seed: int,
    sample_order: str,
) -> list[G1SampleInfo]:
    if obs_horizon <= 0 or pred_horizon <= 0:
        raise ValueError("obs_horizon and pred_horizon must be positive")
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")
    if sample_order not in {"random", "episode"}:
        raise ValueError(f"sample_order must be random or episode, got {sample_order!r}")
    samples: list[G1SampleInfo] = []
    for clip in clips:
        length = len(clip.source_frame_indices)
        for action_start in range(obs_horizon - 1, length - pred_horizon + 1, frame_stride):
            samples.append(
                G1SampleInfo(
                    task=clip.task,
                    episode=clip.episode,
                    source_id=clip.source_id,
                    video_path=clip.video_path,
                    joints_path=clip.joints_path,
                    action_root=clip.action_root,
                    source_frame_indices=clip.source_frame_indices,
                    obs_start=action_start - obs_horizon + 1,
                    action_start=action_start,
                )
            )
    if sample_order == "random":
        random.Random(seed).shuffle(samples)
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError("no G1 samples built; check horizons, stride and max sample limits")
    return samples


def load_g1_state_table(joints_path: str, state_keys: list[str]) -> dict[int, tuple[int, np.ndarray]]:
    import pandas as pd

    columns = list(dict.fromkeys(["frame_index", *state_keys]))
    df = pd.read_parquet(joints_path, columns=columns)
    required = {"frame_index", *state_keys}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"G1 joints parquet missing columns {missing}: {joints_path}")
    table: dict[int, tuple[int, np.ndarray]] = {}
    for local_index, (_, row) in enumerate(df.iterrows()):
        frame_index = int(row["frame_index"])
        table[local_index] = (
            frame_index,
            concat_vector(row, state_keys, context=f"{joints_path}:local={local_index}:frame={frame_index}"),
        )
    return table


def load_g1_action_tables(action_root: str, action_keys: list[str]) -> dict[int, dict[int, np.ndarray]]:
    import pandas as pd

    data_dir = Path(action_root) / "data"
    parquet_paths = sorted(data_dir.glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"G1 action parquet files not found under: {data_dir}")
    columns = list(dict.fromkeys(["episode_index", "frame_index", *action_keys]))
    frames = [pd.read_parquet(path, columns=columns) for path in parquet_paths]
    df = pd.concat(frames, ignore_index=True)
    required = {"episode_index", "frame_index", *action_keys}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"G1 action parquet missing columns {missing}: {action_root}")
    tables: dict[int, dict[int, np.ndarray]] = {}
    for _, row in df.iterrows():
        episode = int(row["episode_index"])
        frame_index = int(row["frame_index"])
        episode_table = tables.setdefault(episode, {})
        if frame_index in episode_table:
            raise ValueError(f"duplicate action episode={episode} frame_index={frame_index} in {action_root}")
        episode_table[frame_index] = concat_vector(
            row,
            action_keys,
            context=f"{action_root}:episode={episode}:frame={frame_index}",
        )
    return tables


class G1Lookup:
    def __init__(self, state_keys: list[str], action_keys: list[str], state_cache_size: int) -> None:
        self.state_keys = state_keys
        self.action_keys = action_keys
        self.state_cache_size = int(state_cache_size)
        if self.state_cache_size < 0:
            raise ValueError(f"state_cache_size must be non-negative, got {self.state_cache_size}")
        self._state_cache: OrderedDict[str, dict[int, tuple[int, np.ndarray]]] = OrderedDict()
        self._action_cache: dict[str, dict[int, dict[int, np.ndarray]]] = {}

    def _state_table(self, joints_path: str) -> dict[int, tuple[int, np.ndarray]]:
        if self.state_cache_size == 0:
            return load_g1_state_table(joints_path, self.state_keys)
        cached = self._state_cache.get(joints_path)
        if cached is not None:
            self._state_cache.move_to_end(joints_path)
            return cached
        table = load_g1_state_table(joints_path, self.state_keys)
        self._state_cache[joints_path] = table
        if len(self._state_cache) > self.state_cache_size:
            self._state_cache.popitem(last=False)
        return table

    def state_matrix(self, joints_path: str, frame_indices: tuple[int, ...]) -> np.ndarray:
        table = self._state_table(joints_path)
        missing = [idx for idx in frame_indices if idx not in table]
        if missing:
            raise ValueError(f"G1 segment-local state frames missing in {joints_path}: {missing[:8]}")
        return np.stack([table[idx][1] for idx in frame_indices], axis=0).astype(np.float32)

    def episode_frame_indices(self, joints_path: str, frame_indices: tuple[int, ...]) -> tuple[int, ...]:
        table = self._state_table(joints_path)
        missing = [idx for idx in frame_indices if idx not in table]
        if missing:
            raise ValueError(f"G1 segment-local frames missing in {joints_path}: {missing[:8]}")
        return tuple(int(table[idx][0]) for idx in frame_indices)

    def _action_tables(self, action_root: str) -> dict[int, dict[int, np.ndarray]]:
        cached = self._action_cache.get(action_root)
        if cached is not None:
            return cached
        tables = load_g1_action_tables(action_root, self.action_keys)
        self._action_cache[action_root] = tables
        return tables

    def action_matrix(self, action_root: str, episode: int, frame_indices: tuple[int, ...]) -> np.ndarray:
        tables = self._action_tables(action_root)
        if episode not in tables:
            raise ValueError(f"G1 action episode {episode} not found in {action_root}")
        table = tables[episode]
        missing = [idx for idx in frame_indices if idx not in table]
        if missing:
            raise ValueError(f"G1 action frames missing in {action_root} episode={episode}: {missing[:8]}")
        return np.stack([table[idx] for idx in frame_indices], axis=0).astype(np.float32)


def load_g1_video_tensor(video_path: str, resize: tuple[int, int], expected_frames: int) -> np.ndarray:
    height, width = resize
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open G1 video: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.float32))
    cap.release()
    if len(frames) != expected_frames:
        raise ValueError(f"G1 video frame count mismatch for {video_path}: {len(frames)} vs {expected_frames}")
    arr = np.stack(frames, axis=0)
    return np.transpose(arr / 127.5 - 1.0, (0, 3, 1, 2)).astype(np.float32)


def compute_g1_norm(clips: list[G1ClipInfo], args: argparse.Namespace) -> NormStats:
    state_keys = parse_csv(args.g1_state_keys)
    action_keys = parse_csv(args.g1_action_keys)
    std_floor = float(args.g1_norm_std_floor)
    if std_floor <= 0.0:
        raise ValueError(f"g1_norm_std_floor must be positive, got {std_floor}")
    lookup = G1Lookup(state_keys, action_keys, args.g1_state_cache_size)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    remaining = args.norm_max_frames
    ordered = list(clips)
    random.Random(args.seed).shuffle(ordered)
    for clip in ordered:
        indices = clip.source_frame_indices
        if remaining > 0 and len(indices) > remaining:
            pick = np.linspace(0, len(indices) - 1, remaining, dtype=np.int64)
            indices = tuple(indices[int(i)] for i in pick)
        states.append(lookup.state_matrix(clip.joints_path, indices))
        episode_indices = lookup.episode_frame_indices(clip.joints_path, indices)
        actions.append(lookup.action_matrix(clip.action_root, clip.episode, episode_indices))
        if remaining > 0:
            remaining -= len(indices)
            if remaining <= 0:
                break
    state_arr = np.concatenate(states, axis=0)
    action_arr = np.concatenate(actions, axis=0)
    return NormStats(
        action_mean=[float(v) for v in action_arr.mean(axis=0).tolist()],
        action_std=[float(v) for v in np.maximum(action_arr.std(axis=0), std_floor).tolist()],
        state_mean=[float(v) for v in state_arr.mean(axis=0).tolist()],
        state_std=[float(v) for v in np.maximum(state_arr.std(axis=0), std_floor).tolist()],
    )


class G1PairDataset(Dataset):
    def __init__(self, samples: list[G1SampleInfo], stats: NormStats, args: argparse.Namespace) -> None:
        self.samples = list(samples)
        self.obs_horizon = int(args.obs_horizon)
        self.pred_horizon = int(args.pred_horizon)
        self.resize = parse_resize(args.resize)
        self.lookup = G1Lookup(parse_csv(args.g1_state_keys), parse_csv(args.g1_action_keys), args.g1_state_cache_size)
        self.video_cache_size = int(args.g1_video_cache_size)
        if self.video_cache_size < 0:
            raise ValueError(f"g1_video_cache_size must be non-negative, got {self.video_cache_size}")
        self._video_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.action_mean = np.asarray(stats.action_mean, dtype=np.float32)
        self.action_std = np.asarray(stats.action_std, dtype=np.float32)
        self.state_mean = np.asarray(stats.state_mean, dtype=np.float32)
        self.state_std = np.asarray(stats.state_std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _video_tensor(self, sample: G1SampleInfo) -> np.ndarray:
        if self.video_cache_size == 0:
            return load_g1_video_tensor(sample.video_path, self.resize, len(sample.source_frame_indices))
        cached = self._video_cache.get(sample.video_path)
        if cached is not None:
            self._video_cache.move_to_end(sample.video_path)
            return cached
        video = load_g1_video_tensor(sample.video_path, self.resize, len(sample.source_frame_indices))
        self._video_cache[sample.video_path] = video
        if len(self._video_cache) > self.video_cache_size:
            self._video_cache.popitem(last=False)
        return video

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        obs_local = tuple(range(sample.obs_start, sample.obs_start + self.obs_horizon))
        action_local = tuple(range(sample.action_start, sample.action_start + self.pred_horizon))
        obs_segment_frames = tuple(sample.source_frame_indices[i] for i in obs_local)
        action_segment_frames = tuple(sample.source_frame_indices[i] for i in action_local)
        video = self._video_tensor(sample)[list(obs_local)]
        state = self.lookup.state_matrix(sample.joints_path, obs_segment_frames)
        action_episode_frames = self.lookup.episode_frame_indices(sample.joints_path, action_segment_frames)
        action = self.lookup.action_matrix(sample.action_root, sample.episode, action_episode_frames)
        state = ((state - self.state_mean) / self.state_std).astype(np.float32)
        action = ((action - self.action_mean) / self.action_std).astype(np.float32)
        return {
            "video": torch.from_numpy(video),
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "task": sample.task,
            "episode": sample.episode,
            "action_start": action_episode_frames[0],
            "source_id": sample.source_id,
        }


def collate(batch: list[dict]) -> dict:
    out = {
        "video": torch.stack([item["video"] for item in batch]),
        "state": torch.stack([item["state"] for item in batch]),
        "action": torch.stack([item["action"] for item in batch]),
        "task": [item["task"] for item in batch],
        "episode": torch.tensor([item["episode"] for item in batch], dtype=torch.long),
        "action_start": torch.tensor([item["action_start"] for item in batch], dtype=torch.long),
    }
    if "source_id" in batch[0]:
        out["source_id"] = [item["source_id"] for item in batch]
    return out


def build_h2r_loaders(args: argparse.Namespace, stats: NormStats | None = None) -> tuple[DataLoader, DataLoader, NormStats, dict]:
    episodes = discover_episodes(args)
    train_eps, val_eps = split_episodes(episodes, args.train_ratio, args.seed)
    train_samples = build_samples(train_eps, args.obs_horizon, args.pred_horizon, args.frame_stride, args.max_train_samples, args.seed)
    val_samples = build_samples(val_eps, args.obs_horizon, args.pred_horizon, args.frame_stride, args.max_val_samples, args.seed + 1)
    if stats is None:
        stats = compute_norm(train_eps, args)
    train_ds = H2RDataset(train_samples, stats, args)
    val_ds = H2RDataset(val_samples, stats, args)
    pin = str(args.device).startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin, collate_fn=collate)
    summary = {
        "format": "h2r_hdf5_direct",
        "data_root": str(args.data_root),
        "tasks": sorted({episode.task for episode in episodes}),
        "episodes_total": len(episodes),
        "episodes_train": len(train_eps),
        "episodes_val": len(val_eps),
        "samples_train": len(train_samples),
        "samples_val": len(val_samples),
        "obs_horizon": args.obs_horizon,
        "pred_horizon": args.pred_horizon,
        "state_dim": len(stats.state_mean),
        "action_dim": len(stats.action_mean),
        "state_keys": parse_csv(args.state_keys),
        "action_key": args.action_key,
        "camera_key": args.camera_key,
        "resize": args.resize,
    }
    return train_loader, val_loader, stats, summary


def build_g1_loaders(args: argparse.Namespace, stats: NormStats | None = None) -> tuple[DataLoader, DataLoader, NormStats, dict]:
    clips = discover_g1_clips(args)
    train_clips, val_clips = split_g1_clips(clips, args.train_ratio, args.seed, args.g1_split_by)
    train_samples = build_g1_samples(
        train_clips,
        args.obs_horizon,
        args.pred_horizon,
        args.frame_stride,
        args.max_train_samples,
        args.seed,
        args.train_sample_order,
    )
    val_samples = build_g1_samples(
        val_clips,
        args.obs_horizon,
        args.pred_horizon,
        args.frame_stride,
        args.max_val_samples,
        args.seed + 1,
        "episode",
    )
    if stats is None:
        stats = compute_g1_norm(train_clips, args)
    train_ds = G1PairDataset(train_samples, stats, args)
    val_ds = G1PairDataset(val_samples, stats, args)
    pin = str(args.device).startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=args.train_sample_order == "random",
        num_workers=args.workers,
        pin_memory=pin,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
        collate_fn=collate,
    )
    summary = {
        "format": "g1_2s_pair_manifest",
        "g1_pair_root": str(args.g1_pair_root),
        "g1_raw_root": str(args.g1_raw_root),
        "g1_segment_root": str(args.g1_segment_root),
        "g1_video_field": args.g1_video_field,
        "tasks": sorted({clip.task for clip in clips}),
        "clips_total": len(clips),
        "clips_train": len(train_clips),
        "clips_val": len(val_clips),
        "samples_train": len(train_samples),
        "samples_val": len(val_samples),
        "train_sample_order": args.train_sample_order,
        "g1_split_by": args.g1_split_by,
        "obs_horizon": args.obs_horizon,
        "pred_horizon": args.pred_horizon,
        "state_dim": len(stats.state_mean),
        "action_dim": len(stats.action_mean),
        "state_keys": parse_csv(args.g1_state_keys),
        "action_keys": parse_csv(args.g1_action_keys),
        "g1_norm_std_floor": args.g1_norm_std_floor,
        "resize": args.resize,
    }
    return train_loader, val_loader, stats, summary


def build_loaders(args: argparse.Namespace, stats: NormStats | None = None) -> tuple[DataLoader, DataLoader, NormStats, dict]:
    if args.dataset_kind == "h2r_hdf5":
        return build_h2r_loaders(args, stats)
    if args.dataset_kind == "g1_2s_pair":
        return build_g1_loaders(args, stats)
    raise ValueError(f"unsupported dataset_kind={args.dataset_kind!r}")


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(half - 1, 1)))
        emb = torch.cat([torch.sin(t.float()[:, None] * freqs[None]), torch.cos(t.float()[:, None] * freqs[None])], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        obs_horizon: int,
        pred_horizon: int,
        action_dim: int,
        state_dim: int,
        hidden_dim: int,
        time_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
        )
        self.cond = nn.Sequential(
            nn.Linear(obs_horizon * (hidden_dim + state_dim), hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time = nn.Sequential(TimeEmbedding(time_dim), nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        in_dim = pred_horizon * action_dim + hidden_dim * 2
        layers: list[nn.Module] = []
        cur = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(cur, hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
            cur = hidden_dim
        layers.append(nn.Linear(cur, pred_horizon * action_dim))
        self.denoiser = nn.Sequential(*layers)

    def encode_condition(self, video: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        bsz, obs, channels, height, width = video.shape
        if obs != self.obs_horizon or channels != 3:
            raise ValueError(f"expected video [B,{self.obs_horizon},3,H,W], got {tuple(video.shape)}")
        if state.shape[1:] != (self.obs_horizon, self.state_dim):
            raise ValueError(f"expected state [B,{self.obs_horizon},{self.state_dim}], got {tuple(state.shape)}")
        image = self.image_encoder(video.reshape(bsz * obs, channels, height, width)).reshape(bsz, obs, self.hidden_dim)
        return self.cond(torch.cat([image, state], dim=-1).reshape(bsz, -1))

    def forward(self, noisy_action: torch.Tensor, t: torch.Tensor, video: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        bsz = noisy_action.shape[0]
        cond = self.encode_condition(video, state)
        time = self.time(t)
        x = torch.cat([noisy_action.reshape(bsz, -1), cond, time], dim=-1)
        return self.denoiser(x).reshape(bsz, self.pred_horizon, self.action_dim)


class DDPMSchedule:
    def __init__(self, steps: int, beta_start: float, beta_end: float, device: torch.device) -> None:
        self.steps = int(steps)
        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        model: DiffusionPolicy,
        video: torch.Tensor,
        state: torch.Tensor,
        shape: tuple[int, int, int],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=video.device, generator=generator)
        for step in reversed(range(self.steps)):
            t = torch.full((shape[0],), step, device=video.device, dtype=torch.long)
            eps = model(x, t, video, state)
            beta = self.betas[step]
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            mean = (x - beta / torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha)
            if step == 0:
                x = mean
            else:
                noise = torch.randn(x.shape, device=x.device, generator=generator)
                x = mean + torch.sqrt(beta) * noise
        return x


def to_device(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    out["video"] = batch["video"].to(device)
    out["state"] = batch["state"].to(device)
    out["action"] = batch["action"].to(device)
    return out


def make_model(args: argparse.Namespace | dict, state_dim: int, action_dim: int) -> DiffusionPolicy:
    get = args.get if isinstance(args, dict) else lambda key: getattr(args, key)
    return DiffusionPolicy(
        obs_horizon=int(get("obs_horizon")),
        pred_horizon=int(get("pred_horizon")),
        action_dim=action_dim,
        state_dim=state_dim,
        hidden_dim=int(get("hidden_dim")),
        time_dim=int(get("time_dim")),
        depth=int(get("depth")),
        dropout=float(get("dropout")),
    )


def denoise_loss(
    model: DiffusionPolicy,
    schedule: DDPMSchedule,
    batch: dict,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    action = batch["action"]
    t = torch.randint(0, schedule.steps, (action.shape[0],), device=action.device, generator=generator)
    noise = torch.randn(action.shape, device=action.device, generator=generator)
    noisy = schedule.add_noise(action, noise, t)
    pred = model(noisy, t, batch["video"], batch["state"])
    return F.mse_loss(pred, noise)


def action_norm_tensors(stats: NormStats, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(stats.action_mean, dtype=torch.float32, device=device).view(1, 1, -1)
    std = torch.tensor(stats.action_std, dtype=torch.float32, device=device).view(1, 1, -1)
    return mean, std


def action_space_metrics(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    stats: NormStats,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, std = action_norm_tensors(stats, pred_norm.device)
    pred = pred_norm * std + mean
    target = target_norm * std + mean
    err = pred - target
    mse = err.pow(2).mean()
    rel = (
        torch.linalg.vector_norm(err.reshape(err.shape[0], -1), dim=1)
        / torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(1e-8)
    )
    per_horizon_rel = (
        torch.linalg.vector_norm(err, dim=2)
        / torch.linalg.vector_norm(target, dim=2).clamp_min(1e-8)
    )
    return mse, torch.sqrt(mse), rel, per_horizon_rel


@torch.no_grad()
def evaluate(
    model: DiffusionPolicy,
    schedule: DDPMSchedule,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    sample_actions: bool,
    stats: NormStats,
    sample_seed: int,
) -> dict:
    model.eval()
    sample_generator: torch.Generator | None = None
    denoise_generator: torch.Generator | None = None
    if sample_actions and sample_seed >= 0:
        sample_generator = torch.Generator(device=device)
        sample_generator.manual_seed(int(sample_seed))
    if sample_seed >= 0:
        denoise_generator = torch.Generator(device=device)
        denoise_generator.manual_seed(int(sample_seed) + 104729)
    losses: list[float] = []
    action_mse: list[float] = []
    action_mse_raw: list[float] = []
    action_rmse_raw: list[float] = []
    action_rel_l2_raw: list[float] = []
    per_horizon_sum: torch.Tensor | None = None
    per_horizon_rel_sum: torch.Tensor | None = None
    per_horizon_count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = to_device(batch, device)
        losses.append(float(denoise_loss(model, schedule, batch, denoise_generator).item()))
        if sample_actions:
            pred = schedule.sample(model, batch["video"], batch["state"], tuple(batch["action"].shape), sample_generator)
            err = (pred - batch["action"]).pow(2)
            action_mse.append(float(err.mean().item()))
            horizon = err.mean(dim=2).sum(dim=0).cpu()
            per_horizon_sum = horizon if per_horizon_sum is None else per_horizon_sum + horizon
            raw_mse, raw_rmse, raw_rel, raw_rel_per_horizon = action_space_metrics(pred, batch["action"], stats)
            action_mse_raw.append(float(raw_mse.item()))
            action_rmse_raw.append(float(raw_rmse.item()))
            action_rel_l2_raw.append(float(raw_rel.mean().item()))
            rel_horizon = raw_rel_per_horizon.sum(dim=0).cpu()
            per_horizon_rel_sum = rel_horizon if per_horizon_rel_sum is None else per_horizon_rel_sum + rel_horizon
            per_horizon_count += int(err.shape[0])
    if not losses:
        raise ValueError("eval produced no batches")
    metrics: dict[str, object] = {"denoise_loss": float(np.mean(losses))}
    if action_mse:
        metrics["sampled_action_mse_norm"] = float(np.mean(action_mse))
        metrics["sampled_action_mse_norm_per_horizon"] = [float(v) for v in (per_horizon_sum / per_horizon_count).tolist()]
        metrics["sampled_action_mse_action"] = float(np.mean(action_mse_raw))
        metrics["sampled_action_rmse_action"] = float(np.mean(action_rmse_raw))
        metrics["sampled_action_relative_l2_action"] = float(np.mean(action_rel_l2_raw))
        metrics["sampled_action_relative_l2_action_per_horizon"] = [
            float(v) for v in (per_horizon_rel_sum / per_horizon_count).tolist()
        ]
    return metrics


def config_from_args(args: argparse.Namespace) -> dict:
    keys = [
        "dataset_kind", "data_root", "tasks", "camera_key", "action_key", "state_keys", "max_episodes_per_task",
        "train_ratio", "obs_horizon", "pred_horizon", "frame_stride", "max_train_samples",
        "max_val_samples", "norm_max_frames", "resize", "batch_size", "workers", "seed",
        "train_sample_order", "g1_pair_root", "g1_raw_root", "g1_segment_root", "g1_video_field",
        "g1_state_keys", "g1_action_keys", "g1_split_by", "g1_max_clips_per_task",
        "g1_video_cache_size", "g1_state_cache_size", "g1_norm_std_floor",
        "hidden_dim", "time_dim", "depth", "dropout", "diffusion_steps", "beta_start", "beta_end",
        "best_metric", "eval_sample_seed",
    ]
    out = {key: getattr(args, key) for key in keys}
    for key in ["data_root", "g1_pair_root", "g1_raw_root", "g1_segment_root"]:
        out[key] = str(out[key])
    return out


def command_inspect(args: argparse.Namespace) -> None:
    train_loader, val_loader, stats, summary = build_loaders(args)
    batch = next(iter(train_loader))
    payload = {
        **summary,
        "normalization": asdict(stats),
        "train_batch_shapes": {
            "video": list(batch["video"].shape),
            "state": list(batch["state"].shape),
            "action": list(batch["action"].shape),
        },
        "val_batches": len(val_loader),
    }
    if args.output_json is not None:
        write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def command_train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    train_loader, val_loader, stats, summary = build_loaders(args)
    model = make_model(args, summary["state_dim"], summary["action_dim"]).to(device)
    schedule = DDPMSchedule(args.diffusion_steps, args.beta_start, args.beta_end, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    if log_path.exists() and not args.resume_log:
        log_path.unlink()
    write_json(args.output_dir / "dataset_summary.json", {**summary, "normalization": asdict(stats)})
    print(json.dumps({**summary, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}, ensure_ascii=False))

    train_iter = iter(train_loader)
    recent: list[float] = []
    best_metric_name = str(args.best_metric)
    best = math.inf
    best_metrics: dict = {}
    for step in range(1, args.steps + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = denoise_loss(model, schedule, batch)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        recent.append(float(loss.item()))
        if step == 1 or step % args.log_every == 0:
            row = {"step": step, "train_denoise_loss": float(np.mean(recent)), "last_train_denoise_loss": float(loss.item())}
            append_jsonl(log_path, row)
            print(json.dumps(row, ensure_ascii=False))
            recent.clear()
        if step == args.steps or step % args.eval_every == 0:
            metrics = evaluate(
                model,
                schedule,
                val_loader,
                device,
                args.eval_max_batches,
                args.eval_sample_actions,
                stats,
                args.eval_sample_seed,
            )
            row = {"step": step, **metrics}
            append_jsonl(log_path, row)
            print(json.dumps(row, ensure_ascii=False))
            if best_metric_name not in metrics:
                raise ValueError(
                    f"best metric {best_metric_name!r} was not produced; available metrics: {sorted(metrics)}"
                )
            metric_value = float(metrics[best_metric_name])
            if metric_value < best:
                best = metric_value
                best_metrics = row
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": config_from_args(args),
                        "normalization": asdict(stats),
                        "summary": summary,
                        "step": step,
                        "metrics": row,
                    },
                    args.output_dir / "best_checkpoint.pt",
                )

    final_metrics = evaluate(model, schedule, val_loader, device, args.eval_max_batches, True, stats, args.eval_sample_seed)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config_from_args(args),
            "normalization": asdict(stats),
            "summary": summary,
            "step": args.steps,
            "metrics": {"step": args.steps, **final_metrics},
        },
        args.output_dir / "last_checkpoint.pt",
    )
    write_json(
        args.output_dir / "train_summary.json",
        {"best_metric": best_metric_name, "best_metrics": best_metrics, "final_metrics": final_metrics, "dataset": summary},
    )


def command_eval(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    if not args.allow_cli_data:
        for key, value in config.items():
            if key == "eval_sample_seed":
                continue
            if hasattr(args, key):
                path_keys = {"data_root", "g1_pair_root", "g1_raw_root", "g1_segment_root"}
                setattr(args, key, Path(value) if key in path_keys else value)
    norm = ckpt["normalization"]
    stats = NormStats(
        action_mean=[float(v) for v in norm["action_mean"]],
        action_std=[float(v) for v in norm["action_std"]],
        state_mean=[float(v) for v in norm["state_mean"]],
        state_std=[float(v) for v in norm["state_std"]],
    )
    _, val_loader, _, summary = build_loaders(args, stats)
    model = make_model(config, summary["state_dim"], summary["action_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    schedule = DDPMSchedule(config["diffusion_steps"], config["beta_start"], config["beta_end"], device)
    metrics = evaluate(model, schedule, val_loader, device, args.eval_max_batches, True, stats, args.eval_sample_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "eval_summary.json", {"metrics": metrics, "dataset": summary})

    rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        sample_generator: torch.Generator | None = None
        if args.eval_sample_seed >= 0:
            sample_generator = torch.Generator(device=device)
            sample_generator.manual_seed(int(args.eval_sample_seed))
        for batch_idx, batch in enumerate(val_loader):
            if args.prediction_batches > 0 and batch_idx >= args.prediction_batches:
                break
            batch = to_device(batch, device)
            pred = schedule.sample(model, batch["video"], batch["state"], tuple(batch["action"].shape), sample_generator)
            err = (pred - batch["action"]).pow(2).mean(dim=(1, 2)).cpu().numpy()
            raw_mean, raw_std = action_norm_tensors(stats, device)
            pred_raw = pred * raw_std + raw_mean
            target_raw = batch["action"] * raw_std + raw_mean
            raw_err = pred_raw - target_raw
            raw_mse = raw_err.pow(2).mean(dim=(1, 2)).cpu().numpy()
            raw_rel = (
                torch.linalg.vector_norm(raw_err.reshape(raw_err.shape[0], -1), dim=1)
                / torch.linalg.vector_norm(target_raw.reshape(target_raw.shape[0], -1), dim=1).clamp_min(1e-8)
            ).cpu().numpy()
            for i, mse in enumerate(err):
                row = {
                    "task": batch["task"][i],
                    "episode": int(batch["episode"][i]),
                    "action_start": int(batch["action_start"][i]),
                    "mse_norm": float(mse),
                    "mse_action": float(raw_mse[i]),
                    "relative_l2_action": float(raw_rel[i]),
                }
                if "source_id" in batch:
                    row["source_id"] = batch["source_id"][i]
                rows.append(row)
    if rows:
        write_rows(args.output_dir / "predictions.csv", rows)
    print(json.dumps({"metrics": metrics, "output_dir": str(args.output_dir)}, ensure_ascii=False))


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-kind", choices=["h2r_hdf5", "g1_2s_pair"], default="h2r_hdf5")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--camera-key", default="robot_camera")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--state-keys", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--max-episodes-per-task", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--pred-horizon", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--norm-max-frames", type=int, default=20000)
    parser.add_argument("--resize", default="96x96")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-sample-order", choices=["random", "episode"], default="random")
    parser.add_argument("--g1-pair-root", type=Path, default=DEFAULT_G1_PAIR_ROOT)
    parser.add_argument("--g1-raw-root", type=Path, default=DEFAULT_G1_RAW_ROOT)
    parser.add_argument("--g1-segment-root", type=Path, default=DEFAULT_G1_SEGMENT_ROOT)
    parser.add_argument("--g1-video-field", default="video")
    parser.add_argument("--g1-state-keys", default=DEFAULT_G1_STATE_KEYS)
    parser.add_argument("--g1-action-keys", default=DEFAULT_G1_ACTION_KEYS)
    parser.add_argument("--g1-split-by", choices=["episode", "source_segment", "clip"], default="episode")
    parser.add_argument("--g1-max-clips-per-task", type=int, default=0)
    parser.add_argument("--g1-video-cache-size", type=int, default=8)
    parser.add_argument("--g1-state-cache-size", type=int, default=128)
    parser.add_argument("--g1-norm-std-floor", type=float, default=1e-2)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="H2R HDF5 Diffusion Policy BC")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_p = sub.add_parser("inspect")
    add_data_args(inspect_p)
    inspect_p.add_argument("--output-json", type=Path, default=None)
    inspect_p.set_defaults(func=command_inspect)

    train_p = sub.add_parser("train")
    add_data_args(train_p)
    add_model_args(train_p)
    train_p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    train_p.add_argument("--steps", type=int, default=1000)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--grad-clip", type=float, default=1.0)
    train_p.add_argument("--log-every", type=int, default=20)
    train_p.add_argument("--eval-every", type=int, default=100)
    train_p.add_argument("--eval-max-batches", type=int, default=0)
    train_p.add_argument("--eval-sample-actions", action="store_true")
    train_p.add_argument("--eval-sample-seed", type=int, default=12345)
    train_p.add_argument("--best-metric", default="denoise_loss")
    train_p.add_argument("--resume-log", action="store_true")
    train_p.set_defaults(func=command_train)

    eval_p = sub.add_parser("eval")
    add_data_args(eval_p)
    add_model_args(eval_p)
    eval_p.add_argument("--checkpoint", type=Path, required=True)
    eval_p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "eval")
    eval_p.add_argument("--eval-max-batches", type=int, default=0)
    eval_p.add_argument("--prediction-batches", type=int, default=4)
    eval_p.add_argument("--eval-sample-seed", type=int, default=12345)
    eval_p.add_argument("--allow-cli-data", action="store_true")
    eval_p.set_defaults(func=command_eval)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
