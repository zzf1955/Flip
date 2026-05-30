"""Action visibility mask utilities for IDM training.

The mask schema is intentionally explicit: IDM labels can be either the
24-dim arm-hand target or the 48-dim full-body target, and visibility is
mapped from robot body parts to the matching action dimensions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.config import MAIN_ROOT

ARM_DIM = 12
HAND_DIM = 12
ACTION_DIM = ARM_DIM + HAND_DIM
ROBOT_Q_DIM = 36
FULL_BODY_ACTION_DIM = ROBOT_Q_DIM + HAND_DIM
ACTION_MASK_VERSION = "action_mask_v1"
TARGET_MODES = ("arm_hand", "full_body")

ARM_HAND_BODY_PART_NAMES = (
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
)

FULL_BODY_PART_NAMES = (
    "torso",
    "left_leg",
    "right_leg",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
)

IDM_BODY_PART_NAMES = ARM_HAND_BODY_PART_NAMES

ARM_HAND_ACTION_DIM_NAMES = tuple(
    [f"ee_left_{idx}" for idx in range(6)]
    + [f"ee_right_{idx}" for idx in range(6)]
    + [f"hand_left_{idx}" for idx in range(6)]
    + [f"hand_right_{idx}" for idx in range(6)]
)

ARM_HAND_ACTION_DIM_PARTS = tuple(
    ["left_arm_or_hand"] * 6
    + ["right_arm_or_hand"] * 6
    + ["left_hand"] * 6
    + ["right_hand"] * 6
)

FULL_BODY_ACTION_DIM_NAMES = tuple(
    [f"robot_root_{idx}" for idx in range(7)]
    + [f"robot_left_leg_{idx}" for idx in range(6)]
    + [f"robot_right_leg_{idx}" for idx in range(6)]
    + [f"robot_waist_{idx}" for idx in range(3)]
    + [f"robot_left_arm_{idx}" for idx in range(7)]
    + [f"robot_right_arm_{idx}" for idx in range(7)]
    + [f"hand_left_{idx}" for idx in range(6)]
    + [f"hand_right_{idx}" for idx in range(6)]
)

FULL_BODY_ACTION_DIM_PARTS = tuple(
    ["torso"] * 7
    + ["left_leg"] * 6
    + ["right_leg"] * 6
    + ["torso"] * 3
    + ["left_arm_or_hand"] * 7
    + ["right_arm_or_hand"] * 7
    + ["left_hand"] * 6
    + ["right_hand"] * 6
)

ACTION_DIM_NAMES = ARM_HAND_ACTION_DIM_NAMES
ACTION_DIM_PARTS = ARM_HAND_ACTION_DIM_PARTS


@dataclass(frozen=True)
class ClipActionMask:
    mask: np.ndarray
    frame_ratios: np.ndarray
    visible_action_count: int
    visible_arm_count: int
    visible_hand_count: int
    visible_action_ratio: float
    visible_arm_ratio: float
    visible_hand_ratio: float
    mask_path: str


def validate_target_mode(target_mode: str) -> str:
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unsupported target_mode={target_mode!r}; expected {TARGET_MODES}")
    return target_mode


def action_dim_for_mode(target_mode: str) -> int:
    target_mode = validate_target_mode(target_mode)
    if target_mode == "arm_hand":
        return ACTION_DIM
    return FULL_BODY_ACTION_DIM


def body_part_names_for_mode(target_mode: str) -> tuple[str, ...]:
    target_mode = validate_target_mode(target_mode)
    if target_mode == "arm_hand":
        return ARM_HAND_BODY_PART_NAMES
    return FULL_BODY_PART_NAMES


def action_dim_names_for_mode(target_mode: str) -> tuple[str, ...]:
    target_mode = validate_target_mode(target_mode)
    if target_mode == "arm_hand":
        return ARM_HAND_ACTION_DIM_NAMES
    return FULL_BODY_ACTION_DIM_NAMES


def action_dim_parts_for_mode(target_mode: str) -> tuple[str, ...]:
    target_mode = validate_target_mode(target_mode)
    if target_mode == "arm_hand":
        return ARM_HAND_ACTION_DIM_PARTS
    return FULL_BODY_ACTION_DIM_PARTS


def default_action_mask_root() -> Path:
    return Path(MAIN_ROOT) / "training_data" / "action_mask"


def action_mask_from_part_visibility(
    part_visibility: np.ndarray,
    part_names: list[str] | tuple[str, ...],
    *,
    target_mode: str = "arm_hand",
) -> np.ndarray:
    """Map per-frame body-part visibility to IDM action dimensions."""

    target_mode = validate_target_mode(target_mode)
    part_visibility = np.asarray(part_visibility, dtype=bool)
    if part_visibility.ndim != 2:
        raise ValueError(
            f"part_visibility must be 2D [frames, parts], got {part_visibility.shape}"
        )
    name_to_idx = {str(name): idx for idx, name in enumerate(part_names)}
    required = set(body_part_names_for_mode(target_mode))
    missing = sorted(required - set(name_to_idx))
    if missing:
        raise ValueError(f"part visibility missing required parts: {missing}")

    n_frames = part_visibility.shape[0]
    action_mask = np.zeros((n_frames, action_dim_for_mode(target_mode)), dtype=bool)
    left_arm_or_hand = (
        part_visibility[:, name_to_idx["left_arm"]]
        | part_visibility[:, name_to_idx["left_hand"]]
    )
    right_arm_or_hand = (
        part_visibility[:, name_to_idx["right_arm"]]
        | part_visibility[:, name_to_idx["right_hand"]]
    )
    if target_mode == "arm_hand":
        action_mask[:, 0:6] = left_arm_or_hand[:, None]
        action_mask[:, 6:12] = right_arm_or_hand[:, None]
        action_mask[:, 12:18] = part_visibility[:, name_to_idx["left_hand"], None]
        action_mask[:, 18:24] = part_visibility[:, name_to_idx["right_hand"], None]
        return action_mask

    action_mask[:, 0:7] = part_visibility[:, name_to_idx["torso"], None]
    action_mask[:, 7:13] = part_visibility[:, name_to_idx["left_leg"], None]
    action_mask[:, 13:19] = part_visibility[:, name_to_idx["right_leg"], None]
    action_mask[:, 19:22] = part_visibility[:, name_to_idx["torso"], None]
    action_mask[:, 22:29] = left_arm_or_hand[:, None]
    action_mask[:, 29:36] = right_arm_or_hand[:, None]
    action_mask[:, 36:42] = part_visibility[:, name_to_idx["left_hand"], None]
    action_mask[:, 42:48] = part_visibility[:, name_to_idx["right_hand"], None]
    return action_mask


def aggregate_clip_action_mask(
    frame_action_mask: np.ndarray,
    frame_indices: list[int] | tuple[int, ...],
    *,
    min_frame_ratio: float,
    target_mode: str = "arm_hand",
) -> tuple[np.ndarray, np.ndarray]:
    """Select the middle-frame action mask for a clip."""

    action_dim = action_dim_for_mode(target_mode)
    if not 0.0 < min_frame_ratio <= 1.0:
        raise ValueError(
            f"min_frame_ratio must be in (0, 1], got {min_frame_ratio}"
        )
    frame_action_mask = np.asarray(frame_action_mask, dtype=bool)
    if frame_action_mask.ndim != 2 or frame_action_mask.shape[1] != action_dim:
        raise ValueError(
            f"frame_action_mask must have shape [frames, {action_dim}], "
            f"got {frame_action_mask.shape}"
        )
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")
    max_idx = max(int(idx) for idx in frame_indices)
    min_idx = min(int(idx) for idx in frame_indices)
    if min_idx < 0 or max_idx >= frame_action_mask.shape[0]:
        raise IndexError(
            f"clip frame indices out of mask range: min={min_idx} max={max_idx} "
            f"mask_frames={frame_action_mask.shape[0]}"
        )
    middle_index = int(frame_indices[len(frame_indices) // 2])
    middle_mask = frame_action_mask[middle_index].astype(bool)
    ratios = middle_mask.astype(np.float32)
    return ratios >= min_frame_ratio, ratios


def summarize_clip_mask(
    clip_mask: np.ndarray,
    frame_ratios: np.ndarray,
    mask_path: Path,
    *,
    target_mode: str = "arm_hand",
) -> ClipActionMask:
    action_dim = action_dim_for_mode(target_mode)
    clip_mask = np.asarray(clip_mask, dtype=bool)
    frame_ratios = np.asarray(frame_ratios, dtype=np.float32)
    if clip_mask.shape != (action_dim,):
        raise ValueError(f"clip action mask must be ({action_dim},), got {clip_mask.shape}")
    if frame_ratios.shape != (action_dim,):
        raise ValueError(
            f"clip action frame ratios must be ({action_dim},), got {frame_ratios.shape}"
        )
    visible_action_count = int(clip_mask.sum())
    if target_mode == "arm_hand":
        arm_mask = clip_mask[:ARM_DIM]
        hand_mask = clip_mask[ARM_DIM:]
    else:
        arm_mask = clip_mask[22:36]
        hand_mask = clip_mask[36:48]
    visible_arm_count = int(arm_mask.sum())
    visible_hand_count = int(hand_mask.sum())
    return ClipActionMask(
        mask=clip_mask.astype(np.float32),
        frame_ratios=frame_ratios.astype(np.float32),
        visible_action_count=visible_action_count,
        visible_arm_count=visible_arm_count,
        visible_hand_count=visible_hand_count,
        visible_action_ratio=float(visible_action_count / action_dim),
        visible_arm_ratio=float(visible_arm_count / len(arm_mask)),
        visible_hand_ratio=float(visible_hand_count / len(hand_mask)),
        mask_path=str(mask_path),
    )


class ActionMaskResolver:
    """Resolve persisted frame-level action masks for segment clips."""

    def __init__(
        self,
        root: str | Path,
        task_short: str,
        *,
        min_frame_ratio: float,
        target_mode: str = "arm_hand",
    ):
        self.root = Path(root)
        self.task_short = task_short
        self.min_frame_ratio = float(min_frame_ratio)
        self.target_mode = validate_target_mode(target_mode)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Action mask root not found: {self.root}")

    def mask_path(self, episode_name: str, segment: str) -> Path:
        return self.root / self.task_short / episode_name / f"{segment}.npz"

    def load_clip(
        self,
        episode_name: str,
        segment: str,
        frame_indices: list[int] | tuple[int, ...],
    ) -> ClipActionMask:
        path = self.mask_path(episode_name, segment)
        if not path.is_file():
            raise FileNotFoundError(f"Action mask artifact not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            if "action_mask" not in data:
                raise KeyError(f"action_mask missing from artifact: {path}")
            frame_action_mask = data["action_mask"].astype(bool)
            if "metadata_json" not in data:
                raise KeyError(f"metadata_json missing from artifact: {path}")
            metadata = json.loads(str(data["metadata_json"]))
        if metadata.get("version") != ACTION_MASK_VERSION:
            raise ValueError(
                f"Unsupported action mask version in {path}: {metadata.get('version')!r}"
            )
        if metadata.get("task") != self.task_short:
            raise ValueError(
                f"Action mask task mismatch for {path}: "
                f"expected {self.task_short!r}, got {metadata.get('task')!r}"
            )
        if metadata.get("episode") != episode_name:
            raise ValueError(
                f"Action mask episode mismatch for {path}: "
                f"expected {episode_name!r}, got {metadata.get('episode')!r}"
            )
        if metadata.get("segment") != segment:
            raise ValueError(
                f"Action mask segment mismatch for {path}: "
                f"expected {segment!r}, got {metadata.get('segment')!r}"
            )
        if metadata.get("target_mode") != self.target_mode:
            raise ValueError(
                f"Action mask target_mode mismatch for {path}: "
                f"expected {self.target_mode!r}, got {metadata.get('target_mode')!r}"
            )
        middle_index = int(frame_indices[len(frame_indices) // 2])
        if metadata.get("clip_middle_only", False):
            rendered = {int(idx) for idx in metadata.get("rendered_frame_indices", [])}
            if middle_index not in rendered:
                raise ValueError(
                    f"Clip middle frame {middle_index} was not rendered in action mask "
                    f"artifact {path}; regenerate masks with matching clip parameters"
                )
        clip_mask, ratios = aggregate_clip_action_mask(
            frame_action_mask,
            frame_indices,
            min_frame_ratio=self.min_frame_ratio,
            target_mode=self.target_mode,
        )
        return summarize_clip_mask(clip_mask, ratios, path, target_mode=self.target_mode)
