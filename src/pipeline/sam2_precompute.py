"""Precompute SAM2 segmentation masks for all training segments.

For each 4s segment (120 frames @ 30fps), runs FK → per-part bbox prompt →
SAM2 video propagation → combined binary mask, saved as compressed .npz.

Output:
  training_data/sam2_mask/{task}/{ep}/{seg}.npz
    masks: uint8 (N_frames, 480, 640), values 0 or 255

Multi-GPU: use --shard-index / --num-shards to split work across processes.
See batch_sam2_precompute.py for the multi-GPU launcher.

Usage:
  python -m src.pipeline.sam2_precompute --task all --device cuda:0
  python -m src.pipeline.sam2_precompute --task all --device cuda:0 \
      --shard-index 0 --num-shards 4
"""

import argparse
import fnmatch
import gc
import os
import re
import sys
import tempfile
import time

import av
import cv2
import numpy as np
import pandas as pd
import pinocchio as pin
import torch

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import (
    ALL_TASKS, BEST_PARAMS, G1_URDF, MAIN_ROOT, MESH_DIR,
    get_hand_type, get_skip_meshes,
)
from src.core.camera import make_camera, make_camera_const, project_points_cv
from src.core.fk import build_q, do_fk, parse_urdf_meshes, preload_meshes

_MAIN_TRAINING_DATA = os.path.join(MAIN_ROOT, "training_data")
_SEGMENT_DIR = os.path.join(_MAIN_TRAINING_DATA, "segment")
_DEFAULT_OUTPUT = os.path.join(_MAIN_TRAINING_DATA, "sam2_mask")

VIDEO_H, VIDEO_W = 480, 640

SAM2_MODEL_IDS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}

BODY_PARTS = {
    "left_arm":  ["left_shoulder_*", "left_elbow_*", "left_wrist_*"],
    "left_hand": ["left_base_link", "left_palm_force_sensor",
                  "left_thumb_*", "left_index_*", "left_middle_*",
                  "left_ring_*", "left_little_*"],
    "right_arm": ["right_shoulder_*", "right_elbow_*", "right_wrist_*"],
    "right_hand":["right_base_link", "right_palm_force_sensor",
                  "right_thumb_*", "right_index_*", "right_middle_*",
                  "right_ring_*", "right_little_*"],
    "left_leg":  ["left_hip_*", "left_knee_*", "left_ankle_*"],
    "right_leg": ["right_hip_*", "right_knee_*", "right_ankle_*"],
    "torso":     ["pelvis", "pelvis_contour_link", "waist_*", "torso_link"],
}

PART_IDS = {name: i + 1 for i, name in enumerate(BODY_PARTS)}


# ── Segment discovery ───────────────────────────────────────────────

def find_segments(tasks: list[str]) -> list[dict]:
    segments = []
    for task in tasks:
        task_dir = os.path.join(_SEGMENT_DIR, task)
        if not os.path.isdir(task_dir):
            continue
        for root, _, files in os.walk(task_dir):
            for fname in sorted(files):
                m = re.match(r"(seg\d+)_video\.mp4$", fname)
                if not m:
                    continue
                seg = m.group(1)
                ep_dir = os.path.relpath(root, task_dir)
                segments.append({
                    "task": task,
                    "episode": ep_dir,
                    "seg": seg,
                    "video_path": os.path.join(root, fname),
                    "parquet_path": os.path.join(root, f"{seg}_joints.parquet"),
                })
    segments.sort(key=lambda s: f"{s['task']}/{s['episode']}/{s['seg']}")
    return segments


# ── FK helpers ──────────────────────────────────────────────────────

def match_links(mesh_cache, patterns, skip_set):
    matched = {}
    for link_name, data in mesh_cache.items():
        if link_name in skip_set:
            continue
        for pat in patterns:
            if fnmatch.fnmatch(link_name, pat):
                matched[link_name] = data
                break
    return matched


def render_mask_for_links(filtered_cache, transforms, params, h, w):
    if not filtered_cache:
        return np.zeros((h, w), dtype=np.uint8)
    K, D, rvec, tvec, R_w2c, t_w2c, _fisheye = make_camera(params, transforms)
    mask = np.zeros((h, w), dtype=np.uint8)
    for link_name, (tris, _) in filtered_cache.items():
        if link_name not in transforms:
            continue
        t_link, R_link = transforms[link_name]
        flat = tris.reshape(-1, 3)
        world = (R_link @ flat.T).T + t_link
        cam_pts = (R_w2c @ world.T).T + t_w2c.flatten()
        z_cam = cam_pts[:, 2]
        pts2d = project_points_cv(
            world.reshape(-1, 1, 3), rvec, tvec, K, D, _fisheye)
        pts2d = pts2d.reshape(-1, 2)
        n_tri = len(tris)
        z_tri = z_cam.reshape(n_tri, 3)
        pts_tri = pts2d.reshape(n_tri, 3, 2)
        valid = (z_tri > 0.01).all(axis=1)
        pts_tri = pts_tri[valid]
        if len(pts_tri) == 0:
            continue
        finite = np.all(np.isfinite(pts_tri), axis=(1, 2))
        pts_tri = pts_tri[finite].astype(np.int32)
        if len(pts_tri) > 0:
            cv2.fillPoly(mask, pts_tri, 255)
    return mask


def mask_to_bbox(mask, margin=0):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    return np.array([
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(w, int(xs.max()) + margin),
        min(h, int(ys.max()) + margin),
    ], dtype=np.float32)


# ── FK model loading ────────────────────────────────────────────────

def load_fk_models():
    model = pin.buildModelFromUrdf(str(G1_URDF), pin.JointModelFreeFlyer())
    data = model.createData()
    link_meshes = parse_urdf_meshes(str(G1_URDF))
    cam_const = make_camera_const(BEST_PARAMS)
    return model, data, link_meshes, cam_const


def build_caches(link_meshes, hand_type):
    skip_set = get_skip_meshes(hand_type)
    mesh_cache = preload_meshes(link_meshes, str(MESH_DIR), skip_set)
    part_caches = {}
    for part_name, patterns in BODY_PARTS.items():
        part_caches[part_name] = match_links(mesh_cache, patterns, skip_set)
    return mesh_cache, part_caches


# ── Core: compute masks for one segment ─────────────────────────────

def compute_segment_mask(
    seg_info: dict,
    predictor,
    fk_model, fk_data, part_caches, cam_const,
    hand_type: str,
    prompt_interval: int,
    bbox_margin: int,
    min_visible_area: int,
    device: str,
) -> np.ndarray:
    """Run SAM2 on one segment. Returns (N_frames, H, W) uint8 mask array."""

    video_path = seg_info["video_path"]
    parquet_path = seg_info["parquet_path"]

    # Read video frames
    container = av.open(video_path)
    stream = container.streams.video[0]
    frames = [f.to_ndarray(format="bgr24") for f in container.decode(stream)]
    container.close()
    n_frames = len(frames)

    # Read parquet
    df = pd.read_parquet(parquet_path)

    # Write frames to temp JPEG dir for SAM2
    jpeg_dir = tempfile.mkdtemp(prefix="sam2_")
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(jpeg_dir, f"{i:05d}.jpg"), frame)

    # Phase A: FK → per-part bbox prompts
    all_prompts = []
    prev_visible = set()
    h, w = frames[0].shape[:2]

    for seq_idx in range(min(n_frames, len(df))):
        row = df.iloc[seq_idx]
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
        q = build_q(fk_model, rq, hs, hand_type=hand_type)
        transforms = do_fk(fk_model, fk_data, q)

        cur_visible = set()
        part_masks = {}
        for part_name, pcache in part_caches.items():
            if not pcache:
                continue
            m = render_mask_for_links(pcache, transforms, BEST_PARAMS, h, w)
            if np.count_nonzero(m) >= min_visible_area:
                cur_visible.add(part_name)
                part_masks[part_name] = m

        is_periodic = (seq_idx == 0 or seq_idx % prompt_interval == 0)
        newly_appeared = cur_visible - prev_visible
        parts_to_prompt = set()
        if is_periodic:
            parts_to_prompt = cur_visible.copy()
        if newly_appeared:
            parts_to_prompt |= newly_appeared

        for part_name in parts_to_prompt:
            bbox = mask_to_bbox(part_masks[part_name], margin=bbox_margin)
            if bbox is not None:
                all_prompts.append((seq_idx, part_name, bbox))

        prev_visible = cur_visible

    # Phase B: SAM2 video propagation
    id_to_part = {v: k for k, v in PART_IDS.items()}
    frame_masks = np.zeros((n_frames, h, w), dtype=np.uint8)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.startswith("cuda")
                    else torch.autocast(device_type="cpu", enabled=False))

    with torch.inference_mode(), autocast_ctx:
        state = predictor.init_state(
            video_path=jpeg_dir, offload_video_to_cpu=True)

        for seq_idx, part_name, bbox in all_prompts:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=seq_idx,
                obj_id=PART_IDS[part_name],
                box=bbox,
            )

        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            masks = (mask_logits > 0.0).cpu().numpy()
            for i, oid in enumerate(obj_ids):
                pn = id_to_part.get(oid)
                if pn is None:
                    continue
                m = masks[i, 0].astype(np.uint8) * 255
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, morph_kernel)
                frame_masks[frame_idx] = np.maximum(frame_masks[frame_idx], m)

    predictor.reset_state(state)
    del state

    # Cleanup temp JPEG dir
    import shutil
    shutil.rmtree(jpeg_dir, ignore_errors=True)

    return frame_masks


# ── Main processing loop ────────────────────────────────────────────

def process_segments(segments: list[dict], predictor, device: str, args):
    fk_model, fk_data, link_meshes, cam_const = load_fk_models()

    cache_by_hand: dict[str, tuple] = {}
    t_start = time.time()
    n_done = 0
    n_skip = 0
    n_fail = 0

    for i, seg in enumerate(segments):
        task_full = f"G1_WBT_{seg['task']}"
        hand_type = get_hand_type(task_full)
        out_dir = os.path.join(args.output, seg["task"], seg["episode"])
        out_path = os.path.join(out_dir, f"{seg['seg']}.npz")

        if args.resume and os.path.isfile(out_path):
            n_skip += 1
            continue

        if not os.path.isfile(seg["parquet_path"]):
            print(f"  WARN: parquet not found: {seg['parquet_path']}")
            n_fail += 1
            continue

        if hand_type not in cache_by_hand:
            cache_by_hand[hand_type] = build_caches(link_meshes, hand_type)
        _, part_caches = cache_by_hand[hand_type]

        seg_id = f"{seg['task']}/{seg['episode']}/{seg['seg']}"
        t0 = time.time()

        masks = compute_segment_mask(
            seg, predictor,
            fk_model, fk_data, part_caches, cam_const,
            hand_type,
            args.prompt_interval, args.bbox_margin, args.min_visible_area,
            device,
        )

        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(out_path, masks=masks)

        n_done += 1
        dt = time.time() - t0
        elapsed = time.time() - t_start
        rate = n_done / elapsed if elapsed > 0 else 0

        if n_done % 10 == 0 or n_done <= 3:
            remaining = len(segments) - n_skip - n_done - n_fail
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{n_done}/{len(segments)}] {seg_id} "
                  f"{dt:.1f}s ({masks.shape[0]} frames, "
                  f"{os.path.getsize(out_path)/1024:.0f}KB) "
                  f"rate={rate:.2f}/s eta={eta/3600:.1f}h")

        if n_done % 50 == 0:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.time() - t_start
    print(f"\nDone. {n_done} processed, {n_skip} skipped, {n_fail} failed. "
          f"{elapsed:.1f}s ({elapsed/3600:.1f}h)")


# ── CLI ─────────────────────────────────────────────────────────────

def parse_tasks(task_arg: str) -> list[str]:
    key = task_arg.strip().lower()
    groups = {"all": None, "inspire": "Inspire_", "brainco": "Brainco_"}
    if key in groups:
        prefix = groups[key]
        short = [t.replace("G1_WBT_", "") for t in ALL_TASKS]
        return [t for t in short if prefix is None or t.startswith(prefix)]
    return [t.strip() for t in task_arg.split(",")]


def main():
    ap = argparse.ArgumentParser(
        description="Precompute SAM2 masks for training segments")
    ap.add_argument("--task", required=True,
                    help="task filter: 'all', 'inspire', 'brainco', or comma-separated")
    ap.add_argument("--sam2-model", choices=list(SAM2_MODEL_IDS), default="tiny")
    ap.add_argument("--prompt-interval", type=int, default=30)
    ap.add_argument("--bbox-margin", type=int, default=0)
    ap.add_argument("--min-visible-area", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=_DEFAULT_OUTPUT)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-segments", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = args.device
    tasks = parse_tasks(args.task)

    print(f"SAM2 Mask Precompute")
    print(f"  tasks:          {tasks}")
    print(f"  sam2-model:     {args.sam2_model}")
    print(f"  device:         {device}")
    print(f"  shard:          {args.shard_index}/{args.num_shards}")
    print(f"  output:         {args.output}")

    all_segs = find_segments(tasks)
    print(f"  found {len(all_segs)} segments")

    if args.max_segments > 0:
        all_segs = all_segs[:args.max_segments]
        print(f"  limited to {len(all_segs)} segments")

    if args.num_shards > 1:
        all_segs = [s for i, s in enumerate(all_segs)
                    if i % args.num_shards == args.shard_index]
        print(f"  shard {args.shard_index}: {len(all_segs)} segments")

    if not all_segs:
        print("No segments to process")
        return

    # Load SAM2
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    model_id = SAM2_MODEL_IDS[args.sam2_model]
    print(f"Loading SAM2 ({model_id})...")
    predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)

    process_segments(all_segs, predictor, device, args)


if __name__ == "__main__":
    main()
