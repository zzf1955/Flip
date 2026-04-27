"""Batch pipeline: segment inpaint + human overlay.

For each 4s video segment:
  Phase A: FK → SAM2 prompts (extract frames + compute FK masks)
  Phase B: SAM2 video segmentation (robot mask)
  Phase C: Mask postprocess → ProPainter/LaMa inpaint (clean background)
  Phase D: SMPLH retarget → render human overlay on clean background

Usage:
  python -m src.pipeline.segment_pipeline \
    --manifest training/data/segment/manifest.json \
    --limit 20 --device cuda:1
"""

import argparse
import fnmatch
import gc
import json
import os
import shutil
import subprocess
import sys
import time

import av
import cv2
import numpy as np
import pandas as pd
import pinocchio as pin
import torch

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import (
    BASE_DIR, G1_URDF, MESH_DIR, BEST_PARAMS, CAMERA_MODEL,
    PROPAINTER_ROOT, MAIN_ROOT,
    SEGMENT_DIR, SEGMENT_PIPELINE_DIR, OVERLAY_4S_DIR,
    get_hand_type, get_skip_meshes,
)
from src.core.camera import make_camera, make_camera_const, project_points_cv
from src.core.fk import build_q, do_fk, parse_urdf_meshes, preload_meshes
from src.core.render import render_mask_and_overlay, render_mesh_on_image
from src.core.mask import postprocess_mask, init_lama, run_lama
from src.core.smplh import SMPLHForIK, extract_g1_targets, R_SMPLH_TO_G1_NP
from src.core.retarget import (
    retarget_frame, refine_arms,
    compute_g1_rest_transforms, scale_hands, apply_finger_curl_from_g1,
    DEFAULT_BODY_SCALE, DEFAULT_HAND_SCALE, DEFAULT_ROOT_OFFSET_G1,
)
from src.core.data import open_video_writer, write_frame, close_video

DEFAULT_INTERMEDIATE_ROOT = SEGMENT_PIPELINE_DIR
DEFAULT_FINAL_ROOT = OVERLAY_4S_DIR

# ── Body part definitions (from sam2_inpaint.py) ──

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

PART_COLORS = {
    "left_arm":  (0, 100, 255),
    "left_hand": (0, 200, 255),
    "right_arm": (255, 100, 0),
    "right_hand":(255, 200, 0),
    "left_leg":  (200, 0, 200),
    "right_leg": (100, 0, 200),
    "torso":     (100, 200, 0),
}


# ── Helper functions (from sam2_inpaint.py) ──

def draw_box_prompt(img, part_name, box):
    color = PART_COLORS[part_name]
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, part_name, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

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


def mask_to_point_prompts(mask, bbox, neg_margin=16):
    """Build one positive centroid and four outside negative point prompts."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or bbox is None:
        return None, None
    h, w = mask.shape
    points = [[float(xs.mean()), float(ys.mean())]]
    labels = [1]

    x1, y1, x2, y2 = bbox.astype(float)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    candidates = [
        [cx, y1 - neg_margin],
        [cx, y2 + neg_margin],
        [x1 - neg_margin, cy],
        [x2 + neg_margin, cy],
    ]
    for x, y in candidates:
        x = float(np.clip(x, 0, w - 1))
        y = float(np.clip(y, 0, h - 1))
        if mask[int(round(y)), int(round(x))] == 0:
            points.append([x, y])
            labels.append(0)
    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)


# ── Model loading ──

def load_models(device, sam2_model, inpaint_method, skip_human=False):
    """Load all models once. Returns a dict of models."""
    models = {}

    # G1 URDF + FK
    print("Loading G1 URDF...")
    model_pin = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model_pin.createData()
    link_meshes = parse_urdf_meshes(G1_URDF)
    models["pin_model"] = model_pin
    models["pin_data"] = data_pin
    models["link_meshes"] = link_meshes

    # Camera
    models["cam_params"] = BEST_PARAMS
    models["cam_const"] = make_camera_const(BEST_PARAMS)

    # SAM2
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    MODEL_IDS = {
        "tiny": "facebook/sam2.1-hiera-tiny",
        "small": "facebook/sam2.1-hiera-small",
        "base": "facebook/sam2.1-hiera-base-plus",
        "large": "facebook/sam2.1-hiera-large",
    }
    model_id = MODEL_IDS[sam2_model]
    print(f"Loading SAM2 ({model_id})...")
    predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)
    models["sam2"] = predictor

    # Inpaint
    models["inpaint_method"] = inpaint_method
    if inpaint_method == "lama":
        print("Loading LaMa...")
        models["lama"] = init_lama()

    # SMPLH (for human overlay)
    if not skip_human:
        print(f"Loading SMPLH (device={device})...")
        smplh = SMPLHForIK(device=device)
        J_shaped, v_shaped = smplh.shape_blend(
            None, body_scale=DEFAULT_BODY_SCALE)
        J_shaped, v_shaped = scale_hands(
            smplh, J_shaped, v_shaped, DEFAULT_HAND_SCALE)

        # Drop head/neck faces
        HEAD_JOINTS = [12, 15]
        weights_np = smplh.weights.cpu().numpy()
        v_head_w = weights_np[:, HEAD_JOINTS].sum(axis=1)
        faces_all = smplh.faces
        face_head_w = v_head_w[faces_all].max(axis=1)
        faces_nohead = faces_all[face_head_w < 0.3]

        rest_transforms = compute_g1_rest_transforms()

        models["smplh"] = smplh
        models["J_shaped"] = J_shaped
        models["v_shaped"] = v_shaped
        models["faces_nohead"] = faces_nohead
        models["rest_transforms"] = rest_transforms

    models["device"] = device
    print("All models loaded.\n")
    return models


def build_caches(link_meshes, hand_type):
    """Build mesh_cache and part_caches for a given hand_type."""
    skip_set = get_skip_meshes(hand_type)
    mesh_cache = preload_meshes(link_meshes, MESH_DIR, skip_set)
    part_caches = {}
    for part_name, patterns in BODY_PARTS.items():
        part_caches[part_name] = match_links(mesh_cache, patterns, skip_set)
    return mesh_cache, part_caches, skip_set


# ── Single segment processor ──

def process_segment(seg_info, task_manifest, models, args):
    """Process one segment through the full pipeline.

    Outputs:
      Intermediate (output/segment_pipeline/{task}/ep{N}/seg{M}/):
        01_mesh_overlay.mp4, 02_mesh_mask.mp4, 03_mesh_bbox.mp4,
        04_sam2_mask.mp4, 05_sam2_postproc.mp4, 06_sam2_overlay.mp4,
        07_inpaint.mp4
      Final (training_data/overlay/4s/{task}/ep{N}/):
        seg{M}_human.mp4
    """
    task_short = task_manifest["task_short"]
    hand_type = task_manifest["hand_type"]
    ep = seg_info["episode"]
    seg_idx = seg_info["segment_index"]

    seg_id = f"{task_short}/ep{ep:03d}/seg{seg_idx:02d}"
    seg_base = os.path.join(args.segment_root, task_short)
    seg_video = os.path.join(seg_base, seg_info["video"])
    seg_parquet = os.path.join(seg_base, seg_info["joints"])

    # Output paths: intermediate stages + final human render
    inter_dir = os.path.join(args.intermediate_root, task_short,
                              f"ep{ep:03d}", f"seg{seg_idx:02d}")
    os.makedirs(inter_dir, exist_ok=True)
    out01_overlay = os.path.join(inter_dir, "01_mesh_overlay.mp4")
    out02_mask = os.path.join(inter_dir, "02_mesh_mask.mp4")
    out03_bbox = os.path.join(inter_dir, "03_mesh_bbox.mp4")
    out04_sam2 = os.path.join(inter_dir, "04_sam2_mask.mp4")
    out05_postproc = os.path.join(inter_dir, "05_sam2_postproc.mp4")
    out06_overlay = os.path.join(inter_dir, "06_sam2_overlay.mp4")
    out07_inpaint = os.path.join(inter_dir, "07_inpaint.mp4")

    final_dir = os.path.join(args.final_root, task_short, f"ep{ep:03d}")
    os.makedirs(final_dir, exist_ok=True)
    human_out = os.path.join(final_dir, f"seg{seg_idx:02d}_human.mp4")

    def _exists(p):
        return os.path.isfile(p) and os.path.getsize(p) > 0

    # Resume: if human done (and we want it), skip entirely
    human_done = _exists(human_out)
    inpaint_done = _exists(out07_inpaint)
    if args.resume and human_done and (args.skip_human or True):
        print(f"  [{seg_id}] skipped (human exists)")
        return True
    if args.resume and args.skip_human and inpaint_done:
        print(f"  [{seg_id}] skipped (inpaint exists, skip-human)")
        return True

    if not os.path.isfile(seg_video):
        print(f"  [{seg_id}] WARN: video not found: {seg_video}")
        return False

    seg_df = pd.read_parquet(seg_parquet)
    n_frames = len(seg_df)

    model_pin = models["pin_model"]
    data_pin = models["pin_data"]
    cam_params = models["cam_params"]
    cam_const = models["cam_const"]
    mesh_cache, part_caches, skip_set = models["_caches"]

    fps = 30
    h, w = 480, 640
    device = models["device"]

    tmp_dir = os.path.join(args.intermediate_root, ".tmp",
                            f"{task_short}_ep{ep}_seg{seg_idx}")
    jpeg_dir = os.path.join(tmp_dir, "frames")
    os.makedirs(jpeg_dir, exist_ok=True)

    t_seg_start = time.time()
    inpaint_frames = None  # populated either by Phase C or by reading 07_inpaint

    # Skip Phase A-C if inpaint already exists
    skip_inpaint = args.resume and inpaint_done

    if skip_inpaint:
        # Read inpaint from disk for Phase D
        c = av.open(out07_inpaint)
        s = c.streams.video[0]
        inpaint_frames = []
        for av_frame in c.decode(s):
            inpaint_frames.append(av_frame.to_ndarray(format='bgr24'))
        c.close()
        processed = len(inpaint_frames)
        if processed > 0:
            h, w = inpaint_frames[0].shape[:2]
        print(f"  [{seg_id}] reusing inpaint ({processed} frames)")
    else:
        # ──────────────────────────────────────
        # Phase A: Extract frames + FK + prompts
        #   Writes: 01_mesh_overlay, 02_mesh_mask, 03_mesh_bbox
        # ──────────────────────────────────────
        container_in = av.open(seg_video)
        stream_in = container_in.streams.video[0]

        all_prompts = []
        per_frame_prompts = {}  # seq_idx -> list of (part_name, bbox) for stage 03 viz
        prev_visible = set()
        frames_bgr = []

        w01 = w02 = w03 = None  # writers init on first frame

        for seq_idx_f, av_frame in enumerate(container_in.decode(stream_in)):
            if seq_idx_f >= n_frames:
                break

            img = av_frame.to_ndarray(format='bgr24')
            h, w = img.shape[:2]
            frames_bgr.append(img)
            cv2.imwrite(os.path.join(jpeg_dir, f"{seq_idx_f:05d}.jpg"), img)

            row = seg_df.iloc[seq_idx_f]
            rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
            hs = np.array(row["observation.state.hand_state"], dtype=np.float64)

            q = build_q(model_pin, rq, hs, hand_type=hand_type)
            transforms = do_fk(model_pin, data_pin, q)

            # FK mesh overlay + binary mask (single pass)
            fk_mask, fk_overlay = render_mask_and_overlay(
                img, mesh_cache, transforms, cam_params, h, w, cam_const)

            if w01 is None:
                w01 = open_video_writer(out01_overlay, w, h, fps)
                w02 = open_video_writer(out02_mask, w, h, fps)
                w03 = open_video_writer(out03_bbox, w, h, fps)
            write_frame(*w01, fk_overlay)
            write_frame(*w02, cv2.cvtColor(fk_mask, cv2.COLOR_GRAY2BGR))

            # Per-part visibility + prompts
            cur_visible = set()
            part_masks = {}
            for part_name, pcache in part_caches.items():
                if not pcache:
                    continue
                m = render_mask_for_links(pcache, transforms, cam_params, h, w)
                if np.count_nonzero(m) >= args.min_visible_area:
                    cur_visible.add(part_name)
                    part_masks[part_name] = m

            is_periodic = (seq_idx_f == 0 or seq_idx_f % args.prompt_interval == 0)
            newly_appeared = cur_visible - prev_visible
            parts_to_prompt = set()
            if is_periodic:
                parts_to_prompt = cur_visible.copy()
            if newly_appeared:
                parts_to_prompt |= newly_appeared

            frame_prompts = []
            for part_name in parts_to_prompt:
                bbox = mask_to_bbox(part_masks[part_name], margin=args.bbox_margin)
                if bbox is not None:
                    points = labels = None
                    if args.point_prompts:
                        points, labels = mask_to_point_prompts(
                            part_masks[part_name], bbox,
                            neg_margin=args.negative_point_margin)
                    all_prompts.append((seq_idx_f, part_name, bbox, points, labels))
                    frame_prompts.append((part_name, bbox))
            per_frame_prompts[seq_idx_f] = frame_prompts

            # 03_mesh_bbox.mp4: draw bboxes on image
            bbox_img = img.copy()
            for part_name, bbox in frame_prompts:
                draw_box_prompt(bbox_img, part_name, bbox)
            write_frame(*w03, bbox_img)

            prev_visible = cur_visible

        container_in.close()
        processed = len(frames_bgr)
        if w01: close_video(*w01)
        if w02: close_video(*w02)
        if w03: close_video(*w03)

        # ──────────────────────────────────────
        # Phase B: SAM2 segmentation
        #   Writes: 04_sam2_mask
        # ──────────────────────────────────────
        predictor = models["sam2"]
        id_to_part = {v: k for k, v in PART_IDS.items()}
        frame_part_masks = {}

        # Wrap in autocast to avoid bfloat16/float32 mismatch that accumulates
        # across multiple init_state/reset_state calls in the same process.
        autocast_ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                        if device.startswith('cuda')
                        else torch.autocast(device_type='cpu', enabled=False))
        with torch.inference_mode(), autocast_ctx:
            state = predictor.init_state(
                video_path=jpeg_dir, offload_video_to_cpu=True)

            for seq_i, part_name, bbox, points, labels in all_prompts:
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=seq_i,
                    obj_id=PART_IDS[part_name], box=bbox,
                    points=points, labels=labels)

            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                masks = (mask_logits > 0.0).cpu().numpy()
                parts = {}
                for i, oid in enumerate(obj_ids):
                    pn = id_to_part.get(oid)
                    if pn is None:
                        continue
                    m = masks[i, 0].astype(np.uint8) * 255
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
                    parts[pn] = m
                frame_part_masks[frame_idx] = parts

        predictor.reset_state(state)
        del state
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        # ──────────────────────────────────────
        # Phase C: Mask postprocess + inpaint
        #   Writes: 04_sam2_mask, 05_sam2_postproc, 06_sam2_overlay, 07_inpaint
        # ──────────────────────────────────────
        mask_png_dir = os.path.join(tmp_dir, "masks")
        os.makedirs(mask_png_dir, exist_ok=True)

        w04 = open_video_writer(out04_sam2, w, h, fps)
        w05 = open_video_writer(out05_postproc, w, h, fps)
        w06 = open_video_writer(out06_overlay, w, h, fps)

        for i in range(processed):
            img = frames_bgr[i]
            parts = frame_part_masks.get(i, {})

            # 04_sam2_mask: colored per-part raw masks
            sam2_mask_img = np.zeros_like(img)
            sam2_overlay_img = img.copy()
            combined_binary = np.zeros((h, w), dtype=np.uint8)
            for pn, m in parts.items():
                color = PART_COLORS.get(pn, (128, 128, 128))
                roi = m > 128
                sam2_mask_img[roi] = color
                sam2_overlay_img[roi] = (
                    0.6 * sam2_overlay_img[roi] + 0.4 * np.array(color)
                ).astype(np.uint8)
                contours, _ = cv2.findContours(
                    m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(sam2_overlay_img, contours, -1, color, 1, cv2.LINE_AA)
                combined_binary = np.maximum(combined_binary, m)

            # 05: postprocessed binary mask
            final_mask = postprocess_mask(combined_binary)
            mask_binary = (final_mask > 128).astype(np.uint8) * 255

            write_frame(*w04, sam2_mask_img)
            write_frame(*w05, cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR))
            write_frame(*w06, sam2_overlay_img)

            cv2.imwrite(os.path.join(mask_png_dir, f"{i:05d}.png"), mask_binary)

        close_video(*w04)
        close_video(*w05)
        close_video(*w06)
        del frame_part_masks

        # 07: inpaint
        inpaint_method = models["inpaint_method"]

        if inpaint_method == "propainter":
            propainter_dir = PROPAINTER_ROOT
            pp_out_dir = os.path.join(tmp_dir, "pp_out")

            cmd = [
                sys.executable,
                os.path.join(propainter_dir, "inference_propainter.py"),
                "--video", os.path.abspath(jpeg_dir),
                "--mask", os.path.abspath(mask_png_dir),
                "--output", os.path.abspath(pp_out_dir),
                "--save_fps", str(fps),
                "--mask_dilation", "4",
                "--subvideo_length", "80",
                "--fp16", "--save_frames",
            ]
            sub_env = os.environ.copy()
            if "CUDA_VISIBLE_DEVICES" not in sub_env and device.startswith("cuda:"):
                sub_env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
            result = subprocess.run(cmd, cwd=propainter_dir, env=sub_env,
                                    capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  [{seg_id}] ProPainter FAILED: {result.stderr[-200:]}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return False

            pp_frames_dir = None
            for root, dirs, files in os.walk(pp_out_dir):
                pngs = [f for f in files if f.endswith('.png')]
                if pngs:
                    pp_frames_dir = root
                    break
            if pp_frames_dir is None:
                print(f"  [{seg_id}] ProPainter no output frames")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return False

            inpaint_c, inpaint_s = open_video_writer(out07_inpaint, w, h, fps)
            inpaint_frames = []
            for i in range(processed):
                fpath = os.path.join(pp_frames_dir, f"{i:04d}.png")
                if not os.path.exists(fpath):
                    break
                frame = cv2.imread(fpath)
                if frame is None:
                    break
                write_frame(inpaint_c, inpaint_s, frame)
                inpaint_frames.append(frame)
            close_video(inpaint_c, inpaint_s)
        else:  # lama
            lama = models["lama"]
            inpaint_c, inpaint_s = open_video_writer(out07_inpaint, w, h, fps)
            inpaint_frames = []
            for i in range(processed):
                img = frames_bgr[i]
                mask = cv2.imread(os.path.join(mask_png_dir, f"{i:05d}.png"),
                                  cv2.IMREAD_GRAYSCALE)
                inpainted = run_lama(lama, img, mask)
                write_frame(inpaint_c, inpaint_s, inpainted)
                inpaint_frames.append(inpainted)
            close_video(inpaint_c, inpaint_s)

        del frames_bgr

    # ──────────────────────────────────────
    # Phase D: SMPLH human overlay
    # ──────────────────────────────────────
    if not args.skip_human and "smplh" in models:
        smplh = models["smplh"]
        J_shaped = models["J_shaped"]
        v_shaped = models["v_shaped"]
        faces_nohead = models["faces_nohead"]
        rest_transforms = models["rest_transforms"]

        human_c, human_s = open_video_writer(human_out, w, h, fps)

        for seq_idx_f in range(min(processed, len(inpaint_frames))):
            bg = inpaint_frames[seq_idx_f]
            row = seg_df.iloc[seq_idx_f]
            rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
            hs = np.array(row["observation.state.hand_state"], dtype=np.float64)

            # FK
            q = build_q(model_pin, rq, hs, hand_type=hand_type)
            transforms = do_fk(model_pin, data_pin, q)
            targets = extract_g1_targets(transforms)

            hand_L_np, hand_R_np = apply_finger_curl_from_g1(
                hs, hand_type=hand_type)
            hand_L_t = torch.tensor(
                hand_L_np, dtype=torch.float64, device=device)
            hand_R_t = torch.tensor(
                hand_R_np, dtype=torch.float64, device=device)

            # Retarget
            root_trans_np, root_orient_np, body_pose_np = retarget_frame(
                transforms, rest_transforms, smplh, J_shaped,
                wrist_rot_deg=(0, 0, 0))
            root_trans_np = root_trans_np + R_SMPLH_TO_G1_NP.T @ DEFAULT_ROOT_OFFSET_G1

            root_t = torch.tensor(root_trans_np, dtype=torch.float64, device=device)
            root_o = torch.tensor(root_orient_np, dtype=torch.float64, device=device)
            body_p = torch.tensor(body_pose_np, dtype=torch.float64, device=device)

            # Foot-plant Z correction
            with torch.no_grad():
                positions, rotations = smplh.forward_kinematics(
                    root_t, root_o, body_p, J_shaped,
                    left_hand_pose=hand_L_t, right_hand_pose=hand_R_t)
                ee_tmp = smplh.end_effector_positions(positions, rotations)
            g1_toe_mid = 0.5 * (targets['L_toe_pos'] + targets['R_toe_pos'])
            smplh_toe_mid = 0.5 * (ee_tmp['L_toe_pos'].cpu().numpy()
                                   + ee_tmp['R_toe_pos'].cpu().numpy())
            shift_g1 = np.array([0.0, 0.0, g1_toe_mid[2] - smplh_toe_mid[2]])
            shift_smplh = R_SMPLH_TO_G1_NP.T @ shift_g1
            root_trans_np = root_trans_np + shift_smplh
            root_t = torch.tensor(root_trans_np, dtype=torch.float64, device=device)

            # IK refine arms
            body_pose_np = refine_arms(
                smplh, J_shaped, targets,
                root_trans_np, root_orient_np, body_pose_np,
                device=device, w_drift=10.0,
                hand_L=hand_L_np, hand_R=hand_R_np)
            body_p = torch.tensor(body_pose_np, dtype=torch.float64, device=device)

            # LBS + render
            with torch.no_grad():
                v_g1 = smplh.lbs_to_g1(
                    root_t, root_o, body_p, J_shaped, v_shaped,
                    left_hand_pose=hand_L_t, right_hand_pose=hand_R_t)

            frame_out = render_mesh_on_image(
                bg, v_g1, faces_nohead, transforms, cam_params,
                cam_const=cam_const)
            write_frame(human_c, human_s, frame_out)

        close_video(human_c, human_s)

    del inpaint_frames

    # Cleanup tmp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.time() - t_seg_start
    print(f"  [{seg_id}] done in {elapsed:.1f}s ({processed} frames)")
    return True


# ── Segment collection ──

def collect_segments(manifest_path, tasks_filter, episodes_filter, offset, limit):
    """Read global manifest and collect segment list."""
    segment_root = os.path.dirname(manifest_path)

    with open(manifest_path) as f:
        global_manifest = json.load(f)

    all_segments = []
    for task_short, task_info in global_manifest["tasks"].items():
        if tasks_filter and task_short not in tasks_filter and \
           task_info["full_name"] not in tasks_filter:
            continue

        task_manifest_path = os.path.join(segment_root, task_short, "manifest.json")
        if not os.path.isfile(task_manifest_path):
            print(f"WARN: no manifest for {task_short}")
            continue

        with open(task_manifest_path) as f:
            task_manifest = json.load(f)

        for seg in task_manifest["segments"]:
            if episodes_filter is not None and seg["episode"] not in episodes_filter:
                continue
            all_segments.append((seg, task_manifest))

    # Apply offset + limit
    if offset:
        all_segments = all_segments[offset:]
    if limit:
        all_segments = all_segments[:limit]

    return all_segments, segment_root


# ── CLI ──

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch inpaint + human overlay for segments")
    p.add_argument("--manifest", type=str,
                   default=os.path.join(SEGMENT_DIR, "manifest.json"))
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Filter tasks (default: all)")
    p.add_argument("--episodes", nargs="+", type=int, default=None,
                   help="Filter episode indices (default: all)")
    p.add_argument("--offset", type=int, default=0,
                   help="Skip first N segments (for multi-GPU splits)")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only N segments after offset (0=all)")
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--inpaint-method", default="propainter",
                   choices=["lama", "propainter"])
    p.add_argument("--sam2-model", default="small",
                   choices=["tiny", "small", "base", "large"])
    p.add_argument("--prompt-interval", type=int, default=30)
    p.add_argument("--bbox-margin", type=int, default=0)
    p.add_argument("--point-prompts", action="store_true",
                   help="Add mesh-mask centroid positive point and outside negative points to each SAM2 box prompt")
    p.add_argument("--negative-point-margin", type=int, default=16,
                   help="Pixel distance from bbox edge for SAM2 negative point prompts")
    p.add_argument("--min-visible-area", type=int, default=50)
    p.add_argument("--intermediate-root", type=str,
                   default=DEFAULT_INTERMEDIATE_ROOT,
                   help="Output root for stages 1-7 (default: output/segment_pipeline)")
    p.add_argument("--final-root", type=str, default=DEFAULT_FINAL_ROOT,
                   help="Output root for stage 8 human render (default: training_data/overlay/4s)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-human", action="store_true",
                   help="Only run inpaint, skip human overlay")
    return p.parse_args()


def main():
    args = parse_args()

    print("Segment Pipeline: Inpaint + Human Overlay")
    print(f"  Manifest: {args.manifest}")
    print(f"  Device: {args.device}")
    print(f"  Inpaint: {args.inpaint_method}")
    print(f"  SAM2: {args.sam2_model}")
    print(f"  Intermediate: {args.intermediate_root}")
    print(f"  Final: {args.final_root}")
    if args.offset:
        print(f"  Offset: {args.offset}")
    if args.limit:
        print(f"  Limit: {args.limit} segments")
    if args.skip_human:
        print(f"  Skip human overlay: yes")

    # Collect segments
    episodes_filter = set(args.episodes) if args.episodes else None
    segments, segment_root = collect_segments(
        args.manifest, args.tasks, episodes_filter, args.offset, args.limit)
    print(f"\n{len(segments)} segments to process\n")
    args.segment_root = segment_root

    if not segments:
        print("No segments found.")
        return

    # Load models
    models = load_models(
        args.device, args.sam2_model, args.inpaint_method, args.skip_human)

    # Track current hand_type for cache rebuilding
    current_hand_type = None
    os.makedirs(args.intermediate_root, exist_ok=True)
    os.makedirs(args.final_root, exist_ok=True)

    t_total = time.time()
    n_done = 0
    n_fail = 0

    for i, (seg_info, task_manifest) in enumerate(segments):
        hand_type = task_manifest["hand_type"]

        # Rebuild mesh caches if hand_type changed
        if hand_type != current_hand_type:
            print(f"Building mesh caches for hand_type={hand_type}...")
            mesh_cache, part_caches, skip_set = build_caches(
                models["link_meshes"], hand_type)
            models["_caches"] = (mesh_cache, part_caches, skip_set)
            current_hand_type = hand_type

        seg_id = (f"{task_manifest['task_short']}/ep{seg_info['episode']:03d}"
                  f"/seg{seg_info['segment_index']:02d}")
        print(f"[{i+1}/{len(segments)}] {seg_id}")

        ok = process_segment(seg_info, task_manifest, models, args)
        if ok:
            n_done += 1
        else:
            n_fail += 1

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Done: {n_done} succeeded, {n_fail} failed, {elapsed:.1f}s total")
    if n_done > 0:
        print(f"  Avg: {elapsed/n_done:.1f}s per segment")
    print(f"Intermediate: {args.intermediate_root}")
    print(f"Final: {args.final_root}")


if __name__ == "__main__":
    main()
