"""Grid-search SMPLH overlay parameters against G1 mesh projection masks."""

import argparse
import csv
import itertools
import os
from dataclasses import dataclass

import av
import cv2
import numpy as np
import pandas as pd
import pinocchio as pin
import torch

from src.core.camera import make_camera_const
from src.core.config import (
    BEST_PARAMS, G1_URDF, MAIN_ROOT, MESH_DIR, OUTPUT_DIR, get_skip_meshes,
)
from src.core.fk import build_q, do_fk, parse_urdf_meshes, preload_meshes
from src.core.render import render_mask, render_mask_and_overlay, render_smplh_mask, render_mesh_on_image
from src.core.retarget import (
    apply_finger_curl_from_g1, build_default_hand_pose,
    compute_g1_rest_transforms, refine_arms, retarget_frame, scale_hands,
)
from src.core.smplh import R_SMPLH_TO_G1_NP, SMPLHForIK, extract_g1_targets
from src.pipeline.segment_pipeline import BODY_PARTS, match_links, render_mask_for_links


@dataclass(frozen=True)
class FitParams:
    body_scale: float
    hand_scale: float
    root_x: float
    root_z: float


def parse_csv_floats(value):
    return [float(x) for x in value.split(",") if x.strip()]


def parse_csv_ints(value):
    return [int(x) for x in value.split(",") if x.strip()]


def read_frame(video_path, frame_idx):
    container = av.open(video_path)
    stream = container.streams.video[0]
    try:
        for idx, frame in enumerate(container.decode(stream)):
            if idx == frame_idx:
                return frame.to_ndarray(format="bgr24")
    finally:
        container.close()
    raise IndexError(f"frame_idx out of range: {frame_idx}")


def mask_metrics(pred, target):
    pred_b = pred > 0
    target_b = target > 0
    inter = np.logical_and(pred_b, target_b).sum()
    union = np.logical_or(pred_b, target_b).sum()
    pred_area = pred_b.sum()
    target_area = target_b.sum()
    return {
        "iou": float(inter / union) if union else 0.0,
        "recall": float(inter / target_area) if target_area else 0.0,
        "precision": float(inter / pred_area) if pred_area else 0.0,
        "pred_area": int(pred_area),
        "target_area": int(target_area),
    }


def combined_score(metrics):
    return 0.55 * metrics["body_iou"] + 0.20 * metrics["hand_iou"] + 0.15 * metrics["body_recall"] + 0.10 * metrics["hand_recall"]


def hand_faces_for_side(smplh, side):
    hand_joints = list(range(22, 37)) if side == "left" else list(range(37, 52))
    weights = smplh.weights.detach().cpu().numpy()
    face_weight = weights[:, hand_joints].sum(axis=1)[smplh.faces].max(axis=1)
    return smplh.faces[face_weight > 0.25]


def no_head_faces(smplh):
    weights = smplh.weights.detach().cpu().numpy()
    face_head_w = weights[:, [12, 15]].sum(axis=1)[smplh.faces].max(axis=1)
    return smplh.faces[face_head_w < 0.3]


def render_variant(smplh, J_shaped, v_shaped, faces, hand_faces, rest_transforms,
                   transforms, targets, hand_L_np, hand_R_np, params, cam_const,
                   image_shape, device, fit_params):
    hand_L_t = torch.tensor(hand_L_np, dtype=torch.float64, device=device)
    hand_R_t = torch.tensor(hand_R_np, dtype=torch.float64, device=device)
    root_trans_np, root_orient_np, body_pose_np = retarget_frame(
        transforms, rest_transforms, smplh, J_shaped, wrist_rot_deg=(0, 0, 0))

    root_t = torch.tensor(root_trans_np, dtype=torch.float64, device=device)
    root_o = torch.tensor(root_orient_np, dtype=torch.float64, device=device)
    body_p = torch.tensor(body_pose_np, dtype=torch.float64, device=device)

    with torch.no_grad():
        positions, rotations = smplh.forward_kinematics(
            root_t, root_o, body_p, J_shaped,
            left_hand_pose=hand_L_t, right_hand_pose=hand_R_t)
        ee_tmp = smplh.end_effector_positions(positions, rotations)
    g1_toe_mid = 0.5 * (targets["L_toe_pos"] + targets["R_toe_pos"])
    smplh_toe_mid = 0.5 * (
        ee_tmp["L_toe_pos"].cpu().numpy() + ee_tmp["R_toe_pos"].cpu().numpy())
    shift_g1 = np.array([
        fit_params.root_x,
        0.0,
        fit_params.root_z + g1_toe_mid[2] - smplh_toe_mid[2],
    ])
    root_trans_np = root_trans_np + R_SMPLH_TO_G1_NP.T @ shift_g1
    root_t = torch.tensor(root_trans_np, dtype=torch.float64, device=device)

    body_pose_np = refine_arms(
        smplh, J_shaped, targets,
        root_trans_np, root_orient_np, body_pose_np,
        device=device, w_drift=10.0,
        hand_L=hand_L_np, hand_R=hand_R_np)
    body_p = torch.tensor(body_pose_np, dtype=torch.float64, device=device)

    with torch.no_grad():
        v_g1 = smplh.lbs_to_g1(
            root_t, root_o, body_p, J_shaped, v_shaped,
            left_hand_pose=hand_L_t, right_hand_pose=hand_R_t)

    body_mask = render_smplh_mask(image_shape, v_g1, faces, transforms, params, cam_const)
    hand_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for side_faces in hand_faces:
        hand_mask = np.maximum(hand_mask, render_smplh_mask(
            image_shape, v_g1, side_faces, transforms, params, cam_const))
    return v_g1, body_mask, hand_mask


def load_frame_items(task, episode, segment, frames, model_pin, data_pin, mesh_cache,
                     hand_caches, params, cam_const):
    seg_root = os.path.join(MAIN_ROOT, "training_data", "segment", task)
    video_path = os.path.join(seg_root, f"ep{episode:03d}", f"seg{segment:02d}_video.mp4")
    joints_path = os.path.join(seg_root, f"ep{episode:03d}", f"seg{segment:02d}_joints.parquet")
    df = pd.read_parquet(joints_path)
    items = []
    for frame_idx in frames:
        img = read_frame(video_path, frame_idx)
        h, w = img.shape[:2]
        row = df.iloc[frame_idx]
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
        q = build_q(model_pin, rq, hs, hand_type="inspire")
        transforms = do_fk(model_pin, data_pin, q)
        target_body = render_mask(mesh_cache, transforms, params, h, w, cam_const)
        target_hand = np.zeros((h, w), dtype=np.uint8)
        for cache in hand_caches.values():
            target_hand = np.maximum(target_hand, render_mask_for_links(
                cache, transforms, params, h, w))
        items.append({
            "frame_idx": frame_idx,
            "img": img,
            "hs": hs,
            "transforms": transforms,
            "targets": extract_g1_targets(transforms),
            "target_body": target_body,
            "target_hand": target_hand,
        })
    return items


def evaluate_params(smplh, rest_transforms, faces, hand_faces, frame_items,
                    params, cam_const, device, fit_params, dynamic_hands):
    J_shaped, v_shaped = smplh.shape_blend(None, body_scale=fit_params.body_scale)
    J_shaped, v_shaped = scale_hands(smplh, J_shaped, v_shaped, fit_params.hand_scale)
    sums = {"body_iou": 0.0, "body_recall": 0.0, "body_precision": 0.0,
            "hand_iou": 0.0, "hand_recall": 0.0, "hand_precision": 0.0}
    for item in frame_items:
        if dynamic_hands:
            hand_L_np, hand_R_np = apply_finger_curl_from_g1(item["hs"], hand_type="inspire")
        else:
            hand_L_np, hand_R_np = build_default_hand_pose()
        _, body_mask, hand_mask = render_variant(
            smplh, J_shaped, v_shaped, faces, hand_faces, rest_transforms,
            item["transforms"], item["targets"], hand_L_np, hand_R_np,
            params, cam_const, item["img"].shape, device, fit_params)
        body = mask_metrics(body_mask, item["target_body"])
        hand = mask_metrics(hand_mask, item["target_hand"])
        sums["body_iou"] += body["iou"]
        sums["body_recall"] += body["recall"]
        sums["body_precision"] += body["precision"]
        sums["hand_iou"] += hand["iou"]
        sums["hand_recall"] += hand["recall"]
        sums["hand_precision"] += hand["precision"]
    count = len(frame_items)
    avg = {k: v / count for k, v in sums.items()}
    avg["score"] = combined_score(avg)
    return avg


def overlay_mask(img, mask, color):
    out = img.copy()
    roi = mask > 0
    out[roi] = (0.55 * out[roi] + 0.45 * np.array(color)).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 1, cv2.LINE_AA)
    return out


def put_label(img, label):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_debug_images(smplh, rest_transforms, faces, hand_faces, frame_items,
                       mesh_cache, params, cam_const, device, baseline, best,
                       output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for item in frame_items:
        panels = []
        _, g1_overlay = render_mask_and_overlay(
            item["img"], mesh_cache, item["transforms"], params,
            item["img"].shape[0], item["img"].shape[1], cam_const)
        panels.append(put_label(g1_overlay, f"G1 target f{item['frame_idx']}"))
        for label, fit_params in [("baseline", baseline), ("best", best)]:
            J_shaped, v_shaped = smplh.shape_blend(None, body_scale=fit_params.body_scale)
            J_shaped, v_shaped = scale_hands(smplh, J_shaped, v_shaped, fit_params.hand_scale)
            hand_L_np, hand_R_np = apply_finger_curl_from_g1(item["hs"], hand_type="inspire")
            v_g1, body_mask, hand_mask = render_variant(
                smplh, J_shaped, v_shaped, faces, hand_faces, rest_transforms,
                item["transforms"], item["targets"], hand_L_np, hand_R_np,
                params, cam_const, item["img"].shape, device, fit_params)
            render = render_mesh_on_image(
                item["img"], v_g1, faces, item["transforms"], params,
                cam_const=cam_const)
            mask_panel = overlay_mask(item["img"], item["target_body"], (0, 255, 255))
            mask_panel = overlay_mask(mask_panel, body_mask, (255, 0, 255))
            panels.append(put_label(render, f"SMPLH {label}"))
            panels.append(put_label(mask_panel, f"mask {label}: target=yellow human=magenta"))
        sheet = np.hstack([cv2.resize(p, (320, 240)) for p in panels])
        path = os.path.join(output_dir, f"fit_f{item['frame_idx']:03d}.png")
        cv2.imwrite(path, sheet)
        print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Inspire_Put_Clothes_Into_Basket")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--segment", type=int, default=1)
    parser.add_argument("--frames", default="0,30,60,90")
    parser.add_argument("--body-scales", default="0.66,0.70,0.74,0.78,0.82")
    parser.add_argument("--hand-scales", default="1.2,1.4,1.6,1.8")
    parser.add_argument("--root-x", default="-0.04,-0.02,0.0,0.02,0.04")
    parser.add_argument("--root-z", default="-0.03,0.0,0.03")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=os.path.join(OUTPUT_DIR, "body_fit_search"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    params = BEST_PARAMS
    cam_const = make_camera_const(params)
    model_pin = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model_pin.createData()
    link_meshes = parse_urdf_meshes(G1_URDF)
    skip_set = get_skip_meshes("inspire")
    mesh_cache = preload_meshes(link_meshes, MESH_DIR, skip_set=skip_set)
    hand_caches = {
        "left": match_links(mesh_cache, BODY_PARTS["left_hand"], skip_set),
        "right": match_links(mesh_cache, BODY_PARTS["right_hand"], skip_set),
    }
    frames = parse_csv_ints(args.frames)
    frame_items = load_frame_items(
        args.task, args.episode, args.segment, frames, model_pin, data_pin,
        mesh_cache, hand_caches, params, cam_const)

    smplh = SMPLHForIK(device=args.device)
    faces = no_head_faces(smplh)
    hand_faces = [hand_faces_for_side(smplh, "left"), hand_faces_for_side(smplh, "right")]
    rest_transforms = compute_g1_rest_transforms()

    body_scales = parse_csv_floats(args.body_scales)
    hand_scales = parse_csv_floats(args.hand_scales)
    root_xs = parse_csv_floats(args.root_x)
    root_zs = parse_csv_floats(args.root_z)
    rows = []
    total = len(body_scales) * len(hand_scales) * len(root_xs) * len(root_zs)
    print(f"searching {total} candidates on {len(frame_items)} frames")
    for idx, values in enumerate(itertools.product(body_scales, hand_scales, root_xs, root_zs), 1):
        fit_params = FitParams(*values)
        metrics = evaluate_params(
            smplh, rest_transforms, faces, hand_faces, frame_items,
            params, cam_const, args.device, fit_params, dynamic_hands=True)
        row = {"idx": idx, **fit_params.__dict__, **metrics}
        rows.append(row)
        if idx % 20 == 0 or idx == total:
            best_so_far = max(rows, key=lambda r: r["score"])
            print(f"{idx}/{total} best score={best_so_far['score']:.4f} params="
                  f"body={best_so_far['body_scale']} hand={best_so_far['hand_scale']} "
                  f"x={best_so_far['root_x']} z={best_so_far['root_z']}")

    rows.sort(key=lambda r: r["score"], reverse=True)
    csv_path = os.path.join(args.output_dir, "body_fit_search.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)
    for row in rows[:10]:
        print(row)

    baseline = FitParams(body_scale=0.75, hand_scale=1.3, root_x=0.0, root_z=0.0)
    best = FitParams(
        body_scale=rows[0]["body_scale"], hand_scale=rows[0]["hand_scale"],
        root_x=rows[0]["root_x"], root_z=rows[0]["root_z"])
    write_debug_images(
        smplh, rest_transforms, faces, hand_faces, frame_items, mesh_cache,
        params, cam_const, args.device, baseline, best, args.output_dir)


if __name__ == "__main__":
    main()
