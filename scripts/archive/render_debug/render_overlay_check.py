"""
Render mesh overlay on multiple Inspire task videos to check camera param generalization.

Uses PSO-optimized parameters from test_results/kp_pso_all/best_params.json.
Outputs short MP4 clips with FK mesh overlay.

Usage:
  python scripts/render_overlay_check.py
  python scripts/render_overlay_check.py --duration 3 --device cuda:2
"""

import sys
import os
import json
import argparse
import numpy as np
import cv2
import pinocchio as pin

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

import pandas as pd

from config import (G1_URDF, MESH_DIR, SKIP_MESHES, DATASET_ROOT, OUTPUT_DIR,
                     get_hand_type, get_skip_meshes, CAMERA_MODEL, BEST_PARAMS)
from camera_models import get_model
from video_inpaint import (build_q, do_fk, parse_urdf_meshes, preload_meshes,
                            make_camera_const, render_mask_and_overlay)

# ── Tasks & episodes to check ──
CLIPS = [
    {"task": "G1_WBT_Inspire_Pickup_Pillow_MainCamOnly",               "episode": 0,   "start": 0},
    {"task": "G1_WBT_Inspire_Pickup_Pillow_MainCamOnly",               "episode": 100, "start": 0},
    {"task": "G1_WBT_Inspire_Put_Clothes_Into_Basket",                  "episode": 0,   "start": 0},
    {"task": "G1_WBT_Inspire_Put_Clothes_Into_Basket",                  "episode": 50,  "start": 0},
    {"task": "G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly", "episode": 0, "start": 0},
    {"task": "G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly", "episode": 100, "start": 0},
    {"task": "G1_WBT_Inspire_Collect_Clothes_MainCamOnly",              "episode": 0,   "start": 0},
    {"task": "G1_WBT_Inspire_Collect_Clothes_MainCamOnly",              "episode": 50,  "start": 0},
]


def load_episode_info(ep, data_dir):
    """Load episode meta, auto-detect camera stream name."""
    meta = pd.read_parquet(os.path.join(data_dir, "meta", "episodes",
                                         "chunk-000", "file-000.parquet"))
    ep_meta = meta[meta["episode_index"] == ep]
    if len(ep_meta) == 0:
        raise ValueError(f"Episode {ep} not found in meta")
    ep_meta = ep_meta.iloc[0]

    # Auto-detect camera: head_stereo_left or cam_0
    for cam in ("head_stereo_left", "cam_0"):
        key = f"videos/observation.images.{cam}/file_index"
        if key in ep_meta.index:
            cam_name = cam
            break
    else:
        raise ValueError("No known camera stream found in meta")

    prefix = f"videos/observation.images.{cam_name}"
    file_idx = int(ep_meta[f"{prefix}/file_index"])
    from_ts = float(ep_meta[f"{prefix}/from_timestamp"])
    to_ts = float(ep_meta[f"{prefix}/to_timestamp"])

    video_path = os.path.join(data_dir, "videos",
                               f"observation.images.{cam_name}",
                               "chunk-000", f"file-{file_idx:03d}.mp4")

    data_fi = int(ep_meta.get("data/file_index", 0))
    df = pd.read_parquet(os.path.join(data_dir, "data", "chunk-000",
                                       f"file-{data_fi:03d}.parquet"))
    ep_df = df[df["episode_index"] == ep].sort_values("frame_index")
    return video_path, from_ts, to_ts, ep_df


def open_video_writer(path, w, h, fps=30):
    import av
    container = av.open(path, mode='w')
    stream = container.add_stream('libx264', rate=int(fps))
    stream.width = w
    stream.height = h
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '18', 'preset': 'medium'}
    return container, stream


def write_frame(container, stream, img_bgr):
    import av
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    frame = av.VideoFrame.from_ndarray(img_rgb, format='rgb24')
    for packet in stream.encode(frame):
        container.mux(packet)


def close_video(container, stream):
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=5, help="Seconds per clip")
    args = parser.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, "overlay_check")
    os.makedirs(out_dir, exist_ok=True)

    # ── Use config BEST_PARAMS ──
    params = BEST_PARAMS
    print(f"Camera model: {CAMERA_MODEL}")
    print(f"Params from config.py:")
    for k, v in params.items():
        print(f"  {k}: {v:.4f}")

    # ── Load URDF + meshes ──
    print("\nLoading URDF...")
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model.createData()
    link_meshes = parse_urdf_meshes(G1_URDF)

    # Precompute camera constants
    cam_const = make_camera_const(params)

    # ── Process each clip ──
    for clip in CLIPS:
        task = clip["task"]
        ep = clip["episode"]
        start = clip["start"]
        hand_type = get_hand_type(task)
        skip_meshes = get_skip_meshes(hand_type)

        # Load meshes with appropriate skip set
        mesh_cache = preload_meshes(link_meshes, MESH_DIR,
                                    skip_set=skip_meshes, subsample=2)

        tag = f"{task.replace('G1_WBT_Inspire_', '')}_ep{ep}"
        print(f"\n{'='*60}")
        print(f"  {tag}")

        data_dir = os.path.join(DATASET_ROOT, task)
        try:
            video_path, from_ts, to_ts, ep_df = load_episode_info(
                ep, data_dir=data_dir)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        import av
        container_in = av.open(video_path)
        stream_in = container_in.streams.video[0]
        fps = float(stream_in.average_rate)

        n_frames = min(int(args.duration * fps), len(ep_df))
        print(f"  video={video_path}")
        print(f"  fps={fps:.0f}, frames={n_frames}, hand={hand_type}")

        # Seek to episode start
        seek_sec = from_ts + start / fps
        if seek_sec > 1.0:
            target_pts = int((seek_sec - 1.0) / stream_in.time_base)
            container_in.seek(max(target_pts, 0), stream=stream_in)

        # Build frame_index -> row lookup
        fi_to_row = {}
        for _, row in ep_df.iterrows():
            fi_to_row[int(row["frame_index"])] = row

        out_path = os.path.join(out_dir, f"{tag}.mp4")
        writer = None
        frame_count = 0

        for av_frame in container_in.decode(stream_in):
            pts_sec = float(av_frame.pts * stream_in.time_base)
            ep_fi = int(round((pts_sec - from_ts) * fps))

            if ep_fi < start:
                continue
            if ep_fi >= start + n_frames:
                break

            if ep_fi not in fi_to_row:
                continue

            row = fi_to_row[ep_fi]
            img = av_frame.to_ndarray(format='bgr24')
            h, w = img.shape[:2]

            if writer is None:
                writer = open_video_writer(out_path, w, h, fps=fps)

            rq = np.array(row["observation.state.robot_q_current"],
                          dtype=np.float64)
            hs = np.array(row["observation.state.hand_state"],
                          dtype=np.float64)

            q = build_q(model, rq, hs, hand_type=hand_type)
            transforms = do_fk(model, data_pin, q)

            _, overlay = render_mask_and_overlay(
                img, mesh_cache, transforms, params, h, w, cam_const)

            # Add info text
            cv2.putText(overlay, f"{tag}  frame={ep_fi}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1)

            write_frame(*writer, overlay)
            frame_count += 1

            if frame_count % 30 == 0:
                print(f"  frame {frame_count}/{n_frames}")

        container_in.close()
        if writer is not None:
            close_video(*writer)
            print(f"  -> {out_path} ({frame_count} frames)")
        else:
            print(f"  SKIP: no frames decoded")

    print(f"\nDone. Output: {out_dir}/")


if __name__ == "__main__":
    main()
