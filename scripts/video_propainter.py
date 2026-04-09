"""
Two-stage video inpainting with ProPainter:
  Stage 1: Generate per-frame masks (FK + GrabCut + postprocess)
  Stage 2: Run ProPainter for temporally consistent inpainting

Usage:
  python scripts/video_propainter.py --episode 4 --start 5 --duration 5
"""

import sys
import os
import argparse
import time
import shutil
import subprocess
import numpy as np
import pandas as pd
import cv2
import pinocchio as pin

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import shared functions from video_inpaint
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))
from video_inpaint import (
    URDF_PATH, MESH_DIR, BEST_PARAMS, SKIP_MESHES,
    build_q, do_fk, parse_urdf_meshes, preload_meshes,
    render_mask, render_overlay, grabcut_refine, postprocess_mask,
    open_video_writer, write_frame, close_video,
)
from video_inpaint import load_episode_info, DATA_DIR

OUTPUT_DIR = os.path.join(BASE_DIR, "test_results", "inpaint_video")
PROPAINTER_DIR = os.path.join(BASE_DIR, "ProPainter")


def main():
    import av

    parser = argparse.ArgumentParser(description="ProPainter video inpainting")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start", type=float, default=0,
                        help="Start offset in seconds within episode")
    parser.add_argument("--duration", type=float, default=0,
                        help="Duration in seconds (0 = entire episode)")
    parser.add_argument("--mask-dilation", type=int, default=4,
                        help="ProPainter mask dilation iterations")
    parser.add_argument("--subvideo-length", type=int, default=80,
                        help="ProPainter sub-video chunk length")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ep = args.episode

    # ====== Stage 1: Generate frames + masks ======
    print(f"=== Stage 1: Generating masks for episode {ep} ===")
    video_path, from_ts, to_ts, ep_df = load_episode_info(ep)
    n_total = len(ep_df)
    fps = 30

    start_frame = int(args.start * fps) if args.start > 0 else 0
    if args.duration > 0:
        end_frame = min(start_frame + int(args.duration * fps), n_total)
    else:
        end_frame = n_total
    ep_df = ep_df.iloc[start_frame:end_frame]
    n_frames = len(ep_df)
    print(f"Episode {ep}: {n_total} total frames, processing {start_frame}-{end_frame-1} ({n_frames} frames)")

    # Build frame data lookup
    frame_data = {}
    for _, row in ep_df.iterrows():
        fi = int(row["frame_index"])
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
        frame_data[fi] = (rq, hs)

    # Load URDF + meshes
    print("Loading URDF and meshes...")
    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    data_pin = model.createData()
    link_meshes = parse_urdf_meshes(URDF_PATH)
    mesh_cache = preload_meshes(link_meshes, MESH_DIR)

    # Temp dirs for ProPainter input
    tag = f"ep{ep:03d}"
    if args.start > 0 or args.duration > 0:
        tag += f"_{start_frame}-{end_frame}"

    tmp_dir = os.path.join(OUTPUT_DIR, f".tmp_{tag}")
    frames_dir = os.path.join(tmp_dir, "frames")
    masks_dir = os.path.join(tmp_dir, "masks")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Skip Stage 1 if temp files already exist
    existing_frames = len([f for f in os.listdir(frames_dir) if f.endswith('.png')]) if os.path.isdir(frames_dir) else 0
    if existing_frames >= n_frames:
        print(f"Stage 1 skipped: {existing_frames} frames already exist in {frames_dir}")
        processed = existing_frames
        # Still need overlay/original/mask videos - check if they exist
        if not all(os.path.exists(p) for p in [overlay_path, original_path, mask_path]):
            print("  (video files missing, will regenerate)")
            existing_frames = 0

    # Open overlay video writer (write directly, not via ProPainter)
    overlay_path = os.path.join(OUTPUT_DIR, f"{tag}_overlay.mp4")
    original_path = os.path.join(OUTPUT_DIR, f"{tag}_original.mp4")
    mask_path = os.path.join(OUTPUT_DIR, f"{tag}_mask.mp4")

    if existing_frames >= n_frames:
        # Jump to Stage 2
        pass
    else:
        _run_stage1(video_path, from_ts, start_frame, end_frame, fps,
                    frame_data, n_frames, model, data_pin, mesh_cache,
                    frames_dir, masks_dir, overlay_path, original_path, mask_path)

    # ====== Stage 2: ProPainter inference ======
    print(f"\n=== Stage 2: ProPainter inference ===")
    _run_stage2(args, tag, tmp_dir, frames_dir, masks_dir, int(vid_fps))
    return


def _run_stage1(video_path, from_ts, start_frame, end_frame, fps,
                frame_data, n_frames, model, data_pin, mesh_cache,
                frames_dir, masks_dir, overlay_path, original_path, mask_path):
    import av

    # Seek to episode start
    container_in = av.open(video_path)
    stream_in = container_in.streams.video[0]
    vid_fps = float(stream_in.average_rate)

    seek_ts = from_ts + start_frame / fps
    if seek_ts > 1.0:
        target_pts = int((seek_ts - 1.0) / stream_in.time_base)
        container_in.seek(max(target_pts, 0), stream=stream_in)

    writers_ready = False
    writers = {}
    processed = 0
    t_start = time.time()

    print(f"Generating {n_frames} frames + masks...")
    for av_frame in container_in.decode(stream_in):
        pts_sec = float(av_frame.pts * stream_in.time_base)
        ep_fi = int(round((pts_sec - from_ts) * fps))

        if ep_fi < start_frame:
            continue
        if ep_fi >= end_frame:
            break
        if ep_fi not in frame_data:
            continue

        img = av_frame.to_ndarray(format='bgr24')

        if not writers_ready:
            h, w = img.shape[:2]
            for vname, vpath in [("overlay", overlay_path),
                                  ("original", original_path),
                                  ("mask", mask_path)]:
                c, s = open_video_writer(vpath, w, h, fps=int(vid_fps))
                writers[vname] = (c, s)
            writers_ready = True

        rq, hs = frame_data[ep_fi]
        q = build_q(model, rq, hs)
        transforms = do_fk(model, data_pin, q)

        # Mask
        raw_mask = render_mask(mesh_cache, transforms, BEST_PARAMS, h, w)
        gc_mask = grabcut_refine(img, raw_mask)
        final_mask = postprocess_mask(gc_mask)

        # Save frame + mask as PNG for ProPainter
        idx_str = f"{processed:05d}.png"
        # ProPainter expects RGB frames, but reads with cv2 (BGR), so save as BGR
        cv2.imwrite(os.path.join(frames_dir, idx_str), img)
        # Mask: binary white on black
        mask_binary = (final_mask > 128).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(masks_dir, idx_str), mask_binary)

        # Overlay
        overlay_img = render_overlay(img, mesh_cache, transforms, BEST_PARAMS)

        # Write videos
        write_frame(*writers["original"], img)
        write_frame(*writers["mask"], cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))
        write_frame(*writers["overlay"], overlay_img)

        processed += 1
        if processed % 50 == 0 or processed == 1:
            elapsed = time.time() - t_start
            fps_proc = processed / elapsed
            eta = (n_frames - processed) / fps_proc if fps_proc > 0 else 0
            print(f"  {processed}/{n_frames} ({fps_proc:.1f} fps, ETA {eta:.0f}s)")

    container_in.close()
    for vname, (c, s) in writers.items():
        close_video(c, s)

    elapsed = time.time() - t_start
    print(f"Stage 1 done: {processed} frames in {elapsed:.1f}s")

    # ====== Stage 2: ProPainter inference ======
    print(f"\n=== Stage 2: ProPainter inference ===")

    pp_output = os.path.join(tmp_dir, "pp_out")
    os.makedirs(pp_output, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(PROPAINTER_DIR, "inference_propainter.py"),
        "--video", frames_dir,
        "--mask", masks_dir,
        "--output", pp_output,
        "--save_fps", str(int(vid_fps)),
        "--mask_dilation", str(args.mask_dilation),
        "--subvideo_length", str(args.subvideo_length),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROPAINTER_DIR, capture_output=False)

    if result.returncode != 0:
        print(f"ProPainter failed with code {result.returncode}")
        print(f"Temp files kept at: {tmp_dir}")
        return

    # Find and copy the output video
    pp_video = os.path.join(pp_output, "frames", "inpaint_out.mp4")
    if not os.path.exists(pp_video):
        # ProPainter outputs to {output}/{video_name}/inpaint_out.mp4
        for root, dirs, files in os.walk(pp_output):
            for f in files:
                if f == "inpaint_out.mp4":
                    pp_video = os.path.join(root, f)
                    break

    inpaint_out = os.path.join(OUTPUT_DIR, f"{tag}_inpaint.mp4")
    if os.path.exists(pp_video):
        shutil.copy2(pp_video, inpaint_out)
        print(f"Inpaint video: {inpaint_out}")
    else:
        print(f"WARNING: ProPainter output not found in {pp_output}")

    # Cleanup temp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.1f}s. Output: {OUTPUT_DIR}/{tag}_*.mp4")


if __name__ == "__main__":
    main()
