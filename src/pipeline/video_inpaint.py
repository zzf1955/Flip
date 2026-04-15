"""Per-frame video inpainting pipeline:
  1. Decode video sequentially for one episode
  2. FK -> mesh mask -> GrabCut -> smooth mask per frame
  3. Render overlay per frame
  4. LaMa inpaint per frame
  5. Encode 4 output videos: original, mask, overlay, inpaint

Usage:
  python -m src.pipeline.video_inpaint --episode 0
  python -m src.pipeline.video_inpaint --episode 4 --start 5 --duration 5
"""

import sys
import os
import argparse
import time
import numpy as np
import pinocchio as pin

sys.stdout.reconfigure(line_buffering=True)

from src.core.config import (
    G1_URDF, MESH_DIR, BEST_PARAMS, OUTPUT_DIR,
    get_hand_type, get_skip_meshes,
)
from src.core.data import load_episode_info, open_video_writer, write_frame, close_video
from src.core.fk import build_q, do_fk, parse_urdf_meshes, preload_meshes
from src.core.render import render_mask, render_overlay
from src.core.mask import postprocess_mask, grabcut_refine, init_lama, run_lama


def main():
    import av
    import cv2

    parser = argparse.ArgumentParser(description="Per-frame video inpainting")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start", type=float, default=0,
                        help="Start offset in seconds within episode (default 0)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Duration in seconds (default 0 = entire episode)")
    args = parser.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, "inpaint/per_frame_lama")
    os.makedirs(out_dir, exist_ok=True)
    ep = args.episode

    # Load episode info
    print(f"Episode: {ep}")
    video_path, from_ts, to_ts, ep_df = load_episode_info(ep)
    n_total = len(ep_df)
    fps = 30
    ep_duration = n_total / fps
    print(f"Episode length: {n_total} frames ({ep_duration:.1f}s)")
    print(f"Video file: {os.path.basename(video_path)}, ts={from_ts:.1f}-{to_ts:.1f}")

    # Apply start/duration trim
    start_frame = int(args.start * fps) if args.start > 0 else 0
    if args.duration > 0:
        end_frame = min(start_frame + int(args.duration * fps), n_total)
    else:
        end_frame = n_total

    ep_df = ep_df.iloc[start_frame:end_frame]
    n_frames = len(ep_df)
    print(f"Processing frames {start_frame}-{end_frame - 1} ({n_frames} frames, {n_frames/fps:.1f}s)")

    # Build frame data lookup
    frame_data = {}
    for _, row in ep_df.iterrows():
        fi = int(row["frame_index"])
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
        frame_data[fi] = (rq, hs)

    vid_from_frame = int(round((from_ts + start_frame / fps) * fps))
    vid_to_frame = int(round((from_ts + (end_frame - 1) / fps) * fps))

    # Load URDF + meshes
    hand_type = get_hand_type()
    skip_set = get_skip_meshes(hand_type)
    print(f"Loading URDF and meshes... (hand_type={hand_type})")
    model = pin.buildModelFromUrdf(G1_URDF, pin.JointModelFreeFlyer())
    data_pin = model.createData()
    link_meshes = parse_urdf_meshes(G1_URDF)
    mesh_cache = preload_meshes(link_meshes, MESH_DIR, skip_set=skip_set)
    print(f"Loaded {len(mesh_cache)} link meshes")

    # Init LaMa
    print("Loading LaMa model...")
    lama = init_lama()

    # Open input video
    container_in = av.open(video_path)
    stream_in = container_in.streams.video[0]
    vid_fps = float(stream_in.average_rate)

    seek_ts = from_ts + start_frame / fps
    if seek_ts > 1.0:
        target_pts = int((seek_ts - 1.0) / stream_in.time_base)
        container_in.seek(max(target_pts, 0), stream=stream_in)

    # Open output videos
    tag = f"ep{ep:03d}"
    if args.start > 0 or args.duration > 0:
        tag += f"_{start_frame}-{end_frame}"

    writers_ready = False
    writers = {}

    processed = 0
    prev_inpainted = None
    ema_alpha = 0.6
    t_start = time.time()

    print(f"Processing {n_frames} frames...")
    for av_frame in container_in.decode(stream_in):
        pts_sec = float(av_frame.pts * stream_in.time_base)
        vid_fn = int(round(pts_sec * vid_fps))
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
            for vname in ["original", "mask", "overlay", "inpaint"]:
                vpath = os.path.join(out_dir, f"{tag}_{vname}.mp4")
                c, s = open_video_writer(vpath, w, h, fps=int(vid_fps))
                writers[vname] = (c, s)
            writers_ready = True

        rq, hs = frame_data[ep_fi]

        q = build_q(model, rq, hs, hand_type=get_hand_type())
        transforms = do_fk(model, data_pin, q)

        raw_mask = render_mask(mesh_cache, transforms, BEST_PARAMS, h, w)
        gc_mask = grabcut_refine(img, raw_mask)
        final_mask = postprocess_mask(gc_mask)

        overlay_img = render_overlay(img, mesh_cache, transforms, BEST_PARAMS)

        inpainted = run_lama(lama, img, final_mask)

        # Temporal blending
        if prev_inpainted is not None:
            mask_region = final_mask > 128
            blended = inpainted.copy()
            blended[mask_region] = (
                ema_alpha * inpainted[mask_region].astype(np.float32) +
                (1 - ema_alpha) * prev_inpainted[mask_region].astype(np.float32)
            ).astype(np.uint8)
            inpainted = blended
        prev_inpainted = inpainted.copy()

        write_frame(*writers["original"], img)
        write_frame(*writers["mask"], cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))
        write_frame(*writers["overlay"], overlay_img)
        write_frame(*writers["inpaint"], inpainted)

        processed += 1
        if processed % 50 == 0 or processed == 1:
            elapsed = time.time() - t_start
            fps_proc = processed / elapsed
            eta = (n_frames - processed) / fps_proc if fps_proc > 0 else 0
            print(f"  {processed}/{n_frames} frames "
                  f"({fps_proc:.1f} fps, ETA {eta:.0f}s)")

    container_in.close()
    for vname, (c, s) in writers.items():
        close_video(c, s)

    elapsed = time.time() - t_start
    print(f"\nDone: {processed} frames in {elapsed:.1f}s "
          f"({processed/elapsed:.1f} fps)")
    print(f"Output: {out_dir}/{tag}_*.mp4")


if __name__ == "__main__":
    main()
