"""Data loading and video I/O utilities.

- Episode metadata & parquet loading (LeRobot format)
- PyAV video writer helpers
- Keypoint detection from alpha-channel annotations
"""

import os
import numpy as np
import pandas as pd
import cv2

from .config import ACTIVE_DATA_DIR, DATASET_ROOT


# ── Episode / parquet loading ──

def load_episode_info(ep, data_dir=None):
    """Load episode meta and return (video_path, from_ts, to_ts, ep_df).

    Args:
        ep: episode index
        data_dir: dataset directory (defaults to ACTIVE_DATA_DIR)

    Returns:
        video_path: path to MP4 file
        from_ts: start timestamp (seconds)
        to_ts: end timestamp (seconds)
        ep_df: DataFrame with frame_index, robot_q_current, hand_state
    """
    if data_dir is None:
        data_dir = ACTIVE_DATA_DIR
    meta = pd.read_parquet(os.path.join(data_dir, "meta", "episodes",
                                         "chunk-000", "file-000.parquet"))
    ep_meta = meta[meta["episode_index"] == ep]
    if len(ep_meta) == 0:
        raise ValueError(f"Episode {ep} not found in meta")
    ep_meta = ep_meta.iloc[0]

    # Auto-detect primary head camera key (varies by dataset variant):
    #   - regular tasks:   observation.images.head_stereo_left
    #   - MainCamOnly:     observation.images.cam_0
    _CAM_KEY_CANDIDATES = ["head_stereo_left", "cam_0"]
    cam_key = None
    for cand in _CAM_KEY_CANDIDATES:
        if f"videos/observation.images.{cand}/file_index" in ep_meta.index:
            cam_key = cand
            break
    if cam_key is None:
        raise ValueError(
            f"No known head camera key found in meta for {data_dir}. "
            f"Tried: {_CAM_KEY_CANDIDATES}")

    file_idx = int(ep_meta[f"videos/observation.images.{cam_key}/file_index"])
    from_ts = float(ep_meta[f"videos/observation.images.{cam_key}/from_timestamp"])
    to_ts = float(ep_meta[f"videos/observation.images.{cam_key}/to_timestamp"])

    video_path = os.path.join(data_dir, "videos",
                               f"observation.images.{cam_key}",
                               "chunk-000", f"file-{file_idx:03d}.mp4")

    # Load parquet (determine which file)
    data_fi = int(ep_meta.get("data/file_index", 0))
    parquet_path = os.path.join(data_dir, "data", "chunk-000",
                                 f"file-{data_fi:03d}.parquet")
    df = pd.read_parquet(parquet_path)
    ep_df = df[df["episode_index"] == ep].sort_values("frame_index")

    return video_path, from_ts, to_ts, ep_df


def load_all_episode_meta(task_name, dataset_root=None):
    """Load full episode metadata for a task.

    Returns:
        meta_df: DataFrame with all episodes
        cam_key: detected camera key (e.g. 'head_stereo_left' or 'cam_0')
    """
    if dataset_root is None:
        dataset_root = DATASET_ROOT
    data_dir = os.path.join(dataset_root, task_name)
    meta = pd.read_parquet(os.path.join(data_dir, "meta", "episodes",
                                         "chunk-000", "file-000.parquet"))
    _CAM_KEY_CANDIDATES = ["head_stereo_left", "cam_0"]
    cam_key = None
    for cand in _CAM_KEY_CANDIDATES:
        if f"videos/observation.images.{cand}/file_index" in meta.columns:
            cam_key = cand
            break
    if cam_key is None:
        raise ValueError(f"No known head camera key found for {task_name}")
    return meta, cam_key


def load_data_parquet(task_name, file_index, dataset_root=None):
    """Load a data parquet file by file index.

    Returns:
        DataFrame with all rows from that parquet file.
    """
    if dataset_root is None:
        dataset_root = DATASET_ROOT
    data_dir = os.path.join(dataset_root, task_name)
    path = os.path.join(data_dir, "data", "chunk-000",
                        f"file-{file_index:03d}.parquet")
    return pd.read_parquet(path)


def build_frame_data(ep_df):
    """Build frame_index -> (rq, hand_state) lookup from episode DataFrame.

    Returns:
        dict[int, tuple[ndarray, ndarray]]: frame_index -> (rq(36), hs(12))
    """
    frame_data = {}
    for _, row in ep_df.iterrows():
        fi = int(row["frame_index"])
        rq = np.array(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.array(row["observation.state.hand_state"], dtype=np.float64)
        frame_data[fi] = (rq, hs)
    return frame_data


# ── Video I/O ──

def open_video_writer(path, w, h, fps=30):
    """Open H.264 video writer via PyAV."""
    import av
    container = av.open(path, mode='w')
    stream = container.add_stream('libx264', rate=int(fps))
    stream.width = w
    stream.height = h
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '18', 'preset': 'medium'}
    return container, stream


def write_frame(container, stream, img_bgr):
    """Encode a single BGR frame to the video stream."""
    import av
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    frame = av.VideoFrame.from_ndarray(img_rgb, format='rgb24')
    for packet in stream.encode(frame):
        container.mux(packet)


def close_video(container, stream):
    """Flush remaining packets and close the video file."""
    for packet in stream.encode():
        container.mux(packet)
    container.close()


# ── Keypoint detection ──

def detect_keypoints_from_alpha(png_path):
    """Detect keypoint markers from alpha-channel annotations.

    Pixels with alpha != 255 are considered markers. Nearby markers
    (within 15px) are merged into clusters; each cluster centroid is
    returned as a keypoint.

    Returns:
        list of (x, y) tuples, one per detected keypoint cluster.
    """
    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] < 4:
        return []
    alpha = img[:, :, 3]
    mask = (alpha != 255).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []

    # Cluster nearby points (simple greedy merge)
    points = list(zip(xs.tolist(), ys.tolist()))
    clusters = []
    used = [False] * len(points)
    for i, (x, y) in enumerate(points):
        if used[i]:
            continue
        cluster = [(x, y)]
        used[i] = True
        for j in range(i + 1, len(points)):
            if used[j]:
                continue
            dx = points[j][0] - x
            dy = points[j][1] - y
            if dx * dx + dy * dy < 15 * 15:
                cluster.append(points[j])
                used[j] = True
        cx = int(np.mean([p[0] for p in cluster]))
        cy = int(np.mean([p[1] for p in cluster]))
        clusters.append((cx, cy))
    return clusters
