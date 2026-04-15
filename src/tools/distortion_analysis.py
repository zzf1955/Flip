#!/usr/bin/env python3
"""
Distortion analysis for G1 head stereo camera.

Extracts a frame from video, performs line detection and curvature analysis
to determine if the camera has significant lens distortion (fisheye/wide-angle).
"""

import cv2
import numpy as np
import os
import sys

# ── Paths ──
from src.core.config import OUTPUT_DIR, DATASET_ROOT
DIST_DIR = os.path.join(OUTPUT_DIR, "tmp", "distortion")
os.makedirs(DIST_DIR, exist_ok=True)

# Try head_stereo_left first (Brainco), fallback to cam_0 (Inspire)
VIDEO_CANDIDATES = [
    os.path.join(DATASET_ROOT, "G1_WBT_Brainco_Pickup_Pillow/videos/observation.images.head_stereo_left/chunk-000/file-000.mp4"),
    os.path.join(DATASET_ROOT, "G1_WBT_Inspire_Pickup_Pillow_MainCamOnly/videos/observation.images.cam_0/chunk-000/file-000.mp4"),
]

FRAME_IDX = 100


def extract_frame(video_path, frame_idx):
    """Extract a specific frame from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_path}")
    print(f"  Resolution: {w}x{h}, FPS: {fps:.1f}, Total frames: {total}")

    if frame_idx >= total:
        frame_idx = total // 2
        print(f"  Adjusted frame index to {frame_idx}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx}")

    return frame


def compute_line_curvature(img_gray, lines, segment_length_threshold=50):
    """
    For each detected line segment, measure how much it deviates from
    a perfect straight line by sampling points along it and measuring
    perpendicular distance to the line joining endpoints.

    Returns list of (line, max_deviation, length, normalized_curvature).
    """
    results = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < segment_length_threshold:
            continue
        results.append((line[0], 0.0, length, 0.0))  # HoughLinesP returns straight segments
    return results


def detect_edge_curvature(img_gray, region_mask=None):
    """
    Use Canny edges + contour fitting to measure curvature in a region.

    For each contour, fit a line and measure the max deviation.
    Returns average normalized curvature (deviation / length).
    """
    if region_mask is not None:
        masked = cv2.bitwise_and(img_gray, img_gray, mask=region_mask)
    else:
        masked = img_gray

    # Canny edge detection
    edges = cv2.Canny(masked, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    curvatures = []
    long_contours = []

    for cnt in contours:
        if len(cnt) < 20:  # Skip very short contours
            continue

        # Fit a line to the contour
        [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)

        # Measure max perpendicular distance from the fitted line
        points = cnt.reshape(-1, 2).astype(np.float64)
        # Direction vector
        d = np.array([vx[0], vy[0]])
        p0 = np.array([x0[0], y0[0]])

        # Perpendicular distances
        diffs = points - p0
        projections = diffs @ d
        perp_vecs = diffs - np.outer(projections, d)
        perp_dists = np.linalg.norm(perp_vecs, axis=1)

        max_dev = np.max(perp_dists)

        # Arc length of contour
        arc_length = cv2.arcLength(cnt, closed=False)
        if arc_length < 30:
            continue

        # Normalized curvature: max_deviation / arc_length
        norm_curv = max_dev / arc_length

        # Only consider roughly linear contours (not circles/blobs)
        # A line-like contour should have small width relative to length
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        aspect = min(w, h) / (max(w, h) + 1e-6)

        if aspect < 0.3:  # Roughly elongated (line-like)
            curvatures.append(norm_curv)
            long_contours.append((cnt, norm_curv, arc_length, max_dev))

    return curvatures, long_contours


def analyze_grid_distortion(img_gray, grid_rows=3, grid_cols=3):
    """
    Divide image into grid and analyze curvature in each cell.
    Returns grid of average curvatures.
    """
    h, w = img_gray.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    grid_results = np.zeros((grid_rows, grid_cols))
    grid_counts = np.zeros((grid_rows, grid_cols), dtype=int)

    for r in range(grid_rows):
        for c in range(grid_cols):
            y_start = r * cell_h
            y_end = (r + 1) * cell_h if r < grid_rows - 1 else h
            x_start = c * cell_w
            x_end = (c + 1) * cell_w if c < grid_cols - 1 else w

            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y_start:y_end, x_start:x_end] = 255

            curvatures, contours = detect_edge_curvature(img_gray, mask)

            if curvatures:
                grid_results[r, c] = np.median(curvatures)
                grid_counts[r, c] = len(curvatures)
            else:
                grid_results[r, c] = 0.0

    return grid_results, grid_counts


def main():
    # ── 1. Extract frame ──
    frame = None
    video_used = None
    for vp in VIDEO_CANDIDATES:
        if os.path.exists(vp):
            frame = extract_frame(vp, FRAME_IDX)
            video_used = vp
            break

    if frame is None:
        print("ERROR: No video file found!")
        sys.exit(1)

    frame_path = os.path.join(DIST_DIR, "distortion_frame.png")
    cv2.imwrite(frame_path, frame)
    print(f"  Saved frame to {frame_path}")

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    print(f"  Image size: {w}x{h}")

    # ── 2. HoughLinesP detection ──
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=60, maxLineGap=10)

    print(f"\n=== HoughLinesP Detection ===")
    if lines is not None:
        print(f"  Detected {len(lines)} line segments")

        # Sort by length
        lengths = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            l = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            lengths.append(l)
        lengths = np.array(lengths)

        print(f"  Length stats: min={lengths.min():.1f}, max={lengths.max():.1f}, "
              f"mean={lengths.mean():.1f}, median={np.median(lengths):.1f}")

        # Top 10 longest lines
        top_idx = np.argsort(lengths)[::-1][:10]
        print(f"\n  Top 10 longest lines:")
        for i, idx in enumerate(top_idx):
            x1, y1, x2, y2 = lines[idx][0]
            print(f"    #{i+1}: ({x1},{y1})->({x2},{y2}), length={lengths[idx]:.1f}")
    else:
        print("  No lines detected!")
        lines = []

    # ── 3. Contour-based curvature analysis ──
    print(f"\n=== Contour-based Curvature Analysis ===")
    all_curvatures, all_contours = detect_edge_curvature(img_gray)

    if all_curvatures:
        curv_arr = np.array(all_curvatures)
        print(f"  Analyzed {len(all_curvatures)} line-like contours")
        print(f"  Normalized curvature (max_dev/arc_length):")
        print(f"    Mean:   {curv_arr.mean():.4f}")
        print(f"    Median: {np.median(curv_arr):.4f}")
        print(f"    Std:    {curv_arr.std():.4f}")
        print(f"    Max:    {curv_arr.max():.4f}")
        print(f"    P90:    {np.percentile(curv_arr, 90):.4f}")
        print(f"    P95:    {np.percentile(curv_arr, 95):.4f}")

    # ── 4. Grid-based distortion analysis ──
    print(f"\n=== Grid-based Distortion Analysis (3x3) ===")
    grid_curv, grid_counts = analyze_grid_distortion(img_gray)

    labels = [["TL", "TC", "TR"],
              ["ML", "MC", "MR"],
              ["BL", "BC", "BR"]]

    print(f"\n  Median normalized curvature per cell:")
    for r in range(3):
        row_str = "  "
        for c in range(3):
            row_str += f"  {labels[r][c]}: {grid_curv[r,c]:.4f} (n={grid_counts[r,c]})"
        print(row_str)

    # Edge cells vs center cell
    center_curv = grid_curv[1, 1]
    edge_cells = []
    for r in range(3):
        for c in range(3):
            if (r, c) != (1, 1):
                edge_cells.append(grid_curv[r, c])
    edge_mean = np.mean(edge_cells)
    corner_cells = [grid_curv[0,0], grid_curv[0,2], grid_curv[2,0], grid_curv[2,2]]
    corner_mean = np.mean(corner_cells)

    print(f"\n  Center cell curvature:  {center_curv:.4f}")
    print(f"  Edge cells mean:       {edge_mean:.4f}")
    print(f"  Corner cells mean:     {corner_mean:.4f}")
    print(f"  Edge/Center ratio:     {edge_mean/(center_curv+1e-8):.2f}")
    print(f"  Corner/Center ratio:   {corner_mean/(center_curv+1e-8):.2f}")

    # ── 5. FOV estimation from focal length ──
    print(f"\n=== FOV Estimation ===")
    # From config.py: fx=290.78, fy=287.35, image 640x480
    fx, fy = 290.78, 287.35
    fov_h = 2 * np.degrees(np.arctan(w / (2 * fx)))
    fov_v = 2 * np.degrees(np.arctan(h / (2 * fy)))
    fov_d = 2 * np.degrees(np.arctan(np.sqrt(w**2 + h**2) / (2 * np.sqrt(fx**2 + fy**2))))
    print(f"  Calibrated fx={fx}, fy={fy}")
    print(f"  Image: {w}x{h}")
    print(f"  Horizontal FOV: {fov_h:.1f} deg")
    print(f"  Vertical FOV:   {fov_v:.1f} deg")
    print(f"  Diagonal FOV:   {fov_d:.1f} deg")
    print(f"  (For reference: typical webcam ~60-70 deg, fisheye >120 deg)")

    # ── 6. Additional: check for barrel/pincushion distortion ──
    # Sample long edges near image borders and measure curvature
    print(f"\n=== Border Region Long-Edge Analysis ===")
    border_margin = min(w, h) // 6  # ~80px for 480p

    regions = {
        "top_strip":    (0, 0, w, border_margin),
        "bottom_strip": (0, h - border_margin, w, h),
        "left_strip":   (0, 0, border_margin, h),
        "right_strip":  (w - border_margin, 0, w, h),
        "center":       (w//4, h//4, 3*w//4, 3*h//4),
    }

    for name, (x1r, y1r, x2r, y2r) in regions.items():
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1r:y2r, x1r:x2r] = 255
        curvs, contours = detect_edge_curvature(img_gray, mask)
        if curvs:
            ca = np.array(curvs)
            print(f"  {name:15s}: median_curv={np.median(ca):.4f}, "
                  f"mean={ca.mean():.4f}, n={len(ca)}, "
                  f"max_dev_px={max(c[3] for c in contours):.1f}")
        else:
            print(f"  {name:15s}: no line-like contours found")

    # ── 7. Create visualization ──
    fig_h = 900
    fig_w = 1400
    canvas = np.ones((fig_h, fig_w, 3), dtype=np.uint8) * 255

    # Panel 1: Original frame with detected lines (top-left)
    panel_h, panel_w = 400, 640
    frame_vis = frame.copy()
    if lines is not None and len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            l = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if l > 80:
                cv2.line(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # Draw top 5 longest in red
        if len(lines) > 0:
            lens = [np.sqrt((l[0][2]-l[0][0])**2 + (l[0][3]-l[0][1])**2) for l in lines]
            top5 = np.argsort(lens)[::-1][:5]
            for idx in top5:
                x1, y1, x2, y2 = lines[idx][0]
                cv2.line(frame_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Resize to fit panel
    scale = min(panel_w / w, panel_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame_vis, (new_w, new_h))
    y_off, x_off = 10, 10
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    cv2.putText(canvas, "Detected Lines (green=long, red=top5)",
                (x_off, y_off + new_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Panel 2: Canny edges (top-right)
    edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    resized_edges = cv2.resize(edges_vis, (new_w, new_h))
    x_off2 = 10 + new_w + 30
    canvas[y_off:y_off+new_h, x_off2:x_off2+new_w] = resized_edges
    cv2.putText(canvas, "Canny Edges",
                (x_off2, y_off + new_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Panel 3: Grid curvature heatmap (bottom-left)
    cell_size = 80
    grid_vis_w = 3 * cell_size
    grid_vis_h = 3 * cell_size
    x_off3 = 10
    y_off3 = y_off + new_h + 40

    max_curv = grid_curv.max() if grid_curv.max() > 0 else 1.0
    for r in range(3):
        for c in range(3):
            val = grid_curv[r, c] / max_curv
            # Color: green (low curv) -> red (high curv)
            color = (0, int(255 * (1 - val)), int(255 * val))
            y1g = y_off3 + r * cell_size
            x1g = x_off3 + c * cell_size
            cv2.rectangle(canvas, (x1g, y1g), (x1g + cell_size, y1g + cell_size), color, -1)
            cv2.rectangle(canvas, (x1g, y1g), (x1g + cell_size, y1g + cell_size), (0, 0, 0), 1)
            cv2.putText(canvas, f"{grid_curv[r,c]:.3f}",
                        (x1g + 5, y1g + cell_size // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(canvas, f"n={grid_counts[r,c]}",
                        (x1g + 5, y1g + cell_size // 2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    cv2.putText(canvas, "Grid Curvature (normalized, green=low, red=high)",
                (x_off3, y_off3 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Panel 4: Statistics text (bottom-right)
    x_off4 = x_off3 + grid_vis_w + 40
    y_off4 = y_off3 + 5

    text_lines = [
        f"Video: .../{os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(video_used))))}/",
        f"Frame: #{FRAME_IDX}   Image: {w}x{h}",
        "",
        "--- FOV (from calibrated focal length) ---",
        f"  fx={fx:.1f}, fy={fy:.1f}",
        f"  Horizontal FOV: {fov_h:.1f} deg",
        f"  Vertical FOV:   {fov_v:.1f} deg",
        f"  Diagonal FOV:   {fov_d:.1f} deg",
        f"  (Typical: webcam 60-70, GoPro 120-170)",
        "",
        "--- Distortion Analysis ---",
        f"  HoughLinesP: {len(lines) if lines is not None else 0} segments",
        f"  Line-like contours: {len(all_curvatures)}",
    ]

    if all_curvatures:
        ca = np.array(all_curvatures)
        text_lines += [
            f"  Curvature median: {np.median(ca):.4f}",
            f"  Curvature P95:    {np.percentile(ca, 95):.4f}",
            "",
            f"  Center curvature:   {center_curv:.4f}",
            f"  Edge mean curv:     {edge_mean:.4f}",
            f"  Corner mean curv:   {corner_mean:.4f}",
            f"  Edge/Center ratio:  {edge_mean/(center_curv+1e-8):.2f}x",
            f"  Corner/Center ratio:{corner_mean/(center_curv+1e-8):.2f}x",
        ]

    text_lines += [
        "",
        "--- Conclusion ---",
    ]

    # Determine conclusion
    if fov_h < 80:
        text_lines.append(f"  FOV={fov_h:.0f} deg < 80: NOT wide-angle")
    else:
        text_lines.append(f"  FOV={fov_h:.0f} deg >= 80: wide-angle or fisheye")

    if all_curvatures:
        ratio = edge_mean / (center_curv + 1e-8)
        if ratio < 1.3 and np.median(np.array(all_curvatures)) < 0.05:
            text_lines.append("  Edge curvature ~ center: minimal distortion")
            text_lines.append("  => Pinhole model is appropriate")
        elif ratio > 1.5:
            text_lines.append("  Edge curvature >> center: significant distortion")
            text_lines.append("  => Fisheye model recommended")
        else:
            text_lines.append("  Moderate edge/center curvature difference")
            if np.median(np.array(all_curvatures)) < 0.03:
                text_lines.append("  But overall curvature is very low")
                text_lines.append("  => Pinhole model likely sufficient")
            else:
                text_lines.append("  => Mild distortion, consider light radial correction")

    for i, txt in enumerate(text_lines):
        cv2.putText(canvas, txt, (x_off4, y_off4 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Save
    output_path = os.path.join(DIST_DIR, "distortion_analysis.png")
    cv2.imwrite(output_path, canvas)
    print(f"\n=== Saved analysis to {output_path} ===")

    # ── Final summary ──
    print(f"\n{'='*60}")
    print(f"DISTORTION ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"  Camera: G1 head_stereo_left")
    print(f"  Image:  {w}x{h}")
    print(f"  Horizontal FOV: {fov_h:.1f} deg (based on fx={fx:.1f})")
    print(f"  Vertical FOV:   {fov_v:.1f} deg (based on fy={fy:.1f})")
    if fov_h < 80:
        print(f"  => This is a NORMAL lens, NOT wide-angle/fisheye.")
    else:
        print(f"  => This is a WIDE-ANGLE lens.")

    if all_curvatures:
        ca = np.array(all_curvatures)
        print(f"  Overall curvature (median): {np.median(ca):.4f}")
        print(f"  Edge/Center curvature ratio: {edge_mean/(center_curv+1e-8):.2f}x")
        if np.median(ca) < 0.03 and edge_mean / (center_curv + 1e-8) < 1.3:
            print(f"  => Distortion is NEGLIGIBLE. Pinhole model is appropriate.")
            print(f"  => config.py \u4e2d k1-k4 \u8bbe\u4e3a 0 \u662f\u5408\u7406\u7684\u3002")
        elif np.median(ca) < 0.05:
            print(f"  => Distortion is MILD. Pinhole model is likely sufficient.")
        else:
            print(f"  => Distortion is SIGNIFICANT. Consider fisheye correction.")


if __name__ == "__main__":
    main()
