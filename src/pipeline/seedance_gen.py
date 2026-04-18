"""
Seedance 2.0 robot-to-human video generation.

Uploads a robot video to a temporary file host, calls the Volcengine Ark
Seedance API for video-to-video generation, polls until completion, and
downloads + resizes the result.

Input requirements:
  - Aspect ratio: 4:3 (strict)
  - Duration: 4s (strict, ±0.5s tolerance)
  - Pixel count: ≥ 409,600 (e.g. 640x640 or 800x600)

Usage:
  python -m src.pipeline.seedance_gen \
    --input training_data/long/pair_0000/robot.mp4 \
    --output training_data/long/pair_0000/human.mp4 \
    --target-size 640x480

Environment:
  ARK_API_KEY   – Volcengine Ark API key (required)
  https_proxy   – proxy for catbox upload (default http://127.0.0.1:20171)
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

sys.stdout.reconfigure(line_buffering=True)

# ── constants ──────────────────────────────────────────────────────────

ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"
CATBOX_API = "https://litterbox.catbox.moe/resources/internals/api.php"
MIN_PIXELS = 409_600
FFMPEG = os.environ.get(
    "FFMPEG_BIN",
    "/home/leadtek/miniconda3/envs/flip/bin/ffmpeg",
)
FFPROBE = os.environ.get(
    "FFPROBE_BIN",
    "/home/leadtek/miniconda3/envs/flip/bin/ffprobe",
)

DEFAULT_PROMPT = (
    "将视频中的机器人完全替换为真人。真人为亚洲男性，穿白色短袖T恤，黑色长裤，灰色拖鞋。"
    "双手裸露，没有手套、没有机械关节、没有金属部件。手臂为裸露皮肤，没有任何机械结构。"
    "头部为正常人类头部，有黑色短发，没有摄像头、传感器或机械面罩。"
    "人物动作、姿态和轨迹与原视频完全一致。"
    "光影与室内环境匹配，皮肤有自然的漫反射和高光。"
    "背景、地面、家具保持不变，仅替换机器人本体。"
)

MODEL_STANDARD = "doubao-seedance-2-0-260128"
MODEL_FAST = "doubao-seedance-2-0-fast-260128"
DEFAULT_MODEL = MODEL_STANDARD


# ── helpers ────────────────────────────────────────────────────────────

def get_video_info(path: str) -> dict:
    """Return width, height, fps, duration via ffprobe."""
    cmd = [
        FFPROBE, "-v", "quiet", "-print_format", "json",
        "-show_streams", path,
    ]
    out = subprocess.check_output(cmd)
    for s in json.loads(out)["streams"]:
        if s["codec_type"] == "video":
            fps_num, fps_den = map(int, s["r_frame_rate"].split("/"))
            return {
                "width": int(s["width"]),
                "height": int(s["height"]),
                "fps": fps_num / fps_den,
                "duration": float(s.get("duration", 0)),
            }
    raise RuntimeError(f"no video stream in {path}")


def validate_input(info: dict):
    """Strict pre-flight checks: 4:3 aspect ratio, 4s duration, min pixels."""
    w, h = info["width"], info["height"]
    pixels = w * h

    # aspect ratio: allow small rounding tolerance (e.g. 640/480 = 1.3333)
    ratio = w / h
    expected = 4 / 3
    if abs(ratio - expected) > 0.02:
        raise ValueError(
            f"input aspect ratio {w}:{h} ({ratio:.4f}) is not 4:3 "
            f"(expected {expected:.4f})")

    # duration: must be ~4s (tolerance ±0.5s)
    dur = info["duration"]
    if abs(dur - 4.0) > 0.5:
        raise ValueError(
            f"input duration {dur:.2f}s is not 4s (expected 3.5–4.5s)")

    # pixel count
    if pixels < MIN_PIXELS:
        raise ValueError(
            f"input {w}x{h} = {pixels}px < minimum {MIN_PIXELS}px. "
            f"Need at least {math.ceil(MIN_PIXELS ** 0.5)}x"
            f"{math.ceil(MIN_PIXELS ** 0.5)} equivalent. "
            f"Re-encode at higher resolution before calling this script.")


def upload_catbox(path: str, expiry: str = "24h") -> str:
    """Upload file to litterbox.catbox.moe, return public URL."""
    print(f"[upload] uploading {os.path.basename(path)} to catbox …")
    proxy = os.environ.get("https_proxy", "http://127.0.0.1:20171")
    cmd = [
        "curl", "-s",
        "--proxy", proxy,
        "-F", "reqtype=fileupload",
        "-F", f"time={expiry}",
        "-F", f"fileToUpload=@{path}",
        CATBOX_API,
    ]
    result = subprocess.check_output(cmd, timeout=120).decode().strip()
    if not result.startswith("http"):
        raise RuntimeError(f"catbox upload failed: {result}")
    print(f"[upload] → {result}")
    return result


def ark_request(method: str, endpoint: str, api_key: str,
                body: dict | None = None) -> dict:
    """Make a request to the Volcengine Ark API."""
    url = f"{ARK_BASE}/{endpoint}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    # ark API is in China mainland, bypass proxy
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    try:
        with opener.open(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        raise RuntimeError(f"Ark API {e.code}: {err_body}") from e


def create_task(api_key: str, url: str, prompt: str, model: str,
                resolution: str, ratio: str, duration: int) -> str:
    """Create a Seedance generation task. Returns task_id."""
    body = {
        "model": model,
        "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": url},
             "role": "reference_video"},
        ],
        "resolution": resolution,
        "ratio": ratio,
        "duration": duration,
        "watermark": False,
    }
    resp = ark_request("POST", "contents/generations/tasks", api_key, body)
    if "error" in resp:
        raise RuntimeError(f"create task failed: {resp['error']}")
    task_id = resp["id"]
    print(f"[task] created: {task_id}")
    return task_id


def poll_task(api_key: str, task_id: str,
              interval: float = 10, timeout: float = 600) -> dict:
    """Poll until task completes. Returns full response."""
    t0 = time.time()
    while True:
        resp = ark_request("GET",
                           f"contents/generations/tasks/{task_id}", api_key)
        status = resp.get("status", "unknown")
        elapsed = time.time() - t0
        print(f"\r[poll] {status}  ({elapsed:.0f}s)", end="", flush=True)

        if status == "succeeded":
            print()
            return resp
        if status in ("failed", "cancelled", "expired"):
            print()
            raise RuntimeError(f"task {status}: {json.dumps(resp, indent=2)}")
        if elapsed > timeout:
            raise TimeoutError(
                f"task still {status} after {timeout}s")
        time.sleep(interval)


def download(url: str, path: str):
    """Download a URL to local path."""
    print(f"[download] → {path}")
    # TOS URL is in China, bypass proxy
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    req = urllib.request.Request(url)
    with opener.open(req, timeout=120) as resp, open(path, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def resize_video(src: str, dst: str, width: int, height: int):
    """Resize video to exact dimensions."""
    print(f"[resize] → {width}x{height}")
    subprocess.check_call([
        FFMPEG, "-y", "-i", src,
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        dst,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── main ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Seedance 2.0 robot→human video generation")
    p.add_argument("--input", required=True,
                   help="input robot video path")
    p.add_argument("--output", required=True,
                   help="output human video path")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="generation prompt")
    p.add_argument("--fast", action="store_true",
                   help="use fast model (default: standard)")
    p.add_argument("--resolution", default="480p",
                   choices=["480p", "720p", "1080p", "2k"])
    p.add_argument("--target-size", default=None,
                   help="resize output to WxH, e.g. 640x480")
    p.add_argument("--api-key", default=None,
                   help="Ark API key (or set ARK_API_KEY env)")
    p.add_argument("--poll-interval", type=float, default=10)
    p.add_argument("--poll-timeout", type=float, default=600)
    args = p.parse_args()

    api_key = args.api_key or os.environ.get("ARK_API_KEY")
    if not api_key:
        p.error("provide --api-key or set ARK_API_KEY env variable")

    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.getcwd(), input_path)
    if not os.path.isfile(input_path):
        p.error(f"input not found: {input_path}")

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.getcwd(), output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── info + validate ──
    info = get_video_info(input_path)
    print(f"[input] {input_path}")
    print(f"        {info['width']}x{info['height']}, "
          f"{info['fps']}fps, {info['duration']:.1f}s")
    validate_input(info)

    # ── upload ──
    public_url = upload_catbox(input_path)

    # ── create task ──
    model = MODEL_FAST if args.fast else MODEL_STANDARD
    task_id = create_task(
        api_key, public_url, args.prompt, model,
        args.resolution, ratio="4:3", duration=4)

    # ── poll ──
    result = poll_task(api_key, task_id,
                       args.poll_interval, args.poll_timeout)
    video_url = result["content"]["video_url"]
    print(f"[done] tokens={result['usage']['total_tokens']}, "
          f"seed={result.get('seed','?')}")

    # ── download ──
    if args.target_size:
        raw_path = output_path + ".raw.mp4"
    else:
        raw_path = output_path
    download(video_url, raw_path)

    # ── resize ──
    if args.target_size:
        tw, th = map(int, args.target_size.lower().split("x"))
        resize_video(raw_path, output_path, tw, th)
        os.remove(raw_path)

    # ── summary ──
    out_info = get_video_info(output_path)
    print(f"[output] {output_path}")
    print(f"         {out_info['width']}x{out_info['height']}, "
          f"{out_info['fps']}fps, {out_info['duration']:.1f}s")


if __name__ == "__main__":
    main()
