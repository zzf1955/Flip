"""
Mitty cache: encode (human_video, robot_video, prompt) → separate VAE + T5 caches.

VAE cache (per sample):
    Output: training_data/cache/vae/<data_type>/<duration>/<task>/pair_NNNN.pth
    Keys:   human_latent, robot_latent, prompt, source_id, task metadata

T5 cache (per unique prompt, shared within one dataset):
    Output: training_data/cache/t5/<dataset>/prompt_NNN.pth  (positive)
            training_data/cache/t5/<dataset>/negative.pth
    Keys:   embedding (1, 512, 4096), prompt (str)

Usage:
    python -m src.pipeline.mitty_cache \\
        --pair-dir training_data/pair/1s/train \\
        --output   training_data/cache/vae/pair_1s/train \\
        --device   cuda:0 \\
        --batch-size 4 \\
        --prefetch-workers 8 \\
        --prefetch-batches 2 \\
        --save-workers 1

    # T5 cache is written to --t5-cache-dir. If omitted, it is inferred from
    # --output, e.g. training_data/cache/vae/pair_1s/train ->
    # training_data/cache/t5/pair_1s.

Legacy (old format with embedded T5 + frames):
    python -m src.pipeline.mitty_cache \\
        --pair-dir training_data/pair/1s/train \\
        --output   output/mitty_cache_1s/train \\
        --legacy --device cuda:0
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.stdout.reconfigure(line_buffering=True)

import av
import numpy as np
import torch
from PIL import Image

from src.core.config import MAIN_ROOT, T5_CACHE_DIR
from src.core.wan_loader import load_text_encoder, load_tokenizer, load_vae


MANUAL_DIR = os.path.join(
    "/disk_n/zzf/.cache/huggingface/hub",
    "models--Wan-AI--Wan2.2-TI2V-5B", "manual",
)
T5_PATH = os.path.join(MANUAL_DIR, "models_t5_umt5-xxl-enc-bf16.pth")
VAE_PATH = os.path.join(MANUAL_DIR, "Wan2.2_VAE.pth")
TOKENIZER_DIR = os.path.join(MANUAL_DIR, "google", "umt5-xxl")

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def load_video_as_pil(path: str) -> list[Image.Image]:
    c = av.open(path)
    frames = [f.to_image() for f in c.decode(video=0)]
    c.close()
    return frames


def load_video_as_rgb_array(path: str) -> np.ndarray:
    """Decode video to uint8 RGB array with shape (T, H, W, 3)."""
    c = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    return np.stack(frames, axis=0)


def parse_cpu_affinity(spec: str) -> set[int]:
    """Parse Linux taskset-style CPU list, e.g. "0-17,72-89"."""
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError(f"Invalid --cpu-affinity: {spec!r}")
    return cpus


def apply_cpu_affinity(spec: str):
    """Bind this process and subsequently-created worker threads."""
    if not spec:
        return
    cpus = parse_cpu_affinity(spec)
    os.sched_setaffinity(0, cpus)
    print(f"CPU affinity: {spec} ({len(cpus)} CPUs)")


def build_models(device: str, skip_t5: bool = False):
    """Load T5 + VAE directly to GPU (no DiffSynth pipeline)."""
    t5 = tokenizer = seq_len = None
    if not skip_t5:
        t5 = load_text_encoder(T5_PATH, device, torch.bfloat16)
        tokenizer, seq_len = load_tokenizer(TOKENIZER_DIR)
    vae = load_vae(VAE_PATH, torch.bfloat16, home_device=device)
    return t5, vae, tokenizer, seq_len


def _preprocess_video(frames: list[Image.Image], device: str,
                      dtype=torch.bfloat16) -> torch.Tensor:
    """PIL frames → (1, 3, T, H, W) in [-1, 1], matching DiffSynth's preprocess_video."""
    tensors = []
    for f in frames:
        t = torch.from_numpy(np.array(f, dtype=np.float32))
        t = t.to(dtype=dtype, device=device)
        t = t * (2.0 / 255.0) - 1.0
        t = t.permute(2, 0, 1)  # (C, H, W)
        tensors.append(t)
    return torch.stack(tensors, dim=1).unsqueeze(0)  # (1, C, T, H, W)


def _preprocess_video_arrays(videos: list[np.ndarray], device: str,
                             dtype=torch.bfloat16) -> torch.Tensor:
    """uint8 RGB videos (T,H,W,C) → (B,C,T,H,W) in [-1,1] on GPU."""
    arr = np.stack(videos, axis=0)
    tensor = torch.from_numpy(arr).to(device=device, dtype=dtype)
    tensor = tensor.permute(0, 4, 1, 2, 3).contiguous()
    return tensor.mul_(2.0 / 255.0).sub_(1.0)


@torch.no_grad()
def encode_prompt(t5, tokenizer, seq_len: int, prompt: str,
                  device: str) -> torch.Tensor:
    """Encode prompt to T5 embedding. Returns (1, 512, 4096) bf16."""
    inputs = tokenizer(
        [prompt], return_tensors="pt",
        padding="max_length", truncation=True,
        max_length=seq_len, add_special_tokens=True,
    )
    ids = inputs.input_ids.to(device)
    mask = inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    emb = t5(ids, mask)
    for i, v in enumerate(seq_lens):
        emb[:, v:] = 0
    return emb


@torch.no_grad()
def encode_video(vae, frames: list[Image.Image], device: str) -> torch.Tensor:
    """VAE-encode frames. Wan 2.2 VAE: spatial 16x, temporal 4x, z_dim=48.

    480x640 / 17 frames → (1, 48, 5, 30, 40)
    """
    video_tensor = _preprocess_video(frames, device)
    latent = vae.encode(video_tensor, device=device)
    return latent.to(dtype=torch.bfloat16)


@torch.no_grad()
def encode_video_batch(vae, videos: list[list[Image.Image]],
                       device: str) -> torch.Tensor:
    """VAE-encode multiple videos in one forward pass.

    480x640 / 17-frame clips → (B, 48, 5, 30, 40).
    """
    batch = torch.cat([_preprocess_video(frames, device) for frames in videos], dim=0)
    latent = vae.single_encode(batch, device=device)
    return latent.to(dtype=torch.bfloat16)


@torch.no_grad()
def encode_video_array_batch(vae, videos: list[np.ndarray],
                             device: str) -> torch.Tensor:
    """VAE-encode predecoded uint8 RGB videos."""
    batch = _preprocess_video_arrays(videos, device)
    latent = vae.single_encode(batch, device=device)
    return latent.to(dtype=torch.bfloat16)


def encode_pair_batch(vae, human_videos: list[np.ndarray],
                      robot_videos: list[np.ndarray],
                      device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode human and robot batches separately after CPU predecode."""
    human_lats = encode_video_array_batch(vae, human_videos, device).cpu()
    robot_lats = encode_video_array_batch(vae, robot_videos, device).cpu()
    return human_lats, robot_lats


BatchItem = tuple[int, dict, Path, str]
DecodedItem = tuple[BatchItem, np.ndarray, np.ndarray]
DecodedBatch = tuple[list[BatchItem], list[np.ndarray], list[np.ndarray]]


def decode_item(pair_dir: Path, item: BatchItem) -> DecodedItem:
    """Decode one paired sample on CPU for later batch assembly."""
    _, row, _, _ = item
    human_video = load_video_as_rgb_array(str(pair_dir / row["control_video"]))
    robot_video = load_video_as_rgb_array(str(pair_dir / row["video"]))
    return item, human_video, robot_video


def decode_batch(pair_dir: Path, batch_items: list[BatchItem]) -> DecodedBatch:
    """Decode a batch of paired videos on CPU for later GPU VAE encoding."""
    decoded = [decode_item(pair_dir, item) for item in batch_items]
    items = [item for item, _, _ in decoded]
    human_videos = [human_video for _, human_video, _ in decoded]
    robot_videos = [robot_video for _, _, robot_video in decoded]
    return items, human_videos, robot_videos


def assemble_decoded_batch(decoded: list[DecodedItem]) -> DecodedBatch:
    """Assemble decoded per-sample items into one ordered batch."""
    batch_items = [item for item, _, _ in decoded]
    human_videos = [human_video for _, human_video, _ in decoded]
    robot_videos = [robot_video for _, _, robot_video in decoded]
    return batch_items, human_videos, robot_videos


def save_cache_item(out_path: Path, data: dict):
    torch.save(data, str(out_path))


def _prompt_filename(prompt: str) -> str:
    """Deterministic filename for a prompt: prompt_<first8chars_of_sha256>.pth"""
    h = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    return f"prompt_{h}.pth"


def _save_t5_cache(t5_dir: str, prompts: list[str], neg_prompt: str,
                   t5, tokenizer, seq_len: int, device: str):
    """Encode and save T5 embeddings to shared cache dir (skip if already exist)."""
    t5_dir = Path(t5_dir)
    t5_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for prompt in prompts:
        fname = _prompt_filename(prompt)
        out_path = t5_dir / fname
        if out_path.exists():
            print(f"  T5 cache exists: {fname}")
            continue
        emb = encode_prompt(t5, tokenizer, seq_len, prompt, device).cpu()
        torch.save({"embedding": emb, "prompt": prompt}, str(out_path))
        print(f"  T5 cache saved: {fname} shape={tuple(emb.shape)}")
        saved += 1

    neg_path = t5_dir / "negative.pth"
    if not neg_path.exists():
        emb = encode_prompt(t5, tokenizer, seq_len, neg_prompt, device).cpu()
        torch.save({"embedding": emb, "prompt": neg_prompt}, str(neg_path))
        print(f"  T5 cache saved: negative.pth shape={tuple(emb.shape)}")
        saved += 1
    else:
        print(f"  T5 cache exists: negative.pth")

    return saved


def infer_t5_cache_dir(output_dir: Path) -> Path:
    """Infer dataset-matched T5 cache dir from a VAE cache output path."""
    resolved_output = output_dir.resolve()
    vae_root = (Path(MAIN_ROOT) / "training_data" / "cache" / "vae").resolve()
    try:
        rel = resolved_output.relative_to(vae_root)
    except ValueError as exc:
        raise ValueError(
            "--t5-cache-dir is required when --output is not under "
            f"{vae_root}/<dataset> or {vae_root}/<data_type>/<duration>/<task>: "
            f"{resolved_output}"
        ) from exc
    parts = rel.parts
    if len(parts) < 1:
        raise ValueError(
            "Cannot infer T5 cache dir from --output; expected "
            f"{vae_root}/<dataset> or {vae_root}/<data_type>/<duration>/<task>, "
            f"got {resolved_output}"
        )
    if len(parts) >= 3 and parts[1].endswith("s"):
        return Path(T5_CACHE_DIR) / parts[0] / parts[1]
    return Path(T5_CACHE_DIR) / parts[0]


def _read_pair_manifest(pair_dir: Path) -> dict[str, dict]:
    manifest_path = pair_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        return {}
    records = {}
    with manifest_path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {manifest_path}:{line_no}") from exc
            video = record.get("video")
            if video:
                records[Path(video).name] = record
    return records


def _source_record_for(
    row: dict, source_map: dict, manifest_by_video: dict[str, dict], split_name: str,
) -> dict:
    video_name = Path(row["video"]).name
    if video_name in manifest_by_video:
        return dict(manifest_by_video[video_name])
    source_key = f"{split_name}/{video_name}"
    return dict(source_map.get(source_key, {}))


def _cache_manifest_record(
    row: dict, out_path: Path, source_record: dict, pair_dir: Path,
) -> dict:
    record = dict(source_record)
    record.setdefault("source_id", source_record.get("source_id", ""))
    record.setdefault("source_segment_id", source_record.get("source_segment_id", ""))
    record.setdefault("robot_task", source_record.get("task", ""))
    record.setdefault("data_type", source_record.get("data_type", ""))
    record.setdefault("duration", source_record.get("duration", ""))
    record["prompt"] = row["prompt"]
    record["cache_path"] = out_path.name
    record["pair_dir"] = str(pair_dir)
    return record


def _write_jsonl(path: Path, rows: list[dict]):
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Encode pair videos + prompt to .pth cache for Mitty training")
    ap.add_argument("--pair-dir", required=True,
                    help="pair split dir with video/ + control_video/ + metadata.csv")
    ap.add_argument("--output", required=True, help="VAE output cache dir")
    ap.add_argument("--t5-cache-dir", default="",
                    help="T5 cache dir; default infers training_data/cache/t5/<dataset> from --output")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true",
                    help="skip pairs whose .pth already exists")
    ap.add_argument("--legacy", action="store_true",
                    help="write old format (embedded T5 + optional PIL frames)")
    ap.add_argument("--no-frames", action="store_true",
                    help="(legacy mode only) skip saving PIL frames")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="number of samples to VAE-encode per forward pass")
    ap.add_argument("--prefetch-workers", type=int, default=0,
                    help="CPU decode workers per process (0 = synchronous)")
    ap.add_argument("--prefetch-batches", type=int, default=2,
                    help="max decoded batches buffered ahead of GPU encode")
    ap.add_argument("--save-workers", type=int, default=1,
                    help="threads for async .pth writes in new cache format")
    ap.add_argument("--cpu-affinity", default="",
                    help="optional CPU list for this process, e.g. 0-17,72-89")
    args = ap.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.prefetch_workers < 0:
        raise ValueError("--prefetch-workers must be >= 0")
    if args.prefetch_batches < 1:
        raise ValueError("--prefetch-batches must be >= 1")
    if args.save_workers < 1:
        raise ValueError("--save-workers must be >= 1")
    if args.legacy and (
        args.batch_size != 1 or args.prefetch_workers != 0 or args.save_workers != 1
    ):
        raise ValueError(
            "--legacy only supports --batch-size 1, --prefetch-workers 0, "
            "and --save-workers 1"
        )

    apply_cpu_affinity(args.cpu_affinity)

    # Resolve relative paths against MAIN_ROOT
    if not os.path.isabs(args.pair_dir):
        args.pair_dir = os.path.join(MAIN_ROOT, args.pair_dir)
    if not os.path.isabs(args.output):
        args.output = os.path.join(MAIN_ROOT, args.output)
    pair_dir = Path(args.pair_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t5_dir = args.t5_cache_dir
    if not t5_dir:
        t5_dir = str(infer_t5_cache_dir(output_dir))
    elif not os.path.isabs(t5_dir):
        t5_dir = os.path.join(MAIN_ROOT, t5_dir)

    metadata_path = pair_dir / "metadata.csv"
    if not metadata_path.exists():
        video_dir = pair_dir / "video"
        assert video_dir.is_dir(), \
            f"Neither metadata.csv nor video/ found in {pair_dir}"
        files = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
        assert files, f"No .mp4 files in {video_dir}"
        default_prompt = ("A first-person view robot arm performing "
                          "household tasks flip_v2v")
        with open(metadata_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["video", "prompt", "control_video"])
            for fn in files:
                w.writerow([f"video/{fn}", default_prompt,
                            f"control_video/{fn}"])
        print(f"Auto-generated {metadata_path} ({len(files)} entries)")

    source_map_path = pair_dir.parent / "source_map.json"
    source_map = json.load(open(source_map_path)) if source_map_path.exists() else {}
    manifest_by_video = _read_pair_manifest(pair_dir)
    split_name = pair_dir.name

    rows = list(csv.DictReader(open(metadata_path)))
    prompts = sorted(set(r["prompt"] for r in rows))
    print(f"Pairs:    {len(rows)} from {pair_dir}")
    print(f"Output:   {output_dir}")
    print(f"T5 dir:   {t5_dir}")
    print(f"Format:   {'legacy (embedded T5)' if args.legacy else 'new (separate T5)'}")
    print(f"Prompts:  {len(prompts)} unique")
    print(f"Device:   {args.device}")
    print(f"Batch:    {args.batch_size}")
    print(f"Prefetch: workers={args.prefetch_workers} batches={args.prefetch_batches}")
    print(f"Save:     workers={args.save_workers}")

    # In new mode, T5 is only needed if cache files don't exist yet
    need_t5 = args.legacy or not all(
        (Path(t5_dir) / _prompt_filename(p)).exists() for p in prompts
    ) or not (Path(t5_dir) / "negative.pth").exists()

    t5, vae, tokenizer, seq_len = build_models(args.device, skip_t5=not need_t5)
    if need_t5:
        print(f"Models loaded: T5 + VAE on {args.device}")
    else:
        print(f"Models loaded: VAE on {args.device} (T5 cache already exists)")

    # T5 encoding
    t0 = time.time()
    if args.legacy:
        prompt_emb_cache = {
            p: encode_prompt(t5, tokenizer, seq_len, p, args.device).cpu()
            for p in prompts
        }
        nega_emb = encode_prompt(
            t5, tokenizer, seq_len, NEGATIVE_PROMPT, args.device).cpu()
        print(f"T5 encode done in {time.time() - t0:.1f}s (in-memory, legacy)")
    else:
        if need_t5:
            _save_t5_cache(t5_dir, prompts, NEGATIVE_PROMPT,
                           t5, tokenizer, seq_len, args.device)
            print(f"T5 cache saved in {time.time() - t0:.1f}s → {t5_dir}")
            # Free T5 memory — only VAE needed from here
            del t5, tokenizer
            torch.cuda.empty_cache()

    # VAE encoding
    skipped = 0
    encoded = 0
    cache_manifest_rows: list[dict] = []

    if args.legacy:
        for idx, row in enumerate(rows):
            name = Path(row["video"]).stem  # pair_NNNN
            out_path = output_dir / f"{name}.pth"
            if args.resume and out_path.exists():
                skipped += 1
                continue

            t0 = time.time()
            human_path = pair_dir / row["control_video"]
            robot_path = pair_dir / row["video"]

            human_frames = load_video_as_pil(str(human_path))
            robot_frames = load_video_as_pil(str(robot_path))

            human_lat = encode_video(vae, human_frames, args.device).cpu()
            robot_lat = encode_video(vae, robot_frames, args.device).cpu()

            source_record = _source_record_for(
                row, source_map, manifest_by_video, split_name)

            data = {
                "human_latent": human_lat,
                "robot_latent": robot_lat,
                "context_posi": prompt_emb_cache[row["prompt"]],
                "context_nega": nega_emb,
                "prompt": row["prompt"],
                "source_id": source_record.get("source_id", ""),
            }
            for key in ("data_type", "duration", "robot_task", "task",
                        "source_segment_id", "episode", "seg", "clip_idx",
                        "input_role", "target_role"):
                if key in source_record:
                    data[key] = source_record[key]
            if not args.no_frames:
                data["human_frames"] = human_frames
                data["robot_frames"] = robot_frames
            torch.save(data, str(out_path))
            cache_manifest_rows.append(
                _cache_manifest_record(row, out_path, source_record, pair_dir))
            encoded += 1

            dt = time.time() - t0
            if (idx + 1) % 10 == 0 or idx == len(rows) - 1:
                print(f"  [{idx+1}/{len(rows)}] {name} shapes "
                      f"h={tuple(human_lat.shape)} r={tuple(robot_lat.shape)} ({dt:.1f}s)")
    else:
        batches: list[list[BatchItem]] = []
        pending: list[BatchItem] = []
        for idx, row in enumerate(rows):
            name = Path(row["video"]).stem  # pair_NNNN
            out_path = output_dir / f"{name}.pth"
            if args.resume and out_path.exists():
                skipped += 1
                continue
            pending.append((idx, row, out_path, name))
            if len(pending) == args.batch_size:
                batches.append(pending)
                pending = []
        if pending:
            batches.append(pending)

        save_futures: deque[Future] = deque()
        save_limit = args.save_workers * 8

        def submit_cache_save(save_pool: ThreadPoolExecutor, out_path: Path,
                              data: dict):
            save_futures.append(save_pool.submit(save_cache_item, out_path, data))
            while len(save_futures) >= save_limit:
                save_futures.popleft().result()

        def process_decoded_batch(save_pool: ThreadPoolExecutor,
                                  decoded: DecodedBatch):
            nonlocal encoded
            batch_items, human_videos, robot_videos = decoded
            t0 = time.time()
            human_lats = encode_video_array_batch(vae, human_videos, args.device).cpu()
            t_human = time.time()
            robot_lats = encode_video_array_batch(vae, robot_videos, args.device).cpu()
            t_robot = time.time()

            for batch_idx, (_, row, out_path, _) in enumerate(batch_items):
                source_record = _source_record_for(
                    row, source_map, manifest_by_video, split_name)
                data = {
                    "human_latent": human_lats[batch_idx:batch_idx + 1],
                    "robot_latent": robot_lats[batch_idx:batch_idx + 1],
                    "prompt": row["prompt"],
                    "source_id": source_record.get("source_id", ""),
                }
                for key in ("data_type", "duration", "robot_task", "task",
                            "source_segment_id", "episode", "seg", "clip_idx",
                            "input_role", "target_role"):
                    if key in source_record:
                        data[key] = source_record[key]
                submit_cache_save(save_pool, out_path, data)
                cache_manifest_rows.append(
                    _cache_manifest_record(row, out_path, source_record, pair_dir))
            t_submit = time.time()

            encoded += len(batch_items)
            last_idx, _, _, name = batch_items[-1]
            dt = t_submit - t0
            if (last_idx + 1) % 10 == 0 or last_idx == len(rows) - 1:
                print(
                    f"  [{last_idx+1}/{len(rows)}] {name} "
                    f"batch={len(batch_items)} shapes "
                    f"h={tuple(human_lats.shape)} r={tuple(robot_lats.shape)} "
                    f"({dt:.1f}s human={t_human-t0:.1f}s "
                    f"robot={t_robot-t_human:.1f}s submit={t_submit-t_robot:.1f}s)"
                )

        with ThreadPoolExecutor(max_workers=args.save_workers) as save_pool:
            if args.prefetch_workers == 0:
                for batch_items in batches:
                    process_decoded_batch(
                        save_pool, decode_batch(pair_dir, batch_items))
            else:
                with ThreadPoolExecutor(
                    max_workers=args.prefetch_workers,
                    thread_name_prefix="mitty_decode",
                ) as decode_pool:
                    items = [item for batch_items in batches for item in batch_items]
                    item_iter = iter(items)
                    item_futures: deque[Future] = deque()
                    prefetch_items = args.prefetch_batches * args.batch_size
                    for _ in range(prefetch_items):
                        try:
                            item = next(item_iter)
                        except StopIteration:
                            break
                        item_futures.append(
                            decode_pool.submit(decode_item, pair_dir, item))

                    decoded_batch: list[DecodedItem] = []
                    while item_futures:
                        decoded_batch.append(item_futures.popleft().result())
                        try:
                            item = next(item_iter)
                        except StopIteration:
                            item = None
                        if item is not None:
                            item_futures.append(
                                decode_pool.submit(decode_item, pair_dir, item))
                        if len(decoded_batch) == args.batch_size:
                            process_decoded_batch(
                                save_pool, assemble_decoded_batch(decoded_batch))
                            decoded_batch = []

                    if decoded_batch:
                        process_decoded_batch(
                            save_pool, assemble_decoded_batch(decoded_batch))

            while save_futures:
                save_futures.popleft().result()

    if cache_manifest_rows:
        _write_jsonl(output_dir / "manifest.jsonl",
                     sorted(cache_manifest_rows, key=lambda r: r["cache_path"]))
    print(f"\nDone. {encoded} encoded, {skipped} skipped → {output_dir}")


if __name__ == "__main__":
    main()
