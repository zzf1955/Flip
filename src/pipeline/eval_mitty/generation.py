"""Model loading and video generation for offline Mitty evaluation."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from diffsynth.diffusion.flow_match import FlowMatchScheduler

from src.core.train_utils import load_sample, save_video, tensor_to_frames
from src.pipeline.backbones import get_mitty_spec
from src.pipeline.eval_mitty.run_specs import RunSpec, resolve_pair_media
from src.pipeline.train_mitty import DEFAULT_DIT_DIR, DEFAULT_TOKENIZER, DEFAULT_VAE
from src.tools.eval_metrics import read_video_frames


def load_model(
    run: RunSpec,
    device: str,
    lora_rank: int | None,
    lora_target_modules: str | None,
    lora_attn_types: str,
    lora_attn_projections: str,
    dit_dir: str,
    vae_path: str,
    tokenizer_dir: str,
):
    spec = get_mitty_spec()
    extra_kwargs = {}
    if run.merge_lora_paths:
        extra_kwargs["merge_lora_paths"] = [str(p) for p in run.merge_lora_paths]
    model = spec.training_module_factory(
        device=device,
        dit_dir=dit_dir,
        vae_path=vae_path,
        tokenizer_dir=tokenizer_dir,
        lora_rank=lora_rank,
        lora_target_modules=lora_target_modules,
        lora_attn_types=lora_attn_types,
        lora_attn_projections=lora_attn_projections,
        use_gradient_checkpointing=False,
        load_vae=True,
        init_lora_path=str(run.checkpoint),
        **extra_kwargs,
    )
    model.eval()
    model.pipe.dit.eval()
    return model, spec


def video_triplet_complete(paths: dict[str, Path]) -> bool:
    """Return true when an interrupted eval already wrote a decodable triplet."""
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        return False
    frame_counts = []
    frame_shapes = []
    for label in ("gen", "gt", "ctrl"):
        try:
            frames = read_video_frames(str(paths[label]))
        except (RuntimeError, ValueError):
            return False
        frame_counts.append(len(frames))
        frame_shapes.append(frames.shape[1:])
    return (
        frame_counts[0] > 0
        and len(set(frame_counts)) == 1
        and len(set(frame_shapes)) == 1
    )


@torch.no_grad()
def generate_split(
    model,
    spec,
    records: list[dict],
    out_dir: Path,
    device: str,
    num_inference_steps: int,
    cfg_scale: float,
    t5_pos: dict[str, torch.Tensor],
    t5_neg: torch.Tensor,
    resume_existing: bool,
    show_progress: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = model.pipe
    sched = FlowMatchScheduler("Wan")
    sched.set_timesteps(
        num_inference_steps=num_inference_steps,
        denoising_strength=1.0,
        shift=5.0,
    )

    for idx, record in enumerate(records):
        sample_id = f"{idx:05d}"
        paths = {
            "gen": out_dir / f"gen_{sample_id}.mp4",
            "gt": out_dir / f"gt_{sample_id}.mp4",
            "ctrl": out_dir / f"ctrl_{sample_id}.mp4",
        }
        if resume_existing and video_triplet_complete(paths):
            if show_progress:
                print(
                    f"  Generate {out_dir.name}: {idx + 1}/{len(records)} "
                    f"sample={sample_id} skip existing",
                    flush=True,
                )
            continue

        if show_progress:
            print(
                f"  Generate {out_dir.name}: {idx + 1}/{len(records)} sample={sample_id}",
                flush=True,
            )
        sample = load_sample(
            record["cache_path"],
            device=device,
            t5_pos=t5_pos,
            t5_neg=t5_neg,
        )
        denoised = spec.eval_denoise_fn(
            pipe=pipe,
            sample=sample,
            sched=sched,
            device=device,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
        )

        pipe.load_models_to_device(["vae"])
        gen_video = pipe.vae.decode(denoised, device=device, tiled=False)
        save_video(tensor_to_frames(gen_video), str(paths["gen"]))

        gt_path = resolve_pair_media(record, "video")
        ctrl_path = resolve_pair_media(record, "control_video")
        if not gt_path.is_file():
            raise FileNotFoundError(f"GT video not found: {gt_path}")
        if not ctrl_path.is_file():
            raise FileNotFoundError(f"Control video not found: {ctrl_path}")
        shutil.copy2(gt_path, paths["gt"])
        shutil.copy2(ctrl_path, paths["ctrl"])


__all__ = [
    "DEFAULT_DIT_DIR",
    "DEFAULT_TOKENIZER",
    "DEFAULT_VAE",
    "generate_split",
    "load_model",
]
