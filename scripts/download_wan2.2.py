"""Download WAN2.2 models from HuggingFace.

Supports hf-mirror.com (default, faster in China) and HTTP proxy.
"""
import argparse
import os
import sys

from huggingface_hub import snapshot_download

MODELS = {
    "I2V-A14B": {
        "repo": "Wan-AI/Wan2.2-I2V-A14B",
        "desc": "Image-to-Video 14B (MoE, 27B total / 14B active), 480P & 720P",
    },
    "T2V-A14B": {
        "repo": "Wan-AI/Wan2.2-T2V-A14B",
        "desc": "Text-to-Video 14B (MoE, 27B total / 14B active), 480P & 720P",
    },
    "TI2V-5B": {
        "repo": "Wan-AI/Wan2.2-TI2V-5B",
        "desc": "Text+Image-to-Video 5B, 720P@24fps (single GPU friendly)",
    },
    "S2V-14B": {
        "repo": "Wan-AI/Wan2.2-S2V-14B",
        "desc": "Audio-driven cinematic video generation 14B",
    },
    "Animate-14B": {
        "repo": "Wan-AI/Wan2.2-Animate-14B",
        "desc": "Character animation and replacement 14B",
    },
}

HF_MIRROR = "https://hf-mirror.com"

DEFAULT_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "wan2.2")


def main():
    parser = argparse.ArgumentParser(description="Download WAN2.2 models from HuggingFace")
    parser.add_argument(
        "--model",
        nargs="+",
        choices=list(MODELS.keys()),
        default=["I2V-A14B"],
        help="Models to download (default: I2V-A14B)",
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help=f"Base directory for downloads (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--proxy",
        default="http://127.0.0.1:7897",
        help="HTTP proxy (default: http://127.0.0.1:7897 for Clash)",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Disable proxy",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        default=True,
        help="Use hf-mirror.com (default: enabled)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Use original huggingface.co",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available WAN2.2 models:\n")
        for key, info in MODELS.items():
            local_dir = os.path.join(args.base_dir, key)
            exists = os.path.isdir(local_dir) and os.listdir(local_dir)
            status = "EXISTS" if exists else "MISSING"
            print(f"  [{status}] {key}")
            print(f"    {info['desc']}")
            print(f"    Repo: {info['repo']}")
            print(f"    Local: {local_dir}")
            print()
        return

    use_mirror = args.mirror and not args.no_mirror

    # Configure endpoint and proxy
    if use_mirror:
        os.environ["HF_ENDPOINT"] = HF_MIRROR
        print(f"Using mirror: {HF_MIRROR}")
    elif not args.no_proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
        print(f"Using proxy: {args.proxy}")

    print("=" * 60)
    print("Download WAN2.2 Models")
    print("=" * 60)

    for key in args.model:
        info = MODELS[key]
        local_dir = os.path.join(args.base_dir, key)
        print(f"\n[{key}] {info['desc']}")
        print(f"  Repo: {info['repo']}")
        print(f"  Local: {local_dir}")

        try:
            snapshot_download(
                repo_id=info["repo"],
                local_dir=local_dir,
                resume_download=True,
            )
            print(f"  Done: {key}")
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            sys.exit(1)

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
