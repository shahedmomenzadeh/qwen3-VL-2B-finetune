#!/usr/bin/env python3
"""
upload_to_hub.py — Upload a local model directory to the Hugging Face Hub.

Usage (standalone):
    python src/upload_to_hub.py \\
        --local-dir output/sft_video_merged \\
        --repo-id your-username/qwen3-vl-2b-cataract-sft \\
        [--private] \\
        [--token hf_xxxx]

Environment variables (can substitute CLI flags):
    HF_TOKEN       — HuggingFace access token (write permission required)
    HF_HUB_REPO    — Target repo id (e.g. "username/model-name")
    HF_PRIVATE     — Set to "1" to create a private repo (default: "0")

The script:
  1. Logs into the Hub (token via --token or $HF_TOKEN).
  2. Creates the repository if it does not exist.
  3. Uploads all files in --local-dir using upload_folder (LFS-aware).
  4. Prints the URL of the uploaded model on success.
"""

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload a merged model to HuggingFace Hub.")
    p.add_argument(
        "--local-dir",
        required=True,
        help="Local directory containing the model to upload (e.g. output/sft_video_merged).",
    )
    p.add_argument(
        "--repo-id",
        default=os.environ.get("HF_HUB_REPO", ""),
        help=(
            "HuggingFace repository ID in the form 'username/model-name'. "
            "Can also be set via $HF_HUB_REPO."
        ),
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace write-access token. Can also be set via $HF_TOKEN.",
    )
    p.add_argument(
        "--private",
        action="store_true",
        default=(os.environ.get("HF_PRIVATE", "0") == "1"),
        help="Create or keep the repository as private (default: public).",
    )
    p.add_argument(
        "--commit-message",
        default="Upload fine-tuned Qwen3-VL-2B SFT checkpoint",
        help="Commit message for the upload.",
    )
    p.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=["optimizer.pt", "scheduler.pt", "__pycache__/*", "*.bin.index.json"],
        help="Glob patterns of files to exclude from the upload.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Validation ────────────────────────────────────────────────────────────
    if not args.repo_id:
        print(
            "[upload_to_hub] ERROR: No --repo-id provided and $HF_HUB_REPO is not set.\n"
            "  Example: HF_HUB_REPO=username/my-model bash train_sft.sh",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.token:
        print(
            "[upload_to_hub] ERROR: No --token provided and $HF_TOKEN is not set.\n"
            "  Example: HF_TOKEN=hf_xxxx bash train_sft.sh",
            file=sys.stderr,
        )
        sys.exit(1)

    local_dir = os.path.abspath(args.local_dir)
    if not os.path.isdir(local_dir):
        print(
            f"[upload_to_hub] ERROR: --local-dir '{local_dir}' does not exist or is not a directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Import huggingface_hub ────────────────────────────────────────────────
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print(
            "[upload_to_hub] ERROR: 'huggingface_hub' is not installed.\n"
            "  Install it with: pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Login ─────────────────────────────────────────────────────────────────
    print("[upload_to_hub] Logging in to HuggingFace Hub...")
    login(token=args.token, add_to_git_credential=False)

    api = HfApi()

    # ── Create repo if needed ─────────────────────────────────────────────────
    print(f"[upload_to_hub] Ensuring repository '{args.repo_id}' exists...")
    repo_url = api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,  # no-op if the repo already exists
    )
    print(f"[upload_to_hub] Repository URL: {repo_url}")

    # ── Upload ────────────────────────────────────────────────────────────────
    visibility = "private" if args.private else "public"
    print(
        f"[upload_to_hub] Uploading '{local_dir}' -> '{args.repo_id}' ({visibility})...\n"
        f"  Ignoring patterns: {args.ignore_patterns}"
    )

    api.upload_folder(
        folder_path=local_dir,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
        ignore_patterns=args.ignore_patterns,
    )

    model_url = f"https://huggingface.co/{args.repo_id}"
    print(f"\n[upload_to_hub] Upload complete!")
    print(f"[upload_to_hub] Model available at: {model_url}")


if __name__ == "__main__":
    main()
