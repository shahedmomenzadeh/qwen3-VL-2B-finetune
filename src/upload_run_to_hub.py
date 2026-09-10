#!/usr/bin/env python3
"""
upload_run_to_hub.py — Publish a finished training run to the Hugging Face Hub.

Two uploads (each skippable, everything re-runnable/idempotent):

  1. Whole output dir (adapters, checkpoints, logs, merged models — everything)
     -> a DATASET repo, e.g. username/qwen3-vl-2b-sft-stage2-runs
  2. Final merged model dir -> a MODEL repo, plus:
       - training_logs/logs.zip  (zipped logs dir: train.log, gpu.csv,
         losses.csv, summary.txt, config.txt, merge.log, ...)
       - training_configs/<...>  (config.txt + cmd.txt from each run dir,
         i.e. the exact configs the model was trained with)

Usage (on the server after training finishes):
    HF_TOKEN=hf_xxxx .venv/bin/python src/upload_run_to_hub.py \\
        --output-dir output \\
        --merged-dir output/sft_stage2_merged \\
        --logs-dir output/logs \\
        --runs-repo username/qwen3-vl-2b-sft-stage2-runs \\
        --model-repo username/qwen3-vl-2b-cataract-sft-stage2 [--private]

    # Only one side:
    ... --skip-runs      # just the model (+ logs.zip + configs)
    ... --skip-model     # just the full-output dataset repo

Environment variables (substitute for CLI flags):
    HF_TOKEN           — write-access token (required)
    HF_HUB_RUNS_REPO   — default for --runs-repo
    HF_HUB_REPO        — default for --model-repo (same var train_sft.sh uses)
    HF_PRIVATE         — "1" for private repos (default: "0")

Tip: for multi-GB uploads install the Rust transfer backend once
(`.venv/bin/pip install hf_transfer`) and export HF_HUB_ENABLE_HF_TRANSFER=1.
"""

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


CONFIG_FILENAMES = ("config.txt", "cmd.txt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish a finished run (outputs + model) to HF Hub.")
    p.add_argument("--output-dir", default="output",
                   help="Whole run output dir for the dataset-repo upload.")
    p.add_argument("--merged-dir", default="",
                   help="Final merged model dir for the model-repo upload.")
    p.add_argument("--logs-dir", default="",
                   help="Logs dir to zip into the model repo (default: <output-dir>/logs).")
    p.add_argument("--runs-repo", default=os.environ.get("HF_HUB_RUNS_REPO", ""),
                   help="Dataset repo id for the full output upload (or $HF_HUB_RUNS_REPO).")
    p.add_argument("--model-repo", default=os.environ.get("HF_HUB_REPO", ""),
                   help="Model repo id for the merged model (or $HF_HUB_REPO).")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN", ""),
                   help="Write-access token (or $HF_TOKEN).")
    p.add_argument("--private", action="store_true",
                   default=(os.environ.get("HF_PRIVATE", "0") == "1"),
                   help="Create repos as private (default: public).")
    p.add_argument("--skip-runs", action="store_true",
                   help="Skip the full-output dataset-repo upload.")
    p.add_argument("--skip-model", action="store_true",
                   help="Skip the merged-model upload.")
    return p.parse_args()


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def fmt_gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def build_logs_zip(logs_dir: Path, dest: Path) -> Path:
    """Zip the whole logs tree (train.log, gpu.csv, losses.csv, configs, ...)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(logs_dir.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(logs_dir))
    return dest


def collect_configs(logs_dir: Path):
    """Yield (local_path, repo_path) for training config files."""
    for f in sorted(logs_dir.rglob("*")):
        if f.is_file() and f.name in CONFIG_FILENAMES:
            yield f, Path("training_configs") / f.relative_to(logs_dir)


def main() -> None:
    args = parse_args()

    if not args.token:
        print("[upload-run] ERROR: no --token and $HF_TOKEN is not set.", file=sys.stderr)
        sys.exit(1)
    if not args.skip_runs and not args.runs_repo:
        print("[upload-run] ERROR: no --runs-repo and $HF_HUB_RUNS_REPO is not set.", file=sys.stderr)
        sys.exit(1)
    if not args.skip_model and not args.model_repo:
        print("[upload-run] ERROR: no --model-repo/--merged-dir (or $HF_HUB_REPO).", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    merged_dir = Path(args.merged_dir) if args.merged_dir else None
    logs_dir = Path(args.logs_dir) if args.logs_dir else output_dir / "logs"
    if not output_dir.is_dir():
        print(f"[upload-run] ERROR: --output-dir '{output_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("[upload-run] ERROR: 'huggingface_hub' not installed.", file=sys.stderr)
        sys.exit(1)
    try:
        import hf_transfer  # noqa: F401
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") != "1":
            print("[upload-run] NOTE: hf_transfer installed but HF_HUB_ENABLE_HF_TRANSFER!=1 "
                  "(export it for faster multi-GB uploads).")
    except ImportError:
        print("[upload-run] NOTE: 'hf_transfer' not installed — uploads will be slower "
              "(`pip install hf_transfer` + HF_HUB_ENABLE_HF_TRANSFER=1).")

    print("[upload-run] Logging in...")
    login(token=args.token, add_to_git_credential=False)
    api = HfApi()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    vis = "private" if args.private else "public"

    # ── 1. Whole output dir -> dataset repo ──────────────────────────────
    if not args.skip_runs:
        print(f"[upload-run] Output dir size: {fmt_gb(dir_size(output_dir))} ({output_dir})")
        print(f"[upload-run] Ensuring dataset repo '{args.runs_repo}' ({vis})...")
        api.create_repo(repo_id=args.runs_repo, repo_type="dataset",
                        private=args.private, exist_ok=True)
        print(f"[upload-run] Uploading '{output_dir}' -> dataset '{args.runs_repo}'...")
        api.upload_folder(folder_path=str(output_dir), repo_id=args.runs_repo,
                          repo_type="dataset",
                          commit_message=f"Training run outputs ({stamp})",
                          ignore_patterns=["__pycache__/*", "*.pyc"])
        print(f"[upload-run] Runs published: https://huggingface.co/datasets/{args.runs_repo}")

    # ── 2. Merged model (+ logs.zip + configs) -> model repo ─────────────
    if not args.skip_model:
        if merged_dir is None or not merged_dir.is_dir():
            print(f"[upload-run] ERROR: merged dir '{merged_dir}' not found.", file=sys.stderr)
            sys.exit(1)
        print(f"[upload-run] Ensuring model repo '{args.model_repo}' ({vis})...")
        api.create_repo(repo_id=args.model_repo, repo_type="model",
                        private=args.private, exist_ok=True)
        print(f"[upload-run] Uploading '{merged_dir}' -> model '{args.model_repo}'...")
        api.upload_folder(folder_path=str(merged_dir), repo_id=args.model_repo,
                          repo_type="model",
                          commit_message=f"Fine-tuned model ({stamp})")

        if logs_dir.is_dir():
            with tempfile.TemporaryDirectory(prefix="run_logs_") as tmp:
                zip_path = Path(tmp) / "logs.zip"
                print(f"[upload-run] Zipping '{logs_dir}' ({fmt_gb(dir_size(logs_dir))})...")
                build_logs_zip(logs_dir, zip_path)
                print(f"[upload-run] logs.zip: {fmt_gb(zip_path.stat().st_size)} -> "
                      f"'{args.model_repo}' @ training_logs/logs.zip")
                api.upload_file(path_or_fileobj=str(zip_path),
                                path_in_repo="training_logs/logs.zip",
                                repo_id=args.model_repo, repo_type="model",
                                commit_message=f"Training logs ({stamp})")
            configs = list(collect_configs(logs_dir))
            for local, repo_path in configs:
                api.upload_file(path_or_fileobj=str(local), path_in_repo=str(repo_path),
                                repo_id=args.model_repo, repo_type="model",
                                commit_message=f"Training configs ({stamp})")
            print(f"[upload-run] Uploaded {len(configs)} config file(s) under training_configs/.")
        else:
            print(f"[upload-run] WARNING: logs dir '{logs_dir}' not found — skipping logs.zip/configs.")

        print(f"[upload-run] Model published: https://huggingface.co/{args.model_repo}")

    print("[upload-run] Done.")


if __name__ == "__main__":
    main()
