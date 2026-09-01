#!/usr/bin/env python3
"""
Build balanced lite benchmark datasets covering all SFT and GRPO task types.
Outputs: data/lite_e2e/{sft_train,sft_val,grpo_train,grpo_val}.json
"""
import json
import os
import sys
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)


def classify_sft_sample(sample):
    """Classify SFT sample into youtube_full, youtube_clip, or phase_clip."""
    video = sample.get("video", "")
    parts = video.split("/")
    if len(parts) < 2:
        return "unknown"
    parent = parts[1] if len(parts) > 1 else parts[0]
    basename = parts[-1] if parts else ""
    if basename == "full_video.mp4":
        return "youtube_full"
    elif parent.startswith("PH_"):
        return "phase_clip"
    else:
        return "youtube_clip"


def verify_media_exists(sample, image_folder):
    """Check that the video/image file exists on disk."""
    media = sample.get("video") or sample.get("image")
    if not media:
        return False
    path = os.path.join(image_folder, media)
    return os.path.exists(path)


def build_sft_subsets(train_path, val_path, sft_folder):
    """Build balanced SFT train/val subsets."""
    with open(train_path) as f:
        train_data = json.load(f)
    with open(val_path) as f:
        val_data = json.load(f)

    # Group by type
    train_by_type = defaultdict(list)
    for s in train_data:
        t = classify_sft_sample(s)
        if verify_media_exists(s, sft_folder):
            train_by_type[t].append(s)

    val_by_type = defaultdict(list)
    for s in val_data:
        t = classify_sft_sample(s)
        if verify_media_exists(s, sft_folder):
            val_by_type[t].append(s)

    print("=== SFT Train Available ===")
    for k, v in sorted(train_by_type.items()):
        print(f"  {k}: {len(v)}")
    print("=== SFT Val Available ===")
    for k, v in sorted(val_by_type.items()):
        print(f"  {k}: {len(v)}")

    # Select balanced train subset
    train_subset = []
    # 2 full video narrations
    random.shuffle(train_by_type["youtube_full"])
    train_subset.extend(train_by_type["youtube_full"][:2])
    # 4 youtube clips (from different videos)
    random.shuffle(train_by_type["youtube_clip"])
    train_subset.extend(train_by_type["youtube_clip"][:4])
    # 4 phase clips
    random.shuffle(train_by_type["phase_clip"])
    train_subset.extend(train_by_type["phase_clip"][:4])

    # Select balanced val subset
    val_subset = []
    random.shuffle(val_by_type["youtube_full"])
    val_subset.extend(val_by_type["youtube_full"][:1])
    random.shuffle(val_by_type["youtube_clip"])
    val_subset.extend(val_by_type["youtube_clip"][:2])
    random.shuffle(val_by_type["phase_clip"])
    val_subset.extend(val_by_type["phase_clip"][:2])

    return train_subset, val_subset


def build_grpo_subsets(train_path, val_path, grpo_folder, train_samples=None, val_samples=None, train_per_task=None, val_per_task=None):
    """Build balanced GRPO train/val subsets covering all 7 task types.

    Args:
        train_samples: total train samples to sample (balanced across tasks).
                       If set, overrides train_per_task.
        val_samples: total val samples.
        train_per_task: samples per task (default 2 -> 14). Ignored if train_samples set.
        val_per_task: samples per task for val (default 1 -> 7).
    """
    with open(train_path) as f:
        train_data = json.load(f)
    with open(val_path) as f:
        val_data = json.load(f)

    # Group by question_type
    train_by_type = defaultdict(list)
    for s in train_data:
        qt = s.get("question_type", "unknown")
        if verify_media_exists(s, grpo_folder):
            train_by_type[qt].append(s)

    val_by_type = defaultdict(list)
    for s in val_data:
        qt = s.get("question_type", "unknown")
        if verify_media_exists(s, grpo_folder):
            val_by_type[qt].append(s)

    print("\n=== GRPO Train Available ===")
    for k, v in sorted(train_by_type.items()):
        print(f"  {k}: {len(v)}")
    print("=== GRPO Val Available ===")
    for k, v in sorted(val_by_type.items()):
        print(f"  {k}: {len(v)}")

    target_tasks = [
        "step_identification",
        "visual_observation",
        "instrument_identification",
        "boundary_detection",
        "temporal_localization",
        "timestamp_to_phase",
        "contextual_phase_recognition",
    ]

    def _balanced_subset(by_type, total=None, per_task=None, split_name="train"):
        subset = []
        if total is not None:
            # Distribute total samples as evenly as possible across tasks
            n_tasks = len(target_tasks)
            base = total // n_tasks
            remainder = total % n_tasks
            # Randomize which tasks get the extra one to avoid bias
            order = target_tasks.copy()
            random.shuffle(order)
            per_task_map = {t: base + (1 if i < remainder else 0) for i, t in enumerate(order)}
            for task in target_tasks:
                n = per_task_map[task]
                samples = by_type.get(task, [])
                random.shuffle(samples)
                take = min(n, len(samples))
                subset.extend(samples[:take])
                if take < n:
                    print(f"  WARNING: only {take}/{n} {split_name} samples for {task} (requested {n})")
            # If we couldn't fill due to scarcity, top up randomly from remaining pool
            if len(subset) < total:
                remaining = []
                for task in target_tasks:
                    taken = per_task_map[task]
                    pool = by_type.get(task, [])
                    remaining.extend(pool[taken:])
                random.shuffle(remaining)
                need = total - len(subset)
                subset.extend(remaining[:need])
                if len(subset) < total:
                    print(f"  WARNING: only {len(subset)}/{total} {split_name} samples available total")
        else:
            # Fixed per-task
            ppt = per_task if per_task is not None else (2 if split_name == "train" else 1)
            for task in target_tasks:
                samples = by_type.get(task, [])
                random.shuffle(samples)
                n = min(ppt, len(samples))
                subset.extend(samples[:n])
                if n < ppt:
                    print(f"  WARNING: only {n} {split_name} samples for {task} (requested {ppt})")
        random.shuffle(subset)
        return subset

    train_subset = _balanced_subset(train_by_type, total=train_samples, per_task=train_per_task, split_name="train")
    val_subset = _balanced_subset(val_by_type, total=val_samples, per_task=val_per_task, split_name="val")

    return train_subset, val_subset


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build balanced lite benchmark subsets")
    parser.add_argument("--output-dir", type=str, default="data/lite_e2e", help="Output directory for lite JSONs")
    parser.add_argument("--grpo-train-samples", type=int, default=None, help="Total GRPO train samples (balanced across 7 tasks, overrides per-task). E.g. 30 -> ~4 per task.")
    parser.add_argument("--grpo-val-samples", type=int, default=None, help="Total GRPO val samples (default 7).")
    parser.add_argument("--grpo-train-per-task", type=int, default=None, help="GRPO train per-task (default 2 -> 14 total).")
    parser.add_argument("--grpo-val-per-task", type=int, default=None, help="GRPO val per-task (default 1 -> 7 total).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_folder = "dataset_sft"
    grpo_folder = "dataset_grpo"

    # Prepared JSON paths
    sft_train_json = "data/sft_train_dataset_sft.json"
    sft_val_json = "data/sft_val_dataset_sft.json"
    grpo_train_json = "data/grpo_train_dataset_grpo.json"
    grpo_val_json = "data/grpo_val_dataset_grpo.json"

    for p in [sft_train_json, sft_val_json, grpo_train_json, grpo_val_json]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run prepare scripts first.")
            sys.exit(1)

    # Build SFT subsets
    print("Building SFT subsets...")
    sft_train, sft_val = build_sft_subsets(sft_train_json, sft_val_json, sft_folder)

    # Build GRPO subsets
    print("\nBuilding GRPO subsets...")
    grpo_train, grpo_val = build_grpo_subsets(
        grpo_train_json,
        grpo_val_json,
        grpo_folder,
        train_samples=args.grpo_train_samples,
        val_samples=args.grpo_val_samples,
        train_per_task=args.grpo_train_per_task,
        val_per_task=args.grpo_val_per_task,
    )

    # Save
    for name, data in [
        ("sft_train", sft_train),
        ("sft_val", sft_val),
        ("grpo_train", grpo_train),
        ("grpo_val", grpo_val),
    ]:
        path = out_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {path}: {len(data)} samples")

    # Summary
    print("\n" + "=" * 60)
    print("Lite E2E Benchmark Dataset Summary")
    print("=" * 60)
    print(f"SFT Train: {len(sft_train)} samples")
    for s in sft_train:
        print(f"  [{classify_sft_sample(s)}] {s.get('video', s.get('image', '?'))}")
    print(f"SFT Val:   {len(sft_val)} samples")
    for s in sft_val:
        print(f"  [{classify_sft_sample(s)}] {s.get('video', s.get('image', '?'))}")
    print(f"GRPO Train: {len(grpo_train)} samples")
    for s in grpo_train:
        print(f"  [{s.get('question_type')}] {s.get('video', s.get('image', '?'))}")
    print(f"GRPO Val:  {len(grpo_val)} samples")
    for s in grpo_val:
        print(f"  [{s.get('question_type')}] {s.get('video', s.get('image', '?'))}")


if __name__ == "__main__":
    main()
