#!/usr/bin/env python3
"""
scripts/build_grpo_subset.py

Build a stratified subset of GRPO training and validation data.
Specifically tailored for testing with custom proportions:
e.g. 10% total training data with:
- 1% of total training data from YouTube-sourced videos
- 9% of total training data from Phase data
Total = 10% of total training data.

Balanced across question types within each corpus.
Verifies on-disk existence of all media files.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def classify_sample(sample):
    """
    Returns 'youtube' or 'phase' based on video path.
    YouTube clips are under {YT_ID}/ (no PH_ prefix).
    Phase clips are under PH_*/.
    """
    video = sample.get("video") or sample.get("image") or ""
    parts = video.split("/")
    # Format is usually Train/{folder}/clip_*.mp4 or {folder}/clip_*.mp4
    folder = parts[1] if len(parts) > 1 else parts[0]
    if folder.startswith("PH_"):
        return "phase"
    return "youtube"


def verify_media_exists(sample, grpo_folder):
    media = sample.get("video") or sample.get("image")
    if not media:
        return False
    path = os.path.join(grpo_folder, media)
    return os.path.exists(path)


def balanced_sample_by_qtype(pool, count, name=""):
    """
    Sample `count` items from `pool` as evenly as possible across question_types.
    """
    if count >= len(pool):
        return list(pool)

    by_qtype = defaultdict(list)
    for s in pool:
        by_qtype[s.get("question_type", "unknown")].append(s)

    qtypes = sorted(by_qtype.keys())
    if not qtypes:
        return []

    # Shuffle each bucket
    for qt in qtypes:
        random.shuffle(by_qtype[qt])

    base_per_type = count // len(qtypes)
    rem = count % len(qtypes)

    # Randomize which types get remainder
    shuffled_types = list(qtypes)
    random.shuffle(shuffled_types)

    allocated = {qt: base_per_type + (1 if i < rem else 0) for i, qt in enumerate(shuffled_types)}

    selected = []
    leftovers = []

    for qt in qtypes:
        target = allocated[qt]
        items = by_qtype[qt]
        take = min(target, len(items))
        selected.extend(items[:take])
        leftovers.extend(items[take:])

    # If any category had fewer samples than target, fill remainder from leftovers
    shortfall = count - len(selected)
    if shortfall > 0 and leftovers:
        random.shuffle(leftovers)
        selected.extend(leftovers[:shortfall])

    random.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description="Build stratified GRPO subset (e.g. 10% total: 1% YouTube, 9% Phase).")
    parser.add_argument("--train-path", type=str, required=True, help="Path to full grpo train json")
    parser.add_argument("--val-path", type=str, default=None, help="Path to full grpo val json")
    parser.add_argument("--grpo-folder", type=str, default="dataset_grpo", help="Root folder of GRPO dataset")
    parser.add_argument("--subset-ratio", type=float, default=0.10, help="Total fraction of training data to use (default 0.10)")
    parser.add_argument("--youtube-ratio", type=float, default=0.01, help="Fraction of total training data from YouTube (default 0.01)")
    parser.add_argument("--phase-ratio", type=float, default=0.09, help="Fraction of total training data from Phase (default 0.09)")
    parser.add_argument("--output-train", type=str, required=True, help="Output path for subset train json")
    parser.add_argument("--output-val", type=str, default=None, help="Output path for subset val json")
    parser.add_argument("--val-samples", type=int, default=20, help="Number of val samples to retain (default 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 65)
    print("Building Stratified GRPO Subset")
    print("=" * 65)
    print(f"Train Source   : {args.train_path}")
    print(f"GRPO Folder    : {args.grpo_folder}")
    print(f"Target Ratios  : Total={args.subset_ratio*100:.1f}%, YouTube={args.youtube_ratio*100:.1f}%, Phase={args.phase_ratio*100:.1f}%")

    with open(args.train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    total_train = len(train_data)
    print(f"\nTotal training samples in manifest: {total_train}")

    # Separate and verify existence
    yt_pool = []
    ph_pool = []
    missing_media = 0

    for s in train_data:
        if not verify_media_exists(s, args.grpo_folder):
            missing_media += 1
            continue
        c = classify_sample(s)
        if c == "youtube":
            yt_pool.append(s)
        else:
            ph_pool.append(s)

    if missing_media > 0:
        print(f"Warning: {missing_media} samples skipped because media was missing.")

    print(f"Available Verified YouTube Samples : {len(yt_pool)} ({len(yt_pool)/total_train*100:.1f}% of total)")
    print(f"Available Verified Phase Samples   : {len(ph_pool)} ({len(ph_pool)/total_train*100:.1f}% of total)")

    # Calculate exact counts
    n_yt = round(total_train * args.youtube_ratio)
    n_ph = round(total_train * args.phase_ratio)
    target_total = n_yt + n_ph

    print(f"\nTarget Selection:")
    print(f"  YouTube samples to select : {n_yt} ({n_yt/total_train*100:.2f}% of total, {n_yt/target_total*100:.1f}% of subset)")
    print(f"  Phase samples to select   : {n_ph} ({n_ph/total_train*100:.2f}% of total, {n_ph/target_total*100:.1f}% of subset)")
    print(f"  Total subset size         : {target_total} ({target_total/total_train*100:.2f}% of total)")

    if n_yt > len(yt_pool):
        print(f"Warning: Requested {n_yt} YouTube samples but only {len(yt_pool)} available. Using all.")
        n_yt = len(yt_pool)
    if n_ph > len(ph_pool):
        print(f"Warning: Requested {n_ph} Phase samples but only {len(ph_pool)} available. Using all.")
        n_ph = len(ph_pool)

    yt_selected = balanced_sample_by_qtype(yt_pool, n_yt, name="YouTube")
    ph_selected = balanced_sample_by_qtype(ph_pool, n_ph, name="Phase")

    subset_train = yt_selected + ph_selected
    random.shuffle(subset_train)

    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_train, "w", encoding="utf-8") as f:
        json.dump(subset_train, f, indent=2, ensure_ascii=False)

    print(f"\nSaved Train Subset -> {args.output_train} ({len(subset_train)} samples)")
    print("\nQuestion type distribution in Train Subset:")
    q_counts = Counter(s.get("question_type") for s in subset_train)
    for qt, count in sorted(q_counts.items(), key=lambda x: -x[1]):
        print(f"  {qt:<35}: {count}")

    # Process validation set if provided
    if args.val_path and os.path.exists(args.val_path) and args.output_val:
        with open(args.val_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)

        val_pool = [s for s in val_data if verify_media_exists(s, args.grpo_folder)]
        n_val = min(args.val_samples, len(val_pool))
        val_subset = balanced_sample_by_qtype(val_pool, n_val, name="Val")

        Path(args.output_val).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_val, "w", encoding="utf-8") as f:
            json.dump(val_subset, f, indent=2, ensure_ascii=False)

        print(f"\nSaved Val Subset -> {args.output_val} ({len(val_subset)} samples)")
        print("Question type distribution in Val Subset:")
        val_q_counts = Counter(s.get("question_type") for s in val_subset)
        for qt, count in sorted(val_q_counts.items(), key=lambda x: -x[1]):
            print(f"  {qt:<35}: {count}")

    print("=" * 65)


if __name__ == "__main__":
    main()
