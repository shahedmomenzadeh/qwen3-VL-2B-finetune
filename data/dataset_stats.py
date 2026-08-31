#!/usr/bin/env python3
"""
Analyze the cataract surgery dataset statistics.
"""
import argparse
import json
import os
import glob
from collections import Counter


def analyze_split(input_dir, split_name):
    """Analyze all files in a split directory."""
    print(f"\n{'='*60}")
    print(f"Dataset Split: {split_name}")
    print(f"{'='*60}")

    # Count videos
    video_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    print(f"Video directories: {len(video_dirs)}")

    total_clips = 0
    total_full_videos = 0
    total_sft_lines = 0
    total_grpo_lines = 0
    clip_durations = []
    question_types = Counter()
    reward_types = Counter()

    for vid_dir in video_dirs:
        dir_path = os.path.join(input_dir, vid_dir)

        # Count clip files
        mp4_files = [f for f in os.listdir(dir_path) if f.endswith('.mp4')]
        clip_mp4s = [f for f in mp4_files if f.startswith('clip_')]
        full_mp4s = [f for f in mp4_files if f.startswith('full_video')]

        total_clips += len(clip_mp4s)
        total_full_videos += len(full_mp4s)

        # Analyze SFT files
        sft_files = glob.glob(os.path.join(dir_path, "*_sft.jsonl"))
        for sf in sft_files:
            with open(sf, 'r') as f:
                lines = [l for l in f if l.strip()]
                total_sft_lines += len(lines)

        # Analyze GRPO files
        grpo_files = glob.glob(os.path.join(dir_path, "*_grpo.jsonl"))
        for gf in grpo_files:
            with open(gf, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        total_grpo_lines += 1
                        qt = data.get("question_type", "unknown")
                        rt = data.get("reward_type", "unknown")
                        question_types[qt] += 1
                        reward_types[rt] += 1

    print(f"Clips (MP4): {total_clips}")
    print(f"Full videos: {total_full_videos}")
    print(f"SFT samples: {total_sft_lines}")
    print(f"GRPO samples: {total_grpo_lines}")

    if question_types:
        print(f"\nQuestion types:")
        for qt, count in question_types.most_common():
            print(f"  {qt}: {count}")

    if reward_types:
        print(f"\nReward types:")
        for rt, count in reward_types.most_common():
            print(f"  {rt}: {count}")

    return {
        "split": split_name,
        "videos": len(video_dirs),
        "clips": total_clips,
        "full_videos": total_full_videos,
        "sft_samples": total_sft_lines,
        "grpo_samples": total_grpo_lines,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze cataract dataset statistics")
    parser.add_argument("--dataset-dir", type=str, default="dataset_sft",
                        help="Path to dataset root directory (dataset_sft or dataset_grpo)")

    args = parser.parse_args()
    dataset_dir = os.path.abspath(args.dataset_dir)

    splits = ["Train", "Validation", "Test"]
    all_stats = []

    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if os.path.isdir(split_dir):
            stats = analyze_split(split_dir, split)
            all_stats.append(stats)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    total = {}
    for stat in all_stats:
        for k, v in stat.items():
            if k != "split":
                total[k] = total.get(k, 0) + v

    for k, v in total.items():
        print(f"Total {k}: {v}")


if __name__ == "__main__":
    main()
