#!/usr/bin/env python3
"""Estimate which annotated clips are visible to full-video frame sampling.

The estimate mirrors the useful property of uniform full-video sampling:
sampled timestamps are distributed across the complete parent video. A clip is
eligible when at least ``min_frames`` sampled timestamps fall inside its parent
interval.

Example:
    python3 data/estimate_full_video_coverage.py \
        --dataset-dir dataset_sft --nframes 60 --min-frames 6
"""

import argparse
import json
import re
import subprocess
from pathlib import Path


def parse_timestamp(value):
    """Parse MM:SS or HH:MM:SS timestamps into seconds."""
    parts = [float(part) for part in value.strip().split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported timestamp: {value!r}")


def probe_duration(path):
    """Read the full-video duration using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def sampled_timestamps(duration, nframes):
    """Return timestamps for evenly spaced full-video frame sampling."""
    if nframes < 1:
        raise ValueError("nframes must be positive")
    if duration <= 0:
        return []
    if nframes == 1:
        return [0.0]
    return [duration * index / (nframes - 1) for index in range(nframes)]


def count_frames_in_interval(samples, start, end):
    """Count sampled timestamps inside an annotated interval."""
    # Small tolerance handles timestamps rounded to whole seconds in JSONL.
    tolerance = 0.05
    return sum(start - tolerance <= timestamp <= end + tolerance for timestamp in samples)


def read_clip_metadata(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    return None


def analyze_video(video_dir, nframes, min_frames):
    full_video = video_dir / "full_video.mp4"
    if not full_video.exists():
        return None

    duration = probe_duration(full_video)
    samples = sampled_timestamps(duration, nframes)
    clips = []

    metadata_paths = [
        path
        for path in video_dir.glob("clip_*.jsonl")
        if re.fullmatch(r"clip_\d+\.jsonl", path.name)
    ]
    for metadata_path in sorted(metadata_paths):
        metadata = read_clip_metadata(metadata_path)
        if not metadata:
            continue
        try:
            start = parse_timestamp(metadata["timestamp_start_in_parent"])
            end = parse_timestamp(metadata["timestamp_end_in_parent"])
        except (KeyError, TypeError, ValueError):
            continue

        observed = count_frames_in_interval(samples, start, end)
        clips.append(
            {
                "clip": metadata_path.stem,
                "start": start,
                "end": end,
                "duration": max(0.0, end - start),
                "fraction": max(0.0, end - start) / duration,
                "sampled_frames": observed,
                "selected": observed >= min_frames,
            }
        )

    return {
        "video_id": video_dir.name,
        "duration": duration,
        "clips": clips,
    }


def format_seconds(seconds):
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_sft"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Validation", "Test"],
        choices=["Train", "Validation", "Test"],
    )
    parser.add_argument("--nframes", type=int, default=60)
    parser.add_argument("--min-frames", type=int, default=6)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    totals = {split: {"videos": 0, "clips": 0, "selected": 0} for split in args.splits}

    print(
        f"Full-video coverage estimate: nframes={args.nframes}, "
        f"minimum_target_frames={args.min_frames}"
    )
    print("Sampling model: evenly spaced timestamps over each full video")

    for split in args.splits:
        split_dir = args.dataset_dir / split
        if not split_dir.is_dir():
            print(f"{split}: missing")
            continue

        split_videos = 0
        split_clips = 0
        split_selected = 0
        selected_fractions = []

        for video_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            analysis = analyze_video(video_dir, args.nframes, args.min_frames)
            if analysis is None:
                continue
            split_videos += 1
            split_clips += len(analysis["clips"])
            selected = [clip for clip in analysis["clips"] if clip["selected"]]
            split_selected += len(selected)
            selected_fractions.extend(clip["fraction"] for clip in selected)

            if args.details:
                for clip in analysis["clips"]:
                    status = "SELECT" if clip["selected"] else "skip"
                    print(
                        f"  {split}/{analysis['video_id']}/{clip['clip']}: {status} "
                        f"{format_seconds(clip['start'])}-{format_seconds(clip['end'])}, "
                        f"{clip['sampled_frames']} sampled frames, "
                        f"{clip['fraction']:.1%} of video"
                    )

        totals[split] = {
            "videos": split_videos,
            "clips": split_clips,
            "selected": split_selected,
        }
        percentage = 100 * split_selected / split_clips if split_clips else 0.0
        print(
            f"{split}: videos={split_videos}, clips={split_clips}, "
            f"selected={split_selected} ({percentage:.1f}%)"
        )

    all_videos = sum(value["videos"] for value in totals.values())
    all_clips = sum(value["clips"] for value in totals.values())
    all_selected = sum(value["selected"] for value in totals.values())
    percentage = 100 * all_selected / all_clips if all_clips else 0.0
    print(
        f"Total: videos={all_videos}, clips={all_clips}, "
        f"selected={all_selected} ({percentage:.1f}%)"
    )


if __name__ == "__main__":
    main()
