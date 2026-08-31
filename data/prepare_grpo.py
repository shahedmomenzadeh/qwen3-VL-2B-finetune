#!/usr/bin/env python3
"""
Convert clip-level cataract surgery records to GRPO format JSON.

Usage:
    python data/prepare_grpo.py --input-dir dataset_grpo/Train --output data/grpo_train_dataset_grpo.json --split train --data-type all
"""
import argparse
import json
import os
import glob
from collections import Counter


def collect_grpo_jsonl_files(input_dir, data_type="all"):
    """Collect YouTube MCQ and phase-task GRPO JSONL files.

    The separated GRPO dataset contains two clip-level sources:
    ``clip_*_grpo.jsonl`` for YouTube MCQs and
    ``grpo_*_<task>_grpo.jsonl`` for phase tasks.  Full-video narration is
    SFT-only, so no full-video GRPO files are collected here.
    """
    if data_type not in {"all", "clip", "phase"}:
        raise ValueError("GRPO data type must be one of: all, clip, phase")

    files = []
    if data_type in {"all", "clip"}:
        files.extend(glob.glob(os.path.join(input_dir, "*", "clip_*_grpo.jsonl")))
    if data_type in {"all", "phase"}:
        files.extend(glob.glob(os.path.join(input_dir, "*", "grpo_*_grpo.jsonl")))
    return sorted(files)


def process_grpo_file(jsonl_path, input_dir):
    """Process a single GRPO JSONL file and return list of GRPO-format samples."""
    samples = []
    video_id_dir = os.path.basename(os.path.dirname(jsonl_path))
    basename = os.path.basename(jsonl_path)
    file_stem = os.path.splitext(basename)[0]
    parent_split = os.path.basename(input_dir)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue

            data = json.loads(line)
            prompt = data["prompt"]
            correct_answer = data.get("correct_answer", "")
            question_type = data.get("question_type", "")
            reference_reasoning = data.get("reference_reasoning", "")
            reward_type = data.get("reward_type", "")

            video_rel_path = None
            media_type = "video"
            for part in prompt[0]["content"]:
                if part["type"] in ("video", "image"):
                    video_rel_path = part[part["type"]]
                    if part["type"] == "image":
                        media_type = "image"
                    break

            if not video_rel_path:
                print(f"Warning: No video/image found in {jsonl_path}:{line_idx}")
                continue

            video_rel_path = f"{parent_split}/{video_rel_path}"

            text_parts = []
            for part in prompt[0]["content"]:
                if part["type"] == "video":
                    text_parts.append("<video>")
                elif part["type"] == "image":
                    text_parts.append("<image>")
                elif part["type"] == "text":
                    text_parts.append(part.get("text", ""))

            prompt_text = "\n".join(text_parts)
            conversations = [
                {"from": "human", "value": prompt_text},
                {"from": "gpt", "value": ""},
            ]

            sample_id = f"grpo_{video_id_dir}_{file_stem}_{line_idx}"

            sample = {
                "id": sample_id,
                media_type: video_rel_path,
                "conversations": conversations,
                "correct_answer": correct_answer,
                "question_type": question_type,
                "reference_reasoning": reference_reasoning,
                "reward_type": reward_type,
            }

            samples.append(sample)

    expected_samples = 1 if basename.startswith("grpo_") or video_id_dir.startswith("PH_") else 3
    if len(samples) != expected_samples:
        print(
            f"Warning: {jsonl_path} contains {len(samples)} samples; "
            f"expected {expected_samples} for this dataset source"
        )

    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare GRPO dataset from cataract JSONL files")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to split directory (e.g., dataset_grpo/Train)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split name (train/val/test)")
    parser.add_argument("--data-type", choices=["all", "clip", "phase"], default="all",
                        help="GRPO source: YouTube clip MCQs, phase tasks, or both")

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    jsonl_files = collect_grpo_jsonl_files(input_dir, data_type=args.data_type)
    print(f"Found {len(jsonl_files)} GRPO JSONL files in {input_dir} (--data-type={args.data_type})")

    all_samples = []
    for jsonl_path in jsonl_files:
        samples = process_grpo_file(jsonl_path, input_dir)
        all_samples.extend(samples)
        print(f"  {os.path.relpath(jsonl_path, input_dir)}: {len(samples)} samples")

    print(f"\nTotal GRPO samples: {len(all_samples)}")

    qtypes = Counter(s["question_type"] for s in all_samples)
    print(f"By question type: {dict(qtypes)}")

    rtypes = Counter(s["reward_type"] for s in all_samples)
    print(f"By reward type: {dict(rtypes)}")

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
