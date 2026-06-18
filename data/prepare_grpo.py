#!/usr/bin/env python3
"""
Convert cataract surgery dataset (OpenAI chat format JSONL) to GRPO format JSON.

Usage:
    python data/prepare_grpo.py \
        --input-dir ../dataset/Train \
        --output data/grpo_train.json \
        --split train

    python data/prepare_grpo.py \
        --input-dir ../dataset/Validation \
        --output data/grpo_val.json \
        --split val

The output JSON contains samples with:
    {
        "id": "...",
        "video": "YT_ID/clip_xx.mp4",
        "conversations": [
            {"from": "human", "value": "<video>\\nQuestion..."},
            {"from": "gpt", "value": "B"}
        ],
        "correct_answer": "B",
        "question_type": "step_identification",
        "reference_reasoning": "...",
        "reward_type": "deterministic"
    }
"""
import argparse
import json
import os
import glob


def collect_grpo_jsonl_files(input_dir):
    """Collect all *grpo.jsonl files from clip directories and full_video files."""
    files = []

    # Find clip_*_grpo.jsonl files
    clip_pattern = os.path.join(input_dir, "*", "clip_*_grpo.jsonl")
    files.extend(glob.glob(clip_pattern))

    # Find full_video_grpo.jsonl files
    full_video_pattern = os.path.join(input_dir, "*", "full_video_grpo.jsonl")
    files.extend(glob.glob(full_video_pattern))

    return sorted(files)


def process_grpo_file(jsonl_path, input_dir):
    """Process a single GRPO JSONL file and return list of GRPO-format samples."""
    samples = []
    video_id_dir = os.path.basename(os.path.dirname(jsonl_path))
    basename = os.path.basename(jsonl_path)
    is_full_video = basename.startswith("full_video")

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

            # Extract video path from the prompt
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

            # Build prompt text from OpenAI format
            text_parts = []
            for part in prompt[0]["content"]:
                if part["type"] == "video":
                    text_parts.append("<video>")
                elif part["type"] == "image":
                    text_parts.append("<image>")
                elif part["type"] == "text":
                    text_parts.append(part["text"])

            prompt_text = "\n".join(text_parts)

            # Build LLaVA conversations format
            conversations = [
                {"from": "human", "value": prompt_text},
                {"from": "gpt", "value": ""},  # Empty for GRPO - model generates
            ]

            # Build sample id
            prefix = "full_" if is_full_video else ""
            sample_id = f"grpo_{prefix}{video_id_dir}_{line_idx}"

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

    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare GRPO dataset from cataract JSONL files")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to split directory (e.g., dataset/Train)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split name (train/val/test)")

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    jsonl_files = collect_grpo_jsonl_files(input_dir)
    print(f"Found {len(jsonl_files)} GRPO JSONL files in {input_dir}")

    all_samples = []
    for jsonl_path in jsonl_files:
        samples = process_grpo_file(jsonl_path, input_dir)
        all_samples.extend(samples)
        print(f"  {os.path.relpath(jsonl_path, input_dir)}: {len(samples)} samples")

    print(f"\nTotal GRPO samples: {len(all_samples)}")

    # Count by question_type
    qtypes = {}
    for s in all_samples:
        qt = s["question_type"]
        qtypes[qt] = qtypes.get(qt, 0) + 1
    print(f"By question type: {qtypes}")

    # Count by reward_type
    rtypes = {}
    for s in all_samples:
        rt = s["reward_type"]
        rtypes[rt] = rtypes.get(rt, 0) + 1
    print(f"By reward type: {rtypes}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
