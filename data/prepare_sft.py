#!/usr/bin/env python3
"""
Convert cataract surgery dataset (OpenAI chat format JSONL) to LLaVA format JSON for SFT training.

Usage:
    python data/prepare_sft.py \
        --input-dir ../dataset/Train \
        --output data/sft_train.json \
        --split train

    python data/prepare_sft.py \
        --input-dir ../dataset/Validation \
        --output data/sft_val.json \
        --split val

The output JSON can be passed to `--data_path` in the SFT training script.
Set `--image_folder` to the parent directory containing Train/Validation/Test.
"""
import argparse
import json
import os
import glob


def collect_jsonl_files(input_dir):
    """Collect all *sft.jsonl files from clip directories and full_video files."""
    files = []

    # Find clip_*_sft.jsonl files
    clip_pattern = os.path.join(input_dir, "*", "clip_*_sft.jsonl")
    files.extend(glob.glob(clip_pattern))

    # Find full_video_sft.jsonl files
    full_video_pattern = os.path.join(input_dir, "*", "full_video_sft.jsonl")
    files.extend(glob.glob(full_video_pattern))

    return sorted(files)


def parse_video_path(video_path):
    """Parse the video path from the JSONL to extract the relative path.

    The dataset uses relative paths like: '7-A4bHZrelA/clip_01.mp4'
    """
    return video_path


def convert_messages_to_llava(messages, is_video=True):
    """Convert OpenAI chat format messages to LLaVA conversations format."""
    conversations = []
    role_map = {"user": "human", "assistant": "gpt"}

    for msg in messages:
        role = role_map.get(msg["role"], msg["role"])
        content_parts = msg["content"]

        # Build the text value by replacing video/image tokens
        text_parts = []
        for part in content_parts:
            if part["type"] == "video":
                text_parts.append("<video>")
            elif part["type"] == "image":
                text_parts.append("<image>")
            elif part["type"] == "text":
                text_parts.append(part["text"])

        value = "\n".join(text_parts)

        conversation = {
            "from": role,
            "value": value,
        }

        # Include reasoning if present (for reasoning support)
        if "reasoning" in msg:
            conversation["reasoning"] = msg["reasoning"]

        conversations.append(conversation)

    return conversations


def process_sft_file(jsonl_path, input_dir):
    """Process a single SFT JSONL file and return list of LLaVA-format samples."""
    samples = []
    video_id_dir = os.path.basename(os.path.dirname(jsonl_path))

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue

            data = json.loads(line)
            messages = data["messages"]

            # Extract video path from the first content element
            video_rel_path = None
            for part in messages[0]["content"]:
                if part["type"] in ("video", "image"):
                    video_rel_path = part[part["type"]]
                    break

            if not video_rel_path:
                print(f"Warning: No video/image found in {jsonl_path}:{line_idx}")
                continue

            # Get the parent split name (Train/Validation/Test)
            parent_split = os.path.basename(input_dir)

            # Build sample id
            sample_id = f"{video_id_dir}_{line_idx}"

            # Determine if video or image
            media_type = "video"
            for part in messages[0]["content"]:
                if part["type"] == "image":
                    media_type = "image"
                    break

            # Convert messages to LLaVA conversations
            conversations = convert_messages_to_llava(messages, is_video=media_type == "video")

            sample = {
                "id": sample_id,
                media_type: video_rel_path,
                "conversations": conversations,
            }

            samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT dataset from cataract JSONL files")
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

    jsonl_files = collect_jsonl_files(input_dir)
    print(f"Found {len(jsonl_files)} SFT JSONL files in {input_dir}")

    all_samples = []
    for jsonl_path in jsonl_files:
        samples = process_sft_file(jsonl_path, input_dir)
        all_samples.extend(samples)
        print(f"  {os.path.relpath(jsonl_path, input_dir)}: {len(samples)} samples")

    print(f"\nTotal SFT samples: {len(all_samples)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
