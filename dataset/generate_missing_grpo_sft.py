import os
import json
import time
import glob
import argparse
from openai import OpenAI
from pydantic import BaseModel, ValidationError


# ==========================================
# Pydantic Schemas
# ==========================================
class QAPair(BaseModel):
    question_type: str
    question: str
    options: dict[str, str]
    correct_answer: str
    reference_reasoning: str


class QAList(BaseModel):
    qa_pairs: list[QAPair]


# ==========================================
# Prompts
# ==========================================
SYSTEM_PROMPT = """You are an expert surgical educator creating multiple-choice questions for a Vision-Language Model training dataset based on cataract surgery video clips.

Your task is to generate high-quality questions that test understanding of surgical procedures, visual observations, and instrument identification.

CRITICAL RULES:
1. Write naturally as if you're watching the video directly - NEVER mention metadata field names like "visual_description", "step_title", "instruments list", etc.
2. In your reasoning, describe what is visible in the video using natural language (e.g., "In the video, the surgeon uses sharp tips to puncture..." NOT "The visual_description states...")
3. Generate EXACTLY 3 questions covering these categories:
   - "step_identification": Identify the surgical step being performed
   - "visual_observation": Ask about specific visual details, techniques, or observations
   - "instrument_identification": Identify surgical tools being used

Each question object MUST include ALL of these fields:
- "question_type": one of ["step_identification", "visual_observation", "instrument_identification"]
- "question": the question text
- "options": a dictionary with exactly 4 keys: "A", "B", "C", and "D"
- "correct_answer": just the letter (e.g., "A")
- "reference_reasoning": natural explanation as if describing what you see in the video

Example format:
{
  "qa_pairs": [
    {
      "question_type": "step_identification",
      "question": "What surgical step is being performed in this clip?",
      "options": {
        "A": "Capsulorhexis",
        "B": "Phacoemulsification",
        "C": "IOL insertion",
        "D": "Corneal incision"
      },
      "correct_answer": "A",
      "reference_reasoning": "In this video clip, the surgeon is creating a circular opening in the anterior capsule of the lens, which is the defining characteristic of capsulorhexis."
    }
  ]
}
"""


# ==========================================
# Core Logic
# ==========================================
def find_clip_mp4s(dataset_dir):
    """Find all clip_XX.mp4 files across Train/Validation/Test splits."""
    clips = []
    for split in ["Train", "Validation", "Test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for vid_id in sorted(os.listdir(split_dir)):
            vid_dir = os.path.join(split_dir, vid_id)
            if not os.path.isdir(vid_dir):
                continue
            for mp4 in sorted(glob.glob(os.path.join(vid_dir, "clip_*.mp4"))):
                base_name = os.path.basename(mp4).replace(".mp4", "")
                clips.append({
                    "split": split,
                    "vid_id": vid_id,
                    "base_name": base_name,
                    "mp4_path": mp4,
                    "vid_dir": vid_dir,
                })
    return clips


def read_clip_metadata(vid_dir, base_name):
    """Read the clip_XX.jsonl metadata file."""
    meta_path = os.path.join(vid_dir, f"{base_name}.jsonl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.loads(f.readline())


def generate_qa_pairs(client, model_name, clip_data, max_retries=3):
    """Call LLM to generate 3 QA pairs for a clip."""
    clip_id = clip_data.get("clip_id", "")
    user_prompt = f"""You are watching a cataract surgery video clip (ID: {clip_id}).

Based on this surgical context, generate exactly 3 multiple-choice questions:

Surgical Step: {clip_data.get("step_title", "")}

What's happening in the video:
{clip_data.get("visual_description", "")}

Instruments visible: {", ".join(clip_data.get("instruments", []))}
Anatomical structures: {", ".join(clip_data.get("anatomy", []))}

Generate:
1. One "step_identification" question about what surgical step is being performed
2. One "visual_observation" question about specific visual details or techniques shown
3. One "instrument_identification" question about the surgical tools being used

IMPORTANT: Write your reasoning naturally, as if describing what you observe in the video. Do NOT reference metadata field names."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=2048,
            )
            raw_output = response.choices[0].message.content
            parsed_json = json.loads(raw_output)
            validated = QAList(**parsed_json)
            return validated.qa_pairs
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"  [attempt {attempt+1}] Validation error: {e}")
        except Exception as e:
            print(f"  [attempt {attempt+1}] LLM error: {e}")
            time.sleep(2)
    return None


def write_sft_file(vid_dir, base_name, video_rel_path, clip_data, qa_pairs):
    """Write clip_XX_sft.jsonl with description + QA pairs."""
    path = os.path.join(vid_dir, f"{base_name}_sft.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        # Line 1: pure visual description
        sft_desc = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_rel_path},
                        {"type": "text", "text": "Describe what is happening in this cataract surgical video clip."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": clip_data.get("visual_description", ""),
                },
            ]
        }
        f.write(json.dumps(sft_desc) + "\n")

        # Lines 2-4: QA pairs
        for qa in qa_pairs:
            options_text = "\n".join([f"{k}) {v}" for k, v in qa.options.items()])
            question_text = f"{qa.question}\n{options_text}\n\nProvide your reasoning first, then state your answer."
            assistant_answer = f"{qa.reference_reasoning} Therefore the answer is {qa.correct_answer}) {qa.options[qa.correct_answer]}."

            sft_qa = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video_rel_path},
                            {"type": "text", "text": question_text},
                        ],
                    },
                    {"role": "assistant", "content": assistant_answer},
                ]
            }
            f.write(json.dumps(sft_qa) + "\n")


def write_grpo_file(vid_dir, base_name, video_rel_path, qa_pairs):
    """Write clip_XX_grpo.jsonl from QA pairs."""
    path = os.path.join(vid_dir, f"{base_name}_grpo.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            options_text = "\n".join([f"{k}) {v}" for k, v in qa.options.items()])
            question_text = f"{qa.question}\n{options_text}\n\nProvide your reasoning first, then state your answer."

            reward_type = "deterministic" if qa.question_type in ["step_identification", "instrument_identification"] else "llm_judge"
            grpo_qa = {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video_rel_path},
                            {"type": "text", "text": question_text},
                        ],
                    }
                ],
                "correct_answer": qa.correct_answer,
                "question_type": qa.question_type,
                "reference_reasoning": qa.reference_reasoning,
                "reward_type": reward_type,
            }
            f.write(json.dumps(grpo_qa) + "\n")


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Generate missing clip SFT/GRPO files using an LLM.")
    parser.add_argument("dataset_dir", help="Path to the dataset root (containing Train/Validation/Test)")
    parser.add_argument("--base-url", default="https://api.gapgpt.app/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="sk-qfAZiv5r45oTbP7XBCAVNBJVzltURT8KebSG6C6ixBomBcuJ", help="API key")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Model name to use for generation")
    parser.add_argument("--retries", type=int, default=3, help="Max retries per clip")
    parser.add_argument("--dry-run", action="store_true", help="Only show what's missing, don't generate")
    args = parser.parse_args()

    # Find all clip MP4s
    all_clips = find_clip_mp4s(args.dataset_dir)
    print(f"Found {len(all_clips)} total clip_*.mp4 files across all splits.\n")

    # Check what's missing
    missing_sft = []
    missing_grpo = []
    for clip in all_clips:
        sft_path = os.path.join(clip["vid_dir"], f"{clip['base_name']}_sft.jsonl")
        grpo_path = os.path.join(clip["vid_dir"], f"{clip['base_name']}_grpo.jsonl")
        if not os.path.exists(sft_path):
            missing_sft.append(clip)
        if not os.path.exists(grpo_path):
            missing_grpo.append(clip)

    print(f"Missing _sft.jsonl: {len(missing_sft)}")
    print(f"Missing _grpo.jsonl: {len(missing_grpo)}")
    print()

    if args.dry_run:
        if missing_sft:
            print("Missing SFT files:")
            for c in missing_sft:
                print(f"  {c['split']}/{c['vid_id']}/{c['base_name']}_sft.jsonl")
        if missing_grpo:
            print("Missing GRPO files:")
            for c in missing_grpo:
                print(f"  {c['split']}/{c['vid_id']}/{c['base_name']}_grpo.jsonl")
        return

    # Union of clips that need at least one file
    clips_to_process = {}
    for c in missing_sft + missing_grpo:
        key = (c["split"], c["vid_id"], c["base_name"])
        if key not in clips_to_process:
            clips_to_process[key] = c

    if not clips_to_process:
        print("All clips already have both _sft.jsonl and _grpo.jsonl. Nothing to do.")
        return

    print(f"Clips to process: {len(clips_to_process)}\n")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    success_count = 0
    error_count = 0

    for (split, vid_id, base_name), clip in clips_to_process.items():
        clip_data = read_clip_metadata(clip["vid_dir"], base_name)
        if not clip_data:
            print(f"[SKIP] {split}/{vid_id}/{base_name} — no metadata .jsonl found")
            error_count += 1
            continue

        video_rel_path = f"{vid_id}/{base_name}.mp4"
        print(f"[{split}] {vid_id}/{base_name} ... ", end="", flush=True)

        qa_pairs = generate_qa_pairs(client, args.model, clip_data, max_retries=args.retries)
        if qa_pairs is None:
            print("FAILED")
            error_count += 1
            continue

        # Write only the missing files
        if (split, vid_id, base_name) in {(c["split"], c["vid_id"], c["base_name"]) for c in missing_sft}:
            write_sft_file(clip["vid_dir"], base_name, video_rel_path, clip_data, qa_pairs)
        if (split, vid_id, base_name) in {(c["split"], c["vid_id"], c["base_name"]) for c in missing_grpo}:
            write_grpo_file(clip["vid_dir"], base_name, video_rel_path, qa_pairs)

        print(f"OK (3 QA pairs)")
        success_count += 1

    print(f"\nDone. Success: {success_count}, Errors: {error_count}")


if __name__ == "__main__":
    main()
