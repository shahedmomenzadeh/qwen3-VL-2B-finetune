# Cataract Surgery VLM Training Dataset (SFT & GRPO)

## Overview

This dataset is a structured, multimodal collection of annotated video clips and full-length surgical videos from cataract surgery. It is specifically curated and formatted to train Vision-Language Models (VLMs) to understand surgical workflows, identify intraocular instruments, recognize anatomical boundaries, and perform visual reasoning.

### Dual-Track Training Compatibility

The dataset supports two complementary training approaches:

- **Supervised Fine-Tuning (SFT)**: For standard visual description and natural language question-answering
- **Group Relative Policy Optimization (GRPO)**: For reinforcement learning based on multi-choice question tracks with deterministic or LLM-judged reward setups

---

## 📊 Dataset Statistics

| Split | Videos | Clip MP4s | Full Videos | Clip SFTs | Clip GRPOs |
|---|---|---|---|---|---|
| **Train** | 108 | 1,141 | 108 | 1,141 | 1,141 |
| **Validation** | 13 | 137 | 13 | 137 | 137 |
| **Test** | 15 | 162 | 15 | 162 | 162 |
| **Total** | **136** | **1,440** | **136** | **1,440** | **1,440** |

---

## 📂 Dataset Structure

The root directory contains three traditional machine learning splits: **Train**, **Validation**, and **Test**. Within each split, data is isolated by its source YouTube Video ID to guarantee complete parent-level splitting and prevent data leakage across clips. Each video directory contains both clip-level segments (8-15 second slices) and the full uncut surgical video, along with their corresponding annotation files.

```
dataset/
├── Train/
├── Validation/
└── Test/
    ├── zOlICw1iEhI/              # Unique YouTube Video ID
    │   ├── full_video.mp4        # Complete uncut surgical video
    │   ├── full_video_sft.jsonl   # SFT timestamped narration for full video
    │   ├── full_video_grpo.jsonl  # GRPO sequence-ordering task for full video
    │   ├── clip_01.mp4           # Raw video slice (8-15 seconds)
    │   ├── clip_01.jsonl         # Ground-truth surgery metadata
    │   ├── clip_01_sft.jsonl     # SFT Multi-turn dialogue / description
    │   ├── clip_01_grpo.jsonl    # Reinforcement learning reasoning prompts
    │   └── [clip_XX.* ...]       # Additional clip-level files
    └── [Other_YT_IDs]/
```

---

## 📋 File Specifications

### 1. Core Metadata (`clip_xx.jsonl`)

Contains the underlying clinical parameters and ground-truth text annotations sourced from video transcripts and surgical analysis.

**Example:**

```json
{
  "clip_id": "zOlICw1iEhI_clip_01",
  "parent_video_id": "zOlICw1iEhI",
  "clip_filename": "clip_01.mp4",
  "parent_video_title": "Achieving the psychological state of Flow during Cataract Surgery",
  "parent_video_url": "https://www.youtube.com/watch?v=zOlICw1iEhI",
  "step_number": 1,
  "clip_duration_seconds": 8,
  "timestamp_start_in_parent": "00:07",
  "timestamp_end_in_parent": "00:15",
  "step_title": "viscoelastic injection",
  "visual_description": "A cannula is inserted into the anterior chamber through a paracentesis to inject viscoelastic, deepening and stabilizing the chamber.",
  "transcript_context": "",
  "instruments": ["Viscoelastic cannula"],
  "anatomy": ["anterior chamber", "cornea", "iris"]
}
```

### 2. SFT Dataset (`clip_xx_sft.jsonl`)

Formatted using the OpenAI/Hugging Face multi-turn chat scheme. It maps a video token payload directly to a descriptive output string or an expanded chain-of-thought QA string.

**Example:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "video",
          "video": "zOlICw1iEhI/clip_01.mp4"
        },
        {
          "type": "text",
          "text": "Describe what is happening in this cataract surgical video clip."
        }
      ]
    },
    {
      "role": "assistant",
      "content": "A cannula is inserted into the anterior chamber through a paracentesis to inject viscoelastic, deepening and stabilizing the chamber."
    }
  ]
}
```

**Format Notes:**
- Each `clip_*_sft.jsonl` file contains **multiple lines** (typically 1-4): Line 1 is always a pure visual description; subsequent lines are multi-choice QA pairs with chain-of-thought reasoning
- The `video` key utilizes a clean relative path schema (`YT_ID/clip_xx.mp4`), allowing you to prepend your local dataset root path seamlessly during batch loading

### 3. GRPO Dataset (`clip_xx_grpo.jsonl`)

Formatted explicitly for rule-based or RL-guided preference modeling. It separates the evaluation query, multi-choice choices, targeted correct letter, and clinical reasoning background.

**Example:**

```json
{
  "prompt": [
    {
      "role": "user",
      "content": [
        {
          "type": "video",
          "video": "zOlICw1iEhI/clip_01.mp4"
        },
        {
          "type": "text",
          "text": "Which surgical step is being performed in this clip?\nA) Creation of a corneal incision\nB) Viscoelastic injection to deepen the anterior chamber\nC) Phacoemulsification of the cataract nucleus\nD) Insertion of the intraocular lens\n\nProvide your reasoning first, then state your answer."
        }
      ]
    }
  ],
  "correct_answer": "B",
  "question_type": "step_identification",
  "reference_reasoning": "In the video the surgeon introduces a thin cannula through a small paracentesis and slowly injects a clear, jelly-like substance, causing the anterior chamber to expand and become more stable. This maneuver is the classic viscoelastic injection step.",
  "reward_type": "deterministic"
}
```

### 4. Full-Video SFT Dataset (`full_video_sft.jsonl`)

Contains SFT training data for the entire uncut surgical video. Each `full_video_sft.jsonl` file contains exactly **2 lines** (2 separate multi-turn dialogues):

- **Line 1 — Timestamped Narration:** The assistant provides a detailed step-by-step narration of the full procedure with `[MM:SS - MM:SS]` intervals.
- **Line 2 — Step-Ordering Chain-of-Thought:** The assistant receives a shuffled list of surgical steps and re-orders them into correct chronological sequence with full reasoning.

**Example:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "video",
          "video": "zOlICw1iEhI/full_video.mp4"
        },
        {
          "type": "text",
          "text": "Provide a detailed timestamped narration of this surgical procedure."
        }
      ]
    },
    {
      "role": "assistant",
      "content": "[00:07 - 00:15] viscoelastic injection: A cannula is inserted into the anterior chamber through a paracentesis to inject viscoelastic, deepening and stabilizing the chamber.\n[00:16 - 00:30] capsulorhexis: Capsulorhexis forceps are used to create a continuous curvilinear capsulorhexis...\n..."
    }
  ]
}
```

**Format Notes:**
- The assistant response in Line 1 uses the format: `[MM:SS - MM:SS] step_name: detailed description`
- Line 1 covers the complete surgical procedure from start to finish
- Line 2 is a step-ordering chain-of-thought dialogue (shuffled steps → correct chronological order with reasoning)

### 5. Full-Video GRPO Dataset (`full_video_grpo.jsonl`)

Formatted for sequence-ordering reinforcement learning over the entire procedure. The model must reorder all surgical steps into their correct chronological sequence as they appear in the full video.

**Example:**

```json
{
  "prompt": [
    {
      "role": "user",
      "content": [
        {
          "type": "video",
          "video": "zOlICw1iEhI/full_video.mp4"
        },
        {
          "type": "text",
          "text": "Reorder the following surgical steps into the exact chronological sequence they appear in the video:\nA) viscoelastic injection\nB) capsulorhexis\nC) phacoemulsification\nD) intraocular lens (iol) implantation\nE) paracentesis creation\n\nOutput only the correct sequence of letters separated by commas (e.g., B, D, A, C)."
        }
      ]
    }
  ],
  "correct_answer": "E, A, B, C, D",
  "question_type": "sequence_ordering",
  "reward_type": "deterministic"
}
```

**Format Notes:**
- `correct_answer` is a comma-separated string of letters representing the true chronological order
- `question_type` is always `"sequence_ordering"` for full-video GRPO
- `reward_type` is always `"deterministic"` (exact string match validation)
- Each video directory contains exactly **1 line** per video

### 6. Clip-Level GRPO Question Types & Reward Types

The clip-level `clip_*_grpo.jsonl` files use multiple `question_type` and `reward_type` combinations:

| `question_type` | Description |
|---|---|
| `step_identification` | Identifies which surgical step is shown in the clip |
| `instrument_identification` | Identifies the surgical instrument being used |
| `visual_observation` | Identifies visual cues or anatomical changes in the clip |

| `reward_type` | Description |
|---|---|
| `deterministic` | Exact string match validation (e.g., answer is "B") |
| `llm_judge` | Pass the model's response and `reference_reasoning` to a teacher LLM for grading |

Each `clip_*_grpo.jsonl` file typically contains **1-3 lines**, one per question. Every clip is guaranteed to have at least one `step_identification` question with a `deterministic` reward.

---

## 🎯 Primary Use Cases

### 1. Visual Instruction Tuning (SFT)

**Goal:** Teach a generic VLM (e.g., LLaVA, Video-LLaMA, Qwen2-VL) specialized ophthalmic vocabulary and scene dynamics.

**Implementation:**
- **Clip-level:** Parse all `clip_*_sft.jsonl` lines across the Train directory for short-segment description training
- **Full-video:** Parse all `full_video_sft.jsonl` lines for long-context timestamped narration training
- Feed the multi-modal dictionary directly into standard training frameworks like Hugging Face SFTTrainer or LLaMA-Factory

### 2. Reinforcement Learning / Alignment (GRPO)

**Goal:** Mitigate model hallucinations during medical scenarios, ensuring the model reasons logically step-by-step before selecting a surgical decision.

**Implementation:**
- **Clip-level:** Use `clip_*_grpo.jsonl` for single-step identification tasks with multi-choice questions
- **Full-video:** Use `full_video_grpo.jsonl` for sequence-ordering tasks where the model must reconstruct the correct chronological order of all surgical steps
- Implement a custom reward function that checks if the string contains the target `correct_answer` regex pattern (e.g., "Therefore the answer is B" or the comma-separated sequence)
- For rows where `reward_type` is `"llm_judge"`, pass the model's generated sequence alongside the `reference_reasoning` to a teacher LLM to grade the scientific accuracy of its surgical analysis