# Qwen3-VL-2B-Instruct SFT + GRPO Fine-tuning

Fine-tuning **Qwen/Qwen3-VL-2B-Instruct** using LoRA with 4-bit QLoRA on cataract surgery video data.

Pipeline:
1. **SFT** (Supervised Fine-Tuning) with LoRA + QLoRA on video description/QA data
2. **Merge** LoRA weights with base model
3. **GRPO** (Group Relative Policy Optimization) with LoRA + QLoRA on multi-choice reward data

---

## Project Structure

```
qwen3vl_2b_finetune/
├── src/                     # Core training code
│   ├── params.py            # All training arguments
│   ├── constants.py         # Token constants
│   ├── utils.py             # Model loading, reward func loading
│   ├── merge_lora.py        # Merge LoRA weights
│   ├── dataset/             # Dataset classes
│   │   ├── sft_dataset.py   # SFT dataset (video + image)
│   │   ├── grpo_dataset.py  # GRPO dataset (video + image)
│   │   └── data_utils.py    # Video/image processing utilities
│   ├── trainer/             # Custom trainers
│   │   ├── sft_trainer.py   # SFT trainer with generation-based eval
│   │   └── grpo_trainer.py  # GRPO trainer with multimodal support
│   ├── train/               # Training entry points
│   │   ├── train_sft.py     # SFT training entry point
│   │   ├── train_grpo.py    # GRPO training entry point
│   │   ├── reward_funcs.py  # Custom reward functions
│   │   └── train_utils.py   # State dict helpers
│   │   ├── monkey_patch_forward.py  # Qwen-VL mixed modality forward
│   │   └── monkey_patch_vision.py   # Vision transformer patches
│   └── model/
│       └── load_model.py    # Model loading with monkey patches
├── scripts/                 # Training & merge scripts
│   ├── finetune_sft_lora.sh
│   ├── finetune_grpo_lora.sh
│   ├── merge_lora.sh
│   ├── zero2.json
│   ├── zero3.json
│   ├── zero2_offload.json
│   └── zero3_offload.json
├── configs/                 # YAML config files
│   ├── sft_config.yaml
│   └── grpo_config.yaml
├── data/                    # Data preparation
│   ├── prepare_sft.py       # Convert JSONL → LLaVA format for SFT
│   ├── prepare_grpo.py      # Convert JSONL → GRPO format
│   └── dataset_stats.py     # Analyze dataset statistics
├── eval/
│   └── compute_metrics.py   # Custom evaluation metrics
├── pyproject.toml
├── requirements.txt
├── environment.yaml           # DEPRECATED — use uv
└── README.md
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- NVIDIA GPU with CUDA 12.4
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)

### Setup

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create and activate a virtual environment
uv venv
source .venv/bin/activate

# 3. Install all dependencies (including torch with CUDA 12.4)
uv sync

# 4. Install flash-attn separately (C++ extension, needs --no-build-isolation)
uv pip install flash-attn --no-build-isolation

# 5. Verify GPU is detected
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Quick reference

| Command | Description |
|---------|-------------|
| `uv sync` | Install / sync all deps from pyproject.toml |
| `uv add <pkg>` | Add a new dependency |
| `uv run <cmd>` | Run a command in the venv |
| `uv pip install flash-attn --no-build-isolation` | Install flash-attn |

---

## Dataset Preparation

The dataset is in `dataset/Train/`, `dataset/Validation/`, `dataset/Test/`.

### Step 1: Prepare SFT data

```bash
# Training set
uv run python data/prepare_sft.py \
    --input-dir dataset/Train \
    --output data/sft_train_clip.json \
    --split train

# Validation set
uv run python data/prepare_sft.py \
    --input-dir dataset/Validation \
    --output data/sft_val_clip.json \
    --split val
```

### Step 2: Prepare GRPO data

```bash
uv run python data/prepare_grpo.py \
    --input-dir dataset/Train \
    --output data/grpo_train_clip.json \
    --split train
```

---

## Training Pipeline

### Phase 1: Clip SFT

```bash
MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct \
DATA_PATH=data/sft_train_clip.json \
EVAL_PATH=data/sft_val_clip.json \
OUTPUT_DIR=output/sft_clip_lora \
NFRAMES=60 \
bash scripts/finetune_sft_lora.sh
```

### Phase 2: Video SFT

```bash
MODEL_NAME=output/sft_clip_merged \
DATA_PATH=data/sft_train_video.json \
EVAL_PATH=data/sft_val_video.json \
OUTPUT_DIR=output/sft_video_lora \
NFRAMES=60 \
bash scripts/finetune_sft_lora.sh
```

### Merge After SFT

```bash
# Phase 1 merge
MODEL_PATH=output/sft_clip_lora \
MODEL_BASE=Qwen/Qwen3-VL-2B-Instruct \
SAVE_MODEL_PATH=output/sft_clip_merged \
bash scripts/merge_lora.sh

# Phase 2 merge
MODEL_PATH=output/sft_video_lora \
MODEL_BASE=output/sft_clip_merged \
SAVE_MODEL_PATH=output/sft_video_merged \
bash scripts/merge_lora.sh
```

### Phase 3: Clip GRPO

```bash
MODEL_NAME=output/sft_video_merged \
DATA_PATH=data/grpo_train_clip.json \
OUTPUT_DIR=output/grpo_clip_lora \
NFRAMES=60 \
bash scripts/finetune_grpo_lora.sh
```

### Phase 4: Video GRPO

```bash
MODEL_NAME=output/sft_video_merged \
DATA_PATH=data/grpo_train_video.json \
OUTPUT_DIR=output/grpo_video_lora \
NFRAMES=60 \
bash scripts/finetune_grpo_lora.sh
```

### Merge Final Model

```bash
MODEL_PATH=output/grpo_video_lora \
MODEL_BASE=output/sft_video_merged \
SAVE_MODEL_PATH=output/grpo_merged \
bash scripts/merge_lora.sh
```

All parameters are adjustable via environment variables. For example:

```bash
GLOBAL_BATCH_SIZE=16 \
BATCH_PER_DEVICE=1 \
NUM_DEVICES=1 \
VIDEO_MIN_PIXELS=$((64 * 32 * 32)) \
VIDEO_MAX_PIXELS=$((128 * 32 * 32)) \
LEARNING_RATE=5e-5 \
DATA_PATH=data/sft_train_clip.json \
EVAL_PATH=data/sft_val_clip.json \
bash scripts/finetune_sft_lora.sh
```

---

## Key Parameters

| Parameter | SFT Default | GRPO Default | Description |
|-----------|-------------|--------------|-------------|
| `NFRAMES` | 60 | 60 | Max frames to sample (caps at video length) |
| `FPS` | — | — | Alternative to NFRAMES (mutually exclusive) |
| `VIDEO_MIN_PIXELS` | 128×32×32 | 128×32×32 | Min video resolution |
| `VIDEO_MAX_PIXELS` | 256×32×32 | 256×32×32 | Max video resolution |
| `LORA_RANK` | 32 | 32 | LoRA rank |
| `LORA_ALPHA` | 64 | 64 | LoRA alpha |
| `LORA_DROPOUT` | 0.05 | 0.05 | LoRA dropout |
| `LEARNING_RATE` | 1e-4 | 5e-6 | LLM learning rate |
| `VISION_LR` | 2e-6 | 2e-6 | Vision tower LR |
| `MERGER_LR` | 1e-5 | 1e-5 | Merger LR |
| `BITS` | 4 | 4 | Quantization (4/8/16) |
| `NUM_GENERATIONS` | N/A | 4 | GRPO group size |
| `BETA` | N/A | 0.04 | KL coefficient |
| `TEMPERATURE` | N/A | 0.9 | Generation temperature |

> **Note for Qwen3-VL models**: The pixel grid is `token * 32 * 32` (patch_size=16), NOT `token * 28 * 28`.

> **Important**: Do NOT set `FPS` and `NFRAMES` at the same time. They are mutually exclusive.

> **Known issue**: If using CuDNN errors, run `unset LD_LIBRARY_PATH` before training.

---

## Evaluation During Training

Set `SFT_COMPUTE_METRICS` env var to load custom metrics:

```bash
export SFT_COMPUTE_METRICS=/path/to/qwen3vl_2b_finetune/eval/compute_metrics.py
bash scripts/finetune_sft_lora.sh
```

---

## Training Notes

- **A100 80GB** with ZeRO-3 offload: comfortably fits the 2B model with 4-bit LoRA
- **4-bit QLoRA** + LoRA: set `--bits 4`, vision tower remains in full precision
- **Liger-Kernel**: enabled by default for memory-efficient training
- **Mixed-modality**: supports both clip-level (8-15s) and full-video (several minutes) data
- **Video resolution**: lower for full videos (`--video_min_pixels $((64 * 32 * 32)) --video_max_pixels $((128 * 32 * 32))`)
