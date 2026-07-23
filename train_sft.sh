#!/bin/bash
# train_sft.sh — End-to-end SFT pipeline for Qwen3-VL-2B (clips + full_video)
#
# Stages:
#   1. Environment setup (uv + venv + install deps)
#   2. Dataset download from Google Drive (if not present)
#   3. Data preparation (JSONL → LLaVA-format JSON)
#   4. OPTIONAL: subsample training data via SUBSET_RATIO (0-1)
#   5. SFT on clips  → merge → output/sft_clip_merged
#   6. SFT on videos → merge → output/sft_video_merged  (final model)
#
# Usage:
#   bash train_sft.sh                          # full training, defaults
#   SUBSET_RATIO=0.3 bash train_sft.sh         # 30% of training data (test run)
#   BITS=16 NFRAMES=48 bash train_sft.sh       # 16-bit LoRA (no quantization)
#
# ── REQUIRED: fill in Google Drive file IDs below (or set via env vars) ──
TRAIN_ZIP_ID="${TRAIN_ZIP_ID:-PUT_TRAIN_ZIP_GOOGLE_DRIVE_ID_HERE}"
VAL_ZIP_ID="${VAL_ZIP_ID:-PUT_VAL_ZIP_GOOGLE_DRIVE_ID_HERE}"
TEST_ZIP_ID="${TEST_ZIP_ID:-PUT_TEST_ZIP_GOOGLE_DRIVE_ID_HERE}"
#
# Also set HF_TOKEN if you need to download the model from HuggingFace:
#   export HF_TOKEN=hf_xxxxx
# ============================================================================

set -euo pipefail

log()  { echo -e "\033[1;32m[train_sft]\033[0m $1"; }
warn() { echo -e "\033[1;33m[train_sft]\033[0m $1"; }
err()  { echo -e "\033[1;31m[train_sft] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ════════════════════════════════════════════════════════════════════════════
# 1. ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════════
log "=== 1. Environment Setup ==="

if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH:-}"
fi
log "uv version: $(uv --version)"

if [ ! -d ".venv" ]; then
    log "Creating virtual environment..."
    uv venv --python 3.12
fi

VENV_PYTHON=".venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    err ".venv/bin/python not found — venv creation failed"
fi
log "Python: $($VENV_PYTHON --version)"

# Install PyTorch (CUDA 13.0)
log "Installing PyTorch..."
uv pip install --python "$VENV_PYTHON" torch torchvision \
    --index-url https://download.pytorch.org/whl/cu130

# Install transformers from GitHub main (latest Qwen3-VL support)
log "Installing transformers from GitHub main..."
uv pip install --python "$VENV_PYTHON" \
    "git+https://github.com/huggingface/transformers.git"

# Install remaining dependencies
log "Installing remaining dependencies..."
uv pip install --python "$VENV_PYTHON" \
    --extra-index-url https://pypi.org/simple \
    "accelerate" \
    "bitsandbytes>=0.43.0" \
    "qwen-vl-utils[decord]" \
    "peft>=0.13.0" \
    "trl>=1.8.0" \
    "liger-kernel>=0.8.0" \
    "ujson" \
    "tensorboard" \
    "einops" \
    "sentencepiece" \
    "protobuf" \
    "pillow" \
    "tqdm" \
    "datasets" \
    "imageio" \
    "gdown"

# Optional: flash-attn for faster attention (skip if build fails)
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
    log "Installing flash-attn (this may take a few minutes to compile)..."
    uv pip install --python "$VENV_PYTHON" flash-attn --no-build-isolation 2>&1 || {
        warn "flash-attn install failed — continuing with SDPA attention"
        warn "To use flash attention, install manually: uv pip install flash-attn --no-build-isolation"
    }
fi

# Verify core imports
log "Verifying installation..."
$VENV_PYTHON -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
import transformers; print(f'transformers: {transformers.__version__}')
from transformers import AutoModelForImageTextToText; print('AutoModelForImageTextToText: OK')
import peft; print(f'peft: {peft.__version__}')
import trl; print(f'trl: {trl.__version__}')
import liger_kernel; print('liger_kernel: OK')
import bitsandbytes; print(f'bitsandbytes: {bitsandbytes.__version__}')
import qwen_vl_utils; print('qwen_vl_utils: OK')
" || err "Package verification failed"

log "Environment setup complete."

# ════════════════════════════════════════════════════════════════════════════
# 2. HUGGINGFACE CACHE + TOKEN
# ════════════════════════════════════════════════════════════════════════════
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"
HF_TOKEN="${HF_TOKEN:-}"
[ -n "$HF_TOKEN" ] && export HF_TOKEN
export PYTHONPATH="src:${PYTHONPATH:-}"

# Enable generation-based eval metrics (set ENABLE_GEN_EVAL=0 to disable)
ENABLE_GEN_EVAL="${ENABLE_GEN_EVAL:-1}"
if [ "$ENABLE_GEN_EVAL" = "1" ]; then
    export SFT_COMPUTE_METRICS="eval/compute_metrics.py"
fi

# ════════════════════════════════════════════════════════════════════════════
# 3. DATASET DOWNLOAD
# ════════════════════════════════════════════════════════════════════════════
log "=== 2. Dataset Setup ==="

mkdir -p dataset

# Helper: download + unzip a split from Google Drive if its dir is missing
download_split() {
    local split_name="$1"
    local zip_id="$2"
    local split_dir="dataset/$split_name"

    if [ -d "$split_dir" ] && [ -n "$(ls -A "$split_dir" 2>/dev/null)" ]; then
        log "  $split_name/ already exists, skipping download."
        return
    fi

    if [ -z "$zip_id" ] || echo "$zip_id" | grep -q "PUT_.*_HERE"; then
        warn "  $split_name: Google Drive ID not set — skipping download."
        warn "  Set ${split_name^^}_ZIP_ID and re-run, or populate dataset/$split_name manually."
        return
    fi

    log "  Downloading $split_name.zip from Google Drive (ID: $zip_id)..."
    $VENV_PYTHON -m gdown "$zip_id" --output "$split_name.zip" 2>&1 \
        || err "gdown failed for $split_name (ID: $zip_id)"

    log "  Extracting $split_name.zip into dataset/..."
    unzip -o -q "$split_name.zip" -d dataset/ || err "unzip failed for $split_name.zip"
    rm -f "$split_name.zip"
    log "  $split_name ready."
}

download_split "Train"      "$TRAIN_ZIP_ID"
download_split "Validation" "$VAL_ZIP_ID"
download_split "Test"        "$TEST_ZIP_ID"

# Verify dataset structure
for split in Train Validation; do
    if [ ! -d "dataset/$split" ] || [ -z "$(ls -A "dataset/$split" 2>/dev/null)" ]; then
        err "dataset/$split/ is empty or missing. Set ${split^^}_ZIP_ID or copy data manually."
    fi
done
log "Dataset verified: dataset/Train/ and dataset/Validation/ present."

# ════════════════════════════════════════════════════════════════════════════
# 4. CONFIGURATION (all overridable via env vars)
# ════════════════════════════════════════════════════════════════════════════
log "=== 3. Configuration ==="

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"
DATA_PREFIX="${DATA_PREFIX:-data}"

# QLoRA
BITS="${BITS:-4}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Batch
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_DEVICES="${NUM_DEVICES:-1}"
GLOBAL_BATCH_SIZE=$((BATCH_PER_DEVICE * GRAD_ACCUM * NUM_DEVICES))

# Learning
LR="${LR:-1e-4}"
VISION_LR="${VISION_LR:-2e-6}"
MERGER_LR="${MERGER_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"

# Video
NFRAMES="${NFRAMES:-60}"
FPS="${FPS:-}"            # leave empty to use NFRAMES; set to override
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-$((128 * 32 * 32))}"   # 131072
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-$((256 * 32 * 32))}"   # 262144

# Attention
DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-0}"

# Eval / save
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
EVAL_STEPS="${EVAL_STEPS:-300}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-300}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
REPORT_TO="${REPORT_TO:-tensorboard}"

# Subset ratio (0.0–1.0): use only a fraction of the training data.
# Useful for smoke testing the pipeline on the GPU server.
SUBSET_RATIO="${SUBSET_RATIO:-1.0}"

# Force regenerate prepared JSONs even if they already exist
FORCE_REPREPARE="${FORCE_REPREPARE:-0}"

log "  MODEL_ID=$MODEL_ID"
log "  BITS=$BITS  LORA_RANK=$LORA_RANK  LORA_ALPHA=$LORA_ALPHA"
log "  BATCH=$BATCH_PER_DEVICE  GRAD_ACCUM=$GRAD_ACCUM  GLOBAL_BS=$GLOBAL_BATCH_SIZE"
log "  LR=$LR  VISION_LR=$VISION_LR  MERGER_LR=$MERGER_LR"
log "  EPOCHS=$NUM_EPOCHS  NFRAMES=$NFRAMES  FPS=${FPS:-unset}"
log "  VIDEO_MIN=$VIDEO_MIN_PIXELS  VIDEO_MAX=$VIDEO_MAX_PIXELS"
log "  SUBSET_RATIO=$SUBSET_RATIO"

# ════════════════════════════════════════════════════════════════════════════
# 5. DATA PREPARATION
# ════════════════════════════════════════════════════════════════════════════
log "=== 4. Data Preparation ==="

NEED_PREPARE=0
[ "$FORCE_REPREPARE" = "1" ] && NEED_PREPARE=1
for f in sft_train_clip.json sft_train_video.json sft_val_clip.json sft_val_video.json; do
    [ ! -f "$DATA_PREFIX/$f" ] && NEED_PREPARE=1
done

if [ "$NEED_PREPARE" = "1" ]; then
    log "Running prepare_sft.py for all splits and data types..."
    $VENV_PYTHON data/prepare_sft.py --input-dir dataset/Train      --output "$DATA_PREFIX/sft_train_clip.json" --data-type clip
    $VENV_PYTHON data/prepare_sft.py --input-dir dataset/Train      --output "$DATA_PREFIX/sft_train_video.json" --data-type full_video
    $VENV_PYTHON data/prepare_sft.py --input-dir dataset/Validation --output "$DATA_PREFIX/sft_val_clip.json"   --data-type clip
    $VENV_PYTHON data/prepare_sft.py --input-dir dataset/Validation --output "$DATA_PREFIX/sft_val_video.json"   --data-type full_video
else
    log "Prepared JSONs already exist (set FORCE_REPREPARE=1 to regenerate)."
fi

# Apply SUBSET_RATIO to training files (only train, not eval)
if [ "$SUBSET_RATIO" != "1.0" ] && [ "$SUBSET_RATIO" != "1" ]; then
    log "Subsampling training data to ${SUBSET_RATIO} of original..."

    $VENV_PYTHON -c "
import json, random, os
random.seed(42)  # reproducible

ratio = float('${SUBSET_RATIO}')
data_dir = '${DATA_PREFIX}'

for fname in ['sft_train_clip.json', 'sft_train_video.json']:
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)

    n = max(1, int(len(data) * ratio))
    random.shuffle(data)
    subset = data[:n]

    # Save subset to a separate file so the original is preserved
    out = path.replace('.json', f'_sub{int(ratio*100)}.json')
    with open(out, 'w') as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    print(f'  {fname}: {len(data)} -> {len(subset)} samples -> {out}')
"
    # Override the train file paths for training
    SUB_SUFFIX="_sub$($VENV_PYTHON -c "print(int(float('${SUBSET_RATIO}')*100))")"
    TRAIN_CLIP_DATA="$DATA_PREFIX/sft_train_clip${SUB_SUFFIX}.json"
    TRAIN_VIDEO_DATA="$DATA_PREFIX/sft_train_video${SUB_SUFFIX}.json"
else
    TRAIN_CLIP_DATA="$DATA_PREFIX/sft_train_clip.json"
    TRAIN_VIDEO_DATA="$DATA_PREFIX/sft_train_video.json"
fi

VAL_CLIP_DATA="$DATA_PREFIX/sft_val_clip.json"
VAL_VIDEO_DATA="$DATA_PREFIX/sft_val_video.json"

log "  Train clip data:  $TRAIN_CLIP_DATA"
log "  Train video data: $TRAIN_VIDEO_DATA"
log "  Val clip data:    $VAL_CLIP_DATA"
log "  Val video data:   $VAL_VIDEO_DATA"

# ════════════════════════════════════════════════════════════════════════════
# 6. COMMON TRAINING ARGS
# ════════════════════════════════════════════════════════════════════════════
# Video frame args (exactly one of fps/nframes)
if [ -n "$FPS" ]; then
    VIDEO_FRAME_ARGS="--fps $FPS"
else
    VIDEO_FRAME_ARGS="--nframes $NFRAMES"
fi

# Flash attention
if [ "$DISABLE_FLASH_ATTN2" = "1" ]; then
    FLASH_ATTN_FLAG="--disable_flash_attn2 True"
else
    FLASH_ATTN_FLAG="--disable_flash_attn2 False"
fi

COMMON_ARGS=(
    --bits "$BITS"
    --lora_enable True --vision_lora True --use_dora False
    --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT"
    --num_lora_modules -1
    --lora_namespan_exclude "['lm_head', 'embed_tokens']"
    --freeze_vision_tower True --freeze_llm True --freeze_merger True
    --bf16 True --fp16 False --tf32 True
    $FLASH_ATTN_FLAG
    --use_liger_kernel True
    --num_train_epochs "$NUM_EPOCHS"
    --per_device_train_batch_size "$BATCH_PER_DEVICE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate "$LR" --vision_lr "$VISION_LR" --merger_lr "$MERGER_LR"
    --weight_decay "$WEIGHT_DECAY" --warmup_ratio "$WARMUP_RATIO"
    --lr_scheduler_type "$LR_SCHEDULER"
    --video_min_pixels "$VIDEO_MIN_PIXELS" --video_max_pixels "$VIDEO_MAX_PIXELS"
    $VIDEO_FRAME_ARGS
    --max_seq_length 32768
    --gradient_checkpointing True
    --lazy_preprocess True --remove_unused_columns False
    --dataloader_num_workers 4
    --logging_steps "$LOGGING_STEPS"
    --save_strategy "$SAVE_STRATEGY"
    --save_steps "$SAVE_STEPS"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --eval_strategy "$EVAL_STRATEGY"
    --eval_steps "$EVAL_STEPS"
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
    --generation_max_new_tokens 256
    --prediction_loss_only False
    --report_to "$REPORT_TO"
    --image_folder dataset
)

# ════════════════════════════════════════════════════════════════════════════
# 7. STAGE 1: SFT on clips
# ════════════════════════════════════════════════════════════════════════════
log "=== 5. SFT Stage 1: Clips ==="

SFT_CLIP_OUT="$OUTPUT_ROOT/sft_clip_lora"

if [ -f "$SFT_CLIP_OUT/adapter_config.json" ]; then
    log "  sft_clip_lora already exists, skipping Stage 1."
    log "  (Delete $SFT_CLIP_OUT to retrain.)"
else
    $VENV_PYTHON -u src/train/train_sft.py \
        --model_id "$MODEL_ID" \
        --data_path "$TRAIN_CLIP_DATA" \
        --eval_path "$VAL_CLIP_DATA" \
        --output_dir "$SFT_CLIP_OUT" \
        "${COMMON_ARGS[@]}" 2>&1 | sed "s/^/[SFT-clip] /"
fi
log "Stage 1 (clips) complete."

# ════════════════════════════════════════════════════════════════════════════
# 8. MERGE Stage 1 LoRA → full model
# ════════════════════════════════════════════════════════════════════════════
log "=== 6. Merge Stage 1 (clips) ==="

SFT_CLIP_MERGED="$OUTPUT_ROOT/sft_clip_merged"

if [ -f "$SFT_CLIP_MERGED/config.json" ]; then
    log "  sft_clip_merged already exists, skipping merge."
else
    $VENV_PYTHON src/merge_lora.py \
        --model-path "$SFT_CLIP_OUT" \
        --model-base "$MODEL_ID" \
        --save-model-path "$SFT_CLIP_MERGED" \
        --safe-serialization
fi
log "Clip LoRA merged to $SFT_CLIP_MERGED"

# ════════════════════════════════════════════════════════════════════════════
# 9. STAGE 2: SFT on full videos (base = merged clip model)
# ════════════════════════════════════════════════════════════════════════════
log "=== 7. SFT Stage 2: Full Videos ==="

SFT_VIDEO_OUT="$OUTPUT_ROOT/sft_video_lora"
SFT_VIDEO_BASE="$SFT_CLIP_MERGED"

if [ -f "$SFT_VIDEO_OUT/adapter_config.json" ]; then
    log "  sft_video_lora already exists, skipping Stage 2."
    log "  (Delete $SFT_VIDEO_OUT to retrain.)"
else
    $VENV_PYTHON -u src/train/train_sft.py \
        --model_id "$SFT_VIDEO_BASE" \
        --data_path "$TRAIN_VIDEO_DATA" \
        --eval_path "$VAL_VIDEO_DATA" \
        --output_dir "$SFT_VIDEO_OUT" \
        "${COMMON_ARGS[@]}" 2>&1 | sed "s/^/[SFT-video] /"
fi
log "Stage 2 (full videos) complete."

# ════════════════════════════════════════════════════════════════════════════
# 10. MERGE Stage 2 LoRA → final model
# ════════════════════════════════════════════════════════════════════════════
log "=== 8. Merge Stage 2 (final model) ==="

SFT_VIDEO_MERGED="$OUTPUT_ROOT/sft_video_merged"

if [ -f "$SFT_VIDEO_MERGED/config.json" ]; then
    log "  sft_video_merged already exists, skipping merge."
else
    $VENV_PYTHON src/merge_lora.py \
        --model-path "$SFT_VIDEO_OUT" \
        --model-base "$SFT_VIDEO_BASE" \
        --save-model-path "$SFT_VIDEO_MERGED" \
        --safe-serialization
fi

# ════════════════════════════════════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════════════════════════════════════
log "=========================================="
log "SFT pipeline complete!"
log "  Stage 1 adapter:    $SFT_CLIP_OUT"
log "  Stage 1 merged:     $SFT_CLIP_MERGED"
log "  Stage 2 adapter:    $SFT_VIDEO_OUT"
log "  FINAL merged model: $SFT_VIDEO_MERGED"
log "=========================================="
if [ "$SUBSET_RATIO" != "1.0" ] && [ "$SUBSET_RATIO" != "1" ]; then
    log "NOTE: Trained on SUBSET_RATIO=${SUBSET_RATIO} of the data."
    log "      For full training, re-run with SUBSET_RATIO=1.0"
fi