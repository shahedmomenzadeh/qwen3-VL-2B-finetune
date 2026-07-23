#!/bin/bash
# train.sh — Full training pipeline for Qwen3-VL-2B LoRA SFT + GRPO
#
# Stages:
#   1. SFT on clips    → merge → output/sft_clip_merged
#   2. SFT on videos   → merge → output/sft_video_merged
#   3. GRPO on clips   (base = sft_video_merged)
#   4. GRPO on videos  (base = merged clip GRPO)  → merge → output/grpo_video_merged
#
# Override any param via env:  MODEL_NAME=... BITS=... ./train.sh

set -euo pipefail

log() { echo -e "\033[1;32m[train.sh]\033[0m $1"; }
warn() { echo -e "\033[1;33m[train.sh]\033[0m $1"; }
err() { echo -e "\033[1;31m[train.sh] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ────────────────────────────────────────────────────────────
# 1. SETUP: UV + VIRTUAL ENVIRONMENT
# ────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
log "uv version: $(uv --version)"

if [ ! -d ".venv" ]; then
    log "Creating virtual environment (uv sync from pyproject.toml)..."
    uv sync
fi

# Detect python inside the venv
if [ -f ".venv/bin/python" ]; then
    VENV_PYTHON=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    VENV_PYTHON=".venv/Scripts/python.exe"
else
    err "Cannot find python in .venv"
fi
log "Python: $($VENV_PYTHON --version)"

# GPU check
if ! $VENV_PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    warn "CUDA not available — training will be very slow or fail."
fi

export PYTHONPATH="src:${PYTHONPATH:-}"

# ────────────────────────────────────────────────────────────
# 2. CONFIGURATION (all overridable via env vars)
# ────────────────────────────────────────────────────────────
HF_TOKEN="${HF_TOKEN:-}"
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
fi
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"

# --- Model ---
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"

# --- Dataset ---
# The prepared JSONs store video paths relative to the split directory
# (e.g., YT_ID/clip.mp4 relative to dataset/Train/).
# We post-process them to include the split prefix (Train/YT_ID/clip.mp4)
# so that a single IMAGE_FOLDER=dataset works for both train and eval.
DATA_PREFIX="${DATA_PREFIX:-data}"

# --- Output root ---
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"

# --- Hyperparameters ---
BITS="${BITS:-4}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_DEVICES="${NUM_DEVICES:-1}"
GLOBAL_BATCH_SIZE=$((BATCH_PER_DEVICE * GRAD_ACCUM * NUM_DEVICES))

LR="${LR:-1e-4}"
GRPO_LR="${GRPO_LR:-5e-6}"
VISION_LR="${VISION_LR:-2e-6}"
MERGER_LR="${MERGER_LR:-1e-5}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
EPOCHS_SFT="${EPOCHS_SFT:-2}"
EPOCHS_GRPO="${EPOCHS_GRPO:-1}"

# --- Video ---
NFRAMES="${NFRAMES:-60}"
FPS="${FPS:-}"
VIDEO_MIN_PIXELS=$((128 * 32 * 32))   # 131072
VIDEO_MAX_PIXELS=$((256 * 32 * 32))   # 262144

# --- GRPO specific ---
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
BETA="${BETA:-0.04}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-1.0}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"

# --- DeepSpeed ---
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero3_offload.json}"

# --- Logging ---
REPORT_TO="${REPORT_TO:-tensorboard}"

log "Configuration:"
log "  MODEL_NAME=$MODEL_NAME"
log "  BITS=$BITS  BATCH/DEV=$BATCH_PER_DEVICE  GRAD_ACCUM=$GRAD_ACCUM  DEVICES=$NUM_DEVICES"
log "  GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
log "  LR=$LR  VISION_LR=$VISION_LR  MERGER_LR=$MERGER_LR  EPOCHS_SFT=$EPOCHS_SFT  EPOCHS_GRPO=$EPOCHS_GRPO"
log "  NFRAMES=$NFRAMES  FPS=${FPS:-unset}"

# ────────────────────────────────────────────────────────────
# 3. DATA PREPARATION
# ────────────────────────────────────────────────────────────
log "Preparing data..."

# Re-generate prepared JSONs (optional — skip if already present)
if [ ! -f "$DATA_PREFIX/sft_train_clip.json" ] || [ "${FORCE_REPREPARE:-0}" = "1" ]; then
    log "Running prepare_sft.py ..."
    $VENV_PYTHON data/prepare_sft.py --input-dir dataset/Train --output "$DATA_PREFIX/sft_train_clip.json" --data-type clip
    $VENV_PYTHON data/prepare_sft.py --input-dir dataset/Train --output "$DATA_PREFIX/sft_train_video.json" --data-type full_video
    $VENV_PYTHON data/prepare_sft.py --input-dir dataset/Validation --output "$DATA_PREFIX/sft_val_clip.json" --data-type clip
    $VENV_PYTHON data/prepare_sft.py --input-dir dataset/Validation --output "$DATA_PREFIX/sft_val_video.json" --data-type full_video
fi

if [ ! -f "$DATA_PREFIX/grpo_train_clip.json" ] || [ "${FORCE_REPREPARE:-0}" = "1" ]; then
    log "Running prepare_grpo.py ..."
    $VENV_PYTHON data/prepare_grpo.py --input-dir dataset/Train --output "$DATA_PREFIX/grpo_train_clip.json" --data-type clip
    $VENV_PYTHON data/prepare_grpo.py --input-dir dataset/Train --output "$DATA_PREFIX/grpo_train_video.json" --data-type full_video
    $VENV_PYTHON data/prepare_grpo.py --input-dir dataset/Validation --output "$DATA_PREFIX/grpo_val_clip.json" --data-type clip
    $VENV_PYTHON data/prepare_grpo.py --input-dir dataset/Validation --output "$DATA_PREFIX/grpo_val_video.json" --data-type full_video
fi

log "Data preparation done."

# ────────────────────────────────────────────────────────────
# Video frame args (exactly one of fps/nframes must be set)
# ────────────────────────────────────────────────────────────
VIDEO_FRAME_ARGS=""
if [ -n "$FPS" ]; then
    VIDEO_FRAME_ARGS="--fps $FPS"
else
    VIDEO_FRAME_ARGS="--nframes $NFRAMES"
fi

# ────────────────────────────────────────────────────────────
# Helper: run deepspeed training
# ────────────────────────────────────────────────────────────
run_training() {
    local STAGE="$1"
    shift
    log "=== Running $STAGE ==="
    $VENV_PYTHON -u "$@" 2>&1 | sed "s/^/[$STAGE] /"
    local EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        err "$STAGE failed with exit code $EXIT_CODE"
    fi
    log "=== $STAGE complete ==="
}

# ────────────────────────────────────────────────────────────
# 4. STAGE 1: SFT on clips
# ────────────────────────────────────────────────────────────
SFT_CLIP_OUT="${OUTPUT_ROOT}/sft_clip_lora"

if [ ! -f "${SFT_CLIP_OUT}/adapter_config.json" ]; then
    run_training "SFT-clip" \
        src/train/train_sft.py \
        --deepspeed "$DEEPSPEED_CONFIG" \
        --model_id "$MODEL_NAME" \
        --data_path "$DATA_PREFIX/sft_train_clip.json" \
        --eval_path "$DATA_PREFIX/sft_val_clip.json" \
        --image_folder dataset \
        --output_dir "$SFT_CLIP_OUT" \
        --bits "$BITS" \
        --lora_enable True \
        --vision_lora True \
        --use_dora False \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --num_lora_modules -1 \
        --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
        --freeze_vision_tower True \
        --freeze_llm True \
        --freeze_merger False \
        --bf16 True --fp16 False --tf32 True \
        --disable_flash_attn2 False \
        --use_liger_kernel True \
        --num_train_epochs "$EPOCHS_SFT" \
        --per_device_train_batch_size "$BATCH_PER_DEVICE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LR" \
        --vision_lr "$VISION_LR" \
        --merger_lr "$MERGER_LR" \
        --weight_decay 0.1 \
        --warmup_steps 10 \
        --lr_scheduler_type cosine \
        --video_min_pixels "$VIDEO_MIN_PIXELS" \
        --video_max_pixels "$VIDEO_MAX_PIXELS" \
        $VIDEO_FRAME_ARGS \
        --max_seq_length 32768 \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --remove_unused_columns False \
        --dataloader_num_workers 4 \
        --logging_steps 1 \
        --save_strategy steps \
        --save_steps 500 \
        --save_total_limit 3 \
        --eval_strategy steps \
        --eval_steps 500 \
        --per_device_eval_batch_size 1 \
        --generation_max_new_tokens 256 \
        --prediction_loss_only False \
        --report_to "$REPORT_TO"
else
    log "SFT-clip already exists at $SFT_CLIP_OUT, skipping."
fi

# ────────────────────────────────────────────────────────────
# 5. MERGE SFT clip
# ────────────────────────────────────────────────────────────
SFT_CLIP_MERGED="${OUTPUT_ROOT}/sft_clip_merged"

if [ ! -f "${SFT_CLIP_MERGED}/config.json" ]; then
    log "Merging SFT clip LoRA..."
    $VENV_PYTHON src/merge_lora.py \
        --model-path "$SFT_CLIP_OUT" \
        --model-base "$MODEL_NAME" \
        --save-model-path "$SFT_CLIP_MERGED" \
        --safe-serialization
    log "SFT clip merged to $SFT_CLIP_MERGED"
else
    log "SFT clip merged already exists at $SFT_CLIP_MERGED, skipping."
fi

# ────────────────────────────────────────────────────────────
# 6. STAGE 2: SFT on videos (base = merged clip)
# ────────────────────────────────────────────────────────────
SFT_VIDEO_OUT="${OUTPUT_ROOT}/sft_video_lora"
SFT_VIDEO_BASE="$SFT_CLIP_MERGED"

if [ ! -f "${SFT_VIDEO_OUT}/adapter_config.json" ]; then
    run_training "SFT-video" \
        src/train/train_sft.py \
        --deepspeed "$DEEPSPEED_CONFIG" \
        --model_id "$SFT_VIDEO_BASE" \
        --data_path "$DATA_PREFIX/sft_train_video.json" \
        --eval_path "$DATA_PREFIX/sft_val_video.json" \
        --image_folder dataset \
        --output_dir "$SFT_VIDEO_OUT" \
        --bits "$BITS" \
        --lora_enable True \
        --vision_lora True \
        --use_dora False \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --num_lora_modules -1 \
        --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
        --freeze_vision_tower True \
        --freeze_llm True \
        --freeze_merger False \
        --bf16 True --fp16 False --tf32 True \
        --disable_flash_attn2 False \
        --use_liger_kernel True \
        --num_train_epochs "$EPOCHS_SFT" \
        --per_device_train_batch_size "$BATCH_PER_DEVICE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LR" \
        --vision_lr "$VISION_LR" \
        --merger_lr "$MERGER_LR" \
        --weight_decay 0.1 \
        --warmup_steps 10 \
        --lr_scheduler_type cosine \
        --video_min_pixels "$VIDEO_MIN_PIXELS" \
        --video_max_pixels "$VIDEO_MAX_PIXELS" \
        $VIDEO_FRAME_ARGS \
        --max_seq_length 32768 \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --remove_unused_columns False \
        --dataloader_num_workers 4 \
        --logging_steps 1 \
        --save_strategy steps \
        --save_steps 500 \
        --save_total_limit 3 \
        --eval_strategy steps \
        --eval_steps 500 \
        --per_device_eval_batch_size 1 \
        --generation_max_new_tokens 256 \
        --prediction_loss_only False \
        --report_to "$REPORT_TO"
else
    log "SFT-video already exists at $SFT_VIDEO_OUT, skipping."
fi

# ────────────────────────────────────────────────────────────
# 7. MERGE SFT video
# ────────────────────────────────────────────────────────────
SFT_VIDEO_MERGED="${OUTPUT_ROOT}/sft_video_merged"

if [ ! -f "${SFT_VIDEO_MERGED}/config.json" ]; then
    log "Merging SFT video LoRA..."
    $VENV_PYTHON src/merge_lora.py \
        --model-path "$SFT_VIDEO_OUT" \
        --model-base "$SFT_VIDEO_BASE" \
        --save-model-path "$SFT_VIDEO_MERGED" \
        --safe-serialization
    log "SFT video merged to $SFT_VIDEO_MERGED"
else
    log "SFT video merged already exists at $SFT_VIDEO_MERGED, skipping."
fi

# ────────────────────────────────────────────────────────────
# 8. STAGE 3: GRPO on clips (base = merged SFT video)
# ────────────────────────────────────────────────────────────
GRPO_CLIP_BASE="$SFT_VIDEO_MERGED"
GRPO_CLIP_OUT="${OUTPUT_ROOT}/grpo_clip_lora"

if [ ! -f "${GRPO_CLIP_OUT}/adapter_config.json" ]; then
    run_training "GRPO-clip" \
        src/train/train_grpo.py \
        --deepspeed "$DEEPSPEED_CONFIG" \
        --model_id "$GRPO_CLIP_BASE" \
        --data_path "$DATA_PREFIX/grpo_train_clip.json" \
        --image_folder dataset \
        --output_dir "$GRPO_CLIP_OUT" \
        --bits "$BITS" \
        --lora_enable True \
        --vision_lora True \
        --use_dora False \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --num_lora_modules -1 \
        --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
        --freeze_vision_tower True \
        --freeze_llm True \
        --freeze_merger False \
        --bf16 True --fp16 False --tf32 True \
        --disable_flash_attn2 False \
        --use_liger_kernel True \
        --loss_type dapo \
        --num_train_epochs "$EPOCHS_GRPO" \
        --num_generations "$NUM_GENERATIONS" \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --max_completion_length "$MAX_COMPLETION_LENGTH" \
        --learning_rate "$GRPO_LR" \
        --vision_lr "$VISION_LR" \
        --merger_lr "$MERGER_LR" \
        --beta "$BETA" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --weight_decay 0.1 \
        --warmup_steps 10 \
        --lr_scheduler_type cosine \
        --video_min_pixels "$VIDEO_MIN_PIXELS" \
        --video_max_pixels "$VIDEO_MAX_PIXELS" \
        $VIDEO_FRAME_ARGS \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --remove_unused_columns False \
        --dataloader_num_workers 4 \
        --logging_steps 1 \
        --save_strategy epoch \
        --save_total_limit 3 \
        --report_to "$REPORT_TO"
else
    log "GRPO-clip already exists at $GRPO_CLIP_OUT, skipping."
fi

# Merge GRPO clip (optional intermediate merge)
GRPO_CLIP_MERGED="${OUTPUT_ROOT}/grpo_clip_merged"

if [ ! -f "${GRPO_CLIP_MERGED}/config.json" ]; then
    log "Merging GRPO clip LoRA..."
    $VENV_PYTHON src/merge_lora.py \
        --model-path "$GRPO_CLIP_OUT" \
        --model-base "$GRPO_CLIP_BASE" \
        --save-model-path "$GRPO_CLIP_MERGED" \
        --safe-serialization
    log "GRPO clip merged to $GRPO_CLIP_MERGED"
fi

# ────────────────────────────────────────────────────────────
# 9. STAGE 4: GRPO on videos (base = merged GRPO clip)
# ────────────────────────────────────────────────────────────
GRPO_VIDEO_BASE="${GRPO_CLIP_MERGED}"
GRPO_VIDEO_OUT="${OUTPUT_ROOT}/grpo_video_lora"

if [ ! -f "${GRPO_VIDEO_OUT}/adapter_config.json" ]; then
    run_training "GRPO-video" \
        src/train/train_grpo.py \
        --deepspeed "$DEEPSPEED_CONFIG" \
        --model_id "$GRPO_VIDEO_BASE" \
        --data_path "$DATA_PREFIX/grpo_train_video.json" \
        --image_folder dataset \
        --output_dir "$GRPO_VIDEO_OUT" \
        --bits "$BITS" \
        --lora_enable True \
        --vision_lora True \
        --use_dora False \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --num_lora_modules -1 \
        --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
        --freeze_vision_tower True \
        --freeze_llm True \
        --freeze_merger False \
        --bf16 True --fp16 False --tf32 True \
        --disable_flash_attn2 False \
        --use_liger_kernel True \
        --loss_type dapo \
        --num_train_epochs "$EPOCHS_GRPO" \
        --num_generations "$NUM_GENERATIONS" \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --max_completion_length "$MAX_COMPLETION_LENGTH" \
        --learning_rate "$GRPO_LR" \
        --vision_lr "$VISION_LR" \
        --merger_lr "$MERGER_LR" \
        --beta "$BETA" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --weight_decay 0.1 \
        --warmup_steps 10 \
        --lr_scheduler_type cosine \
        --video_min_pixels "$VIDEO_MIN_PIXELS" \
        --video_max_pixels "$VIDEO_MAX_PIXELS" \
        $VIDEO_FRAME_ARGS \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --remove_unused_columns False \
        --dataloader_num_workers 4 \
        --logging_steps 1 \
        --save_strategy epoch \
        --save_total_limit 3 \
        --report_to "$REPORT_TO"
else
    log "GRPO-video already exists at $GRPO_VIDEO_OUT, skipping."
fi

# ────────────────────────────────────────────────────────────
# 10. FINAL MERGE
# ────────────────────────────────────────────────────────────
GRPO_VIDEO_MERGED="${OUTPUT_ROOT}/grpo_video_merged"

if [ ! -f "${GRPO_VIDEO_MERGED}/config.json" ]; then
    log "Final merge: GRPO video LoRA..."
    $VENV_PYTHON src/merge_lora.py \
        --model-path "$GRPO_VIDEO_OUT" \
        --model-base "$GRPO_VIDEO_BASE" \
        --save-model-path "$GRPO_VIDEO_MERGED" \
        --safe-serialization
    log "Final model at $GRPO_VIDEO_MERGED"
fi

# ────────────────────────────────────────────────────────────
# DONE
# ────────────────────────────────────────────────────────────
log "======================================================"
log "Training pipeline complete!"
log "Final model: $GRPO_VIDEO_MERGED"
log "Intermediate outputs in: $OUTPUT_ROOT/"
log "======================================================"
