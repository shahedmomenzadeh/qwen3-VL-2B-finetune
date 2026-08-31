#!/bin/bash
# train.sh — Full training pipeline for Qwen3-VL-2B LoRA SFT + GRPO
#
# Stages:
#   1. SFT on dataset_sft (clips + full videos) → merge → output/sft_merged
#   2. GRPO on dataset_grpo (YouTube + phase tasks) → merge → output/grpo_merged
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

# --- Datasets ---
# The separated datasets contain SFT and GRPO annotations independently.
# Prepared JSONs receive a split prefix so each stage can use its own dataset root.
SFT_DATASET_ROOT="${SFT_DATASET_ROOT:-dataset_sft}"
GRPO_DATASET_ROOT="${GRPO_DATASET_ROOT:-dataset_grpo}"
DATA_PREFIX="${DATA_PREFIX:-data}"

for split in Train Validation; do
    [ -d "$SFT_DATASET_ROOT/$split" ] || err "$SFT_DATASET_ROOT/$split missing"
    [ -d "$GRPO_DATASET_ROOT/$split" ] || err "$GRPO_DATASET_ROOT/$split missing"
done

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

prepared_data_needs_refresh() {
    local dataset_root="$1"
    local prepared_path="$2"
    if [ ! -f "$prepared_path" ]; then
        return 0
    fi
    [ -n "$(find "$dataset_root/Train" "$dataset_root/Validation" -type f -newer "$prepared_path" -print -quit 2>/dev/null)" ]
}

SFT_PREPARE=0
for prepared_file in sft_train_dataset_sft.json sft_val_dataset_sft.json; do
    if prepared_data_needs_refresh "$SFT_DATASET_ROOT" "$DATA_PREFIX/$prepared_file"; then
        SFT_PREPARE=1
    fi
done

# Re-generate prepared JSONs (optional — skip if already present)
if [ "$SFT_PREPARE" = "1" ] || [ "${FORCE_REPREPARE:-0}" = "1" ]; then
    log "Running prepare_sft.py (clips + full videos combined)..."
    $VENV_PYTHON data/prepare_sft.py --input-dir "$SFT_DATASET_ROOT/Train" --output "$DATA_PREFIX/sft_train_dataset_sft.json" --data-type all
    $VENV_PYTHON data/prepare_sft.py --input-dir "$SFT_DATASET_ROOT/Validation" --output "$DATA_PREFIX/sft_val_dataset_sft.json" --data-type all
fi

GRPO_PREPARE=0
for prepared_file in grpo_train_dataset_grpo.json grpo_val_dataset_grpo.json; do
    if prepared_data_needs_refresh "$GRPO_DATASET_ROOT" "$DATA_PREFIX/$prepared_file"; then
        GRPO_PREPARE=1
    fi
done

if [ "$GRPO_PREPARE" = "1" ] || [ "${FORCE_REPREPARE:-0}" = "1" ]; then
    log "Running prepare_grpo.py ..."
    $VENV_PYTHON data/prepare_grpo.py --input-dir "$GRPO_DATASET_ROOT/Train" --output "$DATA_PREFIX/grpo_train_dataset_grpo.json" --data-type all
    $VENV_PYTHON data/prepare_grpo.py --input-dir "$GRPO_DATASET_ROOT/Validation" --output "$DATA_PREFIX/grpo_val_dataset_grpo.json" --data-type all
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
# 4. STAGE 1: SFT on all data (clips + full videos → merged SFT base)
# ────────────────────────────────────────────────────────────
SFT_BASE="$MODEL_NAME"
SFT_OUT="${OUTPUT_ROOT}/sft_lora"

if [ ! -f "${SFT_OUT}/adapter_config.json" ]; then
    run_training "SFT" \
        src/train/train_sft.py \
        --deepspeed "$DEEPSPEED_CONFIG" \
        --model_id "$SFT_BASE" \
        --data_path "$DATA_PREFIX/sft_train_dataset_sft.json" \
        --eval_path "$DATA_PREFIX/sft_val_dataset_sft.json" \
        --image_folder "$SFT_DATASET_ROOT" \
        --output_dir "$SFT_OUT" \
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
    log "SFT already exists at $SFT_OUT, skipping."
fi

# ────────────────────────────────────────────────────────────
# 5. MERGE SFT → full model (base for GRPO)
# ────────────────────────────────────────────────────────────
SFT_MERGED="${OUTPUT_ROOT}/sft_merged"

if [ ! -f "${SFT_MERGED}/config.json" ]; then
    log "Merging SFT LoRA..."
    $VENV_PYTHON src/merge_lora.py \
        --model-path "$SFT_OUT" \
        --model-base "$SFT_BASE" \
        --save-model-path "$SFT_MERGED" \
        --safe-serialization
    log "SFT merged to $SFT_MERGED"
else
    log "SFT merged already exists at $SFT_MERGED, skipping."
fi
# ────────────────────────────────────────────────────────────
# 6. STAGE 2: GRPO on dataset_grpo (base = SFT merged)
# ────────────────────────────────────────────────────────────
GRPO_BASE="$SFT_MERGED"
GRPO_OUT="${OUTPUT_ROOT}/grpo_lora"

if [ ! -f "${GRPO_OUT}/adapter_config.json" ]; then
    run_training "GRPO" \
        src/train/train_grpo.py \
        --deepspeed "$DEEPSPEED_CONFIG" \
        --model_id "$GRPO_BASE" \
        --data_path "$DATA_PREFIX/grpo_train_dataset_grpo.json" \
        --eval_path "$DATA_PREFIX/grpo_val_dataset_grpo.json" \
        --image_folder "$GRPO_DATASET_ROOT" \
        --output_dir "$GRPO_OUT" \
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
        --eval_strategy steps \
        --eval_steps 500 \
        --per_device_eval_batch_size 1 \
        --save_strategy epoch \
        --save_total_limit 3 \
        --report_to "$REPORT_TO"
else
    log "GRPO already exists at $GRPO_OUT, skipping."
fi


# ────────────────────────────────────────────────────────────
# 7. FINAL MERGE: GRPO adapter
# ────────────────────────────────────────────────────────────
GRPO_MERGED="${OUTPUT_ROOT}/grpo_merged"

if [ ! -f "${GRPO_MERGED}/config.json" ]; then
    log "Final merge: GRPO LoRA..."
    $VENV_PYTHON src/merge_lora.py \
        --model-path "$GRPO_OUT" \
        --model-base "$GRPO_BASE" \
        --save-model-path "$GRPO_MERGED" \
        --safe-serialization
    log "Final model at $GRPO_MERGED"
fi

# ────────────────────────────────────────────────────────────
# DONE
# ────────────────────────────────────────────────────────────
log "======================================================"
log "Training pipeline complete!"
log "Final model: $GRPO_MERGED"
log "Intermediate outputs in: $OUTPUT_ROOT/"
log "======================================================"
