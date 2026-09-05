#!/bin/bash
# train.sh — Complete Two-Stage Pipeline for Qwen3-VL-2B (SFT -> Merge -> GRPO -> Merge)
#
# Stage 1: Supervised Fine-Tuning (SFT) on dataset_sft (clips + full videos)
# Stage 2: Group Relative Policy Optimization (GRPO) on dataset_grpo (MCQ + Phase tasks)
#
# Override defaults via env vars, e.g.:
#   BITS=16 NFRAMES=48 bash train.sh

set -euo pipefail

log() { echo -e "\033[1;32m[train.sh]\033[0m $1"; }
warn() { echo -e "\033[1;33m[train.sh]\033[0m $1"; }
err() { echo -e "\033[1;31m[train.sh] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Setup Venv ─────────────────────────────────────────────────────────────
if [ -f ".venv/bin/python" ]; then
    VENV_PYTHON=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    VENV_PYTHON=".venv/Scripts/python.exe"
else
    err "Cannot find python in .venv — please run setup.sh"
fi

export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"
export TOKENIZERS_PARALLELISM=false

# ── 2. Config ─────────────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
SFT_DATASET_ROOT="${SFT_DATASET_ROOT:-dataset_sft}"
GRPO_DATASET_ROOT="${GRPO_DATASET_ROOT:-dataset_grpo}"
DATA_PREFIX="${DATA_PREFIX:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"

# Auto-restore SFT data from HF Hub on a fresh machine (no-op when present)
SFT_DATASET_ROOT="$SFT_DATASET_ROOT" bash "$SCRIPT_DIR/scripts/ensure_dataset_sft.sh"

BITS="${BITS:-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"

LR="${LR:-1e-4}"
GRPO_LR="${GRPO_LR:-1e-4}"
VISION_LR="${VISION_LR:-2e-6}"
MERGER_LR="${MERGER_LR:-1e-5}"
EPOCHS_SFT="${EPOCHS_SFT:-2}"
EPOCHS_GRPO="${EPOCHS_GRPO:-1}"

NFRAMES="${NFRAMES:-60}"
FPS="${FPS:-}"
VIDEO_MIN_PIXELS=$((128 * 32 * 32))   # 131072
VIDEO_MAX_PIXELS=$((256 * 32 * 32))   # 262144

BETA="${BETA:-0.04}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-1.0}"
REPORT_TO="${REPORT_TO:-tensorboard}"

log "Configuration:"
log "  MODEL_NAME=$MODEL_NAME"
log "  BITS=$BITS BATCH=$BATCH_PER_DEVICE GRAD_ACCUM=$GRAD_ACCUM NFRAMES=$NFRAMES"
log "  SFT_LR=$LR GRPO_LR=$GRPO_LR VISION_LR=$VISION_LR MERGER_LR=$MERGER_LR"

# ── 3. Data Prep ──────────────────────────────────────────────────────────────
mkdir -p "$DATA_PREFIX" "$OUTPUT_ROOT"

if [ ! -f "$DATA_PREFIX/sft_train_dataset_sft.json" ] || [ ! -f "$DATA_PREFIX/sft_val_dataset_sft.json" ]; then
    log "Preparing SFT JSON datasets..."
    $VENV_PYTHON data/prepare_sft.py --input-dir "$SFT_DATASET_ROOT/Train" --output "$DATA_PREFIX/sft_train_dataset_sft.json" --data-type all
    $VENV_PYTHON data/prepare_sft.py --input-dir "$SFT_DATASET_ROOT/Validation" --output "$DATA_PREFIX/sft_val_dataset_sft.json" --data-type all
fi

if [ ! -f "$DATA_PREFIX/grpo_train_dataset_grpo.json" ] || [ ! -f "$DATA_PREFIX/grpo_val_dataset_grpo.json" ]; then
    log "Preparing GRPO JSON datasets..."
    $VENV_PYTHON data/prepare_grpo.py --input-dir "$GRPO_DATASET_ROOT/Train" --output "$DATA_PREFIX/grpo_train_dataset_grpo.json" --data-type all
    $VENV_PYTHON data/prepare_grpo.py --input-dir "$GRPO_DATASET_ROOT/Validation" --output "$DATA_PREFIX/grpo_val_dataset_grpo.json" --data-type all
fi

VIDEO_FRAME_ARGS=""
if [ -n "$FPS" ]; then
    VIDEO_FRAME_ARGS="--fps $FPS"
else
    VIDEO_FRAME_ARGS="--nframes $NFRAMES"
fi

# ── 4. Stage 1: SFT ───────────────────────────────────────────────────────────
SFT_BASE="$MODEL_NAME"
SFT_OUT="${OUTPUT_ROOT}/sft_lora"
SFT_MERGED="${OUTPUT_ROOT}/sft_merged"

if [ ! -f "${SFT_OUT}/adapter_config.json" ]; then
    log "=== Stage 1: SFT Training ==="
    $VENV_PYTHON src/train/train_sft.py \
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
        --lora_dropout 0.05 \
        --num_lora_modules -1 \
        --lora_namespan_exclude "['lm_head', 'embed_tokens', 'merger', 'pos_embed']" \
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
        --save_steps 300 \
        --save_total_limit 3 \
        --eval_strategy steps \
        --eval_steps 300 \
        --per_device_eval_batch_size 1 \
        --generation_max_new_tokens 256 \
        --prediction_loss_only False \
        --report_to "$REPORT_TO"
else
    log "SFT checkpoint already exists at $SFT_OUT, skipping SFT."
fi

# ── 5. Merge SFT Model ────────────────────────────────────────────────────────
if [ ! -f "${SFT_MERGED}/config.json" ]; then
    log "=== Merging SFT Adapter ==="
    $VENV_PYTHON src/merge_lora.py \
        --model-path "$SFT_OUT" \
        --model-base "$SFT_BASE" \
        --save-model-path "$SFT_MERGED" \
        --safe-serialization
else
    log "Merged SFT already exists at $SFT_MERGED, skipping merge."
fi

# ── 6. Stage 2: GRPO ──────────────────────────────────────────────────────────
GRPO_BASE="$SFT_MERGED"
GRPO_OUT="${OUTPUT_ROOT}/grpo_lora"
GRPO_MERGED="${OUTPUT_ROOT}/grpo_merged"

if [ ! -f "${GRPO_OUT}/adapter_config.json" ]; then
    log "=== Stage 2: GRPO Training ==="
    $VENV_PYTHON src/train/train_grpo.py \
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
        --lora_dropout 0.0 \
        --num_lora_modules -1 \
        --lora_namespan_exclude "['lm_head', 'embed_tokens', 'merger', 'pos_embed']" \
        --freeze_vision_tower True \
        --freeze_llm True \
        --freeze_merger False \
        --bf16 True --fp16 False --tf32 True \
        --disable_flash_attn2 False \
        --use_liger_kernel True \
        --num_train_epochs "$EPOCHS_GRPO" \
        --num_generations "$NUM_GENERATIONS" \
        --per_device_train_batch_size "$BATCH_PER_DEVICE" \
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
        --eval_steps 300 \
        --per_device_eval_batch_size 1 \
        --save_strategy steps \
        --save_steps 300 \
        --save_total_limit 3 \
        --report_to "$REPORT_TO"
else
    log "GRPO checkpoint already exists at $GRPO_OUT, skipping GRPO."
fi

# ── 7. Final Merge GRPO Model ──────────────────────────────────────────────────
if [ ! -f "${GRPO_MERGED}/config.json" ]; then
    log "=== Final Merge: GRPO Adapter ==="
    $VENV_PYTHON src/merge_lora.py \
        --model-path "$GRPO_OUT" \
        --model-base "$GRPO_BASE" \
        --save-model-path "$GRPO_MERGED" \
        --safe-serialization
fi

log "======================================================"
log "Full Pipeline Complete!"
log "Final Model: $GRPO_MERGED"
log "======================================================"
