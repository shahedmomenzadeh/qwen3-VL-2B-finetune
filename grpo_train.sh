#!/bin/bash
# grpo_train.sh — Full GRPO training for Qwen3-VL-2B (QLoRA)
#
# Hardware: 1× 48GB GPU (or multi-GPU), ~20-30GB peak
# Dataset: dataset_grpo/Train + dataset_grpo/Validation (YouTube MCQs + Phase tasks)
# Uses 100% deterministic rule-based JSON rewards (reward_funcs.py)
#
# Progression: SFT model (output/sft_merged or base) → GRPO LoRA → Merge → output/grpo_merged

set -euo pipefail

log()  { echo -e "\033[1;32m[grpo]\033[0m $1"; }
warn() { echo -e "\033[1;33m[grpo]\033[0m $1"; }
err()  { echo -e "\033[1;31m[grpo] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Env Check ──────────────────────────────────────────────────────────────
if [ -f ".venv/bin/python" ]; then
    VENV_PYTHON=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    VENV_PYTHON=".venv/Scripts/python.exe"
else
    err ".venv not found — run bash setup.sh or bash train_sft.sh first"
fi

export HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# ── 2. Base Model (Default: merged SFT) ─────────────────────────────────────────
FULL_SFT_MERGED="$SCRIPT_DIR/output/sft_merged"
DEFAULT_BASE="Qwen/Qwen3-VL-2B-Instruct"

if [ -n "${GRPO_BASE:-}" ]; then
    MODEL_ID="$GRPO_BASE"
elif [ -f "$FULL_SFT_MERGED/config.json" ]; then
    MODEL_ID="$FULL_SFT_MERGED"
    log "Using merged SFT checkpoint: $MODEL_ID"
else
    MODEL_ID="${MODEL_ID:-$DEFAULT_BASE}"
    warn "Full SFT not found at $FULL_SFT_MERGED — initializing from $MODEL_ID"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"
DATA_PREFIX="${DATA_PREFIX:-data}"
GRPO_DATASET_ROOT="${GRPO_DATASET_ROOT:-dataset_grpo}"

# ── 3. Hyperparameters & Configuration ────────────────────────────────────────
BITS="${BITS:-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"

BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_COMP="${MAX_COMP:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"

NFRAMES="${NFRAMES:-60}"
FPS="${FPS:-}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-$((128 * 32 * 32))}"  # 131072
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-$((256 * 32 * 32))}"  # 262144

BETA="${BETA:-0.04}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-1.0}"
LR="${LR:-1e-4}"
VISION_LR="${VISION_LR:-2e-6}"
MERGER_LR="${MERGER_LR:-1e-5}"

USE_LIGER="${USE_LIGER:-1}"
DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-0}"
FORCE_REPREPARE="${FORCE_REPREPARE:-0}"
REPORT_TO="${REPORT_TO:-tensorboard}"

log "MODEL_ID=$MODEL_ID"
log "BITS=$BITS RANK=$LORA_RANK BATCH=${BATCH_PER_DEVICE}x${GRAD_ACCUM} NG=$NUM_GENERATIONS MAX_COMP=$MAX_COMP EPOCHS=$NUM_EPOCHS"
log "NFRAMES=$NFRAMES VIDEO_MIN=$VIDEO_MIN_PIXELS VIDEO_MAX=$VIDEO_MAX_PIXELS"
log "OUTPUT=$OUTPUT_ROOT/grpo_lora -> $OUTPUT_ROOT/grpo_merged"

# ── 4. Data Preparation ────────────────────────────────────────────────────────
for split in Train Validation; do
    [ -d "$GRPO_DATASET_ROOT/$split" ] || err "$GRPO_DATASET_ROOT/$split missing"
done

TRAIN_JSON="$DATA_PREFIX/grpo_train_dataset_grpo.json"
VAL_JSON="$DATA_PREFIX/grpo_val_dataset_grpo.json"

if [ ! -f "$TRAIN_JSON" ] || [ "$FORCE_REPREPARE" = "1" ]; then
    log "Preparing GRPO train dataset..."
    $VENV_PYTHON data/prepare_grpo.py --input-dir "$GRPO_DATASET_ROOT/Train" --output "$TRAIN_JSON" --split train --data-type all
fi

if [ ! -f "$VAL_JSON" ] || [ "$FORCE_REPREPARE" = "1" ]; then
    log "Preparing GRPO val dataset..."
    $VENV_PYTHON data/prepare_grpo.py --input-dir "$GRPO_DATASET_ROOT/Validation" --output "$VAL_JSON" --split val --data-type all
fi

# ── 5. Run GRPO Training ───────────────────────────────────────────────────────
VIDEO_ARGS=""
if [ -n "$FPS" ]; then
    VIDEO_ARGS="--fps $FPS"
else
    VIDEO_ARGS="--nframes $NFRAMES"
fi

GRPO_OUT="$OUTPUT_ROOT/grpo_lora"
log "=== Launching GRPO Training ==="

$VENV_PYTHON src/train/train_grpo.py \
    --model_id "$MODEL_ID" \
    --data_path "$TRAIN_JSON" \
    --eval_path "$VAL_JSON" \
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
    --lora_namespan_exclude "['lm_head', 'embed_tokens', 'merger', 'pos_embed']" \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True --fp16 False --tf32 True \
    --disable_flash_attn2 $([ "$DISABLE_FLASH_ATTN2" = "1" ] && echo True || echo False) \
    --use_liger_kernel False \
    --num_train_epochs "$NUM_EPOCHS" \
    --num_generations "$NUM_GENERATIONS" \
    --per_device_train_batch_size "$BATCH_PER_DEVICE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --max_completion_length "$MAX_COMP" \
    --learning_rate "$LR" \
    --vision_lr "$VISION_LR" \
    --merger_lr "$MERGER_LR" \
    --beta "$BETA" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --weight_decay 0.0 \
    --warmup_steps 0 \
    --lr_scheduler_type constant \
    --video_min_pixels "$VIDEO_MIN_PIXELS" \
    --video_max_pixels "$VIDEO_MAX_PIXELS" \
    $VIDEO_ARGS \
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

log "GRPO training complete: $GRPO_OUT"

# ── 6. Merge GRPO Adapter ──────────────────────────────────────────────────────
GRPO_MERGED="$OUTPUT_ROOT/grpo_merged"
log "Merging GRPO LoRA into $GRPO_MERGED..."

$VENV_PYTHON src/merge_lora.py \
    --model-path "$GRPO_OUT" \
    --model-base "$MODEL_ID" \
    --save-model-path "$GRPO_MERGED" \
    --safe-serialization

log "=== Final GRPO Model Ready at $GRPO_MERGED ==="
