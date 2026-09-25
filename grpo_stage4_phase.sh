#!/bin/bash
# grpo_stage4_phase.sh — Targeted Stage 4 GRPO Training for Cataract Surgical Phase Specialization
#
# Hardware: 1× 24GB GPU (RTX 3090/4090)
# Dataset: data/grpo_train_stage4_phase.json (506 samples, Option A)
# Exploration: G=8 rollouts per prompt with GRPO_MICRO_PROMPTS=1
# Stacking: Appends to existing train.log and merges TensorBoard with +532 step offset
# Completion: Merges LoRA into output/grpo_stage4_merged and publishes to Hugging Face
#

set -euo pipefail

log()  { echo -e "\033[1;32m[stage4]\033[0m $1"; }
warn() { echo -e "\033[1;33m[stage4]\033[0m $1"; }
err()  { echo -e "\033[1;31m[stage4] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

log "=== 1. Environment & Paths Configuration ==="

VENV_PYTHON=".venv/bin/python"
[ -f "$VENV_PYTHON" ] || err ".venv python not found at $VENV_PYTHON"

export HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export APPEND_LOGS=1

# Base model is the merged 16-bit model from Run 1
BASE_MODEL="$SCRIPT_DIR/output/grpo_merged"
[ -f "$BASE_MODEL/config.json" ] || err "Base merged model not found at $BASE_MODEL"
log "Base model: $BASE_MODEL"

# Dataset
TRAIN_JSON="$SCRIPT_DIR/data/grpo_train_stage4_phase.json"
VAL_JSON="$SCRIPT_DIR/data/grpo_val_dataset_grpo.json"
GRPO_DATASET_ROOT="${GRPO_DATASET_ROOT:-$SCRIPT_DIR/dataset_grpo}"
[ -f "$TRAIN_JSON" ] || err "Stage 4 dataset not found at $TRAIN_JSON"

# Output directories
OUTPUT_ROOT="$SCRIPT_DIR/output"
STAGE4_LORA_OUT="$OUTPUT_ROOT/grpo_stage4_lora"
STAGE4_MERGED_OUT="$OUTPUT_ROOT/grpo_stage4_merged"
GRPO_LOG_DIR="$OUTPUT_ROOT/logs/grpo"
mkdir -p "$STAGE4_LORA_OUT" "$STAGE4_MERGED_OUT" "$GRPO_LOG_DIR"

# Hyperparameters
BITS=4
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.0
BATCH_PER_DEVICE=1
GRAD_ACCUM=8
NUM_GENERATIONS=8          # Strategy 1: Boosted exploration (G=8)
GRPO_MICRO_PROMPTS=1       # Essential for 24GB VRAM
MAX_COMP=128
NUM_EPOCHS=1
SAVE_STEPS=20
SAVE_TOTAL_LIMIT=5

NFRAMES=32
VIDEO_MIN_PIXELS=131072
VIDEO_MAX_PIXELS=131072

BETA=0.05                  # Preserves base representations while learning phases
TEMPERATURE=0.9
TOP_P=1.0
LR=4e-5                    # Tuned fine-tuning learning rate
VISION_LR=1e-6
MERGER_LR=5e-6
REPORT_TO="tensorboard"

log "=== Stage 4 Configuration ==="
log "Data: $TRAIN_JSON (506 samples)"
log "Batch: ${BATCH_PER_DEVICE}x${GRAD_ACCUM} (8 prompts/step) | Generations: $NUM_GENERATIONS | Micro: $GRPO_MICRO_PROMPTS"
log "LR: $LR | Beta: $BETA | Epochs: $NUM_EPOCHS"

# Append resumption banner to train.log
printf '\n[=== STARTING TRAINING STAGE 4: PHASE SPECIALIZATION (G=8) AT %s ===]\n' "$(date -u)" >> "$GRPO_LOG_DIR/train.log"

log "=== Launching Stage 4 GRPO Training ==="
bash "$SCRIPT_DIR/scripts/run_instrumented.sh" "$GRPO_LOG_DIR" "stage4_grpo" \
    $VENV_PYTHON -u src/train/train_grpo.py \
    --model_id "$BASE_MODEL" \
    --data_path "$TRAIN_JSON" \
    --eval_path "$VAL_JSON" \
    --image_folder "$GRPO_DATASET_ROOT" \
    --output_dir "$STAGE4_LORA_OUT" \
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
    --disable_flash_attn2 True \
    --use_liger_kernel False \
    --num_train_epochs "$NUM_EPOCHS" \
    --num_generations "$NUM_GENERATIONS" \
    --grpo_micro_prompts "$GRPO_MICRO_PROMPTS" \
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
    --nframes "$NFRAMES" \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --remove_unused_columns False \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 2 \
    --dataloader_persistent_workers True \
    --logging_steps 1 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --report_to "$REPORT_TO" \
    || err "Stage 4 GRPO failed — check $GRPO_LOG_DIR/train.log"

log "=== Stage 4 GRPO Training Complete! ==="

# ── 2. Merge TensorBoard Events with +532 step offset ──
log "=== Merging TensorBoard scalar streams with +532 step offset ==="
PRIMARY_TB_DIR="$OUTPUT_ROOT/grpo_lora/runs/Sep23_13-22-05_31815fea88ba"
$VENV_PYTHON "$SCRIPT_DIR/scripts/merge_tensorboard.py" \
    "$STAGE4_LORA_OUT/runs" \
    "$PRIMARY_TB_DIR" \
    532

# ── 3. Merge LoRA Adapters into Final Standalone Model ──
log "=== Merging Stage 4 LoRA adapters into standalone 16-bit model ==="
$VENV_PYTHON src/merge_lora.py \
    --model-path "$STAGE4_LORA_OUT" \
    --model-base "$BASE_MODEL" \
    --save-model-path "$STAGE4_MERGED_OUT" \
    --safe-serialization 2>&1 | tee "$GRPO_LOG_DIR/merge_stage4.log"

log "=== Final Merged Stage 4 Model Ready at $STAGE4_MERGED_OUT ==="
