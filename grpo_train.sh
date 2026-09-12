#!/bin/bash
# grpo_train.sh — Complete GRPO training pipeline for Qwen3-VL-2B (QLoRA)
#
# Hardware: 24 GB+ VRAM (tested on RTX 3090 24GB)
# Base Model: shahedm2001/qwen3-vl-2b-cataract-sft-stage2
# Dataset: shahedm2001/dataset_grpo (YouTube MCQs + Phase tasks)
# Uses 100% deterministic rule-based JSON rewards (reward_funcs.py)
#
# Progression: SFT model (HF repo or local) -> GRPO LoRA -> Merge -> output/grpo_merged
#
# Usage examples:
#   # 10% test run (1% YouTube, 9% Phase, 50 frames, G=5, batch=2):
#   SUBSET_RATIO=0.10 YOUTUBE_RATIO=0.01 PHASE_RATIO=0.09 bash grpo_train.sh
#
#   # Quick smoke test (e.g. 5 steps):
#   MAX_STEPS=5 SUBSET_RATIO=0.10 YOUTUBE_RATIO=0.01 PHASE_RATIO=0.09 bash grpo_train.sh
#
#   # Full run (100% data):
#   bash grpo_train.sh

set -euo pipefail

log()  { echo -e "\033[1;32m[grpo]\033[0m $1"; }
warn() { echo -e "\033[1;33m[grpo]\033[0m $1"; }
err()  { echo -e "\033[1;31m[grpo] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Env Check ──────────────────────────────────────────────────────────────
if [ ! -f ".venv/bin/python" ]; then
    log "Creating .venv and syncing dependencies..."
    uv sync
fi

VENV_PYTHON=".venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    err ".venv/bin/python not found — uv sync failed"
fi

export HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Single-threaded OpenMP per dataloader worker (avoid oversubscription).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# ── 2. Dataset Verification & Auto-Download ──────────────────────────────────
GRPO_DATASET_ROOT="${GRPO_DATASET_ROOT:-dataset_grpo}"
GRPO_HF_REPO="${GRPO_HF_REPO:-shahedm2001/dataset_grpo}"

# Auto-download and restore from HF Hub if not present
GRPO_DATASET_ROOT="$GRPO_DATASET_ROOT" GRPO_HF_REPO="$GRPO_HF_REPO" \
    bash "$SCRIPT_DIR/scripts/ensure_dataset_grpo.sh"

for split in Train Validation; do
    [ -d "$GRPO_DATASET_ROOT/$split" ] || err "$GRPO_DATASET_ROOT/$split missing"
done
log "GRPO dataset verified: $GRPO_DATASET_ROOT/Train/ and $GRPO_DATASET_ROOT/Validation/ present."

# ── 3. Base Model Configuration ──────────────────────────────────────────────
DEFAULT_BASE="shahedm2001/qwen3-vl-2b-cataract-sft-stage2"
MODEL_ID="${MODEL_ID:-${GRPO_BASE:-$DEFAULT_BASE}}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"
DATA_PREFIX="${DATA_PREFIX:-data}"
mkdir -p "$OUTPUT_ROOT" "$DATA_PREFIX"

# ── 4. Hyperparameters & Configuration ────────────────────────────────────────
# Quantization & LoRA
BITS="${BITS:-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"  # Must be 0.0 for GRPO deterministic logprobs

# Batch & Generations (configured for 24 GB VRAM)
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-5}"
MAX_COMP="${MAX_COMP:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"

# Video frame & resolution config
NFRAMES="${NFRAMES:-50}"
FPS="${FPS:-}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-$((64 * 32 * 32))}"   # 65536
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-$((128 * 32 * 32))}" # 131072

# Policy & Optimization
BETA="${BETA:-0.04}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-1.0}"
LR="${LR:-1e-4}"
VISION_LR="${VISION_LR:-2e-6}"
MERGER_LR="${MERGER_LR:-1e-5}"

# Dataloader workers
if [ -z "${DATALOADER_WORKERS:-}" ]; then
    _NPROC=$(nproc 2>/dev/null || echo 8)
    DATALOADER_WORKERS=$((_NPROC - 2))
    [ "$DATALOADER_WORKERS" -lt 2 ] && DATALOADER_WORKERS=2
    [ "$DATALOADER_WORKERS" -gt 16 ] && DATALOADER_WORKERS=16
fi
DATALOADER_PREFETCH="${DATALOADER_PREFETCH:-2}"
DATALOADER_PERSISTENT="${DATALOADER_PERSISTENT:-True}"

# Attention: user requested NO flash attention
DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-1}"

# Subsetting parameters
SUBSET_RATIO="${SUBSET_RATIO:-1.0}"
YOUTUBE_RATIO="${YOUTUBE_RATIO:-}"
PHASE_RATIO="${PHASE_RATIO:-}"
SEED="${SEED:-42}"

# Eval & Saving
EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
REPORT_TO="${REPORT_TO:-tensorboard}"
FORCE_REPREPARE="${FORCE_REPREPARE:-0}"

log "Configuration:"
log "  MODEL_ID=$MODEL_ID"
log "  BITS=$BITS  LORA_RANK=$LORA_RANK  LORA_ALPHA=$LORA_ALPHA  DROPOUT=$LORA_DROPOUT"
log "  BATCH_PER_DEVICE=$BATCH_PER_DEVICE  GRAD_ACCUM=$GRAD_ACCUM  NUM_GENERATIONS=$NUM_GENERATIONS"
log "  NFRAMES=$NFRAMES  VIDEO_MIN=$VIDEO_MIN_PIXELS  VIDEO_MAX=$VIDEO_MAX_PIXELS"
log "  MAX_COMP=$MAX_COMP  EPOCHS=$NUM_EPOCHS  MAX_STEPS=${MAX_STEPS:-auto}"
log "  DISABLE_FLASH_ATTN2=$DISABLE_FLASH_ATTN2"
log "  SUBSET_RATIO=$SUBSET_RATIO  YOUTUBE_RATIO=${YOUTUBE_RATIO:-unset}  PHASE_RATIO=${PHASE_RATIO:-unset}"

# ── 5. Data Preparation ────────────────────────────────────────────────────────
TRAIN_JSON_MASTER="$DATA_PREFIX/grpo_train_dataset_grpo.json"
VAL_JSON_MASTER="$DATA_PREFIX/grpo_val_dataset_grpo.json"

if [ ! -f "$TRAIN_JSON_MASTER" ] || [ "$FORCE_REPREPARE" = "1" ]; then
    log "Preparing full GRPO train dataset..."
    $VENV_PYTHON data/prepare_grpo.py --input-dir "$GRPO_DATASET_ROOT/Train" --output "$TRAIN_JSON_MASTER" --split train --data-type all
fi

if [ ! -f "$VAL_JSON_MASTER" ] || [ "$FORCE_REPREPARE" = "1" ]; then
    log "Preparing full GRPO val dataset..."
    $VENV_PYTHON data/prepare_grpo.py --input-dir "$GRPO_DATASET_ROOT/Validation" --output "$VAL_JSON_MASTER" --split val --data-type all
fi

# Apply subsetting if requested
TRAIN_JSON="$TRAIN_JSON_MASTER"
VAL_JSON="$VAL_JSON_MASTER"

IS_SUBSET=0
if [ "$SUBSET_RATIO" != "1.0" ] && [ "$SUBSET_RATIO" != "1" ]; then
    IS_SUBSET=1
fi
if [ -n "$YOUTUBE_RATIO" ] || [ -n "$PHASE_RATIO" ]; then
    IS_SUBSET=1
fi

if [ "$IS_SUBSET" = "1" ]; then
    SUBSET_TRAIN_JSON="$DATA_PREFIX/grpo_train_subset_${SUBSET_RATIO}.json"
    SUBSET_VAL_JSON="$DATA_PREFIX/grpo_val_subset_${SUBSET_RATIO}.json"
    log "Subsampling dataset with scripts/build_grpo_subset.py..."
    
    SUBSET_CMD=(
        "$VENV_PYTHON" scripts/build_grpo_subset.py
        --train-path "$TRAIN_JSON_MASTER"
        --val-path "$VAL_JSON_MASTER"
        --grpo-folder "$GRPO_DATASET_ROOT"
        --subset-ratio "$SUBSET_RATIO"
        --output-train "$SUBSET_TRAIN_JSON"
        --output-val "$SUBSET_VAL_JSON"
        --seed "$SEED"
    )
    if [ -n "$YOUTUBE_RATIO" ]; then
        SUBSET_CMD+=(--youtube-ratio "$YOUTUBE_RATIO")
    fi
    if [ -n "$PHASE_RATIO" ]; then
        SUBSET_CMD+=(--phase-ratio "$PHASE_RATIO")
    fi

    "${SUBSET_CMD[@]}"
    TRAIN_JSON="$SUBSET_TRAIN_JSON"
    VAL_JSON="$SUBSET_VAL_JSON"
fi

VIDEO_ARGS=""
if [ -n "$FPS" ]; then
    VIDEO_ARGS="--fps $FPS"
else
    VIDEO_ARGS="--nframes $NFRAMES"
fi

OPTIONAL_TRAIN_ARGS=()
if [ -n "$MAX_STEPS" ]; then
    OPTIONAL_TRAIN_ARGS+=(--max_steps "$MAX_STEPS")
fi

# ── 6. Run GRPO Training ───────────────────────────────────────────────────────
GRPO_OUT="$OUTPUT_ROOT/grpo_lora"
GRPO_LOG_DIR="$OUTPUT_ROOT/logs/grpo"

CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
if [ "$CLEAN_OUTPUT" = "1" ]; then
    log "CLEAN_OUTPUT=1 — cleaning stale output and log directories for a fresh run..."
    rm -rf "$GRPO_OUT" "$GRPO_LOG_DIR"
fi
mkdir -p "$GRPO_LOG_DIR" "$GRPO_OUT"

log "=== Launching GRPO Training ==="
{
    echo "model=$MODEL_ID"
    echo "train_data=$TRAIN_JSON val_data=$VAL_JSON"
    echo "bits=$BITS lora_r=$LORA_RANK lora_alpha=$LORA_ALPHA dropout=$LORA_DROPOUT"
    echo "batch_per_device=$BATCH_PER_DEVICE grad_accum=$GRAD_ACCUM ngen=$NUM_GENERATIONS max_comp=$MAX_COMP epochs=$NUM_EPOCHS"
    echo "lr=$LR vision_lr=$VISION_LR merger_lr=$MERGER_LR beta=$BETA temp=$TEMPERATURE top_p=$TOP_P"
    echo "nframes=$NFRAMES fps=${FPS:-unset} px_min=$VIDEO_MIN_PIXELS px_max=$VIDEO_MAX_PIXELS"
    echo "disable_flash_attn2=$DISABLE_FLASH_ATTN2 report_to=$REPORT_TO"
} > "$GRPO_LOG_DIR/config.txt" 2>&1 || true

bash "$SCRIPT_DIR/scripts/run_instrumented.sh" "$GRPO_LOG_DIR" "grpo" \
    $VENV_PYTHON -u src/train/train_grpo.py \
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
    --dataloader_num_workers "$DATALOADER_WORKERS" \
    --dataloader_prefetch_factor "$DATALOADER_PREFETCH" \
    --dataloader_persistent_workers "$DATALOADER_PERSISTENT" \
    --logging_steps 1 \
    --eval_strategy "$EVAL_STRATEGY" \
    --eval_steps "$EVAL_STEPS" \
    --per_device_eval_batch_size 1 \
    --save_strategy "$SAVE_STRATEGY" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --report_to "$REPORT_TO" \
    "${OPTIONAL_TRAIN_ARGS[@]}" \
    || err "GRPO failed — see $GRPO_LOG_DIR/train.log + summary.txt"

log "GRPO training complete: $GRPO_OUT"

# ── 7. Verify LoRA Weights ───────────────────────────────────────────────────
log "Verifying GRPO LoRA weights..."
$VENV_PYTHON check_lora_weights.py "$GRPO_OUT" || warn "check_lora_weights reported warnings"

# ── 8. Merge GRPO Adapter ──────────────────────────────────────────────────────
GRPO_MERGED="$OUTPUT_ROOT/grpo_merged"
log "Merging GRPO LoRA into $GRPO_MERGED..."

$VENV_PYTHON src/merge_lora.py \
    --model-path "$GRPO_OUT" \
    --model-base "$MODEL_ID" \
    --save-model-path "$GRPO_MERGED" \
    --safe-serialization 2>&1 | tee "$GRPO_LOG_DIR/merge.log"

log "=== Final GRPO Model Ready at $GRPO_MERGED ==="
