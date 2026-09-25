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

# ── 1. Environment Setup (mirrors train_sft.sh: uv -> .venv via uv.lock) ──
log "=== 1. Environment Setup ==="

if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [ -f "${HOME}/.local/bin/env" ]; then
        # shellcheck source=/dev/null
        source "${HOME}/.local/bin/env"
    else
        export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH:-}"
    fi
    if ! command -v uv &>/dev/null; then
        err "uv was installed but is still not on PATH. Add ~/.cargo/bin or ~/.local/bin to PATH and re-run."
    fi
fi
log "uv version: $(uv --version)"

if [ ! -d ".venv" ]; then
    log "Creating virtual environment..."
    uv venv --python 3.12
fi

if [ -f ".venv/bin/python" ]; then
    VENV_PYTHON=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    VENV_PYTHON=".venv/Scripts/python.exe"
else
    err ".venv creation failed — no python binary found in .venv"
fi
log "Python: $($VENV_PYTHON --version)"

# Deterministic env via uv.lock (fast no-op when already in sync).
# Set FORCE_REINSTALL=1 to reinstall every locked package from scratch.
if [ "${FORCE_REINSTALL:-0}" = "1" ]; then
    log "FORCE_REINSTALL=1 — reinstalling locked environment..."
    uv sync --reinstall
else
    log "Syncing environment (uv.lock)..."
    uv sync
fi

# Optional: flash-attn for faster attention (skip if build fails)
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
    log "Installing flash-attn (this may take a few minutes to compile)..."
    uv pip install --python "$VENV_PYTHON" flash-attn --no-build-isolation 2>&1 || {
        warn "flash-attn install failed — continuing with SDPA attention"
        warn "To use flash attention, install manually: uv pip install flash-attn --no-build-isolation"
    }
fi

# Auto-fallback: if flash_attn isn't importable (install skipped or build
# failed), force SDPA so the model loader doesn't request flash_attention_2.
if ! "$VENV_PYTHON" -c "import flash_attn" 2>/dev/null; then
    if [ "${DISABLE_FLASH_ATTN2:-0}" != "1" ]; then
        warn "flash_attn not importable — forcing SDPA (DISABLE_FLASH_ATTN2=1)."
        DISABLE_FLASH_ATTN2=1
    fi
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

export HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
# Single-threaded OpenMP per dataloader worker (avoid oversubscription).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export APPEND_LOGS="${APPEND_LOGS:-1}"
HF_TOKEN="${HF_TOKEN:-}"
[ -n "$HF_TOKEN" ] && export HF_TOKEN

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

# Data loading: auto-size workers to CPUs (leave 2 for main/system), clamp
# to [2, 16] — video decodes are ~150MB+ in flight per sample, so more
# workers risk RAM exhaustion before adding speed. Override any of these.
if [ -z "${DATALOADER_WORKERS:-}" ]; then
    _NPROC=$(nproc 2>/dev/null || echo 8)
    DATALOADER_WORKERS=$((_NPROC - 2))
    [ "$DATALOADER_WORKERS" -lt 2 ] && DATALOADER_WORKERS=2
    [ "$DATALOADER_WORKERS" -gt 16 ] && DATALOADER_WORKERS=16
fi
DATALOADER_PREFETCH="${DATALOADER_PREFETCH:-2}"
DATALOADER_PERSISTENT="${DATALOADER_PERSISTENT:-True}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
GRPO_MICRO_PROMPTS="${GRPO_MICRO_PROMPTS:-1}"
MAX_COMP="${MAX_COMP:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-20}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
EVAL_STEPS="${EVAL_STEPS:-300}"

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
log "BITS=$BITS RANK=$LORA_RANK BATCH=${BATCH_PER_DEVICE}x${GRAD_ACCUM} NG=$NUM_GENERATIONS MICRO=$GRPO_MICRO_PROMPTS MAX_COMP=$MAX_COMP EPOCHS=$NUM_EPOCHS"
log "SAVE_STEPS=$SAVE_STEPS (keep $SAVE_TOTAL_LIMIT) EVAL_STRATEGY=$EVAL_STRATEGY ($EVAL_STEPS)"
log "WORKERS=$DATALOADER_WORKERS PREFETCH=$DATALOADER_PREFETCH PERSISTENT=$DATALOADER_PERSISTENT"
log "NFRAMES=$NFRAMES VIDEO_MIN=$VIDEO_MIN_PIXELS VIDEO_MAX=$VIDEO_MAX_PIXELS"
log "OUTPUT=$OUTPUT_ROOT/grpo_lora -> $OUTPUT_ROOT/grpo_merged"

# ── 4. Data Preparation ────────────────────────────────────────────────────────
# Auto-restore GRPO data from HF Hub on a fresh machine (no-op when present)
GRPO_DATASET_ROOT="$GRPO_DATASET_ROOT" bash "$SCRIPT_DIR/scripts/ensure_dataset_grpo.sh"

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
GRPO_LOG_DIR="$OUTPUT_ROOT/logs/grpo"
log "=== Launching GRPO Training ==="

# (flash_attn presence already handled in §1: installed or SDPA forced)

mkdir -p "$GRPO_LOG_DIR"
{
    echo "model=$MODEL_ID"
    echo "train_data=$TRAIN_JSON val_data=$VAL_JSON"
    echo "bits=$BITS lora_r=$LORA_RANK lora_alpha=$LORA_ALPHA dropout=$LORA_DROPOUT"
    echo "batch_per_device=$BATCH_PER_DEVICE grad_accum=$GRAD_ACCUM ngen=$NUM_GENERATIONS micro=$GRPO_MICRO_PROMPTS max_comp=$MAX_COMP epochs=$NUM_EPOCHS"
    echo "lr=$LR vision_lr=$VISION_LR merger_lr=$MERGER_LR beta=$BETA temp=$TEMPERATURE top_p=$TOP_P"
    echo "nframes=$NFRAMES fps=${FPS:-unset} px_min=$VIDEO_MIN_PIXELS px_max=$VIDEO_MAX_PIXELS"
    echo "disable_flash_attn2=$DISABLE_FLASH_ATTN2 report_to=$REPORT_TO"
} > "$GRPO_LOG_DIR/config.txt" 2>&1 || true

EXISTING_TB_RUN=$(ls -td "$GRPO_OUT"/runs/* 2>/dev/null | head -n1 || true)
if [ -n "$EXISTING_TB_RUN" ] && [ -d "$EXISTING_TB_RUN" ]; then
    export TENSORBOARD_LOGGING_DIR="$EXISTING_TB_RUN"
    log "Preserving existing TensorBoard run: $TENSORBOARD_LOGGING_DIR"
fi

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
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --report_to "$REPORT_TO" \
    || err "GRPO failed — see $GRPO_LOG_DIR/train.log + summary.txt"

log "GRPO training complete: $GRPO_OUT"

# ── 6. Merge GRPO Adapter ──────────────────────────────────────────────────────
GRPO_MERGED="$OUTPUT_ROOT/grpo_merged"
log "Merging GRPO LoRA into $GRPO_MERGED..."

$VENV_PYTHON src/merge_lora.py \
    --model-path "$GRPO_OUT" \
    --model-base "$MODEL_ID" \
    --save-model-path "$GRPO_MERGED" \
    --safe-serialization 2>&1 | tee "$GRPO_LOG_DIR/merge.log"

log "=== Final GRPO Model Ready at $GRPO_MERGED ==="
