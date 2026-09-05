#!/bin/bash
# train_sft.sh — End-to-end SFT pipeline for Qwen3-VL-2B (clips + full_video)
#
# Stages:
#   1. Environment setup (uv + venv + install deps)
#   2. Verify the separated SFT dataset
#   3. Data preparation (JSONL → LLaVA-format JSON)
#   4. OPTIONAL: subsample training data via SUBSET_RATIO (0-1)
#   5. SFT on clips + full videos → merge → output/sft_merged  (final model)
#
# Usage:
#   bash train_sft.sh                          # full training, defaults
#   SUBSET_RATIO=0.3 bash train_sft.sh         # 30% of training data (test run)
#   BITS=16 NFRAMES=48 bash train_sft.sh       # 16-bit LoRA (no quantization)
#
# Repeat runs reuse the existing .venv (installs are skipped once imports
# verify). Set FORCE_REINSTALL=1 to force a full package reinstall.
#
# Set SETUP_ONLY=1 to stop after env + dataset + data prep (validates a
# fresh-machine bootstrap without launching training).
#
# Also set HF_TOKEN if you need to download the model from HuggingFace:
#   export HF_TOKEN=hf_xxxxx
#
# ── OPTIONAL: auto-upload the final SFT checkpoint to HuggingFace Hub ──────
# Set HF_UPLOAD_ENABLED=1 to enable automatic upload after the final merge.
# Requires a HuggingFace write-access token and a target repo ID.
#
#   HF_UPLOAD_ENABLED=1 \
#   HF_TOKEN=hf_xxxx \
#   HF_HUB_REPO=your-username/qwen3-vl-2b-cataract-sft \
#   bash train_sft.sh
#
# HF_PRIVATE=1 creates a private repository (default: public).
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
    # Source the env file the installer creates (sets PATH correctly).
    # Installers write to ~/.local/bin (env file) or ~/.cargo/bin depending on distro.
    if [ -f "${HOME}/.local/bin/env" ]; then
        # shellcheck source=/dev/null
        source "${HOME}/.local/bin/env"
    else
        export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH:-}"
    fi
    # Hard-fail early if uv is still not reachable after install
    if ! command -v uv &>/dev/null; then
        err "uv was installed but is still not on PATH. Add ~/.cargo/bin or ~/.local/bin to PATH and re-run."
    fi
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

# Deterministic env via uv.lock (fast no-op when already in sync, so repeat
# runs start immediately). uv.lock pins torch/cu130 + transformers@git-main.
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
# 3. DATASET VERIFICATION
# ════════════════════════════════════════════════════════════════════════════
log "=== 2. Dataset Setup ==="

SFT_DATASET_ROOT="${SFT_DATASET_ROOT:-dataset_sft}"

# Auto-restore from HF Hub on a fresh machine (no-op when present)
SFT_DATASET_ROOT="$SFT_DATASET_ROOT" bash "$SCRIPT_DIR/scripts/ensure_dataset_sft.sh"

for split in Train Validation; do
    if [ ! -d "$SFT_DATASET_ROOT/$split" ] || [ -z "$(ls -A "$SFT_DATASET_ROOT/$split" 2>/dev/null)" ]; then
        err "$SFT_DATASET_ROOT/$split/ is empty or missing. Populate the separated SFT dataset first."
    fi
done
log "SFT dataset verified: $SFT_DATASET_ROOT/Train/ and $SFT_DATASET_ROOT/Validation/ present."

# ════════════════════════════════════════════════════════════════════════════
# 4. CONFIGURATION (all overridable via env vars)
# ════════════════════════════════════════════════════════════════════════════
log "=== 3. Configuration ==="

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"
DATA_PREFIX="${DATA_PREFIX:-data}"

# QLoRA — production baseline r=16 alpha=32 (merger full-trainable, pos_embed frozen)
BITS="${BITS:-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Batch (Optimized for 24 GB VRAM — effective batch = 1 * 16 = 16)
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
NUM_DEVICES="${NUM_DEVICES:-1}"
GLOBAL_BATCH_SIZE=$((BATCH_PER_DEVICE * GRAD_ACCUM * NUM_DEVICES))

# Learning
LR="${LR:-1e-4}"
VISION_LR="${VISION_LR:-2e-6}"
MERGER_LR="${MERGER_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"

# Video (Optimized for 24 GB VRAM — 100 frames across ~12 min videos)
NFRAMES="${NFRAMES:-100}"
FPS="${FPS:-}"            # leave empty to use NFRAMES; set to override
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-$((96 * 32 * 32))}"    # 98304
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-$((192 * 32 * 32))}"  # 196608

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

prepared_data_needs_refresh() {
    local prepared_path="$1"
    if [ ! -f "$prepared_path" ]; then
        return 0
    fi
    [ -n "$(find "$SFT_DATASET_ROOT/Train" "$SFT_DATASET_ROOT/Validation" -type f -newer "$prepared_path" -print -quit)" ]
}

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
for f in sft_train_dataset_sft.json sft_val_dataset_sft.json; do
    if prepared_data_needs_refresh "$DATA_PREFIX/$f"; then
        NEED_PREPARE=1
    fi
done
[ "$FORCE_REPREPARE" = "1" ] && NEED_PREPARE=1

if [ "$NEED_PREPARE" = "1" ]; then
    log "Running prepare_sft.py (clips + full videos combined)..."
    $VENV_PYTHON data/prepare_sft.py --input-dir "$SFT_DATASET_ROOT/Train"      --output "$DATA_PREFIX/sft_train_dataset_sft.json" --data-type all
    $VENV_PYTHON data/prepare_sft.py --input-dir "$SFT_DATASET_ROOT/Validation" --output "$DATA_PREFIX/sft_val_dataset_sft.json"   --data-type all
else
    log "Prepared JSONs already exist (set FORCE_REPREPARE=1 to regenerate)."
fi

# Apply SUBSET_RATIO to training data (only train, not eval)
if [ "$SUBSET_RATIO" != "1.0" ] && [ "$SUBSET_RATIO" != "1" ]; then
    log "Subsampling training data to ${SUBSET_RATIO} of original..."

    $VENV_PYTHON -c "
import json, random
random.seed(42)  # reproducible

ratio = float('${SUBSET_RATIO}')
path = '${DATA_PREFIX}/sft_train_dataset_sft.json'
with open(path) as f:
    data = json.load(f)

n = max(1, int(len(data) * ratio))
random.shuffle(data)
subset = data[:n]

# Save subset to a separate file so the original is preserved
out = path.replace('.json', f'_sub{int(ratio*100)}.json')
with open(out, 'w') as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)
print(f'  sft_train_dataset_sft.json: {len(data)} -> {len(subset)} samples -> {out}')
"
    # Override the train file path for training
    SUB_SUFFIX="_sub$($VENV_PYTHON -c "print(int(float('${SUBSET_RATIO}')*100))")"
    TRAIN_DATA="$DATA_PREFIX/sft_train_dataset_sft${SUB_SUFFIX}.json"
else
    TRAIN_DATA="$DATA_PREFIX/sft_train_dataset_sft.json"
fi

VAL_DATA="$DATA_PREFIX/sft_val_dataset_sft.json"

log "  Train data: $TRAIN_DATA"
log "  Val data:   $VAL_DATA"

if [ "${SETUP_ONLY:-0}" = "1" ]; then
    log "SETUP_ONLY=1 — bootstrap (env + dataset + data prep) complete, stopping before training."
    exit 0
fi

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
    --lora_namespan_exclude "['lm_head', 'embed_tokens', 'merger', 'pos_embed']"
    --freeze_vision_tower True --freeze_llm True --freeze_merger False
    --bf16 True --fp16 False --tf32 True
    $FLASH_ATTN_FLAG
    --use_liger_kernel True
    --num_train_epochs "$NUM_EPOCHS"
    --per_device_train_batch_size "$BATCH_PER_DEVICE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate "$LR" --vision_lr "$VISION_LR" --merger_lr "$MERGER_LR"
    --weight_decay "$WEIGHT_DECAY" --warmup_steps "$WARMUP_STEPS"
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
    --image_folder "$SFT_DATASET_ROOT"
)

# ════════════════════════════════════════════════════════════════════════════
# 7. SFT on clips + full videos
# ════════════════════════════════════════════════════════════════════════════
log "=== 5. SFT (clips + full videos) ==="

SFT_OUT="$OUTPUT_ROOT/sft_lora"
SFT_LOG_DIR="$OUTPUT_ROOT/logs/sft"

if [ -f "$SFT_OUT/adapter_config.json" ]; then
    log "  sft_lora already exists, skipping SFT."
    log "  (Delete $SFT_OUT to retrain.)"
else
    mkdir -p "$SFT_LOG_DIR"
    # Effective run config (explicit vars only — never secrets like HF_TOKEN)
    {
        echo "model=$MODEL_ID"
        echo "train_data=$TRAIN_DATA val_data=$VAL_DATA"
        echo "bits=$BITS lora_r=$LORA_RANK lora_alpha=$LORA_ALPHA dropout=$LORA_DROPOUT"
        echo "batch_per_device=$BATCH_PER_DEVICE grad_accum=$GRAD_ACCUM global_batch=$GLOBAL_BATCH_SIZE"
        echo "lr=$LR vision_lr=$VISION_LR merger_lr=$MERGER_LR wd=$WEIGHT_DECAY warmup=$WARMUP_STEPS sched=$LR_SCHEDULER epochs=$NUM_EPOCHS"
        echo "nframes=$NFRAMES fps=${FPS:-unset} px_min=$VIDEO_MIN_PIXELS px_max=$VIDEO_MAX_PIXELS"
        echo "disable_flash_attn2=$DISABLE_FLASH_ATTN2 report_to=$REPORT_TO subset=$SUBSET_RATIO"
        $VENV_PYTHON -c "
import json
for p in ('$TRAIN_DATA', '$VAL_DATA'):
    try:
        print(f'{p}: {len(json.load(open(p)))} samples')
    except Exception as e:
        print(f'{p}: unreadable ({e})')
"
    } > "$SFT_LOG_DIR/config.txt" 2>&1 || true
    bash "$SCRIPT_DIR/scripts/run_instrumented.sh" "$SFT_LOG_DIR" "sft" \
        $VENV_PYTHON -u src/train/train_sft.py \
        --model_id "$MODEL_ID" \
        --data_path "$TRAIN_DATA" \
        --eval_path "$VAL_DATA" \
        --output_dir "$SFT_OUT" \
        "${COMMON_ARGS[@]}" \
        || err "SFT failed — see $SFT_LOG_DIR/train.log + summary.txt"
fi
log "SFT complete."

# ════════════════════════════════════════════════════════════════════════════
# 8. MERGE LoRA → final model
# ════════════════════════════════════════════════════════════════════════════
log "=== 6. Merge SFT ==="

SFT_MERGED="$OUTPUT_ROOT/sft_merged"

if [ -f "$SFT_MERGED/config.json" ]; then
    log "  sft_merged already exists, skipping merge."
else
    mkdir -p "$SFT_LOG_DIR"
    $VENV_PYTHON src/merge_lora.py \
        --model-path "$SFT_OUT" \
        --model-base "$MODEL_ID" \
        --save-model-path "$SFT_MERGED" \
        --safe-serialization 2>&1 | tee "$SFT_LOG_DIR/merge.log"
fi
log "SFT LoRA merged to $SFT_MERGED"

# DONE
# ════════════════════════════════════════════════════════════════════════════
log "=========================================="
log "SFT pipeline complete!"
log "  SFT adapter:        $SFT_OUT"
log "  FINAL merged model: $SFT_MERGED"
log "=========================================="
if [ "$SUBSET_RATIO" != "1.0" ] && [ "$SUBSET_RATIO" != "1" ]; then
    log "NOTE: Trained on SUBSET_RATIO=${SUBSET_RATIO} of the data."
    log "      For full training, re-run with SUBSET_RATIO=1.0"
fi

# ════════════════════════════════════════════════════════════════════════════
# 9. UPLOAD to HuggingFace Hub (optional — set HF_UPLOAD_ENABLED=1)
# ════════════════════════════════════════════════════════════════════════════
HF_UPLOAD_ENABLED="${HF_UPLOAD_ENABLED:-0}"

if [ "$HF_UPLOAD_ENABLED" = "1" ]; then
    log "=== 9. Uploading final SFT model to HuggingFace Hub ==="

    # HF_HUB_REPO and HF_TOKEN must be set (as env vars or exported before calling this script)
    if [ -z "${HF_HUB_REPO:-}" ]; then
        err "HF_UPLOAD_ENABLED=1 but HF_HUB_REPO is not set. Example: HF_HUB_REPO=username/my-model"
    fi
    if [ -z "${HF_TOKEN:-}" ]; then
        err "HF_UPLOAD_ENABLED=1 but HF_TOKEN is not set. Example: HF_TOKEN=hf_xxxx"
    fi

    log "  Source:  $SFT_MERGED"
    log "  Target:  $HF_HUB_REPO"
    [ "${HF_PRIVATE:-0}" = "1" ] && log "  Visibility: private" || log "  Visibility: public"

    HF_COMMIT_MSG="${HF_COMMIT_MSG:-Upload Qwen3-VL-2B cataract-surgery SFT ($(date -u +%Y-%m-%d))}"

    $VENV_PYTHON src/upload_to_hub.py \
        --local-dir "$SFT_MERGED" \
        --repo-id   "$HF_HUB_REPO" \
        --token     "$HF_TOKEN" \
        --commit-message "$HF_COMMIT_MSG" \
        ${HF_PRIVATE:+--private}

    log "Upload complete! Model is at: https://huggingface.co/$HF_HUB_REPO"
else
    log "Skipping HuggingFace Hub upload (set HF_UPLOAD_ENABLED=1 to enable)."
    log "  To upload manually:  python src/upload_to_hub.py --local-dir $SFT_MERGED --repo-id your-username/my-model"
fi
