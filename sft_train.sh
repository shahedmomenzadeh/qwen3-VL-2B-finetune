#!/bin/bash
# sft_train.sh — Full SFT for 48GB VRAM (Qwen3-VL-2B, QLoRA)
# End-to-end: dataset → full SFT JSON → SFT (2 epochs) → merge → output/sft_merged
#
# Hardware: 1× 48GB (A6000/L4/4090 24GB*2), ~28-35GB peak, ~3-4s/step (NFRAMES=80)
# Dataset: dataset_sft/Train + dataset_sft/Validation, clips+full combined
#   Uses 100 frames temporal coverage (≈1 frame/7s for 12-min surgery) — see config_setup.md
#
# Usage:
#   bash sft_train.sh                                    # full 48GB defaults
#   MODEL_ID=Qwen/Qwen3-VL-2B-Instruct bash sft_train.sh
#   BITS=16 NFRAMES=60 bash sft_train.sh                  # no quant, fewer frames
#   SUBSET_RATIO=0.5 bash sft_train.sh                    # 50% data ablation
#
# Outputs:
#   Full adapter: output/sft_lora
#   Full merged:  output/sft_merged  (base for grpo_train.sh)
#
# Env overrides: MODEL_ID, BITS, LORA_RANK, BATCH_PER_DEVICE, NFRAMES, SUBSET_RATIO, etc.
# For 24GB, set: BATCH_PER_DEVICE=1 GRAD_ACCUM=16 NFRAMES=100 VIDEO_MAX_PIXELS=$((192*32*32))
# For 32GB, set: BATCH_PER_DEVICE=2 GRAD_ACCUM=8  NFRAMES=100 VIDEO_MAX_PIXELS=$((320*32*32))

set -euo pipefail

log()  { echo -e "\033[1;32m[sft]\033[0m $1"; }
warn() { echo -e "\033[1;33m[sft]\033[0m $1"; }
err()  { echo -e "\033[1;31m[sft] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Env ─────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [ -f "${HOME}/.local/bin/env" ]; then source "${HOME}/.local/bin/env"; else export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH:-}"; fi
    command -v uv &>/dev/null || err "uv not on PATH"
fi
log "uv $(uv --version)"

if [ ! -d ".venv" ]; then log "Creating venv..."; uv venv --python 3.12; fi
if [ -f ".venv/bin/python" ]; then VENV_PYTHON=".venv/bin/python"; else VENV_PYTHON=".venv/Scripts/python.exe"; fi
log "Python: $($VENV_PYTHON --version)"

log "Installing PyTorch cu130..."
uv pip install --python "$VENV_PYTHON" torch torchvision --index-url https://download.pytorch.org/whl/cu130 --quiet || true
log "Installing transformers (GitHub main)..."
uv pip install --python "$VENV_PYTHON" "git+https://github.com/huggingface/transformers.git" --quiet || true
log "Installing deps..."
uv pip install --python "$VENV_PYTHON" --extra-index-url https://pypi.org/simple accelerate "bitsandbytes>=0.43.0" "qwen-vl-utils[decord]" "peft>=0.13.0" "trl>=1.8.0" "liger-kernel>=0.8.0" ujson tensorboard einops sentencepiece protobuf pillow tqdm datasets imageio gdown --quiet || true

INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
    log "Installing flash-attn (may take minutes)..."
    uv pip install --python "$VENV_PYTHON" flash-attn --no-build-isolation 2>&1 || warn "flash-attn failed — using SDPA"
fi

$VENV_PYTHON -c "import torch, transformers, peft, trl; print(f'torch {torch.__version__} cuda {torch.cuda.is_available()}'); print(f'transformers {transformers.__version__} trl {trl.__version__}')" || err "verify failed"

export HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"
export HF_TOKEN="${HF_TOKEN:-}"
[ -n "$HF_TOKEN" ] && export HF_TOKEN
export PYTHONPATH="src:${PYTHONPATH:-}"
ENABLE_GEN_EVAL="${ENABLE_GEN_EVAL:-1}"
[ "$ENABLE_GEN_EVAL" = "1" ] && export SFT_COMPUTE_METRICS="eval/compute_metrics.py"

# ── 2. Separated SFT dataset ─────────────────────────────────────────────────
SFT_DATASET_ROOT="${SFT_DATASET_ROOT:-dataset_sft}"
for s in Train Validation; do
    [ -d "$SFT_DATASET_ROOT/$s" ] && [ -n "$(ls -A "$SFT_DATASET_ROOT/$s" 2>/dev/null)" ] \
        || err "$SFT_DATASET_ROOT/$s missing"
done
log "SFT dataset OK: $SFT_DATASET_ROOT"

# ── 3. Config (48GB — 28-35GB peak) ──────────────────────────────────────────
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"
DATA_PREFIX="${DATA_PREFIX:-data}"

# QLoRA full
BITS="${BITS:-4}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Batch 48GB: global 16 (2×8)
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LR="${LR:-1e-4}"
VISION_LR="${VISION_LR:-2e-6}"
MERGER_LR="${MERGER_LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"

# Video 48GB: 80 frames (or 100 for max temporal), 131k-327k px
NFRAMES="${NFRAMES:-80}"
FPS="${FPS:-}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-$((128 * 32 * 32))}"  # 131072
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-$((320 * 32 * 32))}"  # 327680 — high-res for 48GB
# For 24GB fallback: VIDEO_MAX=$((192*32*32))=196608 NFRAMES=100 BATCH=1 GRAD_ACCUM=16

DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-0}"
EVAL_STEPS="${EVAL_STEPS:-300}"
SAVE_STEPS="${SAVE_STEPS:-300}"
SUBSET_RATIO="${SUBSET_RATIO:-1.0}"
FORCE_REPREPARE="${FORCE_REPREPARE:-0}"

log "MODEL_ID=$MODEL_ID BITS=$BITS RANK=$LORA_RANK BATCH=${BATCH_PER_DEVICE}×${GRAD_ACCUM}=$(($BATCH_PER_DEVICE*$GRAD_ACCUM)) NFRAMES=$NFRAMES VIDEO_MAX=$VIDEO_MAX_PIXELS EPOCHS=$NUM_EPOCHS"

# ── 4. Data prep ─────────────────────────────────────────────────────────────
prepared_needs_refresh() { [ ! -f "$1" ] || [ -n "$(find "$SFT_DATASET_ROOT/Train" "$SFT_DATASET_ROOT/Validation" -type f -newer "$1" -print -quit 2>/dev/null)" ]; }
NEED=0; for f in sft_train_dataset_sft.json sft_val_dataset_sft.json; do prepared_needs_refresh "$DATA_PREFIX/$f" && NEED=1; done; [ "$FORCE_REPREPARE" = "1" ] && NEED=1
if [ "$NEED" = "1" ]; then
    log "Preparing SFT JSONs (clips+full)..."
    $VENV_PYTHON data/prepare_sft.py --input-dir "$SFT_DATASET_ROOT/Train" --output "$DATA_PREFIX/sft_train_dataset_sft.json" --data-type all
    $VENV_PYTHON data/prepare_sft.py --input-dir "$SFT_DATASET_ROOT/Validation" --output "$DATA_PREFIX/sft_val_dataset_sft.json" --data-type all
else log "SFT JSONs exist"; fi

if [ "$SUBSET_RATIO" != "1.0" ] && [ "$SUBSET_RATIO" != "1" ]; then
    log "Subsampling to $SUBSET_RATIO..."
    $VENV_PYTHON -c "import json,random; random.seed(42); r=float('$SUBSET_RATIO'); d=json.load(open('$DATA_PREFIX/sft_train_dataset_sft.json')); n=max(1,int(len(d)*r)); random.shuffle(d); open('$DATA_PREFIX/sft_train_dataset_sft_sub'+str(int(r*100))+'.json','w').write(json.dumps(d[:n],indent=2)); print(f'{len(d)}->{len(d[:n])}')"
    TRAIN_DATA="$DATA_PREFIX/sft_train_dataset_sft_sub$($VENV_PYTHON -c "print(int(float('$SUBSET_RATIO')*100))").json"
else TRAIN_DATA="$DATA_PREFIX/sft_train_dataset_sft.json"; fi
VAL_DATA="$DATA_PREFIX/sft_val_dataset_sft.json"
log "Train $TRAIN_DATA Val $VAL_DATA"

# ── 5. Train ─────────────────────────────────────────────────────────────────
if [ -n "$FPS" ]; then VIDEO_ARGS="--fps $FPS"; else VIDEO_ARGS="--nframes $NFRAMES"; fi
if [ "$DISABLE_FLASH_ATTN2" = "1" ]; then FLASH="--disable_flash_attn2 True"; else FLASH="--disable_flash_attn2 False"; fi

COMMON_ARGS=(
    --bits "$BITS" --lora_enable True --vision_lora True --use_dora False --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" --num_lora_modules -1 --lora_namespan_exclude "['lm_head','embed_tokens']"
    --freeze_vision_tower True --freeze_llm True --freeze_merger True --bf16 True --fp16 False --tf32 True $FLASH --use_liger_kernel True
    --num_train_epochs "$NUM_EPOCHS" --per_device_train_batch_size "$BATCH_PER_DEVICE" --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate "$LR" --vision_lr "$VISION_LR" --merger_lr "$MERGER_LR" --weight_decay 0.1 --warmup_steps "$WARMUP_STEPS" --lr_scheduler_type cosine
    --video_min_pixels "$VIDEO_MIN_PIXELS" --video_max_pixels "$VIDEO_MAX_PIXELS" $VIDEO_ARGS --max_seq_length 32768 --gradient_checkpointing True --lazy_preprocess True --remove_unused_columns False --dataloader_num_workers 4
    --logging_steps 1 --save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit 3 --eval_strategy steps --eval_steps "$EVAL_STEPS" --per_device_eval_batch_size 1 --generation_max_new_tokens 256 --prediction_loss_only False --report_to tensorboard --image_folder "$SFT_DATASET_ROOT"
)

SFT_OUT="$OUTPUT_ROOT/sft_lora"
if [ -f "$SFT_OUT/adapter_config.json" ]; then log "sft_lora exists, skip"; else
    log "=== SFT full (48GB, $NUM_EPOCHS epochs, ~3-4s/step) ==="
    $VENV_PYTHON -u src/train/train_sft.py --model_id "$MODEL_ID" --data_path "$TRAIN_DATA" --eval_path "$VAL_DATA" --output_dir "$SFT_OUT" "${COMMON_ARGS[@]}"
fi

# ── 6. Merge ─────────────────────────────────────────────────────────────────
SFT_MERGED="$OUTPUT_ROOT/sft_merged"
if [ -f "$SFT_MERGED/config.json" ]; then log "sft_merged exists"; else
    log "Merging → $SFT_MERGED"
    $VENV_PYTHON src/merge_lora.py --model-path "$SFT_OUT" --model-base "$MODEL_ID" --save-model-path "$SFT_MERGED" --safe-serialization
fi
log "SFT complete → $SFT_MERGED"
# Optional Hub upload
if [ "${HF_UPLOAD_ENABLED:-0}" = "1" ]; then
    [ -z "${HF_HUB_REPO:-}" ] && err "HF_HUB_REPO not set"
    [ -z "${HF_TOKEN:-}" ] && err "HF_TOKEN not set"
    $VENV_PYTHON src/upload_to_hub.py --local-dir "$SFT_MERGED" --repo-id "$HF_HUB_REPO" --token "$HF_TOKEN" --commit-message "${HF_COMMIT_MSG:-Upload SFT 48GB $(date -u +%Y-%m-%d)}" ${HF_PRIVATE:+--private}
fi
