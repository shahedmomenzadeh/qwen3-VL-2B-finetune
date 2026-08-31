#!/bin/bash
# grpo_train.sh — Full GRPO for 48GB VRAM (Qwen3-VL-2B, QLoRA, from full SFT)
# End-to-end: dataset_grpo → full GRPO JSON → GRPO from sft_merged → merge → output/grpo_merged
#
# Hardware: 1× 48GB, ~30-38GB peak, ~30-60s/iter (gen heavy)
# Dataset: YouTube MCQs + four phase tasks; no full-video GRPO
#   Train 4252 records, Val 573 records (current separated dataset)
#   Uses deterministic JSON rewards — see src/train/reward_funcs.py
#
# Usage:
#   bash sft_train.sh && bash grpo_train.sh                     # full pipeline (default: from output/sft_merged)
#   bash grpo_train.sh                                          # standalone (will use sft_merged if present)
#   GRPO_BASE=Qwen/Qwen3-VL-2B-Instruct bash grpo_train.sh       # from base (no SFT)
#   BITS=16 NFRAMES=48 bash grpo_train.sh                       # 16-bit, fewer frames
#
# Outputs:
#   Full GRPO adapter: output/grpo_lora  (or output/grpo_clip_lora for legacy)
#   Full GRPO merged:  output/grpo_merged
#
# Env overrides: GRPO_BASE, BITS, LORA_RANK, NFRAMES, BATCH_PER_DEVICE, NUM_GENERATIONS, MAX_COMP, etc.

set -euo pipefail

log()  { echo -e "\033[1;32m[grpo]\033[0m $1"; }
warn() { echo -e "\033[1;33m[grpo]\033[0m $1"; }
err()  { echo -e "\033[1;31m[grpo] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Env ─────────────────────────────────────────────────────────────────
if [ ! -f ".venv/bin/python" ] && [ ! -f ".venv/Scripts/python.exe" ]; then
    err ".venv not found — run bash setup.sh or bash sft_train.sh first"
fi
if [ -f ".venv/bin/python" ]; then VENV_PYTHON=".venv/bin/python"; else VENV_PYTHON=".venv/Scripts/python.exe"; fi
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
[ -d "$HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct" ] && export HF_HUB_OFFLINE=1 || true

# ── 2. Base (default: full SFT) ─────────────────────────────────────────────
BASE_SNAPSHOT="$HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
DEFAULT_BASE="Qwen/Qwen3-VL-2B-Instruct"
[ -f "$BASE_SNAPSHOT/config.json" ] && DEFAULT_BASE="$BASE_SNAPSHOT"

FULL_SFT_MERGED="$SCRIPT_DIR/output/sft_merged"
if [ -n "${GRPO_BASE:-}" ]; then
    MODEL_ID="$GRPO_BASE"
elif [ -f "$FULL_SFT_MERGED/config.json" ]; then
    MODEL_ID="$FULL_SFT_MERGED"
    log "Using full SFT checkpoint: $MODEL_ID (set GRPO_BASE to override)"
else
    MODEL_ID="${MODEL_ID:-$DEFAULT_BASE}"
    warn "Full SFT not found at $FULL_SFT_MERGED — using base $MODEL_ID"
    warn "For SFT→GRPO, run bash sft_train.sh first"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"
DATA_PREFIX="${DATA_PREFIX:-data}"
GRPO_DATASET_ROOT="${GRPO_DATASET_ROOT:-dataset_grpo}"

# ── 3. Config (48GB — gen heavy) ────────────────────────────────────────────
# 48GB can handle BATCH 2-4 with NG 4, but generation_batch = BATCH * GRAD_ACCUM must % NG == 0
BITS="${BITS:-4}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"  # gen_batch = 2*4=8 → 32 completions per gen (8*4)
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_COMP="${MAX_COMP:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"

# Video 48GB: 64 frames balances temporal vs VRAM (100 frames → +30% VRAM/time)
NFRAMES="${NFRAMES:-64}"
FPS="${FPS:-}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-$((128 * 32 * 32))}"  # 131072
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-$((256 * 32 * 32))}"  # 262144
# Max-res 48GB: VIDEO_MAX=$((320*32*32))=327680 NFRAMES=80 BATCH=1

BETA="${BETA:-0.04}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-1.0}"
LR="${LR:-5e-6}"
VISION_LR="${VISION_LR:-2e-6}"
MERGER_LR="${MERGER_LR:-1e-5}"
LOSS_TYPE="${LOSS_TYPE:-dapo}"

USE_LIGER="${USE_LIGER:-1}"
DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-0}"

FORCE_REPREPARE="${FORCE_REPREPARE:-0}"

log "MODEL_ID=$MODEL_ID"
log "BITS=$BITS RANK=$LORA_RANK BATCH=${BATCH_PER_DEVICE}×${GRAD_ACCUM}=$(($BATCH_PER_DEVICE*$GRAD_ACCUM)) NG=$NUM_GENERATIONS MAX_COMP=$MAX_COMP EPOCHS=$NUM_EPOCHS"
log "NFRAMES=$NFRAMES VIDEO_MAX=$VIDEO_MAX_PIXELS BETA=$BETA LOSS=$LOSS_TYPE"
log "OUTPUT=$OUTPUT_ROOT/grpo_lora → $OUTPUT_ROOT/grpo_merged"

# ── 4. Data prep (all clip-level GRPO tasks) ─────────────────────────────────
for split in Train Validation; do
    [ -d "$GRPO_DATASET_ROOT/$split" ] || err "$GRPO_DATASET_ROOT/$split missing"
done

prepared_needs_refresh() { [ ! -f "$1" ] || [ -n "$(find "$GRPO_DATASET_ROOT/Train" "$GRPO_DATASET_ROOT/Validation" -type f -newer "$1" -print -quit 2>/dev/null)" ]; }
NEED=0; for f in grpo_train_dataset_grpo.json grpo_val_dataset_grpo.json; do prepared_needs_refresh "$DATA_PREFIX/$f" && NEED=1; done; [ "$FORCE_REPREPARE" = "1" ] && NEED=1
if [ "$NEED" = "1" ]; then
    log "Preparing GRPO JSONs (YouTube + phase tasks)..."
    $VENV_PYTHON data/prepare_grpo.py --input-dir "$GRPO_DATASET_ROOT/Train" --output "$DATA_PREFIX/grpo_train_dataset_grpo.json" --data-type all
    $VENV_PYTHON data/prepare_grpo.py --input-dir "$GRPO_DATASET_ROOT/Validation" --output "$DATA_PREFIX/grpo_val_dataset_grpo.json" --data-type all
else log "GRPO JSONs exist"; fi

TRAIN_DATA="$DATA_PREFIX/grpo_train_dataset_grpo.json"
VAL_DATA="$DATA_PREFIX/grpo_val_dataset_grpo.json"
log "Train $TRAIN_DATA ($($VENV_PYTHON -c "import json; print(len(json.load(open('$TRAIN_DATA')))") samples) Val $VAL_DATA"

# ── 5. Train ─────────────────────────────────────────────────────────────────
if [ -n "$FPS" ]; then VIDEO_ARGS="--fps $FPS"; else VIDEO_ARGS="--nframes $NFRAMES"; fi
if [ "$USE_LIGER" = "1" ]; then LIGER_FLAG="--use_liger_kernel True"; else LIGER_FLAG="--use_liger_kernel False"; fi
if [ "$DISABLE_FLASH_ATTN2" = "1" ]; then FLASH="--disable_flash_attn2 True"; else FLASH="--disable_flash_attn2 False"; fi

GRPO_OUT="$OUTPUT_ROOT/grpo_lora"
# Legacy compat: if output/grpo_clip_lora exists, use it
[ -d "$OUTPUT_ROOT/grpo_clip_lora" ] && [ ! -f "$GRPO_OUT/adapter_config.json" ] && GRPO_OUT="$OUTPUT_ROOT/grpo_clip_lora" || true

if [ -f "$GRPO_OUT/adapter_config.json" ]; then
    log "grpo_lora exists at $GRPO_OUT, skip (rm -rf $GRPO_OUT to retrain)"
else
    log "=== GRPO full (48GB, NG=$NUM_GENERATIONS, MAX_COMP=$MAX_COMP, ~30-60s/iter) ==="
    $VENV_PYTHON -u src/train/train_grpo.py \
        --model_id "$MODEL_ID" \
        --data_path "$TRAIN_DATA" \
        --eval_path "$VAL_DATA" \
        --image_folder "$GRPO_DATASET_ROOT" \
        --output_dir "$GRPO_OUT" \
        --bits "$BITS" \
        --lora_enable True --vision_lora True --use_dora False --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" --num_lora_modules -1 --lora_namespan_exclude "['lm_head','embed_tokens']" \
        --freeze_vision_tower True --freeze_llm True --freeze_merger True --bf16 True --fp16 False --tf32 True $FLASH $LIGER_FLAG --loss_type "$LOSS_TYPE" \
        --num_train_epochs "$NUM_EPOCHS" --num_generations "$NUM_GENERATIONS" --per_device_train_batch_size "$BATCH_PER_DEVICE" --gradient_accumulation_steps "$GRAD_ACCUM" --max_completion_length "$MAX_COMP" \
        --learning_rate "$LR" --vision_lr "$VISION_LR" --merger_lr "$MERGER_LR" --weight_decay 0.1 --warmup_steps 10 --lr_scheduler_type cosine --beta "$BETA" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
        --video_min_pixels "$VIDEO_MIN_PIXELS" --video_max_pixels "$VIDEO_MAX_PIXELS" $VIDEO_ARGS --gradient_checkpointing True --lazy_preprocess True --remove_unused_columns False --dataloader_num_workers 4 --logging_steps 1 --eval_strategy steps --eval_steps 500 --per_device_eval_batch_size 1 --save_strategy epoch --save_total_limit 3 --report_to tensorboard
fi

# ── 6. Merge ─────────────────────────────────────────────────────────────────
GRPO_MERGED="$OUTPUT_ROOT/grpo_merged"
if [ -f "$GRPO_MERGED/config.json" ]; then log "grpo_merged exists"; else
    log "Merging GRPO LoRA → $GRPO_MERGED ..."
    $VENV_PYTHON src/merge_lora.py --model-path "$GRPO_OUT" --model-base "$MODEL_ID" --save-model-path "$GRPO_MERGED" --safe-serialization 2>&1 | tee "$OUTPUT_ROOT/merge_grpo.log"
fi

log "=========================================="
log "GRPO complete → $GRPO_MERGED"
log "  Base was: $MODEL_ID"
log "  Adapter:  $GRPO_OUT"
log "=========================================="
