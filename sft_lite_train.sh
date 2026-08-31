#!/bin/bash
# sft_lite_train.sh — Lite SFT for RTX 4060 8GB (Qwen3-VL-2B)
# End-to-end: dataset → lite JSON → SFT (QLoRA) → merge → output/lite/sft_merged
#
# Hardware: 1× 8GB VRAM (RTX 4060 Laptop), ~4GB peak, ~9s/iter (gen not used)
# Dataset: picks smallest YouTube video (MruUgO5HFZI, 9 samples) for minimal VRAM
#   SFT lite uses clips + full_video combined (9 samples train, ~17 val)
#   Full prepared dataset remains untouched at data/sft_train_dataset_sft.json
#
# Usage:
#   bash sft_lite_train.sh                          # defaults (rank 4, 10 steps)
#   LORA_RANK=8 MAX_STEPS=20 bash sft_lite_train.sh  # override any param
#   FORCE_REPREPARE=1 bash sft_lite_train.sh        # regenerate lite JSONs
#
# Outputs:
#   Lite adapter:  output/lite/sft_lora
#   Lite merged:   output/lite/sft_merged  (used as base for grpo_lit_train.sh)
#
# Env overrides: MODEL_ID, BITS, LORA_RANK, LORA_ALPHA, BATCH_PER_DEVICE, NFRAMES, etc.
# Require: dataset_sft/Train + dataset_sft/Validation populated, .venv with torch/transformers/peft/trl

set -euo pipefail

log()  { echo -e "\033[1;32m[sft-lite]\033[0m $1"; }
warn() { echo -e "\033[1;33m[sft-lite]\033[0m $1"; }
err()  { echo -e "\033[1;31m[sft-lite] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Env ─────────────────────────────────────────────────────────────────
if [ ! -f ".venv/bin/python" ] && [ ! -f ".venv/Scripts/python.exe" ]; then
    err ".venv not found — run bash setup.sh or bash train_sft.sh first"
fi
if [ -f ".venv/bin/python" ]; then VENV_PYTHON=".venv/bin/python"; else VENV_PYTHON=".venv/Scripts/python.exe"; fi
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
# Use offline cache if model already downloaded
[ -d "$HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct" ] && export HF_HUB_OFFLINE=1 || true

# SFT eval uses generation metrics; disable with ENABLE_GEN_EVAL=0
ENABLE_GEN_EVAL="${ENABLE_GEN_EVAL:-0}"  # lite: skip gen eval by default for speed
if [ "$ENABLE_GEN_EVAL" = "1" ]; then export SFT_COMPUTE_METRICS="eval/compute_metrics.py"; fi

# ── 2. Config (RTX 4060 8GB — 4GB peak) ─────────────────────────────────────
MODEL_ID="${MODEL_ID:-$HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203}"
# Fallback to HF Hub ID if local snapshot not found
[ ! -f "$MODEL_ID/config.json" ] && MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"

OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/lite}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data/lite}"
mkdir -p "$OUTPUT_ROOT" "$DATA_DIR"
export SFT_DATASET_ROOT DATA_DIR

# QLoRA lite
BITS="${BITS:-4}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-8}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Batch / opt (effective batch = 1)
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-4}"
VISION_LR="${VISION_LR:-2e-6}"
MERGER_LR="${MERGER_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"

# Video lite (8 frames keeps prompt ~500 tok, ~4GB)
NFRAMES="${NFRAMES:-8}"
FPS="${FPS:-}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-$((64 * 32 * 32))}"   # 65536
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-$((128 * 32 * 32))}"  # 131072

# Steps (not epochs) for lite smoke test
MAX_STEPS="${MAX_STEPS:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
REPORT_TO="${REPORT_TO:-none}"

FORCE_REPREPARE="${FORCE_REPREPARE:-0}"
SFT_DATASET_ROOT="${SFT_DATASET_ROOT:-dataset_sft}"
export SFT_DATASET_ROOT

log "MODEL_ID=$MODEL_ID"
log "BITS=$BITS RANK=$LORA_RANK ALPHA=$LORA_ALPHA BATCH=$BATCH_PER_DEVICE GRAD_ACCUM=$GRAD_ACCUM"
log "NFRAMES=$NFRAMES VIDEO_MIN=$VIDEO_MIN_PIXELS VIDEO_MAX=$VIDEO_MAX_PIXELS MAX_STEPS=$MAX_STEPS"
log "OUTPUT=$OUTPUT_ROOT/sft_lora → $OUTPUT_ROOT/sft_merged"
log "DATA=$DATA_DIR/sft_train_dataset_sft.json"

# ── 3. Dataset check ────────────────────────────────────────────────────────
for split in Train Validation; do
    if [ ! -d "$SFT_DATASET_ROOT/$split" ] || [ -z "$(ls -A "$SFT_DATASET_ROOT/$split" 2>/dev/null)" ]; then
        err "$SFT_DATASET_ROOT/$split missing"
    fi
done

# ── 4. Lite data prep (smallest video MruUgO5HFZI) ──────────────────────────
if [ "$FORCE_REPREPARE" = "1" ] || [ ! -f "$DATA_DIR/sft_train_all.json" ] || [ ! -f "$DATA_DIR/sft_val_all.json" ]; then
    log "Preparing lite SFT JSONs (smallest video)..."
    $VENV_PYTHON << 'PY'
import json, os
from collections import defaultdict
from pathlib import Path

train_path = Path("data/sft_train_dataset_sft.json")
val_path = Path("data/sft_val_dataset_sft.json")
# Use full prepared JSONs if they exist to derive lite subset; else fall back to prepare_sft.py
if train_path.exists() and val_path.exists():
    train = json.loads(open(train_path).read())
    val = json.loads(open(val_path).read())
    parent_counts = defaultdict(list)
    for s in train:
        vid = s['video'].split('/')[1]
        parent_counts[vid].append(s)
    has_full = set(s['video'].split('/')[1] for s in train if 'full_video' in s['video'])
    sorted_parents = sorted([(vid, len(v)) for vid, v in parent_counts.items() if vid in has_full], key=lambda x: x[1])
    target = sorted_parents[0][0] if sorted_parents else list(parent_counts.keys())[0]
    subset = [s for s in train if target in s['video']]
    # val
    parent_counts_v = defaultdict(list)
    for s in val:
        vid = s['video'].split('/')[1]
        parent_counts_v[vid].append(s)
    has_full_v = set(s['video'].split('/')[1] for s in val if 'full_video' in s['video'])
    sorted_v = sorted([(vid, len(v)) for vid, v in parent_counts_v.items() if vid in has_full_v], key=lambda x: x[1])
    target_v = sorted_v[0][0] if sorted_v else list(parent_counts_v.keys())[0]
    val_subset = [s for s in val if target_v in s['video']]
    print(f"  Lite SFT train: {target} {len(subset)} samples, val: {target_v} {len(val_subset)}")
else:
    # Fallback: run prepare_sft.py then pick smallest
    import subprocess, sys
    dataset_root = os.environ["SFT_DATASET_ROOT"]
    subprocess.run([sys.executable, "data/prepare_sft.py", "--input-dir", f"{dataset_root}/Train", "--output", "data/sft_train_dataset_sft.json", "--data-type", "all"], check=True)
    subprocess.run([sys.executable, "data/prepare_sft.py", "--input-dir", f"{dataset_root}/Validation", "--output", "data/sft_val_dataset_sft.json", "--data-type", "all"], check=True)
    train = json.loads(open("data/sft_train_dataset_sft.json").read())
    val = json.loads(open("data/sft_val_dataset_sft.json").read())
    parent_counts = defaultdict(list)
    for s in train:
        vid = s['video'].split('/')[1]
        parent_counts[vid].append(s)
    has_full = set(s['video'].split('/')[1] for s in train if 'full_video' in s['video'])
    sorted_parents = sorted([(vid, len(v)) for vid, v in parent_counts.items() if vid in has_full], key=lambda x: x[1])
    target = sorted_parents[0][0]
    subset = [s for s in train if target in s['video']]
    parent_counts_v = defaultdict(list)
    for s in val:
        vid = s['video'].split('/')[1]
        parent_counts_v[vid].append(s)
    has_full_v = set(s['video'].split('/')[1] for s in val if 'full_video' in s['video'])
    sorted_v = sorted([(vid, len(v)) for vid, v in parent_counts_v.items() if vid in has_full_v], key=lambda x: x[1])
    target_v = sorted_v[0][0]
    val_subset = [s for s in val if target_v in s['video']]

lite_dir = Path(os.environ["DATA_DIR"])
lite_dir.mkdir(parents=True, exist_ok=True)
json.dump(subset, open(lite_dir / "sft_train_all.json", "w"), indent=2)
json.dump(val_subset, open(lite_dir / "sft_val_all.json", "w"), indent=2)
print(f"  Saved {lite_dir / 'sft_train_all.json'} ({len(subset)}) and sft_val_all.json ({len(val_subset)})")
PY
else
    log "Lite SFT JSONs exist (FORCE_REPREPARE=1 to regenerate)"
fi

TRAIN_DATA="$DATA_DIR/sft_train_all.json"
VAL_DATA="$DATA_DIR/sft_val_all.json"
[ -f "$TRAIN_DATA" ] || err "Missing $TRAIN_DATA"
log "Train: $TRAIN_DATA ($(jq length "$TRAIN_DATA" 2>/dev/null || $VENV_PYTHON -c "import json; print(len(json.load(open('$TRAIN_DATA'))) )") samples)"

# ── 5. Training args ─────────────────────────────────────────────────────────
if [ -n "$FPS" ]; then VIDEO_ARGS="--fps $FPS"; else VIDEO_ARGS="--nframes $NFRAMES"; fi
if [ "$BITS" = "4" ] || [ "$BITS" = "8" ]; then EXTRA_DTYPE="--bf16 True --fp16 False"; else EXTRA_DTYPE="--bf16 True --fp16 False"; fi

COMMON_ARGS=(
    --bits "$BITS"
    --lora_enable True --vision_lora True --use_dora False
    --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT"
    --num_lora_modules -1 --lora_namespan_exclude "['lm_head','embed_tokens']"
    --freeze_vision_tower True --freeze_llm True --freeze_merger True
    --bf16 True --fp16 False --tf32 True --disable_flash_attn2 True --use_liger_kernel True
    --num_train_epochs "$NUM_EPOCHS" --max_steps "$MAX_STEPS"
    --per_device_train_batch_size "$BATCH_PER_DEVICE" --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate "$LR" --vision_lr "$VISION_LR" --merger_lr "$MERGER_LR"
    --weight_decay "$WEIGHT_DECAY" --warmup_steps "$WARMUP_STEPS" --lr_scheduler_type "$LR_SCHEDULER"
    --video_min_pixels "$VIDEO_MIN_PIXELS" --video_max_pixels "$VIDEO_MAX_PIXELS" $VIDEO_ARGS
    --max_seq_length 8192 --gradient_checkpointing True --lazy_preprocess True --remove_unused_columns False
    --dataloader_num_workers 0 --logging_steps "$LOGGING_STEPS" --save_strategy "$SAVE_STRATEGY" --eval_strategy "$EVAL_STRATEGY" --report_to "$REPORT_TO"
    --image_folder "$SFT_DATASET_ROOT"
)

# ── 6. SFT ───────────────────────────────────────────────────────────────────
SFT_OUT="$OUTPUT_ROOT/sft_lora"
SFT_MERGED="$OUTPUT_ROOT/sft_merged"

if [ -f "$SFT_OUT/adapter_config.json" ]; then
    log "sft_lora exists at $SFT_OUT, skipping (rm -rf $SFT_OUT to retrain)"
else
    log "=== SFT lite ($MAX_STEPS steps, ~2s/step, ~22s total + 35s load) ==="
    START=$(date +%s.%N)
    $VENV_PYTHON -u src/train/train_sft.py \
        --model_id "$MODEL_ID" \
        --data_path "$TRAIN_DATA" \
        --eval_path "$VAL_DATA" \
        --output_dir "$SFT_OUT" \
        "${COMMON_ARGS[@]}" 2>&1 | tee "$OUTPUT_ROOT/sft.log"
    END=$(date +%s.%N)
    python3 -c "print(f'[sft-lite] wall {float(\"$END\")-float(\"$START\"):.1f}s')"
fi

# Check LoRA actually trained
log "Checking LoRA..."
$VENV_PYTHON check_lora_weights.py "$SFT_OUT" 2>&1 | tee "$OUTPUT_ROOT/sft_check.log" || warn "LoRA check failed"

# ── 7. Merge ─────────────────────────────────────────────────────────────────
if [ -f "$SFT_MERGED/config.json" ]; then
    log "sft_merged exists at $SFT_MERGED, skipping"
else
    log "Merging LoRA → $SFT_MERGED ..."
    $VENV_PYTHON src/merge_lora.py --model-path "$SFT_OUT" --model-base "$MODEL_ID" --save-model-path "$SFT_MERGED" --safe-serialization 2>&1 | tee "$OUTPUT_ROOT/merge_sft.log"
fi

log "=========================================="
log "SFT lite complete!"
log "  Adapter: $SFT_OUT"
log "  Merged:  $SFT_MERGED"
log "  Next: bash grpo_lit_train.sh  (uses $SFT_MERGED as base)"
log "=========================================="
