#!/bin/bash
# test_sft_run.sh — Quick end-to-end SFT test on 1 video (clips + full_video)
#
# Runs a single SFT pass (clips + full_video of one video ID) with minimal VRAM.
# Designed for 8 GB GPUs. Does NOT use DeepSpeed.
#
# Usage:  bash test_sft_run.sh

set -euo pipefail

log() { echo -e "\033[1;32m[test_sft]\033[0m $1"; }
err()  { echo -e "\033[1;31m[test_sft] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. VENV CHECK ──────────────────────────────────────────────────────────
if [ ! -f ".venv/bin/python" ]; then
    err ".venv not found — run ./setup.sh first"
fi
# HF_TOKEN is read from the environment — do NOT hardcode it here.
# Export it before running: export HF_TOKEN=hf_xxxx
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="$SCRIPT_DIR/hf_cache"

log "Python: $(.venv/bin/python --version)"

# ── 2. CREATE TINY SUBSET (1 video, all its clips + full_video) ────────────
# Pick a video with few clips to keep data small from the separated SFT data.
TINY_DATA="data/tiny_test"
mkdir -p "$TINY_DATA"

log "Creating tiny subset from 1 video ID..."
.venv/bin/python -c "
import json

# Pick a small video from the combined train data
with open('data/sft_train_dataset_sft.json') as f:
    clips = json.load(f)

# Restrict candidates to YouTube parents that have a full video.
from collections import Counter
youtube_parents = {
    s['video'].split('/')[1]
    for s in clips
    if s['video'].endswith('/full_video.mp4') and not s['video'].split('/')[1].startswith('PH_')
}
if not youtube_parents:
    raise RuntimeError('No YouTube parent with full_video.mp4 found in combined SFT data')

# Count samples per eligible parent and pick the smallest.
vid_counts = Counter()
for s in clips:
    parts = s['video'].split('/')
    if parts[1] in youtube_parents:
        vid_counts[parts[1]] += 1

# Pick smallest eligible video (clips + full_video for that video)
target_vid = sorted(vid_counts.items(), key=lambda x: x[1])[0][0]
print(f'Using video: {target_vid} ({vid_counts[target_vid]} samples)')

subset = [s for s in clips if target_vid in s['video']]
out = '$TINY_DATA/sft_train_all.json'
with open(out, 'w') as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)
print(f'  {out}: {len(subset)} samples')

# Tiny val: 1 validation YouTube video
with open('data/sft_val_dataset_sft.json') as f:
    vdata = json.load(f)
val_parents = sorted({
    s['video'].split('/')[1]
    for s in vdata
    if s['video'].endswith('/full_video.mp4') and not s['video'].split('/')[1].startswith('PH_')
})
if not val_parents:
    raise RuntimeError('No validation YouTube parent with full_video.mp4 found')
vid = val_parents[0]
vsubset = [s for s in vdata if vid in s['video']]
vout = '$TINY_DATA/sft_val_all.json'
with open(vout, 'w') as f:
    json.dump(vsubset, f, ensure_ascii=False, indent=2)
print(f'  {vout}: {len(vsubset)} samples')
"

# ── 3. COMMON MINIMAL ARGS ─────────────────────────────────────────────────
MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
OUTPUT_ROOT="output/tiny_sft_test"
BITS=4
LORA_RANK=16
LR=1e-4
VISION_LR=2e-6
MERGER_LR=1e-5
NFRAMES=32                                 # 32 frames for temporal coverage
VIDEO_MIN=$((64 * 32 * 32))               # 65536 pixels
VIDEO_MAX=$((128 * 32 * 32))              # 131072 pixels
BATCH_SIZE=1
GRAD_ACCUM=1

COMMON_ARGS=(
    --bits $BITS
    --lora_enable True --vision_lora True --use_dora False
    --lora_rank $LORA_RANK --lora_alpha 32 --lora_dropout 0.05
    --num_lora_modules -1
    --lora_namespan_exclude "['lm_head', 'embed_tokens', 'merger', 'pos_embed']"
    --freeze_vision_tower True --freeze_llm True --freeze_merger False
    --bf16 True --fp16 False --tf32 True
    --disable_flash_attn2 True
    --use_liger_kernel True
    --num_train_epochs 1
    --per_device_train_batch_size $BATCH_SIZE
    --gradient_accumulation_steps $GRAD_ACCUM
    --learning_rate $LR --vision_lr $VISION_LR --merger_lr $MERGER_LR
    --weight_decay 0.0 --warmup_steps 0
    --lr_scheduler_type constant
    --video_min_pixels "$VIDEO_MIN" --video_max_pixels "$VIDEO_MAX"
    --nframes $NFRAMES
    --max_seq_length 16384
    --gradient_checkpointing True
    --lazy_preprocess True --remove_unused_columns False
    --dataloader_num_workers 0
    --logging_steps 1 --save_strategy "no"
    --eval_strategy "no"
    --report_to none
    --image_folder dataset_sft
)

# ── 4. SFT (clips + full_video) ─────────────────────────────────────────────
log "====== SFT (clips + full_video) ======"
SFT_OUT="$OUTPUT_ROOT/sft_lora"

.venv/bin/python -u src/train/train_sft.py \
    --model_id "$MODEL_ID" \
    --data_path "$TINY_DATA/sft_train_all.json" \
    --output_dir "$SFT_OUT" \
    "${COMMON_ARGS[@]}"
log "SFT complete."

# ── 5. MERGE LoRA ──────────────────────────────────────────────────────────
log "====== MERGE ======"
SFT_MERGED="$OUTPUT_ROOT/sft_merged"

.venv/bin/python src/merge_lora.py \
    --model-path "$SFT_OUT" \
    --model-base "$MODEL_ID" \
    --save-model-path "$SFT_MERGED" \
    --safe-serialization
log "Final model: $SFT_MERGED"

log "======================================"
log "Test run PASSED"
log "Outputs at: $OUTPUT_ROOT/"
log "======================================"
