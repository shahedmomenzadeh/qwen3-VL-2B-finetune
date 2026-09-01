#!/bin/bash
# lite_grpo_test.sh — Smoke test for GRPO stage on a tiny subset
# Runs 2 steps of GRPO and verifies LoRA weight updates on lora_B parameters.

set -euo pipefail

log() { echo -e "\033[1;32m[lite-grpo-test]\033[0m $1"; }
warn() { echo -e "\033[1;33m[lite-grpo-test]\033[0m $1"; }
err() { echo -e "\033[1;31m[lite-grpo-test] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f ".venv/bin/python" ]; then
    err ".venv not found — run ./setup.sh first"
fi

export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR="output/lite_grpo_test"
DATA_DIR="output/lite_grpo_test"
mkdir -p "$OUTPUT_DIR" "$DATA_DIR"

MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
GRPO_DATASET_ROOT="dataset_grpo"

log "Creating tiny GRPO dataset subset (1 video)..."
.venv/bin/python -c "
import json
with open('data/grpo_train_clip.json') as f:
    clips = json.load(f)

# Pick first video with at least 3 samples
vid_counts = {}
for s in clips:
    v = s['video'].split('/')[1]
    vid_counts[v] = vid_counts.get(v, 0) + 1

target_vid = [v for v, c in vid_counts.items() if c >= 3][0]
print(f'Selected train video: {target_vid} ({vid_counts[target_vid]} samples)')

subset = [s for s in clips if target_vid in s['video']]
with open('$DATA_DIR/grpo_train.json', 'w') as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)

with open('data/grpo_val_clip.json') as f:
    vclips = json.load(f)

v_subset = vclips[:4]
with open('$DATA_DIR/grpo_val.json', 'w') as f:
    json.dump(v_subset, f, ensure_ascii=False, indent=2)
"

log "Starting GRPO smoke test..."
.venv/bin/python src/train/train_grpo.py \
    --model_id "$MODEL_ID" \
    --data_path "$DATA_DIR/grpo_train.json" \
    --eval_path "$DATA_DIR/grpo_val.json" \
    --image_folder "$GRPO_DATASET_ROOT" \
    --output_dir "$OUTPUT_DIR/output" \
    --bits 4 \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True --fp16 False --tf32 True \
    --disable_flash_attn2 True \
    --use_liger_kernel True \
    --max_steps 2 \
    --num_generations 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_completion_length 64 \
    --learning_rate 1e-4 \
    --vision_lr 2e-6 \
    --merger_lr 1e-5 \
    --beta 0.04 \
    --temperature 0.9 \
    --top_p 1.0 \
    --weight_decay 0.0 \
    --warmup_steps 0 \
    --lr_scheduler_type constant \
    --video_min_pixels $((64 * 32 * 32)) \
    --video_max_pixels $((128 * 32 * 32)) \
    --nframes 8 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --remove_unused_columns False \
    --dataloader_num_workers 0 \
    --logging_steps 1 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 2 \
    --save_total_limit 1 \
    --report_to none

log "GRPO smoke test finished. Verifying LoRA weights..."
.venv/bin/python check_lora_weights.py "$OUTPUT_DIR/output"
log "=== GRPO verification completed successfully ==="
