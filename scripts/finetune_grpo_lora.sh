#!/bin/bash
# GRPO LoRA + QLoRA Fine-tuning for Qwen3-VL-2B-Instruct
# This uses the SFT-merged model as base for RL fine-tuning.
# Dataset: Cataract surgery video clips (multi-choice QA)

# Use the SFT merged model as base
MODEL_NAME=${MODEL_NAME:-"output/sft_merged"}

export PYTHONPATH=src:$PYTHONPATH

# ---------- Hyperparameters (override via env vars) ----------
DATA_PATH=${DATA_PATH:-"data/grpo_train.json"}
IMAGE_FOLDER=${IMAGE_FOLDER:-"dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"output/grpo_lora"}
NUM_EPOCHS=${NUM_EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
VISION_LR=${VISION_LR:-2e-6}
MERGER_LR=${MERGER_LR:-1e-5}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
BITS=${BITS:-4}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-512}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
BETA=${BETA:-0.04}
TEMPERATURE=${TEMPERATURE:-0.9}
TOP_P=${TOP_P:-1.0}

# Video — set ONLY ONE of FPS or NFRAMES
# NFRAMES: caps sampled frames. Videos with fewer frames return all frames.
FPS=${FPS:-}
NFRAMES=${NFRAMES:-60}

VIDEO_MIN_PIXELS=${VIDEO_MIN_PIXELS:-$((128 * 32 * 32))}
VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-$((256 * 32 * 32))}

# DeepSpeed
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"scripts/zero3_offload.json"}

# Liger GRPO loss variant
# Options: grpo | bnpo | dr_grpo | dapo (default) | cispo | sapo | luspo
LIGER_GRPO_LOSS_TYPE=${LIGER_GRPO_LOSS_TYPE:-"dapo"}

# ---------- LoRA exclude ----------
LORA_EXCLUDE=${LORA_EXCLUDE:-"['lm_head', 'embed_tokens']"}

deepspeed src/train/train_grpo.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --use_liger_loss True \
    --liger_grpo_loss_type $LIGER_GRPO_LOSS_TYPE \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "$LORA_EXCLUDE" \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_lora_modules -1 \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --num_generations $NUM_GENERATIONS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --video_min_pixels $VIDEO_MIN_PIXELS \
    --video_max_pixels $VIDEO_MAX_PIXELS \
    $( [ -n "$FPS" ] && echo "--fps $FPS" ) \
    $( [ -n "$NFRAMES" ] && echo "--nframes $NFRAMES" ) \
    --learning_rate $LEARNING_RATE \
    --merger_lr $MERGER_LR \
    --vision_lr $VISION_LR \
    --remove_unused_columns False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --beta $BETA \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --dataloader_num_workers 0 \
    --bits $BITS
