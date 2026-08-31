#!/bin/bash
# SFT LoRA + QLoRA Fine-tuning for Qwen3-VL-2B-Instruct
# Target: Qwen/Qwen3-VL-2B-Instruct
# Dataset: Cataract surgery video clips

# Single SFT stage: clips + full videos combined, starting from the base HuggingFace model
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-VL-2B-Instruct"}

export PYTHONPATH=src:${PYTHONPATH:-}

# Enable generation-based eval metrics (exact_match, contains_match, etc.)
# Set to empty string to disable: SFT_COMPUTE_METRICS="" ./finetune_sft_lora.sh
export SFT_COMPUTE_METRICS="${SFT_COMPUTE_METRICS:-eval/compute_metrics.py}"

# ---------- Hyperparameters (override via env vars) ----------
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-2}
NUM_DEVICES=${NUM_DEVICES:-1}
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

DATA_PATH=${DATA_PATH:-"data/sft_train_dataset_sft.json"}
EVAL_PATH=${EVAL_PATH:-"data/sft_val_dataset_sft.json"}
# Image folder is the separated SFT dataset root (prepared paths include Train/ or Validation/)
IMAGE_FOLDER=${IMAGE_FOLDER:-"dataset_sft"}
OUTPUT_DIR=${OUTPUT_DIR:-"output/sft_lora"}
NUM_EPOCHS=${NUM_EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
VISION_LR=${VISION_LR:-2e-6}
MERGER_LR=${MERGER_LR:-1e-5}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
BITS=${BITS:-4}

# Video — set ONLY ONE of FPS or NFRAMES
# NFRAMES: caps sampled frames to this value (e.g., 60). If video has fewer frames,
#          all frames are used. Clips longer than NFRAMES will be uniformly sampled.
# FPS: frames per second (alternative to NFRAMES)
FPS=${FPS:-}
NFRAMES=${NFRAMES:-60}
VIDEO_FRAME_ARGS=""
if [ -n "$FPS" ]; then
    VIDEO_FRAME_ARGS="--fps $FPS"
else
    VIDEO_FRAME_ARGS="--nframes $NFRAMES"
fi

# Pixel resolution for Qwen3-VL (patch_size=16 → 32×32 tokens per patch)
VIDEO_MIN_PIXELS=${VIDEO_MIN_PIXELS:-$((128 * 32 * 32))}  # 131072 pixels
VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-$((256 * 32 * 32))}  # 262144 pixels

# ---------- DeepSpeed config ----------
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"scripts/zero3_offload.json"}

# ---------- LoRA exclude ----------
# IMPORTANT: If you want to tune embed_token with LoRA, remove 'embed_tokens' from the exclude list
# and also remove 'lm_head' from the exclude list.
LORA_EXCLUDE=${LORA_EXCLUDE:-"['lm_head', 'embed_tokens']"}

# If you want to include vision in LoRA:
# VISION_LORA="True"
# FREEZE_VISION="False"
# If you want to freeze vision and train only LLM with LoRA:
# VISION_LORA="False"
# FREEZE_VISION="True"

deepspeed src/train/train_sft.py \
    --use_liger_kernel True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "$LORA_EXCLUDE" \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_lora_modules -1 \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_min_pixels $VIDEO_MIN_PIXELS \
    --video_max_pixels $VIDEO_MAX_PIXELS \
    $VIDEO_FRAME_ARGS \
    --learning_rate $LEARNING_RATE \
    --merger_lr $MERGER_LR \
    --vision_lr $VISION_LR \
    --weight_decay 0.1 \
    --warmup_steps 10 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --dataloader_num_workers 0 \
    --bits $BITS \
    --eval_path $EVAL_PATH \
    --eval_strategy steps \
    --eval_steps 500 \
    --per_device_eval_batch_size 1 \
    --generation_max_new_tokens 256 \
    --prediction_loss_only False
