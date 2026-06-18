#!/bin/bash
# Merge LoRA weights with base model.
# After SFT training, merge to get a full model for GRPO base.
# After GRPO training, merge to get the final model.

export PYTHONPATH=src:$PYTHONPATH

# Path to LoRA checkpoint (output from training)
MODEL_PATH=${MODEL_PATH:-"output/sft_lora"}

# Base model
# For SFT merge: use the original HuggingFace model
MODEL_BASE=${MODEL_BASE:-"Qwen/Qwen3-VL-2B-Instruct"}
# For GRPO merge: use the SFT merged model as base:
# MODEL_BASE="output/sft_merged"

# Output path
SAVE_MODEL_PATH=${SAVE_MODEL_PATH:-"output/sft_merged"}

# For quantized checkpoints, pass --load-in-8bit or --load-in-4bit
LOAD_IN_8BIT=${LOAD_IN_8BIT:-""}
LOAD_IN_4BIT=${LOAD_IN_4BIT:-""}

python src/merge_lora.py \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --save-model-path $SAVE_MODEL_PATH \
    --safe-serialization \
    $LOAD_IN_8BIT \
    $LOAD_IN_4BIT
