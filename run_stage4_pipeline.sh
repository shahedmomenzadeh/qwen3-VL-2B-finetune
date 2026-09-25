#!/bin/bash
# run_stage4_pipeline.sh — Launch Stage 4 GRPO Training and Watchdog Daemon in tmux sessions
set -euo pipefail

tmux kill-session -t stage4_watchdog 2>/dev/null || true
tmux kill-session -t stage4_training 2>/dev/null || true

echo "Launching Stage 4 Watchdog Daemon..."
tmux new-session -d -s stage4_watchdog "\
cd /workspace/qwen3-VL-2B-finetune && \
PYTHONPATH=/workspace/qwen3-VL-2B-finetune \
CHECKPOINTS_DIR=/workspace/qwen3-VL-2B-finetune/output/grpo_stage4_lora \
MERGED_DIR=/workspace/qwen3-VL-2B-finetune/output/grpo_stage4_merged \
MODEL_BASE=/workspace/qwen3-VL-2B-finetune/output/grpo_merged \
STATE_FILE=/workspace/qwen3-VL-2B-finetune/output/.sync_state_stage4.json \
DATASET_LABEL=stage4_phase \
TB_OFFSET=532 \
TB_MERGE_DEST=/workspace/qwen3-VL-2B-finetune/output/grpo_lora/runs/Sep23_13-22-05_31815fea88ba \
.venv/bin/python -u scripts/monitor_and_sync.py"

echo "Launching Stage 4 GRPO Training..."
tmux new-session -d -s stage4_training "\
cd /workspace/qwen3-VL-2B-finetune && \
./grpo_stage4_phase.sh"

echo "Stage 4 pipeline launched successfully!"
