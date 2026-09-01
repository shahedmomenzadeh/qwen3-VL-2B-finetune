#!/bin/bash
# setup.sh — Create environment and install dependencies for Qwen3-VL finetuning
#
# Run once before training:
#   chmod +x setup.sh && ./setup.sh

set -euo pipefail

log() { echo -e "\033[1;32m[setup.sh]\033[0m $1"; }
err() { echo -e "\033[1;31m[setup.sh] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. CHECK AND INSTALL UV ───────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
log "uv version: $(uv --version)"

# ── 2. CREATE VIRTUAL ENVIRONMENT ──────────────────────────────────────────
if [ -d ".venv" ]; then
    log "Removing old .venv..."
    rm -rf .venv
fi

uv venv --python 3.12
log "Virtual environment created at .venv/"

# ── 3. INSTALL PYTORCH (CUDA 13.0) ─────────────────────────────────────────
log "Installing PyTorch with CUDA 13.0..."
uv pip install --python .venv/bin/python \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu130

# ── 4. INSTALL TRANSFORMERS FROM GITHUB (latest, supports Qwen3-VL) ────────
log "Installing transformers from GitHub main..."
uv pip install --python .venv/bin/python \
    "git+https://github.com/huggingface/transformers.git"

# ── 5. INSTALL REMAINING DEPENDENCIES ──────────────────────────────────────
log "Installing remaining dependencies..."
uv pip install --python .venv/bin/python \
    --extra-index-url https://pypi.org/simple \
    "accelerate" \
    "bitsandbytes>=0.43.0" \
    "qwen-vl-utils[decord]" \
    "peft>=0.13.0" \
    "trl>=1.8.0" \
    "liger-kernel>=0.8.0" \
    "ujson" \
    "tensorboard" \
    "einops" \
    "sentencepiece" \
    "protobuf" \
    "pillow" \
    "tqdm" \
    "datasets" \
    "openai" \
    "imageio"

# ── 6. VERIFY ──────────────────────────────────────────────────────────────
log "Verifying installation..."
.venv/bin/python -c "
import torch; print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')

import transformers; print(f'transformers: {transformers.__version__}')

from transformers import AutoModelForImageTextToText
print('AutoModelForImageTextToText: OK')

import qwen_vl_utils; print('qwen_vl_utils: OK')
import peft; print(f'peft: {peft.__version__}')
import trl; print(f'trl: {trl.__version__}')
import liger_kernel; print('liger_kernel: OK')
import accelerate; print(f'accelerate: {accelerate.__version__}')
import bitsandbytes; print(f'bitsandbytes: {bitsandbytes.__version__}')
try:
    import deepspeed; print(f'deepspeed: {deepspeed.__version__}')
except ImportError:
    print('deepspeed: not installed (optional, single-GPU OK)')
"

log "=============================="
log "Setup complete!"
log "Next:  ./train.sh"
log "Or:     bash scripts/finetune_sft_lora.sh"
log "=============================="
