#!/usr/bin/env bash
# lite_sft_test.sh — Lightweight SFT smoke test (mirrors lite_grpo_test.sh)
#
# Runs a minimal SFT training on a tiny subset (1 video or few samples)
# to verify the SFT pipeline end-to-end without full dataset/VRAM.
#
# Usage:
#   bash lite_sft_test.sh
#   MODEL_ID=Qwen/Qwen3-VL-2B-Instruct BITS=16 bash lite_sft_test.sh

set -euo pipefail

log() { echo -e "\033[1;32m[lite-sft-test]\033[0m $1"; }
warn() { echo -e "\033[1;33m[lite-sft-test]\033[0m $1"; }
err() { echo -e "\033[1;31m[lite-sft-test] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x ".venv/bin/python" ]; then
    VENV_PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
    VENV_PYTHON=".venv/Scripts/python.exe"
else
    err ".venv Python not found. Run setup.sh first."
fi

SFT_DATASET_ROOT="${SFT_DATASET_ROOT:-dataset_sft}"
BENCH_DIR="${BENCH_DIR:-output/lite_sft_test}"
HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
NFRAMES="${NFRAMES:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
BITS="${BITS:-4}"

for split in Train Validation; do
    [ -d "$SFT_DATASET_ROOT/$split" ] || err "$SFT_DATASET_ROOT/$split is missing"
done

mkdir -p "$BENCH_DIR"
export HF_HOME
export PYTHONPATH="src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

if [ -f "$HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203/config.json" ]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
fi

# Prepare tiny SFT manifest: pick 1 YouTube video with few clips (like test_sft_run.sh)
TINY_TRAIN="$BENCH_DIR/sft_train.json"
TINY_VAL="$BENCH_DIR/sft_val.json"
PREPARED_TRAIN="data/sft_train_dataset_sft.json"
PREPARED_VAL="data/sft_val_dataset_sft.json"

# Ensure prepared SFT JSONs exist (run prepare if needed)
if [ ! -f "$PREPARED_TRAIN" ] || [ ! -f "$PREPARED_VAL" ]; then
    log "Prepared SFT JSONs not found, running prepare_sft.py..."
    $VENV_PYTHON data/prepare_sft.py --input-dir "$SFT_DATASET_ROOT/Train" --output "$PREPARED_TRAIN" --data-type all > "$BENCH_DIR/prepare.log" 2>&1
    $VENV_PYTHON data/prepare_sft.py --input-dir "$SFT_DATASET_ROOT/Validation" --output "$PREPARED_VAL" --data-type all >> "$BENCH_DIR/prepare.log" 2>&1
else
    log "Using existing prepared SFT JSONs"
fi

log "Creating tiny SFT subset (1 video)..."
$VENV_PYTHON - "$PREPARED_TRAIN" "$PREPARED_VAL" "$TINY_TRAIN" "$TINY_VAL" <<'PY'
import json, sys
from pathlib import Path
from collections import Counter

train_path, val_path, tiny_train, tiny_val = map(Path, sys.argv[1:])

with open(train_path) as f:
    clips = json.load(f)

# Find YouTube parents with full_video
youtube_parents = {
    s['video'].split('/')[1]
    for s in clips
    if s['video'].endswith('/full_video.mp4') and not s['video'].split('/')[1].startswith('PH_')
}
if not youtube_parents:
    raise RuntimeError('No YouTube parent with full_video found')

vid_counts = Counter()
for s in clips:
    parts = s['video'].split('/')
    if len(parts) > 1 and parts[1] in youtube_parents:
        vid_counts[parts[1]] += 1

target_vid = sorted(vid_counts.items(), key=lambda x: x[1])[0][0]
print(f'Selected train video: {target_vid} ({vid_counts[target_vid]} samples)')
subset = [s for s in clips if target_vid in s['video']]
Path(tiny_train).write_text(json.dumps(subset, indent=2))
print(f'  {tiny_train}: {len(subset)} samples')

with open(val_path) as f:
    vdata = json.load(f)
val_parents = sorted({
    s['video'].split('/')[1]
    for s in vdata
    if s['video'].endswith('/full_video.mp4') and not s['video'].split('/')[1].startswith('PH_')
})
vid = val_parents[0]
vsubset = [s for s in vdata if vid in s['video']]
Path(tiny_val).write_text(json.dumps(vsubset, indent=2))
print(f'  {tiny_val}: {len(vsubset)} samples')
PY

TRAIN_LOG="$BENCH_DIR/train.log"
GPU_LOG="$BENCH_DIR/gpu.csv"
TIME_LOG="$BENCH_DIR/time.log"
STATUS_LOG="$BENCH_DIR/exit_code"
START_LOG="$BENCH_DIR/start_time"
END_LOG="$BENCH_DIR/end_time"
OUTPUT_DIR="$BENCH_DIR/output"

rm -rf "$OUTPUT_DIR"
rm -f "$TRAIN_LOG" "$GPU_LOG" "$TIME_LOG" "$STATUS_LOG" "$START_LOG" "$END_LOG"

log "Starting SFT smoke test"
log "  MODEL_ID=$MODEL_ID"
log "  BITS=$BITS, NFRAMES=$NFRAMES, MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH"
log "  Train: $(cat $TINY_TRAIN | grep -c '"video"') samples, Val: $(cat $TINY_VAL | grep -c '"video"') samples"

START="$(date +%s.%N)"
printf '%s\n' "$START" > "$START_LOG"

set +e
(
    HF_HOME="$HF_HOME" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
    PYTHONPATH="$PYTHONPATH" \
    TOKENIZERS_PARALLELISM=false \
    /usr/bin/time -v -o "$TIME_LOG" \
    $VENV_PYTHON -u src/train/train_sft.py \
        --model_id "$MODEL_ID" \
        --data_path "$TINY_TRAIN" \
        --eval_path "$TINY_VAL" \
        --image_folder "$SFT_DATASET_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --bits "$BITS" \
        --lora_enable True \
        --vision_lora True \
        --use_dora False \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.0 \
        --num_lora_modules -1 \
        --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
        --freeze_vision_tower True \
        --freeze_llm True \
        --freeze_merger True \
        --bf16 True \
        --fp16 False \
        --tf32 True \
        --disable_flash_attn2 True \
        --use_liger_kernel True \
        --num_train_epochs 1 \
        --max_steps 5 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --per_device_eval_batch_size 1 \
        --learning_rate 1e-4 \
        --vision_lr 2e-6 \
        --merger_lr 1e-5 \
        --weight_decay 0.0 \
        --warmup_steps 0 \
        --lr_scheduler_type constant \
        --video_min_pixels 65536 \
        --video_max_pixels 131072 \
        --nframes "$NFRAMES" \
        --max_seq_length "$MAX_SEQ_LENGTH" \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --remove_unused_columns False \
        --dataloader_num_workers 0 \
        --logging_steps 1 \
        --eval_strategy steps \
        --eval_steps 5 \
        --save_strategy no \
        --report_to none \
        > "$TRAIN_LOG" 2>&1
    printf '%s\n' "$?" > "$STATUS_LOG"
) &
TRAIN_PID=$!

(
    printf 'timestamp,gpu_util_percent,memory_used_mib,memory_total_mib,temperature_c,power_w\n' > "$GPU_LOG"
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        timestamp="$(date +%s.%N)"
        row="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || true)"
        if [ -n "$row" ]; then
            printf '%s,%s\n' "$timestamp" "$(printf '%s' "$row" | tr -d ' ')" >> "$GPU_LOG"
        fi
        sleep 0.5
    done
) &
MONITOR_PID=$!

wait "$TRAIN_PID"
wait "$MONITOR_PID" 2>/dev/null || true
END="$(date +%s.%N)"
printf '%s\n' "$END" > "$END_LOG"
STATUS="$(cat "$STATUS_LOG" 2>/dev/null || printf '1')"
set -e

log "SFT process exited with status $STATUS"

$VENV_PYTHON - "$BENCH_DIR" <<'PY'
import csv, sys
from pathlib import Path
bench_dir = Path(sys.argv[1])
def read_float(name):
    return float((bench_dir / name).read_text().strip())
elapsed = read_float("end_time") - read_float("start_time")
rows = []
with (bench_dir / "gpu.csv").open(newline="") as handle:
    for row in csv.DictReader(handle):
        try:
            rows.append({k: float(v) for k, v in row.items()})
        except: pass
print(f"Wall time: {elapsed:.2f} s")
print(f"GPU samples: {len(rows)}")
for k, label in [("gpu_util_percent","GPU utilization"),("memory_used_mib","GPU memory"),("temperature_c","GPU temperature"),("power_w","GPU power")]:
    vals = [r[k] for r in rows]
    if vals: print(f"{label}: max={max(vals):.1f}, avg={sum(vals)/len(vals):.1f}")
time_log = bench_dir / "time.log"
if time_log.exists():
    for line in time_log.read_text().splitlines():
        if line.startswith(("User time","System time","Maximum resident","Elapsed")):
            print(line)
PY

echo
log "Logs: $BENCH_DIR"
log "Training log: $TRAIN_LOG"
log "GPU telemetry: $GPU_LOG"

if [ "$STATUS" -ne 0 ]; then
    warn "SFT did not complete. See $TRAIN_LOG"
    exit "$STATUS"
fi

log "Lite SFT benchmark completed successfully"
log "Verifying LoRA weights..."
$VENV_PYTHON check_lora_weights.py "$OUTPUT_DIR" || warn "LoRA check failed"

