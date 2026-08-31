#!/usr/bin/env bash
# Controlled GRPO benchmark: one sample per task type from the base model.
#
# This uses bf16 compute with 4-bit weights. Liger is disabled and the
# custom trainer uses the regular GRPO loss explicitly.
#
# Usage:
#   bash lite_grpo_test.sh
#   MODEL_ID=/path/to/base BENCH_DIR=/tmp/grpo_test bash lite_grpo_test.sh
#
# The benchmark writes its manifest, logs, timing, telemetry, and adapter under
# BENCH_DIR (default: output/lite_grpo_test).

set -euo pipefail

log() { echo -e "\033[1;32m[lite-grpo-test]\033[0m $1"; }
warn() { echo -e "\033[1;33m[lite-grpo-test]\033[0m $1"; }
err() { echo -e "\033[1;31m[lite-grpo-test] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x ".venv/bin/python" ]; then
    VENV_PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
    VENV_PYTHON=".venv/Scripts/python.exe"
else
    err ".venv Python not found. Run setup.sh first."
fi

GRPO_DATASET_ROOT="${GRPO_DATASET_ROOT:-dataset_grpo}"
BENCH_DIR="${BENCH_DIR:-output/lite_grpo_test}"
HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
# Stress knobs. The first smoke test (4 frames / 32 tokens) peaked at ~2.9 GiB;
# these heavier defaults push the multimodal sequence length up to exercise
# roughly 90% of the 8 GiB laptop GPU. Override if you need a lighter/different run.
NFRAMES="${NFRAMES:-48}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"

for split in Train Validation; do
    [ -d "$GRPO_DATASET_ROOT/$split" ] || err "$GRPO_DATASET_ROOT/$split is missing"
done

mkdir -p "$BENCH_DIR"
export HF_HOME
export PYTHONPATH="src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

if [ -f "$HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203/config.json" ]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
fi

MANIFEST="$BENCH_DIR/train_one_each.json"
PREPARED="$BENCH_DIR/grpo_train.json"

log "Preparing the GRPO manifest from $GRPO_DATASET_ROOT/Train"
"$VENV_PYTHON" data/prepare_grpo.py \
    --input-dir "$GRPO_DATASET_ROOT/Train" \
    --output "$PREPARED" \
    --data-type all \
    > "$BENCH_DIR/prepare.log"

"$VENV_PYTHON" - "$PREPARED" "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

source_path, output_path = map(Path, sys.argv[1:])
task_order = [
    "step_identification",
    "visual_observation",
    "instrument_identification",
    "boundary_detection",
    "temporal_localization",
    "timestamp_to_phase",
    "contextual_phase_recognition",
]
records = json.loads(source_path.read_text())
selected = []
for task in task_order:
    match = next((record for record in records if record.get("question_type") == task), None)
    if match is None:
        raise SystemExit(f"No GRPO record found for {task}")
    selected.append(match)
output_path.write_text(json.dumps(selected, indent=2))
print(f"Selected {len(selected)} records")
for record in selected:
    print(f"  {record['question_type']}: {record['video']}")
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

log "Starting base-model GRPO benchmark"
log "  MODEL_ID=$MODEL_ID"
log "  Samples=7, one per task; num_generations=2; max_steps=7"
log "  4-bit weights, bf16 compute, frames=$NFRAMES, max completion=$MAX_COMPLETION_LENGTH"

START="$(date +%s.%N)"
printf '%s\n' "$START" > "$START_LOG"

set +e
(
    HF_HOME="$HF_HOME" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
    PYTHONPATH="$PYTHONPATH" \
    TOKENIZERS_PARALLELISM=false \
    /usr/bin/time -v -o "$TIME_LOG" \
    "$VENV_PYTHON" -u src/train/train_grpo.py \
        --model_id "$MODEL_ID" \
        --data_path "$MANIFEST" \
        --image_folder "$GRPO_DATASET_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --bits 4 \
        --lora_enable True \
        --vision_lora True \
        --use_dora False \
        --lora_rank 4 \
        --lora_alpha 8 \
        --lora_dropout 0.0 \
        --num_lora_modules -1 \
        --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
        --freeze_vision_tower True \
        --freeze_llm True \
        --freeze_merger True \
        --bf16 False \
        --fp16 True \
        --tf32 False \
        --disable_flash_attn2 True \
        --use_liger_kernel False \
        --loss_type grpo \
        --num_train_epochs 1 \
        --max_steps 7 \
        --num_generations 2 \
        --generation_batch_size 2 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --max_completion_length "$MAX_COMPLETION_LENGTH" \
        --learning_rate 5e-6 \
        --weight_decay 0.0 \
        --warmup_steps 0 \
        --lr_scheduler_type constant \
        --beta 0.04 \
        --temperature 0.9 \
        --top_p 1.0 \
        --video_min_pixels 65536 \
        --video_max_pixels 131072 \
        --nframes "$NFRAMES" \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --remove_unused_columns False \
        --dataloader_num_workers 0 \
        --logging_steps 1 \
        --eval_strategy no \
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

log "Benchmark process exited with status $STATUS"

"$VENV_PYTHON" - "$BENCH_DIR" <<'PY'
import csv
import sys
from pathlib import Path

bench_dir = Path(sys.argv[1])
def read_float(name):
    return float((bench_dir / name).read_text().strip())

elapsed = read_float("end_time") - read_float("start_time")
rows = []
with (bench_dir / "gpu.csv").open(newline="") as handle:
    for row in csv.DictReader(handle):
        try:
            rows.append({key: float(value) for key, value in row.items()})
        except (TypeError, ValueError):
            pass

print(f"Wall time: {elapsed:.2f} s")
print(f"GPU samples: {len(rows)}")
for key, label in [
    ("gpu_util_percent", "GPU utilization"),
    ("memory_used_mib", "GPU memory"),
    ("temperature_c", "GPU temperature"),
    ("power_w", "GPU power"),
]:
    values = [row[key] for row in rows]
    if values:
        print(f"{label}: max={max(values):.1f}, avg={sum(values) / len(values):.1f}")

time_log = bench_dir / "time.log"
if time_log.exists():
    for line in time_log.read_text().splitlines():
        if line.startswith(("User time", "System time", "Maximum resident", "Elapsed")):
            print(line)
PY

echo
log "Logs: $BENCH_DIR"
log "Training log: $TRAIN_LOG"
log "GPU telemetry: $GPU_LOG"

if [ "$STATUS" -ne 0 ]; then
    warn "Training did not complete. See $TRAIN_LOG"
    exit "$STATUS"
fi

log "Lite GRPO benchmark completed successfully"
