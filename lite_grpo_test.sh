#!/bin/bash
# lite_grpo_test.sh — GRPO probe matching lite_e2e benchmark (G=4, nframes=8)
# Runs GRPO on balanced 30/7 subset (realistic lite) from SFT-merged checkpoint.
# Default: GRPO_TRAIN_SAMPLES=30 (~4-5 per 7 tasks), GRPO_MAX_STEPS=8, max_completion=1024.
# Mirrors the probe that succeeded with G=4/nframes=4 and is stable on 8GB.

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
# Allow override: GRPO_TRAIN_SAMPLES=30 (realistic lite) vs 14 (minimal), GRPO_MAX_STEPS scales with samples
GRPO_TRAIN_SAMPLES="${GRPO_TRAIN_SAMPLES:-30}"
GRPO_VAL_SAMPLES="${GRPO_VAL_SAMPLES:-7}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-8}"   # 8 steps for 30 samples (~27% epoch); set 30 for full epoch, 4 for quick
mkdir -p "$OUTPUT_DIR"
# Clean stale GRPO output (prevents rank mismatch on resume: old rank 16 vs new rank 32)
rm -rf "$OUTPUT_DIR/output" "$OUTPUT_DIR/output_merged"
mkdir -p "$OUTPUT_DIR/output"

# Probe config (matches /tmp/run_grpo_probe.sh that is stable on RTX 4060 8GB)
# Prefer SFT-merged checkpoints in order: lite_sft_test/merged -> lite_benchmark/sft_merged -> base
SFT_MERGED_CANDIDATES=(
    "output/lite_sft_test/merged"
    "output/lite_benchmark/sft_merged"
)
SFT_ADAPTER_CANDIDATES=(
    "output/lite_sft_test/output"
    "output/lite_benchmark/sft_lora"
)
MODEL_ID_RESOLVED=""
for cand in "${SFT_MERGED_CANDIDATES[@]}"; do
    if [ -f "$cand/config.json" ]; then
        MODEL_ID_RESOLVED="$cand"
        log "Using SFT-merged checkpoint: $MODEL_ID_RESOLVED"
        break
    fi
done
if [ -z "$MODEL_ID_RESOLVED" ]; then
    # Try to merge SFT adapter on-the-fly if merged not found but adapter exists
    for cand in "${SFT_ADAPTER_CANDIDATES[@]}"; do
        if [ -f "$cand/adapter_config.json" ]; then
            BASE_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
            MERGE_DST="output/lite_sft_test/merged"
            log "No merged SFT found, merging adapter $cand (base $BASE_ID) -> $MERGE_DST ..."
            rm -rf "$MERGE_DST"
            .venv/bin/python src/merge_lora.py --model-path "$cand" --model-base "$BASE_ID" --save-model-path "$MERGE_DST" --safe-serialization
            if [ -f "$MERGE_DST/config.json" ]; then
                MODEL_ID_RESOLVED="$MERGE_DST"
                log "Merged SFT adapter -> $MODEL_ID_RESOLVED"
                break
            else
                warn "Merge failed for $cand"
            fi
        fi
    done
fi
if [ -z "$MODEL_ID_RESOLVED" ]; then
    MODEL_ID_RESOLVED="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
    warn "No SFT checkpoint found, falling back to base: $MODEL_ID_RESOLVED"
fi
MODEL_ID="$MODEL_ID_RESOLVED"
GRPO_DATASET_ROOT="dataset_grpo"
# Realistic lite GRPO uses 30 train / 7 val balanced across 7 tasks (~4-5 per task) under OUTPUT_DIR
# (isolated from data/lite_e2e which stays 14/7 for lite_e2e_benchmark.sh).
DATA_TRAIN="$OUTPUT_DIR/grpo_train.json"
DATA_VAL="$OUTPUT_DIR/grpo_val.json"

# Build / verify 30-sample balanced GRPO subset (isolated from shared lite_e2e)
need_rebuild=0
if [ ! -f "$DATA_TRAIN" ] || [ ! -f "$DATA_VAL" ]; then
    need_rebuild=1
else
    have_train=$(grep -c '"question_type"' "$DATA_TRAIN" 2>/dev/null || echo 0)
    have_val=$(grep -c '"question_type"' "$DATA_VAL" 2>/dev/null || echo 0)
    if [ "$have_train" != "$GRPO_TRAIN_SAMPLES" ] || [ "$have_val" != "$GRPO_VAL_SAMPLES" ]; then
        log "Existing GRPO lite has train=$have_train val=$have_val, want $GRPO_TRAIN_SAMPLES/$GRPO_VAL_SAMPLES -> rebuilding"
        need_rebuild=1
    fi
fi
if [ "$need_rebuild" -eq 1 ]; then
    log "Building balanced GRPO lite: train=$GRPO_TRAIN_SAMPLES val=$GRPO_VAL_SAMPLES (7 tasks) -> $OUTPUT_DIR"
    # Reuse build_lite_benchmark_data.py but output to a temp dir then copy GRPO files, or build directly inline
    # Inline balanced sampling from prepared GRPO JSONs (avoids polluting data/lite_e2e)
    .venv/bin/python - "$DATA_TRAIN" "$DATA_VAL" "$GRPO_TRAIN_SAMPLES" "$GRPO_VAL_SAMPLES" <<'PY'
import json, random, os, sys
from collections import defaultdict
from pathlib import Path
train_out, val_out = Path(sys.argv[1]), Path(sys.argv[2])
n_train, n_val = int(sys.argv[3]), int(sys.argv[4])
random.seed(42)
train_path, val_path = "data/grpo_train_dataset_grpo.json", "data/grpo_val_dataset_grpo.json"
grpo_folder = "dataset_grpo"
def exists_ok(s):
    media = s.get("video") or s.get("image")
    return os.path.exists(os.path.join(grpo_folder, media)) if media else False
with open(train_path) as f: train_data = json.load(f)
with open(val_path) as f: val_data = json.load(f)
train_by = defaultdict(list)
for s in train_data:
    if exists_ok(s): train_by[s.get("question_type","unknown")].append(s)
val_by = defaultdict(list)
for s in val_data:
    if exists_ok(s): val_by[s.get("question_type","unknown")].append(s)
tasks = ["step_identification","visual_observation","instrument_identification","boundary_detection","temporal_localization","timestamp_to_phase","contextual_phase_recognition"]
def balanced(by, total, name):
    n_tasks=len(tasks); base=total//n_tasks; rem=total%n_tasks
    order=tasks.copy(); random.shuffle(order)
    per={t: base + (1 if i < rem else 0) for i,t in enumerate(order)}
    out=[]
    for t in tasks:
        n=per[t]; pool=by.get(t,[]); random.shuffle(pool)
        take=min(n, len(pool)); out.extend(pool[:take])
        if take < n: print(f"WARNING: only {take}/{n} {name} for {t}")
    if len(out) < total:
        rem_pool=[]
        for t in tasks:
            pool=by.get(t,[]); taken=per[t]
            rem_pool.extend(pool[taken:])
        random.shuffle(rem_pool); out.extend(rem_pool[:total-len(out)])
    random.shuffle(out)
    return out
train_subset=balanced(train_by, n_train, "train")
val_subset=balanced(val_by, n_val, "val")
train_out.parent.mkdir(parents=True, exist_ok=True)
with open(train_out,"w") as f: json.dump(train_subset,f,indent=2,ensure_ascii=False)
with open(val_out,"w") as f: json.dump(val_subset,f,indent=2,ensure_ascii=False)
print(f"Saved {train_out}: {len(train_subset)}")
print(f"Saved {val_out}: {len(val_subset)}")
from collections import Counter
print("train",Counter(s.get("question_type") for s in train_subset))
print("val",Counter(s.get("question_type") for s in val_subset))
PY
    # Also keep shared data/lite_e2e for e2e benchmark if missing (14/7)
    if [ ! -f "data/lite_e2e/grpo_train.json" ]; then
        log "Also populating shared data/lite_e2e (14/7) for lite_e2e_benchmark.sh"
        .venv/bin/python scripts/build_lite_benchmark_data.py >/dev/null 2>&1 || true
    fi
fi
log "Dataset: train=$(grep -c '"question_type"' "$DATA_TRAIN") val=$(grep -c '"question_type"' "$DATA_VAL") (expect $GRPO_TRAIN_SAMPLES/$GRPO_VAL_SAMPLES)"
.venv/bin/python - <<'PY'
import json, collections, os
for p in ["output/lite_grpo_test/grpo_train.json","output/lite_grpo_test/grpo_val.json"]:
    if os.path.exists(p):
        d=json.load(open(p))
        print(p, len(d), collections.Counter(s.get("question_type") for s in d))
PY

log "Starting GRPO probe (G=4, nframes=8, max_completion=1024, rank=32, samples=$GRPO_TRAIN_SAMPLES, steps=$GRPO_MAX_STEPS)..."
.venv/bin/python -u src/train/train_grpo.py \
    --model_id "$MODEL_ID" \
    --data_path "$DATA_TRAIN" \
    --eval_path "$DATA_VAL" \
    --image_folder "$GRPO_DATASET_ROOT" \
    --output_dir "$OUTPUT_DIR/output" \
    --bits 4 \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True --fp16 False --tf32 True \
    --disable_flash_attn2 True \
    --use_liger_kernel True \
    --max_steps "$GRPO_MAX_STEPS" \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_completion_length 1024 \
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
    --save_steps "$GRPO_MAX_STEPS" \
    --save_total_limit 1 \
    --report_to none

log "GRPO probe finished. Verifying LoRA weights..."
.venv/bin/python check_lora_weights.py "$OUTPUT_DIR/output"
log "=== GRPO verification completed successfully ==="
