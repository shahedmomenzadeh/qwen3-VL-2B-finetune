#!/usr/bin/env bash
# test_dtypes.sh — dtype smoke matrix: bf16 / fp16 / fp32 x (SFT + GRPO), adapter-only.
#
# Purpose: prove every compute dtype trains end-to-end (forward + backward +
# adapter save with non-zero lora_B) on both trainers:
#   - src/train/train_sft.py  (QwenSFTTrainer, use_liger_kernel=True prod path)
#   - src/train/train_grpo.py (QwenGRPOTrainer, custom token-mean loss + generate)
#
# Adapter-only: each run saves just the LoRA adapter (+ non_lora_state_dict.bin)
# via the trainers' end-of-training save. No merge_lora.py / full-model
# checkpoint is produced. Uses --save_strategy no so only the final adapter
# is written (no intermediate checkpoint-* dirs).
#
# Dtype mapping (mirrors train_{sft,grpo}.py compute_dtype logic):
#   bf16 -> --bf16 True  --fp16 False  (torch.bfloat16)
#   fp16 -> --bf16 False --fp16 True   (torch.float16, HF fp16 scaler)
#   fp32 -> --bf16 False --fp16 False  (torch.float32, GRPO uses nullcontext)
# BITS applies to all dtypes equally (default 4: 4-bit base + <dtype> compute,
# so fp32 also fits 8 GB; BITS=16 fp32 full weights needs ~24 GB+).
#
# Resource profile (RTX 4060 8 GB safe): nframes=4, 64k/128k pixels, batch=1,
# SFT 3 steps / GRPO G=2 x 2 steps x 64 completion tokens.
#
# Usage:
#   bash test_dtypes.sh                        # full 3x2 matrix
#   DTYPES="bf16" bash test_dtypes.sh          # single dtype
#   STAGES="sft" bash test_dtypes.sh           # single stage (sft|grpo)
#   BITS=4 NFRAMES=4 bash test_dtypes.sh       # override footprint
#
# Creates NO changes to existing code; all outputs under output/dtype_test/.

set -euo pipefail

log()  { echo -e "\033[1;32m[dtype-test]\033[0m $1"; }
warn() { echo -e "\033[1;33m[dtype-test]\033[0m $1"; }
err()  { echo -e "\033[1;31m[dtype-test] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x ".venv/bin/python" ]; then
    VENV_PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
    VENV_PYTHON=".venv/Scripts/python.exe"
else
    err ".venv Python not found. Run setup.sh first."
fi

# ── Config (env-overridable) ──────────────────────────────────────────────
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
DTYPES="${DTYPES:-bf16 fp16 fp32}"       # space-separated subset of: bf16 fp16 fp32
STAGES="${STAGES:-sft grpo}"             # subset of: sft grpo
BITS="${BITS:-4}"                        # 4 keeps fp32-compute runnable on 8 GB
NFRAMES="${NFRAMES:-4}"
SFT_STEPS="${SFT_STEPS:-3}"
GRPO_STEPS="${GRPO_STEPS:-2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"  # G=2 smoke (prod G=5); same code path
MAX_COMP="${MAX_COMP:-64}"               # short completions: dtype smoke, not quality
SFT_DATASET_ROOT="${SFT_DATASET_ROOT:-dataset_sft}"
GRPO_DATASET_ROOT="${GRPO_DATASET_ROOT:-dataset_grpo}"
OUT_ROOT="${OUT_ROOT:-output/dtype_test}"
HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"

mkdir -p "$OUT_ROOT"
export HF_HOME
export PYTHONPATH="src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# Auto-restore SFT data from HF Hub on a fresh machine (no-op when present)
SFT_DATASET_ROOT="$SFT_DATASET_ROOT" VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}" bash "$SCRIPT_DIR/scripts/ensure_dataset_sft.sh"

if [ -f "$HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203/config.json" ]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
fi

for split in Train Validation; do
    [ -d "$SFT_DATASET_ROOT/$split" ] || err "$SFT_DATASET_ROOT/$split is missing (auto-restore failed?)"
    [ -d "$GRPO_DATASET_ROOT/$split" ] || err "$GRPO_DATASET_ROOT/$split is missing"
done
for f in data/sft_train_dataset_sft.json data/sft_val_dataset_sft.json \
         data/grpo_train_dataset_grpo.json data/grpo_val_dataset_grpo.json; do
    [ -f "$f" ] || err "$f is missing — run data preparation first"
done

# ── Tiny manifests (built once, shared by all dtype runs) ─────────────────
SFT_TRAIN="$OUT_ROOT/sft_train.json"
SFT_VAL="$OUT_ROOT/sft_val.json"
GRPO_TRAIN="$OUT_ROOT/grpo_train.json"
GRPO_VAL="$OUT_ROOT/grpo_val.json"

log "Building tiny manifests under $OUT_ROOT ..."
$VENV_PYTHON - "$SFT_TRAIN" "$SFT_VAL" "$GRPO_TRAIN" "$GRPO_VAL" <<'PY'
import json, os, random, sys
from collections import Counter, defaultdict
from pathlib import Path

sft_train_out, sft_val_out, grpo_train_out, grpo_val_out = map(Path, sys.argv[1:])
random.seed(42)

# SFT: smallest YouTube video (clips + full_video), like lite_sft_test.sh
clips = json.load(open("data/sft_train_dataset_sft.json"))
parents = {s["video"].split("/")[1] for s in clips
           if s["video"].endswith("/full_video.mp4") and not s["video"].split("/")[1].startswith("PH_")}
counts = Counter(s["video"].split("/")[1] for s in clips if len(s["video"].split("/")) > 1 and s["video"].split("/")[1] in parents)
vid = sorted(counts.items(), key=lambda x: x[1])[0][0]
subset = [s for s in clips if vid in s["video"]]
sft_train_out.write_text(json.dumps(subset, indent=2))
print(f"SFT train: video {vid}, {len(subset)} samples -> {sft_train_out}")

vdata = json.load(open("data/sft_val_dataset_sft.json"))
vparents = sorted({s["video"].split("/")[1] for s in vdata
                   if s["video"].endswith("/full_video.mp4") and not s["video"].split("/")[1].startswith("PH_")})
vvid = vparents[0]
vsubset = [s for s in vdata if vvid in s["video"]]
sft_val_out.write_text(json.dumps(vsubset, indent=2))
print(f"SFT val: video {vvid}, {len(vsubset)} samples -> {sft_val_out}")

# GRPO: round-robin over tasks so reward fns get coverage (4 train / 2 val)
def exists_ok(folder, s):
    media = s.get("video") or s.get("image")
    return os.path.exists(os.path.join(folder, media)) if media else False

def round_robin(path, total):
    data = [s for s in json.load(open(path)) if exists_ok("dataset_grpo", s)]
    by = defaultdict(list)
    for s in data:
        by[s.get("question_type", "unknown")].append(s)
    for pool in by.values():
        random.shuffle(pool)
    out, tasks = [], sorted(by)
    i = 0
    while len(out) < total and any(by[t][i:i+1] for t in tasks):
        for t in tasks:
            if len(out) >= total:
                break
            if i < len(by[t]):
                out.append(by[t][i])
        i += 1
    random.shuffle(out)
    return out

gt = round_robin("data/grpo_train_dataset_grpo.json", 4)
gv = round_robin("data/grpo_val_dataset_grpo.json", 2)
grpo_train_out.write_text(json.dumps(gt, indent=2, ensure_ascii=False))
grpo_val_out.write_text(json.dumps(gv, indent=2, ensure_ascii=False))
print(f"GRPO train: {len(gt)} {Counter(s.get('question_type') for s in gt)} -> {grpo_train_out}")
print(f"GRPO val: {len(gv)} {Counter(s.get('question_type') for s in gv)} -> {grpo_val_out}")
PY

# ── Helpers ───────────────────────────────────────────────────────────────
# verify_adapter <adapter_dir> <train_log>: exit 0 iff adapter saved AND lora_B moved.
verify_adapter() {
    local dir="$1" tlog="$2"
    local adapter=""
    if [ -f "$dir/adapter_model.safetensors" ]; then
        adapter="$dir/adapter_model.safetensors"
    elif [ -f "$dir/adapter_model.bin" ]; then
        adapter="$dir/adapter_model.bin"
    else
        warn "no adapter file in $dir"
        return 1
    fi
    # 1) lora_B zeros->non-zero proves gradients flowed in this dtype
    if ! $VENV_PYTHON check_lora_weights.py "$dir" > "$dir/check.log" 2>&1; then
        warn "check_lora_weights.py failed for $dir (see $dir/check.log)"
        return 1
    fi
    grep -q "VERDICT: LoRA weights CHANGED" "$dir/check.log" || {
        warn "lora_B all zeros in $dir — no learning happened"
        return 1
    }
    # 2) report saved adapter tensor dtypes (proof of the active precision)
    $VENV_PYTHON - "$adapter" >> "$dir/check.log" 2>&1 <<'PY'
import sys
try:
    from safetensors.torch import load_file
    sd = load_file(sys.argv[1])
except Exception:  # .bin fallback
    import torch
    sd = torch.load(sys.argv[1], map_location="cpu")
from collections import Counter
print("adapter tensor dtypes:", dict(Counter(str(v.dtype) for v in sd.values())))
PY
    grep "adapter tensor dtypes" "$dir/check.log" || true
    # 3) surface dtype errors even on success (best-effort signal, non-fatal)
    grep -ioE "expected scalar type [A-Za-z]+ but found [A-Za-z]+|cuda out of memory|[A-Za-z]*dtype[A-Za-z]* (mismatch|error)[A-Za-z ]*" "$tlog" | sort -u | head -n 5 || true
    return 0
}

SUMMARY="$OUT_ROOT/summary.txt"
printf 'dtype stage result adapter note\n' > "$SUMMARY"
PASS=0; FAIL=0

run_one() {
    # run_one <dtype> <stage: sft|grpo>
    local dtype="$1" stage="$2"
    local bf16_flag=false fp16_flag=false
    case "$dtype" in
        bf16) bf16_flag=true ;;
        fp16) fp16_flag=true ;;
        fp32) ;;
        *) err "unknown dtype '$dtype' (want bf16|fp16|fp32)" ;;
    esac
    local out="$OUT_ROOT/${dtype}_${stage}"
    local tlog="$OUT_ROOT/${dtype}_${stage}.log"
    rm -rf "$out"; mkdir -p "$out"
    rm -f "$tlog"

    log "=== [$dtype/$stage] bf16=$bf16_flag fp16=$fp16_flag bits=$BITS -> $out ==="

    set +e
    if [ "$stage" = "sft" ]; then
        $VENV_PYTHON -u src/train/train_sft.py \
            --model_id "$MODEL_ID" \
            --data_path "$SFT_TRAIN" \
            --eval_path "$SFT_VAL" \
            --image_folder "$SFT_DATASET_ROOT" \
            --output_dir "$out" \
            --bits "$BITS" \
            --lora_enable True --vision_lora True --use_dora False \
            --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
            --num_lora_modules -1 \
            --lora_namespan_exclude "['lm_head', 'embed_tokens', 'merger', 'pos_embed']" \
            --freeze_vision_tower True --freeze_llm True --freeze_merger False \
            --bf16 "$bf16_flag" --fp16 "$fp16_flag" --tf32 True \
            --disable_flash_attn2 True --use_liger_kernel True \
            --num_train_epochs 1 --max_steps "$SFT_STEPS" \
            --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
            --per_device_eval_batch_size 1 \
            --learning_rate 1e-4 --vision_lr 2e-6 --merger_lr 1e-5 \
            --weight_decay 0.0 --warmup_steps 0 --lr_scheduler_type constant \
            --video_min_pixels 65536 --video_max_pixels 131072 \
            --nframes "$NFRAMES" --max_seq_length 4096 \
            --gradient_checkpointing True --lazy_preprocess True \
            --remove_unused_columns False --dataloader_num_workers 0 \
            --logging_steps 1 --eval_strategy no --save_strategy no \
            --report_to none \
            > "$tlog" 2>&1
    else
        $VENV_PYTHON -u src/train/train_grpo.py \
            --model_id "$MODEL_ID" \
            --data_path "$GRPO_TRAIN" \
            --eval_path "$GRPO_VAL" \
            --image_folder "$GRPO_DATASET_ROOT" \
            --output_dir "$out" \
            --bits "$BITS" \
            --lora_enable True --vision_lora True --use_dora False \
            --lora_rank 16 --lora_alpha 32 --lora_dropout 0.0 \
            --num_lora_modules -1 \
            --lora_namespan_exclude "['lm_head', 'embed_tokens', 'merger', 'pos_embed']" \
            --freeze_vision_tower True --freeze_llm True --freeze_merger False \
            --bf16 "$bf16_flag" --fp16 "$fp16_flag" --tf32 True \
            --disable_flash_attn2 True --use_liger_kernel False \
            --num_train_epochs 1 --max_steps "$GRPO_STEPS" \
            --num_generations "$NUM_GENERATIONS" \
            --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
            --max_completion_length "$MAX_COMP" \
            --learning_rate 1e-4 --vision_lr 2e-6 --merger_lr 1e-5 \
            --beta 0.04 --temperature 0.9 --top_p 1.0 \
            --weight_decay 0.0 --warmup_steps 0 --lr_scheduler_type constant \
            --video_min_pixels 65536 --video_max_pixels 131072 \
            --nframes "$NFRAMES" \
            --gradient_checkpointing True --lazy_preprocess True \
            --remove_unused_columns False --dataloader_num_workers 0 \
            --logging_steps 1 --eval_strategy no --save_strategy no \
            --report_to none \
            > "$tlog" 2>&1
    fi
    local rc=$?
    set -e

    local result=FAIL note=""
    if [ $rc -ne 0 ]; then
        note="train exit=$rc"
        warn "[$dtype/$stage] FAILED: $note — tail of $tlog:"
        tail -n 20 "$tlog" || true
    elif verify_adapter "$out" "$tlog"; then
        result=PASS
        note="adapter ok, lora_B non-zero"
        # best-effort loss line for the summary (warn-only if absent)
        loss_line=$(grep -oE "'loss': '?[0-9.e+-]+'?|\"loss\": ?[0-9.e+-]+|loss = [0-9.e+-]+" "$tlog" | tail -n 1 || true)
        note="$note; ${loss_line:-loss n/a}"
    else
        note="adapter verification failed"
    fi

    printf '%-5s %-5s %-6s %s -> %s\n' "$dtype" "$stage" "$result" "$out" "$note" | tee -a "$SUMMARY"
    if [ "$result" = "PASS" ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
}

# ── Matrix ────────────────────────────────────────────────────────────────
for dtype in $DTYPES; do
    for stage in $STAGES; do
        run_one "$dtype" "$stage"
    done
done

echo
log "Results ($PASS passed, $FAIL failed):"
column -t "$SUMMARY" || cat "$SUMMARY"

if [ "$FAIL" -ne 0 ]; then
    err "$FAIL run(s) failed — see $OUT_ROOT/<dtype>_<stage>.log"
fi
log "All dtype runs passed (adapters only, no full checkpoints)."
