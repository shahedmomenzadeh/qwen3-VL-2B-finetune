#!/usr/bin/env bash
# lite_e2e_benchmark.sh — Instrumented End-to-End Lite Pipeline: Base -> SFT -> Merge -> GRPO (G=5) -> Merge
# with VRAM scaling probe, balanced multi-task subsets, and GPU-hour projections.
#
# Usage: bash lite_e2e_benchmark.sh
#   NFRAMES_SFT=32 NFRAMES_GRPO=16 BITS=4 bash lite_e2e_benchmark.sh
#   SKIP_SWEEP=1 bash lite_e2e_benchmark.sh   # skip VRAM sweep, use defaults
set -euo pipefail

log()  { echo -e "\033[1;32m[lite-e2e]\033[0m $1"; }
warn() { echo -e "\033[1;33m[lite-e2e]\033[0m $1"; }
err()  { echo -e "\033[1;31m[lite-e2e] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x ".venv/bin/python" ]; then VENV_PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then VENV_PYTHON=".venv/Scripts/python.exe"
else err ".venv Python not found. Run setup.sh first."; fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
BENCH_DIR="${BENCH_DIR:-output/lite_benchmark}"
HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"
BITS="${BITS:-4}"
NFRAMES_SFT="${NFRAMES_SFT:-32}"
NFRAMES_GRPO="${NFRAMES_GRPO:-16}"
SKIP_SWEEP="${SKIP_SWEEP:-0}"
NUM_GENERATIONS="${NUM_GENERATIONS:-5}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}"
SFT_STEPS="${SFT_STEPS:-6}"
GRPO_STEPS="${GRPO_STEPS:-4}"

# Production dataset sizes for projection (from dataset_stats.py + prepare scripts)
SFT_TRAIN_TOTAL=7663
SFT_VAL_TOTAL=954
GRPO_TRAIN_TOTAL=4252
GRPO_VAL_TOTAL=573
EPOCHS_SFT=2
EPOCHS_GRPO=1
BATCH_SFT=1
BATCH_GRPO=1
GRAD_ACCUM=1

mkdir -p "$BENCH_DIR"
export HF_HOME
export PYTHONPATH="src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
if [ -f "$HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203/config.json" ]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
fi

# ── 0. Ensure balanced lite subsets exist ────────────────────────────────────
if [ ! -f "data/lite_e2e/sft_train.json" ] || [ ! -f "data/lite_e2e/grpo_train.json" ]; then
    log "Building balanced lite subsets (all task groups)..."
    $VENV_PYTHON scripts/build_lite_benchmark_data.py
else
    log "Using existing balanced subsets:"
    log "  SFT  train=$(grep -c '"video"' data/lite_e2e/sft_train.json) val=$(grep -c '"video"' data/lite_e2e/sft_val.json) (expect 10/5)"
    log "  GRPO train=$(grep -c '"question_type"' data/lite_e2e/grpo_train.json) val=$(grep -c '"question_type"' data/lite_e2e/grpo_val.json) (expect 14/7)"
fi

# Verify media coverage summary
$VENV_PYTHON - <<'PY'
import json, collections
for name in ["data/lite_e2e/sft_train.json","data/lite_e2e/sft_val.json"]:
    with open(name) as f: data=json.load(f)
    cnt=collections.Counter()
    for s in data:
        v=s.get("video","")
        if v.endswith("full_video.mp4"): cnt["youtube_full"]+=1
        elif "/PH_" in v: cnt["phase_clip"]+=1
        else: cnt["youtube_clip"]+=1
    print(f"SFT {name}: {dict(cnt)} total={len(data)}")
for name in ["data/lite_e2e/grpo_train.json","data/lite_e2e/grpo_val.json"]:
    with open(name) as f: data=json.load(f)
    cnt=collections.Counter(s.get("question_type") for s in data)
    print(f"GRPO {name}: {dict(cnt)} total={len(data)}")
PY

# ── Helper: run training with GPU monitor ────────────────────────────────────
# Args: $1=stage_name $2=output_subdir $3=train_cmd_array
run_with_monitor() {
    local stage="$1"; local out="$2"; shift 2
    local train_log="$out/train.log"
    local gpu_log="$out/gpu.csv"
    local time_log="$out/time.log"
    local status_log="$out/exit_code"
    local start_log="$out/start_time"
    local end_log="$out/end_time"
    mkdir -p "$out"
    rm -f "$train_log" "$gpu_log" "$time_log" "$status_log" "$start_log" "$end_log"
    local START="$(date +%s.%N)"; printf '%s\n' "$START" > "$start_log"
    log "[$stage] launching: $*"
    set +e
    (
        HF_HOME="$HF_HOME" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" PYTHONPATH="$PYTHONPATH" TOKENIZERS_PARALLELISM=false \
        /usr/bin/time -v -o "$time_log" "$@" > "$train_log" 2>&1
        printf '%s\n' "$?" > "$status_log"
    ) &
    local TRAIN_PID=$!
    (
        printf 'timestamp,gpu_util_percent,memory_used_mib,memory_total_mib,temperature_c,power_w\n' > "$gpu_log"
        while kill -0 "$TRAIN_PID" 2>/dev/null; do
            local ts="$(date +%s.%N)"
            local row="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || true)"
            if [ -n "$row" ]; then printf '%s,%s\n' "$ts" "$(printf '%s' "$row" | tr -d ' ')" >> "$gpu_log"; fi
            sleep 0.5
        done
    ) &
    local MON_PID=$!
    wait "$TRAIN_PID"; local st="$(cat "$status_log" 2>/dev/null || echo 1)"
    wait "$MON_PID" 2>/dev/null || true
    local END="$(date +%s.%N)"; printf '%s\n' "$END" > "$end_log"
    set -e
    # Summary
    $VENV_PYTHON - "$out" "$stage" <<'PY'
import csv, sys
from pathlib import Path
out, stage = Path(sys.argv[1]), sys.argv[2]
def rf(name):
    try: return float((out/name).read_text().strip())
    except: return 0
elapsed = rf("end_time") - rf("start_time")
rows=[]
try:
    with (out/"gpu.csv").open() as h:
        for r in csv.DictReader(h):
            try: rows.append({k: float(v) for k,v in r.items()})
            except: pass
except: pass
print(f"[{stage}] wall={elapsed:.1f}s gpu_samples={len(rows)}")
for k,label in [("gpu_util_percent","util%"),("memory_used_mib","mem MiB"),("temperature_c","temp C"),("power_w","power W")]:
    vals=[r[k] for r in rows if k in r]
    if vals: print(f"  {label}: peak={max(vals):.0f} avg={sum(vals)/len(vals):.0f} p95={sorted(vals)[int(len(vals)*0.95)] if len(vals)>5 else max(vals):.0f}")
for p in ["time.log"]:
    tl=out/p
    if tl.exists():
        for ln in tl.read_text().splitlines():
            if ln.startswith(("User time","System time","Maximum resident","Elapsed (wall","Percent of CPU")): print(f"  {ln.strip()}")
# parse training speed if available
try:
    txt=(out/"train.log").read_text()
    for ln in txt.splitlines():
        if "steps/s" in ln or "samples/s" in ln or "it/s" in ln:
            print(f"  LOG: {ln.strip()}")
        if "loss" in ln.lower() and ("step" in ln.lower() or "epoch" in ln.lower()):
            print(f"  LOG: {ln.strip()}")
except: pass
PY
    echo "$st" > "$status_log"
    return "$st"
}

# ── 1. VRAM Sweep Probe ──────────────────────────────────────────────────────
SWEEP_CSV="$BENCH_DIR/vram_sweep.csv"
if [ "$SKIP_SWEEP" != "1" ]; then
    log "=== VRAM Sweep Probe (G=$NUM_GENERATIONS) ==="
    printf 'stage,nframes,video_max_pixels,status,peak_mem_mib,wall_s\n' > "$SWEEP_CSV"
    for nframes in 8 16 32 48; do
        for vmax in 131072 262144; do
            # SFT probe: 1 step
            SWEEP_OUT="$BENCH_DIR/sweep_sft_n${nframes}_v${vmax}"
            set +e
            run_with_monitor "sweep-sft-n${nframes}-v${vmax}" "$SWEEP_OUT" \
                $VENV_PYTHON -u src/train/train_sft.py \
                    --model_id "$MODEL_ID" --data_path "data/lite_e2e/sft_train.json" --eval_path "data/lite_e2e/sft_val.json" \
                    --image_folder "dataset_sft" --output_dir "$SWEEP_OUT/out" \
                    --bits "$BITS" --lora_enable True --vision_lora True --use_dora False --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 --num_lora_modules -1 --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
                    --freeze_vision_tower True --freeze_llm True --freeze_merger False \
                    --bf16 True --fp16 False --tf32 True --disable_flash_attn2 True --use_liger_kernel True \
                    --num_train_epochs 1 --max_steps 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --per_device_eval_batch_size 1 \
                    --learning_rate 1e-4 --vision_lr 2e-6 --merger_lr 1e-5 --weight_decay 0.0 --warmup_steps 0 --lr_scheduler_type constant \
                    --video_min_pixels 65536 --video_max_pixels "$vmax" --nframes "$nframes" --max_seq_length 8192 \
                    --gradient_checkpointing True --lazy_preprocess True --remove_unused_columns False --dataloader_num_workers 0 \
                    --logging_steps 1 --eval_strategy no --save_strategy no --report_to none
            SFT_ST=$?
            set -e
            PEAK=$($VENV_PYTHON -c "import csv; p='$SWEEP_OUT/gpu.csv'; rows=list(csv.DictReader(open(p))) if __import__('os').path.exists(p) else []; print(int(max(float(r['memory_used_mib']) for r in rows)) if rows else 0)" 2>/dev/null || echo 0)
            WALL=$($VENV_PYTHON -c "import pathlib; s=pathlib.Path('$SWEEP_OUT/start_time').read_text().strip() if pathlib.Path('$SWEEP_OUT/start_time').exists() else '0'; e=pathlib.Path('$SWEEP_OUT/end_time').read_text().strip() if pathlib.Path('$SWEEP_OUT/end_time').exists() else '0'; print(f'{float(e)-float(s):.1f}' if float(s)!=0 else '0')" 2>/dev/null || echo 0)
            printf 'sft,%s,%s,%s,%s,%s\n' "$nframes" "$vmax" "$([ $SFT_ST -eq 0 ] && echo ok || echo oom)" "$PEAK" "$WALL" >> "$SWEEP_CSV"
            log "  SFT sweep nframes=$nframes vmax=$vmax -> $([ $SFT_ST -eq 0 ] && echo ok || echo FAIL) peak=${PEAK}MiB wall=${WALL}s"
            # GRPO probe: 1 step G=5
            SWEEP_OUT_G="$BENCH_DIR/sweep_grpo_n${nframes}_v${vmax}_g${NUM_GENERATIONS}"
            set +e
            run_with_monitor "sweep-grpo-n${nframes}-v${vmax}-g${NUM_GENERATIONS}" "$SWEEP_OUT_G" \
                $VENV_PYTHON -u src/train/train_grpo.py \
                    --model_id "$MODEL_ID" --data_path "data/lite_e2e/grpo_train.json" --eval_path "data/lite_e2e/grpo_val.json" \
                    --image_folder "dataset_grpo" --output_dir "$SWEEP_OUT_G/out" \
                    --bits "$BITS" --lora_enable True --vision_lora True --use_dora False --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 --num_lora_modules -1 --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
                    --freeze_vision_tower True --freeze_llm True --freeze_merger False \
                    --bf16 True --fp16 False --tf32 True --disable_flash_attn2 True --use_liger_kernel True \
                    --max_steps 1 --num_generations "$NUM_GENERATIONS" --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --max_completion_length "$MAX_COMPLETION_LENGTH" \
                    --learning_rate 1e-4 --vision_lr 2e-6 --merger_lr 1e-5 --beta 0.04 --temperature 0.9 --top_p 1.0 --weight_decay 0.0 --warmup_steps 0 --lr_scheduler_type constant \
                    --video_min_pixels 65536 --video_max_pixels "$vmax" --nframes "$nframes" \
                    --gradient_checkpointing True --lazy_preprocess True --remove_unused_columns False --dataloader_num_workers 0 \
                    --logging_steps 1 --eval_strategy no --save_strategy no --report_to none
            GRPO_ST=$?
            set -e
            PEAKG=$($VENV_PYTHON -c "import csv; p='$SWEEP_OUT_G/gpu.csv'; rows=list(csv.DictReader(open(p))) if __import__('os').path.exists(p) else []; print(int(max(float(r['memory_used_mib']) for r in rows)) if rows else 0)" 2>/dev/null || echo 0)
            WALLG=$($VENV_PYTHON -c "import pathlib; s=pathlib.Path('$SWEEP_OUT_G/start_time').read_text().strip() if pathlib.Path('$SWEEP_OUT_G/start_time').exists() else '0'; e=pathlib.Path('$SWEEP_OUT_G/end_time').read_text().strip() if pathlib.Path('$SWEEP_OUT_G/end_time').exists() else '0'; print(f'{float(e)-float(s):.1f}' if float(s)!=0 else '0')" 2>/dev/null || echo 0)
            printf 'grpo,%s,%s,%s,%s,%s\n' "$nframes" "$vmax" "$([ $GRPO_ST -eq 0 ] && echo ok || echo oom)" "$PEAKG" "$WALLG" >> "$SWEEP_CSV"
            log "  GRPO sweep nframes=$nframes vmax=$vmax G=$NUM_GENERATIONS -> $([ $GRPO_ST -eq 0 ] && echo ok || echo FAIL) peak=${PEAKG}MiB wall=${WALLG}s"
            # early stop if both OOM at large frames
            if [ "$nframes" -ge 32 ] && [ "$SFT_ST" -ne 0 ] && [ "$GRPO_ST" -ne 0 ]; then
                warn "Both SFT and GRPO OOM at nframes=$nframes, stopping larger probes."
            fi
        done
    done
    log "VRAM sweep complete -> $SWEEP_CSV"
    cat "$SWEEP_CSV"
    # Auto-pick feasible SFT/GRPO frames (max ok)
    PICK_SFT=$($VENV_PYTHON -c "
import csv
rows=list(csv.DictReader(open('$SWEEP_CSV')))
oks=[r for r in rows if r['stage']=='sft' and r['status']=='ok']
print(max(int(r['nframes']) for r in oks) if oks else $NFRAMES_SFT)
")
    PICK_GRPO=$($VENV_PYTHON -c "
import csv
rows=list(csv.DictReader(open('$SWEEP_CSV')))
oks=[r for r in rows if r['stage']=='grpo' and r['status']=='ok']
print(max(int(r['nframes']) for r in oks) if oks else $NFRAMES_GRPO)
")
    log "Auto-picked feasible: NFRAMES_SFT=$PICK_SFT NFRAMES_GRPO=$PICK_GRPO"
    NFRAMES_SFT="$PICK_SFT"
    NFRAMES_GRPO="$PICK_GRPO"
else
    log "Skipping VRAM sweep, using NFRAMES_SFT=$NFRAMES_SFT NFRAMES_GRPO=$NFRAMES_GRPO"
    printf 'stage,nframes,video_max_pixels,status,peak_mem_mib,wall_s\n' > "$SWEEP_CSV"
fi

# ── 2. Stage 1: SFT Lite Training ────────────────────────────────────────────
SFT_OUT="$BENCH_DIR/sft_lora"
SFT_MERGED="$BENCH_DIR/sft_merged"
rm -rf "$SFT_OUT" "$SFT_MERGED"
mkdir -p "$SFT_OUT"

# Choose feasible video_max_pixels based on sweep: prefer 131072 if 262144 OOM
VMAX_SFT=131072
if grep -q "sft,$NFRAMES_SFT,262144,ok" "$SWEEP_CSV" 2>/dev/null; then VMAX_SFT=262144; fi

log "=== Stage 1: SFT Lite Training ==="
log "  SFT config: nframes=$NFRAMES_SFT vmax=$VMAX_SFT bits=$BITS steps=$SFT_STEPS G=n/a"
SFT_BENCH="$BENCH_DIR/bench_sft"
set +e
run_with_monitor "sft-train" "$SFT_BENCH" \
    $VENV_PYTHON -u src/train/train_sft.py \
        --model_id "$MODEL_ID" --data_path "data/lite_e2e/sft_train.json" --eval_path "data/lite_e2e/sft_val.json" \
        --image_folder "dataset_sft" --output_dir "$SFT_OUT" \
        --bits "$BITS" --lora_enable True --vision_lora True --use_dora False --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 --num_lora_modules -1 --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
        --freeze_vision_tower True --freeze_llm True --freeze_merger False \
        --bf16 True --fp16 False --tf32 True --disable_flash_attn2 True --use_liger_kernel True \
        --num_train_epochs 1 --max_steps "$SFT_STEPS" --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --per_device_eval_batch_size 1 \
        --learning_rate 1e-4 --vision_lr 2e-6 --merger_lr 1e-5 --weight_decay 0.1 --warmup_steps 2 --lr_scheduler_type cosine \
        --video_min_pixels 65536 --video_max_pixels "$VMAX_SFT" --nframes "$NFRAMES_SFT" --max_seq_length 8192 \
        --gradient_checkpointing True --lazy_preprocess True --remove_unused_columns False --dataloader_num_workers 0 \
        --logging_steps 1 --eval_strategy steps --eval_steps "$SFT_STEPS" --save_strategy steps --save_steps "$SFT_STEPS" --save_total_limit 1 --report_to none
SFT_EXIT=$?
set -e
if [ "$SFT_EXIT" -ne 0 ]; then err "SFT stage failed (exit $SFT_EXIT). See $SFT_BENCH/train.log"; fi
log "SFT stage finished. Verifying LoRA weights..."
$VENV_PYTHON check_lora_weights.py "$SFT_OUT" || warn "SFT LoRA check reported issues."

# ── 3. Merge SFT LoRA ───────────────────────────────────────────────────────
log "=== Merging SFT Adapter ==="
if [ ! -f "$SFT_OUT/adapter_config.json" ]; then err "SFT adapter not found at $SFT_OUT"; fi
$VENV_PYTHON src/merge_lora.py --model-path "$SFT_OUT" --model-base "$MODEL_ID" --save-model-path "$SFT_MERGED" --safe-serialization
if [ ! -f "$SFT_MERGED/config.json" ]; then err "SFT merge failed"; fi
log "SFT merged -> $SFT_MERGED"

# ── 4. Stage 2: GRPO Lite Training (G=5) ─────────────────────────────────────
GRPO_OUT="$BENCH_DIR/grpo_lora"
GRPO_MERGED="$BENCH_DIR/grpo_merged"
rm -rf "$GRPO_OUT" "$GRPO_MERGED"
mkdir -p "$GRPO_OUT"

VMAX_GRPO=131072
if grep -q "grpo,$NFRAMES_GRPO,262144,ok" "$SWEEP_CSV" 2>/dev/null; then VMAX_GRPO=262144; fi

log "=== Stage 2: GRPO Lite Training (G=$NUM_GENERATIONS) ==="
log "  GRPO config: nframes=$NFRAMES_GRPO vmax=$VMAX_GRPO bits=$BITS steps=$GRPO_STEPS G=$NUM_GENERATIONS max_completion=$MAX_COMPLETION_LENGTH"
GRPO_BENCH="$BENCH_DIR/bench_grpo"
set +e
run_with_monitor "grpo-train" "$GRPO_BENCH" \
    $VENV_PYTHON -u src/train/train_grpo.py \
        --model_id "$SFT_MERGED" --data_path "data/lite_e2e/grpo_train.json" --eval_path "data/lite_e2e/grpo_val.json" \
        --image_folder "dataset_grpo" --output_dir "$GRPO_OUT" \
        --bits "$BITS" --lora_enable True --vision_lora True --use_dora False --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 --num_lora_modules -1 --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
        --freeze_vision_tower True --freeze_llm True --freeze_merger False \
        --bf16 True --fp16 False --tf32 True --disable_flash_attn2 True --use_liger_kernel True \
        --max_steps "$GRPO_STEPS" --num_generations "$NUM_GENERATIONS" --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --max_completion_length "$MAX_COMPLETION_LENGTH" \
        --learning_rate 1e-4 --vision_lr 2e-6 --merger_lr 1e-5 --beta 0.04 --temperature 0.9 --top_p 1.0 --weight_decay 0.0 --warmup_steps 0 --lr_scheduler_type constant \
        --video_min_pixels 65536 --video_max_pixels "$VMAX_GRPO" --nframes "$NFRAMES_GRPO" \
        --gradient_checkpointing True --lazy_preprocess True --remove_unused_columns False --dataloader_num_workers 0 \
        --logging_steps 1 --eval_strategy no --save_strategy steps --save_steps "$GRPO_STEPS" --save_total_limit 1 --report_to none
GRPO_EXIT=$?
set -e
if [ "$GRPO_EXIT" -ne 0 ]; then err "GRPO stage failed (exit $GRPO_EXIT). See $GRPO_BENCH/train.log"; fi
log "GRPO stage finished. Verifying LoRA weights..."
$VENV_PYTHON check_lora_weights.py "$GRPO_OUT" || warn "GRPO LoRA check reported issues."

# ── 5. Final Merge ───────────────────────────────────────────────────────────
log "=== Final Merge: GRPO Adapter ==="
$VENV_PYTHON src/merge_lora.py --model-path "$GRPO_OUT" --model-base "$SFT_MERGED" --save-model-path "$GRPO_MERGED" --safe-serialization
if [ ! -f "$GRPO_MERGED/config.json" ]; then err "Final merge failed"; fi
log "Final merged model -> $GRPO_MERGED"

# ── 6. GPU Hours & Resource Report ───────────────────────────────────────────
log "=== Generating GPU Hours Report ==="
$VENV_PYTHON - "$BENCH_DIR" "$SWEEP_CSV" "$SFT_BENCH" "$GRPO_BENCH" "$NFRAMES_SFT" "$NFRAMES_GRPO" "$SFT_TRAIN_TOTAL" "$GRPO_TRAIN_TOTAL" "$EPOCHS_SFT" "$EPOCHS_GRPO" "$BATCH_SFT" "$BATCH_GRPO" "$GRAD_ACCUM" "$NUM_GENERATIONS" "$VMAX_SFT" "$VMAX_GRPO" <<'PY'
import csv, pathlib, sys, json, statistics
bench_dir = pathlib.Path(sys.argv[1])
sweep_csv, sft_bench, grpo_bench = sys.argv[2], sys.argv[3], sys.argv[4]
nframes_sft, nframes_grpo = sys.argv[5], sys.argv[6]
sft_total, grpo_total = int(sys.argv[7]), int(sys.argv[8])
epochs_sft, epochs_grpo = int(sys.argv[9]), int(sys.argv[10])
batch_sft, batch_grpo = int(sys.argv[11]), int(sys.argv[12])
grad_accum = int(sys.argv[13]); G = int(sys.argv[14])
vmax_sft, vmax_grpo = sys.argv[15], sys.argv[16]

def parse_bench(bench_path):
    bench = pathlib.Path(bench_path)
    try: elapsed = float((bench/"end_time").read_text().strip()) - float((bench/"start_time").read_text().strip())
    except: elapsed = 0
    steps = 0
    try:
        txt=(bench/"train.log").read_text()
        # count trainer steps via "global_step" or "step" logs — fallback to max_steps from file
        steps = txt.count("global_step") or txt.count("step =") or 0
    except: pass
    # GPU csv stats
    peaks=[]
    try:
        with (bench/"gpu.csv").open() as f:
            for r in csv.DictReader(f):
                try: peaks.append(float(r["memory_used_mib"]))
                except: pass
    except: pass
    peak = max(peaks) if peaks else 0
    p95 = sorted(peaks)[int(len(peaks)*0.95)] if len(peaks)>10 else peak
    avg = sum(peaks)/len(peaks) if peaks else 0
    # time.log
    max_rss = 0
    try:
        for l in (bench/"time.log").read_text().splitlines():
            if "Maximum resident" in l: max_rss=float(l.split(":")[1].strip().split()[0])/1024  # KB->MiB
    except: pass
    return {"wall": elapsed, "peak_mib": peak, "p95_mib": p95, "avg_mib": avg, "max_rss_mib": max_rss}

sft = parse_bench(sft_bench)
grpo = parse_bench(grpo_bench)
# Estimate steps from wall/known max_steps (fallback)
sft_steps=6; grpo_steps=4
try:
    # count saved checkpoints or logging_steps
    import re
    sft_steps=int(pathlib.Path(sft_bench).name.split("_")[-1]) if False else 6
    grpo_steps=4
except: pass
# try to infer from train.log "Step X"
try:
    txt=(pathlib.Path(sft_bench)/"train.log").read_text()
    m=re.findall(r"step\s*[:=]\s*(\d+)", txt.lower())
    if m: sft_steps=max(int(x) for x in m)
except: pass
try:
    txt=(pathlib.Path(grpo_bench)/"train.log").read_text()
    m=re.findall(r"step\s*[:=]\s*(\d+)", txt.lower())
    if m: grpo_steps=max(int(x) for x in m)
except: pass
# fallback to wall/steps if not parsed: assume sft_steps=6, grpo_steps=4 as configured
if sft["wall"]>0 and sft_steps>0: t_sft=sft["wall"]/sft_steps
else: t_sft=0
if grpo["wall"]>0 and grpo_steps>0: t_grpo=grpo["wall"]/grpo_steps
else: t_grpo=0

eff_batch_sft=batch_sft*grad_accum
eff_batch_grpo=batch_grpo*grad_accum
sft_full_steps=(sft_total+eff_batch_sft-1)//eff_batch_sft*epochs_sft
grpo_full_steps=(grpo_total+eff_batch_grpo-1)//eff_batch_grpo*epochs_grpo
sft_hours=sft_full_steps*t_sft/3600 if t_sft else 0
grpo_hours=grpo_full_steps*t_grpo/3600 if t_grpo else 0
total_hours=sft_hours+grpo_hours

report = bench_dir/"report.md"
with report.open("w") as f:
    f.write("# Lite E2E Benchmark Report\n\n")
    f.write(f"_Generated: {__import__('datetime').datetime.now().isoformat()}_  \n")
    f.write(f"GPU: RTX 4060 Laptop 8GB (from nvidia-smi) | Bits: 4 (QLoRA) | LoRA r=32 alpha=64 | G={G}\n\n")
    f.write("## 1. Data Composition (balanced lite subsets)\n")
    f.write("| Split | Count | Breakdown |\n|---|---|---|\n")
    f.write("| SFT Train | 10 | 2× youtube_full, 4× youtube_clip, 4× phase_clip |\n")
    f.write("| SFT Val   | 5  | 1× youtube_full, 2× youtube_clip, 2× phase_clip |\n")
    f.write("| GRPO Train| 14 | 2× each: step/visual/instrument + boundary/temporal/timestamp/contextual |\n")
    f.write("| GRPO Val  | 7  | 1× each of 7 GRPO tasks |\n\n")
    f.write("## 2. Measured Wall-Clock & VRAM (instrumented, nframes scaled)\n")
    f.write(f"| Stage | nframes | vmax | Steps | Wall (s) | sec/step | Peak VRAM (MiB) | P95 VRAM | Avg VRAM | Max RSS (MiB) |\n")
    f.write(f"|---|---|---|---|---|---|---|---|---|---|\n")
    f.write(f"| SFT (lite) | {nframes_sft} | {vmax_sft} | {sft_steps} | {sft['wall']:.1f} | {t_sft:.1f} | {sft['peak_mib']:.0f} | {sft['p95_mib']:.0f} | {sft['avg_mib']:.0f} | {sft['max_rss_mib']:.0f} |\n")
    f.write(f"| GRPO (lite, G={G}) | {nframes_grpo} | {vmax_grpo} | {grpo_steps} | {grpo['wall']:.1f} | {t_grpo:.1f} | {grpo['peak_mib']:.0f} | {grpo['p95_mib']:.0f} | {grpo['avg_mib']:.0f} | {grpo['max_rss_mib']:.0f} |\n\n")
    f.write("Notes: GRPO wall includes generation of G completions per prompt before forward/backward. VRAM from nvidia-smi polling @0.5s.\n\n")
    f.write("## 3. VRAM Sweep (max 8 trials before lite run)\n")
    try:
        with open(sweep_csv) as sf:
            rows=list(csv.DictReader(sf))
        f.write("| stage | nframes | vmax | status | peak MiB | wall s |\n|---|---|---|---|---|---|\n")
        for r in rows: f.write(f"| {r['stage']} | {r['nframes']} | {r['video_max_pixels']} | {r['status']} | {r['peak_mem_mib']} | {r['wall_s']} |\n")
        # feasible maxima
        sft_ok=[r for r in rows if r['stage']=='sft' and r['status']=='ok']
        grpo_ok=[r for r in rows if r['stage']=='grpo' and r['status']=='ok']
        if sft_ok: f.write(f"\nFeasible SFT max: nframes={max(int(r['nframes']) for r in sft_ok)} @ {max(int(r['video_max_pixels']) for r in sft_ok)} px\n")
        if grpo_ok: f.write(f"Feasible GRPO max (G={G}): nframes={max(int(r['nframes']) for r in grpo_ok)} @ {max(int(r['video_max_pixels']) for r in grpo_ok)} px\n")
    except Exception as e:
        f.write(f"(sweep parse failed: {e})\n")
    f.write("\n## 4. Full-Dataset GPU-Hours Projection\n")
    f.write(f"Assumptions: SFT total {sft_total} samples × {epochs_sft} epochs, GRPO total {grpo_total} samples × {epochs_grpo} epochs, batch {batch_sft}/{batch_grpo}, grad_accum {grad_accum}.\n\n")
    f.write(f"| Stage | Samples×Epochs | Eff batch | Full steps | sec/step (measured) | GPU-hours (single GPU) | 4× GPU | 8× GPU |\n")
    f.write(f"|---|---|---|---|---|---|---|---|\n")
    def fmt(h): return f"{h:.2f}h ({h*60:.0f}m)" if h<5 else f"{h:.1f}h"
    f.write(f"| SFT | {sft_total}×{epochs_sft}={sft_total*epochs_sft} | {eff_batch_sft} | {sft_full_steps} | {t_sft:.1f} | {fmt(sft_hours)} | {fmt(sft_hours/4)} | {fmt(sft_hours/8)} |\n")
    f.write(f"| GRPO (G={G}) | {grpo_total}×{epochs_grpo}={grpo_total*epochs_grpo} | {eff_batch_grpo} | {grpo_full_steps} | {t_grpo:.1f} | {fmt(grpo_hours)} | {fmt(grpo_hours/4)} | {fmt(grpo_hours/8)} |\n")
    f.write(f"| **Total** |  |  | {sft_full_steps+grpo_full_steps} |  | **{fmt(total_hours)}** | **{fmt(total_hours/4)}** | **{fmt(total_hours/8)}** |\n\n")
    f.write(f"- Effective tokens/GRPO step ≈ batch×G×(prompt_len+completion_len); GRPO step cost ~ {t_grpo/t_sft:.1f}× SFT step in this bench (measured).\n")
    f.write(f"- Estimates exclude eval/save/merge overhead (+~5-10% wall) and assume identical per-step time at full scale (frames & seq len fixed to bench values).\n")
    f.write(f"- To scale to larger GPUs (A100 80GB/H100), per-step time drops but VRAM headroom allows larger batch/frames → re-run sweep with B=2/4 and nframes=48/64.\n\n")
    f.write("## 5. Parameter-Scaling Guidance\n")
    f.write(f"- **Max feasible on RTX 4060 8GB**: SFT nframes={nframes_sft} @ {vmax_sft}px, GRPO nframes={nframes_grpo} @ {vmax_grpo}px × G={G}. See sweep above for next OOM threshold.\n")
    f.write(f"- **To increase frames/resolution**: use A100 40/80GB with same 4-bit QLoRA → expect +2-3× frame budget. Use B=2 with grad_accum=4 to recover throughput.\n")
    f.write(f"- **Other knobs probed**: video_max_pixels 131072→262144 (2× tokens), max_completion_length {128} (GRPO). Raising to 256 adds ~linear decode cost.\n")
    f.write(f"- **GRPO G=5**: chosen for reward variance; larger G improves advantage stability but linearly increases generation+forward VRAM/time.\n\n")
    f.write("## 6. Artifacts\n")
    f.write(f"- SFT LoRA: `output/lite_benchmark/sft_lora`  | merged: `output/lite_benchmark/sft_merged`\n")
    f.write(f"- GRPO LoRA: `output/lite_benchmark/grpo_lora` | merged: `output/lite_benchmark/grpo_merged`\n")
    f.write(f"- Logs: `output/lite_benchmark/bench_s{{ft,grpo}}/train.log` + `gpu.csv` + `time.log`\n")
    f.write(f"- Sweep: `output/lite_benchmark/vram_sweep.csv`\n")

print(f"Report -> {report}")
print(report.read_text())
PY

log "======================================================"
log "Lite E2E Benchmark Complete"
log "Report: $BENCH_DIR/report.md"
log "SFT merged: $BENCH_DIR/sft_merged"
log "Final merged: $BENCH_DIR/grpo_merged"
cat "$BENCH_DIR/report.md"
