#!/usr/bin/env bash
# scripts/run_instrumented.sh — run a training command with full instrumentation.
#
# Captures everything the lite scripts capture (train.log, gpu.csv, timing),
# plus parsed losses and a summary, in one shared helper for all training
# entry points (train_sft.sh, train.sh, grpo_train.sh, test_dtypes.sh).
#
# Usage:
#   bash scripts/run_instrumented.sh <run_dir> <label> <cmd> [args...]
#
# Outputs in <run_dir>/:
#   train.log       — full stdout+stderr of the command (also streamed to console)
#   gpu.csv         — nvidia-smi samples: timestamp,gpu_index,util%,mem_used/total,temp,power
#   cmd.txt         — the exact command line run
#   start_time / end_time — wall-clock epoch seconds (float)
#   exit_code       — the training command's exit status
#   losses.csv      — parsed per-step train loss rows (idx,loss,grad_norm,learning_rate,epoch)
#   eval_losses.csv — parsed eval rows, best effort (idx,eval_loss,epoch)
#   summary.txt     — wall time, loss first/last/best, GPU peak/avg
#
# Env:
#   GPU_POLL_SEC — nvidia-smi sampling interval in seconds (default 5;
#                  lite scripts use 0.5s; prod runs are hours long so 5s keeps
#                  gpu.csv small while still catching OOM-adjacent peaks)
#   VENV_PYTHON  — python used for the summary step (falls back to python3)
#
# Exit status: the training command's exit status (callers with `set -e`
# abort on failure; add `|| err ...` for a pointer to train.log).
set -uo pipefail

RUN_DIR="${1:?usage: run_instrumented.sh <run_dir> <label> <cmd...>}"
LABEL="${2:?usage: run_instrumented.sh <run_dir> <label> <cmd...>}"
shift 2
[ "$#" -gt 0 ] || { echo "[run_instrumented] ERROR: no command given" >&2; exit 2; }

mkdir -p "$RUN_DIR"
TRAIN_LOG="$RUN_DIR/train.log"
GPU_LOG="$RUN_DIR/gpu.csv"
rm -f "$TRAIN_LOG" "$GPU_LOG" "$RUN_DIR/start_time" "$RUN_DIR/end_time" \
      "$RUN_DIR/exit_code" "$RUN_DIR/cmd.txt" "$RUN_DIR/losses.csv" \
      "$RUN_DIR/eval_losses.csv" "$RUN_DIR/summary.txt"
printf '%q ' "$@" > "$RUN_DIR/cmd.txt"; printf '\n' >> "$RUN_DIR/cmd.txt"

printf '%s\n' "$(date +%s.%N)" > "$RUN_DIR/start_time"

# Training in background, direct redirect (exit code is exactly the command's).
"$@" > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!

# Stream the log to console; GNU tail --pid exits on its own when training ends.
tail --pid="$TRAIN_PID" -F -n +1 "$TRAIN_LOG" 2>/dev/null &
TAIL_PID=$!

# GPU telemetry (best effort: skipped cleanly when nvidia-smi is absent).
POLL="${GPU_POLL_SEC:-5}"
MON_PID=""
if command -v nvidia-smi &>/dev/null; then
    (
        printf 'timestamp,gpu_index,gpu_util_percent,memory_used_mib,memory_total_mib,temperature_c,power_w\n' > "$GPU_LOG"
        while kill -0 "$TRAIN_PID" 2>/dev/null; do
            ts="$(date +%s.%N)"
            nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
                --format=csv,noheader,nounits 2>/dev/null \
                | tr -d ' ' | while IFS= read -r row; do
                    [ -n "$row" ] && printf '%s,%s\n' "$ts" "$row"
                done >> "$GPU_LOG"
            sleep "$POLL"
        done
    ) &
    MON_PID=$!
else
    printf '# nvidia-smi not found — no GPU telemetry\n' > "$GPU_LOG"
fi

wait "$TRAIN_PID"; STATUS=$?
[ -n "$MON_PID" ] && wait "$MON_PID" 2>/dev/null || true
wait "$TAIL_PID" 2>/dev/null || true
printf '%s\n' "$(date +%s.%N)" > "$RUN_DIR/end_time"
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_code"

# Summary step (must never fail the helper — findings are advisory).
# Logic lives in scripts/summarize_run.py so old run dirs can be
# (re)summarized standalone: .venv/bin/python scripts/summarize_run.py <run_dir> <label>
PY="${VENV_PYTHON:-}"
{ [ -n "$PY" ] && [ -x "$PY" ]; } || PY="python3"
HELPER_DIR="$(cd "$(dirname "$0")" && pwd)"
"$PY" "$HELPER_DIR/summarize_run.py" "$RUN_DIR" "$LABEL" \
    || echo "[run_instrumented] WARNING: summary step failed" >&2
exit "$STATUS"
