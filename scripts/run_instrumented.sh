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
PY="${VENV_PYTHON:-}"
{ [ -n "$PY" ] && [ -x "$PY" ]; } || PY="python3"
"$PY" - "$RUN_DIR" "$LABEL" <<'PY' || echo "[run_instrumented] WARNING: summary step failed" >&2
import csv, re, sys
from pathlib import Path

run = Path(sys.argv[1]); label = sys.argv[2]
out = []

elapsed = float((run / "end_time").read_text().strip()) - float((run / "start_time").read_text().strip())
h, rem = divmod(elapsed, 3600); m, s = divmod(rem, 60)
out.append(f"[{label}] wall time: {int(h)}h {int(m)}m {s:.1f}s ({elapsed:.1f} s)")

# --- losses: HF Trainer dict lines, e.g. {'loss': 2.27, 'grad_norm': 1.5, 'learning_rate': 1e-4, 'epoch': 0.05}
num = r"(-?\d[\d.eE+-]*)"
train_rows, eval_rows = [], []
log_text = (run / "train.log").read_text(errors="replace").splitlines()
for line in log_text:
    mt = re.search(rf"['\"]loss['\"]\s*:\s*{num}", line)
    if mt:
        def grab(key):
            mm = re.search(rf"['\"]{key}['\"]\s*:\s*{num}", line)
            return mm.group(1) if mm else ""
        train_rows.append((mt.group(1), grab("grad_norm"), grab("learning_rate"), grab("epoch")))
        continue
    me = re.search(rf"['\"]eval_loss['\"]\s*:\s*{num}", line)
    if me:
        mg = re.search(rf"['\"]epoch['\"]\s*:\s*{num}", line)
        eval_rows.append((me.group(1), mg.group(1) if mg else ""))

with (run / "losses.csv").open("w", newline="") as h:
    w = csv.writer(h); w.writerow(["idx", "loss", "grad_norm", "learning_rate", "epoch"])
    for i, r in enumerate(train_rows): w.writerow([i, *r])
with (run / "eval_losses.csv").open("w", newline="") as h:
    w = csv.writer(h); w.writerow(["idx", "eval_loss", "epoch"])
    for i, r in enumerate(eval_rows): w.writerow([i, *r])

if train_rows:
    losses = [float(r[0]) for r in train_rows]
    out.append(f"[{label}] train loss: n={len(losses)} first={losses[0]:.4f} last={losses[-1]:.4f} best={min(losses):.4f}")
else:
    out.append(f"[{label}] train loss: no loss lines found in train.log")
if eval_rows:
    elosses = [float(r[0]) for r in eval_rows]
    out.append(f"[{label}] eval loss: n={len(elosses)} last={elosses[-1]:.4f} best={min(elosses):.4f}")

# --- GPU stats
grows = []
try:
    with (run / "gpu.csv").open() as h:
        first = h.readline()
        if first and not first.startswith("#"):
            grows = list(csv.DictReader([first, *h]))
except OSError:
    pass
out.append(f"[{label}] GPU samples: {len(grows)}")
by_gpu = {}
for r in grows:
    try:
        by_gpu.setdefault(r["gpu_index"], []).append((float(r["memory_used_mib"]), float(r["gpu_util_percent"])))
    except (KeyError, ValueError):
        pass
for idx in sorted(by_gpu):
    mems = [x[0] for x in by_gpu[idx]]; utils = [x[1] for x in by_gpu[idx]]
    out.append(f"[{label}] gpu{idx}: mem peak={max(mems):.0f} MiB avg={sum(mems)/len(mems):.0f} MiB | util peak={max(utils):.0f}% avg={sum(utils)/len(utils):.0f}%")
try:
    temps = [float(r["temperature_c"]) for r in grows]; powers = [float(r["power_w"]) for r in grows]
    if temps: out.append(f"[{label}] temp peak={max(temps):.0f}C | power peak={max(powers):.0f}W")
except (KeyError, ValueError):
    pass

status = (run / "exit_code").read_text().strip()
out.append(f"[{label}] exit={status} | logs: {run}/train.log gpu.csv losses.csv summary.txt")

(run / "summary.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
PY

exit "$STATUS"
