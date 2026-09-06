#!/usr/bin/env python3
"""scripts/summarize_run.py — (re)build losses.csv/eval_losses.csv/summary.txt from a run dir.

Usage:
    python scripts/summarize_run.py <run_dir> <label>

Reads <run_dir>/train.log (+ gpu.csv, start_time/end_time/exit_code when
present) and writes losses.csv, eval_losses.csv, summary.txt. Used live by
scripts/run_instrumented.sh and standalone to backfill runs whose summary
step predates a parser fix:

    .venv/bin/python scripts/summarize_run.py output/logs/sft sft
"""

import csv
import re
import sys
from pathlib import Path


def main() -> None:
    run = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else run.name
    out = []

    # --- wall time (optional files: absent for ad-hoc dirs)
    try:
        elapsed = float((run / "end_time").read_text().strip()) - float(
            (run / "start_time").read_text().strip()
        )
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        out.append(f"[{label}] wall time: {int(h)}h {int(m)}m {s:.1f}s ({elapsed:.1f} s)")
    except OSError:
        pass

    # --- losses: HF Trainer dict lines.
    # Values may be quoted strings ('loss': '1.861') or raw numbers — accept both.
    num = r"(-?\d[\d.eE+-]*)"
    val = rf"['\"]?\s*{num}\s*['\"]?"
    train_rows, eval_rows = [], []
    log_text = (run / "train.log").read_text(errors="replace").splitlines()
    for line in log_text:
        mt = re.search(rf"['\"]loss['\"]\s*:\s*{val}", line)
        if mt:
            def grab(key):
                mm = re.search(rf"['\"]{key}['\"]\s*:\s*{val}", line)
                return mm.group(1) if mm else ""

            train_rows.append((mt.group(1), grab("grad_norm"), grab("learning_rate"), grab("epoch")))
            continue
        me = re.search(rf"['\"]eval_loss['\"]\s*:\s*{val}", line)
        if me:
            mg = re.search(rf"['\"]epoch['\"]\s*:\s*{val}", line)
            eval_rows.append((me.group(1), mg.group(1) if mg else ""))

    with (run / "losses.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["idx", "loss", "grad_norm", "learning_rate", "epoch"])
        for i, r in enumerate(train_rows):
            w.writerow([i, *r])
    with (run / "eval_losses.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["idx", "eval_loss", "epoch"])
        for i, r in enumerate(eval_rows):
            w.writerow([i, *r])

    if train_rows:
        losses = [float(r[0]) for r in train_rows]
        out.append(
            f"[{label}] train loss: n={len(losses)} first={losses[0]:.4f} "
            f"last={losses[-1]:.4f} best={min(losses):.4f}"
        )
    else:
        out.append(f"[{label}] train loss: no loss lines found in train.log")
    if eval_rows:
        elosses = [float(r[0]) for r in eval_rows]
        out.append(f"[{label}] eval loss: n={len(elosses)} last={elosses[-1]:.4f} best={min(elosses):.4f}")

    # --- GPU stats (optional file)
    grows = []
    try:
        with (run / "gpu.csv").open() as h:
            first = h.readline()
            if first and not first.startswith("#"):
                grows = list(csv.DictReader([first, *h]))
    except OSError:
        pass
    if grows or (run / "gpu.csv").exists():
        out.append(f"[{label}] GPU samples: {len(grows)}")
        by_gpu: dict = {}
        for r in grows:
            try:
                by_gpu.setdefault(r["gpu_index"], []).append(
                    (float(r["memory_used_mib"]), float(r["gpu_util_percent"]))
                )
            except (KeyError, ValueError):
                pass
        for idx in sorted(by_gpu):
            mems = [x[0] for x in by_gpu[idx]]
            utils = [x[1] for x in by_gpu[idx]]
            out.append(
                f"[{label}] gpu{idx}: mem peak={max(mems):.0f} MiB avg={sum(mems)/len(mems):.0f} MiB | "
                f"util peak={max(utils):.0f}% avg={sum(utils)/len(utils):.0f}%"
            )
        try:
            temps = [float(r["temperature_c"]) for r in grows]
            powers = [float(r["power_w"]) for r in grows]
            if temps:
                out.append(f"[{label}] temp peak={max(temps):.0f}C | power peak={max(powers):.0f}W")
        except (KeyError, ValueError):
            pass

    if (run / "exit_code").exists():
        out.append(f"[{label}] exit={(run / 'exit_code').read_text().strip()}")
    out.append(f"[{label}] logs: {run}/train.log gpu.csv losses.csv summary.txt")

    (run / "summary.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
