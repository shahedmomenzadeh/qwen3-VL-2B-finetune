#!/usr/bin/env python3
"""
merge_tensorboard.py — Merge Stage 4 TensorBoard event files into the primary
TensorBoard run directory with a step offset of +532.
"""

import sys
import glob
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

def merge_tb_runs(src_dir: str, dst_dir: str, step_offset: int = 532):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    if not src_path.exists():
        print(f"Source dir {src_dir} does not exist yet.")
        return

    # Find all event files in source dir
    event_files = sorted(glob.glob(f"{src_dir}/**/events.out.tfevents.*", recursive=True))
    if not event_files:
        print(f"No event files found in {src_dir}")
        return

    print(f"Found {len(event_files)} event file(s) in {src_dir}")
    
    # Target merged directory
    dst_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(dst_path), filename_suffix=".stage4_merged")

    merged_count = 0
    for ef in event_files:
        print(f"Reading events from {ef}...")
        ea = EventAccumulator(ef)
        ea.Reload()
        scalar_tags = ea.Tags().get("scalars", [])
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            for ev in events:
                new_step = ev.step + step_offset
                writer.add_scalar(tag, ev.value, global_step=new_step, walltime=ev.wall_time)
                merged_count += 1

    writer.flush()
    writer.close()
    print(f"Successfully merged {merged_count} scalar events with step_offset={step_offset} into {dst_dir}")

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "/workspace/qwen3-VL-2B-finetune/output/grpo_stage4_lora/runs"
    dst = sys.argv[2] if len(sys.argv) > 2 else "/workspace/qwen3-VL-2B-finetune/output/grpo_lora/runs/Sep23_13-22-05_31815fea88ba"
    offset = int(sys.argv[3]) if len(sys.argv) > 3 else 532
    merge_tb_runs(src, dst, offset)
