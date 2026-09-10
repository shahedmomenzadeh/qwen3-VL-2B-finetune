#!/usr/bin/env python3
"""scripts/build_epoch_subset.py — stratified per-epoch subset for stage-2 SFT.

Groups samples by id prefix (the prepare_sft.py scheme):
  full_* -> full-video samples
  PH_*   -> phase-source clip samples
  else   -> video-source (YouTube) clip samples

The clip budget (CLIP_FRACTION of all clips) is split EQUALLY between the two
clip sources; the full-video budget (FULL_FRACTION of all full videos) is
sampled independently. Sampling is without replacement within one call;
call once per epoch with seed = base + epoch for fresh draws.

Usage:
    python scripts/build_epoch_subset.py <full_json> <out_json> \
        --clip-fraction 0.1 --full-fraction 1.0 --seed 43
"""

import argparse
import json
import random
import sys


def split_groups(samples):
    full, ph, yt = [], [], []
    for s in samples:
        sid = s.get("id", "")
        if sid.startswith("full_"):
            full.append(s)
        elif sid.startswith("PH_"):
            ph.append(s)
        else:
            yt.append(s)
    return full, ph, yt


def main():
    ap = argparse.ArgumentParser(description="Stratified stage-2 epoch subset")
    ap.add_argument("full_json", help="Full prepared train JSON")
    ap.add_argument("out_json", help="Output epoch JSON")
    ap.add_argument("--clip-fraction", type=float, default=1.0)
    ap.add_argument("--full-fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for name, v in (("clip-fraction", args.clip_fraction), ("full-fraction", args.full_fraction)):
        if not 0.0 < v <= 1.0:
            print(f"error: --{name} must be in (0, 1], got {v}", file=sys.stderr)
            sys.exit(2)

    with open(args.full_json) as f:
        samples = json.load(f)
    full, ph, yt = split_groups(samples)
    rng = random.Random(args.seed)

    # Clip budget split equally between the two sources (remainder -> YT pool).
    n_clip = round(args.clip_fraction * (len(ph) + len(yt)))
    n_yt = min(len(yt), (n_clip + 1) // 2)
    n_ph = min(len(ph), n_clip - n_yt)
    # If one source is exhausted, top up from the other.
    n_yt = min(len(yt), n_clip - n_ph)

    # Full-video budget.
    n_full = round(args.full_fraction * len(full))

    subset = rng.sample(yt, n_yt) + rng.sample(ph, n_ph) + rng.sample(full, n_full)
    rng.shuffle(subset)

    with open(args.out_json, "w") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"seed={args.seed}: clips {len(ph) + len(yt)} -> {n_yt} YT + {n_ph} PH | "
          f"full {len(full)} -> {n_full} | epoch total {len(subset)} -> {args.out_json}")


if __name__ == "__main__":
    main()
