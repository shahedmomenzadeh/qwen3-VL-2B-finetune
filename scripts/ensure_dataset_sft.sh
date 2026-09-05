#!/usr/bin/env bash
# scripts/ensure_dataset_sft.sh — ensure dataset_sft exists; fetch + restore from HF Hub if missing.
#
# Called automatically at the top of every SFT entry script (train_sft.sh,
# train.sh, lite_sft_test.sh, lite_e2e_benchmark.sh, test_dtypes.sh).
# Fast no-op when <root>/Train + <root>/Validation already exist and are non-empty.
#
# Server flow (fresh clone): download the *.zip chunks from the HF dataset repo
# (~27 GB, resumable) into <root>/dataset_zips/, then restore the exact
# Train/Validation/Test layout with dataset_sft/unzip_dataset.py.
#
# Env:
#   SFT_DATASET_ROOT — dataset root (default: dataset_sft under the repo root)
#   SFT_HF_REPO      — HF dataset repo holding the zips (default: shahedm2001/dataset_sft)
#   SFT_ZIP_DIR      — zip download dir (default: <root>/dataset_zips)
#   SFT_KEEP_ZIPS    — 1 keep zips after extract (default), 0 delete them (~27 GB saved)
#   VENV_PYTHON      — python with huggingface_hub (default: .venv/bin/python)
#   HF_TOKEN         — honored automatically for private repos (also: hf auth login)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${SFT_DATASET_ROOT:-dataset_sft}"
case "$ROOT" in /*) ;; *) ROOT="$REPO_ROOT/$ROOT";; esac
HF_REPO="${SFT_HF_REPO:-shahedm2001/dataset_sft}"
ZIP_DIR="${SFT_ZIP_DIR:-$ROOT/dataset_zips}"
KEEP_ZIPS="${SFT_KEEP_ZIPS:-1}"

log()  { echo -e "\033[1;32m[ensure-sft]\033[0m $1"; }
warn() { echo -e "\033[1;33m[ensure-sft]\033[0m $1"; }
err()  { echo -e "\033[1;31m[ensure-sft] ERROR:\033[0m $1" >&2; exit 1; }

nonempty() { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

# ── 1. Fast path: already present ──────────────────────────────────────────
if nonempty "$ROOT/Train" && nonempty "$ROOT/Validation"; then
    log "dataset present: $ROOT/Train + $ROOT/Validation — nothing to do."
    exit 0
fi

log "dataset missing/empty at $ROOT — restoring from HF Hub ($HF_REPO)."

# ── 2. Python with huggingface_hub ─────────────────────────────────────────
PY="${VENV_PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
"$PY" -c "import huggingface_hub" 2>/dev/null \
    || err "huggingface_hub not found (tried $PY). Run setup.sh / train_sft.sh env setup first."

# ── 3. Download *.zip chunks (resumable; HF_TOKEN honored if set) ──────────
mkdir -p "$ZIP_DIR"
log "downloading zips -> $ZIP_DIR (resume-safe, ~27 GB total) ..."
"$PY" - "$HF_REPO" "$ZIP_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo_id, local_dir = sys.argv[1], sys.argv[2]
path = snapshot_download(
    repo_id=repo_id, repo_type="dataset",
    allow_patterns=["*.zip"], local_dir=local_dir,
)
print(f"snapshot ready: {path}")
PY

count=$(find "$ZIP_DIR" -maxdepth 1 -name "*.zip" | wc -l)
[ "$count" -gt 0 ] || err "no zips downloaded to $ZIP_DIR — check repo/patterns/HF_TOKEN."
log "downloaded $count zip chunk(s)."

# ── 4. Unzip into the dataset root (restores Train/Validation/Test) ────────
UNZIP_PY="$REPO_ROOT/dataset_sft/unzip_dataset.py"
[ -f "$UNZIP_PY" ] || err "$UNZIP_PY not found — fresh clone incomplete?"
log "extracting -> $ROOT ..."
"$PY" "$UNZIP_PY" --zip-dir "$ZIP_DIR" --out "$ROOT"

# ── 5. Verify + optional zip cleanup ────────────────────────────────────────
nonempty "$ROOT/Train" && nonempty "$ROOT/Validation" \
    || err "extraction incomplete — $ROOT/Train or $ROOT/Validation still missing."
log "dataset ready: $ROOT (Train + Validation present)."
if [ "$KEEP_ZIPS" = "0" ]; then
    log "SFT_KEEP_ZIPS=0 — removing zips to free disk ..."
    rm -f "$ZIP_DIR"/*.zip
fi
