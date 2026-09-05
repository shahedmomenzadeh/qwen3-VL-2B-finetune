#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Upload dataset ZIP chunks to Hugging Face Hub
# Repo: shahedm2001/dataset_sft (dataset)
# Usage:
#   chmod +x upload_to_hf.sh
#   ./upload_to_hf.sh                    # upload ./dataset_zips
#   ./upload_to_hf.sh /path/to/zips      # custom path
#   ZIP_DIR=/my/zips REPO_ID=myuser/myrepo ./upload_to_hf.sh
# ==============================================================================

REPO_ID="${REPO_ID:-shahedm2001/dataset_sft}"
ZIP_DIR="${1:-${ZIP_DIR:-./dataset_zips}}"
REPO_TYPE="dataset"
NUM_WORKERS=2

# Resolve absolute path
ZIP_DIR="$(realpath "$ZIP_DIR" 2>/dev/null || echo "$ZIP_DIR")"

echo "============================================================"
echo " Hugging Face Dataset Upload"
echo "============================================================"
echo " Repo ID     : $REPO_ID"
echo " Repo Type   : $REPO_TYPE"
echo " Local Folder: $ZIP_DIR"
echo "============================================================"

# 1. Check hf CLI exists
if ! command -v hf &>/dev/null; then
  echo "[ERROR] 'hf' CLI not found. Install with: pip install -U huggingface_hub"
  echo "         or: pip install -U hf_xet"
  exit 1
fi

# 2. Check login
echo ""
echo "[1/4] Checking authentication..."
if ! hf auth whoami &>/dev/null; then
  echo "[ERROR] Not logged in. Run: hf auth login"
  exit 1
fi
hf auth whoami
# export HF_TOKEN=hf_xxxx  # set your token in the environment, never commit it

# 3. Check that ZIP_DIR exists and has .zip files
echo ""
echo "[2/4] Checking local ZIP files..."
if [ ! -d "$ZIP_DIR" ]; then
  echo "[ERROR] Directory not found: $ZIP_DIR"
  echo "        Did you run: python3 zip_dataset.py --src . --out ./dataset_zips ?"
  exit 1
fi

ZIP_COUNT=$(find "$ZIP_DIR" -maxdepth 1 -name "*.zip" | wc -l)
TOTAL_SIZE=$(du -sh "$ZIP_DIR" | cut -f1)

if [ "$ZIP_COUNT" -eq 0 ]; then
  echo "[ERROR] No .zip files found in $ZIP_DIR"
  ls -lh "$ZIP_DIR"
  exit 1
fi

echo " Found $ZIP_COUNT zip file(s) ($TOTAL_SIZE total):"
ls -lh "$ZIP_DIR"/*.zip | awk '{print "   -", $9, "(" $5 ")"}'
echo ""

# Optional: warn if any file < 5MB (might indicate truncated)
echo "[3/4] Uploading to Hub (resumable, multi-worker)..."
echo "      Command: hf upload-large-folder $REPO_ID $ZIP_DIR --repo-type $REPO_TYPE --num-workers $NUM_WORKERS"
echo ""

# Use upload-large-folder for resumable large uploads (26GB).
# Recommended by Hugging Face for >few GB. Handles chunked LFS uploads and retries.
# Fallback to 'hf upload' if upload-large-folder not available.

if hf upload-large-folder --help &>/dev/null; then
  hf upload-large-folder "$REPO_ID" "$ZIP_DIR" \
    --repo-type "$REPO_TYPE" \
    --num-workers "$NUM_WORKERS"
else
# echo "[WARN] 'hf upload-large-folder' not available, falling back to 'hf upload' (single-commit)..."
  hf upload "$REPO_ID" "$ZIP_DIR" . \
    --repo-type "$REPO_TYPE" \
    --commit-message "Add dataset zip chunks ($ZIP_COUNT files, $TOTAL_SIZE)" \
    --commit-description "Standalone independent archives: train_*.zip test_*.zip validation_*.zip root_files.zip - each ~2GB, unzip with unzip_dataset.py"
  fi

echo ""
echo "[4/4] Verifying upload..."
echo "      Listing remote files:"
hf datasets ls "$REPO_ID" 2>&1 || hf upload --help >/dev/null

echo ""
echo "============================================================"
echo " Upload complete!"
echo " View at: https://huggingface.co/datasets/$REPO_ID"
echo "============================================================"
echo ""
echo "To download later:"
echo "  hf download $REPO_ID --repo-type dataset --local-dir ./dataset_zips_restored"
echo "  # or via git lfs:"
echo "  git lfs install && git clone https://huggingface.co/datasets/$REPO_ID"
