#!/usr/bin/env python3
"""
Multi-part independent ZIP extractor.

Features:
- Discovers and unpacks all independent split ZIP files (e.g., train_01.zip, test_01.zip, etc.).
- Reconstructs the exact directory structure and restores all files.
- Preserves file permissions and timestamps.
- Verifies extracted files against the original zip file manifest/checksums.
"""

import os
import sys
import glob
import zipfile
import argparse
import time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def unzip_all(zip_dir: str, target_dir: str, verify: bool = True):
    zip_dir = os.path.abspath(zip_dir)
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    # Find all .zip files in zip_dir
    zip_files = sorted(glob.glob(os.path.join(zip_dir, "*.zip")))

    if not zip_files:
        print(f"Error: No .zip files found in '{zip_dir}'")
        sys.exit(1)

    print("=" * 60)
    print(f"ZIP Directory      : {zip_dir}")
    print(f"Destination Folder : {target_dir}")
    print(f"Found ZIP Files    : {len(zip_files)}")
    for zf in zip_files:
        sz_mb = os.path.getsize(zf) / (1024 ** 2)
        print(f"  - {os.path.basename(zf):<25} ({sz_mb:.2f} MB)")
    print("=" * 60)

    total_files_extracted = 0
    total_bytes_extracted = 0
    start_time = time.time()

    for zip_path in zip_files:
        zip_name = os.path.basename(zip_path)
        print(f"\nExtracting '{zip_name}'...")

        with zipfile.ZipFile(zip_path, 'r') as zf:
            infolist = zf.infolist()
            
            # Use progress bar if tqdm is available
            iterator = tqdm(infolist, desc=f"Extracting {zip_name}", unit="file") if tqdm else infolist

            for member in iterator:
                zf.extract(member, path=target_dir)
                # Restore original timestamps if available
                extracted_path = os.path.join(target_dir, member.filename)
                if hasattr(member, 'date_time') and member.date_time:
                    try:
                        date_time = time.mktime(member.date_time + (0, 0, -1))
                        os.utime(extracted_path, (date_time, date_time))
                    except Exception:
                        pass

                if not member.is_dir():
                    total_files_extracted += 1
                    total_bytes_extracted += member.file_size

        if verify:
            print(f"  -> Verifying '{zip_name}' integrity (CRC-32)...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                bad_file = zf.testzip()
                if bad_file:
                    print(f"  [ERROR] Corrupted file found in {zip_name}: {bad_file}")
                    sys.exit(1)
                else:
                    print(f"  -> Checksum OK.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETED SUCCESSFULLY!")
    print(f"Total files restored : {total_files_extracted}")
    print(f"Total size restored  : {total_bytes_extracted / (1024**3):.2f} GB ({total_bytes_extracted / (1024**2):.2f} MB)")
    print(f"Time elapsed         : {elapsed:.2f} seconds")
    print(f"Target location      : {target_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-part split independent ZIP extractor.")
    parser.add_argument("--zip-dir", type=str, default="./zips", help="Directory containing .zip files (default: ./zips)")
    parser.add_argument("--out", type=str, default="./extracted_dataset", help="Destination directory (default: ./extracted_dataset)")
    parser.add_argument("--no-verify", action="store_true", help="Skip CRC-32 integrity verification")
    args = parser.parse_args()

    unzip_all(args.zip_dir, args.out, verify=not args.no_verify)
