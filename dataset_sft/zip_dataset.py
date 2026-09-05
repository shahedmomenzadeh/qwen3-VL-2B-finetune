#!/usr/bin/env python3
"""
Multi-part independent ZIP creator for dataset splits.

Features:
- Splits archives split-wise (e.g., train_01.zip, train_02.zip, test_01.zip, validation_01.zip, etc.).
- Produces completely independent, standalone, standard ZIP files.
- Restores original subfolder structures and preserves relative directory hierarchies.
- Standalone top-level files (e.g., README.md, grpo_procedure_split_map.json) are packed into root_files.zip.
- Configurable chunk size (default: 2.0 GB).
- Shows live file and bytes progress.
"""

import os
import sys
import zipfile
import argparse

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def chunk_files_for_split(split_name: str, src_dir: str, target_size_bytes: int):
    """
    Groups files for a split or file group into independent chunks.
    Only creates a new chunk when there are files in it.
    """
    split_path = os.path.join(src_dir, split_name)
    items = []
    
    if os.path.isdir(split_path):
        for root, dirs, files in os.walk(split_path):
            rel_root = os.path.relpath(root, src_dir)
            # Add files with sizes
            for f in sorted(files):
                rel_f = os.path.normpath(os.path.join(rel_root, f))
                abs_f = os.path.join(root, f)
                items.append((rel_f, os.path.getsize(abs_f)))
    else:
        items.append((split_name, os.path.getsize(split_path)))

    items.sort(key=lambda x: x[0])

    chunks = []
    current_chunk = []
    current_size = 0

    for rel_path, size in items:
        # If adding this file exceeds chunk limit and current chunk is not empty, start a new chunk
        if current_chunk and (current_size + size > target_size_bytes):
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0

        current_chunk.append((rel_path, size))
        current_size += size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def create_zip_chunk(zip_filepath: str, chunk_items: list, src_dir: str, compresslevel: int = 6):
    """
    Creates a single independent ZIP file containing the specified items with relative paths.
    """
    with zipfile.ZipFile(
        zip_filepath,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compresslevel,
        allowZip64=True
    ) as zf:
        iterator = tqdm(chunk_items, desc=f"  Packing {os.path.basename(zip_filepath)}", unit="file", leave=False) if tqdm else chunk_items
        for rel_path, _ in iterator:
            abs_path = os.path.join(src_dir, rel_path)
            zf.write(abs_path, arcname=rel_path)


def split_and_zip(src_dir: str, output_dir: str, chunk_size_gb: float = 2.0, compresslevel: int = 6):
    src_dir = os.path.abspath(src_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    target_size_bytes = int(chunk_size_gb * (1024 ** 3))

    # Identify directories (splits) and root files
    entries = sorted(os.listdir(src_dir))
    splits = [e for e in entries if os.path.isdir(os.path.join(src_dir, e)) and not e.startswith('.') and os.path.abspath(os.path.join(src_dir, e)) != output_dir]
    root_files = [e for e in entries if os.path.isfile(os.path.join(src_dir, e)) and not e.startswith('.') and not e.endswith('.py')]

    created_zips = []

    print("=" * 65)
    print("DATASET INDEPENDENT MULTI-CHUNK ZIP ARCHIVER")
    print("=" * 65)
    print(f"Source Directory   : {src_dir}")
    print(f"Output Directory   : {output_dir}")
    print(f"Target Chunk Size  : {chunk_size_gb} GB ({target_size_bytes / (1024**2):.1f} MB)")
    print(f"Detected Splits    : {', '.join(splits)}")
    print(f"Detected Root Files: {', '.join(root_files) if root_files else 'None'}")
    print("=" * 65)

    # Process each split directory
    for split in splits:
        prefix = split.lower()
        chunks = chunk_files_for_split(split, src_dir, target_size_bytes)
        num_chunks = len(chunks)
        print(f"\n[Split: {split}] -> Generating {num_chunks} independent zip chunk{'s' if num_chunks > 1 else ''}...")

        for idx, chunk_items in enumerate(chunks, start=1):
            zip_filename = f"{prefix}_{idx:02d}.zip"
            zip_filepath = os.path.join(output_dir, zip_filename)
            
            total_raw_bytes = sum(sz for _, sz in chunk_items)
            num_files = len(chunk_items)
            
            print(f" -> {zip_filename}: {num_files} files, ~{total_raw_bytes / (1024**3):.2f} GB raw data...")
            create_zip_chunk(zip_filepath, chunk_items, src_dir, compresslevel=compresslevel)
            
            actual_size = os.path.getsize(zip_filepath)
            created_zips.append((zip_filename, actual_size, num_files))
            print(f"    Done: {zip_filename} (Compressed Size: {actual_size / (1024**2):.1f} MB)")

    # Process root files (README.md, metadata json files, etc.)
    if root_files:
        print(f"\n[Root Files] -> {len(root_files)} standalone file(s)...")
        root_chunks = []
        current_chunk = []
        current_size = 0
        for f in root_files:
            sz = os.path.getsize(os.path.join(src_dir, f))
            if current_chunk and (current_size + sz > target_size_bytes):
                root_chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            current_chunk.append((f, sz))
            current_size += sz
        if current_chunk:
            root_chunks.append(current_chunk)

        for idx, chunk_items in enumerate(root_chunks, start=1):
            zip_filename = f"root_files_{idx:02d}.zip" if len(root_chunks) > 1 else "root_files.zip"
            zip_filepath = os.path.join(output_dir, zip_filename)
            create_zip_chunk(zip_filepath, chunk_items, src_dir, compresslevel=compresslevel)
            actual_size = os.path.getsize(zip_filepath)
            created_zips.append((zip_filename, actual_size, len(chunk_items)))
            print(f" -> Created {zip_filename} ({actual_size / (1024**2):.2f} MB)")

    print("\n" + "=" * 65)
    print("ALL ZIP ARCHIVES CREATED SUCCESSFULLY:")
    print("=" * 65)
    total_archived_size = sum(sz for _, sz, _ in created_zips)
    for name, sz, num_f in created_zips:
        print(f"  - {name:<22} : {num_f:>5} files | {sz / (1024**2):>8.2f} MB ({sz / (1024**3):.2f} GB)")
    print("-" * 65)
    print(f"Total Output Size: {total_archived_size / (1024**3):.2f} GB across {len(created_zips)} independent archives.")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split-wise independent chunked ZIP archiver.")
    parser.add_argument("--src", type=str, default=".", help="Source dataset directory (default: current directory)")
    parser.add_argument("--out", type=str, default="./dataset_zips", help="Output directory for zip files (default: ./dataset_zips)")
    parser.add_argument("--chunk-size-gb", type=float, default=2.0, help="Target uncompressed chunk size in GB (default: 2.0)")
    parser.add_argument("--compresslevel", type=int, default=6, help="Deflate compression level 1-9 (default: 6)")
    args = parser.parse_args()

    split_and_zip(args.src, args.out, chunk_size_gb=args.chunk_size_gb, compresslevel=args.compresslevel)
