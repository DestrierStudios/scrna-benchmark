#!/usr/bin/env python3
"""
Explore raw dataset structure to understand formats before QC.

Usage:
    python scripts/03_explore_raw_data.py
"""

import os
import gzip
import tarfile
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")


def explore_directory(geo_id):
    """List and describe files in a dataset directory."""
    dataset_dir = RAW_DIR / geo_id
    if not dataset_dir.exists():
        print(f"  Directory not found: {dataset_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  {geo_id}")
    print(f"{'='*60}")

    for f in sorted(dataset_dir.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            rel_path = f.relative_to(dataset_dir)
            print(f"  {rel_path} ({size_mb:.1f} MB)")

    # Count specific file types
    gz_files = list(dataset_dir.glob("*.gz"))
    mtx_files = list(dataset_dir.rglob("*.mtx*"))
    tsv_files = list(dataset_dir.rglob("*.tsv*"))
    h5_files = list(dataset_dir.rglob("*.h5*"))
    tar_files = list(dataset_dir.glob("*.tar"))
    txt_files = list(dataset_dir.rglob("*.txt*"))
    rds_files = list(dataset_dir.rglob("*.rds*"))

    print(f"\n  File types: .gz={len(gz_files)}, .mtx={len(mtx_files)}, "
          f".tsv={len(tsv_files)}, .h5={len(h5_files)}, .tar={len(tar_files)}, "
          f".txt={len(txt_files)}, .rds={len(rds_files)}")


def peek_10x_files(geo_id):
    """Check if dataset has 10x-format files (barcodes, features/genes, matrix)."""
    dataset_dir = RAW_DIR / geo_id
    
    # Look for typical 10x patterns in extracted files
    barcodes = list(dataset_dir.rglob("*barcodes*"))
    features = list(dataset_dir.rglob("*features*")) + list(dataset_dir.rglob("*genes*"))
    matrix = list(dataset_dir.rglob("*matrix*"))

    if barcodes or features or matrix:
        print(f"\n  10x-style files detected:")
        for f in barcodes[:3]:
            print(f"    Barcodes: {f.relative_to(dataset_dir)}")
        for f in features[:3]:
            print(f"    Features: {f.relative_to(dataset_dir)}")
        for f in matrix[:3]:
            print(f"    Matrix:   {f.relative_to(dataset_dir)}")
        return True
    return False


def peek_text_matrix(filepath, n_rows=5, n_cols=8):
    """Preview the first few rows/cols of a text matrix file."""
    print(f"\n  Previewing: {filepath.name}")
    
    opener = gzip.open if str(filepath).endswith(".gz") else open
    mode = "rt" if str(filepath).endswith(".gz") else "r"
    
    try:
        with opener(filepath, mode) as f:
            for i, line in enumerate(f):
                if i >= n_rows:
                    break
                fields = line.strip().split("\t")
                if len(fields) > n_cols:
                    preview = "\t".join(fields[:n_cols]) + f"\t... ({len(fields)} cols total)"
                else:
                    preview = "\t".join(fields)
                print(f"    Row {i}: {preview}")
    except Exception as e:
        print(f"    Error reading: {e}")


def count_lines(filepath):
    """Count lines in a (potentially gzipped) file."""
    opener = gzip.open if str(filepath).endswith(".gz") else open
    mode = "rt" if str(filepath).endswith(".gz") else "r"
    count = 0
    try:
        with opener(filepath, mode) as f:
            for _ in f:
                count += 1
    except Exception as e:
        print(f"    Error counting lines: {e}")
    return count


def explore_GSE131928():
    """Glioblastoma — 10x format."""
    geo_id = "GSE131928"
    explore_directory(geo_id)
    has_10x = peek_10x_files(geo_id)
    
    if not has_10x:
        # Check what's inside the tar
        dataset_dir = RAW_DIR / geo_id
        gz_files = sorted(dataset_dir.glob("*.gz"))
        if gz_files:
            print(f"\n  Compressed files found ({len(gz_files)}):")
            for f in gz_files[:5]:
                print(f"    {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
            # Try to peek at first gz file
            peek_text_matrix(gz_files[0], n_rows=3)


def explore_GSE139829():
    """Uveal Melanoma — 10x format."""
    geo_id = "GSE139829"
    explore_directory(geo_id)
    has_10x = peek_10x_files(geo_id)

    if not has_10x:
        dataset_dir = RAW_DIR / geo_id
        gz_files = sorted(dataset_dir.glob("*.gz"))
        if gz_files:
            print(f"\n  Compressed files found ({len(gz_files)}):")
            for f in gz_files[:5]:
                print(f"    {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")


def explore_GSE176078():
    """Breast Cancer — 10x format."""
    geo_id = "GSE176078"
    explore_directory(geo_id)
    has_10x = peek_10x_files(geo_id)

    if not has_10x:
        dataset_dir = RAW_DIR / geo_id
        gz_files = sorted(dataset_dir.glob("*.gz"))
        if gz_files:
            print(f"\n  Compressed files found ({len(gz_files)}):")
            for f in gz_files[:5]:
                print(f"    {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")


def explore_GSE131907():
    """Lung Adenocarcinoma — pre-built UMI matrix."""
    geo_id = "GSE131907"
    explore_directory(geo_id)
    
    dataset_dir = RAW_DIR / geo_id
    matrix_file = dataset_dir / "GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz"
    annot_file = dataset_dir / "GSE131907_Lung_Cancer_cell_annotation.txt.gz"

    if matrix_file.exists():
        peek_text_matrix(matrix_file, n_rows=5)
        print(f"\n  Counting rows (this may take a moment)...")
        n_lines = count_lines(matrix_file)
        print(f"  Matrix: {n_lines} lines (rows ≈ genes if genes x cells)")

    if annot_file.exists():
        print(f"\n  Cell annotation file:")
        peek_text_matrix(annot_file, n_rows=5)
        n_lines = count_lines(annot_file)
        print(f"  Annotations: {n_lines} lines")


def explore_GSE72056():
    """Cutaneous Melanoma — Smart-seq2 text matrix."""
    geo_id = "GSE72056"
    explore_directory(geo_id)
    
    dataset_dir = RAW_DIR / geo_id
    matrix_file = dataset_dir / "GSE72056_melanoma_single_cell_revised_v2.txt.gz"

    if matrix_file.exists():
        peek_text_matrix(matrix_file, n_rows=8, n_cols=6)
        print(f"\n  Counting rows...")
        n_lines = count_lines(matrix_file)
        print(f"  Matrix: {n_lines} lines")


def main():
    print("scRNA-seq Benchmark — Raw Data Exploration")
    print(f"Looking in: {RAW_DIR.resolve()}")
    
    explore_GSE131928()
    explore_GSE139829()
    explore_GSE176078()
    explore_GSE131907()
    explore_GSE72056()

    print(f"\n{'='*60}")
    print("  Exploration complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
