#!/usr/bin/env python3
"""
Download all scRNA-seq datasets from GEO.

Datasets:
    GSE139829 - Uveal Melanoma (Durante et al. 2020) ~60,000 cells, 10x
    GSE176078 - Breast Cancer (Wu et al. 2021) ~130,000 cells, 10x
    GSE131907 - Lung Adenocarcinoma (Kim et al. 2020) ~208,000 cells, 10x
    GSE72056  - Cutaneous Melanoma (Tirosh et al. 2016) ~4,600 cells, Smart-seq2

Note: GSE131928 (Glioblastoma, Neftel et al. 2019) was excluded because only
processed TPM values are available on GEO — raw counts are required for
normalization benchmarking.
"""

import os
import subprocess
import sys
import hashlib
import gzip
import shutil
from pathlib import Path


RAW_DIR = Path("data/raw")

DATASETS = {
    "GSE139829": {
        "description": "Uveal Melanoma (Durante et al. 2020)",
        "platform": "10x",
        "urls": [
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE139nnn/GSE139829/suppl/GSE139829_RAW.tar",
        ],
    },
    "GSE176078": {
        "description": "Breast Cancer (Wu et al. 2021)",
        "platform": "10x",
        "urls": [
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE176nnn/GSE176078/suppl/GSE176078_RAW.tar",
        ],
    },
    "GSE131907": {
        "description": "Lung Adenocarcinoma (Kim et al. 2020)",
        "platform": "10x",
        "urls": [
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131907/suppl/GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz",
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131907/suppl/GSE131907_Lung_Cancer_cell_annotation.txt.gz",
        ],
    },
    "GSE72056": {
        "description": "Cutaneous Melanoma (Tirosh et al. 2016)",
        "platform": "Smart-seq2",
        "urls": [
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE72nnn/GSE72056/suppl/GSE72056_melanoma_single_cell_revised_v2.txt.gz",
        ],
    },
}


def download_file(url, dest_dir):
    """Download a file using wget with resume support."""
    filename = url.split("/")[-1]
    dest_path = dest_dir / filename

    if dest_path.exists():
        print(f"  Already exists: {filename}, skipping")
        return dest_path

    print(f"  Downloading: {filename}")
    cmd = [
        "wget",
        "--continue",           # Resume partial downloads
        "--no-verbose",
        "--show-progress",
        "-O", str(dest_path),
        url,
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR: Failed to download {filename}")
        if dest_path.exists():
            dest_path.unlink()
        return None

    return dest_path


def extract_tar(tar_path, dest_dir):
    """Extract a tar file."""
    print(f"  Extracting: {tar_path.name}")
    cmd = ["tar", "xf", str(tar_path), "-C", str(dest_dir)]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR: Failed to extract {tar_path.name}")
        return False
    return True


def get_dir_size(path):
    """Get total size of a directory in MB."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def download_dataset(geo_id, info):
    """Download and organize one dataset."""
    dest_dir = RAW_DIR / geo_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {geo_id} — {info['description']}")
    print(f"Platform: {info['platform']}")
    print(f"{'='*60}")

    for url in info["urls"]:
        downloaded = download_file(url, dest_dir)
        if downloaded is None:
            print(f"  FAILED: Could not download from {url}")
            continue

        # Extract tar files
        if downloaded.suffix == ".tar":
            if extract_tar(downloaded, dest_dir):
                # Optionally remove the tar after extraction to save space
                # downloaded.unlink()
                pass

    # Report what we got
    files = list(dest_dir.rglob("*"))
    file_count = len([f for f in files if f.is_file()])
    dir_size = get_dir_size(dest_dir)
    print(f"  Result: {file_count} files, {dir_size:.1f} MB")


def verify_downloads():
    """Check that all datasets have files."""
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    all_ok = True
    for geo_id, info in DATASETS.items():
        dest_dir = RAW_DIR / geo_id
        if not dest_dir.exists():
            print(f"  MISSING: {geo_id}")
            all_ok = False
            continue

        files = [f for f in dest_dir.rglob("*") if f.is_file()]
        size = get_dir_size(dest_dir)

        if len(files) == 0:
            print(f"  EMPTY:   {geo_id}")
            all_ok = False
        else:
            print(f"  OK:      {geo_id} — {len(files)} files, {size:.1f} MB")

    return all_ok


def main():
    print("scRNA-seq Benchmark — Dataset Download")
    print(f"Download directory: {RAW_DIR.resolve()}")

    # Check wget is available
    if shutil.which("wget") is None:
        print("ERROR: wget not found. Install with: apt-get install wget")
        sys.exit(1)

    # Download each dataset
    for geo_id, info in DATASETS.items():
        download_dataset(geo_id, info)

    # Verify
    if verify_downloads():
        print("\nAll datasets downloaded successfully!")
    else:
        print("\nSome datasets are missing. Re-run to retry failed downloads.")
        sys.exit(1)


if __name__ == "__main__":
    main()
