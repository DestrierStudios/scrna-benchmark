#!/usr/bin/env python3
"""
Load raw datasets and perform quality control filtering.

Handles four data formats:
  - 10x standard (barcodes/genes/matrix per sample): GSE139829
  - Nested tar.gz (each sample is a tar.gz with 10x files inside): GSE176078
  - Pre-built UMI count matrix (genes x cells text file): GSE131907
  - Smart-seq2 text matrix with metadata rows: GSE72056

Usage:
    python scripts/04_run_qc.py                    # process all datasets
    python scripts/04_run_qc.py --dataset GSE139829 # process one dataset
"""

import argparse
import json
import gzip
import tarfile
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
from scipy.io import mmread

import yaml
import warnings
warnings.filterwarnings("ignore")


def load_config():
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)



def load_10x_sample(barcodes_file, genes_file, matrix_file, sample_name=None):
    """Load a single 10x sample from barcodes/genes/matrix files."""
    # Read matrix — handle gzipped .mtx.gz files
    if str(matrix_file).endswith(".gz"):
        import gzip as _gzip
        with _gzip.open(matrix_file, "rb") as f:
            mat = mmread(f).T.tocsr()
    else:
        mat = mmread(str(matrix_file)).T.tocsr()

    # Read barcodes
    opener = gzip.open if str(barcodes_file).endswith(".gz") else open
    with opener(barcodes_file, "rt") as f:
        barcodes = [line.strip().split("\t")[0] for line in f]

    # Read genes
    opener = gzip.open if str(genes_file).endswith(".gz") else open
    with opener(genes_file, "rt") as f:
        lines = [line.strip().split("\t") for line in f]
    gene_ids = [l[0] for l in lines]
    gene_names = [l[1] if len(l) > 1 else l[0] for l in lines]

    # Add sample prefix to barcodes to avoid collisions
    if sample_name:
        barcodes = [f"{sample_name}_{bc}" for bc in barcodes]

    adata = ad.AnnData(
        X=mat,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame({"gene_ids": gene_ids}, index=gene_names),
    )
    adata.var_names_make_unique()

    if sample_name:
        adata.obs["sample"] = sample_name

    return adata


def load_GSE139829(data_dir):
    """Load Uveal Melanoma — 10x standard format, 11 samples."""
    data_dir = Path(data_dir)
    print("  Loading GSE139829 (Uveal Melanoma) — 10x standard format")

    # Find all sample prefixes (e.g., GSM4147091_BSSR0022)
    matrix_files = sorted(data_dir.glob("*_matrix.mtx.gz"))
    print(f"  Found {len(matrix_files)} samples")

    adatas = []
    for mf in matrix_files:
        # Derive sample name from filename
        # e.g., GSM4147091_BSSR0022_matrix.mtx.gz -> BSSR0022
        parts = mf.stem.replace(".mtx", "").replace("_matrix", "").split("_")
        sample_name = "_".join(parts[1:])  # skip GSM ID
        prefix = mf.name.replace("_matrix.mtx.gz", "")

        bf = data_dir / f"{prefix}_barcodes.tsv.gz"
        gf = data_dir / f"{prefix}_genes.tsv.gz"

        if not bf.exists() or not gf.exists():
            print(f"  WARNING: Missing files for {sample_name}, skipping")
            continue

        print(f"    Loading {sample_name}...", end=" ", flush=True)
        a = load_10x_sample(bf, gf, mf, sample_name=sample_name)
        print(f"{a.n_obs} cells")
        adatas.append(a)

    if not adatas:
        raise RuntimeError("No samples loaded for GSE139829")

    # Concatenate all samples
    adata = ad.concat(adatas, join="outer")
    adata.obs_names_make_unique()
    # Fill NaN with 0 for genes missing in some samples
    if hasattr(adata.X, "toarray"):
        pass  # sparse is fine
    print(f"  Total: {adata.n_obs} cells, {adata.n_vars} genes")
    return adata


def load_GSE176078(data_dir):
    """Load Breast Cancer — nested tar.gz format, 26 samples."""
    data_dir = Path(data_dir)
    print("  Loading GSE176078 (Breast Cancer) — nested tar.gz format")

    tar_gz_files = sorted(data_dir.glob("GSM*.tar.gz"))
    print(f"  Found {len(tar_gz_files)} sample archives")

    adatas = []
    for tgz in tar_gz_files:
        # Extract sample name: GSM5354513_CID3586.tar.gz -> CID3586
        sample_name = tgz.stem.replace(".tar", "").split("_", 1)[-1]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract tar.gz
            with tarfile.open(tgz, "r:gz") as tar:
                tar.extractall(tmpdir)

            # Find the 10x files inside (may be nested in a subdirectory)
            tmppath = Path(tmpdir)
            matrix_files = list(tmppath.rglob("*.mtx*"))
            if not matrix_files:
                matrix_files = list(tmppath.rglob("*matrix*.mtx*"))

            if not matrix_files:
                print(f"  WARNING: No matrix file found in {tgz.name}, skipping")
                continue

            matrix_file = matrix_files[0]
            sample_dir = matrix_file.parent

            # Find barcodes and features/genes
            barcode_files = [f for f in sample_dir.rglob("*") if "barcode" in f.name.lower() and f.suffix in (".tsv", ".gz")]
            feature_files = [f for f in sample_dir.rglob("*") if ("gene" in f.name.lower() or "feature" in f.name.lower()) and f.suffix in (".tsv", ".gz")]

            if not barcode_files or not feature_files:
                print(f"  WARNING: Missing barcodes/features in {tgz.name}, skipping")
                continue

            print(f"    Loading {sample_name}...", end=" ", flush=True)
            a = load_10x_sample(
                barcode_files[0], feature_files[0], matrix_file,
                sample_name=sample_name,
            )
            print(f"{a.n_obs} cells")
            adatas.append(a)

    if not adatas:
        raise RuntimeError("No samples loaded for GSE176078")

    adata = ad.concat(adatas, join="outer")
    adata.obs_names_make_unique()
    print(f"  Total: {adata.n_obs} cells, {adata.n_vars} genes")
    return adata


def load_GSE131907(data_dir):
    """Load Lung Adenocarcinoma — pre-built UMI count matrix."""
    data_dir = Path(data_dir)
    print("  Loading GSE131907 (Lung Adenocarcinoma) — UMI count matrix")

    matrix_file = data_dir / "GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz"
    annot_file = data_dir / "GSE131907_Lung_Cancer_cell_annotation.txt.gz"

    # Read the count matrix (genes x cells, tab-separated)
    # First column is gene name ("Index" header)
    print("    Reading count matrix (this may take several minutes)...")
    df = pd.read_csv(matrix_file, sep="\t", index_col=0)

    # Transpose to cells x genes
    print(f"    Matrix shape: {df.shape[0]} genes x {df.shape[1]} cells")
    adata = ad.AnnData(
        X=csr_matrix(df.values.T.astype(np.float32)),
        obs=pd.DataFrame(index=df.columns),
        var=pd.DataFrame(index=df.index),
    )

    # Load cell annotations
    if annot_file.exists():
        print("    Loading cell annotations...")
        annot = pd.read_csv(annot_file, sep="\t", index_col=0)
        # Merge annotations with obs
        common_cells = adata.obs_names.intersection(annot.index)
        print(f"    Matched {len(common_cells)} / {adata.n_obs} cells to annotations")
        for col in annot.columns:
            adata.obs[col] = annot.reindex(adata.obs_names)[col]

    print(f"  Total: {adata.n_obs} cells, {adata.n_vars} genes")
    return adata


def load_GSE72056(data_dir):
    """Load Cutaneous Melanoma — Smart-seq2 text matrix with metadata rows."""
    data_dir = Path(data_dir)
    print("  Loading GSE72056 (Cutaneous Melanoma) — Smart-seq2 matrix")

    matrix_file = data_dir / "GSE72056_melanoma_single_cell_revised_v2.txt.gz"

    # Read the full file — first 3 rows are metadata, rest is gene expression
    print("    Reading matrix...")
    df = pd.read_csv(matrix_file, sep="\t", index_col=0)

    # Extract metadata rows
    meta_rows = ["tumor", "malignant(1=no,2=yes,0=unresolved)",
                 "non-malignant cell type (1=T,2=B,3=Macro.4=Endo.,5=CAF;6=NK)"]
    meta = {}
    for row_name in meta_rows:
        if row_name in df.index:
            meta[row_name] = df.loc[row_name]
        else:
            # Try partial match
            matches = [idx for idx in df.index if row_name[:20] in str(idx)]
            if matches:
                meta[row_name] = df.loc[matches[0]]

    # Remove metadata rows from expression data
    gene_mask = ~df.index.isin(list(meta.keys()))
    # Also remove rows that look like metadata (non-gene rows)
    first_few = df.index[:5].tolist()
    non_gene_rows = []
    for idx in df.index:
        try:
            # If all values are small integers (0-6), likely metadata
            vals = df.loc[idx].astype(float)
            if vals.max() <= 100 and idx in first_few:
                # Check if it's truly metadata vs a low-expression gene
                if str(idx).startswith(("tumor", "malig", "non-")):
                    non_gene_rows.append(idx)
        except (ValueError, TypeError):
            non_gene_rows.append(idx)

    expr_df = df.drop(index=non_gene_rows, errors="ignore")
    # Drop the known metadata rows
    for key in meta:
        if key in expr_df.index:
            expr_df = expr_df.drop(index=key)

    print(f"    Expression matrix: {expr_df.shape[0]} genes x {expr_df.shape[1]} cells")

    # Note: Smart-seq2 data is already log2(TPM+1), NOT raw counts
    # We store it as-is for cross-platform validation (not normalization benchmarking)
    adata = ad.AnnData(
        X=csr_matrix(expr_df.values.T.astype(np.float32)),
        obs=pd.DataFrame(index=expr_df.columns),
        var=pd.DataFrame(index=expr_df.index),
    )

    # Add metadata to obs
    if "tumor" in meta:
        adata.obs["tumor_id"] = meta["tumor"].reindex(adata.obs_names).values
    for key, vals in meta.items():
        col_name = key.split("(")[0].strip().replace(" ", "_")
        adata.obs[col_name] = vals.reindex(adata.obs_names).values

    # Mark as pre-normalized (not raw counts)
    adata.uns["is_raw_counts"] = False
    adata.uns["normalization"] = "log2(TPM+1)"

    print(f"  Total: {adata.n_obs} cells, {adata.n_vars} genes")
    return adata



def run_qc(adata, qc_params, dataset_name, is_raw_counts=True):
    """Apply QC filters and return filtered AnnData + QC report."""
    print(f"\n  Running QC on {dataset_name}")
    report = {"dataset": dataset_name, "pre_qc": {}, "post_qc": {}, "filters": {}}

    # Pre-QC stats
    report["pre_qc"]["n_cells"] = adata.n_obs
    report["pre_qc"]["n_genes"] = adata.n_vars
    print(f"    Before QC: {adata.n_obs} cells, {adata.n_vars} genes")

    if not is_raw_counts:
        # For pre-normalized data (GSE72056), minimal filtering only
        print("    Data is pre-normalized — applying minimal filters only")
        sc.pp.filter_cells(adata, min_genes=qc_params.get("min_genes_per_cell", 200))
        sc.pp.filter_genes(adata, min_cells=qc_params.get("min_cells_per_gene", 3))
        report["post_qc"]["n_cells"] = adata.n_obs
        report["post_qc"]["n_genes"] = adata.n_vars
        report["filters"]["note"] = "Pre-normalized data, minimal filtering"
        print(f"    After QC: {adata.n_obs} cells, {adata.n_vars} genes")
        return adata, report

    # Identify mitochondrial genes
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    n_mt_genes = adata.var["mt"].sum()
    print(f"    Mitochondrial genes found: {n_mt_genes}")

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)

    # Store pre-filter distributions
    report["pre_qc"]["median_genes_per_cell"] = float(np.median(adata.obs["n_genes_by_counts"]))
    report["pre_qc"]["median_counts_per_cell"] = float(np.median(adata.obs["total_counts"]))
    report["pre_qc"]["median_pct_mito"] = float(np.median(adata.obs["pct_counts_mt"]))

    # Apply filters
    min_genes = qc_params.get("min_genes_per_cell", 200)
    max_genes = qc_params.get("max_genes_per_cell", 8000)
    max_mito = qc_params.get("max_pct_mito", 20)
    min_cells = qc_params.get("min_cells_per_gene", 3)

    report["filters"]["min_genes_per_cell"] = min_genes
    report["filters"]["max_genes_per_cell"] = max_genes
    report["filters"]["max_pct_mito"] = max_mito
    report["filters"]["min_cells_per_gene"] = min_cells

    n_before = adata.n_obs

    # Cell filters
    cell_mask = (
        (adata.obs["n_genes_by_counts"] >= min_genes)
        & (adata.obs["n_genes_by_counts"] <= max_genes)
        & (adata.obs["pct_counts_mt"] <= max_mito)
    )
    n_removed_low_genes = (adata.obs["n_genes_by_counts"] < min_genes).sum()
    n_removed_high_genes = (adata.obs["n_genes_by_counts"] > max_genes).sum()
    n_removed_mito = (adata.obs["pct_counts_mt"] > max_mito).sum()

    adata = adata[cell_mask].copy()

    # Gene filter
    n_genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    n_genes_removed = n_genes_before - adata.n_vars

    report["filters"]["cells_removed_low_genes"] = int(n_removed_low_genes)
    report["filters"]["cells_removed_high_genes"] = int(n_removed_high_genes)
    report["filters"]["cells_removed_mito"] = int(n_removed_mito)
    report["filters"]["cells_removed_total"] = int(n_before - adata.n_obs)
    report["filters"]["genes_removed"] = int(n_genes_removed)

    # Post-QC stats
    report["post_qc"]["n_cells"] = adata.n_obs
    report["post_qc"]["n_genes"] = adata.n_vars
    report["post_qc"]["median_genes_per_cell"] = float(np.median(adata.obs["n_genes_by_counts"]))
    report["post_qc"]["median_counts_per_cell"] = float(np.median(adata.obs["total_counts"]))
    report["post_qc"]["median_pct_mito"] = float(np.median(adata.obs["pct_counts_mt"]))

    pct_retained = 100 * adata.n_obs / n_before
    print(f"    Removed {n_before - adata.n_obs} cells ({100-pct_retained:.1f}%)")
    print(f"      - Low genes (<{min_genes}): {n_removed_low_genes}")
    print(f"      - High genes (>{max_genes}): {n_removed_high_genes}")
    print(f"      - High mito (>{max_mito}%): {n_removed_mito}")
    print(f"    Removed {n_genes_removed} genes (expressed in <{min_cells} cells)")
    print(f"    After QC: {adata.n_obs} cells, {adata.n_vars} genes")

    return adata, report



LOADERS = {
    "GSE139829": load_GSE139829,
    "GSE176078": load_GSE176078,
    "GSE131907": load_GSE131907,
    "GSE72056": load_GSE72056,
}


def process_dataset(geo_id, config):
    """Load, QC, and save one dataset."""
    ds_config = config["datasets"][geo_id]
    qc_params = config["qc"]
    output_dir = Path(config["output"]["processed_data"]) / geo_id

    output_dir.mkdir(parents=True, exist_ok=True)
    output_h5ad = output_dir / "adata_qc.h5ad"
    output_report = output_dir / "qc_report.json"

    # Skip if already processed
    if output_h5ad.exists():
        print(f"\n  {geo_id}: Already processed ({output_h5ad}), skipping")
        print(f"  Delete {output_h5ad} to reprocess")
        return

    print(f"\n{'='*60}")
    print(f"  Processing {geo_id}: {ds_config['description']}")
    print(f"{'='*60}")

    # Load
    loader = LOADERS.get(geo_id)
    if loader is None:
        print(f"  ERROR: No loader for {geo_id}")
        return

    adata = loader(ds_config["path"])

    # Determine if raw counts
    is_raw = adata.uns.get("is_raw_counts", True)

    # QC
    adata, report = run_qc(adata, qc_params, geo_id, is_raw_counts=is_raw)

    # Save
    print(f"\n  Saving to {output_h5ad}")
    adata.write_h5ad(output_h5ad)
    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  QC report saved to {output_report}")

    # Create flag file for Snakemake
    (output_dir / "qc_done.flag").touch()


def main():
    parser = argparse.ArgumentParser(description="Run QC on scRNA-seq datasets")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Process a single dataset (e.g., GSE139829)")
    args = parser.parse_args()

    config = load_config()

    if args.dataset:
        if args.dataset not in config["datasets"]:
            print(f"ERROR: Unknown dataset {args.dataset}")
            print(f"Available: {list(config['datasets'].keys())}")
            sys.exit(1)
        process_dataset(args.dataset, config)
    else:
        for geo_id in config["datasets"]:
            process_dataset(geo_id, config)

    print(f"\n{'='*60}")
    print("  QC complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
