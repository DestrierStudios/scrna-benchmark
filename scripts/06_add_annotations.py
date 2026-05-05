#!/usr/bin/env python3
"""
Add cell type annotations to processed datasets.

- GSE176078: Load annotations from metadata.csv inside each sample tar.gz
- GSE139829: Automated marker-based annotation using canonical markers
- GSE131907: Already has annotations from GEO
- GSE72056: Already has malignant/cell type codes

Usage:
    python scripts/06_add_annotations.py
"""

import tarfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse

import warnings
warnings.filterwarnings("ignore")


def annotate_GSE176078():
    """Load cell type labels from metadata.csv inside each sample tar.gz."""
    print("\n" + "=" * 60)
    print("  GSE176078 (Breast Cancer) — Loading metadata from tar.gz files")
    print("=" * 60)

    h5ad_path = Path("data/processed/GSE176078/adata_qc.h5ad")
    raw_dir = Path("data/raw/GSE176078")

    adata = sc.read_h5ad(h5ad_path)
    print(f"  Loaded: {adata.n_obs} cells")

    # Collect metadata from all tar.gz files
    all_meta = []
    tar_files = sorted(raw_dir.glob("GSM*.tar.gz"))

    for tgz in tar_files:
        sample_name = tgz.stem.replace(".tar", "").split("_", 1)[-1]
        with tarfile.open(tgz, "r:gz") as tar:
            for member in tar.getmembers():
                if "metadata" in member.name.lower() and member.name.endswith(".csv"):
                    f = tar.extractfile(member)
                    meta = pd.read_csv(f)

                    # The first column is the barcode
                    id_col = meta.columns[0]
                    if id_col == "Unnamed: 0":
                        meta = meta.rename(columns={"Unnamed: 0": "barcode"})
                    else:
                        meta = meta.rename(columns={id_col: "barcode"})

                    # Standardize barcode format to match h5ad
                    # In h5ad: {sample}_{barcode}, in metadata: {sample}_{barcode}
                    # Check if barcodes already have sample prefix
                    if not meta["barcode"].iloc[0].startswith(sample_name):
                        meta["barcode"] = sample_name + "_" + meta["barcode"]

                    all_meta.append(meta)
                    print(f"    {sample_name}: {len(meta)} cells, "
                          f"cols={[c for c in meta.columns if 'type' in c.lower() or 'cell' in c.lower() or 'subtype' in c.lower()]}")
                    break

    if not all_meta:
        print("  ERROR: No metadata files found")
        return

    # Combine all metadata
    meta_df = pd.concat(all_meta, ignore_index=True)
    meta_df = meta_df.set_index("barcode")
    print(f"\n  Total metadata: {len(meta_df)} cells")

    # Match to adata
    common = adata.obs_names.intersection(meta_df.index)
    print(f"  Matched: {len(common)} / {adata.n_obs} cells ({100*len(common)/adata.n_obs:.1f}%)")

    # Add annotation columns
    annotation_cols = ["celltype_major", "celltype_minor", "celltype_subset", "subtype"]
    for col in annotation_cols:
        if col in meta_df.columns:
            adata.obs[col] = meta_df.reindex(adata.obs_names)[col]
            n_annotated = adata.obs[col].notna().sum()
            n_unique = adata.obs[col].nunique()
            print(f"    {col}: {n_unique} types, {n_annotated} cells annotated")

    # Save
    adata.write_h5ad(h5ad_path)
    print(f"\n  Saved to {h5ad_path}")


# Canonical markers from Durante et al. 2020 and follow-up studies
UVEAL_MARKERS = {
    "Tumor cells": ["MLANA", "MITF", "PMEL", "TYR", "DCT"],
    "T cells": ["CD3D", "CD3E", "CD2", "CD8A", "CD4"],
    "B cells": ["CD19", "MS4A1", "CD79A", "CD79B"],
    "Plasma cells": ["MZB1", "IGHG1", "CD79A", "JCHAIN", "SDC1"],
    "Macrophages": ["CD68", "CD14", "CSF1R", "FCGR3A", "AIF1"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "FLT1"],
    "Fibroblasts": ["COL1A1", "COL1A2", "DCN", "LUM", "FAP"],
    "Photoreceptors": ["RCVRN", "RHO", "OPN1SW", "PDE6A"],
    "RPE cells": ["RPE65", "BEST1", "RLBP1"],
}


def score_cell_types(adata, markers_dict):
    """Score cells for each cell type using marker gene expression."""
    # Normalize for scoring if not already done
    adata_score = adata.copy()
    if issparse(adata_score.X):
        max_val = adata_score.X.max()
    else:
        max_val = np.max(adata_score.X)

    # If raw counts, normalize temporarily
    if max_val > 50:
        sc.pp.normalize_total(adata_score, target_sum=1e4)
        sc.pp.log1p(adata_score)

    scores = {}
    for cell_type, markers in markers_dict.items():
        # Filter to markers that exist in the dataset
        available = [g for g in markers if g in adata_score.var_names]
        if not available:
            print(f"    WARNING: No markers found for {cell_type}")
            continue

        sc.tl.score_genes(adata_score, available, score_name=f"score_{cell_type}")
        scores[cell_type] = adata_score.obs[f"score_{cell_type}"].values
        print(f"    {cell_type}: {len(available)}/{len(markers)} markers "
              f"({', '.join(available[:3])}{'...' if len(available) > 3 else ''})")

    return scores


def annotate_GSE139829():
    """Annotate GSE139829 using canonical markers."""
    print("\n" + "=" * 60)
    print("  GSE139829 (Uveal Melanoma) — Marker-based annotation")
    print("=" * 60)

    h5ad_path = Path("data/processed/GSE139829/adata_qc.h5ad")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

    # Score cells
    print("\n  Scoring cells with canonical markers:")
    scores = score_cell_types(adata, UVEAL_MARKERS)

    if not scores:
        print("  ERROR: No marker scores computed")
        return

    # Assign cell type by highest score
    score_df = pd.DataFrame(scores, index=adata.obs_names)
    adata.obs["celltype_major"] = score_df.idxmax(axis=1)

    # Also store individual scores for QC
    for ct, vals in scores.items():
        adata.obs[f"score_{ct}"] = vals

    # Print summary
    print("\n  Cell type distribution:")
    counts = adata.obs["celltype_major"].value_counts()
    for ct, n in counts.items():
        pct = 100 * n / adata.n_obs
        print(f"    {ct}: {n:,} ({pct:.1f}%)")

    # Save
    adata.write_h5ad(h5ad_path)
    print(f"\n  Saved to {h5ad_path}")



def verify_all():
    """Print annotation summary for all datasets."""
    print("\n" + "=" * 60)
    print("  Annotation Summary")
    print("=" * 60)

    datasets = {
        "GSE139829": "celltype_major",
        "GSE176078": "celltype_major",
        "GSE131907": "Cell_type",
        "GSE72056": "malignant",
    }

    for ds, col in datasets.items():
        path = Path(f"data/processed/{ds}/adata_qc.h5ad")
        if not path.exists():
            print(f"\n  {ds}: NOT FOUND")
            continue

        adata = sc.read_h5ad(path)
        if col in adata.obs.columns:
            n_annotated = adata.obs[col].notna().sum()
            n_unique = adata.obs[col].nunique()
            print(f"\n  {ds}: {adata.n_obs} cells, {n_annotated} annotated, "
                  f"{n_unique} types via '{col}'")
            print(f"    {adata.obs[col].value_counts().head(5).to_dict()}")
        else:
            print(f"\n  {ds}: Column '{col}' not found. Available: {list(adata.obs.columns)}")


def main():
    annotate_GSE176078()
    annotate_GSE139829()
    verify_all()
    print("\n  All annotations complete!")


if __name__ == "__main__":
    main()
