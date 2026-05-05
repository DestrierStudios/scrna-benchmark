#!/usr/bin/env python3
"""
Subsample large datasets for computational feasibility.

Methods:
  - Geometric sketching (Geosketch): diversity-preserving
  - Stratified random: preserves cell type proportions
  - Random: baseline comparison

Only subsamples datasets with more cells than the target.
GSE72056 (4,645 cells) is skipped since it's already small.

Usage:
    python scripts/08_subsample.py
    python scripts/08_subsample.py --target 20000 --repeats 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

import yaml
import warnings
warnings.filterwarnings("ignore")


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def subsample_geosketch(adata, target_n, seed=42):
    """Diversity-preserving subsampling using geometric sketching."""
    from geosketch import gs

    np.random.seed(seed)

    # Need PCA for sketching
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(3000, a.n_vars))
    sc.pp.pca(a, n_comps=50, use_highly_variable=True)

    sketch_idx = gs(a.obsm["X_pca"], target_n, replace=False)
    return adata[sketch_idx].copy()


def subsample_stratified(adata, target_n, celltype_col, seed=42):
    """Stratified subsampling preserving cell type proportions."""
    np.random.seed(seed)

    # Get cell type proportions
    ct_counts = adata.obs[celltype_col].value_counts()
    ct_fractions = ct_counts / len(adata)

    indices = []
    for ct, frac in ct_fractions.items():
        ct_idx = adata.obs[adata.obs[celltype_col] == ct].index.tolist()
        n_sample = max(50, int(frac * target_n))  # minimum 50 per type
        n_sample = min(n_sample, len(ct_idx))  # can't sample more than available
        sampled = np.random.choice(ct_idx, n_sample, replace=False)
        indices.extend(sampled)

    return adata[indices].copy()


def subsample_random(adata, target_n, seed=42):
    """Simple random subsampling (baseline)."""
    np.random.seed(seed)
    if target_n >= adata.n_obs:
        return adata.copy()
    indices = np.random.choice(adata.obs_names, target_n, replace=False)
    return adata[indices].copy()


def get_celltype_col(dataset_id):
    """Return the appropriate cell type column for each dataset."""
    mapping = {
        "GSE139829": "celltype_major",
        "GSE176078": "celltype_major",
        "GSE131907": "Cell_type",
    }
    return mapping.get(dataset_id)


def subsample_dataset(dataset_id, config, target_n, n_repeats, methods):
    """Subsample one dataset with multiple methods and repeats."""
    h5ad_path = Path(f"data/processed/{dataset_id}/adata_qc.h5ad")
    output_dir = Path(f"data/subsampled/{dataset_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(h5ad_path)
    print(f"\n{'='*60}")
    print(f"  {dataset_id}: {adata.n_obs} cells")
    print(f"{'='*60}")

    if adata.n_obs <= target_n:
        print(f"  Skipping — fewer cells ({adata.n_obs}) than target ({target_n})")
        return

    ct_col = get_celltype_col(dataset_id)
    report = {"dataset": dataset_id, "original_cells": adata.n_obs, "target": target_n}

    for method in methods:
        print(f"\n  Method: {method}")
        report[method] = {}

        for rep in range(n_repeats):
            seed = 42 + rep
            output_file = output_dir / f"{method}_rep{rep}.h5ad"

            if output_file.exists():
                print(f"    Rep {rep}: already exists, skipping")
                continue

            print(f"    Rep {rep} (seed={seed})...", end=" ", flush=True)

            if method == "geosketch":
                sub = subsample_geosketch(adata, target_n, seed=seed)
            elif method == "stratified":
                if ct_col and ct_col in adata.obs.columns:
                    sub = subsample_stratified(adata, target_n, ct_col, seed=seed)
                else:
                    print(f"No cell type column, falling back to random")
                    sub = subsample_random(adata, target_n, seed=seed)
            elif method == "random":
                sub = subsample_random(adata, target_n, seed=seed)
            else:
                raise ValueError(f"Unknown method: {method}")

            sub.write_h5ad(output_file)
            print(f"{sub.n_obs} cells")

            # Track cell type distribution
            if ct_col and ct_col in sub.obs.columns:
                dist = sub.obs[ct_col].value_counts().to_dict()
                report[method][f"rep{rep}"] = {
                    "n_cells": sub.n_obs,
                    "seed": seed,
                    "cell_types": {str(k): int(v) for k, v in dist.items()},
                }

    # Save report
    report_path = output_dir / "subsample_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")


def verify_proportions(dataset_id, target_n):
    """Compare cell type proportions between original and subsampled."""
    ct_col = get_celltype_col(dataset_id)
    if not ct_col:
        return

    h5ad_path = Path(f"data/processed/{dataset_id}/adata_qc.h5ad")
    sub_dir = Path(f"data/subsampled/{dataset_id}")

    adata = sc.read_h5ad(h5ad_path)
    if ct_col not in adata.obs.columns:
        return

    orig_dist = adata.obs[ct_col].value_counts(normalize=True)

    print(f"\n  Cell type proportion comparison ({dataset_id}):")
    print(f"  {'Type':<25} {'Original':>10} {'Geosketch':>10} {'Stratified':>10} {'Random':>10}")
    print(f"  {'-'*65}")

    geo_path = sub_dir / "geosketch_rep0.h5ad"
    strat_path = sub_dir / "stratified_rep0.h5ad"
    rand_path = sub_dir / "random_rep0.h5ad"

    geo_dist = sc.read_h5ad(geo_path).obs[ct_col].value_counts(normalize=True) if geo_path.exists() else pd.Series()
    strat_dist = sc.read_h5ad(strat_path).obs[ct_col].value_counts(normalize=True) if strat_path.exists() else pd.Series()
    rand_dist = sc.read_h5ad(rand_path).obs[ct_col].value_counts(normalize=True) if rand_path.exists() else pd.Series()

    for ct in orig_dist.index:
        o = f"{orig_dist.get(ct, 0):.3f}"
        g = f"{geo_dist.get(ct, 0):.3f}"
        s = f"{strat_dist.get(ct, 0):.3f}"
        r = f"{rand_dist.get(ct, 0):.3f}"
        print(f"  {str(ct):<25} {o:>10} {g:>10} {s:>10} {r:>10}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--methods", nargs="+", default=["geosketch", "stratified", "random"])
    args = parser.parse_args()

    config = load_config()
    primary = config["primary_datasets"]

    print(f"Subsampling target: {args.target} cells")
    print(f"Repeats: {args.repeats}")
    print(f"Methods: {args.methods}")

    for ds in primary:
        subsample_dataset(ds, config, args.target, args.repeats, args.methods)

    # Verify proportions
    print(f"\n{'='*60}")
    print("  Proportion Verification")
    print(f"{'='*60}")
    for ds in primary:
        verify_proportions(ds, args.target)

    print(f"\n{'='*60}")
    print("  Subsampling complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
