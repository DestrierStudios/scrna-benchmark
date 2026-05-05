#!/usr/bin/env python3
"""
Generate supplementary materials for the manuscript.

Tables:
  S1: Full evaluation matrix (all 465 pipelines, all metrics)
  S2: Per-dataset meta-rankings with bootstrap CIs
  S3: Cross-platform validation results
  S4: Published marker recovery per pipeline
  S5: BDS matrix for top pipelines

Usage:
    python scripts/17_supplementary_tables.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import warnings
warnings.filterwarnings("ignore")

SUPP_DIR = Path("results/supplementary")
SUPP_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAMES = {
    "GSE139829": "Uveal Melanoma",
    "GSE176078": "Breast Cancer",
    "GSE131907": "Lung Adenocarcinoma",
}


def table_s1_full_results():
    """S1: Complete evaluation matrix across all datasets."""
    print("  Table S1: Full evaluation matrix...")
    frames = []
    for ds, name in DATASET_NAMES.items():
        path = Path(f"results/{ds}/evaluation_matrix.csv")
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            df["dataset"] = name
            df["dataset_id"] = ds
            frames.append(df)

    if not frames:
        print("    No data found")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Reorder columns
    id_cols = ["dataset", "dataset_id", "pipeline", "normalization", "clustering", "params"]
    metric_cols = ["n_clusters", "ARI", "NMI", "homogeneity", "completeness",
                   "cell_type_purity", "silhouette", "calinski_harabasz", "davies_bouldin"]
    other_cols = [c for c in combined.columns if c not in id_cols + metric_cols]
    cols = [c for c in id_cols + metric_cols + other_cols if c in combined.columns]
    combined = combined[cols]

    combined.to_csv(SUPP_DIR / "table_s1_full_results.csv", index=False)
    print(f"    {len(combined)} rows saved")


def table_s2_rankings():
    """S2: Per-dataset meta-rankings with bootstrap CIs."""
    print("  Table S2: Meta-rankings...")
    frames = []
    for ds, name in DATASET_NAMES.items():
        path = Path(f"results/{ds}/meta_ranking.csv")
        if path.exists():
            df = pd.read_csv(path)
            df["dataset"] = name
            df["dataset_id"] = ds
            frames.append(df)

    if not frames:
        print("    No data found")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(SUPP_DIR / "table_s2_meta_rankings.csv", index=False)
    print(f"    {len(combined)} rows saved")

    # Also save cross-dataset ranking
    cross_path = Path("results/tables/cross_dataset_ranking.csv")
    if cross_path.exists():
        df = pd.read_csv(cross_path)
        df.to_csv(SUPP_DIR / "table_s2b_cross_dataset_ranking.csv", index=False)
        print(f"    Cross-dataset: {len(df)} rows saved")


def table_s3_cross_platform():
    """S3: Cross-platform validation results."""
    print("  Table S3: Cross-platform validation...")

    metrics_path = Path("results/GSE72056/clustering_metrics.csv")
    comparison_path = Path("results/GSE72056/cross_platform_comparison.csv")

    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df.to_csv(SUPP_DIR / "table_s3a_smartseq2_metrics.csv", index=False)
        print(f"    Smart-seq2 metrics: {len(df)} rows")

    if comparison_path.exists():
        df = pd.read_csv(comparison_path, index_col=0)
        df.to_csv(SUPP_DIR / "table_s3b_cross_platform_overlap.csv")
        print(f"    Cross-platform overlap: {len(df)} rows")


def table_s4_marker_recovery():
    """S4: Summary of normalization method performance."""
    print("  Table S4: Normalization summary...")

    rows = []
    for ds, name in DATASET_NAMES.items():
        path = Path(f"results/{ds}/evaluation_matrix.csv")
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        if "normalization" not in df.columns:
            continue

        for norm in ["log", "scran", "sctransform", "pearson_residuals", "analytic_pearson"]:
            mask = df["normalization"] == norm
            if not mask.any():
                continue
            sub = df.loc[mask]
            row = {"dataset": name, "normalization": norm, "n_pipelines": int(mask.sum())}
            for metric in ["ARI", "NMI", "silhouette", "cell_type_purity",
                           "calinski_harabasz", "davies_bouldin"]:
                if metric in sub.columns:
                    vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
                    if len(vals) > 0:
                        row[f"{metric}_mean"] = round(vals.mean(), 4)
                        row[f"{metric}_std"] = round(vals.std(), 4)
                        row[f"{metric}_best"] = round(vals.max(), 4) if metric != "davies_bouldin" else round(vals.min(), 4)
            rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(SUPP_DIR / "table_s4_normalization_summary.csv", index=False)
        print(f"    {len(rows)} rows saved")


def table_s5_clustering_summary():
    """S5: Summary of clustering algorithm performance."""
    print("  Table S5: Clustering summary...")

    rows = []
    for ds, name in DATASET_NAMES.items():
        path = Path(f"results/{ds}/evaluation_matrix.csv")
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        if "clustering" not in df.columns:
            continue

        for clust in ["leiden", "louvain", "hierarchical", "hdbscan"]:
            mask = df["clustering"] == clust
            if not mask.any():
                continue
            sub = df.loc[mask]
            row = {"dataset": name, "clustering": clust, "n_pipelines": int(mask.sum())}
            for metric in ["ARI", "NMI", "silhouette", "cell_type_purity"]:
                if metric in sub.columns:
                    vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
                    if len(vals) > 0:
                        row[f"{metric}_mean"] = round(vals.mean(), 4)
                        row[f"{metric}_std"] = round(vals.std(), 4)
            rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(SUPP_DIR / "table_s5_clustering_summary.csv", index=False)
        print(f"    {len(rows)} rows saved")


def main():
    print("Generating supplementary materials...")
    print(f"Output: {SUPP_DIR}\n")

    table_s1_full_results()
    table_s2_rankings()
    table_s3_cross_platform()
    table_s4_marker_recovery()
    table_s5_clustering_summary()

    print(f"\nGenerated files:")
    for f in sorted(SUPP_DIR.glob("*.csv")):
        size = f.stat().st_size / 1024
        print(f"  {f.name} ({size:.0f} KB)")

    print("\nSupplementary materials complete!")


if __name__ == "__main__":
    main()
