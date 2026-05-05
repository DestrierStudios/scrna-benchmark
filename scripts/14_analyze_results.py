#!/usr/bin/env python3
"""
Phase 3 analysis: Meta-ranking, targeted BDS, and cross-dataset aggregation.

Loads evaluation matrices from all datasets, computes:
1. Meta-ranking per dataset (Borda count + bootstrap CIs)
2. BDS on top-20 pipelines only (avoids OOM from full 155)
3. Cross-dataset aggregated ranking
4. Summary statistics for the paper

Usage:
    python scripts/14_analyze_results.py
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import yaml
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from meta_ranking import borda_rank, bootstrap_borda, aggregate_rankings
from cluster import cluster
from bds import extract_markers, compute_bds_pairwise


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def load_results(dataset_id):
    """Load evaluation matrix for one dataset."""
    path = Path(f"results/{dataset_id}/evaluation_matrix.csv")
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return None
    df = pd.read_csv(path, index_col=0)
    print(f"  {dataset_id}: {len(df)} pipelines loaded")
    return df


def clean_scores(df):
    """Extract numeric metric columns and clean for ranking."""
    # Metrics to use for ranking
    rank_metrics = [
        "ARI", "NMI", "homogeneity", "completeness",
        "cell_type_purity", "silhouette", "calinski_harabasz", "davies_bouldin",
    ]
    available = [m for m in rank_metrics if m in df.columns]

    scores = df[available].copy()
    # Convert to numeric
    for col in scores.columns:
        scores[col] = pd.to_numeric(scores[col], errors="coerce")

    # Drop rows that are all NaN
    scores = scores.dropna(how="all")

    # Set pipeline names as index
    if "pipeline" in df.columns:
        scores.index = df.loc[scores.index, "pipeline"].values

    return scores


def compute_targeted_bds(dataset_id, top_pipelines, config, n_top=20):
    """Compute BDS only for top-N pipelines vs reference."""
    print(f"\n  Computing BDS for top {n_top} pipelines...")

    cache_dir = Path(f"results/{dataset_id}/cache")
    seed = config["reproducibility"]["random_seed"]

    # Reference: log + leiden + res0.8
    ref_cache = cache_dir / "log_reduced.h5ad"
    if not ref_cache.exists():
        print(f"    WARNING: Reference cache not found, skipping BDS")
        return {}

    ref_adata = sc.read_h5ad(ref_cache)
    ref_adata = cluster(ref_adata, "leiden", param_str="res0.8", seed=seed)
    _, ref_markers = extract_markers(ref_adata, n_markers=50)
    del ref_adata

    bds_results = {}
    for i, pipeline_name in enumerate(top_pipelines[:n_top]):
        parts = pipeline_name.split("_")

        # Parse normalization (may contain underscore)
        norm = None
        for n in ["log", "scran", "sctransform", "pearson_residuals", "analytic_pearson"]:
            if pipeline_name.startswith(n + "_"):
                norm = n
                break
        if norm is None:
            continue

        remainder = pipeline_name[len(norm) + 1:]
        # Parse clustering method
        clust_method = None
        for c in ["leiden", "louvain", "hierarchical", "hdbscan"]:
            if remainder.startswith(c + "_"):
                clust_method = c
                break
        if clust_method is None:
            continue

        param = remainder[len(clust_method) + 1:]
        cache_file = cache_dir / f"{norm}_reduced.h5ad"

        if not cache_file.exists():
            continue

        try:
            adata = sc.read_h5ad(cache_file)
            adata = cluster(adata, clust_method, param_str=param, seed=seed)
            _, test_markers = extract_markers(adata, n_markers=50)
            bds = compute_bds_pairwise(test_markers, ref_markers)
            bds_results[pipeline_name] = bds
            print(f"    [{i+1}/{min(n_top, len(top_pipelines))}] {pipeline_name}: BDS={bds:.4f}")
            del adata
        except Exception as e:
            print(f"    [{i+1}] {pipeline_name}: FAILED — {e}")
            bds_results[pipeline_name] = np.nan

    return bds_results


def analyze_dataset(dataset_id, config):
    """Full analysis for one dataset."""
    print(f"\n{'='*60}")
    print(f"  Analyzing {dataset_id}")
    print(f"{'='*60}")

    df = load_results(dataset_id)
    if df is None:
        return None

    scores = clean_scores(df)
    print(f"  Ranking {len(scores)} pipelines on {len(scores.columns)} metrics")
    print(f"  Metrics: {list(scores.columns)}")

    # Basic Borda ranking
    ranking = borda_rank(scores)
    print(f"\n  Top 10 pipelines:")
    for _, row in ranking.head(10).iterrows():
        print(f"    #{int(row['rank'])}: {row['pipeline']} (score={row['borda_score']:.1f})")

    # Bootstrap ranking
    print(f"\n  Bootstrap ranking (500 iterations)...")
    boot_ranking = bootstrap_borda(scores, n_bootstrap=500, seed=42)

    print(f"\n  {'Pipeline':<45} {'Rank':>4} {'Mean':>5} {'95% CI':>10} {'P(top5)':>8}")
    print(f"  {'-'*72}")
    for _, row in boot_ranking.head(15).iterrows():
        ci = f"[{row['rank_ci_lower']:.0f}-{row['rank_ci_upper']:.0f}]"
        print(f"  {row['pipeline']:<45} {int(row['rank']):>4} {row['mean_rank']:>5.1f} "
              f"{ci:>10} {row['prob_top5']:>7.0%}")

    # Targeted BDS on top 20
    top_names = boot_ranking.head(20)["pipeline"].tolist()
    bds_scores = compute_targeted_bds(dataset_id, top_names, config, n_top=20)

    # Add BDS to ranking
    if bds_scores:
        boot_ranking["global_bds"] = boot_ranking["pipeline"].map(bds_scores)

    # Save results
    output_dir = Path(f"results/{dataset_id}")
    boot_ranking.to_csv(output_dir / "meta_ranking.csv", index=False)
    print(f"\n  Saved: {output_dir / 'meta_ranking.csv'}")

    # Summary stats for paper
    summary = {
        "dataset": dataset_id,
        "n_pipelines": len(scores),
        "n_metrics": len(scores.columns),
        "top_pipeline": boot_ranking.iloc[0]["pipeline"],
        "top_borda_score": float(boot_ranking.iloc[0]["borda_score"]),
    }

    # Normalization breakdown
    print(f"\n  Normalization method summary (mean rank):")
    norm_ranks = {}
    for norm in ["log", "scran", "sctransform", "pearson_residuals", "analytic_pearson"]:
        mask = boot_ranking["pipeline"].str.startswith(norm + "_")
        if mask.any():
            mean_r = boot_ranking.loc[mask, "mean_rank"].mean()
            norm_ranks[norm] = mean_r
            print(f"    {norm:<25} mean rank: {mean_r:.1f}")

    # Clustering breakdown
    print(f"\n  Clustering method summary (mean rank):")
    for clust in ["leiden", "louvain", "hierarchical", "hdbscan"]:
        mask = boot_ranking["pipeline"].str.contains(f"_{clust}_")
        if mask.any():
            mean_r = boot_ranking.loc[mask, "mean_rank"].mean()
            print(f"    {clust:<25} mean rank: {mean_r:.1f}")

    return boot_ranking


def cross_dataset_analysis(all_rankings):
    """Aggregate rankings across datasets."""
    print(f"\n{'='*60}")
    print(f"  Cross-Dataset Aggregation")
    print(f"{'='*60}")

    agg = aggregate_rankings(all_rankings, method="mean_rank")
    print(f"\n  Overall ranking ({len(agg)} pipelines):")
    print(f"  {'Pipeline':<45} {'Overall Rank':>12} {'Mean Rank':>10}")
    print(f"  {'-'*67}")
    for _, row in agg.head(20).iterrows():
        print(f"  {row['pipeline']:<45} {int(row['overall_rank']):>12} "
              f"{row['mean_rank_across_datasets']:>10.1f}")

    # Save
    output_path = Path("results/tables/cross_dataset_ranking.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")

    # Key findings for paper
    print(f"\n  Key findings:")
    top5 = agg.head(5)["pipeline"].tolist()
    norms_in_top5 = [p.split("_")[0] if not p.startswith("pearson") and not p.startswith("analytic")
                     else "_".join(p.split("_")[:2]) for p in top5]
    print(f"    Top 5 pipelines: {top5}")
    print(f"    Normalization methods in top 5: {set(norms_in_top5)}")

    # Best per normalization
    print(f"\n  Best pipeline per normalization method:")
    for norm in ["log", "scran", "sctransform", "pearson_residuals", "analytic_pearson"]:
        mask = agg["pipeline"].str.startswith(norm + "_")
        if mask.any():
            best = agg.loc[mask].iloc[0]
            print(f"    {norm:<25} #{int(best['overall_rank'])}: {best['pipeline']}")

    return agg


def main():
    config = load_config()
    primary = config["primary_datasets"]

    print("Phase 3 Analysis: Meta-Ranking + BDS + Cross-Dataset Aggregation")
    print(f"Datasets: {primary}")

    # Analyze each dataset
    all_rankings = {}
    for ds in primary:
        ranking = analyze_dataset(ds, config)
        if ranking is not None:
            all_rankings[ds] = ranking

    # Cross-dataset aggregation
    if len(all_rankings) >= 2:
        agg = cross_dataset_analysis(all_rankings)

    # Save combined summary
    summary_path = Path("results/tables/experiment_summary.json")
    summary = {
        "n_datasets": len(all_rankings),
        "n_pipelines_per_dataset": 155,
        "total_pipelines": 155 * len(all_rankings),
        "datasets": list(all_rankings.keys()),
        "normalizations": ["log", "scran", "sctransform", "pearson_residuals", "analytic_pearson"],
        "clustering": ["leiden", "louvain", "hierarchical", "hdbscan"],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Analysis complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
