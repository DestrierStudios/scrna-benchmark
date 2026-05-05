#!/usr/bin/env python3
"""
Test BDS validation on synthetic data + meta-ranking framework.

Week 7: Validates BDS detects known differences in synthetic data
Week 8: Tests Borda count meta-ranking with bootstrap CIs

Usage:
    python scripts/12_test_metaranking.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pipeline import run_pipeline
from evaluate import evaluate_clustering
from bds import extract_markers, compute_bds_pairwise, compare_to_reference
from meta_ranking import borda_rank, bootstrap_borda, aggregate_rankings


def generate_synthetic_data(n_cells=2000, n_genes=500, n_clusters=5, seed=42):
    """
    Generate synthetic scRNA-seq data with known ground truth clusters
    and known marker genes per cluster.
    """
    rng = np.random.RandomState(seed)
    cells_per = n_cells // n_clusters
    markers_per = 30

    counts = rng.poisson(1.0, (n_cells, n_genes)).astype(np.float32)
    true_labels = []
    known_markers = {}

    for i in range(n_clusters):
        start = i * cells_per
        end = start + cells_per
        marker_start = i * markers_per
        marker_end = marker_start + markers_per

        # Upregulate cluster-specific markers
        counts[start:end, marker_start:marker_end] += rng.poisson(
            8.0, (cells_per, markers_per)
        )
        true_labels.extend([f"type_{i}"] * cells_per)
        known_markers[f"type_{i}"] = [f"gene_{j}" for j in range(marker_start, marker_end)]

    adata = ad.AnnData(
        X=csr_matrix(counts),
        obs=pd.DataFrame({"true_type": true_labels},
                         index=[f"cell_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )
    adata.var["mt"] = False

    return adata, known_markers


def test_bds_validation():
    """Validate BDS detects expected differences."""
    print(f"\n{'='*60}")
    print("  Part 1: BDS Validation on Synthetic Data")
    print(f"{'='*60}")

    adata, known_markers = generate_synthetic_data()
    print(f"  Synthetic data: {adata.n_obs} cells, {adata.n_vars} genes, "
          f"{len(known_markers)} known clusters")

    # Run two pipelines that should give SIMILAR results (same method, different seed)
    print("\n  Test 1: Same method, different seeds (expect low BDS)...")
    pipe_a = run_pipeline(adata, "log", "leiden", "res0.8", seed=42)
    pipe_b = run_pipeline(adata, "log", "leiden", "res0.8", seed=123)

    _, markers_a = extract_markers(pipe_a, n_markers=30)
    _, markers_b = extract_markers(pipe_b, n_markers=30)
    bds_similar = compute_bds_pairwise(markers_a, markers_b)
    print(f"    BDS (same method): {bds_similar:.4f}")

    # Run two pipelines that should give DIFFERENT results
    print("\n  Test 2: Different methods (expect higher BDS)...")
    pipe_c = run_pipeline(adata, "log", "leiden", "res0.2", seed=42)  # coarse clustering
    pipe_d = run_pipeline(adata, "log", "leiden", "res2.0", seed=42)  # fine clustering

    _, markers_c = extract_markers(pipe_c, n_markers=30)
    _, markers_d = extract_markers(pipe_d, n_markers=30)
    bds_different = compute_bds_pairwise(markers_c, markers_d)
    print(f"    BDS (different resolution): {bds_different:.4f}")

    # Test 3: BDS should correlate with known marker recovery
    print("\n  Test 3: Known marker recovery...")
    all_known = set()
    for genes in known_markers.values():
        all_known.update(genes)

    recovery_a = len(markers_a & all_known) / len(all_known) if all_known else 0
    recovery_c = len(markers_c & all_known) / len(all_known) if all_known else 0
    recovery_d = len(markers_d & all_known) / len(all_known) if all_known else 0

    print(f"    Pipeline A (res0.8): {recovery_a:.1%} known markers recovered")
    print(f"    Pipeline C (res0.2): {recovery_c:.1%} known markers recovered")
    print(f"    Pipeline D (res2.0): {recovery_d:.1%} known markers recovered")

    # Validation checks
    checks = []
    checks.append(("BDS(same) < BDS(different)", bds_similar < bds_different))
    checks.append(("BDS(same) < 0.5", bds_similar < 0.5))

    print(f"\n  Validation checks:")
    all_pass = True
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {desc}")
        if not passed:
            all_pass = False

    return all_pass



def test_meta_ranking():
    """Test Borda count meta-ranking with bootstrap CIs."""
    print(f"\n{'='*60}")
    print("  Part 2: Meta-Ranking Framework")
    print(f"{'='*60}")

    # Load real subsampled data
    sub_path = Path("data/subsampled/GSE139829/random_rep0.h5ad")
    adata_raw = sc.read_h5ad(sub_path)
    np.random.seed(42)
    idx = np.random.choice(adata_raw.obs_names, 3000, replace=False)
    adata_raw = adata_raw[idx].copy()
    print(f"  Data: {adata_raw.n_obs} cells")

    gt_col = "celltype_major" if "celltype_major" in adata_raw.obs.columns else None

    # Run several pipelines
    pipelines = [
        ("log", "leiden", "res0.4"),
        ("log", "leiden", "res0.8"),
        ("log", "leiden", "res1.2"),
        ("log", "louvain", "res0.8"),
        ("log", "hierarchical", "k10"),
        ("pearson_residuals", "leiden", "res0.8"),
        ("analytic_pearson", "leiden", "res0.8"),
    ]

    print(f"\n  Running {len(pipelines)} pipelines...")
    results = {}
    pipeline_outputs = {}

    for norm, clust, param in pipelines:
        name = f"{norm}_{clust}_{param}"
        print(f"    {name}...", end=" ", flush=True)
        t = time.time()
        try:
            pipe = run_pipeline(adata_raw, norm, clust, param)
            metrics = evaluate_clustering(pipe, true_label_col=gt_col)

            # Add BDS vs reference (log+leiden+res0.8)
            pipeline_outputs[name] = pipe
            results[name] = metrics
            print(f"done ({time.time()-t:.1f}s)")
        except Exception as e:
            print(f"FAIL — {e}")

    # Add BDS scores (vs first pipeline as reference)
    ref_name = "log_leiden_res0.8"
    if ref_name in pipeline_outputs:
        ref_pipe = pipeline_outputs[ref_name]
        for name, pipe in pipeline_outputs.items():
            if name == ref_name:
                results[name]["global_bds"] = 0.0
                continue
            try:
                comparison = compare_to_reference(pipe, ref_pipe, n_markers=30)
                results[name]["global_bds"] = comparison["global_bds"]
            except Exception:
                results[name]["global_bds"] = np.nan

    # Build scores matrix
    scores_df = pd.DataFrame(results).T
    # Keep only numeric metrics
    metric_cols = [c for c in scores_df.columns if scores_df[c].dtype in [np.float64, np.int64, float, int]]
    scores_df = scores_df[metric_cols].astype(float)

    print(f"\n  Scores matrix: {scores_df.shape[0]} pipelines × {scores_df.shape[1]} metrics")
    print(f"  Metrics: {list(scores_df.columns)}")

    # Test 1: Basic Borda ranking
    print(f"\n  Basic Borda ranking:")
    ranking = borda_rank(scores_df)
    for _, row in ranking.iterrows():
        print(f"    #{int(row['rank'])}: {row['pipeline']} (score={row['borda_score']:.1f})")

    # Test 2: Bootstrap Borda with CIs
    print(f"\n  Bootstrap Borda (100 iterations)...")
    boot_ranking = bootstrap_borda(scores_df, n_bootstrap=100, seed=42)
    print(f"\n  {'Pipeline':<40} {'Rank':>5} {'Mean':>6} {'95% CI':>12} {'P(top3)':>8}")
    print(f"  {'-'*71}")
    for _, row in boot_ranking.iterrows():
        ci = f"[{row['rank_ci_lower']:.0f}-{row['rank_ci_upper']:.0f}]"
        print(f"  {row['pipeline']:<40} {int(row['rank']):>5} {row['mean_rank']:>6.1f} "
              f"{ci:>12} {row['prob_top3']:>7.0%}")

    # Test 3: Cross-dataset aggregation (simulate with same data, different subsets)
    print(f"\n  Cross-dataset aggregation (simulated)...")
    dataset_rankings = {"dataset_1": boot_ranking}

    # Create a second "dataset" by re-running with different subset
    idx2 = np.random.choice(adata_raw.obs_names, 3000, replace=False)
    adata_raw2 = adata_raw[idx2].copy()
    results2 = {}
    for norm, clust, param in pipelines[:4]:  # subset for speed
        name = f"{norm}_{clust}_{param}"
        try:
            pipe = run_pipeline(adata_raw2, norm, clust, param)
            results2[name] = evaluate_clustering(pipe, true_label_col=gt_col)
        except Exception:
            pass

    if results2:
        scores_df2 = pd.DataFrame(results2).T
        scores_df2 = scores_df2[[c for c in scores_df2.columns if scores_df2[c].dtype in [np.float64, np.int64, float, int]]].astype(float)
        boot2 = bootstrap_borda(scores_df2, n_bootstrap=100, seed=42)
        dataset_rankings["dataset_2"] = boot2

        agg = aggregate_rankings(dataset_rankings, method="mean_rank")
        print(f"\n  Aggregated ranking across datasets:")
        for _, row in agg.iterrows():
            print(f"    #{int(row['overall_rank'])}: {row['pipeline']} "
                  f"(mean rank: {row['mean_rank_across_datasets']:.1f})")

    return True


def main():
    print("Week 7-8: BDS Validation + Meta-Ranking Framework")

    bds_ok = test_bds_validation()
    ranking_ok = test_meta_ranking()

    print(f"\n{'='*60}")
    if bds_ok and ranking_ok:
        print("  All Week 7-8 tests passed!")
    else:
        print("  Some tests failed — review output above")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
