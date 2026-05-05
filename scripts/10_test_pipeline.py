#!/usr/bin/env python3
"""
Test clustering methods and full pipeline on subsampled data.

Usage:
    python scripts/10_test_pipeline.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import scanpy as sc

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from normalize import normalize
from cluster import METHODS as CLUST_METHODS, cluster, parse_param_string
from pipeline import run_pipeline


def test_clustering(adata_reduced):
    """Test all clustering methods on pre-reduced data."""
    print(f"\n{'='*60}")
    print(f"  Testing 4 clustering methods")
    print(f"{'='*60}")

    test_params = {
        "leiden": ["res0.4", "res0.8", "res1.2"],
        "louvain": ["res0.8"],
        "hierarchical": ["k10", "k20"],
        "hdbscan": ["mcs50_ms10"],
    }

    results = {}
    for method, params in test_params.items():
        for param_str in params:
            name = f"{method}_{param_str}"
            print(f"\n  {name}:", end=" ", flush=True)
            t = time.time()
            try:
                result = cluster(adata_reduced, method, param_str=param_str)
                elapsed = time.time() - t
                n_clusters = result.obs["clusters"].nunique()
                print(f"PASS ({elapsed:.1f}s) — {n_clusters} clusters")
                results[name] = True
            except Exception as e:
                elapsed = time.time() - t
                print(f"FAIL ({elapsed:.1f}s) — {e}")
                results[name] = False

    return results


def test_full_pipelines(adata_raw):
    """Test a few representative full pipelines."""
    print(f"\n{'='*60}")
    print(f"  Testing full pipelines (norm → PCA → cluster)")
    print(f"{'='*60}")

    test_combos = [
        ("log", "leiden", "res0.8"),
        ("log", "hierarchical", "k15"),
        ("pearson_residuals", "leiden", "res0.8"),
        ("scran", "louvain", "res0.8"),
        ("sctransform", "leiden", "res1.0"),
        ("analytic_pearson", "hdbscan", "mcs50_ms10"),
    ]

    results = {}
    for norm, clust, param in test_combos:
        name = f"{norm}_{clust}_{param}"
        print(f"\n  {name}:", end=" ", flush=True)
        t = time.time()
        try:
            result = run_pipeline(adata_raw, norm, clust, param)
            elapsed = time.time() - t
            n_clusters = result.obs["clusters"].nunique()
            timings = result.uns["pipeline"]["timings"]
            print(f"PASS ({elapsed:.1f}s) — {n_clusters} clusters "
                  f"[norm={timings['normalization']:.1f}s, "
                  f"dim={timings['dim_reduction']:.1f}s, "
                  f"clust={timings['clustering']:.1f}s]")
            results[name] = True
        except Exception as e:
            elapsed = time.time() - t
            print(f"FAIL ({elapsed:.1f}s) — {e}")
            results[name] = False

    return results


def main():
    # Load subsampled data
    sub_path = Path("data/subsampled/GSE139829/random_rep0.h5ad")
    print(f"Loading: {sub_path}")
    adata_raw = sc.read_h5ad(sub_path)

    # Use 5K subset for speed
    np.random.seed(42)
    idx = np.random.choice(adata_raw.obs_names, 5000, replace=False)
    adata_raw = adata_raw[idx].copy()
    print(f"Test data: {adata_raw.n_obs} cells, {adata_raw.n_vars} genes")

    # Prepare a pre-normalized + reduced dataset for clustering-only tests
    print("\nPreparing normalized + reduced data for clustering tests...")
    adata_norm = normalize(adata_raw, "log")
    from pipeline import run_dim_reduction
    adata_reduced = run_dim_reduction(adata_norm)
    print(f"  PCA: {adata_reduced.obsm['X_pca'].shape[1]} components")

    # Test clustering
    clust_results = test_clustering(adata_reduced)

    # Test full pipelines
    pipe_results = test_full_pipelines(adata_raw)

    # Summary
    all_results = {**clust_results, **pipe_results}
    n_pass = sum(1 for v in all_results.values() if v)
    n_fail = sum(1 for v in all_results.values() if not v)

    print(f"\n{'='*60}")
    print(f"  Summary: {n_pass}/{len(all_results)} passed, {n_fail} failed")
    print(f"{'='*60}")

    if n_fail > 0:
        print("\n  Failed tests:")
        for name, passed in all_results.items():
            if not passed:
                print(f"    - {name}")
        sys.exit(1)
    else:
        print("\n  All tests passed!")


if __name__ == "__main__":
    main()
