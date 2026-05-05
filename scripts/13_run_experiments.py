#!/usr/bin/env python3
"""
Full experiment runner — Phase 3.

Runs all normalization × clustering × parameter combinations across all
primary datasets on subsampled data. Uses smart caching: normalizes once
per method, then runs all clustering variants on each normalized result.

Architecture:
  For each dataset:
    For each normalization method:
      1. Normalize (slow for R methods, cached)
      2. Dim reduction (cached)
      For each clustering method × parameter:
        3. Cluster (fast)
        4. Evaluate metrics
        5. Compute BDS vs reference
    Aggregate into results CSV

Usage:
    python scripts/13_run_experiments.py                    # all datasets
    python scripts/13_run_experiments.py --dataset GSE139829  # one dataset
    python scripts/13_run_experiments.py --dry-run           # show what would run
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import yaml
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from normalize import normalize
from cluster import cluster
from pipeline import run_dim_reduction
from evaluate import evaluate_clustering
from bds import extract_markers, compute_bds_pairwise, compare_to_reference


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def get_all_cluster_params(config):
    """Build list of (method, param_string) for all clustering configurations."""
    params = []
    for method, cfg in config["clustering"].items():
        if method in ("leiden", "louvain"):
            for r in cfg["resolutions"]:
                params.append((method, f"res{r}"))
        elif method == "hierarchical":
            for k in cfg["n_clusters"]:
                params.append((method, f"k{k}"))
        elif method == "hdbscan":
            for mcs in cfg["min_cluster_sizes"]:
                for ms in cfg["min_samples"]:
                    params.append((method, f"mcs{mcs}_ms{ms}"))
    return params


def get_gt_col(dataset_id):
    """Return ground truth column name for each dataset."""
    mapping = {
        "GSE139829": "celltype_major",
        "GSE176078": "celltype_major",
        "GSE131907": "Cell_type",
    }
    return mapping.get(dataset_id)


def run_dataset_experiments(dataset_id, config, subsample_rep=0, dry_run=False):
    """Run all pipeline combinations for one dataset."""
    norm_methods = list(config["normalization"].keys())
    cluster_params = get_all_cluster_params(config)
    gt_col = get_gt_col(dataset_id)
    seed = config["reproducibility"]["random_seed"]

    # Paths
    sub_path = Path(f"data/subsampled/{dataset_id}/random_rep{subsample_rep}.h5ad")
    cache_dir = Path(f"results/{dataset_id}/cache")
    results_dir = Path(f"results/{dataset_id}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_csv = results_dir / "evaluation_matrix.csv"

    total_combos = len(norm_methods) * len(cluster_params)
    print(f"\n{'='*60}")
    print(f"  {dataset_id}: {len(norm_methods)} norms × {len(cluster_params)} "
          f"cluster configs = {total_combos} pipelines")
    print(f"{'='*60}")

    if dry_run:
        for norm in norm_methods:
            for clust, param in cluster_params:
                print(f"    {norm}_{clust}_{param}")
        return None

    # Load data
    print(f"  Loading {sub_path}...")
    adata_raw = sc.read_h5ad(sub_path)
    print(f"  {adata_raw.n_obs} cells, {adata_raw.n_vars} genes")

    # Load existing results if resuming
    existing_results = {}
    if results_csv.exists():
        existing_df = pd.read_csv(results_csv, index_col=0)
        for idx in existing_df.index:
            existing_results[idx] = existing_df.loc[idx].to_dict()
        print(f"  Resuming: {len(existing_results)} pipelines already completed")

    all_results = dict(existing_results)
    reference_pipe = None  # log + leiden + res0.8
    completed = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    for norm_idx, norm in enumerate(norm_methods):
        # Cache normalized + dim-reduced data
        cache_file = cache_dir / f"{norm}_reduced.h5ad"

        if cache_file.exists():
            print(f"\n  [{norm}] Loading cached normalized data...")
            adata_norm = sc.read_h5ad(cache_file)
        else:
            print(f"\n  [{norm}] Normalizing ({norm_idx+1}/{len(norm_methods)})...",
                  end=" ", flush=True)
            t = time.time()
            try:
                adata_norm = normalize(adata_raw, norm)
                adata_norm = run_dim_reduction(adata_norm, seed=seed)
                adata_norm.write_h5ad(cache_file)
                print(f"done ({time.time()-t:.1f}s)")
            except Exception as e:
                print(f"FAILED — {e}")
                failed += len(cluster_params)
                continue

        # Run all clustering variants on this normalized data
        for clust, param in cluster_params:
            pipeline_name = f"{norm}_{clust}_{param}"

            if pipeline_name in existing_results:
                skipped += 1
                continue

            t = time.time()
            try:
                adata_clust = cluster(adata_norm, clust, param_str=param, seed=seed)
                elapsed = time.time() - t

                # Evaluate
                metrics = evaluate_clustering(
                    adata_clust, true_label_col=gt_col, seed=seed
                )
                metrics["normalization"] = norm
                metrics["clustering"] = clust
                metrics["params"] = param
                metrics["pipeline"] = pipeline_name
                metrics["time_cluster"] = elapsed

                # Store reference pipeline
                if norm == "log" and clust == "leiden" and param == "res0.8":
                    reference_pipe = adata_clust.copy()

                all_results[pipeline_name] = metrics
                completed += 1

                # Progress
                total_done = completed + skipped
                pct = 100 * total_done / total_combos
                elapsed_total = time.time() - start_time
                rate = total_done / max(elapsed_total, 1)
                remaining = (total_combos - total_done) / max(rate, 0.01)
                n_clust = metrics.get("n_clusters", "?")
                print(f"    {pipeline_name}: {n_clust} clusters "
                      f"({pct:.0f}%, ~{remaining:.0f}s remaining)")

            except Exception as e:
                all_results[pipeline_name] = {
                    "normalization": norm, "clustering": clust,
                    "params": param, "pipeline": pipeline_name,
                    "error": str(e),
                }
                failed += 1
                print(f"    {pipeline_name}: FAILED — {e}")

        # Save intermediate results after each normalization method
        results_df = pd.DataFrame(all_results).T
        results_df.to_csv(results_csv)

    # Compute BDS vs reference for all pipelines
    if reference_pipe is not None:
        print(f"\n  Computing BDS vs reference (log_leiden_res0.8)...")
        ref_markers_dict, ref_markers_set = extract_markers(reference_pipe, n_markers=50)

        for pipeline_name, metrics in all_results.items():
            if "error" in metrics:
                continue
            if pipeline_name == "log_leiden_res0.8":
                all_results[pipeline_name]["global_bds"] = 0.0
                continue

            try:
                # Reconstruct pipeline output from cache
                norm = metrics["normalization"]
                clust_method = metrics["clustering"]
                param = metrics["params"]

                cache_file = cache_dir / f"{norm}_reduced.h5ad"
                if not cache_file.exists():
                    continue

                adata_norm = sc.read_h5ad(cache_file)
                adata_clust = cluster(adata_norm, clust_method, param_str=param, seed=seed)

                _, test_markers = extract_markers(adata_clust, n_markers=50)
                bds = compute_bds_pairwise(test_markers, ref_markers_set)
                all_results[pipeline_name]["global_bds"] = bds

            except Exception:
                all_results[pipeline_name]["global_bds"] = np.nan

    # Final save
    results_df = pd.DataFrame(all_results).T
    results_df.to_csv(results_csv)

    elapsed_total = time.time() - start_time
    print(f"\n  Done: {completed} completed, {skipped} skipped, {failed} failed")
    print(f"  Total time: {elapsed_total/60:.1f} minutes")
    print(f"  Results: {results_csv}")

    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run one dataset (e.g., GSE139829)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    parser.add_argument("--rep", type=int, default=0,
                        help="Subsample replicate to use (0-4)")
    args = parser.parse_args()

    config = load_config()
    primary = config["primary_datasets"]

    if args.dataset:
        if args.dataset not in primary:
            print(f"ERROR: {args.dataset} not in primary datasets: {primary}")
            sys.exit(1)
        datasets = [args.dataset]
    else:
        datasets = primary

    print(f"Phase 3: Full Experiments")
    print(f"Datasets: {datasets}")

    all_dfs = {}
    for ds in datasets:
        df = run_dataset_experiments(ds, config, subsample_rep=args.rep,
                                     dry_run=args.dry_run)
        if df is not None:
            all_dfs[ds] = df

    if not args.dry_run and all_dfs:
        # Save combined results
        combined = pd.concat(
            [df.assign(dataset=ds) for ds, df in all_dfs.items()],
            ignore_index=True,
        )
        combined_path = Path("results/tables/all_results.csv")
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined results: {combined_path} ({len(combined)} rows)")

    print("\nExperiments complete!")


if __name__ == "__main__":
    main()
