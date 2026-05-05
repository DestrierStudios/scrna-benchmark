#!/usr/bin/env python3
"""
Test evaluation metrics and BDS on real pipeline outputs.

Runs two different pipelines, then computes all metrics including BDS.

Usage:
    python scripts/11_test_evaluation.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import yaml

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pipeline import run_pipeline
from evaluate import evaluate_clustering, compute_supervised_metrics
from bds import extract_markers, compute_bds_pairwise, compare_to_reference, compute_marker_recovery


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

    # Check for ground truth labels
    gt_col = "celltype_major" if "celltype_major" in adata_raw.obs.columns else None
    if gt_col:
        n_types = adata_raw.obs[gt_col].nunique()
        print(f"Ground truth: '{gt_col}' with {n_types} types")
    else:
        print("No ground truth labels available")

    print(f"\n{'='*60}")
    print("  Running pipelines")
    print(f"{'='*60}")

    # Pipeline A: log + Leiden (baseline/reference)
    print("\n  Pipeline A (log + Leiden res0.8)...", end=" ", flush=True)
    t = time.time()
    pipe_a = run_pipeline(adata_raw, "log", "leiden", "res0.8")
    print(f"done ({time.time()-t:.1f}s, {pipe_a.obs['clusters'].nunique()} clusters)")

    # Pipeline B: pearson + Leiden
    print("  Pipeline B (pearson + Leiden res0.8)...", end=" ", flush=True)
    t = time.time()
    pipe_b = run_pipeline(adata_raw, "pearson_residuals", "leiden", "res0.8")
    print(f"done ({time.time()-t:.1f}s, {pipe_b.obs['clusters'].nunique()} clusters)")

    # Pipeline C: log + Hierarchical (different clustering)
    print("  Pipeline C (log + Hierarchical k15)...", end=" ", flush=True)
    t = time.time()
    pipe_c = run_pipeline(adata_raw, "log", "hierarchical", "k15")
    print(f"done ({time.time()-t:.1f}s, {pipe_c.obs['clusters'].nunique()} clusters)")

    print(f"\n{'='*60}")
    print("  Evaluation metrics")
    print(f"{'='*60}")

    for name, pipe in [("A (log+leiden)", pipe_a),
                       ("B (pearson+leiden)", pipe_b),
                       ("C (log+hierarchical)", pipe_c)]:
        print(f"\n  Pipeline {name}:")
        metrics = evaluate_clustering(pipe, true_label_col=gt_col)
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    print(f"\n{'='*60}")
    print("  Biological Discordance Score (BDS)")
    print(f"{'='*60}")

    # Extract markers
    print("\n  Extracting markers...")
    markers_dict_a, markers_set_a = extract_markers(pipe_a, n_markers=50)
    markers_dict_b, markers_set_b = extract_markers(pipe_b, n_markers=50)
    markers_dict_c, markers_set_c = extract_markers(pipe_c, n_markers=50)

    print(f"    Pipeline A: {len(markers_set_a)} unique markers across {len(markers_dict_a)} clusters")
    print(f"    Pipeline B: {len(markers_set_b)} unique markers across {len(markers_dict_b)} clusters")
    print(f"    Pipeline C: {len(markers_set_c)} unique markers across {len(markers_dict_c)} clusters")

    # Pairwise BDS
    bds_ab = compute_bds_pairwise(markers_set_a, markers_set_b)
    bds_ac = compute_bds_pairwise(markers_set_a, markers_set_c)
    bds_bc = compute_bds_pairwise(markers_set_b, markers_set_c)

    print(f"\n  Pairwise BDS (0=identical, 1=completely different):")
    print(f"    A vs B (different norm, same clust):    {bds_ab:.4f}")
    print(f"    A vs C (same norm, different clust):    {bds_ac:.4f}")
    print(f"    B vs C (different norm & clust):        {bds_bc:.4f}")

    # Detailed comparison A vs B
    print(f"\n  Detailed comparison (A vs B):")
    comparison = compare_to_reference(pipe_b, pipe_a, n_markers=50)
    print(f"    Global BDS: {comparison['global_bds']:.4f}")
    print(f"    Mean cluster BDS: {comparison['mean_cluster_bds']:.4f}")
    print(f"    Shared markers: {comparison['n_shared']} / "
          f"{comparison['n_markers_test']} (test) / {comparison['n_markers_ref']} (ref)")

    print(f"\n{'='*60}")
    print("  Published marker recovery")
    print(f"{'='*60}")

    markers_yaml = Path("config/markers.yaml")
    if markers_yaml.exists():
        with open(markers_yaml) as f:
            all_markers = yaml.safe_load(f)

        ds_markers = all_markers.get("GSE139829", {}).get("cell_types", {})
        if ds_markers:
            published = {ct: info["markers"] for ct, info in ds_markers.items()}

            for name, marker_set in [("A (log+leiden)", markers_set_a),
                                     ("B (pearson+leiden)", markers_set_b)]:
                recovery = compute_marker_recovery(marker_set, published)
                overall = recovery.get("_overall", {})
                print(f"\n  Pipeline {name}:")
                print(f"    Overall: {overall.get('n_recovered', 0)}/{overall.get('n_published', 0)} "
                      f"({overall.get('fraction', 0):.1%}) published markers recovered")
                for ct, info in recovery.items():
                    if ct.startswith("_"):
                        continue
                    print(f"    {ct}: {info['n_recovered']}/{info['n_published']} "
                          f"({info['fraction']:.0%})")
    else:
        print("  markers.yaml not found, skipping")

    print(f"\n{'='*60}")
    print("  All evaluation tests complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
