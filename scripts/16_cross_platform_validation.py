#!/usr/bin/env python3
"""
Cross-platform validation: GSE72056 (Smart-seq2) vs 10x datasets.

GSE72056 is already log2(TPM+1) normalized, so we can't benchmark
normalization methods. Instead we:
1. Cluster the Smart-seq2 data with multiple algorithms
2. Extract marker genes from each clustering
3. Compare marker overlap with markers from the 10x datasets
4. Test whether top-ranked 10x pipelines identify markers that
   transfer across platforms

This validates that our pipeline recommendations generalize beyond 10x.

Usage:
    python scripts/16_cross_platform_validation.py
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
from cluster import cluster, METHODS as CLUST_METHODS
from pipeline import run_dim_reduction
from evaluate import evaluate_clustering
from bds import extract_markers, compute_bds_pairwise, compute_marker_recovery


def load_markers_yaml():
    with open("config/markers.yaml") as f:
        return yaml.safe_load(f)


def decode_gse72056_labels(adata):
    """
    Decode numeric cell type codes from Tirosh et al. 2016.
    malignant: 0=unresolved, 1=no, 2=yes
    non-malignant_cell_type: 0=unresolved, 1=T, 2=B, 3=Macro, 4=Endo, 5=CAF, 6=NK
    """
    label_map = {
        0.0: "Unresolved",
        1.0: "T cells",
        2.0: "B cells",
        3.0: "Macrophages",
        4.0: "Endothelial",
        5.0: "CAFs",
        6.0: "NK cells",
    }

    cell_types = []
    for _, row in adata.obs.iterrows():
        if row.get("malignant", 0) == 2.0:
            cell_types.append("Malignant")
        elif row.get("non-malignant_cell_type", 0) in label_map:
            cell_types.append(label_map[row["non-malignant_cell_type"]])
        else:
            cell_types.append("Unresolved")

    adata.obs["celltype_major"] = pd.Categorical(cell_types)
    return adata


def run_smartseq2_clustering(adata):
    """Run multiple clustering configs on pre-normalized Smart-seq2 data."""
    print(f"\n  Dimensionality reduction...")
    adata_red = adata.copy()

    # HVG selection (data is already normalized)
    sc.pp.highly_variable_genes(adata_red, n_top_genes=min(3000, adata_red.n_vars),
                                 flavor="seurat")
    # PCA + neighbors
    adata_red = run_dim_reduction(adata_red, n_pcs=50, n_neighbors=15, seed=42)

    configs = [
        ("leiden", "res0.4"),
        ("leiden", "res0.8"),
        ("leiden", "res1.2"),
        ("louvain", "res0.8"),
        ("hierarchical", "k7"),  # 7 known cell types
        ("hierarchical", "k10"),
        ("hierarchical", "k15"),
        ("hdbscan", "mcs50_ms10"),
    ]

    results = {}
    for method, param in configs:
        name = f"{method}_{param}"
        print(f"    {name}...", end=" ", flush=True)
        t = time.time()
        try:
            result = cluster(adata_red, method, param_str=param, seed=42)
            n_clust = result.obs["clusters"].nunique()

            # Evaluate
            metrics = evaluate_clustering(result, true_label_col="celltype_major")

            # Extract markers
            _, marker_set = extract_markers(result, n_markers=50)

            results[name] = {
                "adata": result,
                "metrics": metrics,
                "markers": marker_set,
                "n_clusters": n_clust,
            }
            print(f"{n_clust} clusters, ARI={metrics.get('ARI', 0):.3f} ({time.time()-t:.1f}s)")
        except Exception as e:
            print(f"FAILED — {e}")

    return results, adata_red


def cross_platform_marker_comparison(smartseq_results, tenx_datasets):
    """Compare markers between Smart-seq2 and 10x datasets."""
    print(f"\n{'='*60}")
    print(f"  Cross-Platform Marker Comparison")
    print(f"{'='*60}")

    # For each 10x dataset, load the cached reference pipeline markers
    tenx_markers = {}
    for ds in tenx_datasets:
        cache_file = Path(f"results/{ds}/cache/log_reduced.h5ad")
        if not cache_file.exists():
            continue

        try:
            adata = sc.read_h5ad(cache_file)
            adata = cluster(adata, "leiden", param_str="res0.8", seed=42)
            _, markers = extract_markers(adata, n_markers=50)
            tenx_markers[ds] = markers
            print(f"\n  10x {ds} (log+leiden+res0.8): {len(markers)} markers")
            del adata
        except Exception as e:
            print(f"  10x {ds}: FAILED — {e}")

    if not tenx_markers:
        print("  No 10x markers available, skipping comparison")
        return {}

    # Compare each Smart-seq2 clustering with each 10x dataset
    comparison_results = {}
    print(f"\n  Smart-seq2 vs 10x marker overlap:")
    print(f"  {'Smart-seq2 pipeline':<30} {'10x dataset':<15} {'Shared':>7} {'Jaccard':>8}")
    print(f"  {'-'*60}")

    for ss_name, ss_data in smartseq_results.items():
        ss_markers = ss_data["markers"]
        if not ss_markers:
            continue

        for ds, tx_markers in tenx_markers.items():
            shared = len(ss_markers & tx_markers)
            union = len(ss_markers | tx_markers)
            jaccard = shared / union if union > 0 else 0
            bds = 1.0 - jaccard

            key = f"{ss_name}_vs_{ds}"
            comparison_results[key] = {
                "smartseq_pipeline": ss_name,
                "tenx_dataset": ds,
                "smartseq_markers": len(ss_markers),
                "tenx_markers": len(tx_markers),
                "shared_markers": shared,
                "jaccard": jaccard,
                "bds": bds,
            }
            ds_short = ds.replace("GSE", "")
            print(f"  {ss_name:<30} {ds_short:<15} {shared:>7} {jaccard:>8.3f}")

    return comparison_results


def published_marker_validation(smartseq_results, markers_yaml):
    """Check recovery of published melanoma markers across platforms."""
    print(f"\n{'='*60}")
    print(f"  Published Marker Recovery (GSE72056)")
    print(f"{'='*60}")

    gse72056_markers = markers_yaml.get("GSE72056", {}).get("cell_types", {})
    if not gse72056_markers:
        print("  No published markers for GSE72056 in markers.yaml")
        return

    published = {ct: info["markers"] for ct, info in gse72056_markers.items()}

    print(f"\n  {'Pipeline':<30} {'Overall':>8} ", end="")
    for ct in list(published.keys())[:5]:
        print(f"{ct[:10]:>10} ", end="")
    print()
    print(f"  {'-'*80}")

    for name, data in smartseq_results.items():
        marker_set = data["markers"]
        if not marker_set:
            continue

        recovery = compute_marker_recovery(marker_set, published)
        overall = recovery.get("_overall", {})
        frac = overall.get("fraction", 0)
        print(f"  {name:<30} {frac:>7.0%} ", end="")

        for ct in list(published.keys())[:5]:
            ct_info = recovery.get(ct, {})
            ct_frac = ct_info.get("fraction", 0)
            print(f"{ct_frac:>9.0%} ", end="")
        print()


def main():
    print("Cross-Platform Validation: GSE72056 (Smart-seq2)")

    # Load Smart-seq2 data
    h5ad_path = Path("data/processed/GSE72056/adata_qc.h5ad")
    if not h5ad_path.exists():
        print(f"ERROR: {h5ad_path} not found")
        sys.exit(1)

    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"Obs columns: {list(adata.obs.columns)}")

    # Decode cell type labels
    adata = decode_gse72056_labels(adata)
    print(f"Cell types: {adata.obs['celltype_major'].value_counts().to_dict()}")

    # Run clustering
    print(f"\n{'='*60}")
    print(f"  Clustering Smart-seq2 Data")
    print(f"{'='*60}")
    ss_results, adata_red = run_smartseq2_clustering(adata)

    # Load markers.yaml
    markers_yaml = load_markers_yaml()

    # Published marker recovery
    published_marker_validation(ss_results, markers_yaml)

    # Cross-platform comparison
    tenx_datasets = ["GSE139829", "GSE176078", "GSE131907"]
    comparison = cross_platform_marker_comparison(ss_results, tenx_datasets)

    # Save results
    output_dir = Path("results/GSE72056")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save clustering metrics
    metrics_rows = []
    for name, data in ss_results.items():
        row = {"pipeline": name, **data["metrics"]}
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(output_dir / "clustering_metrics.csv", index=False)

    # Save cross-platform comparison
    if comparison:
        pd.DataFrame(comparison).T.to_csv(output_dir / "cross_platform_comparison.csv")

    # Summary for paper
    print(f"\n{'='*60}")
    print(f"  Summary for Manuscript")
    print(f"{'='*60}")

    if ss_results:
        best = max(ss_results.items(), key=lambda x: x[1]["metrics"].get("ARI", 0))
        print(f"  Best Smart-seq2 pipeline: {best[0]}")
        print(f"    ARI={best[1]['metrics'].get('ARI', 0):.3f}, "
              f"NMI={best[1]['metrics'].get('NMI', 0):.3f}, "
              f"{best[1]['n_clusters']} clusters")

    if comparison:
        jaccards = [v["jaccard"] for v in comparison.values()]
        print(f"\n  Cross-platform marker overlap (Jaccard):")
        print(f"    Mean: {np.mean(jaccards):.3f}")
        print(f"    Range: {np.min(jaccards):.3f} — {np.max(jaccards):.3f}")
        print(f"    This represents the baseline marker overlap expected")
        print(f"    between platforms (10x vs Smart-seq2)")

    print(f"\n  Results saved to {output_dir}")
    print(f"\n  Cross-platform validation complete!")


if __name__ == "__main__":
    main()
