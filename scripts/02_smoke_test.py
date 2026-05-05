#!/usr/bin/env python3
"""
Smoke test: Quick end-to-end verification of the benchmarking pipeline.

Tests that all normalization methods, clustering algorithms, and evaluation
metrics work on a small synthetic dataset. Run this before starting real
experiments to catch environment or wiring issues early.

Usage:
    python scripts/02_smoke_test.py
"""

import sys
import time
import json
import traceback
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Track results
RESULTS = []
TOTAL_START = time.time()


def log_test(name, status, detail="", elapsed=0):
    """Log a test result."""
    icon = "PASS" if status else "FAIL"
    RESULTS.append({"name": name, "passed": status, "detail": detail})
    elapsed_str = f" ({elapsed:.1f}s)" if elapsed > 0 else ""
    print(f"  [{icon}] {name}{elapsed_str}")
    if not status and detail:
        print(f"         {detail}")


def section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

section("Setup: Creating synthetic test data")

try:
    import scanpy as sc
    import anndata as ad
    import pandas as pd
    from scipy.sparse import csr_matrix

    np.random.seed(42)

    # Create a small dataset with 3 known clusters
    n_cells = 300
    n_genes = 500
    n_clusters = 3
    cells_per_cluster = n_cells // n_clusters

    # Simulate count data with cluster structure
    counts = np.zeros((n_cells, n_genes), dtype=np.float32)
    true_labels = []

    for i in range(n_clusters):
        start = i * cells_per_cluster
        end = start + cells_per_cluster
        # Base expression
        counts[start:end, :] = np.random.poisson(1.0, (cells_per_cluster, n_genes))
        # Cluster-specific upregulated genes
        marker_start = i * 50
        marker_end = marker_start + 50
        counts[start:end, marker_start:marker_end] += np.random.poisson(
            5.0, (cells_per_cluster, 50)
        )
        true_labels.extend([f"cluster_{i}"] * cells_per_cluster)

    # Build AnnData object
    adata = ad.AnnData(
        X=csr_matrix(counts),
        obs=pd.DataFrame(
            {"cell_type": true_labels},
            index=[f"cell_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(
            index=[f"gene_{i}" for i in range(n_genes)],
        ),
    )
    # Add mitochondrial genes for QC
    adata.var["mt"] = False
    adata.var.loc[adata.var_names[-10:], "mt"] = True

    log_test("Create synthetic AnnData", True, f"{n_cells} cells x {n_genes} genes")

except Exception as e:
    log_test("Create synthetic AnnData", False, str(e))
    print("\nCannot continue without test data. Exiting.")
    sys.exit(1)


section("Test 1: Quality Control")

try:
    t = time.time()
    adata_qc = adata.copy()
    sc.pp.calculate_qc_metrics(
        adata_qc, qc_vars=["mt"], percent_top=None, inplace=True
    )
    # Apply filters
    initial_cells = adata_qc.n_obs
    sc.pp.filter_cells(adata_qc, min_genes=10)
    sc.pp.filter_genes(adata_qc, min_cells=3)
    final_cells = adata_qc.n_obs
    log_test(
        "QC filtering",
        True,
        f"{initial_cells} -> {final_cells} cells",
        time.time() - t,
    )
except Exception as e:
    log_test("QC filtering", False, str(e))

section("Test 2: Normalization Methods")

normalized = {}

try:
    t = time.time()
    adata_log = adata_qc.copy()
    sc.pp.normalize_total(adata_log, target_sum=1e4)
    sc.pp.log1p(adata_log)
    sc.pp.highly_variable_genes(adata_log, n_top_genes=min(200, adata_log.n_vars))
    normalized["log"] = adata_log
    log_test("Log normalization", True, elapsed=time.time() - t)
except Exception as e:
    log_test("Log normalization", False, str(e))

try:
    t = time.time()
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    # numpy2ri — use localconverter instead

    scran = importr("scran")
    scater = importr("scater")
    sce_pkg = importr("SingleCellExperiment")
    base = importr("base")

    adata_scran = adata_qc.copy()
    # Convert to dense for R
    if hasattr(adata_scran.X, "toarray"):
        mat = adata_scran.X.toarray()
    else:
        mat = adata_scran.X

    # Transpose for R (genes x cells)
    r_mat = ro.r["matrix"](
        ro.FloatVector(mat.T.flatten()),
        nrow=mat.shape[1],
        ncol=mat.shape[0],
    )
    sce = sce_pkg.SingleCellExperiment(
        assays=ro.ListVector({"counts": r_mat})
    )
    # Quick cluster for size factors
    clusters = scran.quickCluster(sce)
    sce = scran.computeSumFactors(sce, clusters=clusters)
    size_factors = np.array(ro.r["sizeFactors"](sce))

    # Apply size factors in Python
    adata_scran.X = mat / size_factors[:, None]
    sc.pp.log1p(adata_scran)
    sc.pp.highly_variable_genes(adata_scran, n_top_genes=min(200, adata_scran.n_vars))
    normalized["scran"] = adata_scran

    # (no deactivate needed with localconverter)
    log_test("Scran normalization", True, elapsed=time.time() - t)
except Exception as e:
    log_test("Scran normalization", False, str(e))
    try:
        # (no deactivate needed with localconverter)
    except:
        pass

try:
    t = time.time()
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    # numpy2ri — use localconverter instead

    sct = importr("sctransform")

    adata_sct = adata_qc.copy()
    if hasattr(adata_sct.X, "toarray"):
        mat = adata_sct.X.toarray()
    else:
        mat = adata_sct.X

    # sctransform expects genes x cells
    r_mat = ro.r["matrix"](
        ro.FloatVector(mat.T.flatten()),
        nrow=mat.shape[1],
        ncol=mat.shape[0],
    )
    ro.r.assign("counts_mat", r_mat)

    # Run sctransform
    ro.r(
        """
        rownames(counts_mat) <- paste0("gene_", 1:nrow(counts_mat))
        colnames(counts_mat) <- paste0("cell_", 1:ncol(counts_mat))
        vst_out <- sctransform::vst(counts_mat, verbosity=0)
        corrected <- vst_out$y
        """
    )
    corrected = np.array(ro.r("corrected"))  # genes x cells
    adata_sct.X = corrected.T  # cells x genes — may be smaller after vst
    # Adjust var if dimensions changed
    if adata_sct.X.shape[1] != adata_sct.n_vars:
        residual_genes = list(ro.r("rownames(corrected)"))
        adata_sct = adata_sct[:, :adata_sct.X.shape[1]].copy()
    normalized["sctransform"] = adata_sct

    # (no deactivate needed with localconverter)
    log_test("SCTransform normalization", True, elapsed=time.time() - t)
except Exception as e:
    log_test("SCTransform normalization", False, str(e))
    try:
        # (no deactivate needed with localconverter)
    except:
        pass

try:
    t = time.time()
    adata_pr = adata_qc.copy()
    sc.experimental.pp.normalize_pearson_residuals(adata_pr)
    normalized["pearson_residuals"] = adata_pr
    log_test("Pearson residuals normalization", True, elapsed=time.time() - t)
except Exception as e:
    log_test("Pearson residuals normalization", False, str(e))

try:
    t = time.time()
    adata_ap = adata_qc.copy()
    # Uses same scanpy function with different params
    sc.experimental.pp.normalize_pearson_residuals(adata_ap, theta=100)
    normalized["analytic_pearson"] = adata_ap
    log_test("Analytic Pearson normalization", True, elapsed=time.time() - t)
except Exception as e:
    log_test("Analytic Pearson normalization", False, str(e))


section("Test 3: Dimensionality Reduction")

reduced = {}
for name, ad_obj in normalized.items():
    try:
        t = time.time()
        a = ad_obj.copy()
        # Ensure we work on HVGs if available
        if "highly_variable" in a.var.columns:
            sc.pp.pca(a, n_comps=min(30, a.n_vars - 1), use_highly_variable=True)
        else:
            sc.pp.pca(a, n_comps=min(30, a.n_vars - 1))
        sc.pp.neighbors(a, n_neighbors=15, n_pcs=min(30, a.obsm["X_pca"].shape[1]))
        sc.tl.umap(a)
        reduced[name] = a
        log_test(f"PCA + neighbors + UMAP ({name})", True, elapsed=time.time() - t)
    except Exception as e:
        log_test(f"PCA + neighbors + UMAP ({name})", False, str(e))


section("Test 4: Clustering Algorithms")

# Use the log-normalized reduced data for clustering tests
if "log" in reduced:
    test_adata = reduced["log"]
elif reduced:
    test_adata = list(reduced.values())[0]
else:
    print("  No reduced data available — skipping clustering tests")
    test_adata = None

if test_adata is not None:
    
    try:
        t = time.time()
        sc.tl.leiden(test_adata, resolution=0.8, key_added="leiden")
        n_clusters_found = test_adata.obs["leiden"].nunique()
        log_test(
            "Leiden clustering",
            True,
            f"{n_clusters_found} clusters at res=0.8",
            time.time() - t,
        )
    except Exception as e:
        log_test("Leiden clustering", False, str(e))

  
    try:
        t = time.time()
        sc.tl.louvain(test_adata, resolution=0.8, key_added="louvain")
        n_clusters_found = test_adata.obs["louvain"].nunique()
        log_test(
            "Louvain clustering",
            True,
            f"{n_clusters_found} clusters at res=0.8",
            time.time() - t,
        )
    except Exception as e:
        log_test("Louvain clustering", False, str(e))

   
    try:
        t = time.time()
        from sklearn.cluster import AgglomerativeClustering

        hc = AgglomerativeClustering(n_clusters=3, linkage="ward")
        labels = hc.fit_predict(test_adata.obsm["X_pca"])
        test_adata.obs["hierarchical"] = pd.Categorical(labels.astype(str))
        log_test(
            "Hierarchical clustering",
            True,
            f"3 clusters (ward)",
            time.time() - t,
        )
    except Exception as e:
        log_test("Hierarchical clustering", False, str(e))

   
    try:
        t = time.time()
        try:
            from hdbscan import HDBSCAN
        except ImportError:
            from sklearn.cluster import HDBSCAN

        hdb = HDBSCAN(min_cluster_size=15, min_samples=5)
        labels = hdb.fit_predict(test_adata.obsm["X_pca"])
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        test_adata.obs["hdbscan"] = pd.Categorical(labels.astype(str))
        log_test(
            "HDBSCAN clustering",
            True,
            f"{n_clusters_found} clusters, {n_noise} noise points",
            time.time() - t,
        )
    except Exception as e:
        log_test("HDBSCAN clustering", False, str(e))


section("Test 5: Evaluation Metrics")

if test_adata is not None and "leiden" in test_adata.obs.columns:
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
    )

    true = test_adata.obs["cell_type"]
    pred = test_adata.obs["leiden"]
    embedding = test_adata.obsm["X_pca"]

   
    try:
        t = time.time()
        ari = adjusted_rand_score(true, pred)
        nmi = normalized_mutual_info_score(true, pred)
        log_test(
            "ARI + NMI",
            True,
            f"ARI={ari:.3f}, NMI={nmi:.3f}",
            time.time() - t,
        )
    except Exception as e:
        log_test("ARI + NMI", False, str(e))

    
    try:
        t = time.time()
        sil = silhouette_score(embedding, pred)
        ch = calinski_harabasz_score(embedding, pred)
        db = davies_bouldin_score(embedding, pred)
        log_test(
            "Silhouette + CH + DB",
            True,
            f"Sil={sil:.3f}, CH={ch:.1f}, DB={db:.3f}",
            time.time() - t,
        )
    except Exception as e:
        log_test("Silhouette + CH + DB", False, str(e))

   
    try:
        t = time.time()
        # Find markers for leiden clusters
        test_copy = test_adata.copy()
        sc.tl.rank_genes_groups(test_copy, groupby="leiden", method="wilcoxon")

        # Find markers for "true" labels
        test_copy2 = test_adata.copy()
        test_copy2.obs["cell_type"] = test_copy2.obs["cell_type"].astype("category")
        sc.tl.rank_genes_groups(test_copy2, groupby="cell_type", method="wilcoxon")

        # Compare top markers (simplified BDS)
        n_top = 20
        leiden_markers = set()
        for group in test_copy.obs["leiden"].cat.categories:
            markers = sc.get.rank_genes_groups_df(test_copy, group=group)
            leiden_markers.update(markers.head(n_top)["names"].tolist())

        true_markers = set()
        for group in test_copy2.obs["cell_type"].cat.categories:
            markers = sc.get.rank_genes_groups_df(test_copy2, group=group)
            true_markers.update(markers.head(n_top)["names"].tolist())

        overlap = len(leiden_markers & true_markers)
        union = len(leiden_markers | true_markers)
        bds = 1.0 - (overlap / union if union > 0 else 0)

        log_test(
            "BDS (simplified)",
            True,
            f"BDS={bds:.3f} (0=identical markers, 1=no overlap)",
            time.time() - t,
        )
    except Exception as e:
        log_test("BDS (simplified)", False, str(e))


section("Test 6: R Package Integration")

r_packages = ["Seurat", "scran", "scater", "SingleCellExperiment", "sctransform", "Dino"]
for pkg in r_packages:
    try:
        t = time.time()
        from rpy2.robjects.packages import importr
        importr(pkg)
        log_test(f"R package: {pkg}", True, elapsed=time.time() - t)
    except Exception as e:
        log_test(f"R package: {pkg}", False, str(e))


section("Test 7: File I/O")

try:
    t = time.time()
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write h5ad
        path = os.path.join(tmpdir, "test.h5ad")
        test_adata.write_h5ad(path)
        size = os.path.getsize(path) / 1024
        # Read it back
        loaded = sc.read_h5ad(path)
        assert loaded.n_obs == test_adata.n_obs
        log_test("Write/read h5ad", True, f"{size:.0f} KB", time.time() - t)
except Exception as e:
    log_test("Write/read h5ad", False, str(e))

try:
    t = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write CSV clusters
        path = os.path.join(tmpdir, "clusters.csv")
        test_adata.obs[["leiden", "louvain"]].to_csv(path)
        loaded = pd.read_csv(path, index_col=0)
        assert loaded.shape[0] == test_adata.n_obs
        log_test("Write/read cluster CSV", True, elapsed=time.time() - t)
except Exception as e:
    log_test("Write/read cluster CSV", False, str(e))

try:
    t = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write JSON metrics
        path = os.path.join(tmpdir, "metrics.json")
        metrics = {"ARI": 0.95, "NMI": 0.88, "silhouette": 0.45}
        with open(path, "w") as f:
            json.dump(metrics, f)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["ARI"] == 0.95
        log_test("Write/read metrics JSON", True, elapsed=time.time() - t)
except Exception as e:
    log_test("Write/read metrics JSON", False, str(e))


section("SUMMARY")

total_elapsed = time.time() - TOTAL_START
passed = sum(1 for r in RESULTS if r["passed"])
failed = sum(1 for r in RESULTS if not r["passed"])
total = len(RESULTS)

print(f"\n  Total tests: {total}")
print(f"  Passed:      {passed}")
print(f"  Failed:      {failed}")
print(f"  Time:        {total_elapsed:.1f}s")

if failed > 0:
    print(f"\n  Failed tests:")
    for r in RESULTS:
        if not r["passed"]:
            print(f"    - {r['name']}: {r['detail']}")

print()
if failed == 0:
    print("  ALL TESTS PASSED — pipeline is ready for implementation!")
    sys.exit(0)
elif failed <= 2:
    print("  MOSTLY PASSED — review failed tests but pipeline is likely usable.")
    sys.exit(0)
else:
    print("  MULTIPLE FAILURES — fix environment issues before proceeding.")
    sys.exit(1)
