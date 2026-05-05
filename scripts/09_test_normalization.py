#!/usr/bin/env python3
"""
Test all 5 normalization methods on a subsampled dataset.

Usage:
    python scripts/09_test_normalization.py
    python scripts/09_test_normalization.py --dataset GSE139829
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import scanpy as sc

import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from normalize import METHODS, normalize


def test_method(adata_raw, method_name, **kwargs):
    """Test one normalization method and report results."""
    print(f"\n  {method_name}:", end=" ", flush=True)
    t = time.time()

    try:
        result = normalize(adata_raw, method_name, **kwargs)
        elapsed = time.time() - t

        # Validate output
        assert result.n_obs == adata_raw.n_obs, "Cell count changed"
        assert result.n_vars > 0, "No genes remaining"
        assert "normalization_method" in result.uns, "Missing method annotation"

        # Check for NaN/Inf
        from scipy.sparse import issparse
        if issparse(result.X):
            data = result.X.data
        else:
            data = result.X.flatten()
        n_nan = np.isnan(data).sum()
        n_inf = np.isinf(data).sum()

        # Check HVGs
        n_hvg = result.var["highly_variable"].sum() if "highly_variable" in result.var else 0

        print(f"PASS ({elapsed:.1f}s) — {result.n_vars} genes, {n_hvg} HVGs", end="")
        if n_nan > 0 or n_inf > 0:
            print(f" [WARNING: {n_nan} NaN, {n_inf} Inf]", end="")
        print()

        return True, elapsed

    except Exception as e:
        elapsed = time.time() - t
        print(f"FAIL ({elapsed:.1f}s) — {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="GSE139829")
    args = parser.parse_args()

    # Use a subsampled file for speed
    sub_path = Path(f"data/subsampled/{args.dataset}/random_rep0.h5ad")
    if not sub_path.exists():
        # Fall back to full processed file
        sub_path = Path(f"data/processed/{args.dataset}/adata_qc.h5ad")

    print(f"Loading test data: {sub_path}")
    adata = sc.read_h5ad(sub_path)
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    # Verify it's raw counts
    from scipy.sparse import issparse
    if issparse(adata.X):
        sample = adata.X[:10, :10].toarray()
    else:
        sample = adata.X[:10, :10]
    is_integer = np.allclose(sample, np.round(sample), atol=0.01)
    print(f"  Raw counts check: {'PASS' if is_integer else 'WARNING — not integers'}")

    print(f"\n{'='*60}")
    print(f"  Testing {len(METHODS)} normalization methods")
    print(f"{'='*60}")

    results = {}
    for method_name in METHODS:
        passed, elapsed = test_method(adata, method_name)
        results[method_name] = {"passed": passed, "time": elapsed}

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    n_pass = sum(1 for r in results.values() if r["passed"])
    n_fail = sum(1 for r in results.values() if not r["passed"])
    total_time = sum(r["time"] for r in results.values())

    for name, r in results.items():
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {name}: {r['time']:.1f}s")

    print(f"\n  {n_pass}/{len(results)} passed, total time: {total_time:.0f}s")

    if n_fail > 0:
        print("\n  Some methods failed — check errors above")
        sys.exit(1)
    else:
        print("\n  All normalization methods working!")


if __name__ == "__main__":
    main()
