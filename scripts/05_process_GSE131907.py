#!/usr/bin/env python3
"""
Process GSE131907 (Lung Adenocarcinoma) with chunked memory-efficient loading.

Reads the 29K genes x 208K cells matrix in chunks of genes, building
a sparse sub-matrix per chunk, then stacks them. This keeps peak memory
under ~4GB instead of the ~12GB+ needed to accumulate all COO data at once.

Usage:
    python scripts/05_process_GSE131907.py
"""

import gc
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import coo_matrix, csr_matrix, hstack

import yaml
import warnings
warnings.filterwarnings("ignore")

CHUNK_SIZE = 2000  # genes per chunk — keeps peak memory low


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def load_GSE131907_chunked(data_dir):
    """Load GSE131907 in gene chunks to limit peak memory."""
    data_dir = Path(data_dir)
    matrix_file = data_dir / "GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz"
    annot_file = data_dir / "GSE131907_Lung_Cancer_cell_annotation.txt.gz"

    print("  Loading GSE131907 (Lung Adenocarcinoma) — chunked mode")
    start_time = time.time()

    # Pass 1: read header and count genes
    print("    Pass 1: Reading header and counting genes...")
    with gzip.open(matrix_file, "rt") as f:
        header = f.readline().strip().split("\t")
        cell_names = header[1:]
        n_cells = len(cell_names)
        n_genes = sum(1 for _ in f)

    print(f"    Found {n_genes} genes x {n_cells} cells")
    print(f"    Will process in chunks of {CHUNK_SIZE} genes")
    n_chunks = (n_genes + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"    Total chunks: {n_chunks}")

    # Pass 2: read in chunks, build sparse sub-matrices
    print("    Pass 2: Building sparse matrix in chunks...")
    sparse_chunks = []
    gene_names = []

    with gzip.open(matrix_file, "rt") as f:
        f.readline()  # skip header

        chunk_rows = []
        chunk_cols = []
        chunk_vals = []
        chunk_gene_names = []
        gene_in_chunk = 0
        total_genes_read = 0
        total_nnz = 0

        for line in f:
            parts = line.split("\t")
            gene_name = parts[0]
            chunk_gene_names.append(gene_name)

            # Parse non-zero values for this gene
            for cell_idx in range(1, len(parts)):
                val = parts[cell_idx]
                if val != "0" and val != "0\n" and val != "0.0" and val.strip() not in ("0", ""):
                    try:
                        v = float(val)
                        if v != 0:
                            chunk_rows.append(cell_idx - 1)
                            chunk_cols.append(gene_in_chunk)
                            chunk_vals.append(v)
                    except ValueError:
                        pass

            gene_in_chunk += 1
            total_genes_read += 1

            # When chunk is full, build sparse matrix and reset
            if gene_in_chunk == CHUNK_SIZE or total_genes_read == n_genes:
                nnz = len(chunk_vals)
                total_nnz += nnz

                # Build sparse matrix for this chunk (cells x genes_in_chunk)
                chunk_sparse = coo_matrix(
                    (np.array(chunk_vals, dtype=np.float32),
                     (np.array(chunk_rows, dtype=np.int32),
                      np.array(chunk_cols, dtype=np.int32))),
                    shape=(n_cells, gene_in_chunk),
                ).tocsc()

                sparse_chunks.append(chunk_sparse)
                gene_names.extend(chunk_gene_names)

                elapsed = time.time() - start_time
                chunk_num = len(sparse_chunks)
                print(f"      Chunk {chunk_num}/{n_chunks}: "
                      f"genes {total_genes_read}/{n_genes}, "
                      f"nnz={nnz:,} ({elapsed:.0f}s)", flush=True)

                # Reset chunk accumulators
                chunk_rows = []
                chunk_cols = []
                chunk_vals = []
                chunk_gene_names = []
                gene_in_chunk = 0
                gc.collect()

    # Stack all chunks horizontally (cells x all_genes)
    print(f"    Stacking {len(sparse_chunks)} chunks...")
    X = hstack(sparse_chunks, format="csr")
    del sparse_chunks
    gc.collect()

    elapsed = time.time() - start_time
    density = 100 * total_nnz / (n_cells * n_genes)
    print(f"    Done: {X.shape[0]} cells x {X.shape[1]} genes, "
          f"{total_nnz:,} non-zeros ({density:.1f}% density)")
    print(f"    Loading time: {elapsed:.0f}s")

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=gene_names),
    )

    # Load annotations
    if annot_file.exists():
        print("    Loading cell annotations...")
        annot = pd.read_csv(annot_file, sep="\t", index_col=0)
        common = adata.obs_names.intersection(annot.index)
        print(f"    Matched {len(common)} / {adata.n_obs} cells")
        for col in annot.columns:
            adata.obs[col] = annot.reindex(adata.obs_names)[col]

    return adata


def run_qc(adata, qc_params):
    """Apply QC filters."""
    print(f"\n  Running QC on GSE131907")
    print(f"    Before QC: {adata.n_obs} cells, {adata.n_vars} genes")

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    print(f"    Mitochondrial genes: {adata.var['mt'].sum()}")

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)

    min_genes = qc_params.get("min_genes_per_cell", 200)
    max_genes = qc_params.get("max_genes_per_cell", 8000)
    max_mito = qc_params.get("max_pct_mito", 20)
    min_cells = qc_params.get("min_cells_per_gene", 3)

    n_before = adata.n_obs

    cell_mask = (
        (adata.obs["n_genes_by_counts"] >= min_genes)
        & (adata.obs["n_genes_by_counts"] <= max_genes)
        & (adata.obs["pct_counts_mt"] <= max_mito)
    )

    n_low = (adata.obs["n_genes_by_counts"] < min_genes).sum()
    n_high = (adata.obs["n_genes_by_counts"] > max_genes).sum()
    n_mito = (adata.obs["pct_counts_mt"] > max_mito).sum()

    adata = adata[cell_mask].copy()
    gc.collect()

    n_genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)

    print(f"    Removed {n_before - adata.n_obs} cells ({100*(n_before-adata.n_obs)/n_before:.1f}%)")
    print(f"      - Low genes (<{min_genes}): {n_low}")
    print(f"      - High genes (>{max_genes}): {n_high}")
    print(f"      - High mito (>{max_mito}%): {n_mito}")
    print(f"    Removed {n_genes_before - adata.n_vars} genes")
    print(f"    After QC: {adata.n_obs} cells, {adata.n_vars} genes")

    report = {
        "dataset": "GSE131907",
        "pre_qc": {"n_cells": n_before, "n_genes": int(n_genes_before)},
        "post_qc": {"n_cells": adata.n_obs, "n_genes": adata.n_vars},
        "filters": {
            "min_genes_per_cell": min_genes,
            "max_genes_per_cell": max_genes,
            "max_pct_mito": max_mito,
            "min_cells_per_gene": min_cells,
            "cells_removed_total": int(n_before - adata.n_obs),
            "genes_removed": int(n_genes_before - adata.n_vars),
        },
    }
    return adata, report


def main():
    config = load_config()
    qc_params = config["qc"]

    output_dir = Path(config["output"]["processed_data"]) / "GSE131907"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_h5ad = output_dir / "adata_qc.h5ad"
    output_report = output_dir / "qc_report.json"

    if output_h5ad.exists():
        print(f"  Already processed: {output_h5ad}")
        print(f"  Delete to reprocess.")
        return

    adata = load_GSE131907_chunked(config["datasets"]["GSE131907"]["path"])

    adata, report = run_qc(adata, qc_params)

    print(f"\n  Saving to {output_h5ad}")
    adata.write_h5ad(output_h5ad)
    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)
    (output_dir / "qc_done.flag").touch()
    print("  GSE131907 processing complete!")


if __name__ == "__main__":
    main()
