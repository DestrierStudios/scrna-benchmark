#!/usr/bin/env python3
"""Reprocess GSE176078 with correct barcodes and metadata."""
import tarfile
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.io import mmread
from scipy.sparse import issparse
import gzip
import yaml
import warnings
warnings.filterwarnings("ignore")

raw_dir = Path("data/raw/GSE176078")
output_path = Path("data/processed/GSE176078/adata_qc.h5ad")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Delete corrupted file
if output_path.exists():
    output_path.unlink()
    print("Deleted corrupted h5ad")

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)
qc = config["qc"]

tar_files = sorted(raw_dir.glob("GSM*.tar.gz"))
print(f"Found {len(tar_files)} sample archives")

adatas = []
all_meta = []

for tgz in tar_files:
    sample = tgz.stem.replace(".tar", "").split("_", 1)[-1]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(tmpdir)
        
        tmppath = Path(tmpdir)
        
        # Find files
        mtx = list(tmppath.rglob("*.mtx*"))
        barcodes = [f for f in tmppath.rglob("*") if "barcode" in f.name.lower() and f.suffix == ".tsv"]
        genes = [f for f in tmppath.rglob("*") if ("gene" in f.name.lower() or "feature" in f.name.lower()) and f.suffix == ".tsv"]
        metadata = list(tmppath.rglob("metadata.csv"))
        
        if not mtx or not barcodes or not genes:
            print(f"  WARNING: Missing files for {sample}, skipping")
            continue
        
        # Load matrix
        if str(mtx[0]).endswith(".gz"):
            with gzip.open(mtx[0], "rb") as f:
                mat = mmread(f).T.tocsr()
        else:
            mat = mmread(str(mtx[0])).T.tocsr()
        
        # Load barcodes — they already contain sample prefix
        with open(barcodes[0]) as f:
            bc_list = [line.strip().split("\t")[0].replace("-1", "") for line in f]
        
        # Load genes
        with open(genes[0]) as f:
            lines = [line.strip().split("\t") for line in f]
        gene_ids = [l[0] for l in lines]
        gene_names = [l[1] if len(l) > 1 else l[0] for l in lines]
        
        a = ad.AnnData(
            X=mat,
            obs=pd.DataFrame(index=bc_list),
            var=pd.DataFrame({"gene_ids": gene_ids}, index=gene_names),
        )
        a.var_names_make_unique()
        a.obs["sample"] = sample
        
        # Load metadata if present
        if metadata:
            meta = pd.read_csv(metadata[0])
            col0 = meta.columns[0]
            if col0 == "Unnamed: 0":
                meta = meta.rename(columns={"Unnamed: 0": "barcode"})
            else:
                meta = meta.rename(columns={col0: "barcode"})
            meta = meta.set_index("barcode")
            
            for col in ["celltype_major", "celltype_minor", "celltype_subset", "subtype"]:
                if col in meta.columns:
                    a.obs[col] = meta.reindex(a.obs_names)[col].values
        
        print(f"  {sample}: {a.n_obs} cells", end="")
        if "celltype_major" in a.obs.columns:
            n_ann = a.obs["celltype_major"].notna().sum()
            print(f", {n_ann} annotated", end="")
        print()
        adatas.append(a)

# Concatenate
print(f"\nConcatenating {len(adatas)} samples...")
adata = ad.concat(adatas, join="outer")
adata.obs_names_make_unique()
print(f"Total: {adata.n_obs} cells, {adata.n_vars} genes")

# QC
print("\nRunning QC...")
adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)

n_before = adata.n_obs
cell_mask = (
    (adata.obs["n_genes_by_counts"] >= qc["min_genes_per_cell"])
    & (adata.obs["n_genes_by_counts"] <= qc["max_genes_per_cell"])
    & (adata.obs["pct_counts_mt"] <= qc["max_pct_mito"])
)
adata = adata[cell_mask].copy()
sc.pp.filter_genes(adata, min_cells=qc["min_cells_per_gene"])
print(f"QC: {n_before} -> {adata.n_obs} cells, {adata.n_vars} genes")

# Verify annotations
for col in ["celltype_major", "celltype_minor", "celltype_subset", "subtype"]:
    if col in adata.obs.columns:
        n = adata.obs[col].notna().sum()
        u = adata.obs[col].nunique()
        print(f"  {col}: {u} types, {n} annotated")

# Drop any underscore-prefixed columns
for col in list(adata.obs.columns):
    if col.startswith("_"):
        adata.obs = adata.obs.drop(columns=[col])

# Save
adata.write_h5ad(output_path)
Path("data/processed/GSE176078/qc_done.flag").touch()
print(f"\nSaved to {output_path}")
print("Done!")
