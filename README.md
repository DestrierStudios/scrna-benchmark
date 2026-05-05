# scRNA-seq Benchmarking: Normalization Choice Drives Biological Interpretation

Systematic benchmarking of 465 computational pipelines across 3 cancer scRNA-seq datasets, introducing the Biological Discordance Score (BDS) to quantify marker gene interpretation stability.

## Citation

Saxena, N. (2026). Normalization choice drives biological interpretation in single-cell RNA-seq cancer studies: A systematic benchmarking of 465 computational pipelines. *Computational Biology and Chemistry*, 124, 109100. https://doi.org/10.1016/j.compbiolchem.2026.109100

## Key Findings

- **Normalization matters more than clustering** for biological interpretation: pipelines differing only in normalization produce up to 86% different marker genes (BDS = 0.86), while clustering differences yield only 29% discordance.
- **Log-normalization** consistently achieves the best balance of clustering performance and marker gene stability across all three cancer types.
- **Traditional metrics are insufficient**: pipelines with similar ARI can show BDS from 0.00 to 0.91, meaning high clustering agreement does not guarantee consistent biological conclusions.
- **Hierarchical clustering** outperforms graph-based methods (Leiden, Louvain) by mean rank across all datasets.

## Datasets

| Dataset | Cancer Type | Cells (post-QC) | Platform |
|---------|------------|-----------------|----------|
| GSE139829 | Uveal melanoma | 125,788 | 10x Chromium |
| GSE176078 | Breast cancer | 99,983 | 10x Chromium |
| GSE131907 | Lung adenocarcinoma | 208,298 | 10x Chromium |
| GSE72056 | Cutaneous melanoma | 4,645 | Smart-seq2 |

## Reproducibility

All analyses run inside a Docker container. To reproduce:

```bash
docker build -t scrna-benchmark .
docker run -it --rm -v $(pwd):/workspace scrna-benchmark
```

Then execute the pipeline scripts in order (`scripts/01_*.py` through `scripts/17_*.py`).

## Pipeline

1. **QC and preprocessing** (scripts 01-03)
2. **Subsampling** with random, Geosketch, and stratified strategies (scripts 04-05)
3. **Normalization** with 5 methods: log, scran, SCTransform, Pearson residuals, analytic Pearson (scripts 06-08)
4. **Clustering** with 4 algorithms x multiple parameters = 31 configurations (scripts 09-10)
5. **Evaluation** with 8 metrics + BDS (scripts 11-13)
6. **Meta-ranking** with Borda count and bootstrap CIs (script 14)
7. **Figure generation** and cross-platform validation (scripts 15-17)

## License

CC BY-NC 4.0
