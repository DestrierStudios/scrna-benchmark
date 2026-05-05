#!/usr/bin/env python3
"""
Generate publication-quality figures for the manuscript.

Figures:
  1. Overview heatmap: metrics across all pipelines (per dataset)
  2. Normalization comparison: boxplots of key metrics by normalization method
  3. BDS analysis: BDS vs traditional metrics scatter
  4. Meta-ranking: forest plot with bootstrap CIs
  5. Cross-dataset consistency: rank correlation between datasets
  6. Clustering algorithm comparison: metrics by clustering method

Usage:
    python scripts/15_generate_figures.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAMES = {
    "GSE139829": "Uveal Melanoma",
    "GSE176078": "Breast Cancer",
    "GSE131907": "Lung Adenocarcinoma",
}

NORM_COLORS = {
    "log": "#2196F3",
    "scran": "#4CAF50",
    "sctransform": "#FF9800",
    "pearson_residuals": "#E91E63",
    "analytic_pearson": "#9C27B0",
}

CLUST_COLORS = {
    "leiden": "#1976D2",
    "louvain": "#388E3C",
    "hierarchical": "#F57C00",
    "hdbscan": "#7B1FA2",
}


def load_all_results():
    """Load evaluation matrices for all datasets."""
    results = {}
    for ds in DATASET_NAMES:
        path = Path(f"results/{ds}/evaluation_matrix.csv")
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            # Parse normalization and clustering from pipeline name
            if "pipeline" in df.columns:
                df["norm"] = df["pipeline"].apply(lambda x: _parse_norm(x))
                df["clust"] = df["pipeline"].apply(lambda x: _parse_clust(x))
            results[ds] = df
    return results


def load_rankings():
    """Load meta-rankings for all datasets."""
    rankings = {}
    for ds in DATASET_NAMES:
        path = Path(f"results/{ds}/meta_ranking.csv")
        if path.exists():
            rankings[ds] = pd.read_csv(path)
    return rankings


def load_cross_dataset():
    """Load cross-dataset ranking."""
    path = Path("results/tables/cross_dataset_ranking.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


def _parse_norm(pipeline_name):
    for n in ["analytic_pearson", "pearson_residuals", "sctransform", "scran", "log"]:
        if str(pipeline_name).startswith(n):
            return n
    return "unknown"


def _parse_clust(pipeline_name):
    for c in ["leiden", "louvain", "hierarchical", "hdbscan"]:
        if f"_{c}_" in str(pipeline_name):
            return c
    return "unknown"



def figure1_heatmap(results):
    """Heatmap of key metrics across top pipelines per dataset."""
    metrics = ["ARI", "NMI", "silhouette", "cell_type_purity", "homogeneity"]

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 8))
    if len(results) == 1:
        axes = [axes]

    for ax, (ds, df) in zip(axes, results.items()):
        available = [m for m in metrics if m in df.columns]
        if not available or "pipeline" not in df.columns:
            continue

        # Top 25 pipelines by ARI
        if "ARI" in df.columns:
            df_sorted = df.nlargest(25, "ARI")
        else:
            df_sorted = df.head(25)

        plot_data = df_sorted.set_index("pipeline")[available].astype(float)
        plot_data.index = [p[:35] for p in plot_data.index]  # truncate names

        sns.heatmap(plot_data, ax=ax, cmap="RdYlBu_r", annot=True, fmt=".2f",
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title(DATASET_NAMES.get(ds, ds), fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Figure 1: Evaluation Metrics Across Top Pipelines", 
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure1_heatmap.pdf")
    fig.savefig(FIGURES_DIR / "figure1_heatmap.png")
    plt.close()
    print("  Figure 1: Heatmap saved")



def figure2_normalization(results):
    """Boxplots comparing normalization methods across metrics."""
    metrics = ["ARI", "NMI", "silhouette", "cell_type_purity"]

    fig, axes = plt.subplots(len(metrics), len(results), 
                             figsize=(5 * len(results), 3.5 * len(metrics)))
    if len(results) == 1:
        axes = axes.reshape(-1, 1)

    for col, (ds, df) in enumerate(results.items()):
        if "norm" not in df.columns:
            continue
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            if metric not in df.columns:
                ax.set_visible(False)
                continue

            plot_df = df[["norm", metric]].dropna()
            plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")

            order = ["log", "scran", "sctransform", "pearson_residuals", "analytic_pearson"]
            colors = [NORM_COLORS.get(n, "gray") for n in order]

            sns.boxplot(data=plot_df, x="norm", y=metric, ax=ax, order=order,
                       palette=colors, width=0.6, fliersize=3)
            ax.set_xlabel("")
            if col == 0:
                ax.set_ylabel(metric, fontweight="bold")
            else:
                ax.set_ylabel("")
            if row == 0:
                ax.set_title(DATASET_NAMES.get(ds, ds), fontweight="bold")
            ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Figure 2: Normalization Method Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure2_normalization.pdf")
    fig.savefig(FIGURES_DIR / "figure2_normalization.png")
    plt.close()
    print("  Figure 2: Normalization comparison saved")


def figure3_bds(rankings):
    """BDS vs traditional metrics — the key novel finding."""
    fig, axes = plt.subplots(1, len(rankings), figsize=(6 * len(rankings), 5))
    if len(rankings) == 1:
        axes = [axes]

    for ax, (ds, df) in zip(axes, rankings.items()):
        if "global_bds" not in df.columns or "borda_score" not in df.columns:
            ax.set_visible(False)
            continue

        plot_df = df[["pipeline", "borda_score", "global_bds", "rank"]].dropna(subset=["global_bds"])
        plot_df["norm"] = plot_df["pipeline"].apply(_parse_norm)

        for norm, color in NORM_COLORS.items():
            mask = plot_df["norm"] == norm
            if mask.any():
                ax.scatter(plot_df.loc[mask, "borda_score"],
                          plot_df.loc[mask, "global_bds"],
                          c=color, label=norm, s=60, alpha=0.8, edgecolors="white")

        ax.set_xlabel("Borda Score (higher = better overall)")
        ax.set_ylabel("BDS (lower = more consistent markers)")
        ax.set_title(DATASET_NAMES.get(ds, ds), fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")

        # Highlight ideal zone (high borda, low BDS)
        ax.axhline(y=0.3, color="gray", linestyle="--", alpha=0.3)
        ax.text(ax.get_xlim()[1] * 0.95, 0.25, "Low discordance zone",
                ha="right", fontsize=8, color="gray", alpha=0.5)

    plt.suptitle("Figure 3: Biological Discordance Score vs Overall Performance",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure3_bds.pdf")
    fig.savefig(FIGURES_DIR / "figure3_bds.png")
    plt.close()
    print("  Figure 3: BDS analysis saved")



def figure4_forest(cross_ranking):
    """Forest plot of cross-dataset ranking with CIs."""
    if cross_ranking is None:
        print("  Figure 4: Skipped (no cross-dataset ranking)")
        return

    top = cross_ranking.head(20).copy()
    top = top.sort_values("mean_rank_across_datasets", ascending=False)
    top["norm"] = top["pipeline"].apply(_parse_norm)

    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = range(len(top))
    colors = [NORM_COLORS.get(n, "gray") for n in top["norm"]]

    ax.barh(y_pos, top["mean_rank_across_datasets"].max() - top["mean_rank_across_datasets"] + 1,
            color=colors, alpha=0.7, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([p[:40] for p in top["pipeline"]], fontsize=9)
    ax.set_xlabel("Inverse Mean Rank (higher = better)")
    ax.set_title("Figure 4: Cross-Dataset Pipeline Ranking (Top 20)",
                 fontweight="bold", fontsize=13)

    # Add rank numbers
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(0.5, i, f"#{int(row['overall_rank'])}",
                va="center", fontsize=8, fontweight="bold", color="white")

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.7)
               for c in NORM_COLORS.values()]
    ax.legend(handles, NORM_COLORS.keys(), loc="lower right", fontsize=9,
              title="Normalization")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure4_forest.pdf")
    fig.savefig(FIGURES_DIR / "figure4_forest.png")
    plt.close()
    print("  Figure 4: Forest plot saved")


def figure5_consistency(results):
    """Rank correlation between datasets."""
    if len(results) < 2:
        print("  Figure 5: Skipped (need 2+ datasets)")
        return

    # Compute ARI rank for each pipeline in each dataset
    rank_data = {}
    for ds, df in results.items():
        if "pipeline" in df.columns and "ARI" in df.columns:
            r = df.set_index("pipeline")["ARI"].astype(float).rank(ascending=False)
            rank_data[DATASET_NAMES.get(ds, ds)] = r

    if len(rank_data) < 2:
        return

    rank_df = pd.DataFrame(rank_data).dropna()

    fig, ax = plt.subplots(figsize=(7, 6))
    corr = rank_df.corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlBu_r",
                vmin=-1, vmax=1, ax=ax, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8})
    ax.set_title("Figure 5: Cross-Dataset Rank Correlation (Spearman, ARI)",
                 fontweight="bold", fontsize=12)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure5_consistency.pdf")
    fig.savefig(FIGURES_DIR / "figure5_consistency.png")
    plt.close()
    print("  Figure 5: Consistency heatmap saved")



def figure6_clustering(results):
    """Compare clustering algorithms across metrics and datasets."""
    metrics = ["ARI", "NMI", "silhouette"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    for ax, metric in zip(axes, metrics):
        all_data = []
        for ds, df in results.items():
            if "clust" in df.columns and metric in df.columns:
                sub = df[["clust", metric]].copy()
                sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
                sub["dataset"] = DATASET_NAMES.get(ds, ds)
                all_data.append(sub)

        if not all_data:
            continue

        combined = pd.concat(all_data)
        order = ["leiden", "louvain", "hierarchical", "hdbscan"]
        colors = [CLUST_COLORS.get(c, "gray") for c in order]

        sns.boxplot(data=combined, x="clust", y=metric, ax=ax,
                   order=order, palette=colors, width=0.6, fliersize=3)
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontweight="bold")
        ax.set_title(metric, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Figure 6: Clustering Algorithm Comparison (All Datasets Combined)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure6_clustering.pdf")
    fig.savefig(FIGURES_DIR / "figure6_clustering.png")
    plt.close()
    print("  Figure 6: Clustering comparison saved")



def generate_summary_table(results, cross_ranking):
    """Generate summary statistics table for the paper."""
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds, df in results.items():
        if "norm" not in df.columns:
            continue
        for norm in ["log", "scran", "sctransform", "pearson_residuals", "analytic_pearson"]:
            mask = df["norm"] == norm
            if not mask.any():
                continue
            sub = df.loc[mask]
            row = {
                "dataset": DATASET_NAMES.get(ds, ds),
                "normalization": norm,
                "n_pipelines": len(sub),
            }
            for metric in ["ARI", "NMI", "silhouette", "cell_type_purity"]:
                if metric in sub.columns:
                    vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
                    if len(vals) > 0:
                        row[f"{metric}_mean"] = f"{vals.mean():.3f}"
                        row[f"{metric}_std"] = f"{vals.std():.3f}"
                        row[f"{metric}_max"] = f"{vals.max():.3f}"
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(tables_dir / "summary_statistics.csv", index=False)
    print(f"  Summary table: {tables_dir / 'summary_statistics.csv'}")

    return summary_df



def main():
    print("Phase 4: Figure Generation")
    print(f"Output: {FIGURES_DIR}")

    # Load data
    results = load_all_results()
    rankings = load_rankings()
    cross_ranking = load_cross_dataset()

    print(f"Loaded: {len(results)} datasets")

    # Generate figures
    print(f"\nGenerating figures...")
    figure1_heatmap(results)
    figure2_normalization(results)
    figure3_bds(rankings)
    figure4_forest(cross_ranking)
    figure5_consistency(results)
    figure6_clustering(results)

    # Summary table
    print(f"\nGenerating tables...")
    summary = generate_summary_table(results, cross_ranking)

    # List outputs
    print(f"\n{'='*60}")
    print(f"  Generated files:")
    print(f"{'='*60}")
    for f in sorted(FIGURES_DIR.glob("*")):
        size = f.stat().st_size / 1024
        print(f"  {f.name} ({size:.0f} KB)")
    for f in sorted(Path("results/tables").glob("*.csv")):
        print(f"  tables/{f.name}")

    print(f"\n  All figures generated!")


if __name__ == "__main__":
    main()
