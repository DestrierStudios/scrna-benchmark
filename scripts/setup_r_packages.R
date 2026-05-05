cat("Installing R packages for scRNA-seq benchmarking...\n\n")

if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

cat("Installing Bioconductor packages...\n")
BiocManager::install("scran", update = FALSE, ask = FALSE)
BiocManager::install("scater", update = FALSE, ask = FALSE)
BiocManager::install("SingleCellExperiment", update = FALSE, ask = FALSE)
BiocManager::install("Dino", update = FALSE, ask = FALSE)

cat("Installing CRAN packages...\n")
install.packages("Seurat", repos = "https://cloud.r-project.org")
install.packages("sctransform", repos = "https://cloud.r-project.org")

cat("\nVerifying installations...\n")
library(scran)
library(scater)
library(SingleCellExperiment)
library(Seurat)
library(sctransform)
library(Dino)

cat("\n[SUCCESS] All R packages installed!\n")