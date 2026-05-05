FROM condaforge/miniforge3:latest

LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="scRNA-seq normalization and clustering benchmarking pipeline"
LABEL version="1.3"

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    cmake \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY docker/environment.yaml /tmp/environment.yaml

RUN mamba env create -f /tmp/environment.yaml && \
    mamba clean --all --yes

ENV PATH=/opt/conda/envs/scrna_benchmark/bin:$PATH
ENV CONDA_DEFAULT_ENV=scrna_benchmark
ENV R_HOME=/opt/conda/envs/scrna_benchmark/lib/R
ENV R_LIBS_USER=/opt/conda/envs/scrna_benchmark/lib/R/library

RUN mamba install -n scrna_benchmark -y -c conda-forge \
    r-curl \
    r-png \
    r-httr \
    r-xml2 \
    r-openssl \
    r-igraph \
    r-data.table \
    r-httpuv \
    r-shiny \
    r-reticulate \
    r-rcpp \
    r-rcpparmadillo \
    r-rcppeigen \
    r-matrix \
    r-irlba \
    r-jsonlite \
    r-mime \
    r-r6 \
    r-rlang \
    r-cli \
    r-glue \
    r-vctrs \
    r-lifecycle \
    r-dplyr \
    r-ggplot2 \
    r-tidyr \
    r-tibble \
    r-purrr \
    r-stringr \
    r-htmltools \
    r-htmlwidgets \
    r-later \
    r-promises \
    r-plotly \
    r-miniui \
    r-cowplot \
    r-patchwork \
    r-uwot \
    r-rann \
    r-pbapply \
    r-fitdistrplus \
    r-rocr \
    r-kernsmooth \
    r-spatstat.explore \
    r-leiden \
    r-rspectra \
    r-lmtest \
    r-fastdummies \
    r-sctransform \
    && mamba clean --all --yes

# Verify the critical compiled packages loaded (catch linking issues early)
RUN /opt/conda/envs/scrna_benchmark/bin/Rscript -e "\
    for (pkg in c('curl', 'png', 'igraph', 'httpuv', 'data.table', 'reticulate', 'plotly', 'shiny')) { \
        if (!require(pkg, character.only=TRUE, quietly=TRUE)) { \
            cat(paste('FAILED:', pkg, '\n')); quit(status=1) \
        } else { \
            cat(paste('OK:', pkg, '\n')) \
        } \
    }; cat('All conda R packages OK\n')"

RUN mamba install -n scrna_benchmark -y -c bioconda -c conda-forge \
    bioconductor-scran \
    bioconductor-scater \
    bioconductor-singlecellexperiment \
    && mamba clean --all --yes

# Install Dino + RcppHNSW from CRAN (not on conda-forge/bioconda)
RUN /opt/conda/envs/scrna_benchmark/bin/Rscript -e "\
    install.packages('BiocManager', repos='https://cloud.r-project.org'); \
    BiocManager::install('Dino', update=FALSE, ask=FALSE); \
    install.packages('RcppHNSW', repos='https://cloud.r-project.org')"

RUN /opt/conda/envs/scrna_benchmark/bin/Rscript -e "\
    install.packages('Seurat', repos='https://cloud.r-project.org', dependencies=NA); \
    if (!require('Seurat', quietly=TRUE)) { \
        cat('ERROR: Seurat failed to install\n'); \
        quit(status=1) \
    } else { \
        cat('OK: Seurat installed successfully\n') \
    }"

RUN /opt/conda/envs/scrna_benchmark/bin/Rscript -e "\
    packages <- c('Seurat', 'scran', 'scater', 'SingleCellExperiment', 'sctransform', 'Dino'); \
    failed <- c(); \
    for (pkg in packages) { \
        if (!require(pkg, character.only=TRUE, quietly=TRUE)) { \
            cat(paste('FAILED:', pkg, '\n')); \
            failed <- c(failed, pkg); \
        } else { \
            cat(paste('OK:', pkg, '\n')); \
        } \
    }; \
    if (length(failed) > 0) { \
        cat(paste('\nMissing packages:', paste(failed, collapse=', '), '\n')); \
        quit(status=1) \
    }; \
    cat('\nAll R packages verified OK\n')"

RUN echo "conda activate scrna_benchmark" >> ~/.bashrc

COPY . /workspace

CMD ["bash", "-l"]
