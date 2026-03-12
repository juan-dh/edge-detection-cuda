FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN cp /etc/skel/.bashrc /root/.bashrc && \
    cp /etc/skel/.profile /root/.profile

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Use bash login shell for RUN
SHELL ["/bin/bash", "--login", "-c"]

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    conda clean -afy


# Accept Conda terms
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create environment from YAML
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Automatically activate environment
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate npp_env" >> ~/.bashrc && \
    echo "cd /root/edge-detection-cuda" >> ~/.bashrc

# Working directory
WORKDIR /root/edge-detection-cuda

CMD ["/bin/bash"]