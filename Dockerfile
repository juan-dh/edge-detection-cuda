FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu22.04

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

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y libfreeimage-dev

# Directorio temporal para clonar los samples
WORKDIR /tmp

# Clona los CUDA Samples de NVIDIA
RUN git clone --depth 1 https://github.com/NVIDIA/cuda-samples.git

# Crea la carpeta de samples dentro de /usr/local/cuda
RUN mkdir -p /usr/local/cuda/samples

# Copia todos los samples a /usr/local/cuda/samples
RUN cp -r cuda-samples/* /usr/local/cuda/samples/

# Limpia el temporal
RUN rm -rf /tmp/cuda-samples

# Working directory
WORKDIR /root/edge-detection-cuda

CMD ["/bin/bash"]