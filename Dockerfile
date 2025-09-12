FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV TORCH_CUDA_ARCH_LIST="8.9" \
    DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:$PATH \
    CPATH=/usr/local/cuda-11.8/targets/x86_64-linux/include:$CPATH \
    LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libgl1-mesa-dev \
    libegl1 \
    libegl1-mesa \
    libgles2-mesa \
    libx11-6 \
    libxext6 \
    libxdamage1 \
    libxfixes3 \
    libx11-xcb1 \
    libxcb1 \
    libxrender1 \
    libxrandr2 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda clean -afy

# Accept Conda Terms of Service
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Prioritize conda-forge channel
RUN conda config --system --add channels conda-forge && \
    conda config --system --set channel_priority strict

RUN conda create -n milo python=3.9 -y

# Install PyTorch with CUDA 11.8 inside the "milo" environment
RUN /bin/bash -c "source activate milo && \
    conda install -y pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia"

# Install ninja build system
RUN /bin/bash -c "source activate milo && conda install -y ninja -c conda-forge"

WORKDIR /workspace/MILo

COPY requirements.txt ./

# Install additional Python dependencies from requirements.txt
RUN /bin/bash -c "source activate milo && pip install -r requirements.txt"

COPY .gitmodules ./
COPY .git ./.git
COPY submodules ./submodules

# Initialize git submodules and convert URLs to HTTPS
RUN git submodule init && \
    git config submodule.submodules/Depth-Anything-V2.url https://github.com/DepthAnything/Depth-Anything-V2.git && \
    git config submodule.submodules/nvdiffrast.url https://github.com/NVlabs/nvdiffrast.git && \
    git submodule update --init --recursive

# Install Gaussian Splatting related submodules
RUN /bin/bash -c "source activate milo && \
    pip install submodules/diff-gaussian-rasterization_ms \
                submodules/diff-gaussian-rasterization \
                submodules/diff-gaussian-rasterization_gof \
                submodules/simple-knn \
                submodules/fused-ssim"

# Build Tetra-Nerf Delaunay Triangulation
WORKDIR /workspace/MILo/submodules/tetra_triangulation
RUN /bin/bash -c "source activate milo && \
    conda install -y cmake && \
    conda install -y -c conda-forge gmp cgal && \
    cmake . && make && pip install -e ."

# Install nvdiffrast for efficient mesh rasterization
WORKDIR /workspace/MILo/submodules/nvdiffrast
RUN /bin/bash -c "source activate milo && pip install ."

WORKDIR /workspace
COPY . MILo/

# Return to the MILo root directory
WORKDIR /workspace/MILo

SHELL ["/bin/bash", "--login", "-c"]
CMD ["bash", "-c", "source activate milo && bash"]
