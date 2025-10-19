FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        cmake \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        clang \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install --no-cache-dir \
        transformers \
        safetensors \
        sentencepiece \
        numpy \
        pybind11 \
        tqdm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY . /workspace

RUN mkdir -p third_party && \
    [ -d third_party/cutlass ] || git clone --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass && \
    cmake -S . -B build -G Ninja && \
    cmake --build build

CMD ["/bin/bash"]
