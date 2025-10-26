ARG CUDA_IMAGE_TAG=12.8.0-cudnn9-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_IMAGE_TAG}

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
        ca-certificates \
        cuda-toolkit-12-8 \
        cuda-nsight-systems-12-8 && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN python3 -m venv "${VIRTUAL_ENV}"

COPY setup/requirements/requirements.txt /tmp/requirements.txt

RUN "${VIRTUAL_ENV}/bin/pip" install --upgrade pip && \
    "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

WORKDIR /workspace

COPY . /workspace

RUN mkdir -p third_party && \
    [ -d third_party/cutlass ] || git clone --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass && \
    cmake -S . -B build -G Ninja && \
    cmake --build build

CMD ["/bin/bash"]
