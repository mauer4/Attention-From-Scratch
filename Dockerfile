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
        pkg-config \
        unzip && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN mkdir -p setup requirements/locks

WORKDIR /workspace

COPY setup/bootstrap_host.sh setup/bootstrap_host.sh
COPY pyproject.toml pyproject.toml
COPY requirements/locks/olmo.lock requirements/locks/olmo.lock

RUN chmod +x setup/bootstrap_host.sh && \
    bash setup/bootstrap_host.sh \
      --system none \
      --python-env "${VIRTUAL_ENV}" \
      --cuda-toolkit none \
      --extras olmo \
      --lock requirements/locks/olmo.lock

COPY . /workspace

CMD ["/bin/bash"]
