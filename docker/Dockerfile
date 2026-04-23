# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=22.04
ARG TORCH_VERSION=2.7.1
ARG TORCH_CUDA_TAG=cu128
ARG FLASH_ATTN_VERSION=2.8.3

# ======================================================================
# Builder: install torch + upstream engine + API deps + flash-attn
# Uses the *devel* CUDA image so nvcc is available for flash-attn's source
# build. The runtime stage below still uses the smaller *runtime* image.
# ======================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG TORCH_VERSION
ARG TORCH_CUDA_TAG
ARG FLASH_ATTN_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        build-essential \
        git \
        ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip wheel setuptools

# PyTorch (CUDA)
RUN pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} \
        torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION}

# HF transfer accelerator (used when HF_HUB_ENABLE_HF_TRANSFER=1)
RUN pip install hf_transfer

# Upstream Qwen3-TTS engine
WORKDIR /build
COPY engine /build/engine
RUN pip install /build/engine

# API-layer deps
COPY docker/requirements.api.txt /tmp/requirements.api.txt
RUN pip install -r /tmp/requirements.api.txt

# Engine extra deps
COPY docker/requirements.engine.txt /tmp/requirements.engine.txt
RUN pip install -r /tmp/requirements.engine.txt

# Flash-Attention 2 (built from source against the torch just installed).
# MAX_JOBS caps nvcc parallelism so the GitHub Actions runner (7 GB RAM) does
# not OOM during the build.
RUN MAX_JOBS=4 pip install --no-build-isolation flash-attn==${FLASH_ATTN_VERSION}

RUN find /opt/venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ======================================================================
# Runtime: only keep deps required at run time
# ======================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/api:/opt/api/engine

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        ffmpeg \
        libsndfile1 \
        libsox-fmt-all \
        sox \
        ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /opt/api

COPY engine /opt/api/engine
COPY app /opt/api/app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/voices", "/root/.cache"]
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
