# syntax=docker/dockerfile:1
#
# Jupyter sidecar: shares the worker's Python environment so every import
# in the notebook resolves the same way as production code.
#
# Source is volume-mounted (see docker-compose.jupyter.yml), not COPY'd,
# so edits to the repo are visible without rebuilds. This differs from the
# main Dockerfile on purpose — the notebook is a live troubleshooting tool.

FROM python:3.11-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libmagic1 is required by python-magic (prepare_document uses it for MIME
# detection). build-essential + curl cover the usual pip-from-source cases.
# libxcb1 + libxrender1 + libsm6 + libglib2.0-0 + libgl1 are OpenCV runtime
# deps — pulled in by docling-ibm-models TableFormer when running Docling
# conversion in-process (the notebook calls DocumentConverter directly;
# the production docling service image already has these).
# tesseract-ocr + eng language pack supply an OCR engine so scanned PDFs
# convert cleanly (docling falls back gracefully if absent but emits a
# warning).
# Using bullseye (not bookworm) because bookworm's libgcrypt20 aborts with
# "requested algo not in md context" when apt-get update runs inside a
# container on a FIPS-enabled host kernel (Linux *-fips). See
# https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=1022199. Bullseye's
# older libgcrypt doesn't hit the same FIPS-policy rejection.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libmagic1 \
        libxcb1 \
        libxrender1 \
        libsm6 \
        libglib2.0-0 \
        libgl1 \
        tesseract-ocr \
        tesseract-ocr-eng \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the same deps the worker does — editable install from pyproject.
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" \
    && pip install --no-cache-dir jupyterlab ipykernel

# CUDA-enabled PyTorch. The pyproject install pulled the default (CPU-only)
# wheel from PyPI; reinstall from PyTorch's CUDA 12.4 index so the wheel
# bundles the CUDA runtime libs open_clip + sentence-transformers need.
# The host driver (nvidia-container-runtime) is mounted in at runtime;
# no cuda-toolkit system package needed. CUDA 12.4 wheels are
# forward-compatible with the host's CUDA 13 driver (PyTorch policy).
RUN pip install --no-cache-dir --upgrade \
      --index-url https://download.pytorch.org/whl/cu124 \
      torch torchvision

EXPOSE 8888

# Token + password disabled: the service is only exposed on localhost via
# the compose port mapping. Do NOT expose this port publicly.
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.token=", \
     "--ServerApp.password="]
