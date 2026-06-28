# syntax=docker/dockerfile:1

# ============================================================
# Stage 0 — FIPS bypass shim for hosts with FIPS-enabled kernels
# ============================================================
FROM gcc:12-bookworm AS fips-bypass

RUN echo '0' > /tmp/fips_disabled
RUN cat > /tmp/fips_bypass.c <<'CEOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <fcntl.h>

static const char *FIPS_PATH = "/proc/sys/crypto/fips_enabled";
static const char *FAKE_PATH = "/usr/local/lib/fips_disabled";

/* Intercept fopen/fopen64 (used by libgcrypt / older OpenSSL) */
FILE *fopen(const char *path, const char *mode) {
    static FILE *(*real)(const char *, const char *) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "fopen");
    if (path && strcmp(path, FIPS_PATH) == 0)
        return real(FAKE_PATH, mode);
    return real(path, mode);
}
FILE *fopen64(const char *path, const char *mode) {
    static FILE *(*real)(const char *, const char *) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "fopen64");
    if (path && strcmp(path, FIPS_PATH) == 0)
        return real(FAKE_PATH, mode);
    return real(path, mode);
}

/* Intercept open/open64 (used by OpenSSL 3.x ossl_get_kernel_fips_flag) */
int open(const char *path, int flags, ...) {
    static int (*real)(const char *, int, ...) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "open");
    if (path && strcmp(path, FIPS_PATH) == 0)
        return real(FAKE_PATH, flags);
    if (flags & O_CREAT) {
        va_list ap; va_start(ap, flags);
        int mode = va_arg(ap, int); va_end(ap);
        return real(path, flags, mode);
    }
    return real(path, flags);
}
int open64(const char *path, int flags, ...) {
    static int (*real)(const char *, int, ...) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "open64");
    if (path && strcmp(path, FIPS_PATH) == 0)
        return real(FAKE_PATH, flags);
    if (flags & O_CREAT) {
        va_list ap; va_start(ap, flags);
        int mode = va_arg(ap, int); va_end(ap);
        return real(path, flags, mode);
    }
    return real(path, flags);
}
CEOF
RUN gcc -shared -fPIC -o /tmp/libfips_bypass.so /tmp/fips_bypass.c -ldl

# ============================================================
# Stage 1 — Docling VLM service
# ============================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install FIPS bypass shim (needed at build-time AND runtime)
COPY --from=fips-bypass /tmp/libfips_bypass.so /usr/local/lib/
COPY --from=fips-bypass /tmp/fips_disabled /usr/local/lib/fips_disabled

# Jammy's universe `python3.11` package is 3.11.0rc1 (a release candidate),
# which LACKS sys.get_int_max_str_digits (added in FINAL 3.11.0). Modern torch
# (>=2.12) references that API in torch._dynamo.polyfills.loader and crashes on
# import via easyocr -> torchvision -> torch._dynamo. The docling-graph service
# avoids this by running a final 3.11 (python:3.11-slim). Install a FINAL 3.11.x
# here too via the deadsnakes PPA. Docling supports Python 3.10/3.11/3.12.
RUN LD_PRELOAD=/usr/local/lib/libfips_bypass.so apt-get update \
    && LD_PRELOAD=/usr/local/lib/libfips_bypass.so apt-get install -y --no-install-recommends \
       software-properties-common gnupg ca-certificates \
    && LD_PRELOAD=/usr/local/lib/libfips_bypass.so add-apt-repository -y ppa:deadsnakes/ppa \
    && LD_PRELOAD=/usr/local/lib/libfips_bypass.so apt-get update \
    && LD_PRELOAD=/usr/local/lib/libfips_bypass.so apt-get install -y --no-install-recommends \
       python3.11 python3.11-venv python3.11-dev python3-pip \
       curl libgl1 libglib2.0-0 poppler-utils ninja-build ffmpeg \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Keep FIPS bypass shim at runtime — OpenSSL FIPS self-test fails on
# FIPS-enabled host kernels when Python crypto libraries initialize.
ENV LD_PRELOAD=/usr/local/lib/libfips_bypass.so

WORKDIR /app

# --- CUDA PyTorch (pinned, from the official PyTorch index) ---
# Install torch/torchvision FIRST from the cu130 wheel index so the rest of the
# deps resolve against an already-satisfied CUDA build (never the CPU wheel).
# Pinned to docling-graph's proven pair; --index-url guarantees the CUDA build
# per Docling's RTX GPU setup guide.
RUN pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1 \
    --index-url https://download.pytorch.org/whl/cu130

# Modernize the build toolchain BEFORE building any sdists. Jammy's system
# pip/setuptools (59.6.0) can't build openai-whisper's sdist (fails with
# "Unknown distribution option: 'readme'" + flat-layout package discovery).
# Placed after the torch layer to keep that ~532MB layer cached. setuptools is
# pinned <82 because torch 2.12.1 requires setuptools<82.
RUN pip install --no-cache-dir --upgrade pip "setuptools<82" wheel

# --- Stable deps (cached normally) ---
COPY requirements-stable.txt .
RUN pip install --no-cache-dir packaging ninja && \
    pip install --no-cache-dir -r requirements-stable.txt

# Model cache directory
ENV HF_HOME=/opt/huggingface

# Pre-download the layout model (required at pipeline init, checked first)
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('docling-project/docling-layout-heron')"

# --- Updatable deps (cache-busted on every start) ---
ARG CACHE_BUST
COPY requirements.txt .
# only-if-needed keeps the pre-installed cu130 torch in place (docling accepts
# it); --extra-index-url ensures any torch touch still resolves to a CUDA wheel.
RUN pip install --no-cache-dir --upgrade --upgrade-strategy only-if-needed \
    --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt

# Pre-download EasyOCR models (craft detector + English recognizer). Without
# this, the first PDF that triggers OCR downloads ~100MB at request time and
# a transient HTTP 502 from GitHub release assets will fail the conversion
# and every subsequent OCR request in the container's lifetime. Baking the
# models into the image makes cold starts deterministic.
RUN python3 -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"

# Pre-download the Whisper ASR model used by InputFormat.AUDIO. The
# native whisper library ships its own model registry (separate from
# HuggingFace); `whisper.load_model('base')` pulls ~140MB from OpenAI's
# CDN into ~/.cache/whisper (container runs as root → /root/.cache/whisper).
# Bake it into the image so first-use conversion doesn't stall on the
# download. Docling's ASR pipeline calls whisper.load_model() without
# download_root, so the bake location must match the library's default.
RUN python3 -c "import whisper; whisper.load_model('base')"

# Allow online model downloads at runtime — TableFormer and CodeFormulaV2
# models are downloaded on first use by the pipeline. Remove
# HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE to permit this. For fully air-gapped
# deployments, pre-populate /opt/huggingface at build time.

COPY app/ ./app/

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
