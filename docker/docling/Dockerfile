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

RUN LD_PRELOAD=/usr/local/lib/libfips_bypass.so apt-get update \
    && LD_PRELOAD=/usr/local/lib/libfips_bypass.so apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    curl libgl1 libglib2.0-0 poppler-utils ninja-build \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Keep FIPS bypass shim at runtime — OpenSSL FIPS self-test fails on
# FIPS-enabled host kernels when Python crypto libraries initialize.
ENV LD_PRELOAD=/usr/local/lib/libfips_bypass.so

WORKDIR /app

# --- Stable deps (cached normally) ---
COPY requirements-stable.txt .
RUN pip install --no-cache-dir packaging ninja && \
    pip install --no-cache-dir -r requirements-stable.txt

# Model downloads (stays cached)
# PdfPipeline uses EasyOCR + TableFormer + CodeFormulaV2 (downloaded on first use)
ENV HF_HOME=/opt/huggingface

# --- Updatable deps (cache-busted on every start) ---
ARG CACHE_BUST
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Lock to offline mode for air-gapped runtime
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY app/ ./app/

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
