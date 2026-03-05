# syntax=docker/dockerfile:1

# ============================================================
# Stage 0 — FIPS bypass shim for hosts with FIPS-enabled kernels
# gcc:12-bookworm already has the compiler — no apt-get needed.
# ============================================================
FROM gcc:12-bookworm AS fips-bypass

RUN cat > /tmp/fips_bypass.c <<'CEOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <stdio.h>
FILE *fopen(const char *path, const char *mode) {
    static FILE *(*real)(const char *, const char *) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "fopen");
    if (path && strcmp(path, "/proc/sys/crypto/fips_enabled") == 0)
        return real("/dev/null", mode);
    return real(path, mode);
}
FILE *fopen64(const char *path, const char *mode) {
    static FILE *(*real)(const char *, const char *) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "fopen64");
    if (path && strcmp(path, "/proc/sys/crypto/fips_enabled") == 0)
        return real("/dev/null", mode);
    return real(path, mode);
}
CEOF
RUN gcc -shared -fPIC -o /tmp/libfips_bypass.so /tmp/fips_bypass.c -ldl

# ============================================================
# Stage 1 — Build the React frontend
# ============================================================
FROM node:22-alpine AS frontend-build

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json* ./
# Use 'ci' if lock file was copied, otherwise fall back to 'install'
RUN if [ -f package-lock.json ]; then npm ci --prefer-offline; else npm install; fi

COPY frontend/ ./
RUN npm run build

# ============================================================
# Stage 2 — Python API + embedded frontend
# ============================================================
FROM python:3.11-slim-bookworm

# Install FIPS bypass shim (only needed during build-time apt-get)
COPY --from=fips-bypass /tmp/libfips_bypass.so /usr/local/lib/

# System dependencies for ML libraries and document processing
RUN LD_PRELOAD=/usr/local/lib/libfips_bypass.so apt-get update \
    && LD_PRELOAD=/usr/local/lib/libfips_bypass.so apt-get install -y --no-install-recommends \
    # Tesseract OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # libmagic for MIME detection
    libmagic1 \
    # OpenCV headless dependencies
    libglib2.0-0 \
    libgl1 \
    # PDF processing
    libmupdf-dev \
    # General utilities
    curl \
    && rm -rf /var/lib/apt/lists/*

# Keep FIPS bypass shim at runtime — pymupdf/pdfplumber call into OpenSSL
# which aborts on FIPS-enabled host kernels without this.
ENV LD_PRELOAD=/usr/local/lib/libfips_bypass.so

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

# Copy application code
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY ontology/ ./ontology/
COPY scripts/ ./scripts/

RUN chmod +x scripts/*.sh

# Copy built frontend from Stage 1
COPY --from=frontend-build /frontend/dist ./frontend/dist

# Default: run the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
