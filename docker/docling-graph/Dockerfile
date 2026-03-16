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
# Stage 1 — Docling-Graph extraction service
# ============================================================
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# Install FIPS bypass shim (needed at build-time for apt-get)
COPY --from=fips-bypass /tmp/libfips_bypass.so /usr/local/lib/
COPY --from=fips-bypass /tmp/fips_disabled /usr/local/lib/fips_disabled

RUN LD_PRELOAD=/usr/local/lib/libfips_bypass.so apt-get update \
    && LD_PRELOAD=/usr/local/lib/libfips_bypass.so apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/local/lib/libfips_bypass.so

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8002

# Derive OLLAMA_HOST and OLLAMA_API_BASE from OLLAMA_BASE_URL so the
# ollama Python package and LiteLLM both find the right endpoint.
# Shell form so env var expansion works at runtime.
CMD sh -c 'export OLLAMA_HOST="${OLLAMA_BASE_URL:-http://ollama:11434}" OLLAMA_API_BASE="${OLLAMA_BASE_URL:-http://ollama:11434}" && exec uvicorn app.main:app --host 0.0.0.0 --port 8002'
