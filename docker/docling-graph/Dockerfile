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
    patch \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH=/app

ARG CACHE_BUST
COPY docker/docling-graph/requirements.txt .
RUN LD_PRELOAD=/usr/local/lib/libfips_bypass.so pip install --no-cache-dir --upgrade -r requirements.txt \
    && LD_PRELOAD=/usr/local/lib/libfips_bypass.so python -m spacy download en_core_web_md \
    && rm -f /usr/local/lib/libfips_bypass.so

COPY docker/docling-graph/app /app/app
COPY docker/docling-graph/repo /app/repo
COPY docker/docling-graph/patches /app/patches
COPY ontology_bundles /app/ontology_bundles

# Spec 2026-05-11-table-aware-chunking §5 — mirror the shared
# table_normalization module into the docling-graph image so the
# integration block at app/main.py can import it. The empty
# app/services/__init__.py is required for Python to recognize
# `app.services` as a package (docling-graph's app/ has its own
# __init__.py from line 91's COPY).
COPY app/services/__init__.py /app/app/services/__init__.py
COPY app/services/table_normalization /app/app/services/table_normalization

# Apply local patches to the vendored docling-graph library, then install
# the patched source as an editable package so Python imports from /app/repo
# instead of the un-patched site-packages copy that pip install pulled
# earlier. Without this final pip install -e step, the patches existed in
# /app/repo but Python loaded the unpatched docling_graph from
# /usr/local/lib/python3.11/site-packages/, making every patch silently
# inactive. Verified with:
#     python -c "import inspect, docling_graph.pipeline.stages as s; \
#                print(inspect.getsourcefile(s))"
# After this step the import resolves to /app/repo/docling_graph/...
# Applied with -p1 so the leading "a/" / "b/" components in the unified
# diff resolve under /app/repo.
RUN cd /app/repo && for p in /app/patches/*.patch; do \
        echo "Applying patch: $p"; \
        patch -p1 < "$p"; \
    done \
    && pip install --no-cache-dir --no-deps -e /app/repo

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
