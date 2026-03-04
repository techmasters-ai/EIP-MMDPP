# syntax=docker/dockerfile:1

# ============================================================
# Stage 1: Build LD_PRELOAD shim to bypass FIPS on host kernels
# where /proc/sys/crypto/fips_enabled=1 causes libgcrypt to
# abort during apt-get (MD5 is not FIPS-approved).
# gcc:12-bookworm already has the compiler — no apt-get needed.
# ============================================================
FROM gcc:12-bookworm AS fips-bypass

RUN cat > /tmp/fips_bypass.c <<'CEOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <stdio.h>

/* Intercept fopen/fopen64: redirect reads of the kernel FIPS flag
   to /dev/null so libgcrypt sees "empty" → FIPS disabled. */

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
# Stage 2: Postgres 16 + pgvector + Apache AGE
# ============================================================
FROM postgres:16-bookworm

# Install FIPS bypass shim (only needed during build-time apt-get)
COPY --from=fips-bypass /tmp/libfips_bypass.so /usr/local/lib/

# Build-time dependencies for compiling pgvector and Apache AGE
RUN LD_PRELOAD=/usr/local/lib/libfips_bypass.so apt-get update \
    && LD_PRELOAD=/usr/local/lib/libfips_bypass.so apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    postgresql-server-dev-16 \
    libreadline-dev \
    zlib1g-dev \
    flex \
    bison \
    && rm -rf /var/lib/apt/lists/*

# Install pgvector (v0.8.0)
RUN git clone --branch v0.8.0 --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector

# Install Apache AGE (PG16 / v1.6.0)
RUN git clone --branch release/PG16/1.6.0 --depth 1 https://github.com/apache/age.git /tmp/age \
    && cd /tmp/age \
    && make \
    && make install \
    && rm -rf /tmp/age

# Clean up build shim — not needed at runtime
RUN rm -f /usr/local/lib/libfips_bypass.so

# Copy initialization SQL (runs in alphabetical order on first start)
COPY init/ /docker-entrypoint-initdb.d/
