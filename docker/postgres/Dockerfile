FROM postgres:16-bookworm

# Build-time dependencies for compiling pgvector and Apache AGE
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
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

# Install Apache AGE (PG16 / v1.5.0)
RUN git clone --branch PG16/v1.5.0 --depth 1 https://github.com/apache/age.git /tmp/age \
    && cd /tmp/age \
    && make \
    && make install \
    && rm -rf /tmp/age

# Copy initialization SQL (runs in alphabetical order on first start)
COPY init/ /docker-entrypoint-initdb.d/
