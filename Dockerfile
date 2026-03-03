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

# System dependencies for ML libraries and document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
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
