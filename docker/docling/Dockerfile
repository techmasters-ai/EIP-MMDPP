FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    curl libgl1 libglib2.0-0 poppler-utils \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model for air-gapped deployment
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('ibm-granite/granite-docling-258M', local_dir='/models/granite-docling-258M')"

ENV HF_HOME=/models
ENV TRANSFORMERS_OFFLINE=1
ENV DOCLING_MODEL_PATH=/models/granite-docling-258M

COPY app/ ./app/

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
