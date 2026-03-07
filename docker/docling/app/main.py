"""Docling document conversion service.

FastAPI wrapper around the Docling library with granite-docling-258M VLM model.
Provides /convert and /health endpoints for the EIP-MMDPP ingest pipeline.

Conversion runs in a threadpool so the async /health endpoint remains responsive
during CPU-bound VLM processing. A semaphore limits concurrent conversions
(capacity=1 on CPU) and returns 503 when saturated.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.converter import convert_document, init_converter, is_model_loaded
from app.schemas import ConvertResponse, HealthResponse

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("DOCLING_MODEL_PATH", "/models/granite-docling-258M")
DEVICE = os.environ.get("DOCLING_DEVICE", "cuda")
MAX_CONCURRENT = int(os.environ.get("DOCLING_MAX_CONCURRENT", "1"))

_convert_semaphore: asyncio.Semaphore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Docling model at startup."""
    global _convert_semaphore
    _convert_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    logger.info("Starting Docling service — loading model...")
    init_converter()
    logger.info("Docling service ready (max_concurrent=%d).", MAX_CONCURRENT)
    yield
    logger.info("Docling service shutting down.")


app = FastAPI(
    title="EIP-MMDPP Docling Service",
    description="Document conversion via granite-docling-258M VLM",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — reports model load status.

    Always responsive (not blocked by in-flight conversions) because
    conversion runs in a threadpool.
    """
    return HealthResponse(
        status="ok" if is_model_loaded() else "loading",
        model_loaded=is_model_loaded(),
        model_name=MODEL_PATH,
        device=DEVICE,
    )


@app.post("/convert", response_model=ConvertResponse)
async def convert(file: UploadFile):
    """Convert an uploaded document through the Docling VLM pipeline.

    Accepts PDF, DOCX, PNG, JPG, TIFF files.
    Returns structured elements (text, tables, images, equations) plus full markdown.

    Returns 503 if the service is already processing the maximum number of
    concurrent conversions.
    """
    assert _convert_semaphore is not None

    if not _convert_semaphore._value:  # noqa: SLF001
        raise HTTPException(
            status_code=503,
            detail=f"Docling at capacity ({MAX_CONCURRENT} concurrent conversion(s)). Retry later.",
        )

    async with _convert_semaphore:
        file_bytes = await file.read()
        filename = file.filename or "document.pdf"

        logger.info("Converting %s (%d bytes)", filename, len(file_bytes))
        result = await run_in_threadpool(convert_document, file_bytes, filename)
        logger.info(
            "Conversion %s: %s — %d elements, %d pages, %.0fms",
            result.status,
            filename,
            len(result.elements),
            result.num_pages,
            result.processing_time_ms,
        )
        return result
