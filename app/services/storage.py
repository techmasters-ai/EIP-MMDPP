"""MinIO / S3-compatible object storage client.

Provides both async (aiobotocore, for FastAPI) and sync (boto3, for Celery workers)
variants to avoid mixing async drivers into synchronous Celery tasks.
"""

import hashlib
import io
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

import aiobotocore.session
import boto3
from botocore.client import Config

from app.config import get_settings

settings = get_settings()


# ---------------------------------------------------------------------------
# Async client (FastAPI)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def get_async_s3_client():
    """Yield an async S3/MinIO client scoped to a single request."""
    session = aiobotocore.session.get_session()
    async with session.create_client(
        "s3",
        endpoint_url=f"http{'s' if settings.minio_secure else ''}://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    ) as client:
        yield client


async def upload_file_async(
    file_content: bytes,
    bucket: str,
    key: str,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload bytes to MinIO. Returns the storage key."""
    async with get_async_s3_client() as client:
        await client.put_object(
            Bucket=bucket,
            Key=key,
            Body=file_content,
            ContentType=content_type,
        )
    return key


async def stream_upload_async(
    stream,
    bucket: str,
    key: str,
    content_type: str = "application/octet-stream",
) -> tuple[str, int, str]:
    """Stream an upload from a file-like async iterator to MinIO.

    Returns (key, bytes_written, sha256_hex).
    Files > 10MB are uploaded using multipart.
    """
    CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB

    async with get_async_s3_client() as client:
        # Buffer first chunk to check if we need multipart
        first_chunk = await stream.read(CHUNK_SIZE)

        if len(first_chunk) < CHUNK_SIZE:
            # Single-part upload
            sha256 = hashlib.sha256(first_chunk).hexdigest()
            await client.put_object(
                Bucket=bucket,
                Key=key,
                Body=first_chunk,
                ContentType=content_type,
            )
            return key, len(first_chunk), sha256

        # Multipart upload for large files
        mpu = await client.create_multipart_upload(
            Bucket=bucket, Key=key, ContentType=content_type
        )
        upload_id = mpu["UploadId"]
        parts = []
        part_number = 1
        total_bytes = 0
        hasher = hashlib.sha256()

        try:
            chunk = first_chunk
            while chunk:
                hasher.update(chunk)
                response = await client.upload_part(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=chunk,
                )
                parts.append({"PartNumber": part_number, "ETag": response["ETag"]})
                total_bytes += len(chunk)
                part_number += 1
                chunk = await stream.read(CHUNK_SIZE)

            await client.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
        except Exception:
            await client.abort_multipart_upload(
                Bucket=bucket, Key=key, UploadId=upload_id
            )
            raise

        return key, total_bytes, hasher.hexdigest()


async def delete_object_async(bucket: str, key: str) -> None:
    """Delete an object from MinIO."""
    async with get_async_s3_client() as client:
        await client.delete_object(Bucket=bucket, Key=key)


async def download_bytes_async(bucket: str, key: str) -> bytes:
    """Download an object from MinIO and return its content as bytes."""
    async with get_async_s3_client() as client:
        response = await client.get_object(Bucket=bucket, Key=key)
        async with response["Body"] as stream:
            return await stream.read()


async def generate_presigned_url_async(
    bucket: str, key: str, expires_in: int = 3600
) -> str:
    """Generate a presigned download URL."""
    async with get_async_s3_client() as client:
        return await client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )


# ---------------------------------------------------------------------------
# Sync client (Celery workers)
# ---------------------------------------------------------------------------

def get_sync_s3_client():
    """Return a synchronous boto3 S3 client for Celery worker use."""
    return boto3.client(
        "s3",
        endpoint_url=f"http{'s' if settings.minio_secure else ''}://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def download_bytes_sync(bucket: str, key: str) -> bytes:
    """Download bytes synchronously (Celery worker context)."""
    client = get_sync_s3_client()
    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def upload_bytes_sync(
    data: bytes,
    bucket: str,
    key: str,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload bytes synchronously (Celery worker context). Returns key."""
    client = get_sync_s3_client()
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    return key
