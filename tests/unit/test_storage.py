"""Unit tests for MinIO/S3 storage client.

Tests sync upload, download, and bucket creation with mocked boto3.
Async variants are tested separately (require aiobotocore).
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("boto3", reason="boto3 not installed")

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# upload_bytes_sync
# ---------------------------------------------------------------------------

class TestUploadBytesSync:
    @patch("app.services.storage.get_sync_s3_client")
    def test_calls_put_object(self, mock_get_client):
        from app.services.storage import upload_bytes_sync
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        result = upload_bytes_sync(b"data", "bucket", "key/file.pdf", content_type="application/pdf")
        mock_client.put_object.assert_called_once_with(
            Bucket="bucket", Key="key/file.pdf", Body=b"data", ContentType="application/pdf",
        )
        assert result == "key/file.pdf"

    @patch("app.services.storage.get_sync_s3_client")
    def test_default_content_type(self, mock_get_client):
        from app.services.storage import upload_bytes_sync
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        upload_bytes_sync(b"data", "bucket", "key/file.bin")
        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["ContentType"] == "application/octet-stream"


# ---------------------------------------------------------------------------
# download_bytes_sync
# ---------------------------------------------------------------------------

class TestDownloadBytesSync:
    @patch("app.services.storage.get_sync_s3_client")
    def test_returns_body_bytes(self, mock_get_client):
        from app.services.storage import download_bytes_sync
        mock_client = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b"file contents"
        mock_client.get_object.return_value = {"Body": mock_body}
        mock_get_client.return_value = mock_client
        result = download_bytes_sync("bucket", "key/file.pdf")
        assert result == b"file contents"
        mock_client.get_object.assert_called_once_with(Bucket="bucket", Key="key/file.pdf")


# ---------------------------------------------------------------------------
# get_sync_s3_client
# ---------------------------------------------------------------------------

class TestGetSyncS3Client:
    @patch("app.services.storage.boto3")
    @patch("app.services.storage.settings")
    def test_creates_client_with_settings(self, mock_settings, mock_boto3):
        from app.services.storage import get_sync_s3_client
        mock_settings.minio_secure = False
        mock_settings.minio_endpoint = "minio:9000"
        mock_settings.minio_access_key = "minioadmin"
        mock_settings.minio_secret_key = "minioadmin"
        client = get_sync_s3_client()
        mock_boto3.client.assert_called_once()
        call_kwargs = mock_boto3.client.call_args[1]
        assert call_kwargs["endpoint_url"] == "http://minio:9000"
        assert call_kwargs["aws_access_key_id"] == "minioadmin"

    @patch("app.services.storage.boto3")
    @patch("app.services.storage.settings")
    def test_secure_endpoint_uses_https(self, mock_settings, mock_boto3):
        from app.services.storage import get_sync_s3_client
        mock_settings.minio_secure = True
        mock_settings.minio_endpoint = "minio:9000"
        mock_settings.minio_access_key = "key"
        mock_settings.minio_secret_key = "secret"
        get_sync_s3_client()
        call_kwargs = mock_boto3.client.call_args[1]
        assert call_kwargs["endpoint_url"] == "https://minio:9000"
