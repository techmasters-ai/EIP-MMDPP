import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Source(Base, TimestampMixin):
    """Named collection of documents (e.g., a program, a project, a corpus)."""

    __tablename__ = "sources"
    __table_args__ = {"schema": "ingest"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    documents: Mapped[list["Document"]] = relationship(back_populates="source")


class Document(Base, TimestampMixin):
    """A single uploaded file within a source."""

    __tablename__ = "documents"
    __table_args__ = {"schema": "ingest"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ingest.sources.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(1024), nullable=False)
    mime_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # SHA-256

    # Object storage reference
    storage_bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(2048), nullable=False)

    # Pipeline state
    pipeline_status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="PENDING"
    )
    # PENDING | PROCESSING | COMPLETE | PARTIAL_COMPLETE | FAILED | PENDING_HUMAN_REVIEW
    pipeline_stage: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    failed_stages: Mapped[Optional[list]] = mapped_column(ARRAY(String), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Task tracking
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    uploaded_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    source: Mapped["Source"] = relationship(back_populates="documents")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="document")
    text_chunks: Mapped[list["TextChunk"]] = relationship(
        back_populates="document",
        primaryjoin="Document.id == foreign(TextChunk.document_id)",
    )
    image_chunks: Mapped[list["ImageChunk"]] = relationship(
        back_populates="document",
        primaryjoin="Document.id == foreign(ImageChunk.document_id)",
    )


class Artifact(Base, TimestampMixin):
    """An extracted content artifact from a document (page text, image, table, schematic)."""

    __tablename__ = "artifacts"
    __table_args__ = {"schema": "ingest"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ingest.documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Artifact type and modality
    artifact_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # text | image | table | schematic | ocr

    # Extracted content
    content_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Storage reference for binary artifacts (images, etc.)
    storage_bucket: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    storage_key: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)

    # Location within the source document
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bounding_box: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # OCR quality
    ocr_confidence: Mapped[Optional[float]] = mapped_column(nullable=True)
    ocr_engine: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    requires_human_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Classification
    classification: Mapped[str] = mapped_column(
        String(100), nullable=False, default="UNCLASSIFIED"
    )

    document: Mapped["Document"] = relationship(back_populates="artifacts")
    text_chunks: Mapped[list["TextChunk"]] = relationship(
        back_populates="artifact",
        primaryjoin="Artifact.id == foreign(TextChunk.artifact_id)",
    )
    image_chunks: Mapped[list["ImageChunk"]] = relationship(
        back_populates="artifact",
        primaryjoin="Artifact.id == foreign(ImageChunk.artifact_id)",
    )


class WatchDir(Base, TimestampMixin):
    """User-registered directory that is automatically polled for new documents."""

    __tablename__ = "watch_dirs"
    __table_args__ = {"schema": "ingest"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ingest.sources.id", ondelete="CASCADE"), nullable=False
    )
    path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    poll_interval_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    file_patterns: Mapped[list] = mapped_column(
        ARRAY(String),
        nullable=False,
        default=["*.pdf", "*.docx", "*.txt", "*.png", "*.jpg", "*.tiff"],
    )
    created_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    watch_logs: Mapped[list["WatchLog"]] = relationship(back_populates="watch_dir")


class WatchLog(Base):
    """Deduplication log for directory watcher."""

    __tablename__ = "watch_logs"
    __table_args__ = (
        UniqueConstraint("watch_dir_id", "file_hash", name="uq_watch_log_dir_hash"),
        {"schema": "ingest"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    watch_dir_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ingest.watch_dirs.id", ondelete="CASCADE"),
        nullable=False,
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA-256
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ingest.documents.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="enqueued")
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    watch_dir: Mapped["WatchDir"] = relationship(back_populates="watch_logs")


# Import to resolve forward references
from app.models.retrieval import TextChunk, ImageChunk  # noqa: E402
