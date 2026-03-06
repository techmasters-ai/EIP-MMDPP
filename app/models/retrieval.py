import os
import uuid
from typing import Optional

from sqlalchemy import Float, ForeignKey, Integer, SmallInteger, String, Text, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.models.base import Base, TimestampMixin

_TEXT_DIM = int(os.environ.get("TEXT_EMBEDDING_DIM", "1024"))
_IMAGE_DIM = int(os.environ.get("IMAGE_EMBEDDING_DIM", "512"))


class TextChunk(Base, TimestampMixin):
    """A chunk of extracted text content with BGE vector embedding."""

    __tablename__ = "text_chunks"
    __table_args__ = {"schema": "retrieval"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ingest.artifacts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ingest.documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Text embedding (BGE-large: 1024-dim; controlled by TEXT_EMBEDDING_DIM env var)
    # Kept during transition — vectors migrate to Qdrant (drop in 0006)
    embedding: Mapped[Optional[list]] = mapped_column(Vector(_TEXT_DIM), nullable=True)

    # Qdrant point ID (set after vector upsert to Qdrant)
    qdrant_point_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )

    # Modality and provenance
    modality: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # text | table | schematic | ocr
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bounding_box: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Classification
    classification: Mapped[str] = mapped_column(
        String(100), nullable=False, default="UNCLASSIFIED"
    )

    artifact: Mapped["Artifact"] = relationship(
        back_populates="text_chunks",
        primaryjoin="TextChunk.artifact_id == Artifact.id",
    )
    document: Mapped["Document"] = relationship(
        back_populates="text_chunks",
        primaryjoin="TextChunk.document_id == Document.id",
    )


class ImageChunk(Base, TimestampMixin):
    """A chunk of extracted image content with CLIP vector embedding."""

    __tablename__ = "image_chunks"
    __table_args__ = {"schema": "retrieval"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ingest.artifacts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ingest.documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Image embedding (CLIP ViT-B/32: 512-dim; controlled by IMAGE_EMBEDDING_DIM)
    # Kept during transition — vectors migrate to Qdrant (drop in 0006)
    embedding: Mapped[Optional[list]] = mapped_column(Vector(_IMAGE_DIM), nullable=True)

    # Qdrant point ID (set after vector upsert to Qdrant)
    qdrant_point_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )

    # Modality and provenance
    modality: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # image
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bounding_box: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Classification
    classification: Mapped[str] = mapped_column(
        String(100), nullable=False, default="UNCLASSIFIED"
    )

    artifact: Mapped["Artifact"] = relationship(
        back_populates="image_chunks",
        primaryjoin="ImageChunk.artifact_id == Artifact.id",
    )
    document: Mapped["Document"] = relationship(
        back_populates="image_chunks",
        primaryjoin="ImageChunk.document_id == Document.id",
    )


class ChunkLink(Base):
    """Document-structure link between chunks (NEXT_CHUNK, SAME_PAGE, etc.)."""

    __tablename__ = "chunk_links"
    __table_args__ = {"schema": "retrieval"}

    source_chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True
    )
    target_chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True
    )
    link_type: Mapped[str] = mapped_column(
        String(50), primary_key=True
    )
    hop: Mapped[int] = mapped_column(
        SmallInteger, primary_key=True, default=1
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ingest.documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.85)
    evidence: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# DEPRECATED: alias for backwards compatibility during transition
Chunk = TextChunk


# Import to resolve forward references
from app.models.ingest import Artifact, Document  # noqa: E402
