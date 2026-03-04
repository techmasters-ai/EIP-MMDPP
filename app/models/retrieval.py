import os
import uuid
from typing import Optional

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.models.base import Base, TimestampMixin

_TEXT_DIM = int(os.environ.get("TEXT_EMBEDDING_DIM", "1024"))
_IMAGE_DIM = int(os.environ.get("IMAGE_EMBEDDING_DIM", "512"))


class Chunk(Base, TimestampMixin):
    """A chunk of extracted content with vector embeddings for retrieval."""

    __tablename__ = "chunks"
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

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Text embedding (BGE-large: 1024-dim in prod; controlled by TEXT_EMBEDDING_DIM env var)
    embedding: Mapped[Optional[list]] = mapped_column(Vector(_TEXT_DIM), nullable=True)

    # Image/visual embedding (CLIP ViT-B/32: 512-dim; controlled by IMAGE_EMBEDDING_DIM)
    image_embedding: Mapped[Optional[list]] = mapped_column(Vector(_IMAGE_DIM), nullable=True)

    # Modality and provenance
    modality: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # text | image | table | schematic
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bounding_box: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Classification
    classification: Mapped[str] = mapped_column(
        String(100), nullable=False, default="UNCLASSIFIED"
    )

    artifact: Mapped["Artifact"] = relationship(back_populates="chunks")


# Import Artifact here to resolve circular reference
from app.models.ingest import Artifact  # noqa: E402
