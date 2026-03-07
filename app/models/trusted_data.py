import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class TrustedDataSubmission(Base, TimestampMixin):
    """Human-reviewed trusted data for vector-indexed retrieval.

    State machine:
        PROPOSED → APPROVED_PENDING_INDEX → APPROVED_INDEXED
                                          → INDEX_FAILED → (reindex) → APPROVED_PENDING_INDEX
        PROPOSED → REJECTED
    """

    __tablename__ = "trusted_data_submissions"
    __table_args__ = {"schema": "governance"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_context: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    proposed_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    # State machine
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="PROPOSED"
    )  # PROPOSED | APPROVED_PENDING_INDEX | APPROVED_INDEXED | INDEX_FAILED | REJECTED

    reviewed_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    review_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Indexing lifecycle
    index_status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    index_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    qdrant_point_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    embedded_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
