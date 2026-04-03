import uuid
from typing import Optional

from sqlalchemy import Boolean, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class QueryProfileRegistry(Base, TimestampMixin):
    """Persisted ontology/query profile registry used to drive exact graph search."""

    __tablename__ = "query_profile_registries"
    __table_args__ = {"schema": "governance"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ingest.sources.id", ondelete="SET NULL"),
        nullable=True,
    )
    ontology_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    ontology_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    ontology_definition: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    profiles: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
