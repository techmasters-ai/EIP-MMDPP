import uuid
from typing import Optional

from sqlalchemy import Boolean, ForeignKey, Index, String, Text
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


class QueryProfile(Base, TimestampMixin):
    """First-class query profile row.

    Replaces the ``QueryProfileRegistry.profiles`` JSONB list — each row here
    is one profile (was previously one element of that list). ``profile_key``
    is the stable string identifier that was previously
    ``QueryProfileDefinition.id`` — dossier ``section_profile_ids`` and
    search ``profile_id`` reference this key, not the internal uuid ``id``.

    ``definition`` holds the remaining profile body not promoted to its own
    column: ``target_entity_types``, ``traversals``, ``section_profile_ids``,
    ``profile_sections``, ``profile_subgroup``, ``include_associated_systems``,
    and ``placeholder_query`` (a live per-profile UI field).
    """

    __tablename__ = "query_profiles"
    __table_args__ = (
        Index("ix_query_profiles_source_id", "source_id"),
        Index("ix_query_profiles_enabled", "enabled"),
        {"schema": "governance"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    profile_key: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    root_entity_types: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    definition: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    source_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ingest.sources.id", ondelete="SET NULL"),
        nullable=True,
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
