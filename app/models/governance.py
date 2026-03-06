import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Feedback(Base, TimestampMixin):
    """User-submitted feedback on a retrieval result."""

    __tablename__ = "feedback"
    __table_args__ = {"schema": "governance"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # What the user was doing
    query_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    text_chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("retrieval.text_chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    image_chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("retrieval.image_chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    artifact_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ingest.artifacts.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Feedback type
    feedback_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    # WRONG_TEXT | WRONG_CLASSIFICATION | INCORRECT_ENTITY | MISSING_RELATIONSHIP
    # MISSING_ENTITY | DELETE_ENTITY | MERGE_ENTITY

    # Proposed correction
    proposed_value: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    submitted_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    patch: Mapped[Optional["Patch"]] = relationship(back_populates="source_feedback")


class Patch(Base, TimestampMixin):
    """A proposed data correction derived from user feedback."""

    __tablename__ = "patches"
    __table_args__ = {"schema": "governance"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_feedback_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("governance.feedback.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    )

    patch_type: Mapped[str] = mapped_column(String(100), nullable=False)
    # entity_update | relationship_add | relationship_delete | entity_delete
    # entity_merge | chunk_text_correction | classification_correction

    # State machine
    state: Mapped[str] = mapped_column(
        String(50), nullable=False, default="DRAFT"
    )
    # DRAFT | UNDER_REVIEW | APPROVED | DUAL_APPROVED | REJECTED | APPLIED | REVERTED

    # Determines if dual approval is required.
    # True for all graph mutations; false for relational/text corrections.
    requires_dual_approval: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # Target of the patch
    target_table: Mapped[str] = mapped_column(String(255), nullable=False)
    target_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)

    # RFC 6902 JSON Patch operations
    patch_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Snapshot of state before patch was applied (populated at APPLY time)
    previous_snapshot: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    created_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    source_feedback: Mapped["Feedback"] = relationship(back_populates="patch")
    approvals: Mapped[list["PatchApproval"]] = relationship(back_populates="patch")
    events: Mapped[list["PatchEvent"]] = relationship(back_populates="patch")


class PatchApproval(Base):
    """Individual curator decision on a patch."""

    __tablename__ = "patch_approvals"
    __table_args__ = (
        UniqueConstraint("patch_id", "curator_id", name="uq_patch_approval_curator"),
        {"schema": "governance"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patch_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("governance.patches.id", ondelete="CASCADE"),
        nullable=False,
    )
    curator_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    decision: Mapped[str] = mapped_column(String(50), nullable=False)  # approved | rejected
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    patch: Mapped["Patch"] = relationship(back_populates="approvals")


class PatchEvent(Base):
    """Immutable audit log of all patch state transitions."""

    __tablename__ = "patch_events"
    __table_args__ = {"schema": "governance"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patch_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("governance.patches.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    actor_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    event_metadata: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    patch: Mapped["Patch"] = relationship(back_populates="events")
