"""Initial schemas: auth, ingest, retrieval, governance, audit

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00.000000
"""

import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# Read embedding dimensions from environment so the test stack (which uses a
# smaller model) creates correctly-sized vector columns.
_TEXT_DIM = int(os.environ.get("TEXT_EMBEDDING_DIM", "1024"))
_IMAGE_DIM = int(os.environ.get("IMAGE_EMBEDDING_DIM", "512"))

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Create schemas
    # ------------------------------------------------------------------
    op.execute("CREATE SCHEMA IF NOT EXISTS auth")
    op.execute("CREATE SCHEMA IF NOT EXISTS ingest")
    op.execute("CREATE SCHEMA IF NOT EXISTS retrieval")
    op.execute("CREATE SCHEMA IF NOT EXISTS governance")
    op.execute("CREATE SCHEMA IF NOT EXISTS audit")
    op.execute("CREATE SCHEMA IF NOT EXISTS ontology")

    # ------------------------------------------------------------------
    # Ensure AGE is available and knowledge graph exists
    # ------------------------------------------------------------------
    op.execute("LOAD 'age'")
    op.execute("SET search_path = ag_catalog, \"$user\", public")
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'kg'
            ) THEN
                PERFORM create_graph('kg');
            END IF;
        END $$;
    """)

    # ------------------------------------------------------------------
    # auth.users
    # ------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("username", sa.String(255), nullable=False, unique=True),
        sa.Column("email", sa.String(512), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(1024), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "classification_level", sa.String(100), nullable=False, server_default="UNCLASSIFIED"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="auth",
    )

    op.create_table(
        "user_roles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("auth.users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(50), nullable=False),
        sa.Column(
            "assigned_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("user_id", "role", name="uq_user_role"),
        schema="auth",
    )

    # ------------------------------------------------------------------
    # ingest.sources
    # ------------------------------------------------------------------
    op.create_table(
        "sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="ingest",
    )

    # ------------------------------------------------------------------
    # ingest.documents
    # ------------------------------------------------------------------
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "source_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ingest.sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(1024), nullable=False),
        sa.Column("mime_type", sa.String(255), nullable=True),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("file_hash", sa.String(64), nullable=True),
        sa.Column("storage_bucket", sa.String(255), nullable=False),
        sa.Column("storage_key", sa.String(2048), nullable=False),
        sa.Column(
            "pipeline_status", sa.String(50), nullable=False, server_default="PENDING"
        ),
        sa.Column("pipeline_stage", sa.String(100), nullable=True),
        sa.Column(
            "failed_stages", postgresql.ARRAY(sa.String()), nullable=True
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column("uploaded_by", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="ingest",
    )
    op.create_index(
        "ix_documents_source_id", "documents", ["source_id"], schema="ingest"
    )
    op.create_index(
        "ix_documents_pipeline_status", "documents", ["pipeline_status"], schema="ingest"
    )

    # ------------------------------------------------------------------
    # ingest.artifacts
    # ------------------------------------------------------------------
    op.create_table(
        "artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ingest.documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("artifact_type", sa.String(50), nullable=False),
        sa.Column("content_text", sa.Text(), nullable=True),
        sa.Column("content_metadata", postgresql.JSONB(), nullable=True),
        sa.Column("storage_bucket", sa.String(255), nullable=True),
        sa.Column("storage_key", sa.String(2048), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("bounding_box", postgresql.JSONB(), nullable=True),
        sa.Column("ocr_confidence", sa.Float(), nullable=True),
        sa.Column("ocr_engine", sa.String(50), nullable=True),
        sa.Column(
            "requires_human_review", sa.Boolean(), nullable=False, server_default="false"
        ),
        sa.Column(
            "classification", sa.String(100), nullable=False, server_default="UNCLASSIFIED"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="ingest",
    )
    op.create_index(
        "ix_artifacts_document_id", "artifacts", ["document_id"], schema="ingest"
    )

    # ------------------------------------------------------------------
    # ingest.watch_dirs
    # ------------------------------------------------------------------
    op.create_table(
        "watch_dirs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "source_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ingest.sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("path", sa.Text(), nullable=False, unique=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("poll_interval_seconds", sa.Integer(), nullable=False, server_default="30"),
        sa.Column(
            "file_patterns",
            postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default="{*.pdf,*.docx,*.png,*.jpg,*.tiff}",
        ),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="ingest",
    )

    # ------------------------------------------------------------------
    # ingest.watch_logs
    # ------------------------------------------------------------------
    op.create_table(
        "watch_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "watch_dir_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ingest.watch_dirs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ingest.documents.id"),
            nullable=True,
        ),
        sa.Column("status", sa.String(50), nullable=False, server_default="enqueued"),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("watch_dir_id", "file_hash", name="uq_watch_log_dir_hash"),
        schema="ingest",
    )

    # ------------------------------------------------------------------
    # retrieval.chunks  (with pgvector columns)
    # ------------------------------------------------------------------
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "artifact_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ingest.artifacts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("modality", sa.String(50), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("bounding_box", postgresql.JSONB(), nullable=True),
        sa.Column(
            "classification", sa.String(100), nullable=False, server_default="UNCLASSIFIED"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="retrieval",
    )

    # Add vector columns via raw SQL (pgvector types).
    # Dimensions are read from env vars so test and prod stacks can differ.
    op.execute(
        f"ALTER TABLE retrieval.chunks ADD COLUMN embedding vector({_TEXT_DIM})"
    )
    op.execute(
        f"ALTER TABLE retrieval.chunks ADD COLUMN image_embedding vector({_IMAGE_DIM})"
    )

    # HNSW indices for fast approximate nearest neighbour search
    op.execute("""
        CREATE INDEX ix_chunks_embedding_hnsw
        ON retrieval.chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        WHERE embedding IS NOT NULL
    """)
    op.execute("""
        CREATE INDEX ix_chunks_image_embedding_hnsw
        ON retrieval.chunks
        USING hnsw (image_embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        WHERE image_embedding IS NOT NULL
    """)
    op.create_index(
        "ix_chunks_artifact_id", "chunks", ["artifact_id"], schema="retrieval"
    )

    # ------------------------------------------------------------------
    # governance.feedback
    # ------------------------------------------------------------------
    op.create_table(
        "feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("query_text", sa.Text(), nullable=True),
        sa.Column(
            "chunk_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("retrieval.chunks.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "artifact_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ingest.artifacts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("feedback_type", sa.String(50), nullable=False),
        sa.Column("proposed_value", postgresql.JSONB(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("submitted_by", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="governance",
    )

    # ------------------------------------------------------------------
    # governance.patches
    # ------------------------------------------------------------------
    op.create_table(
        "patches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "source_feedback_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("governance.feedback.id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column("patch_type", sa.String(100), nullable=False),
        sa.Column("state", sa.String(50), nullable=False, server_default="DRAFT"),
        sa.Column(
            "requires_dual_approval", sa.Boolean(), nullable=False, server_default="false"
        ),
        sa.Column("target_table", sa.String(255), nullable=False),
        sa.Column("target_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("patch_payload", postgresql.JSONB(), nullable=False),
        sa.Column("previous_snapshot", postgresql.JSONB(), nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="governance",
    )
    op.create_index(
        "ix_patches_state", "patches", ["state"], schema="governance"
    )

    # ------------------------------------------------------------------
    # governance.patch_approvals
    # ------------------------------------------------------------------
    op.create_table(
        "patch_approvals",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "patch_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("governance.patches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("curator_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("decision", sa.String(50), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "decided_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("patch_id", "curator_id", name="uq_patch_approval_curator"),
        schema="governance",
    )

    # ------------------------------------------------------------------
    # governance.patch_events (immutable audit log)
    # ------------------------------------------------------------------
    op.create_table(
        "patch_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "patch_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("governance.patches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("actor_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="governance",
    )
    op.create_index(
        "ix_patch_events_patch_id", "patch_events", ["patch_id"], schema="governance"
    )

    # ------------------------------------------------------------------
    # ontology.versions, entity_types, relationship_types
    # ------------------------------------------------------------------
    op.create_table(
        "versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("version_number", sa.String(50), nullable=False, unique=True),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="ontology",
    )

    op.create_table(
        "entity_types",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("label", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "parent_type_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ontology.entity_types.id"),
            nullable=True,
        ),
        sa.Column("properties", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("is_abstract", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="ontology",
    )

    op.create_table(
        "relationship_types",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("label", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "source_type_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ontology.entity_types.id"),
            nullable=True,
        ),
        sa.Column(
            "target_type_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ontology.entity_types.id"),
            nullable=True,
        ),
        sa.Column(
            "cardinality", sa.String(50), nullable=False, server_default="many_to_many"
        ),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="ontology",
    )


def downgrade() -> None:
    op.drop_table("relationship_types", schema="ontology")
    op.drop_table("entity_types", schema="ontology")
    op.drop_table("versions", schema="ontology")
    op.drop_table("patch_events", schema="governance")
    op.drop_table("patch_approvals", schema="governance")
    op.drop_table("patches", schema="governance")
    op.drop_table("feedback", schema="governance")
    op.drop_table("chunks", schema="retrieval")
    op.drop_table("watch_logs", schema="ingest")
    op.drop_table("watch_dirs", schema="ingest")
    op.drop_table("artifacts", schema="ingest")
    op.drop_table("documents", schema="ingest")
    op.drop_table("sources", schema="ingest")
    op.drop_table("user_roles", schema="auth")
    op.drop_table("users", schema="auth")
    op.execute("DROP SCHEMA IF EXISTS ontology CASCADE")
    op.execute("DROP SCHEMA IF EXISTS governance CASCADE")
    op.execute("DROP SCHEMA IF EXISTS retrieval CASCADE")
    op.execute("DROP SCHEMA IF EXISTS ingest CASCADE")
    op.execute("DROP SCHEMA IF EXISTS auth CASCADE")
    op.execute("DROP SCHEMA IF EXISTS audit CASCADE")
