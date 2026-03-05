"""Split chunks into text_chunks + image_chunks; add memory_proposals

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-05 00:00:00.000000
"""

import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

_TEXT_DIM = int(os.environ.get("TEXT_EMBEDDING_DIM", "1024"))
_IMAGE_DIM = int(os.environ.get("IMAGE_EMBEDDING_DIM", "512"))

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # retrieval.text_chunks
    # ------------------------------------------------------------------
    op.create_table(
        "text_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "artifact_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ingest.artifacts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ingest.documents.id", ondelete="CASCADE"),
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

    # Add text embedding vector column
    op.execute(
        f"ALTER TABLE retrieval.text_chunks ADD COLUMN embedding vector({_TEXT_DIM})"
    )

    # HNSW index for text embeddings
    op.execute("""
        CREATE INDEX ix_text_chunks_embedding_hnsw
        ON retrieval.text_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        WHERE embedding IS NOT NULL
    """)
    op.create_index(
        "ix_text_chunks_artifact_id", "text_chunks", ["artifact_id"], schema="retrieval"
    )
    op.create_index(
        "ix_text_chunks_document_id", "text_chunks", ["document_id"], schema="retrieval"
    )

    # ------------------------------------------------------------------
    # retrieval.image_chunks
    # ------------------------------------------------------------------
    op.create_table(
        "image_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "artifact_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ingest.artifacts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ingest.documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=True),  # nullable for images
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

    # Add image embedding vector column
    op.execute(
        f"ALTER TABLE retrieval.image_chunks ADD COLUMN embedding vector({_IMAGE_DIM})"
    )

    # HNSW index for image embeddings
    op.execute("""
        CREATE INDEX ix_image_chunks_embedding_hnsw
        ON retrieval.image_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        WHERE embedding IS NOT NULL
    """)
    op.create_index(
        "ix_image_chunks_artifact_id", "image_chunks", ["artifact_id"], schema="retrieval"
    )
    op.create_index(
        "ix_image_chunks_document_id", "image_chunks", ["document_id"], schema="retrieval"
    )

    # ------------------------------------------------------------------
    # Migrate data from chunks → text_chunks / image_chunks
    # ------------------------------------------------------------------
    op.execute("""
        INSERT INTO retrieval.text_chunks
            (id, artifact_id, document_id, chunk_index, chunk_text, embedding,
             modality, page_number, bounding_box, classification, created_at, updated_at)
        SELECT
            c.id, c.artifact_id, a.document_id, c.chunk_index, c.chunk_text, c.embedding,
            c.modality, c.page_number, c.bounding_box, c.classification, c.created_at, c.updated_at
        FROM retrieval.chunks c
        JOIN ingest.artifacts a ON a.id = c.artifact_id
        WHERE c.embedding IS NOT NULL
    """)

    op.execute("""
        INSERT INTO retrieval.image_chunks
            (id, artifact_id, document_id, chunk_index, chunk_text, embedding,
             modality, page_number, bounding_box, classification, created_at, updated_at)
        SELECT
            c.id, c.artifact_id, a.document_id, c.chunk_index, c.chunk_text, c.image_embedding,
            c.modality, c.page_number, c.bounding_box, c.classification, c.created_at, c.updated_at
        FROM retrieval.chunks c
        JOIN ingest.artifacts a ON a.id = c.artifact_id
        WHERE c.image_embedding IS NOT NULL
    """)

    # ------------------------------------------------------------------
    # Update governance.feedback: add text_chunk_id + image_chunk_id, drop chunk_id
    # ------------------------------------------------------------------
    op.add_column(
        "feedback",
        sa.Column(
            "text_chunk_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("retrieval.text_chunks.id", ondelete="SET NULL"),
            nullable=True,
        ),
        schema="governance",
    )
    op.add_column(
        "feedback",
        sa.Column(
            "image_chunk_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("retrieval.image_chunks.id", ondelete="SET NULL"),
            nullable=True,
        ),
        schema="governance",
    )

    # Migrate existing chunk_id references to text_chunk_id
    op.execute("""
        UPDATE governance.feedback f
        SET text_chunk_id = f.chunk_id
        WHERE f.chunk_id IS NOT NULL
          AND EXISTS (SELECT 1 FROM retrieval.text_chunks tc WHERE tc.id = f.chunk_id)
    """)
    op.execute("""
        UPDATE governance.feedback f
        SET image_chunk_id = f.chunk_id
        WHERE f.chunk_id IS NOT NULL
          AND EXISTS (SELECT 1 FROM retrieval.image_chunks ic WHERE ic.id = f.chunk_id)
    """)

    # Drop old chunk_id FK constraint and column
    op.drop_constraint(
        "feedback_chunk_id_fkey", "feedback", schema="governance", type_="foreignkey"
    )
    op.drop_column("feedback", "chunk_id", schema="governance")

    # ------------------------------------------------------------------
    # Rename chunks → chunks_legacy (keep for rollback)
    # ------------------------------------------------------------------
    op.rename_table("chunks", "chunks_legacy", schema="retrieval")

    # ------------------------------------------------------------------
    # governance.memory_proposals
    # ------------------------------------------------------------------
    op.create_table(
        "memory_proposals",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_context", postgresql.JSONB(), nullable=True),
        sa.Column("proposed_by", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column(
            "status",
            sa.String(50),
            nullable=False,
            server_default="PROPOSED",
        ),
        sa.Column("reviewed_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("review_notes", sa.Text(), nullable=True),
        sa.Column(
            "reviewed_at", sa.DateTime(timezone=True), nullable=True
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
        schema="governance",
    )
    op.create_index(
        "ix_memory_proposals_status",
        "memory_proposals",
        ["status"],
        schema="governance",
    )


def downgrade() -> None:
    # Drop memory proposals
    op.drop_table("memory_proposals", schema="governance")

    # Restore chunks from legacy
    op.rename_table("chunks_legacy", "chunks", schema="retrieval")

    # Restore chunk_id on feedback
    op.add_column(
        "feedback",
        sa.Column(
            "chunk_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("retrieval.chunks.id", ondelete="SET NULL"),
            nullable=True,
        ),
        schema="governance",
    )
    # Migrate text_chunk_id back to chunk_id
    op.execute("""
        UPDATE governance.feedback
        SET chunk_id = COALESCE(text_chunk_id, image_chunk_id)
    """)
    op.drop_column("feedback", "text_chunk_id", schema="governance")
    op.drop_column("feedback", "image_chunk_id", schema="governance")

    # Drop new tables
    op.drop_table("image_chunks", schema="retrieval")
    op.drop_table("text_chunks", schema="retrieval")
