"""Ingest v2 foundation: pipeline runs, stage runs, document elements,
graph extractions, and retrieval chunk links.

All tables are additive — no existing tables modified.

Revision ID: 0004
Revises: 0003
Create Date: 2026-03-05
"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- Pipeline run / stage tracking ---
    op.execute("""
        CREATE TABLE ingest.pipeline_runs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES ingest.documents(id) ON DELETE CASCADE,
            pipeline_version TEXT NOT NULL DEFAULT 'v2',
            status TEXT NOT NULL DEFAULT 'PROCESSING',
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            finished_at TIMESTAMPTZ,
            error_message TEXT
        )
    """)
    op.execute("""
        CREATE INDEX idx_pipeline_runs_doc
            ON ingest.pipeline_runs (document_id, started_at DESC)
    """)

    op.execute("""
        CREATE TABLE ingest.stage_runs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            pipeline_run_id UUID NOT NULL REFERENCES ingest.pipeline_runs(id) ON DELETE CASCADE,
            stage_name TEXT NOT NULL,
            attempt INT NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'PENDING',
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            metrics JSONB DEFAULT '{}',
            error_message TEXT,
            UNIQUE (pipeline_run_id, stage_name, attempt)
        )
    """)

    # --- Canonical document elements ---
    op.execute("""
        CREATE TABLE ingest.document_elements (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES ingest.documents(id) ON DELETE CASCADE,
            artifact_id UUID REFERENCES ingest.artifacts(id) ON DELETE SET NULL,
            element_uid TEXT NOT NULL,
            element_type TEXT NOT NULL,
            element_order INT NOT NULL,
            page_number INT,
            bounding_box JSONB,
            section_path TEXT,
            heading_level INT,
            parent_element_uid TEXT,
            content_text TEXT,
            storage_bucket TEXT,
            storage_key TEXT,
            metadata JSONB DEFAULT '{}',
            element_hash TEXT,
            UNIQUE (document_id, element_uid)
        )
    """)
    op.execute("""
        CREATE INDEX idx_doc_elements_order
            ON ingest.document_elements (document_id, element_order)
    """)
    op.execute("""
        CREATE INDEX idx_doc_elements_page
            ON ingest.document_elements (document_id, page_number)
    """)
    op.execute("""
        CREATE INDEX idx_doc_elements_type
            ON ingest.document_elements (document_id, element_type)
    """)

    # --- Graph extraction (one per document) ---
    op.execute("""
        CREATE TABLE ingest.document_graph_extractions (
            document_id UUID PRIMARY KEY REFERENCES ingest.documents(id) ON DELETE CASCADE,
            provider TEXT,
            model_name TEXT,
            extraction_version TEXT,
            graph_json JSONB,
            status TEXT NOT NULL DEFAULT 'PENDING',
            metrics JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # --- Retrieval chunk links ---
    op.execute("""
        CREATE TABLE retrieval.chunk_links (
            source_chunk_id UUID NOT NULL,
            target_chunk_id UUID NOT NULL,
            document_id UUID NOT NULL REFERENCES ingest.documents(id) ON DELETE CASCADE,
            link_type TEXT NOT NULL,
            hop SMALLINT NOT NULL DEFAULT 1,
            weight REAL NOT NULL DEFAULT 0.85,
            evidence JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (source_chunk_id, target_chunk_id, link_type, hop)
        )
    """)
    op.execute("""
        CREATE INDEX idx_chunk_links_source
            ON retrieval.chunk_links (source_chunk_id)
    """)
    op.execute("""
        CREATE INDEX idx_chunk_links_target
            ON retrieval.chunk_links (target_chunk_id)
    """)
    op.execute("""
        CREATE INDEX idx_chunk_links_doc_type
            ON retrieval.chunk_links (document_id, link_type)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS retrieval.chunk_links CASCADE")
    op.execute("DROP TABLE IF EXISTS ingest.document_graph_extractions CASCADE")
    op.execute("DROP TABLE IF EXISTS ingest.document_elements CASCADE")
    op.execute("DROP TABLE IF EXISTS ingest.stage_runs CASCADE")
    op.execute("DROP TABLE IF EXISTS ingest.pipeline_runs CASCADE")
