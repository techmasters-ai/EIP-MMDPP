"""Unit tests for the Postgres -> GraphRAG input bridge layer."""

from unittest.mock import MagicMock, call

import pytest

pytestmark = pytest.mark.unit


class TestExportDocuments:
    def test_exports_documents_with_full_text(self):
        from app.services.graphrag_bridge import export_documents

        db = MagicMock()
        # First call: document metadata; second call: text chunks
        db.execute.return_value.fetchall.side_effect = [
            [("doc-1", "TM-123-manual.pdf")],
            [
                ("doc-1", "The S-400 system uses..."),
                ("doc-1", "Radar characteristics..."),
            ],
        ]

        df = export_documents(db)
        assert len(df) == 1
        assert "id" in df.columns
        assert "title" in df.columns
        assert "text" in df.columns
        assert df.iloc[0]["title"].startswith("TM-123-manual.pdf_")
        assert "S-400" in df.iloc[0]["text"]
        assert "Radar" in df.iloc[0]["text"]

    def test_empty_result(self):
        from app.services.graphrag_bridge import export_documents

        db = MagicMock()
        db.execute.return_value.fetchall.side_effect = [
            [],   # no documents
            [],   # no chunks
        ]

        df = export_documents(db)
        assert len(df) == 0
        assert "id" in df.columns

    def test_exception_returns_empty_df(self):
        from app.services.graphrag_bridge import export_documents

        db = MagicMock()
        db.execute.side_effect = Exception("connection failed")

        df = export_documents(db)
        assert len(df) == 0


class TestExportAll:
    def test_writes_csv_file(self, tmp_path):
        from app.services.graphrag_bridge import export_all

        db = MagicMock()
        db.execute.return_value.fetchall.side_effect = [
            [("doc-1", "Doc 1")],
            [("doc-1", "Some text content")],
        ]

        stats = export_all(db, tmp_path)

        assert (tmp_path / "documents.csv").exists()
        assert stats["documents"] == 1

    def test_empty_db_returns_zero(self, tmp_path):
        from app.services.graphrag_bridge import export_all

        db = MagicMock()
        db.execute.return_value.fetchall.side_effect = [
            [],
            [],
        ]

        stats = export_all(db, tmp_path)

        assert stats["documents"] == 0
