"""Unit tests for the Neo4j -> GraphRAG Parquet bridge layer."""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


def _mock_neo4j_records(data: list[dict]):
    """Create a mock Neo4j result that iterates over records."""
    result = MagicMock()
    records = []
    for row in data:
        rec = MagicMock()
        rec.data.return_value = row
        rec.__getitem__ = lambda self, key, r=row: r[key]
        records.append(rec)
    result.__iter__ = MagicMock(return_value=iter(records))
    return result


class TestExportEntities:
    def test_exports_entity_dataframe(self, mock_neo4j_driver):
        from app.services.graphrag_bridge import export_entities

        driver, session = mock_neo4j_driver
        session.run.return_value = _mock_neo4j_records([
            {
                "name": "S-400 Triumf",
                "entity_type": "MISSILE_SYSTEM",
                "description": "Russian long-range SAM system",
                "id": "s400-1",
            },
            {
                "name": "91N6E",
                "entity_type": "RADAR_SYSTEM",
                "description": "Battle management radar for S-400",
                "id": "91n6e-1",
            },
        ])

        df = export_entities(driver)
        assert len(df) == 2
        assert "id" in df.columns
        assert "title" in df.columns
        assert "type" in df.columns
        assert "description" in df.columns
        assert "human_readable_id" in df.columns
        assert df.iloc[0]["title"] == "S-400 Triumf"
        assert df.iloc[0]["type"] == "MISSILE_SYSTEM"

    def test_empty_graph_returns_empty_df(self, mock_neo4j_driver):
        from app.services.graphrag_bridge import export_entities

        driver, session = mock_neo4j_driver
        session.run.return_value = _mock_neo4j_records([])

        df = export_entities(driver)
        assert len(df) == 0
        assert "id" in df.columns

    def test_exception_returns_empty_df(self, mock_neo4j_driver):
        from app.services.graphrag_bridge import export_entities

        driver, session = mock_neo4j_driver
        session.run.side_effect = Exception("connection failed")

        df = export_entities(driver)
        assert len(df) == 0


class TestExportRelationships:
    def test_exports_relationship_dataframe(self, mock_neo4j_driver):
        from app.services.graphrag_bridge import export_relationships

        driver, session = mock_neo4j_driver
        session.run.return_value = _mock_neo4j_records([
            {
                "source": "S-400 Triumf",
                "target": "91N6E",
                "relationship": "CONTAINS",
                "description": "S-400 contains 91N6E radar",
            },
        ])

        df = export_relationships(driver)
        assert len(df) == 1
        assert "id" in df.columns
        assert "source" in df.columns
        assert "target" in df.columns
        assert "weight" in df.columns
        assert "description" in df.columns
        assert df.iloc[0]["weight"] == 0.90  # CONTAINS weight from ontology

    def test_default_weight_for_unknown_relationship(self, mock_neo4j_driver):
        from app.services.graphrag_bridge import export_relationships

        driver, session = mock_neo4j_driver
        session.run.return_value = _mock_neo4j_records([
            {
                "source": "A",
                "target": "B",
                "relationship": "UNKNOWN_REL",
                "description": "",
            },
        ])

        df = export_relationships(driver)
        assert df.iloc[0]["weight"] == 0.70  # default weight


class TestExportTextUnits:
    def test_exports_text_units(self):
        from app.services.graphrag_bridge import export_text_units

        db = MagicMock()
        db.execute.return_value.fetchall.return_value = [
            ("chunk-1", "The S-400 system uses...", "doc-1"),
            ("chunk-2", "Radar characteristics...", "doc-1"),
        ]

        df = export_text_units(db)
        assert len(df) == 2
        assert "id" in df.columns
        assert "text" in df.columns
        assert "document_ids" in df.columns
        assert "entity_ids" in df.columns

    def test_empty_result(self):
        from app.services.graphrag_bridge import export_text_units

        db = MagicMock()
        db.execute.return_value.fetchall.return_value = []

        df = export_text_units(db)
        assert len(df) == 0
        assert "id" in df.columns


class TestExportDocuments:
    def test_exports_documents(self):
        from app.services.graphrag_bridge import export_documents

        db = MagicMock()
        db.execute.return_value.fetchall.return_value = [
            ("doc-1", "TM-123-manual.pdf"),
        ]

        df = export_documents(db)
        assert len(df) == 1
        assert "id" in df.columns
        assert "title" in df.columns


class TestExportAll:
    def test_writes_parquet_files(self, tmp_path, mock_neo4j_driver):
        from app.services.graphrag_bridge import export_all

        driver, session = mock_neo4j_driver

        # Mock entity query, then relationship query
        session.run.side_effect = [
            _mock_neo4j_records([
                {"name": "A", "entity_type": "PLATFORM", "description": "test", "id": "a1"},
            ]),
            _mock_neo4j_records([
                {"source": "A", "target": "B", "relationship": "CONTAINS", "description": ""},
            ]),
        ]

        db = MagicMock()
        # text_units query, then documents query
        db.execute.return_value.fetchall.side_effect = [
            [("c1", "text", "d1")],  # text_units
            [("d1", "Doc 1")],       # documents
        ]

        export_all(driver, db, tmp_path)

        assert (tmp_path / "entities.parquet").exists()
        assert (tmp_path / "relationships.parquet").exists()
        assert (tmp_path / "text_units.parquet").exists()
        assert (tmp_path / "documents.parquet").exists()
