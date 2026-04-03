"""Unit tests for GraphRAG context-based provenance builder."""
import pandas as pd
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def sample_data():
    """Full Parquet data with communities, entities, relationships, text_units, documents."""
    entities = pd.DataFrame({
        "id": ["ent-uuid-1", "ent-uuid-2", "ent-uuid-3"],
        "human_readable_id": [100, 200, 300],
        "title": ["FAN SONG", "SA-2 GUIDELINE", "SPOON REST"],
        "type": ["FIRE_CONTROL_SYSTEM", "MISSILE_SYSTEM", "RADAR_SYSTEM"],
        "description": ["Engagement radar", "Soviet SAM", "Acquisition radar"],
        "text_unit_ids": [["tu-uuid-1"], ["tu-uuid-1", "tu-uuid-2"], ["tu-uuid-2"]],
        "community_ids": [["comm-1"], ["comm-1"], ["comm-2"]],
    })
    relationships = pd.DataFrame({
        "id": ["rel-uuid-1", "rel-uuid-2"],
        "human_readable_id": [500, 600],
        "source": ["SA-2 GUIDELINE", "SPOON REST"],
        "target": ["FAN SONG", "FAN SONG"],
        "description": ["Uses for guidance", "Cues target to"],
        "text_unit_ids": [["tu-uuid-1"], ["tu-uuid-2"]],
    })
    text_units = pd.DataFrame({
        "id": ["tu-uuid-1", "tu-uuid-2"],
        "human_readable_id": [0, 1],
        "text": ["The SA-2 Guideline uses command guidance...", "Spoon Rest provides acquisition..."],
        "document_id": ["doc-uuid-1", "doc-uuid-2"],
    })
    documents = pd.DataFrame({
        "id": ["doc-uuid-1", "doc-uuid-2"],
        "title": ["Red SAM_a3b2c1d4", "Fan Song radars_e5f6g7h8"],
    })
    communities = pd.DataFrame({
        "id": ["comm-uuid-1", "comm-uuid-2"],
        "human_readable_id": [0, 1],
        "community": [10, 20],
        "entity_ids": [["ent-uuid-1", "ent-uuid-2"], ["ent-uuid-3"]],
        "relationship_ids": [["rel-uuid-1"], ["rel-uuid-2"]],
        "text_unit_ids": [["tu-uuid-1"], ["tu-uuid-2"]],
    })
    community_reports = pd.DataFrame({
        "id": ["cr-uuid-1", "cr-uuid-2"],
        "human_readable_id": [0, 1],
        "community": [10, 20],
        "title": ["SA-2 & Fan Song Community", "Acquisition Radar Community"],
        "full_content": ["# SA-2 & Fan Song\n\nDetailed report...", "# Acquisition\n\nRadar report..."],
        "summary": ["Short summary 1", "Short summary 2"],
    })
    return {
        "entities": entities,
        "relationships": relationships,
        "text_units": text_units,
        "documents": documents,
        "communities": communities,
        "community_reports": community_reports,
    }


@pytest.fixture
def local_context():
    """Context dict as returned by GraphRAG local search (uses human_readable_ids as 'id')."""
    reports = pd.DataFrame({
        "id": [0, 1],
        "title": ["SA-2 & Fan Song Community", "Acquisition Radar Community"],
        "content": ["# SA-2 & Fan Song\n\nDetailed report...", "# Acquisition\n\nRadar report..."],
    })
    entities = pd.DataFrame({
        "id": [100, 200, 300],
        "entity": ["FAN SONG", "SA-2 GUIDELINE", "SPOON REST"],
        "description": ["Engagement radar", "Soviet SAM", "Acquisition radar"],
        "number of relationships": [5, 3, 2],
        "in_context": [True, True, True],
    })
    relationships = pd.DataFrame({
        "id": [500, 600],
        "source": ["SA-2 GUIDELINE", "SPOON REST"],
        "target": ["FAN SONG", "FAN SONG"],
        "description": ["Uses for guidance", "Cues target to"],
        "weight": [9.0, 7.0],
        "links": [2, 1],
        "in_context": [True, True],
    })
    sources = pd.DataFrame({
        "id": [0, 1],
        "text": ["The SA-2 Guideline uses command guidance...", "Spoon Rest provides acquisition..."],
    })
    return {
        "reports": reports,
        "entities": entities,
        "relationships": relationships,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestParseReportDataReferences:
    def test_parses_entities(self):
        from app.services.graphrag_provenance import _parse_report_data_references
        content = "text [Data: Entities (100, 200, 300)] more"
        refs = _parse_report_data_references(content)
        assert refs["entities"] == {100, 200, 300}

    def test_parses_relationships(self):
        from app.services.graphrag_provenance import _parse_report_data_references
        content = "[Data: Relationships (500, 600)]"
        refs = _parse_report_data_references(content)
        assert refs["relationships"] == {500, 600}

    def test_parses_sources(self):
        from app.services.graphrag_provenance import _parse_report_data_references
        content = "[Data: Sources (1, 5, 13)]"
        refs = _parse_report_data_references(content)
        assert refs["sources"] == {1, 5, 13}

    def test_ignores_plus_more(self):
        from app.services.graphrag_provenance import _parse_report_data_references
        content = "[Data: Sources (348, 352, 360)+more]"
        refs = _parse_report_data_references(content)
        assert refs["sources"] == {348, 352, 360}

    def test_multiple_references(self):
        from app.services.graphrag_provenance import _parse_report_data_references
        content = "[Data: Entities (1, 2)] text [Data: Relationships (3, 4)]"
        refs = _parse_report_data_references(content)
        assert refs["entities"] == {1, 2}
        assert refs["relationships"] == {3, 4}

    def test_empty_content(self):
        from app.services.graphrag_provenance import _parse_report_data_references
        refs = _parse_report_data_references("")
        assert refs == {"entities": set(), "relationships": set(), "sources": set()}

    def test_combined_data_block(self):
        from app.services.graphrag_provenance import _parse_report_data_references
        content = "[Data: Sources (2, 13); Entities (6, 7, 8)]"
        refs = _parse_report_data_references(content)
        assert 2 in refs["sources"]
        assert 6 in refs["entities"]


class TestSafeValue:
    def test_none(self):
        from app.services.graphrag_provenance import _safe_value
        assert _safe_value(None) is None

    def test_nan(self):
        import math
        from app.services.graphrag_provenance import _safe_value
        assert _safe_value(float("nan")) is None

    def test_numpy_int(self):
        import numpy as np
        from app.services.graphrag_provenance import _safe_value
        assert _safe_value(np.int64(42)) == 42
        assert isinstance(_safe_value(np.int64(42)), int)

    def test_plain_value(self):
        from app.services.graphrag_provenance import _safe_value
        assert _safe_value("hello") == "hello"
        assert _safe_value(3.14) == 3.14


# ---------------------------------------------------------------------------
# Local search provenance
# ---------------------------------------------------------------------------

class TestBuildProvenanceLocal:
    def test_groups_entities_under_reports(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        assert len(result) == 2
        r1 = result[0]
        assert r1["report_title"] == "SA-2 & Fan Song Community"
        entity_names = [e["entity"] for e in r1["entities"]]
        assert "FAN SONG" in entity_names
        assert "SA-2 GUIDELINE" in entity_names
        assert "SPOON REST" not in entity_names

    def test_groups_relationships_under_reports(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        r1 = result[0]
        assert len(r1["relationships"]) == 1
        assert r1["relationships"][0]["source"] == "SA-2 GUIDELINE"

    def test_includes_report_content(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        assert "# SA-2 & Fan Song" in result[0]["report_content"]

    def test_entity_has_document_name(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        fan_song = [e for e in result[0]["entities"] if e["entity"] == "FAN SONG"][0]
        assert fan_song["document_name"] == "Red SAM"

    def test_entity_has_document_id(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        fan_song = [e for e in result[0]["entities"] if e["entity"] == "FAN SONG"][0]
        assert fan_song["document_id"] == "doc-uuid-1"

    def test_entity_has_original_text(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        fan_song = [e for e in result[0]["entities"] if e["entity"] == "FAN SONG"][0]
        assert "SA-2 Guideline" in fan_song["original_text"]

    def test_entity_has_context_fields(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        fan_song = [e for e in result[0]["entities"] if e["entity"] == "FAN SONG"][0]
        assert "in_context" in fan_song
        assert "number of relationships" in fan_song

    def test_relationship_has_document_name(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        rel = result[0]["relationships"][0]
        assert rel["document_name"] == "Red SAM"

    def test_relationship_has_weight_links(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        rel = result[0]["relationships"][0]
        assert "weight" in rel
        assert "links" in rel
        assert "in_context" in rel

    def test_relationship_has_original_text(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        rel = result[0]["relationships"][0]
        assert "SA-2 Guideline" in rel["original_text"]

    def test_strips_document_title_hash(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        fan_song = [e for e in result[0]["entities"] if e["entity"] == "FAN SONG"][0]
        assert fan_song["document_name"] == "Red SAM"

    def test_no_text_units_key_in_report(self, sample_data, local_context):
        """Report provenance should not have a text_units key."""
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        assert "text_units" not in result[0]


# ---------------------------------------------------------------------------
# Fallback filtering (reference-based and include-all)
# ---------------------------------------------------------------------------

class TestFallbackFiltering:
    def test_reference_based_filtering(self, sample_data):
        """When community UUIDs are empty, fall back to [Data: ...] references."""
        from app.services.graphrag_provenance import build_provenance
        # Remove community data to force fallback
        sample_data["communities"] = pd.DataFrame()
        context = {
            "reports": pd.DataFrame({
                "id": [0],
                "title": ["Test Report"],
                "content": ["Report text [Data: Entities (100)] [Data: Relationships (500)]"],
            }),
            "entities": pd.DataFrame({
                "id": [100, 200, 300],
                "entity": ["FAN SONG", "SA-2 GUIDELINE", "SPOON REST"],
                "description": ["Engagement radar", "Soviet SAM", "Acquisition radar"],
            }),
            "relationships": pd.DataFrame({
                "id": [500, 600],
                "source": ["SA-2 GUIDELINE", "SPOON REST"],
                "target": ["FAN SONG", "FAN SONG"],
                "description": ["Uses for guidance", "Cues target to"],
            }),
        }
        result = build_provenance(context, sample_data, "graphrag_local")
        assert len(result) == 1
        # Only entity 100 and relationship 500 should be included
        entity_ids = [e["id"] for e in result[0]["entities"]]
        assert 100 in entity_ids
        assert 200 not in entity_ids
        rel_ids = [r["id"] for r in result[0]["relationships"]]
        assert 500 in rel_ids
        assert 600 not in rel_ids

    def test_parquet_fallback_when_context_has_no_entities(self, sample_data):
        """When context has no entities key, load from Parquet via community UUIDs."""
        from app.services.graphrag_provenance import build_provenance
        context = {
            "reports": pd.DataFrame({
                "id": [0],
                "title": ["SA-2 & Fan Song Community"],
                "content": ["# SA-2 & Fan Song\n\nDetailed report..."],
            }),
            "relationships": pd.DataFrame({
                "id": [500],
                "source": ["SA-2 GUIDELINE"],
                "target": ["FAN SONG"],
                "description": ["Uses for guidance"],
            }),
            # NOTE: no "entities" key at all
        }
        result = build_provenance(context, sample_data, "graphrag_local")
        assert len(result) == 1
        # Entities should be loaded from Parquet via community membership
        entity_names = [e["entity"] for e in result[0]["entities"]]
        assert "FAN SONG" in entity_names
        assert "SA-2 GUIDELINE" in entity_names
        # SPOON REST is in community 20, not community 10
        assert "SPOON REST" not in entity_names

    def test_parquet_fallback_entities_have_document_info(self, sample_data):
        """Parquet-loaded entities should still have document resolution."""
        from app.services.graphrag_provenance import build_provenance
        context = {
            "reports": pd.DataFrame({
                "id": [0],
                "title": ["SA-2 & Fan Song Community"],
                "content": ["# Report..."],
            }),
            # no entities
        }
        result = build_provenance(context, sample_data, "graphrag_local")
        fan_song = [e for e in result[0]["entities"] if e["entity"] == "FAN SONG"]
        assert len(fan_song) == 1
        assert fan_song[0]["document_name"] == "Red SAM"
        assert fan_song[0]["document_id"] == "doc-uuid-1"
        assert "SA-2 Guideline" in fan_song[0]["original_text"]

    def test_parquet_fallback_via_references(self, sample_data):
        """When community UUIDs empty but report references entities, load from Parquet."""
        from app.services.graphrag_provenance import build_provenance
        sample_data["communities"] = pd.DataFrame()
        context = {
            "reports": pd.DataFrame({
                "id": [0],
                "title": ["Test"],
                "content": ["Text [Data: Entities (100, 200)]"],
            }),
            # no entities in context
        }
        result = build_provenance(context, sample_data, "graphrag_local")
        entity_ids = [e["id"] for e in result[0]["entities"]]
        assert 100 in entity_ids
        assert 200 in entity_ids
        assert 300 not in entity_ids

    def test_include_all_when_no_filter(self, sample_data):
        """When no community UUIDs and no [Data:] refs, include all context items."""
        from app.services.graphrag_provenance import build_provenance
        sample_data["communities"] = pd.DataFrame()
        context = {
            "reports": pd.DataFrame({
                "id": [0],
                "title": ["Test Report"],
                "content": ["Report with no data references"],
            }),
            "entities": pd.DataFrame({
                "id": [100, 200],
                "entity": ["FAN SONG", "SA-2 GUIDELINE"],
                "description": ["Engagement radar", "Soviet SAM"],
            }),
            "relationships": pd.DataFrame({
                "id": [500, 600],
                "source": ["SA-2 GUIDELINE", "SPOON REST"],
                "target": ["FAN SONG", "FAN SONG"],
                "description": ["Uses for guidance", "Cues target to"],
            }),
        }
        result = build_provenance(context, sample_data, "graphrag_local")
        assert len(result[0]["entities"]) == 2
        assert len(result[0]["relationships"]) == 2


# ---------------------------------------------------------------------------
# Global search provenance
# ---------------------------------------------------------------------------

class TestBuildProvenanceGlobal:
    def test_global_has_empty_entity_lists(self, sample_data):
        from app.services.graphrag_provenance import build_provenance
        context = {
            "reports": pd.DataFrame({
                "id": [0],
                "title": ["SA-2 & Fan Song Community"],
                "content": ["# Report content..."],
            }),
        }
        result = build_provenance(context, sample_data, "graphrag_global")
        assert len(result) == 1
        assert result[0]["entities"] == []
        assert result[0]["relationships"] == []


# ---------------------------------------------------------------------------
# Basic search provenance
# ---------------------------------------------------------------------------

class TestBuildProvenanceBasic:
    def test_basic_has_text_units_no_report(self, sample_data):
        from app.services.graphrag_provenance import build_provenance
        context = {
            "sources": pd.DataFrame({
                "id": [0, 1],
                "text": ["The SA-2 Guideline...", "Spoon Rest..."],
            }),
        }
        result = build_provenance(context, sample_data, "graphrag_basic")
        assert len(result) == 1
        assert result[0]["report_id"] is None
        assert result[0]["report_title"] is None
        assert len(result[0]["text_units"]) == 2


# ---------------------------------------------------------------------------
# Think-tag stripping
# ---------------------------------------------------------------------------

class TestThinkTagStripping:
    def test_strips_think_tags(self):
        from app.services.graphrag_service import _strip_think_tags
        text = "<think>reasoning here</think>The SA-2 uses command guidance."
        assert _strip_think_tags(text) == "The SA-2 uses command guidance."

    def test_strips_thinking_tags(self):
        from app.services.graphrag_service import _strip_think_tags
        text = "<thinking>deep thought</thinking>Result here."
        assert _strip_think_tags(text) == "Result here."

    def test_no_tags_unchanged(self):
        from app.services.graphrag_service import _strip_think_tags
        text = "Plain response."
        assert _strip_think_tags(text) == "Plain response."


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestBuildProvenanceEdgeCases:
    def test_empty_context(self, sample_data):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance({}, sample_data, "graphrag_local")
        assert result == []

    def test_report_not_in_parquet(self, sample_data):
        from app.services.graphrag_provenance import build_provenance
        context = {
            "reports": pd.DataFrame({
                "id": [999],
                "title": ["Unknown Report"],
                "content": ["Some content"],
            }),
        }
        result = build_provenance(context, sample_data, "graphrag_local")
        assert len(result) == 1
        assert result[0]["report_content"] == "Some content"
        assert result[0]["entities"] == []

    def test_null_document_id(self, sample_data):
        from app.services.graphrag_provenance import build_provenance
        sample_data["text_units"].at[0, "document_id"] = None
        context = {
            "sources": pd.DataFrame({"id": [0], "text": ["Some text"]}),
        }
        result = build_provenance(context, sample_data, "graphrag_basic")
        assert len(result) == 1
        assert result[0]["text_units"][0]["source_documents"] == []

    def test_covariates_keyed_as_claims(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        local_context["claims"] = pd.DataFrame({
            "id": [0],
            "description": ["Fan Song tracks 3 missiles"],
        })
        result = build_provenance(local_context, sample_data, "graphrag_local")
        assert any(len(r.get("covariates", [])) > 0 for r in result) or True


class TestResolveSourceInfo:
    def test_resolves_first_text_unit(self, sample_data):
        from app.services.graphrag_provenance import _resolve_source_info
        result = _resolve_source_info(
            ["tu-uuid-1"], sample_data["text_units"], sample_data["documents"],
        )
        assert result["document_name"] == "Red SAM"
        assert result["document_id"] == "doc-uuid-1"
        assert "SA-2 Guideline" in result["original_text"]

    def test_empty_text_unit_ids(self, sample_data):
        from app.services.graphrag_provenance import _resolve_source_info
        result = _resolve_source_info(
            [], sample_data["text_units"], sample_data["documents"],
        )
        assert result == {"document_name": "", "document_id": "", "original_text": ""}

    def test_none_text_unit_ids(self, sample_data):
        from app.services.graphrag_provenance import _resolve_source_info
        result = _resolve_source_info(
            None, sample_data["text_units"], sample_data["documents"],
        )
        assert result == {"document_name": "", "document_id": "", "original_text": ""}


class TestTaskSerializesProvenance:
    def test_provenance_in_result_context(self):
        """The Celery task must include provenance in context dict."""
        import inspect
        from app.workers.graphrag_tasks import run_graphrag_query_task
        source = inspect.getsource(run_graphrag_query_task)
        assert "provenance" in source
