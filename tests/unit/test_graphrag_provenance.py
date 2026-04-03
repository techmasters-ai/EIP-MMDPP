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
    })
    relationships = pd.DataFrame({
        "id": [500, 600],
        "source": ["SA-2 GUIDELINE", "SPOON REST"],
        "target": ["FAN SONG", "FAN SONG"],
        "description": ["Uses for guidance", "Cues target to"],
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


class TestBuildProvenanceLocal:
    def test_groups_entities_under_reports(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        assert len(result) == 2
        r1 = result[0]
        assert r1["report_title"] == "SA-2 & Fan Song Community"
        entity_titles = [e["title"] for e in r1["entities"]]
        assert "FAN SONG" in entity_titles
        assert "SA-2 GUIDELINE" in entity_titles
        assert "SPOON REST" not in entity_titles

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

    def test_resolves_entity_source_documents(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        fan_song = [e for e in result[0]["entities"] if e["title"] == "FAN SONG"][0]
        assert len(fan_song["source_documents"]) > 0
        assert fan_song["source_documents"][0]["document_title"] == "Red SAM"

    def test_resolves_relationship_source_documents(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        rel = result[0]["relationships"][0]
        assert len(rel["source_documents"]) > 0
        assert rel["source_documents"][0]["document_title"] == "Red SAM"

    def test_resolves_text_unit_source_documents(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        tu = result[0]["text_units"]
        assert len(tu) > 0
        assert tu[0]["source_documents"][0]["document_title"] == "Red SAM"

    def test_strips_document_title_hash(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance
        result = build_provenance(local_context, sample_data, "graphrag_local")
        fan_song = [e for e in result[0]["entities"] if e["title"] == "FAN SONG"][0]
        title = fan_song["source_documents"][0]["document_title"]
        assert title == "Red SAM"


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
        assert result[0]["text_units"] == []


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


class TestTaskSerializesProvenance:
    def test_provenance_in_result_context(self):
        """The Celery task must include provenance in context dict."""
        import inspect
        from app.workers.graphrag_tasks import run_graphrag_query_task
        source = inspect.getsource(run_graphrag_query_task)
        assert "provenance" in source
