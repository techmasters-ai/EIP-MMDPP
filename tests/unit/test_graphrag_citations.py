"""Unit tests for GraphRAG citation parsing and resolution."""
import pandas as pd
import pytest

pytestmark = pytest.mark.unit


class TestStripSourcesBlock:
    def test_extracts_and_strips_sources(self):
        from app.services.graphrag_citations import strip_sources_block

        text = (
            "The SA-2 uses command guidance [1].\n\n"
            "## Sources\n"
            "[1] Entity: SA-2 GUIDELINE (3349), Relationship: 4276\n"
            "[2] Entity: SNR-75 FAN SONG (1494)\n"
        )
        clean, block = strip_sources_block(text)
        assert "## Sources" not in clean
        assert "[1]" in clean  # citation markers preserved
        assert "3349" in block
        assert "1494" in block

    def test_no_sources_block(self):
        from app.services.graphrag_citations import strip_sources_block

        text = "The SA-2 uses command guidance."
        clean, block = strip_sources_block(text)
        assert clean == text
        assert block == ""

    def test_strips_think_tags(self):
        from app.services.graphrag_citations import strip_sources_block

        text = "<think>reasoning</think>The SA-2 [1].\n\n## Sources\n[1] Entity: X (1)\n"
        clean, block = strip_sources_block(text)
        assert "<think>" not in clean
        assert "The SA-2 [1]." in clean


class TestParseCitationBlock:
    def test_parses_id_based_citations(self):
        from app.services.graphrag_citations import parse_citation_block

        block = (
            "[1] Entity: SA-2 GUIDELINE (3349), Relationship: 4276\n"
            "[2] Entity: SNR-75 FAN SONG (1494), Relationship: 3494\n"
        )
        citations = parse_citation_block(block, "local")
        assert len(citations) == 2
        assert citations[1]["entity_ids"] == [3349]
        assert citations[1]["relationship_ids"] == [4276]
        assert citations[2]["entity_ids"] == [1494]

    def test_parses_name_based_citations(self):
        from app.services.graphrag_citations import parse_citation_block

        block = "[1] Entity: SA-2 GUIDELINE\n[2] Entity: SNR-75 FAN SONG\n"
        citations = parse_citation_block(block, "global")
        assert citations[1]["entity_names"] == ["SA-2 GUIDELINE"]
        assert citations[2]["entity_names"] == ["SNR-75 FAN SONG"]

    def test_parses_text_based_citations(self):
        from app.services.graphrag_citations import parse_citation_block

        block = '[1] Source: "The SA-2 Guideline surface-to-air missile..."\n'
        citations = parse_citation_block(block, "basic")
        assert "SA-2 Guideline" in citations[1]["source_text"]

    def test_skips_malformed_lines(self):
        from app.services.graphrag_citations import parse_citation_block

        block = "[1] Entity: GOOD (100)\ngarbage line\n[2] Entity: ALSO GOOD (200)\n"
        citations = parse_citation_block(block, "local")
        assert len(citations) == 2

    def test_duplicate_numbers_keeps_first(self):
        from app.services.graphrag_citations import parse_citation_block

        block = "[1] Entity: FIRST (100)\n[1] Entity: SECOND (200)\n"
        citations = parse_citation_block(block, "local")
        assert citations[1]["entity_ids"] == [100]


class TestResolveCitations:
    @pytest.fixture
    def sample_data(self):
        entities = pd.DataFrame({
            "id": ["uuid-1", "uuid-2"],
            "human_readable_id": [3349, 1494],
            "title": ["SA-2 GUIDELINE", "SNR-75 FAN SONG"],
            "type": ["MISSILE_SYSTEM", "FIRE_CONTROL_SYSTEM"],
            "description": ["Soviet SAM system", "Fire control radar"],
            "text_unit_ids": [["tu-1"], ["tu-1", "tu-2"]],
        })
        relationships = pd.DataFrame({
            "id": ["rel-uuid-1"],
            "human_readable_id": [4276],
            "source": ["S-75 DVINA"],
            "target": ["V-750"],
            "description": ["Interceptor component"],
            "text_unit_ids": [["tu-1"]],
        })
        text_units = pd.DataFrame({
            "id": ["tu-1", "tu-2"],
            "human_readable_id": [0, 1],
            "text": ["The SA-2 Guideline uses command guidance...", "Fan Song radar operates..."],
            "document_ids": [["doc-1"], ["doc-2"]],
        })
        documents = pd.DataFrame({
            "id": ["doc-1", "doc-2"],
            "title": ["Red SAM_a3b2c1d4", "Fan Song radars_e5f6g7h8"],
        })
        return {
            "entities": entities,
            "relationships": relationships,
            "text_units": text_units,
            "documents": documents,
        }

    def test_resolves_id_based(self, sample_data):
        from app.services.graphrag_citations import resolve_citations

        parsed = {
            1: {"entity_ids": [3349], "relationship_ids": [4276]},
        }
        sources = resolve_citations(parsed, sample_data, "local")
        assert len(sources) == 1
        assert sources[0]["citation"] == 1
        assert sources[0]["entities"][0]["title"] == "SA-2 GUIDELINE"
        assert sources[0]["relationships"][0]["source"] == "S-75 DVINA"
        assert sources[0]["source_documents"][0]["document_title"] == "Red SAM"

    def test_resolves_name_based(self, sample_data):
        from app.services.graphrag_citations import resolve_citations

        parsed = {1: {"entity_names": ["SA-2 GUIDELINE"]}}
        sources = resolve_citations(parsed, sample_data, "global")
        assert sources[0]["entities"][0]["id"] == 3349

    def test_missing_id_skipped(self, sample_data):
        from app.services.graphrag_citations import resolve_citations

        parsed = {1: {"entity_ids": [9999], "relationship_ids": []}}
        sources = resolve_citations(parsed, sample_data, "local")
        assert sources[0]["entities"] == []

    def test_null_document_id(self, sample_data):
        from app.services.graphrag_citations import resolve_citations

        sample_data["text_units"].at[0, "document_ids"] = None
        parsed = {1: {"entity_ids": [3349], "relationship_ids": []}}
        sources = resolve_citations(parsed, sample_data, "local")
        # Should not crash; source_documents may be empty
        assert len(sources) == 1

    def test_strips_document_title_hash(self, sample_data):
        from app.services.graphrag_citations import resolve_citations

        parsed = {1: {"entity_ids": [3349], "relationship_ids": []}}
        sources = resolve_citations(parsed, sample_data, "local")
        title = sources[0]["source_documents"][0]["document_title"]
        assert title == "Red SAM"  # hash suffix stripped
