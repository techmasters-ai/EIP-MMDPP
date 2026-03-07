"""Unit tests for the docling-graph extraction service.

Tests LLM response parsing, regex NER fallback, and NetworkX graph building.
All tests mock the LLM — no actual model calls are made.
"""

import json

import pytest

nx = pytest.importorskip("networkx", reason="networkx not installed locally")

pytestmark = pytest.mark.unit


SAMPLE_LLM_RESPONSE = json.dumps({
    "entities": [
        {
            "entity_type": "EQUIPMENT_SYSTEM",
            "name": "Patriot PAC-3",
            "properties": {"designation": "MIM-104F"},
            "confidence": 0.95,
        },
        {
            "entity_type": "STANDARD",
            "name": "MIL-STD-1553B",
            "properties": {"designation": "MIL-STD-1553B"},
            "confidence": 0.9,
        },
    ],
    "relationships": [
        {
            "relationship_type": "MEETS_STANDARD",
            "from_name": "Patriot PAC-3",
            "from_type": "EQUIPMENT_SYSTEM",
            "to_name": "MIL-STD-1553B",
            "to_type": "STANDARD",
            "properties": {},
            "confidence": 0.85,
        },
    ],
})


class TestParseLLMResponse:
    def test_parse_valid_json(self):
        from app.services.docling_graph_service import _parse_llm_response

        result = _parse_llm_response(SAMPLE_LLM_RESPONSE)
        assert len(result.entities) == 2
        assert len(result.relationships) == 1
        assert result.entities[0].name == "Patriot PAC-3"

    def test_parse_json_with_markdown_fences(self):
        from app.services.docling_graph_service import _parse_llm_response

        wrapped = f"```json\n{SAMPLE_LLM_RESPONSE}\n```"
        result = _parse_llm_response(wrapped)
        assert len(result.entities) == 2

    def test_parse_invalid_json_raises(self):
        from app.services.docling_graph_service import _parse_llm_response

        with pytest.raises(json.JSONDecodeError):
            _parse_llm_response("this is not json")


class TestFallbackRegexExtraction:
    def test_fallback_produces_entities(self):
        from app.services.docling_graph_service import _fallback_regex_extraction

        text = "The Patriot PAC-3 system complies with MIL-STD-1553B."
        result = _fallback_regex_extraction(text)
        assert len(result.entities) > 0
        entity_types = {e.entity_type for e in result.entities}
        assert "EQUIPMENT_SYSTEM" in entity_types or "STANDARD" in entity_types

    def test_fallback_empty_text(self):
        from app.services.docling_graph_service import _fallback_regex_extraction

        result = _fallback_regex_extraction("")
        assert result.entities == []
        assert result.relationships == []


class TestBuildNetworkxGraph:
    def test_builds_graph_from_extraction(self):
        from app.services.docling_graph_service import _build_networkx_graph
        from app.services.ontology_templates import DocumentExtractionResult, ExtractedEntity, ExtractedRelationship

        extraction = DocumentExtractionResult(
            entities=[
                ExtractedEntity(entity_type="EQUIPMENT_SYSTEM", name="System A", confidence=0.9),
                ExtractedEntity(entity_type="COMPONENT", name="Part B", confidence=0.8),
            ],
            relationships=[
                ExtractedRelationship(
                    relationship_type="CONTAINS",
                    from_name="System A", from_type="EQUIPMENT_SYSTEM",
                    to_name="Part B", to_type="COMPONENT",
                    confidence=0.7,
                ),
            ],
        )

        G = _build_networkx_graph(extraction, "doc-123")
        assert len(G.nodes) == 2
        assert len(G.edges) == 1
        assert G.nodes["EQUIPMENT_SYSTEM:System A"]["name"] == "System A"

    def test_builds_empty_graph(self):
        from app.services.docling_graph_service import _build_networkx_graph
        from app.services.ontology_templates import DocumentExtractionResult

        G = _build_networkx_graph(DocumentExtractionResult(), "doc-empty")
        assert len(G.nodes) == 0
        assert len(G.edges) == 0

    def test_creates_missing_nodes_from_relationships(self):
        from app.services.docling_graph_service import _build_networkx_graph
        from app.services.ontology_templates import DocumentExtractionResult, ExtractedRelationship

        extraction = DocumentExtractionResult(
            entities=[],
            relationships=[
                ExtractedRelationship(
                    relationship_type="RELATED",
                    from_name="X", from_type="A",
                    to_name="Y", to_type="B",
                    confidence=0.5,
                ),
            ],
        )

        G = _build_networkx_graph(extraction, "doc-rel")
        assert len(G.nodes) == 2
        assert G.nodes["A:X"]["confidence"] == 0.3  # low confidence for implicit nodes


class TestNetworkxToNeo4jImport:
    def test_converts_graph_to_neo4j_format(self):
        import networkx as nx
        from app.services.docling_graph_service import networkx_to_neo4j_import

        G = nx.DiGraph()
        G.add_node("A:Sys", entity_type="A", name="Sys", properties={}, confidence=0.9)
        G.add_node("B:Part", entity_type="B", name="Part", properties={}, confidence=0.8)
        G.add_edge("A:Sys", "B:Part", relationship_type="HAS", confidence=0.7)

        result = networkx_to_neo4j_import(G, "artifact-1")
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
        assert result["nodes"][0]["artifact_id"] == "artifact-1"
        assert result["edges"][0]["rel_type"] == "HAS"


class TestDeduplicateExtraction:
    def test_dedup_keeps_highest_confidence(self):
        from app.services.docling_graph_service import _deduplicate_extraction
        from app.services.ontology_templates import DocumentExtractionResult, ExtractedEntity

        result = DocumentExtractionResult(
            entities=[
                ExtractedEntity(entity_type="SYSTEM", name="Patriot", confidence=0.7, properties={"a": "1"}),
                ExtractedEntity(entity_type="SYSTEM", name="Patriot", confidence=0.9, properties={"b": "2"}),
            ],
        )
        deduped = _deduplicate_extraction(result)
        assert len(deduped.entities) == 1
        assert deduped.entities[0].confidence == 0.9
        # Properties from both occurrences should be merged
        assert "a" in deduped.entities[0].properties
        assert "b" in deduped.entities[0].properties

    def test_dedup_relationships_by_key(self):
        from app.services.docling_graph_service import _deduplicate_extraction
        from app.services.ontology_templates import DocumentExtractionResult, ExtractedRelationship

        result = DocumentExtractionResult(
            relationships=[
                ExtractedRelationship(
                    relationship_type="CONTAINS", from_name="A", from_type="SYS",
                    to_name="B", to_type="COMP", confidence=0.6,
                ),
                ExtractedRelationship(
                    relationship_type="CONTAINS", from_name="A", from_type="SYS",
                    to_name="B", to_type="COMP", confidence=0.8,
                ),
            ],
        )
        deduped = _deduplicate_extraction(result)
        assert len(deduped.relationships) == 1
        assert deduped.relationships[0].confidence == 0.8

    def test_dedup_different_entities_preserved(self):
        from app.services.docling_graph_service import _deduplicate_extraction
        from app.services.ontology_templates import DocumentExtractionResult, ExtractedEntity

        result = DocumentExtractionResult(
            entities=[
                ExtractedEntity(entity_type="SYSTEM", name="Patriot", confidence=0.9),
                ExtractedEntity(entity_type="STANDARD", name="MIL-STD-1553B", confidence=0.8),
            ],
        )
        deduped = _deduplicate_extraction(result)
        assert len(deduped.entities) == 2


class TestExtractSinglePass:
    def test_mock_provider_returns_mock(self):
        from app.services.docling_graph_service import _extract_single_pass

        G, provider = _extract_single_pass("some text", "doc-123")
        assert provider == "mock"
        assert len(G.nodes) == 0


class TestChunkedExtraction:
    def test_short_text_single_pass(self):
        from app.services.docling_graph_service import extract_graph_from_text_chunked

        # With mock provider, short text should go through single pass
        G, provider = extract_graph_from_text_chunked("short text", "doc-1", chunk_size=7000)
        assert provider == "mock"
        assert len(G.nodes) == 0


class TestExtractGraphFromText:
    def test_mock_provider_returns_empty_graph(self):
        from app.services.docling_graph_service import extract_graph_from_text

        # LLM_PROVIDER=mock is set in conftest.py env overrides
        G = extract_graph_from_text("some text", "doc-123")
        assert len(G.nodes) == 0
        assert len(G.edges) == 0
