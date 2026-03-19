"""Tests for direct LLM entity extraction and the /extract-all endpoint."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the app package is importable
_DOCLING_GRAPH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_DOCLING_GRAPH_ROOT))

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ONTOLOGY_PATH = _REPO_ROOT / "ontology" / "ontology.yaml"


@pytest.fixture()
def mock_templates():
    from app.templates import build_templates, load_ontology

    for p in [_ONTOLOGY_PATH, Path(os.environ.get("ONTOLOGY_PATH", "/ontology/ontology.yaml"))]:
        if p.exists():
            ontology = load_ontology(p)
            return build_templates(ontology)
    pytest.skip("Ontology file not found")


@pytest.fixture()
def setup_app(mock_templates):
    """Set up the FastAPI app in mock mode."""
    from app import main

    main._templates = mock_templates
    main._ontology_version = "3.0.0"
    # Build schema prompts from ontology
    from app.templates import load_ontology
    for p in [_ONTOLOGY_PATH, Path(os.environ.get("ONTOLOGY_PATH", "/ontology/ontology.yaml"))]:
        if p.exists():
            ontology = load_ontology(p)
            main._group_schema_prompts = main._build_group_schema_prompts(ontology)
            main._relationship_prompt_context = main._build_relationship_prompt_context(ontology)
            break
    return main


@pytest.fixture()
def client(setup_app):
    from fastapi.testclient import TestClient
    return TestClient(setup_app.app)


# ---------------------------------------------------------------------------
# Test helpers — LLM response mock factories
# ---------------------------------------------------------------------------

def _make_llm_msg(content=None, reasoning_content=None, thinking=None, thinking_blocks=None):
    """Create a MagicMock LLM message with the given field values."""
    msg = MagicMock()
    msg.content = content
    msg.reasoning_content = reasoning_content
    msg.thinking = thinking
    msg.thinking_blocks = thinking_blocks
    return msg


def _make_llm_response(content=None, reasoning_content=None, thinking=None, thinking_blocks=None):
    """Create a full litellm.completion() response mock."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = _make_llm_msg(content, reasoning_content, thinking, thinking_blocks)
    return resp


# ---------------------------------------------------------------------------
# _parse_json_from_llm
# ---------------------------------------------------------------------------

class TestParseJsonFromLlm:
    def test_plain_json_object(self, setup_app):
        result = setup_app._parse_json_from_llm('{"entities": []}')
        assert result == {"entities": []}

    def test_plain_json_array(self, setup_app):
        result = setup_app._parse_json_from_llm('[{"name": "test"}]')
        assert result == [{"name": "test"}]

    def test_markdown_wrapped(self, setup_app):
        raw = '```json\n{"entities": [{"name": "test"}]}\n```'
        result = setup_app._parse_json_from_llm(raw)
        assert result == {"entities": [{"name": "test"}]}

    def test_text_before_json(self, setup_app):
        raw = 'Here are the entities: {"entities": [{"name": "AN/MPQ-53"}]}'
        result = setup_app._parse_json_from_llm(raw)
        assert result["entities"][0]["name"] == "AN/MPQ-53"

    def test_none_input(self, setup_app):
        assert setup_app._parse_json_from_llm(None) is None

    def test_empty_string(self, setup_app):
        assert setup_app._parse_json_from_llm("") is None

    def test_no_json(self, setup_app):
        assert setup_app._parse_json_from_llm("no json here") is None

    def test_json_embedded_in_reasoning_text(self, setup_app):
        """JSON after reasoning preamble (from reasoning_content fallback)."""
        raw = (
            "Let me analyze this text for entities.\n"
            "I found the following:\n"
            '{"entities": [{"name": "AN/MPQ-53", "entity_type": "RADAR_SYSTEM"}]}'
        )
        result = setup_app._parse_json_from_llm(raw)
        assert result is not None
        assert result["entities"][0]["name"] == "AN/MPQ-53"

    def test_think_tags_stripped(self, setup_app):
        """<think> tags wrapping reasoning should be stripped before parsing."""
        raw = '<think>Analyzing the text...</think>{"entities": []}'
        result = setup_app._parse_json_from_llm(raw)
        assert result == {"entities": []}


# ---------------------------------------------------------------------------
# _build_group_schema_prompts
# ---------------------------------------------------------------------------

class TestBuildGroupSchemaPrompts:
    def test_generates_all_groups(self, setup_app):
        prompts = setup_app._group_schema_prompts
        assert "reference" in prompts
        assert "equipment" in prompts
        assert "rf_signal" in prompts
        assert "weapon" in prompts
        assert "operational" in prompts

    def test_equipment_group_includes_radar_system(self, setup_app):
        prompt = setup_app._group_schema_prompts["equipment"]
        assert "RADAR_SYSTEM" in prompt
        assert "PLATFORM" in prompt

    def test_prompts_include_property_descriptions(self, setup_app):
        prompt = setup_app._group_schema_prompts["equipment"]
        assert "system_name" in prompt or "nomenclature" in prompt


# ---------------------------------------------------------------------------
# /extract-all (mock mode)
# ---------------------------------------------------------------------------

class TestExtractAllEndpoint:
    def test_mock_mode_returns_entities(self, client, setup_app):
        orig = setup_app.LLM_PROVIDER
        setup_app.LLM_PROVIDER = "mock"
        try:
            resp = client.post(
                "/extract-all",
                json={
                    "document_id": "test-all-1",
                    "text": "The AN/MPQ-53 radar operates in C-band on the Patriot system.",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "entities" in data
            assert "relationships" in data
            assert data["provider"] == "mock"
            assert len(data["entities"]) > 0
        finally:
            setup_app.LLM_PROVIDER = orig

    def test_service_not_ready(self, setup_app):
        """If templates aren't loaded, should return 503."""
        from fastapi.testclient import TestClient

        saved = setup_app._templates
        setup_app._templates = None
        try:
            test_client = TestClient(setup_app.app)
            resp = test_client.post(
                "/extract-all",
                json={"document_id": "test", "text": "text"},
            )
            assert resp.status_code == 503
        finally:
            setup_app._templates = saved


# ---------------------------------------------------------------------------
# _extract_entities_for_group (with mocked LLM)
# ---------------------------------------------------------------------------

class TestExtractEntitiesForGroup:
    @patch("app.main.litellm")
    def test_parses_llm_response(self, mock_litellm, setup_app):
        """Should parse JSON entity array from LLM response."""
        mock_litellm.completion.return_value = _make_llm_response(content=json.dumps({
            "entities": [
                {"name": "S-300", "entity_type": "MISSILE_SYSTEM", "confidence": 0.9, "properties": {}},
                {"name": "Tombstone", "entity_type": "RADAR_SYSTEM", "confidence": 0.85, "properties": {}},
            ]
        }))
        result = setup_app._extract_entities_for_group(
            "The S-300 uses the Tombstone radar.", "equipment"
        )
        assert len(result) == 2
        assert result[0]["name"] == "S-300"
        assert result[1]["entity_type"] == "RADAR_SYSTEM"

    @patch("app.main.litellm")
    def test_llm_failure_returns_empty(self, mock_litellm, setup_app):
        """LLM exception should return empty list, not crash."""
        mock_litellm.completion.side_effect = Exception("LLM timeout")
        result = setup_app._extract_entities_for_group("text", "equipment")
        assert result == []

    @patch("app.main.litellm")
    def test_invalid_json_returns_empty(self, mock_litellm, setup_app):
        """Non-JSON LLM output should return empty list."""
        mock_litellm.completion.return_value = _make_llm_response(
            content="I cannot extract entities from this text."
        )
        result = setup_app._extract_entities_for_group("text", "equipment")
        assert result == []


# ---------------------------------------------------------------------------
# _extract_relationships (with mocked LLM)
# ---------------------------------------------------------------------------

class TestExtractRelationships:
    @patch("app.main.litellm")
    def test_parses_relationship_response(self, mock_litellm, setup_app):
        mock_litellm.completion.return_value = _make_llm_response(content=json.dumps({
            "relationships": [{
                "from_name": "S-300", "from_type": "MISSILE_SYSTEM",
                "rel_type": "HAS_COMPONENT",
                "to_name": "Tombstone", "to_type": "RADAR_SYSTEM",
                "confidence": 0.9,
            }]
        }))
        entities = [
            {"name": "S-300", "entity_type": "MISSILE_SYSTEM"},
            {"name": "Tombstone", "entity_type": "RADAR_SYSTEM"},
        ]
        result = setup_app._extract_relationships("text about S-300", entities)
        assert len(result) == 1
        assert result[0]["rel_type"] == "HAS_COMPONENT"

    @patch("app.main.litellm")
    def test_llm_failure_returns_empty(self, mock_litellm, setup_app):
        mock_litellm.completion.side_effect = Exception("LLM down")
        result = setup_app._extract_relationships("text", [])
        assert result == []


# ---------------------------------------------------------------------------
# _run_full_extraction (with mocked LLM)
# ---------------------------------------------------------------------------

class TestRunFullExtraction:
    @patch("app.main._extract_relationships")
    @patch("app.main._extract_entities_for_group")
    def test_runs_all_groups_and_relationships(self, mock_entities, mock_rels, setup_app):
        """Should call entity extraction for all 5 groups + 1 relationship pass."""
        # Return unique entity per call so dedup doesn't collapse them
        call_count = [0]
        def _unique_entity(*args, **kwargs):
            call_count[0] += 1
            return [{"name": f"Entity-{call_count[0]}", "entity_type": "PLATFORM", "confidence": 0.9, "properties": {}}]
        mock_entities.side_effect = _unique_entity
        mock_rels.return_value = []

        entities, rels = setup_app._run_full_extraction("some document text")

        assert len(entities) == 5  # 5 unique entities (1 per group)
        assert mock_entities.call_count == 5
        assert mock_rels.call_count == 1

    @patch("app.main._extract_relationships")
    @patch("app.main._extract_entities_for_group")
    def test_passes_entities_to_relationship_extraction(self, mock_entities, mock_rels, setup_app):
        """Relationship extraction should receive all extracted entities as context."""
        call_count = [0]
        def _unique_entity(*args, **kwargs):
            call_count[0] += 1
            return [{"name": f"Radar-{call_count[0]}", "entity_type": "RADAR_SYSTEM", "confidence": 0.9, "properties": {}}]
        mock_entities.side_effect = _unique_entity
        mock_rels.return_value = []

        setup_app._run_full_extraction("text")

        rel_call_args = mock_rels.call_args
        entities_context = rel_call_args[0][1]
        assert len(entities_context) == 5  # 5 groups × 1 unique entity each
        assert all(e.get("name") for e in entities_context)


# ---------------------------------------------------------------------------
# _extract_llm_content — thinking-model response shapes
# ---------------------------------------------------------------------------

class TestExtractLlmContent:
    """Test _extract_llm_content with thinking-model response field layouts.

    Reproduces the production failure: gpt-oss:120b with OLLAMA_THINK=high
    returns msg.content="" and the JSON lands in msg.reasoning_content.
    """

    def test_standard_content(self, setup_app):
        msg = _make_llm_msg(content='{"entities": []}')
        assert setup_app._extract_llm_content(msg) == '{"entities": []}'

    def test_content_empty_falls_back_to_reasoning_content(self, setup_app):
        msg = _make_llm_msg(content="", reasoning_content='{"entities": [{"name": "test"}]}')
        assert setup_app._extract_llm_content(msg) == '{"entities": [{"name": "test"}]}'

    def test_content_none_falls_back_to_reasoning_content(self, setup_app):
        msg = _make_llm_msg(reasoning_content='{"entities": []}')
        assert setup_app._extract_llm_content(msg) == '{"entities": []}'

    def test_falls_back_to_thinking_field(self, setup_app):
        msg = _make_llm_msg(content="", reasoning_content="", thinking='{"entities": []}')
        assert setup_app._extract_llm_content(msg) == '{"entities": []}'

    def test_falls_back_to_thinking_blocks(self, setup_app):
        msg = _make_llm_msg(thinking_blocks=[{"type": "thinking", "thinking": '{"entities": []}'}])
        assert "entities" in setup_app._extract_llm_content(msg)

    def test_all_fields_empty_returns_empty_string(self, setup_app):
        assert setup_app._extract_llm_content(_make_llm_msg()) == ""

    def test_prefers_content_over_reasoning_content(self, setup_app):
        msg = _make_llm_msg(
            content='{"entities": [{"name": "from_content"}]}',
            reasoning_content='{"entities": [{"name": "from_reasoning"}]}',
        )
        assert "from_content" in setup_app._extract_llm_content(msg)

    def test_whitespace_only_content_falls_back(self, setup_app):
        msg = _make_llm_msg(content="   \n  ", reasoning_content='{"entities": []}')
        assert setup_app._extract_llm_content(msg) == '{"entities": []}'


# ---------------------------------------------------------------------------
# Thinking-model integration: entity/relationship extraction end-to-end
# ---------------------------------------------------------------------------

class TestThinkingModelExtraction:
    """Reproduces production failure: gpt-oss:120b with OLLAMA_THINK=high
    returns JSON in reasoning_content, content empty → 0 entities.
    """

    @patch("app.main.litellm")
    def test_entities_from_reasoning_content(self, mock_litellm, setup_app):
        mock_litellm.completion.return_value = _make_llm_response(
            content="",
            reasoning_content=json.dumps({"entities": [
                {"name": "S-300", "entity_type": "MISSILE_SYSTEM", "confidence": 0.9, "properties": {}},
            ]}),
        )
        result = setup_app._extract_entities_for_group("The S-300 system.", "equipment")
        assert len(result) == 1
        assert result[0]["name"] == "S-300"

    @patch("app.main.litellm")
    def test_entities_from_reasoning_content_with_preamble(self, mock_litellm, setup_app):
        """Should extract JSON even when reasoning_content has reasoning text before it."""
        mock_litellm.completion.return_value = _make_llm_response(reasoning_content=(
            "Let me analyze this text for equipment entities.\n\n"
            "I can identify the following:\n"
            '{"entities": [{"name": "AN/MPQ-53", "entity_type": "RADAR_SYSTEM",'
            ' "confidence": 0.95, "properties": {}}]}'
        ))
        result = setup_app._extract_entities_for_group("The AN/MPQ-53 radar.", "equipment")
        assert len(result) == 1
        assert result[0]["name"] == "AN/MPQ-53"

    @patch("app.main.litellm")
    def test_relationships_from_reasoning_content(self, mock_litellm, setup_app):
        mock_litellm.completion.return_value = _make_llm_response(
            content="",
            reasoning_content=json.dumps({"relationships": [{
                "from_name": "S-300", "from_type": "MISSILE_SYSTEM",
                "rel_type": "HAS_COMPONENT",
                "to_name": "Tombstone", "to_type": "RADAR_SYSTEM",
                "confidence": 0.9,
            }]}),
        )
        entities = [
            {"name": "S-300", "entity_type": "MISSILE_SYSTEM"},
            {"name": "Tombstone", "entity_type": "RADAR_SYSTEM"},
        ]
        result = setup_app._extract_relationships("text about S-300", entities)
        assert len(result) == 1
        assert result[0]["rel_type"] == "HAS_COMPONENT"

    @patch("app.main.litellm")
    def test_all_fields_empty_returns_empty_not_crash(self, mock_litellm, setup_app):
        mock_litellm.completion.return_value = _make_llm_response()
        result = setup_app._extract_entities_for_group("text", "equipment")
        assert result == []


# ---------------------------------------------------------------------------
# Schema prompt improvements (pattern fields, type hierarchy)
# ---------------------------------------------------------------------------

class TestSchemaPromptImprovements:
    def test_pattern_included_in_schema(self, setup_app):
        """Properties with pattern fields should include format annotation."""
        prompt = setup_app._group_schema_prompts.get("weapon", "")
        # COMPONENT has nsn with pattern ^\\d{4}-\\d{2}-\\d{3}-\\d{4}$
        assert "format:" in prompt

    def test_parent_hierarchy_in_schema(self, setup_app):
        """Entity headers should include parent type when present."""
        prompt = setup_app._group_schema_prompts.get("equipment", "")
        # RADAR_SYSTEM has parent: MilitarySystem
        assert "MilitarySystem" in prompt

    def test_label_in_schema(self, setup_app):
        """Entity headers should include label when present."""
        prompt = setup_app._group_schema_prompts.get("reference", "")
        # DOCUMENT has label: "Source Document"
        assert "Source Document" in prompt

    def test_few_shot_example_in_entity_prompt(self, setup_app):
        """Few-shot examples should be importable and non-empty for all groups."""
        from app.prompts import GROUP_FEW_SHOT_EXAMPLES
        for group in ("reference", "equipment", "rf_signal", "weapon", "operational"):
            assert group in GROUP_FEW_SHOT_EXAMPLES
            assert len(GROUP_FEW_SHOT_EXAMPLES[group]) > 50

    def test_cross_group_dedup_guidance(self, setup_app):
        """Overlapping groups should have dedup notes in their prompts."""
        from app.prompts import GROUP_PROMPTS
        assert "rf_signal group" in GROUP_PROMPTS["equipment"]
        assert "equipment group" in GROUP_PROMPTS["rf_signal"]
        assert "equipment group" in GROUP_PROMPTS["operational"]


# ---------------------------------------------------------------------------
# Relationship prompt from ontology
# ---------------------------------------------------------------------------

class TestRelationshipPromptFromOntology:
    def test_relationship_context_built_at_startup(self, setup_app):
        """_relationship_prompt_context should be populated from ontology."""
        ctx = setup_app._relationship_prompt_context
        assert len(ctx) > 100

    def test_includes_missing_types_from_old_hardcoded_list(self, setup_app):
        """Should include relationship types that were missing from the old prompt."""
        ctx = setup_app._relationship_prompt_context
        assert "LAUNCHES" in ctx
        assert "HAS_SEEKER" in ctx
        assert "HAS_GUIDANCE" in ctx
        assert "TESTED_IN" in ctx
        assert "SUPERSEDES" in ctx

    def test_includes_descriptions(self, setup_app):
        """Relationship types should have descriptions from ontology."""
        ctx = setup_app._relationship_prompt_context
        # PART_OF description: "Component or subsystem is part of a larger system"
        assert "part of a larger system" in ctx.lower() or "Component or subsystem" in ctx

    def test_includes_validation_matrix(self, setup_app):
        """Should include allowed (source, rel, target) triples."""
        ctx = setup_app._relationship_prompt_context
        assert "RADAR_SYSTEM" in ctx
        assert "INSTALLED_ON" in ctx
        assert "PLATFORM" in ctx

    def test_excludes_phantom_types(self, setup_app):
        """Old hardcoded phantom types should not appear."""
        ctx = setup_app._relationship_prompt_context
        # COMPLIES_WITH, FEEDS_INTO, RECEIVES_FROM were phantom types
        assert "COMPLIES_WITH" not in ctx
        assert "FEEDS_INTO" not in ctx
        assert "RECEIVES_FROM" not in ctx


# ---------------------------------------------------------------------------
# Chunking and deduplication
# ---------------------------------------------------------------------------

class TestChunkingAndDedup:
    def test_chunk_text_short_returns_one(self, setup_app):
        result = setup_app._chunk_text("short text", 1000, 100)
        assert len(result) == 1
        assert result[0] == "short text"

    def test_chunk_text_splits_with_overlap(self, setup_app):
        text = "a" * 3000
        result = setup_app._chunk_text(text, 1000, 200)
        assert len(result) >= 3
        # Each chunk should be <= chunk_size
        for chunk in result:
            assert len(chunk) <= 1000

    def test_chunk_text_overlap_content(self, setup_app):
        """Chunks should overlap — the end of chunk N should appear at the start of chunk N+1."""
        text = "ABCDEFGHIJ" * 100  # 1000 chars
        chunks = setup_app._chunk_text(text, 400, 100)
        assert len(chunks) >= 3
        # Last 100 chars of chunk 0 should equal first 100 chars of chunk 1
        assert chunks[0][-100:] == chunks[1][:100]

    def test_dedup_keeps_highest_confidence(self, setup_app):
        entities = [
            {"name": "AN/MPQ-53", "entity_type": "RADAR_SYSTEM", "confidence": 0.7},
            {"name": "AN/MPQ-53", "entity_type": "RADAR_SYSTEM", "confidence": 0.95},
        ]
        result = setup_app._dedup_entities(entities)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.95

    def test_dedup_different_types_kept(self, setup_app):
        """Same name but different types should not be deduped."""
        entities = [
            {"name": "Patriot", "entity_type": "MISSILE_SYSTEM", "confidence": 0.9},
            {"name": "Patriot", "entity_type": "PLATFORM", "confidence": 0.8},
        ]
        result = setup_app._dedup_entities(entities)
        assert len(result) == 2

    def test_dedup_empty_list(self, setup_app):
        assert setup_app._dedup_entities([]) == []


# ---------------------------------------------------------------------------
# JSON parser: new strategies (reasoning markers, backward bracket-match)
# ---------------------------------------------------------------------------

class TestParseJsonReasoningStrategies:
    def test_whole_string_json(self, setup_app):
        """Clean JSON string should parse directly."""
        result = setup_app._parse_json_from_llm('{"entities": []}')
        assert result == {"entities": []}

    def test_thinking_tags_both_variants(self, setup_app):
        """Both <think> and <thinking> tags should be stripped."""
        raw = '<thinking>Let me analyze...</thinking>{"entities": []}'
        assert setup_app._parse_json_from_llm(raw) == {"entities": []}
        raw2 = '<think>reasoning here</think>{"entities": []}'
        assert setup_app._parse_json_from_llm(raw2) == {"entities": []}

    def test_final_answer_marker(self, setup_app):
        """'Final Answer:' followed by JSON should extract the JSON."""
        raw = "Let me think about this...\n\nFinal Answer: {\"entities\": []}"
        result = setup_app._parse_json_from_llm(raw)
        assert result == {"entities": []}

    def test_answer_marker(self, setup_app):
        """'Answer:' followed by JSON should extract the JSON."""
        raw = "Reasoning: I found entities.\n\nAnswer: {\"entities\": [{\"name\": \"test\"}]}"
        result = setup_app._parse_json_from_llm(raw)
        assert result is not None
        assert result["entities"][0]["name"] == "test"

    def test_fenced_json_block(self, setup_app):
        """```json fenced blocks should be extracted."""
        raw = 'Here is the result:\n```json\n{"entities": [{"name": "S-300"}]}\n```'
        result = setup_app._parse_json_from_llm(raw)
        assert result["entities"][0]["name"] == "S-300"

    def test_reasoning_with_stray_brackets_then_json(self, setup_app):
        """Reasoning text with {} characters before the actual JSON answer."""
        raw = (
            "The schema says {\"type\": \"object\"} for RADAR_SYSTEM. "
            "Let me extract entities from the text.\n\n"
            '{"entities": [{"name": "AN/MPQ-53", "entity_type": "RADAR_SYSTEM", "confidence": 0.9}]}'
        )
        result = setup_app._parse_json_from_llm(raw)
        assert result is not None
        assert result["entities"][0]["name"] == "AN/MPQ-53"

    def test_json_array_not_confused_with_inner_object(self, setup_app):
        """[{"name": "test"}] should parse as array, not extract inner object."""
        result = setup_app._parse_json_from_llm('[{"name": "test"}]')
        assert isinstance(result, list)
        assert result[0]["name"] == "test"

    def test_no_json_returns_none(self, setup_app):
        """Pure reasoning text with no JSON should return None."""
        raw = "I analyzed the text but could not find any entities matching the criteria."
        assert setup_app._parse_json_from_llm(raw) is None
