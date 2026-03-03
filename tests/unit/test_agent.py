"""Unit tests for the LangGraph agent context helpers.

These tests exercise build_markdown and build_sources directly from
_agent_helpers, which has no DB/asyncpg dependency — matching the
inline-import pattern used by other unit tests in this suite.
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(**kwargs):
    """Build a QueryResultItem with sensible defaults."""
    from app.schemas.retrieval import QueryResultItem

    defaults = dict(
        score=0.85,
        modality="text",
        content_text="Some chunk text about the Patriot PAC-3 missile.",
        page_number=1,
        classification="UNCLASSIFIED",
        context=None,
    )
    defaults.update(kwargs)
    return QueryResultItem(**defaults)


# ---------------------------------------------------------------------------
# build_markdown
# ---------------------------------------------------------------------------


class TestBuildMarkdown:
    def _call(self, query, results):
        from app.api.v1._agent_helpers import build_markdown
        return build_markdown(query, results)

    def test_empty_results_returns_no_results_message(self):
        md = self._call("patriot missile", [])
        assert "No results found" in md
        assert "patriot missile" in md

    def test_returns_markdown_header(self):
        md = self._call("test", [_make_item()])
        assert md.startswith("## Retrieved Context")

    def test_result_numbered(self):
        items = [_make_item(), _make_item(score=0.7)]
        md = self._call("test", items)
        assert "### Result 1" in md
        assert "### Result 2" in md

    def test_score_formatted_as_percentage(self):
        md = self._call("test", [_make_item(score=0.92)])
        assert "92%" in md

    def test_content_text_included(self):
        md = self._call("test", [_make_item(content_text="Specific chunk text here")])
        assert "Specific chunk text here" in md

    def test_classified_item_shows_classification(self):
        md = self._call("test", [_make_item(classification="SECRET")])
        assert "SECRET" in md

    def test_unclassified_not_explicitly_shown(self):
        """UNCLASSIFIED shouldn't clutter the output with a classification line."""
        md = self._call("test", [_make_item(classification="UNCLASSIFIED")])
        assert "UNCLASSIFIED" not in md

    def test_page_number_included(self):
        md = self._call("test", [_make_item(page_number=42)])
        assert "42" in md

    def test_page_number_none_not_shown(self):
        md = self._call("test", [_make_item(page_number=None)])
        assert "Page" not in md

    def test_graph_node_entity_name_shown(self):
        item = _make_item(
            modality="graph_node",
            content_text=None,
            context={
                "entity": {
                    "properties": {"name": "Patriot PAC-3"},
                    "labels": ["EQUIPMENT_SYSTEM"],
                },
                "rel_type": "IS_SUBSYSTEM_OF",
                "neighbor": {
                    "properties": {"name": "MK-4 Guidance Computer"},
                    "labels": ["SUBSYSTEM"],
                },
            },
        )
        md = self._call("test", [item])
        assert "Patriot PAC-3" in md

    def test_graph_node_relationship_shown(self):
        item = _make_item(
            modality="graph_node",
            content_text=None,
            context={
                "entity": {"properties": {"name": "MK-4 Guidance Computer"}},
                "rel_type": "IS_SUBSYSTEM_OF",
                "neighbor": {"properties": {"name": "Patriot PAC-3"}},
            },
        )
        md = self._call("test", [item])
        assert "IS_SUBSYSTEM_OF" in md
        assert "Patriot PAC-3" in md

    def test_multiple_results_all_present(self):
        items = [_make_item(content_text=f"chunk {i}") for i in range(5)]
        md = self._call("test", items)
        for i in range(1, 6):
            assert f"### Result {i}" in md

    def test_modality_shown_in_header(self):
        md = self._call("test", [_make_item(modality="schematic")])
        assert "schematic" in md


# ---------------------------------------------------------------------------
# build_sources
# ---------------------------------------------------------------------------


class TestBuildSources:
    def _call(self, results):
        from app.api.v1._agent_helpers import build_sources
        return build_sources(results)

    def test_empty_returns_empty_list(self):
        assert self._call([]) == []

    def test_length_matches_input(self):
        items = [_make_item() for _ in range(3)]
        sources = self._call(items)
        assert len(sources) == 3

    def test_score_preserved(self):
        items = [_make_item(score=0.77)]
        sources = self._call(items)
        assert abs(sources[0].score - 0.77) < 1e-6

    def test_modality_preserved(self):
        items = [_make_item(modality="image")]
        sources = self._call(items)
        assert sources[0].modality == "image"

    def test_classification_preserved(self):
        items = [_make_item(classification="CUI")]
        sources = self._call(items)
        assert sources[0].classification == "CUI"

    def test_chunk_id_none_when_not_set(self):
        item = _make_item()
        item.chunk_id = None
        sources = self._call([item])
        assert sources[0].chunk_id is None

    def test_chunk_id_string_when_set(self):
        import uuid
        uid = uuid.uuid4()
        item = _make_item()
        item.chunk_id = uid
        sources = self._call([item])
        assert sources[0].chunk_id == str(uid)
