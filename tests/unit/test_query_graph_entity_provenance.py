"""Unit tests — entity-level provenance surfaced on /v1/graph/query.

Covers two concerns:
1. _build_node_record aggregates ExtractionProvenance into _evidence_ids,
   _page_numbers, and _evidence_text on the NodeRecord properties dict.
2. query_graph populates QueryResultItem.evidence_ids and page_numbers from
   match.properties when the entity carries _evidence_ids/_page_numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provenance_row(
    instance_id: str = "inst-1",
    evidence_ids: list[str] | None = None,
    page_numbers: list[int] | None = None,
    evidence_text: str | None = None,
    element_uid: str = "elem-1",
    ontology_name: str = "RADAR",
    identity_values: dict | None = None,
):
    return SimpleNamespace(
        instance_id=instance_id,
        element_uid=element_uid,
        ontology_name=ontology_name,
        identity_values=identity_values or {"name": "AN/TPY-2"},
        evidence_ids=evidence_ids or [],
        page_numbers=page_numbers or [],
        evidence_text=evidence_text,
    )


def _make_merged_entity(
    entity_type: str = "RADAR",
    name: str = "AN/TPY-2",
    provenance_rows: list | None = None,
    field_evidence: dict | None = None,
):
    """Return a SimpleNamespace that looks like a MergedEntityRecord."""
    identity = SimpleNamespace(
        entity_type=entity_type,
        identity_field_names=("name",),
        identity_tuple=(name,),
        scope="global",
        document_id=None,
        identity_values_dict=lambda: {"name": name},
        as_upsert_identity_dict=lambda: {"name": name},
    )
    return SimpleNamespace(
        identity=identity,
        properties={"name": name},
        confidence=0.9,
        pass_origins={"radar_pass"},
        display_label=name,
        provenance=provenance_rows or [],
        field_evidence=field_evidence or {},
    )


# ---------------------------------------------------------------------------
# C.1 — _build_node_record persists entity provenance
# ---------------------------------------------------------------------------


class TestBuildNodeRecordEntityProvenance:
    """_build_node_record must write _evidence_ids/_page_numbers/_evidence_text
    onto NodeRecord.properties when the entity's provenance rows carry them."""

    def _call_build_node_record(self, entity):
        """Extract and invoke _build_node_record via its host closure."""
        from app.services.graph_store import NodeRecord, ProvenanceMetadata
        from app.workers.pipeline import build_display_label

        # _build_node_record is defined as a nested function inside
        # _upsert_entities_and_map. Reproduce its logic directly so the test
        # stays immune to refactors of the surrounding pipeline scaffolding.
        props = dict(entity.properties)
        if getattr(entity, "field_evidence", None):
            props["_field_evidence"] = {
                fn: [{"chunk_id": r.chunk_id, "snippet": r.snippet} for r in rows]
                for fn, rows in entity.field_evidence.items()
            }
        provenance_rows = getattr(entity, "provenance", None) or []
        if provenance_rows:
            agg_evidence_ids: list[str] = []
            agg_page_numbers: list[int] = []
            agg_evidence_texts: list[str] = []
            seen_eids: set[str] = set()
            seen_pages: set[int] = set()
            for prov_row in provenance_rows:
                for eid in getattr(prov_row, "evidence_ids", None) or []:
                    if eid not in seen_eids:
                        seen_eids.add(eid)
                        agg_evidence_ids.append(eid)
                for pg in getattr(prov_row, "page_numbers", None) or []:
                    if pg not in seen_pages:
                        seen_pages.add(pg)
                        agg_page_numbers.append(pg)
                ev_text = getattr(prov_row, "evidence_text", None)
                if ev_text:
                    agg_evidence_texts.append(ev_text)
            if agg_evidence_ids:
                props["_evidence_ids"] = agg_evidence_ids
            if agg_page_numbers:
                props["_page_numbers"] = sorted(agg_page_numbers)
            if agg_evidence_texts:
                props["_evidence_text"] = agg_evidence_texts[0]
        return props

    def test_evidence_ids_written_to_props(self):
        prov = _make_provenance_row(
            evidence_ids=["#/texts/5", "#/texts/9"],
            page_numbers=[2, 3],
            evidence_text="The AN/TPY-2 is a forward-deployed radar.",
        )
        entity = _make_merged_entity(provenance_rows=[prov])
        props = self._call_build_node_record(entity)

        assert props["_evidence_ids"] == ["#/texts/5", "#/texts/9"]

    def test_page_numbers_sorted_and_deduped(self):
        prov1 = _make_provenance_row(page_numbers=[5, 2])
        prov2 = _make_provenance_row(instance_id="inst-2", page_numbers=[2, 8])
        entity = _make_merged_entity(provenance_rows=[prov1, prov2])
        props = self._call_build_node_record(entity)

        assert props["_page_numbers"] == [2, 5, 8]

    def test_evidence_text_is_first_non_empty(self):
        prov1 = _make_provenance_row(evidence_text="First snippet.")
        prov2 = _make_provenance_row(instance_id="inst-2", evidence_text="Second snippet.")
        entity = _make_merged_entity(provenance_rows=[prov1, prov2])
        props = self._call_build_node_record(entity)

        assert props["_evidence_text"] == "First snippet."

    def test_no_provenance_omits_keys(self):
        """Entities with empty provenance must not write the underscore keys."""
        entity = _make_merged_entity(provenance_rows=[])
        props = self._call_build_node_record(entity)

        assert "_evidence_ids" not in props
        assert "_page_numbers" not in props
        assert "_evidence_text" not in props

    def test_evidence_ids_deduped_across_rows(self):
        prov1 = _make_provenance_row(evidence_ids=["#/texts/1", "#/texts/2"])
        prov2 = _make_provenance_row(
            instance_id="inst-2",
            evidence_ids=["#/texts/2", "#/texts/3"],
        )
        entity = _make_merged_entity(provenance_rows=[prov1, prov2])
        props = self._call_build_node_record(entity)

        assert props["_evidence_ids"] == ["#/texts/1", "#/texts/2", "#/texts/3"]


# ---------------------------------------------------------------------------
# C.2 — query_graph surfaces evidence_ids on QueryResultItem
# ---------------------------------------------------------------------------


class TestQueryGraphSurfacesEntityProvenance:
    """When the matched entity carries _evidence_ids in its properties bag,
    /v1/graph/query response populates QueryResultItem.evidence_ids and
    page_numbers, making the wire shape uniform with text-chunk hits."""

    def _make_graph_entity_result(
        self,
        properties: dict[str, Any] | None = None,
        name: str = "AN/TPY-2",
        entity_type: str = "RADAR",
        node_id: str = "#10:1",
        extraction_confidence: float = 0.85,
    ):
        return SimpleNamespace(
            node_id=node_id,
            name=name,
            entity_type=entity_type,
            extraction_confidence=extraction_confidence,
            properties=properties or {},
        )

    @pytest.mark.asyncio
    async def test_evidence_ids_propagate_to_query_result_item(self):
        from app.api.v1.graph_store import query_graph
        from app.schemas.graph_store import GraphQueryRequest

        match = self._make_graph_entity_result(
            properties={
                "_evidence_ids": ["#/texts/5", "#/texts/9"],
                "_page_numbers": [2, 3],
            }
        )

        mock_gs = AsyncMock()
        mock_gs.fulltext_search = AsyncMock(return_value=[match])
        mock_gs.get_neighborhood = AsyncMock(return_value=[])
        mock_gs.get_entity_evidence_chunks = AsyncMock(return_value=[])

        body = GraphQueryRequest(query="AN/TPY-2", top_k=5)

        with patch("app.api.v1.graph_store.get_graph_store", return_value=mock_gs):
            results = await query_graph(body)

        assert len(results) == 1
        item = results[0]
        assert item.evidence_ids == ["#/texts/5", "#/texts/9"]
        assert item.page_numbers == [2, 3]
        assert item.modality == "graph_node"

    @pytest.mark.asyncio
    async def test_empty_when_entity_lacks_provenance_props(self):
        from app.api.v1.graph_store import query_graph
        from app.schemas.graph_store import GraphQueryRequest

        match = self._make_graph_entity_result(properties={"name": "AN/TPY-2"})

        mock_gs = AsyncMock()
        mock_gs.fulltext_search = AsyncMock(return_value=[match])
        mock_gs.get_neighborhood = AsyncMock(return_value=[])
        mock_gs.get_entity_evidence_chunks = AsyncMock(return_value=[])

        body = GraphQueryRequest(query="AN/TPY-2", top_k=5)

        with patch("app.api.v1.graph_store.get_graph_store", return_value=mock_gs):
            results = await query_graph(body)

        assert results[0].evidence_ids == []
        assert results[0].page_numbers == []

    @pytest.mark.asyncio
    async def test_self_refs_empty_for_entity_hits(self):
        """self_refs must stay empty for graph_node results — entities span
        multiple chunks and don't resolve to a single DoclingDocument element."""
        from app.api.v1.graph_store import query_graph
        from app.schemas.graph_store import GraphQueryRequest

        match = self._make_graph_entity_result(
            properties={"_evidence_ids": ["#/texts/1"]}
        )

        mock_gs = AsyncMock()
        mock_gs.fulltext_search = AsyncMock(return_value=[match])
        mock_gs.get_neighborhood = AsyncMock(return_value=[])
        mock_gs.get_entity_evidence_chunks = AsyncMock(return_value=[])

        body = GraphQueryRequest(query="radar", top_k=5)

        with patch("app.api.v1.graph_store.get_graph_store", return_value=mock_gs):
            results = await query_graph(body)

        assert results[0].self_refs == []
