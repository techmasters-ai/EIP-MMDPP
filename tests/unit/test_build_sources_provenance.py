"""build_sources carries ontology_relation provenance into the structured
citation list (final-review follow-up to acceptance #7)."""
from __future__ import annotations

import uuid

from app.api.v1._agent_helpers import build_sources
from app.schemas.retrieval import QueryResultItem


def test_build_sources_carries_ontology_relation_provenance():
    onto = QueryResultItem(
        chunk_id=uuid.uuid4(), score=0.5, modality="text", content_text="x",
        context={"source": "ontology_relation", "rel_type": "CUES",
                 "related_entity": "Amazonka", "reserved": True},
    )
    plain = QueryResultItem(
        chunk_id=uuid.uuid4(), score=0.4, modality="text", content_text="y", context=None,
    )
    srcs = build_sources([onto, plain])
    assert srcs[0].source == "ontology_relation"
    assert srcs[0].rel_type == "CUES"
    assert srcs[0].related_entity == "Amazonka"
    assert srcs[0].reserved is True
    # Non-graph result: provenance fields default cleanly.
    assert srcs[1].source is None
    assert srcs[1].rel_type is None
    assert srcs[1].related_entity is None
    assert srcs[1].reserved is False
