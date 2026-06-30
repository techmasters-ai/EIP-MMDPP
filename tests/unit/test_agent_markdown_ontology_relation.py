from __future__ import annotations
import uuid
from app.api.v1._agent_helpers import build_markdown
from app.schemas.retrieval import QueryResultItem


def test_markdown_renders_ontology_relation():
    item = QueryResultItem(chunk_id=uuid.uuid4(), score=0.5, modality="text", content_text="Fan Song",
                           context={"source": "ontology_relation", "rel_type": "ASSOCIATED_WITH",
                                    "related_entity": "Fan Song", "reserved": True})
    md = build_markdown("SA-2", [item])
    assert "ASSOCIATED_WITH" in md and "Fan Song" in md
    assert "reserved" in md.lower()
