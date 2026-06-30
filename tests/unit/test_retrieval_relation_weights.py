from __future__ import annotations
import json
import pytest


def test_default_retrieval_relation_weights():
    from app.api.v1._retrieval_helpers import get_retrieval_relation_weights
    w = get_retrieval_relation_weights()
    assert w["VARIANT_OF"] == 0.95
    assert w["ASSOCIATED_WITH"] == 0.85
    assert w["CUES"] == 0.88
    assert w["USES_COMPONENT"] == 0.92
    assert w["CONTAINS"] == 0.90
    assert w["PART_OF"] == 0.90
    assert w.get("EXTRACTED_FROM", w["default"]) == 0.70
    assert w["default"] == 0.70


def test_curated_domain_relations():
    from app.api.v1._retrieval_helpers import get_curated_domain_relations
    rels = set(get_curated_domain_relations())
    assert rels == {"VARIANT_OF", "ASSOCIATED_WITH", "CUES", "PART_OF", "CONTAINS", "USES_COMPONENT"}


def test_env_override_merges(monkeypatch):
    import app.api.v1._retrieval_helpers as h
    monkeypatch.setattr(h, "_settings_cache", None)
    from app.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("RETRIEVAL_DOMAIN_RELATION_WEIGHTS", json.dumps({"CUES": 0.99, "NEW_REL": 0.5}))
    w = h.get_retrieval_relation_weights()
    assert w["CUES"] == 0.99
    assert w["NEW_REL"] == 0.5
    assert w["VARIANT_OF"] == 0.95
    get_settings.cache_clear()


def test_compute_fusion_score_uses_passed_relation_weights():
    from app.api.v1._retrieval_helpers import compute_fusion_score
    s_with = compute_fusion_score(
        semantic_score=0.0, ontology_rel_type="ASSOCIATED_WITH", ontology_hops=1,
        relation_weights={"ASSOCIATED_WITH": 0.85, "default": 0.70},
    )
    s_default = compute_fusion_score(
        semantic_score=0.0, ontology_rel_type="ASSOCIATED_WITH", ontology_hops=1,
        relation_weights={"default": 0.70},
    )
    assert s_with > s_default
