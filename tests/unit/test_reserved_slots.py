from __future__ import annotations
import uuid
from app.schemas.retrieval import QueryResultItem
from app.api.v1.retrieval import _apply_reserved_slots

W = {"ASSOCIATED_WITH": 0.85, "VARIANT_OF": 0.95, "RELATED_TO": 0.75, "default": 0.70}


def _seed(score):
    return QueryResultItem(chunk_id=uuid.uuid4(), score=score, modality="text", content_text="x", context={"source": "text"})


def _onto(score, cosine, rel="ASSOCIATED_WITH"):
    return QueryResultItem(chunk_id=uuid.uuid4(), score=score, modality="text", content_text="x",
                           context={"source": "ontology_relation", "rel_type": rel, "raw_cosine": cosine})


def test_qualifying_ontology_chunk_reserved_over_seed():
    seeds = [_seed(0.9), _seed(0.85), _seed(0.8)]
    onto = _onto(score=0.2, cosine=0.3)
    out = _apply_reserved_slots(seeds + [onto], top_k=3, m=1,
                                min_rel_weight=0.85, min_cosine=0.15, relation_weights=W)
    assert onto in out
    assert len(out) == 3


def test_below_floor_rejected():
    seeds = [_seed(0.9), _seed(0.85), _seed(0.8)]
    onto = _onto(score=0.2, cosine=0.10)
    out = _apply_reserved_slots(seeds + [onto], top_k=3, m=1,
                                min_rel_weight=0.85, min_cosine=0.15, relation_weights=W)
    assert onto not in out


def test_non_tier_relation_rejected():
    seeds = [_seed(0.9), _seed(0.85), _seed(0.8)]
    onto = _onto(score=0.2, cosine=0.5, rel="RELATED_TO")
    out = _apply_reserved_slots(seeds + [onto], top_k=3, m=1,
                                min_rel_weight=0.85, min_cosine=0.15, relation_weights=W)
    assert onto not in out


def test_m_zero_is_pure_ranking():
    seeds = [_seed(0.9), _seed(0.85)]
    onto = _onto(score=0.2, cosine=0.5)
    pool = seeds + [onto]
    out = _apply_reserved_slots(pool, top_k=2, m=0,
                                min_rel_weight=0.85, min_cosine=0.15, relation_weights=W)
    assert out == sorted(pool, key=lambda x: x.score, reverse=True)[:2]
