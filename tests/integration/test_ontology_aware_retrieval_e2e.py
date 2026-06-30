"""E2E acceptance for ontology-aware hybrid retrieval (plan Task 9).

This exercises the LIVE stack (api at localhost:8005 + ArcadeDB), so it is
skipped unless ``OAR_E2E_LIVE=1`` is set. It encodes the gate evidence captured
on 2026-06-30 on branch ``feat/ontology-aware-retrieval``.

GATE DISPOSITION (user-approved 2026-06-30)
-------------------------------------------
The plan's original acceptance item — "full rollback reproduces the baseline
chunk_id order BYTE-IDENTICALLY" — is **unachievable** and was replaced with an
**inert-rollback** criterion. Reason: the retrieval pipeline is *pre-existing*
nondeterministic. Verified on ``main`` (no feature code): the same hybrid query
returned 6 results on one run and 5 on the next (set intersection 4, symmetric
difference 3). Result *set membership* varies run-to-run, so no stable baseline
exists to be byte-identical to — for any change, not just this feature. Logged
as a separate backlog finding (retrieval nondeterminism).

CAPTURED EVIDENCE (feature-on, defaults ENABLED=true / RESERVED_SLOTS=3 / ALPHA=0.6)
-----------------------------------------------------------------------------------
Query: {"query_text": "SA-2 guidance radar", "strategy": "hybrid", "top_k": 10}
Returned 4 ``ontology_relation`` chunks reached via the ``CUES`` domain relation:
  - rel_type=CUES related=Amazonka  reserved=True  raw_cosine=0.529 fused_pre_rerank=0.476
  - rel_type=CUES related=Amazonka  reserved=None  raw_cosine=0.506 fused_pre_rerank=0.461
  - rel_type=CUES related=Amazonka  reserved=True  raw_cosine=0.595 fused_pre_rerank=0.518
  - rel_type=CUES related=SNR-75    reserved=True  raw_cosine=0.513 fused_pre_rerank=0.465
=> 4 ontology_relation, 3 reserved. Amazonka (RD-75 rangefinding radar) and
   SNR-75 (Fan Song nomenclature) are ontological neighbours a pure-semantic
   search would not rank — exactly the intent.

INERT-ROLLBACK (ENABLED=false / RESERVED_SLOTS=0 / ALPHA=1.0)
------------------------------------------------------------
Same query returned 0 ``ontology_relation`` chunks, 0 reserved — the feature is
fully inert when flagged off. (The ``bothE`` MATCH in ``get_related_entity_chunks``
works against real ArcadeDB; no per-relation fallback was needed.)

MANUAL ROLLBACK PROCEDURE (operator)
------------------------------------
  1. In .env set RETRIEVAL_DOMAIN_EXPANSION_ENABLED=false,
     RETRIEVAL_ONTOLOGY_RESERVED_SLOTS=0, RETRIEVAL_RERANK_BLEND_ALPHA=1.0
  2. docker compose -p eip-mmdpp up -d --force-recreate api
  3. The hybrid query then returns no ontology_relation / reserved chunks; the
     feature is disabled with no other behaviour change (the rollback code path
     is the same one main runs — see the m=0 / alpha=1.0 / flag-off unit tests).
"""
from __future__ import annotations

import json
import os
import urllib.request

import pytest

_LIVE = os.environ.get("OAR_E2E_LIVE") == "1"
_API = os.environ.get("OAR_API_BASE", "http://localhost:8005")

pytestmark = pytest.mark.skipif(not _LIVE, reason="set OAR_E2E_LIVE=1 to run against the live stack")


def _hybrid_query(query_text: str, top_k: int = 10) -> list[dict]:
    body = json.dumps({"query_text": query_text, "strategy": "hybrid", "top_k": top_k}).encode()
    req = urllib.request.Request(
        f"{_API}/v1/retrieval/query", data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r).get("results", [])


def test_feature_on_surfaces_reserved_ontology_relation_chunk():
    """With the feature on (server defaults), a curated domain relation surfaces
    a reserved ontology_relation chunk carrying raw_cosine + fused_score_pre_rerank."""
    results = _hybrid_query("SA-2 guidance radar")
    onto = [r for r in results if (r.get("context") or {}).get("source") == "ontology_relation"]
    assert onto, "expected >=1 ontology_relation chunk (feature on)"
    assert any((r.get("context") or {}).get("reserved") for r in onto), "expected >=1 reserved chunk"
    for r in onto:
        ctx = r["context"]
        assert "raw_cosine" in ctx and "fused_score_pre_rerank" in ctx
        assert ctx.get("rel_type") in {
            "VARIANT_OF", "ASSOCIATED_WITH", "CUES", "PART_OF", "CONTAINS", "USES_COMPONENT",
        }


def test_settings_endpoint_exposes_reserved_slots_default():
    with urllib.request.urlopen(f"{_API}/v1/settings/retrieval", timeout=30) as r:
        data = json.load(r)
    assert "ontology_reserved_slots" in data
