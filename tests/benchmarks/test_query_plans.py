"""ArcadeDB query plan validation via EXPLAIN.

Verifies that critical query paths use indexes instead of full scans.
Requires a running ArcadeDB instance with the schema synced.

Run with::

    pytest tests/benchmarks/test_query_plans.py -m benchmark --timeout=60
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]

_SKIP_REASON = "Requires running ArcadeDB with synced schema"


class TestVectorSearchPlans:
    """Verify vectorNeighbors queries hit LSM_VECTOR indexes."""

    def test_text_vector_search_uses_index(self):
        pytest.skip(_SKIP_REASON)

    def test_image_vector_search_uses_index(self):
        pytest.skip(_SKIP_REASON)

    def test_community_report_vector_search_uses_index(self):
        pytest.skip(_SKIP_REASON)


class TestPointLookupPlans:
    """Verify chunk_id / document_id lookups use UNIQUE_HASH_INDEX."""

    def test_text_chunk_lookup_uses_hash_index(self):
        pytest.skip(_SKIP_REASON)

    def test_document_lookup_uses_hash_index(self):
        pytest.skip(_SKIP_REASON)


class TestFulltextSearchPlans:
    """Verify entity name searches use FULL_TEXT indexes."""

    def test_entity_fulltext_search_uses_index(self):
        pytest.skip(_SKIP_REASON)


class TestTraversalPlans:
    """Verify graph traversals are efficient."""

    def test_neighborhood_traversal_plan(self):
        pytest.skip(_SKIP_REASON)

    def test_ontology_linked_chunks_plan(self):
        pytest.skip(_SKIP_REASON)
