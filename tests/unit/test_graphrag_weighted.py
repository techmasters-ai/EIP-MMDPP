"""Tests for ontology-weighted GraphRAG community detection."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.graphrag_service import _detect_communities


def test_weighted_communities_group_equipment_hierarchy():
    """Equipment connected by CONTAINS (high weight) should cluster together.

    Graph:
      Tombstone --CONTAINS--> Antenna   (strong, weight 0.90 via scoring_weights)
      Tombstone --MENTIONED_IN--> TM-9-1425  (weak, default 0.70)
      Patriot  --MENTIONED_IN--> TM-9-1425   (weak, default 0.70)

    Expected: Tombstone and Antenna in same community because the strong
    CONTAINS edge (0.90) binds them tighter than the weak MENTIONED_IN edges
    (0.70) that connect through TM-9-1425.
    """
    entities = [
        {"name": "Tombstone", "entity_type": "RADAR_SYSTEM"},
        {"name": "Antenna", "entity_type": "ANTENNA"},
        {"name": "Patriot", "entity_type": "MISSILE_SYSTEM"},
        {"name": "TM-9-1425", "entity_type": "DOCUMENT"},
    ]
    relationships = [
        {"source": "Tombstone", "target": "Antenna", "relationship": "CONTAINS"},
        {"source": "Tombstone", "target": "TM-9-1425", "relationship": "MENTIONED_IN"},
        {"source": "Patriot", "target": "TM-9-1425", "relationship": "MENTIONED_IN"},
    ]

    settings = MagicMock()
    settings.graphrag_max_cluster_size = 10

    communities = _detect_communities(entities, relationships, settings)

    assert len(communities) >= 1, "Should detect at least one community"

    # Find which community Tombstone and Antenna land in
    tombstone_community = None
    antenna_community = None
    for c in communities:
        if "Tombstone" in c["entity_names"]:
            tombstone_community = c["community_id"]
        if "Antenna" in c["entity_names"]:
            antenna_community = c["community_id"]

    assert tombstone_community is not None, "Tombstone should be in a community"
    assert antenna_community is not None, "Antenna should be in a community"
    assert tombstone_community == antenna_community, (
        "Tombstone and Antenna should be in the same community "
        f"(got {tombstone_community} vs {antenna_community})"
    )


def test_edges_carry_ontology_weights():
    """Verify that edges in the NetworkX graph carry ontology scoring_weights.

    We expose the internal graph by patching _detect_communities' local
    networkx import to capture the constructed graph.
    """
    from unittest.mock import patch
    import networkx as nx

    entities = [
        {"name": "A", "entity_type": "EQUIPMENT_SYSTEM"},
        {"name": "B", "entity_type": "COMPONENT"},
        {"name": "C", "entity_type": "DOCUMENT"},
    ]
    relationships = [
        {"source": "A", "target": "B", "relationship": "CONTAINS"},
        {"source": "A", "target": "C", "relationship": "MENTIONED_IN"},
    ]

    settings = MagicMock()
    settings.graphrag_max_cluster_size = 10

    # We'll use _build_weighted_graph (a helper we extract) indirectly.
    # Easier approach: just call _detect_communities, then rebuild the graph
    # the same way to verify weights are applied.
    from app.services.graphrag_service import _build_weighted_graph

    G = _build_weighted_graph(entities, relationships)

    # CONTAINS should have weight 0.90 from scoring_weights
    ab_weight = G["A"]["B"]["weight"]
    assert ab_weight == 0.90, f"CONTAINS edge weight should be 0.90, got {ab_weight}"

    # MENTIONED_IN is not in scoring_weights — should get default 0.70
    ac_weight = G["A"]["C"]["weight"]
    assert ac_weight == 0.70, f"MENTIONED_IN edge weight should be 0.70 (default), got {ac_weight}"
