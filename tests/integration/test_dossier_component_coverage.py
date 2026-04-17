"""E-7 — Dossier filter-list coverage for demoted components.

Per spec §6.1, the 12 is_entity=False components retained their
ontology_name so dossier + query-profile filter lists don't need to
change when an entity demotes. This integration test verifies that
for each of the 12 demoted ontology_names, the graph store can return
at least one vertex once the test-fixture data has been loaded.

Marked ``@pytest.mark.integration`` so the default unit-test run skips
it when ArcadeDB is unreachable or empty — the test inspects the
shape of real ingested data, not mocks. Add test-fixture ingests (or
run against a Chunk-G-migrated corpus) before expecting it to pass.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


# The 12 entities demoted to is_entity=False in Chunk B. Each retained
# its ontology_name, so dossier / query-profile filters pick them up
# without modification (spec §6.1).
DEMOTED_ONTOLOGY_NAMES = [
    "MODULATION",
    "RF_SIGNATURE",
    "RF_EMISSION",
    "SCAN_PATTERN",
    "IF_AMPLIFIER",
    "SPECIFICATION",
    "MISSILE_PERFORMANCE",
    "MISSILE_PHYSICAL_CHARACTERISTICS",
    "PROPULSION_STACK",
    "PROPULSION_STAGE",
    "RADAR_PERFORMANCE",
    "ENGAGEMENT_TIMELINE",
]


def _graph_store_or_skip():
    """Return a graph_store whose schema classes appear populated, or
    skip the test when the backend is unreachable / empty."""
    try:
        from app.db.session import get_graph_store
        graph_store = get_graph_store()
        graph_store.ensure_ready_sync()
    except Exception as exc:
        pytest.skip(f"graph store unreachable: {exc}")
    # Sanity: at least one SECTION vertex exists. If not, there's no
    # ingested data to check against — skip rather than fail.
    try:
        sections = graph_store.count_ontology_nodes_sync("SECTION")
    except Exception as exc:
        pytest.skip(f"count_ontology_nodes_sync unavailable: {exc}")
    if sections <= 0:
        pytest.skip("no SECTION vertices present — ingest test data first")
    return graph_store


@pytest.mark.parametrize("ontology_name", DEMOTED_ONTOLOGY_NAMES)
def test_demoted_component_has_at_least_one_vertex(ontology_name: str):
    """Every demoted component ontology_name should be discoverable via
    graph_store.count_ontology_nodes_sync when the corpus includes
    relevant material. Asserts ≥ 1 vertex per type."""
    graph_store = _graph_store_or_skip()
    count = graph_store.count_ontology_nodes_sync(ontology_name)
    assert count >= 1, (
        f"no {ontology_name} vertices found — dossier filter lists list "
        f"this ontology_name but the corpus has no matching data. If this "
        f"fires for a small test corpus, expand fixtures to cover "
        f"{ontology_name} before green-lighting dossier retrieval on that type."
    )
