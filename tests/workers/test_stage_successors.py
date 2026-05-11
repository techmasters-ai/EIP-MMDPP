"""STAGE_SUCCESSORS forms the canonical sequential pipeline."""
from app.workers.pipeline import (
    STAGE_SUCCESSORS,
    LEDGER_SEQUENTIAL_STAGES,
    LEDGER_FANOUT_STAGES,
    StageEdge,
)


def test_stage_successors_is_a_dag_from_prepare_document():
    """Every key reachable from prepare_document; derive_ontology_graph terminates."""
    visited = set()
    cur = "prepare_document"
    while cur is not None:
        assert cur in STAGE_SUCCESSORS, f"{cur} missing from STAGE_SUCCESSORS"
        assert cur not in visited, f"cycle at {cur}"
        visited.add(cur)
        edge = STAGE_SUCCESSORS[cur]
        cur = edge.next_stage

    # Every key must be reachable (no orphaned entries).
    assert visited == set(STAGE_SUCCESSORS.keys())
    # Terminal stage has no successor.
    assert STAGE_SUCCESSORS["derive_ontology_graph"].next_stage is None
    assert STAGE_SUCCESSORS["derive_ontology_graph"].next_task is None


def test_ledger_sequential_excludes_fanout_stage():
    """Stage 9 is in STAGE_SUCCESSORS but excluded from sequential sweeper set."""
    assert "derive_ontology_graph" in STAGE_SUCCESSORS
    assert "derive_ontology_graph" not in LEDGER_SEQUENTIAL_STAGES
    assert LEDGER_FANOUT_STAGES == ["derive_ontology_graph"]


def test_stage_edge_is_frozen():
    """StageEdge is immutable (dataclass frozen=True)."""
    import dataclasses
    edge = STAGE_SUCCESSORS["prepare_document"]
    assert dataclasses.is_dataclass(edge)
    try:
        edge.next_stage = "tampered"  # type: ignore[misc]
        raise AssertionError("expected FrozenInstanceError")
    except dataclasses.FrozenInstanceError:
        pass


def test_persisted_stage_names_not_function_names():
    """Text-embedding stage uses the persisted name, not the function name."""
    edge = STAGE_SUCCESSORS["derive_picture_descriptions"]
    assert edge.next_stage == "derive_text_embeddings"
    assert edge.next_task == "app.workers.pipeline.derive_text_chunks_and_embeddings"
