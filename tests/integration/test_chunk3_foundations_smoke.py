"""Chunk 3 foundations smoke test.

Verifies every new symbol introduced in Chunk 3 is importable and that
the Chunk 3 invariants hold:
- ``graph_extraction_engine`` feature flag defaults to ``"legacy"`` —
  nothing has switched traffic to the new path yet.
- ``ArcadeDBGraphStore.delete_extraction_layer_graph_sync`` is present.
- ``ProvenanceMetadata`` accepts the new ``pipeline_run_id`` field.

The legacy end-to-end run lives in ``tests/e2e/test_full_pipeline.py``
and is NOT nested here. Running pytest inside pytest via
``subprocess.run`` is fragile (conftest search paths, fixture state,
cwd); CI should invoke the two suites as separate steps. This smoke
test only asserts the Chunk 3 invariants in process.

Task 3.8 of the extraction-refactor plan.
"""
from __future__ import annotations


def test_imports_all_chunk3_symbols():
    """Every new Chunk 3 symbol must be importable."""
    # Task 3.1 + 3.3: GraphStore protocol extension + ProvenanceMetadata field
    from app.services.graph_store import (  # noqa: F401
        GraphStore,
        NodeRecord,
        ProvenanceMetadata,
        RelationshipRecord,
    )

    # Task 3.2: ArcadeDBGraphStore concrete class
    from app.services.arcadedb_graph import ArcadeDBGraphStore  # noqa: F401

    # Task 3.4: app/services/extraction_merge.py — merge + identity + yield
    from app.services.extraction_merge import (  # noqa: F401
        ChunkForDerivation,
        DerivedEdge,
        ExtractionMetadata,
        LogicalIdentity,
        MergedEdgeRecord,
        MergedEntityRecord,
        MergedExtraction,
        PassResult,
        RelationshipRejectionReason,
        YieldStatus,
        build_display_label,
        classify_yield,
        classify_yield_from_counts,
        merge_and_resolve,
    )

    # Task 3.5: StatusSignals on the bundle loader module
    from app.services.ontology_bundles import StatusSignals  # noqa: F401

    # Task 3.6: IngestDispatchResult dispatch type + graph_extraction_engine flag
    from app.workers.dispatch_types import IngestDispatchResult  # noqa: F401
    from app.config import Settings  # noqa: F401

    # Task 3.7: Source bundle fields + new ReingestRequest
    from app.schemas.sources import (  # noqa: F401
        ReingestRequest,
        SourceCreate,
        SourceResponse,
    )


def test_feature_flag_defaults_to_legacy():
    """The 'nothing switched traffic' invariant for Chunk 3.

    At the end of Chunk 3 the feature flag MUST still default to
    'legacy' so merging PR 2 foundations does not auto-switch
    production. The flip to 'bundle_passes' happens during the soak
    procedure in Chunk 4 (§7.4), not here.
    """
    from app.config import Settings

    s = Settings(_env_file=None, postgres_password="test")
    assert s.graph_extraction_engine == "legacy"


def test_delete_extraction_layer_graph_sync_present_on_arcadedb_store():
    """Task 3.2: the concrete rollback primitive is implemented."""
    from app.services.arcadedb_graph import ArcadeDBGraphStore
    assert hasattr(ArcadeDBGraphStore, "delete_extraction_layer_graph_sync")
    assert callable(ArcadeDBGraphStore.delete_extraction_layer_graph_sync)


def test_delete_extraction_layer_graph_sync_on_protocol():
    """Task 3.1: the abstract method is part of the Protocol too."""
    from app.services.graph_store import GraphStore
    assert hasattr(GraphStore, "delete_extraction_layer_graph_sync")


def test_provenance_metadata_accepts_pipeline_run_id():
    """Task 3.3: new Optional[str] field with default None."""
    from app.services.graph_store import ProvenanceMetadata

    m = ProvenanceMetadata(document_id="doc-1", pipeline_run_id="run-1")
    assert m.pipeline_run_id == "run-1"

    # Existing callers that don't pass pipeline_run_id still work
    m2 = ProvenanceMetadata(document_id="doc-2")
    assert m2.pipeline_run_id is None


def test_ingest_dispatch_result_is_frozen():
    """Task 3.6: dispatch return type is immutable."""
    from dataclasses import FrozenInstanceError

    from app.workers.dispatch_types import IngestDispatchResult

    r = IngestDispatchResult(pipeline_run_id="run-1", celery_task_id="task-1")
    assert r.pipeline_run_id == "run-1"
    assert r.celery_task_id == "task-1"
    try:
        r.pipeline_run_id = "run-2"  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("IngestDispatchResult should be frozen")


def test_reingest_request_accepts_bundle_override():
    """Task 3.7: bundle fields flow through the reingest schema."""
    from app.schemas.sources import ReingestRequest

    r = ReingestRequest(
        mode="graph_only",
        ontology_bundle_key="air_defense_v3",
        use_case_key="demo",
    )
    assert r.mode == "graph_only"
    assert r.ontology_bundle_key == "air_defense_v3"
    assert r.use_case_key == "demo"


def test_status_signals_is_dataclass_with_three_fields():
    """Task 3.5: StatusSignals has exactly {snapshot, is_stale, graph_queryable}."""
    from dataclasses import fields

    from app.services.ontology_bundles import StatusSignals

    field_names = {f.name for f in fields(StatusSignals)}
    assert field_names == {"snapshot", "is_stale", "graph_queryable"}
