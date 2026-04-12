"""Multi-modal ingest pipeline.

Task graph (manifest-first, parallel derivations, idempotent):

    prepare_document  (validate + detect + Docling convert + persist document_elements)
        ↓
    detect_and_translate  (language detection + LLM translation)
        ↓
    derive_document_metadata  (LLM: summary, date, classification, source)
        ↓
    purge_document_derivations  (idempotent cleanup of prior derived data)
        ↓
    derive_picture_descriptions  (LLM: image descriptions with summary context)
        ↓
    derive_text_chunks_and_embeddings  (text chunking + BGE embedding)
        ↓
    derive_image_embeddings  (SigLIP2 image embedding)
        ↓
    derive_ontology_graph  (docling-graph entity/relationship extraction)
        ↓
    collect_derivations  (post-derivation checkpoint)
        ↓
    derive_structure_links  (needs embedding output committed)
        ↓
    derive_canonicalization  (entity alias resolution)
        ↓
    finalize_document
"""

import hashlib
import logging
import uuid
from typing import Optional

import httpx
import redis as redis_lib
from celery import chain
from celery import chain as celery_chain  # alias used by reingest_graph_only (patchable at module level)
from celery.exceptions import Retry as CeleryRetry, SoftTimeLimitExceeded
from celery.signals import worker_ready

from app.workers.celery_app import celery_app

# Unicode chars that cause NaN in bge-m3 embeddings (non-breaking hyphens,
# en/em dashes, narrow no-break spaces).  Normalize at ingest time so they
# never enter the knowledge graph or embedding pipeline.
_UNICODE_NORMALIZE = str.maketrans({
    "\u2011": "-",   # non-breaking hyphen
    "\u2010": "-",   # hyphen
    "\u2012": "-",   # figure dash
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u202f": " ",   # narrow no-break space
    "\u00a0": " ",   # no-break space
})


from app.workers._db import get_worker_db as _get_db
from app.config import get_settings
from app.services.redis_utils import get_redis

logger = logging.getLogger(__name__)
settings = get_settings()

# Shared Redis client (singleton connection pool) — also used for Docling
# concurrency locks and the post-ingest community trigger counter.
_redis_client = get_redis()


# ---------------------------------------------------------------------------
# Chunk 4 (PR 2 orchestrator rewrite) scaffolding — spec §5.4 tracker gate
#
# GraphWriteTracker is the worker-local rollback gate. Phase helpers call
# .mark() immediately before the first graph_store mutation in each phase,
# so failures that happen BEFORE the first mutation leave the flag False
# and the rollback primitive is skipped. This prevents "rollback a
# document that was never actually written to" bugs on early failures
# (gate check, merge, manifest load, etc.).
#
# The helper stubs below let downstream Chunk 4 tasks (4.3, 4.4, 4.5,
# 4.6, 4.7) import these names without breaking the module's import
# graph. Each stub raises NotImplementedError with a task-ID back-pointer
# so partial runs are loud. Task 4.6 is the one that rewrites
# derive_ontology_graph to actually call these helpers.
# ---------------------------------------------------------------------------

import sqlalchemy as sa  # noqa: E402 — used by _write_stage_run partial-index upsert
from dataclasses import dataclass as _dataclass  # noqa: E402
from dataclasses import dataclass  # noqa: E402

from app.services.extraction_merge import classify_yield  # noqa: E402
from app.services.extraction_merge import classify_yield_from_counts  # noqa: E402
from app.services.extraction_merge import build_display_label  # noqa: E402
from app.services.extraction_merge import YieldStatus  # noqa: E402
from app.services.extraction_merge import merge_and_resolve  # noqa: E402
from app.services.ontology_bundles import load_bundle_manifest  # noqa: E402
from app.services.ontology_templates import load_ontology  # noqa: E402
from app.db.session import get_graph_store  # noqa: E402
from app.db.session import get_sync_session  # noqa: E402


# --- Custom exception types for the single-pass dispatcher (spec §5.5 + §6.5) ---

class PassRetryable(Exception):
    """Raised by _call_extract_pass for transport errors, timeouts, HTTP 5xx,
    partial response parse errors, and TransientOllamaBusyError.
    _run_single_pass retries up to pass_max_retries with exponential backoff."""


class PassTerminal(Exception):
    """Raised by _call_extract_pass / _parse_pass_response for HTTP 4xx, Pydantic
    validation failure after salvage, UnknownBundleOrPassError, ManifestValidationError,
    and worker code bugs.  Terminal — no retry."""


class IngestFailed(Exception):
    """Raised by _run_single_pass when a required pass exhausts retries or hits a
    terminal failure.  The caller (derive_ontology_graph) uses this as its
    gate-failure marker."""


class WorkerInvariantError(Exception):
    """Raised by check_required_pass_gate when a required pass has NO StageRun at
    all — this is a worker bug, not a pass failure."""


@dataclass
class GateResult:
    passed: bool
    failures: list  # list[tuple[str, str]] — (pass_name, reason_description)


@_dataclass
class GraphWriteTracker:
    """Worker-local rollback gate per spec §5.4.

    Phase helpers in the new derive_ontology_graph branch (Task 4.6) call
    ``.mark()`` immediately before their first graph_store mutation. If
    the orchestrator catches an exception, it consults
    ``any_mutation_attempted`` to decide whether to invoke
    ``_attempt_rollback``. Failures before the first mark are rollback-
    free because no graph state changed.
    """
    any_mutation_attempted: bool = False

    def mark(self) -> None:
        self.any_mutation_attempted = True


# --- Orchestrator helper stubs (filled in by later Chunk 4 tasks) -----------

def _delete_extraction_layer_graph(document_id: str) -> None:
    """Thin wrapper over graph_store.delete_extraction_layer_graph_sync."""
    graph_store = get_graph_store()
    graph_store.delete_extraction_layer_graph_sync(str(document_id))


def _attempt_rollback(document_id: str) -> str:
    """Calls _delete_extraction_layer_graph; returns empty string on success
    or a diagnostic suffix on failure to concatenate into the stage row's
    error_message."""
    try:
        _delete_extraction_layer_graph(document_id)
        return ""
    except Exception as rollback_exc:
        logger.error("rollback during failure handling also failed: %s", rollback_exc)
        return f"; ROLLBACK_ALSO_FAILED: {rollback_exc}"


def _rel_to_dict(rel_tuple) -> dict:
    """Serialise a rejected_edges tuple (source_pass, raw_rel, reason) to a
    JSON-safe dict for the metrics blob."""
    source_pass, raw_rel, reason = rel_tuple
    reason_val = reason.value if hasattr(reason, "value") else str(reason)
    return {
        "source_pass": source_pass,
        "reason": reason_val,
    }


def _build_rejection_sample(rejected_edges, max_per_pass: int = 20) -> list[dict]:
    """Return up to max_per_pass rejection dicts per (pass, reason) combo.

    Groups by source_pass first so no single pass drowns out others.
    """
    from collections import defaultdict
    by_pass: dict[str, list] = defaultdict(list)
    for tup in rejected_edges:
        source_pass = tup[0]
        by_pass[source_pass].append(tup)

    sample: list[dict] = []
    for pass_name, tups in by_pass.items():
        for tup in tups[:max_per_pass]:
            sample.append(_rel_to_dict(tup))
    return sample


def _build_pass_outcomes_rollup(session, pipeline_run_id) -> dict:
    """Query v_latest_pass_attempts for pass-level metrics.

    Returns a dict keyed by pass_name with counts. Returns {} if no rows.
    """
    from sqlalchemy import text

    try:
        rows = session.execute(
            text(
                "SELECT pass_name, execution_status, yield_status, "
                "primary_entities_extracted, bridge_entities_extracted, "
                "relationships_extracted, relationships_rejected "
                "FROM ingest.v_latest_pass_attempts "
                "WHERE pipeline_run_id = :run_id"
            ),
            {"run_id": str(pipeline_run_id)},
        ).all()
    except Exception:
        return {}

    rollup: dict = {}
    for row in rows:
        rollup[row.pass_name] = {
            "execution_status": row.execution_status,
            "yield_status": row.yield_status,
            "primary_entities_extracted": row.primary_entities_extracted or 0,
            "bridge_entities_extracted": row.bridge_entities_extracted or 0,
            "relationships_extracted": row.relationships_extracted or 0,
            "relationships_rejected": row.relationships_rejected or 0,
        }
    return rollup


def _serialize_for_audit(merged, manifest) -> dict:
    """Build the DocumentGraphExtraction.graph_json audit blob.

    Distinguishes primary vs bridge entities using the manifest's
    bridge_entity_types lists (a type is a bridge iff it appears in
    ANY pass's bridge_entity_types).
    """
    bridge_types: set[str] = set()
    for pass_def in manifest.passes:
        bridge_types.update(pass_def.bridge_entity_types or [])

    primary_total = 0
    bridge_total = 0
    entity_count_by_type: dict[str, int] = {}
    for e in merged.entities:
        etype = e.identity.entity_type
        entity_count_by_type[etype] = entity_count_by_type.get(etype, 0) + 1
        if etype in bridge_types:
            bridge_total += 1
        else:
            primary_total += 1

    edges_accepted = len(merged.edges)
    edges_rejected = len(merged.rejected_edges)

    # Rejection reasons summary
    rejection_reasons: dict[str, int] = {}
    for tup in merged.rejected_edges:
        reason = tup[2]
        reason_val = reason.value if hasattr(reason, "value") else str(reason)
        rejection_reasons[reason_val] = rejection_reasons.get(reason_val, 0) + 1

    return {
        "primary_entities_total": primary_total,
        "bridge_entities_total": bridge_total,
        "entity_count_by_type": entity_count_by_type,
        "edges_accepted": edges_accepted,
        "edges_rejected": edges_rejected,
        "rejection_reasons": rejection_reasons,
    }


def _write_pipeline_run_metrics(pipeline_run_id, merged, manifest) -> None:
    """Populates PipelineRun.metrics with the quality-signal blob.

    Spec §6.6. Queries v_latest_pass_attempts for per-pass outcomes,
    computes document_extraction_anomaly (all core passes non-HIT),
    overall_relationship_rejection_ratio, and a rejection sample.
    """
    from app.models.ingest import PipelineRun

    with get_sync_session() as session:
        run = session.get(PipelineRun, pipeline_run_id)

        # Per-pass rollup from the view
        pass_outcomes = _build_pass_outcomes_rollup(session, pipeline_run_id)

        # document_extraction_anomaly: True if NO pass achieved HIT
        # (all passes ended EMPTY or BRIDGES_ONLY or SKIPPED/FAILED)
        any_hit = any(
            v.get("yield_status") == "HIT"
            for v in pass_outcomes.values()
        )
        document_extraction_anomaly = not any_hit if pass_outcomes else False

        # Count passes in DEGRADED state
        pass_degraded_count = sum(
            1 for v in pass_outcomes.values() if v.get("yield_status") == "DEGRADED"
        )

        # Overall rejection ratio across all merged edges
        total_extracted = len(merged.edges)
        total_rejected = len(merged.rejected_edges)
        total_rels = total_extracted + total_rejected
        overall_rejection_ratio = (
            round(total_rejected / total_rels, 4) if total_rels > 0 else 0.0
        )

        rejected_sample = _build_rejection_sample(merged.rejected_edges)

        bundle_key = getattr(manifest, "bundle_key", None) or getattr(run, "ontology_bundle_key", None)

        run.metrics = {
            "pass_outcomes": pass_outcomes,
            "document_extraction_anomaly": document_extraction_anomaly,
            "pass_degraded_count": pass_degraded_count,
            "overall_relationship_rejection_ratio": overall_rejection_ratio,
            "rejected_relationships_sample": rejected_sample,
            "bundle_legacy": False,
            "bundle_key_display": bundle_key or "",
        }
        session.commit()


def _run_single_pass(
    *,
    pipeline_run_id,
    pass_def,
    manifest,
    ontology: dict,
    bundle_key: str,
    doc_json: dict,
    pass_results: dict,
    upstream_refs: dict,
    document_id: str,
) -> None:
    """Per-pass dispatcher with retry, skip, and required-pass gate handling.

    Spec §5.5 + §6.5.  On success, populates pass_results[pass_def.name] and
    optionally extends upstream_refs for downstream passes that depend on this
    one.  Writes a StageRun row for every attempt (including failures) so
    operators can audit retry history.
    """
    max_retries = getattr(settings, "pass_max_retries", 3)
    attempt = 1

    while True:
        if _should_skip(pass_def, upstream_refs, ontology):
            _write_stage_run(
                pipeline_run_id=pipeline_run_id,
                pass_def=pass_def,
                attempt=attempt,
                execution_status="SKIPPED",
                yield_status=None,
                skip_reason="NO_UPSTREAM_ENDPOINTS",
                counts=None,
                error=None,
            )
            return

        try:
            request_body = _build_extract_pass_request(
                bundle_key=bundle_key,
                pass_def=pass_def,
                doc_json=doc_json,
                upstream_refs=(
                    upstream_refs
                    if pass_def.input_mode == "document_plus_entity_refs"
                    else None
                ),
            )
            response = _call_extract_pass(
                request_body,
                timeout=settings.docling_graph_timeout,
            )
            pass_result = _parse_pass_response(response, pass_def, manifest)

        except PassRetryable as exc:
            _write_stage_run(
                pipeline_run_id=pipeline_run_id,
                pass_def=pass_def,
                attempt=attempt,
                execution_status="FAILED",
                yield_status=None,
                skip_reason=None,
                counts=None,
                error=str(exc),
            )
            if attempt >= max_retries:
                if pass_def.required:
                    raise IngestFailed(
                        f"Required pass {pass_def.name} exhausted retries"
                    ) from exc
                return
            _backoff(attempt)
            attempt += 1
            continue

        except PassTerminal as exc:
            _write_stage_run(
                pipeline_run_id=pipeline_run_id,
                pass_def=pass_def,
                attempt=attempt,
                execution_status="FAILED",
                yield_status=None,
                skip_reason=None,
                counts=None,
                error=str(exc),
            )
            if pass_def.required:
                raise IngestFailed(
                    f"Required pass {pass_def.name} terminal failure"
                ) from exc
            return

        yield_status_val = classify_yield(pass_result, pass_def, ontology)
        # classify_yield returns a YieldStatus enum; normalise to string
        yield_str = (
            yield_status_val.value
            if hasattr(yield_status_val, "value")
            else str(yield_status_val)
        )
        counts = _count_pass_output(pass_result, pass_def, ontology)
        _write_stage_run(
            pipeline_run_id=pipeline_run_id,
            pass_def=pass_def,
            attempt=attempt,
            execution_status="COMPLETE",
            yield_status=yield_str,
            skip_reason=None,
            counts=counts,
            error=None,
        )
        pass_results[pass_def.name] = pass_result

        if _any_downstream_pass_depends_on(manifest, pass_def.name):
            _extend_upstream_refs(upstream_refs, pass_result, pass_def, ontology)
        return


def _should_skip(pass_def, upstream_refs: dict, ontology: dict) -> bool:
    """Return True iff the pass should be skipped per spec §5.5.

    Only relationships_only passes with skip_if_no_upstream_endpoints=True are
    candidates.  The check walks pass_def.depends_on, collects the entity types
    present in upstream_refs (filtered to refs whose pass_origin is in the
    declared depends_on set), then tests whether any (source, rel, target)
    triple in ontology["validation_matrix"] can be satisfied by those types.
    """
    if pass_def.kind != "relationships_only":
        return False
    if not getattr(pass_def, "skip_if_no_upstream_endpoints", False):
        return False

    declared_deps = set(pass_def.depends_on)
    available_types: set[str] = {
        ref.entity_type
        for ref in upstream_refs.values()
        if getattr(ref, "pass_origin", None) in declared_deps
    }
    if not available_types:
        return True

    allowed_rels = set(pass_def.extracted_relationship_types)
    for row in ontology.get("validation_matrix", []):
        if row.get("relationship") not in allowed_rels:
            continue
        if (row.get("source") in available_types
                and row.get("target") in available_types):
            return False

    return True


def _apply_post_merge_yield_updates(pipeline_run_id, merged) -> None:
    """Recomputes yield_status per-pass after merge_and_resolve rejects edges.

    Spec §5.4 + §6.2. The only allowed transition is HIT → DEGRADED; this
    function does NOT promote EMPTY/BRIDGES_ONLY passes upward.

    Updates relationships_extracted and relationships_rejected counts on
    each COMPLETE StageRun row so they reflect post-merge reality.
    """
    from app.models.ingest import StageRun

    # Build per-pass edge counts from the merged result
    extracted_by_pass: dict[str, int] = {}
    for edge in merged.edges:
        pass_name = edge.source_pass
        extracted_by_pass[pass_name] = extracted_by_pass.get(pass_name, 0) + 1

    rejected_by_pass: dict[str, int] = {}
    for tup in merged.rejected_edges:
        pass_name = tup[0]
        rejected_by_pass[pass_name] = rejected_by_pass.get(pass_name, 0) + 1

    with get_sync_session() as session:
        rows = (
            session.query(StageRun)
            .filter(
                StageRun.pipeline_run_id == pipeline_run_id,
                StageRun.stage_name == "derive_ontology_graph",
                StageRun.pass_name.isnot(None),
                StageRun.execution_status == "COMPLETE",
            )
            .all()
        )

        for row in rows:
            pass_name = row.pass_name
            extracted = extracted_by_pass.get(pass_name, 0)
            rejected = rejected_by_pass.get(pass_name, 0)

            # Always update the relationship counts to reflect post-merge reality
            row.relationships_extracted = extracted
            row.relationships_rejected = rejected

            # Only allow HIT → DEGRADED transition (spec §6.2)
            if row.yield_status == "HIT":
                new_yield = classify_yield_from_counts(
                    primary=row.primary_entities_extracted or 0,
                    bridge=row.bridge_entities_extracted or 0,
                    extracted_rels=extracted,
                    rejected_rels=rejected,
                )
                if new_yield == YieldStatus.DEGRADED:
                    row.yield_status = "DEGRADED"

        session.commit()


def _import_graph_phase_nodes(merged, ontology, document_id, tracker):
    """Spec §5.6 phase 2 — node upsert.

    Builds the full NodeRecord list in pure Python FIRST so that any
    pre-mutation failure (e.g. build_display_label raising) leaves
    tracker.any_mutation_attempted == False and the rollback gate
    correctly skips.  tracker.mark() is called AFTER the list is built
    and IMMEDIATELY before the first graph_store mutation.

    Task 4.4.
    """
    from app.services.graph_store import NodeRecord, ProvenanceMetadata

    # Build all records in pure Python first. If this raises, tracker
    # stays False and the rollback gate correctly skips.
    node_records = [
        NodeRecord(
            entity_type=e.identity.entity_type,
            identity_fields=e.identity.as_upsert_identity_dict(),
            name=build_display_label(
                e.identity.entity_type,
                e.identity.identity_values_dict(),
                e.properties,
            ),
            properties=e.properties,
            extraction_confidence=e.confidence,
        )
        for e in merged.entities
    ]

    provenance = ProvenanceMetadata(
        document_id=document_id,
        pipeline_run_id=merged.pipeline_run_id,
    )

    tracker.mark()
    graph_store = get_graph_store()
    node_rids: list[str] = graph_store.upsert_nodes_batch_sync(node_records, provenance)

    identity_to_rid = dict(
        zip(
            (e.identity for e in merged.entities),
            node_rids,
            strict=True,
        )
    )
    return identity_to_rid


def _import_graph_phase_domain_edges(merged, ontology, tracker) -> None:
    """Spec §5.6 phase 3 — domain edge upsert (identity-based).

    Builds RelationshipRecord list in pure Python, calls tracker.mark()
    defensively (idempotent — phase 2 likely already marked), then
    upserts.  An empty edges list still calls upsert_relationships_batch_sync
    with an empty list to match graph_store semantics.

    Task 4.4.
    """
    from app.services.graph_store import RelationshipRecord, ProvenanceMetadata

    rel_records = [
        RelationshipRecord(
            from_type=e.from_identity.entity_type,
            from_identity=e.from_identity.as_upsert_identity_dict(),
            to_type=e.to_identity.entity_type,
            to_identity=e.to_identity.as_upsert_identity_dict(),
            rel_type=e.rel_type,
            extraction_confidence=e.confidence,
        )
        for e in merged.edges
    ]

    provenance = ProvenanceMetadata(
        document_id=merged.document_id,
        pipeline_run_id=merged.pipeline_run_id,
    )

    tracker.mark()  # idempotent — phase 2 likely already marked
    graph_store = get_graph_store()
    graph_store.upsert_relationships_batch_sync(rel_records, provenance)


def _import_graph_phase_structural_edges(
    merged, identity_to_rid, document_id, pipeline_run_id, tracker,
) -> None:
    """Spec §5.6 phase 4 — derived structural edges (MENTIONED_IN, etc.).

    Loads TextChunk vertices from ArcadeDB, looks up the structural
    Document vertex RID, calls derive_rules.derive_structural_edges,
    then writes each DerivedEdge via create_structural_edge_sync.

    tracker.mark() is called inside the loop so an empty derived list is
    a true no-op (tracker state unchanged).

    Task 4.4.
    """
    from ontology_bundles.air_defense_v3 import derive_rules

    chunks = _load_chunks_for_derivation(document_id)
    document_rid = _get_structural_document_rid(document_id)

    derived = derive_rules.derive_structural_edges(
        merged=merged,
        identity_to_rid=identity_to_rid,
        chunks=chunks,
        document_rid=document_rid,
    )

    graph_store = get_graph_store()
    for edge in derived:
        tracker.mark()  # idempotent
        graph_store.create_structural_edge_sync(
            from_id=edge.from_id,
            to_id=edge.to_id,
            rel_type=edge.rel_type,
            properties={
                "document_id": document_id,
                "pipeline_run_id": pipeline_run_id,
                "extraction_confidence": edge.confidence,
                "source": "derive_rules",
            },
        )


def _update_document_pipeline_status(document_id: str, new_status: str) -> None:
    """Writes Document.pipeline_status. Spec §5.4 + §6.9."""
    from app.models.ingest import Document
    db = _get_db()
    try:
        doc = db.get(Document, uuid.UUID(str(document_id)))
        if doc:
            doc.pipeline_status = new_status
            db.commit()
    finally:
        db.close()


def check_required_pass_gate(pipeline_run_id) -> GateResult:
    """Required-pass gate per spec §6.4.

    Queries the latest StageRun per required pass.  COMPLETE and
    authorized-SKIPPED passes pass; FAILED and unauthorized-SKIPPED passes
    accumulate as failures.  Missing StageRun rows for required passes are a
    worker invariant violation (bug, not a pass failure) and raise
    WorkerInvariantError.
    """
    from app.models.ingest import PipelineRun, StageRun

    db = _get_db()
    try:
        run = db.get(PipelineRun, uuid.UUID(str(pipeline_run_id)))
        if run is None:
            raise WorkerInvariantError(f"PipelineRun {pipeline_run_id} not found")
        manifest = load_bundle_manifest(run.ontology_bundle_key)
        required_passes = [p.name for p in manifest.passes if p.required]
        failures: list[tuple[str, str]] = []

        for pass_name in required_passes:
            latest = (
                db.query(StageRun)
                .filter(
                    StageRun.pipeline_run_id == uuid.UUID(str(pipeline_run_id)),
                    StageRun.stage_name == "derive_ontology_graph",
                    StageRun.pass_name == pass_name,
                )
                .order_by(StageRun.attempt.desc())
                .first()
            )
            if latest is None:
                raise WorkerInvariantError(
                    f"Required pass {pass_name} has no StageRun"
                )
            if latest.execution_status == "COMPLETE":
                continue
            if latest.execution_status == "FAILED":
                failures.append((pass_name, f"FAILED: {latest.error_message}"))
                continue
            if latest.execution_status == "SKIPPED":
                if latest.skip_reason in {"NO_UPSTREAM_ENDPOINTS"}:
                    continue
                failures.append(
                    (pass_name, f"unauthorized skip: {latest.skip_reason}")
                )
                continue
    finally:
        db.close()

    return GateResult(passed=(not failures), failures=failures)


def _build_docling_document_json(document_id: str) -> dict:
    """Load the persisted docling_document.json for a document from MinIO.

    Mirrors the pattern used at multiple legacy call sites in this module.
    When the new derive_ontology_graph branch (Task 4.6) calls this, it
    expects the enriched DoclingDocument JSON that earlier stages
    (prepare_document, derive_picture_descriptions) persisted.

    Task 4.4.
    """
    import json as _json_mod
    from app.services.storage import download_bytes_sync

    _s = get_settings()
    base_key = f"artifacts/{document_id}"
    raw = download_bytes_sync(
        _s.minio_bucket_derived,
        f"{base_key}/docling_document.json",
    )
    return _json_mod.loads(raw)


def _load_chunks_for_derivation(document_id: str) -> list:
    """Load TextChunk vertices from ArcadeDB and convert to ChunkForDerivation DTOs.

    The derive_rules MENTIONED_IN logic needs the vertex RID + normalized
    text for substring matching.  Postgres retrieval.text_chunks does NOT
    store the ArcadeDB RID, so this queries ArcadeDB directly.

    ArcadeDB TextChunk vertex schema (see arcadedb_graph.py _build_text_chunk_sql):
      - chunk_id, text, document_id  (plus optional embedding + props)

    Task 4.4.
    """
    import re
    from app.services.extraction_merge import ChunkForDerivation

    graph_store = get_graph_store()
    rows = graph_store._client.query_sync(
        graph_store._database, "sql",
        "SELECT @rid AS rid, text FROM TextChunk WHERE document_id = :doc_id",
        params={"doc_id": document_id},
    )

    whitespace_re = re.compile(r"\s+")
    chunks = []
    for row in rows:
        raw_text = row.get("text") or ""
        text_normalized = whitespace_re.sub(" ", raw_text.strip().lower())
        chunks.append(ChunkForDerivation(
            rid=row["rid"],
            text_normalized=text_normalized,
        ))
    return chunks


def _get_structural_document_rid(document_id: str) -> str:
    """Look up the ArcadeDB @rid of the structural Document vertex.

    The vertex is guaranteed to exist at this point because the
    prepare_document stage created it earlier in the pipeline.  If it
    is missing, that is a worker-invariant violation (bug, not a pass
    failure).

    Task 4.4.
    """
    graph_store = get_graph_store()
    rows = graph_store._client.query_sync(
        graph_store._database, "sql",
        "SELECT @rid AS rid FROM Document WHERE document_id = :doc_id",
        params={"doc_id": document_id},
    )
    if not rows:
        raise WorkerInvariantError(
            f"No structural Document vertex found for document_id={document_id}"
        )
    return rows[0]["rid"]


def _upsert_document_graph_extraction(
    *,
    document_id: str,
    pipeline_run_id: str,
    run,
    merged,
    manifest,
) -> None:
    """Writes the DocumentGraphExtraction snapshot row per spec §5.7.

    graph_json now carries an audit blob (entity/edge counts by type,
    rejection reasons, primary vs bridge totals) rather than a serialised
    NetworkX graph. Accepts manifest so _serialize_for_audit can distinguish
    primary from bridge entities using pass_def.bridge_entity_types lists.

    Upserts: queries by document_id; inserts a new row if none exists,
    otherwise updates the existing row in place.
    """
    import datetime
    from app.models.ingest import DocumentGraphExtraction

    audit_blob = _serialize_for_audit(merged, manifest)

    with get_sync_session() as session:
        existing = (
            session.query(DocumentGraphExtraction)
            .filter_by(document_id=document_id)
            .first()
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        bundle_key = getattr(manifest, "bundle_key", None) or getattr(run, "ontology_bundle_key", None)
        ontology_name = getattr(manifest, "ontology_name", None) or getattr(run, "ontology_name", None)
        ontology_version = getattr(manifest, "ontology_version", None) or getattr(run, "ontology_version", None)
        extraction_profile_version = (
            getattr(manifest, "extraction_profile_version", None)
            or getattr(run, "extraction_profile_version", None)
        )

        if existing is None:
            row = DocumentGraphExtraction(
                document_id=document_id,
                pipeline_run_id=pipeline_run_id,
                graph_json=audit_blob,
                status="COMPLETE",
                updated_at=now,
                ontology_bundle_key=bundle_key,
                ontology_name=ontology_name,
                ontology_version=ontology_version,
                use_case_key=getattr(run, "use_case_key", None),
                extraction_profile_version=extraction_profile_version,
            )
            session.add(row)
        else:
            existing.pipeline_run_id = pipeline_run_id
            existing.graph_json = audit_blob
            existing.status = "COMPLETE"
            existing.updated_at = now
            existing.ontology_bundle_key = bundle_key
            existing.ontology_name = ontology_name
            existing.ontology_version = ontology_version
            existing.use_case_key = getattr(run, "use_case_key", None)
            existing.extraction_profile_version = extraction_profile_version

        session.commit()


def _normalize_text(text: str | None) -> str | None:
    """Replace problematic Unicode chars with ASCII equivalents."""
    if text is None:
        return None
    return text.translate(_UNICODE_NORMALIZE)

# Pipeline status constants
STATUS_PROCESSING = "PROCESSING"
STATUS_COMPLETE = "COMPLETE"
STATUS_PARTIAL_COMPLETE = "PARTIAL_COMPLETE"
STATUS_FAILED = "FAILED"
STATUS_PENDING_REVIEW = "PENDING_HUMAN_REVIEW"


@worker_ready.connect
def _cleanup_stale_runs(sender, **kwargs):
    """Reset documents stuck in PROCESSING from prior worker crashes."""
    from app.db.session import get_sync_session
    from sqlalchemy import text

    db = get_sync_session()
    try:
        result = db.execute(text("""
            UPDATE ingest.documents
            SET pipeline_status = 'PENDING'
            WHERE pipeline_status = 'PROCESSING'
            RETURNING id
        """))
        stale_ids = [str(r[0]) for r in result.fetchall()]

        db.execute(text("""
            UPDATE ingest.stage_runs
            SET status = 'PENDING'
            WHERE status = 'RUNNING'
        """))

        # Mark stale pipeline_runs as FAILED so queued tasks from those runs
        # will be caught by the supersession guard in prepare_document.
        db.execute(text("""
            UPDATE ingest.pipeline_runs
            SET status = 'FAILED', finished_at = NOW()
            WHERE status = 'PROCESSING'
        """))

        db.commit()
        if stale_ids:
            # Also clear Redis singleflight locks for stale documents
            for stale_id in stale_ids:
                _redis_client.delete(f"prepare:{stale_id}")
            logger.info("Cleaned up %d stale PROCESSING documents (+ Redis locks): %s", len(stale_ids), stale_ids)

        # Clear stale Docling concurrency permits — these are Redis locks that
        # may be held by a previous worker that died mid-conversion.
        docling_permits_cleared = 0
        for i in range(settings.docling_concurrency):
            key = f"docling:permit:{i}"
            if _redis_client.delete(key):
                docling_permits_cleared += 1
        if docling_permits_cleared:
            logger.info("Cleared %d stale Docling concurrency permits", docling_permits_cleared)
    except Exception as e:
        logger.warning("Stale document cleanup failed: %s", e)
        db.rollback()
    finally:
        db.close()


def _dedupe_extracted_elements(chunks: list) -> tuple[list, int]:
    """Remove exact duplicate extracted elements conservatively.

    Dedup key: (modality, page_number, section_path, content_text, bounding_box).
    For image/schematic elements, the dedup key also includes a hash of the raw
    image bytes so that different images on the same page are NOT treated as
    duplicates (they often share empty text and null bounding boxes).
    Preserves first-occurrence order. Keeps duplicates across different pages/sections.
    """
    seen: set[str] = set()
    result = []
    for chunk in chunks:
        section_path = (getattr(chunk, "metadata", None) or {}).get("section_path", "")
        key = f"{chunk.modality}|{chunk.page_number}|{section_path}|{chunk.chunk_text}|{chunk.bounding_box}"
        # Images often have empty text + null bbox, so two different images on
        # the same page produce identical keys.  Include a hash of the raw
        # image bytes to distinguish them.
        if chunk.modality in ("image", "schematic") and getattr(chunk, "raw_image_bytes", None):
            import hashlib
            img_hash = hashlib.sha256(chunk.raw_image_bytes).hexdigest()[:12]
            key = f"{key}|{img_hash}"
        if key in seen:
            continue
        seen.add(key)
        result.append(chunk)
    return result, len(chunks) - len(result)


def _synthesize_standalone_image(file_bytes: bytes, mime_type: str) -> list | None:
    """Synthesize a single image element for standalone image files.

    When Docling returns 0 elements for an image MIME type, create an
    ExtractedChunk so the pipeline can CLIP-embed and LLM-describe it.

    Returns a list with one ExtractedChunk, or None if mime_type is not an image.
    """
    if not mime_type.startswith("image/"):
        return None

    from app.services.extraction import ExtractedChunk

    ext = mime_type.split("/")[-1]  # e.g. "png", "jpeg", "tiff"
    return [ExtractedChunk(
        chunk_text="",
        modality="image",
        page_number=1,
        raw_image_bytes=file_bytes,
        metadata={"label": "picture", "ext": ext},
        bounding_box=None,
    )]


def _update_document_status(
    document_id: str,
    status: str,
    stage: Optional[str] = None,
    error: Optional[str] = None,
    failed_stages: Optional[list[str]] = None,
) -> None:
    """Update document pipeline status in the database."""
    from sqlalchemy import update
    from app.models.ingest import Document

    db = _get_db()
    try:
        values = {
            "pipeline_status": status,
            "pipeline_stage": stage,
            "error_message": error,  # None clears previous errors
        }
        if failed_stages is not None:
            values["failed_stages"] = failed_stages

        db.execute(
            update(Document)
            .where(Document.id == uuid.UUID(document_id))
            .values(**values)
        )
        db.commit()
    finally:
        db.close()


def _deterministic_artifact_id(document_id: str, element_uid: str) -> uuid.UUID:
    """Generate a deterministic artifact UUID from document_id + element_uid.

    Uses uuid5 with URL namespace so the same (document_id, element_uid)
    pair always produces the same artifact ID.  This replaces the old
    positional zip-linking approach.
    """
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{element_uid}")


def _persist_extraction_results(db, document_id: str, chunks, element_uids: list[str] | None = None) -> list[uuid.UUID]:
    """Persist ExtractedChunk list as Artifact rows. Stores images in MinIO.

    If *element_uids* is provided (one per chunk, same order), each Artifact
    gets a deterministic ID derived from document_id + element_uid.

    Uses ON CONFLICT DO UPDATE so reingest/retry with the same deterministic
    IDs is idempotent (updates mutable fields, preserves classification).

    Returns the list of artifact IDs (in chunk order).
    """
    import uuid as uuid_mod
    from sqlalchemy import func as sa_func
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from app.models.ingest import Artifact
    from app.services.storage import upload_bytes_sync

    artifact_ids: list[uuid.UUID] = []
    for idx, chunk in enumerate(chunks):
        # Compute artifact_id first so image storage keys are deterministic
        artifact_id = (
            _deterministic_artifact_id(document_id, element_uids[idx])
            if element_uids
            else uuid_mod.uuid4()
        )

        storage_bucket = None
        storage_key = None

        if chunk.raw_image_bytes:
            ext = chunk.metadata.get("ext", "png")
            img_key = f"artifacts/{document_id}/images/{artifact_id}.{ext}"
            upload_bytes_sync(
                chunk.raw_image_bytes,
                settings.minio_bucket_derived,
                img_key,
                content_type=f"image/{ext}",
            )
            storage_bucket = settings.minio_bucket_derived
            storage_key = img_key

        values = {
            "id": artifact_id,
            "document_id": uuid.UUID(document_id),
            "artifact_type": chunk.modality,
            "content_text": chunk.chunk_text,
            "content_metadata": chunk.metadata,
            "storage_bucket": storage_bucket,
            "storage_key": storage_key,
            "page_number": chunk.page_number,
            "bounding_box": chunk.bounding_box,
            "ocr_confidence": chunk.ocr_confidence,
            "ocr_engine": chunk.ocr_engine,
            "requires_human_review": chunk.requires_human_review,
        }

        stmt = pg_insert(Artifact).values(**values)
        stmt = stmt.on_conflict_do_update(
            constraint="artifacts_pkey",
            set_={
                "artifact_type": stmt.excluded.artifact_type,
                "content_text": stmt.excluded.content_text,
                "content_metadata": stmt.excluded.content_metadata,
                "storage_bucket": stmt.excluded.storage_bucket,
                "storage_key": stmt.excluded.storage_key,
                "page_number": stmt.excluded.page_number,
                "bounding_box": stmt.excluded.bounding_box,
                "ocr_confidence": stmt.excluded.ocr_confidence,
                "ocr_engine": stmt.excluded.ocr_engine,
                "requires_human_review": stmt.excluded.requires_human_review,
                "updated_at": sa_func.now(),
            },
        )
        db.execute(stmt)
        artifact_ids.append(artifact_id)

    return artifact_ids


def _legacy_extract(db, document_id: str, doc, file_bytes: bytes) -> None:
    """Fallback: run legacy extraction (pdfplumber/pymupdf/tesseract) inline.

    Creates both Artifact and DocumentElement rows so downstream derivation
    tasks (embedding, graph, structure links) have input to work with.
    """
    from app.services.extraction import extract_pdf, extract_docx, extract_image, extract_txt
    from app.models.ingest import DocumentElement

    mime = doc.mime_type or ""
    chunks = []

    if "pdf" in mime:
        chunks = extract_pdf(file_bytes)
    elif "wordprocessingml" in mime or "msword" in mime:
        chunks = extract_docx(file_bytes)
    elif "image" in mime:
        chunks = extract_image(file_bytes)
    elif "text" in mime:
        chunks = extract_txt(file_bytes)
    # Note: PPTX, XLSX, HTML, MD now route to Docling; legacy fallback
    # only handles formats above. Unknown formats produce empty chunks.

    # Build element_uids first, then persist Artifacts with deterministic IDs
    element_uids: list[str] = []
    for idx, chunk in enumerate(chunks):
        content_hash = hashlib.sha256(
            (chunk.chunk_text or "").encode("utf-8", errors="replace")
        ).hexdigest()[:8]
        element_uids.append(f"legacy-{idx}-{chunk.modality}-{content_hash}")

    artifact_ids = _persist_extraction_results(db, document_id, chunks, element_uids=element_uids)

    # Create DocumentElement rows with artifact_id linked inline (upsert for reingest)
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    for idx, chunk in enumerate(chunks):
        element_uid = element_uids[idx]
        element_hash = hashlib.sha256(
            f"{document_id}:{element_uid}:{chunk.chunk_text or ''}".encode()
        ).hexdigest()

        element_values = {
            "document_id": uuid.UUID(document_id),
            "element_uid": element_uid,
            "element_type": chunk.modality,
            "element_order": idx,
            "page_number": chunk.page_number,
            "bounding_box": chunk.bounding_box,
            "content_text": _normalize_text(chunk.chunk_text),
            "element_metadata": chunk.metadata or {},
            "element_hash": element_hash,
            "artifact_id": artifact_ids[idx],
        }

        stmt = pg_insert(DocumentElement).values(**element_values)
        stmt = stmt.on_conflict_do_update(
            constraint="document_elements_document_id_element_uid_key",
            set_={
                "element_type": stmt.excluded.element_type,
                "element_order": stmt.excluded.element_order,
                "content_text": stmt.excluded.content_text,
                "element_hash": stmt.excluded.element_hash,
                "artifact_id": stmt.excluded.artifact_id,
            },
        )
        db.execute(stmt)


@celery_app.task(bind=True)
def _chord_error_handler(self, request, exc, traceback, document_id: str, run_id: str | None = None):
    """Errback for chord failures (e.g. hard time limit kills a chord member)."""
    logger.error("Chord failed for document %s: %s", document_id, exc)
    _update_document_status(
        document_id, STATUS_FAILED,
        stage="chord_error", error=str(exc),
    )
    if run_id:
        db = _get_db()
        try:
            from app.models.ingest import PipelineRun
            from sqlalchemy import update as sql_update
            import datetime
            db.execute(
                sql_update(PipelineRun)
                .where(PipelineRun.id == uuid.UUID(run_id))
                .values(status="FAILED", finished_at=datetime.datetime.now(datetime.timezone.utc))
            )
            db.commit()
        except Exception as e:
            logger.warning("_chord_error_handler: failed to update pipeline run %s: %s", run_id, e)
            db.rollback()
        finally:
            db.close()


def start_ingest_pipeline(
    document_id: str,
    *,
    ontology_bundle_key: str | None = None,
    use_case_key: str | None = None,
) -> "IngestDispatchResult":
    """Enqueue the ingest pipeline for a document. Returns IngestDispatchResult.

    Resolves the ontology bundle via the three-tier precedence:
        explicit (caller-supplied) → Source default → system default (settings)

    When the FOR UPDATE duplicate-dispatch guard hits an already-PROCESSING run,
    returns IngestDispatchResult(pipeline_run_id=str(active), celery_task_id='')
    — an empty celery_task_id signals "already dispatched by a prior caller."

    Spec §5.2 / Task 4.2.
    """
    from app.workers.dispatch_types import IngestDispatchResult
    from app.models.ingest import PipelineRun, Document
    from app.services.ontology_bundles import resolve_bundle_key, load_bundle_manifest
    from sqlalchemy import select

    _settings = get_settings()

    db = _get_db()
    try:
        # Atomic check: prevent duplicate dispatch if a run is already active
        active = db.execute(
            select(PipelineRun.id)
            .where(
                PipelineRun.document_id == uuid.UUID(document_id),
                PipelineRun.status == "PROCESSING",
            )
            .with_for_update()
            .limit(1)
        ).scalar_one_or_none()

        if active:
            logger.warning(
                "start_ingest_pipeline: skipping document %s — active run %s exists",
                document_id, active,
            )
            db.commit()  # release FOR UPDATE lock
            return IngestDispatchResult(
                pipeline_run_id=str(active),
                celery_task_id="",
            )

        # Resolve bundle key via three-tier precedence (spec §4.5)
        document = db.get(Document, uuid.UUID(document_id))
        source_key = (
            document.source.default_ontology_bundle_key
            if document and document.source
            else None
        )
        resolved_key = resolve_bundle_key(
            run_key=ontology_bundle_key,
            source_key=source_key,
            system_default=_settings.default_ontology_bundle_key,
        )

        manifest = load_bundle_manifest(resolved_key)

        run_id = _create_pipeline_run(
            db,
            document_id,
            mode="full",
            ontology_bundle_key=resolved_key,
            ontology_name=manifest.ontology_name,
            ontology_version=manifest.ontology_version,
            use_case_key=use_case_key,
            extraction_profile_version=manifest.extraction_profile_version,
        )
        db.commit()
    finally:
        db.close()

    logger.info(
        "start_ingest_pipeline: document_id=%s pipeline_run_id=%s bundle=%s",
        document_id, run_id, resolved_key,
    )

    # Fully sequential pipeline — no chords.  Celery 5.x chords with Redis
    # silently drop callbacks regardless of positioning, so we run every stage
    # in a simple chain.  The derivation stages (chunks, embed, graph) lose
    # parallelism but each takes only 10-60s vs 20+ min for picture descriptions,
    # so the throughput impact is negligible.
    pipeline = chain(
        prepare_document.si(document_id, run_id),
        detect_and_translate.si(document_id, run_id),
        derive_document_metadata.si(document_id, run_id),
        purge_document_derivations.si(document_id, run_id),
        derive_picture_descriptions.si(document_id, run_id),
        derive_text_chunks_and_embeddings.si(document_id, run_id),
        derive_image_embeddings.si(document_id, run_id),
        derive_ontology_graph.si(document_id, run_id),
        collect_derivations.si(document_id, run_id),
        derive_structure_links.si(document_id, run_id),
        derive_canonicalization.si(document_id, run_id),
        finalize_document.si(document_id, run_id),
    )
    result = pipeline.apply_async()
    return IngestDispatchResult(
        pipeline_run_id=run_id,
        celery_task_id=result.id,
    )


def _create_pipeline_run(
    db,
    document_id: str,
    *,
    mode: str = "full",
    ontology_bundle_key: str | None = None,
    ontology_name: str | None = None,
    ontology_version: str | None = None,
    use_case_key: str | None = None,
    extraction_profile_version: str | None = None,
) -> str:
    """Create a PipelineRun record and return its id as string.

    This is the single place where PipelineRun row construction happens.
    All callers (start_ingest_pipeline, reingest_graph_only) go through here.
    """
    from app.models.ingest import PipelineRun

    run = PipelineRun(
        document_id=uuid.UUID(document_id),
        pipeline_version="1.0",
        status="PROCESSING",
        mode=mode,
        ontology_bundle_key=ontology_bundle_key,
        ontology_name=ontology_name,
        ontology_version=ontology_version,
        use_case_key=use_case_key,
        extraction_profile_version=extraction_profile_version,
    )
    db.add(run)
    db.flush()
    return str(run.id)


def reingest_graph_only(doc_id, request) -> dict:
    """Dispatch a graph_only reingest. Spec §5.3 + Task 4.2.

    Resolves the bundle via graph_only precedence (explicit →
    inherited from latest run → source default → system default),
    creates a new PipelineRun with mode='graph_only' and the bundle
    snapshot, then dispatches the 3-stage graph-only chain
    (derive_ontology_graph, derive_structure_links, finalize_document).

    Returns a dict matching the legacy route's response shape:
        {
            "pipeline_run_id": str,
            "celery_task_id": str,
            "ontology_bundle_key": str,
        }

    Task 4.6 later changes derive_ontology_graph's signature; Task 4.2
    keeps the legacy (document_id, run_id) positional args so the
    graph_only chain still dispatches correctly today.
    """
    from app.models.ingest import Document, PipelineRun
    from app.services.ontology_bundles import (
        resolve_bundle_key_for_graph_only,
        load_bundle_manifest,
    )

    _settings = get_settings()
    doc_id_str = str(doc_id)

    db = _get_db()
    try:
        document = db.get(Document, uuid.UUID(doc_id_str))
        latest_run = (
            db.query(PipelineRun)
            .filter_by(document_id=uuid.UUID(doc_id_str))
            .order_by(PipelineRun.started_at.desc(), PipelineRun.id.desc())
            .first()
        )
        inherited_bundle = (
            latest_run.ontology_bundle_key
            if latest_run and latest_run.ontology_bundle_key
            else None
        )

        explicit_override = getattr(request, "ontology_bundle_key", None)
        source_key = (
            document.source.default_ontology_bundle_key
            if document and document.source
            else None
        )

        resolved_key = resolve_bundle_key_for_graph_only(
            run_key=explicit_override,
            inherited_from_run=inherited_bundle,
            source_key=source_key,
            system_default=_settings.default_ontology_bundle_key,
        )

        if inherited_bundle is None and latest_run is not None:
            logger.info(
                "reingest_graph_only: latest run for document %s is legacy "
                "(ontology_bundle_key NULL); bundle inferred from source/system default (%s)",
                doc_id_str, resolved_key,
            )

        manifest = load_bundle_manifest(resolved_key)

        explicit_use_case = getattr(request, "use_case_key", None)
        resolved_use_case = explicit_use_case or (
            latest_run.use_case_key if latest_run else None
        )

        run_id = _create_pipeline_run(
            db,
            doc_id_str,
            mode="graph_only",
            ontology_bundle_key=resolved_key,
            ontology_name=manifest.ontology_name,
            ontology_version=manifest.ontology_version,
            use_case_key=resolved_use_case,
            extraction_profile_version=manifest.extraction_profile_version,
        )
        db.commit()
    finally:
        db.close()

    # Same 3-stage chain the legacy reingest route used to build inline,
    # but now with the newly-created run_id.
    result = celery_chain(
        derive_ontology_graph.si(doc_id_str, run_id),
        derive_structure_links.si(doc_id_str, run_id),
        finalize_document.si(doc_id_str, run_id),
    ).apply_async()

    return {
        "pipeline_run_id": run_id,
        "celery_task_id": result.id,
        "ontology_bundle_key": resolved_key,
    }


def _update_stage_run(
    db, pipeline_run_id: str, stage_name: str, status: str,
    attempt: int = 1, metrics: dict | None = None, error: str | None = None,
) -> None:
    """Upsert a StageRun record."""
    from app.models.ingest import StageRun
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    import datetime

    values = {
        "pipeline_run_id": uuid.UUID(pipeline_run_id),
        "stage_name": stage_name,
        "attempt": attempt,
        "status": status,
    }
    if status == "RUNNING":
        values["started_at"] = datetime.datetime.now(datetime.timezone.utc)
    if status in ("COMPLETE", "FAILED"):
        values["finished_at"] = datetime.datetime.now(datetime.timezone.utc)
    if metrics:
        values["metrics"] = metrics
    if error:
        values["error_message"] = error

    from sqlalchemy import text as sa_text

    stmt = pg_insert(StageRun).values(**values).on_conflict_do_update(
        index_elements=["pipeline_run_id", "stage_name", "attempt"],
        index_where=sa_text("pass_name IS NULL"),
        set_={k: v for k, v in values.items() if k not in ("pipeline_run_id", "stage_name", "attempt")},
    )
    try:
        db.execute(stmt)
    except Exception as exc:
        # FK violation if pipeline_run was cleaned up by _cleanup_stale_runs
        db.rollback()
        logger.debug("_update_stage_run skipped (stale run_id %s): %s", pipeline_run_id, exc)


def _write_stage_run(
    *,
    pipeline_run_id,
    pass_def,
    attempt: int,
    execution_status: str,
    yield_status: str | None,
    skip_reason: str | None,
    counts: dict | None,
    error: str | None,
) -> None:
    """Upsert a per-pass StageRun row targeting the partial unique index
    uq_stage_runs_run_pass_attempt (WHERE pass_name IS NOT NULL) added in
    migration 0015.

    Maps execution_status to the legacy celery-level status column so the
    existing monitoring queries still work:
      COMPLETE or SKIPPED → 'COMPLETE'
      FAILED              → 'FAILED'

    counts shape:
      primary_entities_extracted, bridge_entities_extracted,
      relationships_extracted, relationships_rejected, schema_size_chars,
      structured_output_mode, salvaged
    """
    from app.models.ingest import StageRun
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    import datetime

    if execution_status in ("COMPLETE", "SKIPPED"):
        celery_status = "COMPLETE"
    elif execution_status == "FAILED":
        celery_status = "FAILED"
    else:
        celery_status = execution_status

    values: dict = {
        "pipeline_run_id": uuid.UUID(str(pipeline_run_id)),
        "stage_name": "derive_ontology_graph",
        "pass_name": pass_def.name,
        "attempt": attempt,
        "status": celery_status,
        "execution_status": execution_status,
        "yield_status": yield_status,
        "skip_reason": skip_reason,
        "finished_at": datetime.datetime.now(datetime.timezone.utc),
    }
    if counts:
        values.update({
            "primary_entities_extracted": counts.get("primary_entities_extracted"),
            "bridge_entities_extracted": counts.get("bridge_entities_extracted"),
            "relationships_extracted": counts.get("relationships_extracted"),
            "relationships_rejected": counts.get("relationships_rejected"),
            "schema_size_chars": counts.get("schema_size_chars"),
            "structured_output_mode": counts.get("structured_output_mode"),
            "salvaged": counts.get("salvaged"),
        })
    if error:
        values["error_message"] = error

    stmt = (
        pg_insert(StageRun)
        .values(**values)
        .on_conflict_do_update(
            index_elements=["pipeline_run_id", "stage_name", "pass_name", "attempt"],
            index_where=sa.text("pass_name IS NOT NULL"),
            set_={
                k: v
                for k, v in values.items()
                if k not in ("pipeline_run_id", "stage_name", "pass_name", "attempt")
            },
        )
    )
    db = _get_db()
    try:
        db.execute(stmt)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.debug(
            "_write_stage_run skipped (stale run_id %s, pass %s): %s",
            pipeline_run_id, pass_def.name, exc,
        )
    finally:
        db.close()


def _build_extract_pass_request(
    *, bundle_key: str, pass_def, doc_json: dict, upstream_refs: dict | None
) -> dict:
    """Assemble the POST body for /extract-pass."""
    body: dict = {
        "bundle_key": bundle_key,
        "pass_name": pass_def.name,
        "docling_document_json": doc_json,
    }
    if upstream_refs:
        body["upstream_entities"] = [
            {
                "ref_id": ref_id,
                "entity_type": getattr(ref, "entity_type", None),
                "identity_values": getattr(ref, "identity_values", {}) or {},
                "display_label": getattr(ref, "display_label", None),
            }
            for ref_id, ref in upstream_refs.items()
        ]
    return body


def _call_extract_pass(request_body: dict, *, timeout: float) -> dict:
    """Synchronous HTTP POST to the docling-graph /extract-pass endpoint.

    5xx and transport errors are retryable (PassRetryable).
    4xx and JSON decode errors are terminal (PassTerminal).
    """
    url = f"{settings.docling_graph_base_url}/extract-pass"
    try:
        response = httpx.post(url, json=request_body, timeout=timeout)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise PassRetryable(f"transport error: {exc}") from exc

    if response.status_code >= 500:
        raise PassRetryable(f"HTTP {response.status_code}: {response.text[:200]}")
    if response.status_code >= 400:
        raise PassTerminal(f"HTTP {response.status_code}: {response.text[:200]}")

    try:
        return response.json()
    except ValueError as exc:
        raise PassRetryable(f"partial/malformed response: {exc}") from exc


def _parse_pass_response(response_json: dict, pass_def, manifest) -> "object":
    """Validate the /extract-pass response dict into a PassResult with an
    instantiated pass template class.

    Pydantic validation errors are terminal — a malformed response won't
    heal on retry (spec §6.5).
    """
    import importlib
    from pydantic import ValidationError
    from app.services.extraction_merge import PassResult, ExtractionMetadata

    full_module_path = f"ontology_bundles.{manifest.bundle_key}.{pass_def.module}"
    try:
        template_module = importlib.import_module(full_module_path)
        template_cls = getattr(template_module, pass_def.template_class)
    except (ImportError, AttributeError) as exc:
        raise PassTerminal(
            f"cannot load template {full_module_path}.{pass_def.template_class}: {exc}"
        ) from exc

    try:
        template_instance = template_cls.model_validate(
            response_json.get("pass_output", {})
        )
    except ValidationError as exc:
        raise PassTerminal(f"template validation failed: {exc}") from exc

    metadata_dict = response_json.get("metadata", {}) or {}
    return PassResult(
        pass_name=pass_def.name,
        template_instance=template_instance,
        metadata=ExtractionMetadata(
            schema_size_chars=metadata_dict.get("schema_size_chars", 0),
            structured_output_mode=metadata_dict.get("structured_output_mode", "strict"),
        ),
        pre_merge_rejections=[],
    )


def _backoff(attempt: int) -> None:
    """Exponential backoff per spec §6.5: 30s × 2^(attempt-1), capped at 300s."""
    import time
    delay = min(30 * (2 ** (attempt - 1)), 300)
    time.sleep(delay)


def _count_pass_output(pass_result, pass_def, ontology) -> dict:
    """Count primary/bridge entities and relationships emitted by this pass."""
    primary = sum(
        len(list(pass_result.iter_entities_of_type(t)))
        for t in pass_def.primary_entity_types
    ) if hasattr(pass_result, "iter_entities_of_type") else 0
    bridge = sum(
        len(list(pass_result.iter_entities_of_type(t)))
        for t in pass_def.bridge_entity_types
    ) if hasattr(pass_result, "iter_entities_of_type") else 0
    extracted_rels = len(getattr(pass_result, "relationships", []) or [])
    rejected_rels = len(getattr(pass_result, "pre_merge_rejections", []) or [])
    metadata = getattr(pass_result, "metadata", None)
    return {
        "primary_entities_extracted": primary,
        "bridge_entities_extracted": bridge,
        "relationships_extracted": extracted_rels,
        "relationships_rejected": rejected_rels,
        "schema_size_chars": getattr(metadata, "schema_size_chars", None),
        "structured_output_mode": getattr(metadata, "structured_output_mode", None),
        "salvaged": False,
    }


def _any_downstream_pass_depends_on(manifest, pass_name: str) -> bool:
    """True if any later pass in the manifest lists pass_name in its depends_on."""
    seen_current = False
    for p in manifest.passes:
        if seen_current and pass_name in (p.depends_on or []):
            return True
        if p.name == pass_name:
            seen_current = True
    return False


def _extend_upstream_refs(
    upstream_refs: dict, pass_result, pass_def, ontology
) -> None:
    """Add ref_id → ref entries to upstream_refs for every primary entity
    produced by this pass so downstream passes can reference them."""
    from types import SimpleNamespace

    start = len(upstream_refs) + 1
    for entity_type in pass_def.primary_entity_types:
        if not hasattr(pass_result, "iter_entities_of_type"):
            break
        for i, instance in enumerate(
            pass_result.iter_entities_of_type(entity_type), start=start
        ):
            ref_id = f"E{i:03d}"
            upstream_refs[ref_id] = SimpleNamespace(
                pass_origin=pass_def.name,
                entity_type=entity_type,
                identity_values={
                    k: v
                    for k, v in (
                        instance.__dict__
                        if hasattr(instance, "__dict__")
                        else {}
                    ).items()
                    if not k.startswith("_")
                },
                display_label=None,
            )


def _get_pipeline_run_id(db, document_id: str) -> str | None:
    """Get the latest pipeline run id for a document."""
    from app.models.ingest import PipelineRun
    from sqlalchemy import select

    result = db.execute(
        select(PipelineRun.id)
        .where(
            PipelineRun.document_id == uuid.UUID(document_id),
            PipelineRun.pipeline_version == "1.0",
        )
        .order_by(PipelineRun.started_at.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    return str(row) if row else None


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30,
                 soft_time_limit=settings.prepare_soft_time_limit,
                 time_limit=settings.prepare_time_limit)
def prepare_document(self, document_id: str, run_id: str | None = None) -> str:
    """Validate + detect + Docling convert + persist document_elements.

    Creates canonical DocumentElement rows from Docling output, with backward-
    compatible Artifact dual-write.
    """
    import uuid as uuid_mod
    from app.models.ingest import Document, Artifact, DocumentElement
    from app.services.storage import download_bytes_sync, upload_bytes_sync
    from app.services.docling_client import convert_document_sync, check_health_sync
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    import magic

    # Apply env-var configurable retry and time-limit settings
    self.max_retries = settings.prepare_max_retries
    self.default_retry_delay = settings.prepare_retry_delay
    self.soft_time_limit = settings.prepare_soft_time_limit
    self.time_limit = settings.prepare_time_limit

    logger.info("prepare_document: document_id=%s run_id=%s", document_id, run_id)

    # Singleflight lock: prevent concurrent prepare_document for same document
    _singleflight_lock = _redis_client.lock(
        f"prepare:{document_id}",
        timeout=settings.prepare_singleflight_timeout,
        blocking=False,
    )
    if not _singleflight_lock.acquire(blocking=False):
        # Lock held — check if there's genuinely an active PROCESSING run.
        # If not, the lock is stale (orphaned by a crashed/timed-out task).
        _check_db = _get_db()
        try:
            from app.models.ingest import PipelineRun as _PR
            from sqlalchemy import select as _sa_select
            _active_run = _check_db.execute(
                _sa_select(_PR.id).where(
                    _PR.document_id == uuid.UUID(document_id),
                    _PR.status == "PROCESSING",
                ).limit(1)
            ).scalar_one_or_none()
        finally:
            _check_db.close()

        if _active_run:
            logger.warning(
                "prepare_document: singleflight lock held for %s (active run %s) — aborting",
                document_id, _active_run,
            )
            return document_id

        # Lock is stale — force-delete and re-acquire
        logger.warning(
            "prepare_document: stale singleflight lock for %s — no active run, force-releasing",
            document_id,
        )
        _redis_client.delete(f"prepare:{document_id}")
        if not _singleflight_lock.acquire(blocking=False):
            logger.error(
                "prepare_document: failed to acquire lock for %s even after force-release",
                document_id,
            )
            return document_id

    _update_document_status(document_id, STATUS_PROCESSING, stage="prepare_document")

    db = _get_db()
    try:
        # Use passed run_id or create one (backward compat)
        if not run_id:
            run_id = _create_pipeline_run(db, document_id)
            db.commit()

        # Supersession guard: bail if a newer pipeline run exists
        from app.models.ingest import PipelineRun
        from sqlalchemy import select as sa_select
        latest_active = db.execute(
            sa_select(PipelineRun.id)
            .where(
                PipelineRun.document_id == uuid.UUID(document_id),
                PipelineRun.status == "PROCESSING",
            )
            .order_by(PipelineRun.started_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if latest_active and str(latest_active) != run_id:
            logger.warning(
                "prepare_document: run %s superseded by %s for document %s — aborting",
                run_id, latest_active, document_id,
            )
            _update_stage_run(db, run_id, "prepare_document", "FAILED",
                              attempt=self.request.retries + 1, error="superseded")
            db.commit()
            try:
                _singleflight_lock.release()
            except Exception:
                pass
            return document_id

        _update_stage_run(db, run_id, "prepare_document", "RUNNING", attempt=self.request.retries + 1)
        db.commit()

        doc = db.get(Document, uuid.UUID(document_id))
        if not doc:
            raise ValueError(f"Document not found: {document_id}")

        # 2. Download + validate
        file_bytes = download_bytes_sync(doc.storage_bucket, doc.storage_key)
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        mime_type = magic.from_buffer(file_bytes, mime=True)

        # libmagic detects markdown/asciidoc/csv as text/plain — override by extension
        if mime_type == "text/plain" and doc.storage_key:
            ext = doc.storage_key.rsplit(".", 1)[-1].lower() if "." in doc.storage_key else ""
            _EXT_MIME_OVERRIDES = {
                "md": "text/markdown",
                "markdown": "text/markdown",
                "csv": "text/csv",
                "adoc": "text/asciidoc",
                "asciidoc": "text/asciidoc",
                "html": "text/html",
                "htm": "text/html",
            }
            if ext in _EXT_MIME_OVERRIDES:
                mime_type = _EXT_MIME_OVERRIDES[ext]

        from sqlalchemy import update as sql_update
        db.execute(
            sql_update(Document)
            .where(Document.id == uuid.UUID(document_id))
            .values(
                file_size_bytes=len(file_bytes),
                file_hash=file_hash,
                mime_type=mime_type,
            )
        )
        db.commit()

        # 3. Route by format — Docling handles PDF, images, office docs, and markup
        _DOCLING_MIMES = {
            # PDF
            "application/pdf",
            # Images
            "image/png", "image/jpeg", "image/tiff", "image/bmp", "image/gif", "image/webp",
            # Office documents
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # PPTX
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # XLSX
            "application/msword",  # legacy DOC
            "application/vnd.ms-powerpoint",  # legacy PPT
            "application/vnd.ms-excel",  # legacy XLS
            # Markup / text
            "text/html",
            "text/markdown",
            "text/csv",
            "text/asciidoc",
        }
        if mime_type not in _DOCLING_MIMES:
            logger.info("prepare_document: %s not supported by Docling (mime=%s), using legacy extraction", document_id, mime_type)
            _legacy_extract(db, document_id, doc, file_bytes)
            db.commit()

            # Persist extracted text as markdown so derive_document_metadata can run
            try:
                from app.models.ingest import DocumentElement
                from sqlalchemy import select as sql_select
                elems = db.execute(
                    sql_select(DocumentElement.content_text)
                    .where(DocumentElement.document_id == uuid.UUID(document_id))
                    .order_by(DocumentElement.element_order)
                ).scalars().all()
                fallback_md = "\n\n".join(t for t in elems if t and t.strip())
                if fallback_md:
                    fallback_md = _normalize_text(fallback_md)
                    from app.services.storage import upload_bytes_sync
                    _fb_base = f"artifacts/{document_id}"
                    upload_bytes_sync(
                        fallback_md.encode("utf-8"),
                        settings.minio_bucket_derived,
                        f"{_fb_base}/docling_document.md",
                        content_type="text/markdown; charset=utf-8",
                    )
                    logger.info("prepare_document: persisted legacy markdown for %s (%d chars)", document_id, len(fallback_md))
            except Exception as _fb_err:
                logger.warning("prepare_document: failed to persist legacy markdown for %s: %s", document_id, _fb_err)

            _update_stage_run(db, run_id, "prepare_document", "COMPLETE", attempt=self.request.retries + 1, metrics={"fallback": True, "reason": "unsupported_format"})
            db.commit()
            return document_id

        # 4. Docling conversion — acquire semaphore permit with busy-wait
        #    Uses in-task backoff loop instead of self.retry() so capacity waits
        #    do NOT consume the task's retry budget (retries reserved for real errors).
        import time as _time

        docling_lock = None
        _wait_start = _time.monotonic()
        _max_wait = settings.docling_lock_timeout  # 600s default
        _wait_attempt = 0
        while docling_lock is None:
            for _permit_i in range(settings.docling_concurrency):
                _candidate = _redis_client.lock(
                    f"docling:permit:{_permit_i}", timeout=settings.docling_lock_timeout, blocking=False,
                )
                if _candidate.acquire(blocking=False):
                    docling_lock = _candidate
                    break
            if docling_lock is not None:
                break
            _elapsed = _time.monotonic() - _wait_start
            if _elapsed >= _max_wait:
                raise RuntimeError(
                    f"Docling at capacity for {_elapsed:.0f}s — all {settings.docling_concurrency} "
                    f"permits held. Document {document_id} cannot proceed."
                )
            _wait_attempt += 1
            _sleep_time = min(30, 5 * _wait_attempt)  # 5s, 10s, 15s, ... capped at 30s
            logger.info(
                "prepare_document: Docling at capacity (%d/%d), waiting %ds for %s (%.0fs elapsed)",
                settings.docling_concurrency, settings.docling_concurrency,
                _sleep_time, document_id, _elapsed,
            )
            _time.sleep(_sleep_time)

        try:
            # Advisory health check — log but don't fail (health endpoint
            # may time out during long CPU conversions on other docs, but
            # convert itself will work once the semaphore permits it).
            docling_healthy = check_health_sync()
            if not docling_healthy:
                logger.warning(
                    "prepare_document: Docling health check failed (advisory) for %s — proceeding with convert",
                    document_id,
                )

            # In-task retry loop for 503 (Docling busy with ghost request from
            # a previous timed-out attempt).  Sleeps and retries WITHOUT consuming
            # the Celery retry budget.  The Redis lock stays held during this loop
            # so no other tasks attempt to send to Docling concurrently.
            _max_503_retries = settings.docling_503_max_retries
            for _503_attempt in range(_max_503_retries):
                try:
                    result = convert_document_sync(file_bytes, doc.filename or "document")
                    break  # success
                except httpx.HTTPStatusError as _docling_exc:
                    if _docling_exc.response.status_code != 503 or _503_attempt >= _max_503_retries - 1:
                        raise
                    _wait = min(120, 30 * (_503_attempt + 1))  # 30s, 60s, 90s, 120s cap
                    logger.info(
                        "prepare_document: Docling 503 for %s — in-task wait %ds (%d/%d)",
                        document_id, _wait, _503_attempt + 1, _max_503_retries,
                    )
                    _time.sleep(_wait)
            _docling_convert_ok = True
        except Exception:
            # If Celery retries remain, keep the Docling lock held to prevent
            # another task from grabbing it in the gap. Lock TTL auto-expires.
            _docling_convert_ok = False
            if self.request.retries < self.max_retries:
                logger.info(
                    "prepare_document: keeping Docling lock for %s (retries remain, TTL will expire)",
                    document_id,
                )
            raise
        finally:
            if _docling_convert_ok or self.request.retries >= self.max_retries:
                try:
                    docling_lock.release()
                except redis_lib.exceptions.LockNotOwnedError:
                    logger.warning("prepare_document: Docling lock expired before release for %s", document_id)
        logger.info(
            "prepare_document: docling returned %d elements, %d pages, %.0fms",
            len(result.elements), result.num_pages, result.processing_time_ms,
        )

        # 4. Deduplicate extracted elements (conservative: same modality+page+section+text+bbox)
        result.elements, _dups_dropped = _dedupe_extracted_elements(result.elements)
        if _dups_dropped:
            logger.info(
                "prepare_document: %d elements after dedup (%d duplicates dropped) for %s",
                len(result.elements), _dups_dropped, document_id,
            )

        # 4b. Standalone image fallback — synthesize element when Docling
        #     returns 0 elements for an image file (JPEG, PNG, TIFF, etc.)
        if len(result.elements) == 0 and mime_type.startswith("image/"):
            synthesized = _synthesize_standalone_image(file_bytes, mime_type)
            if synthesized:
                result.elements = synthesized
                result.num_pages = max(result.num_pages, 1)
                logger.info(
                    "prepare_document: synthesized standalone image element for %s (mime=%s)",
                    document_id, mime_type,
                )

        # 5. Build element_uids, then persist Artifacts with deterministic IDs
        element_uids: list[str] = []
        elements_created = 0
        for chunk in result.elements:
            element_uid = (chunk.metadata or {}).get("element_uid")
            if not element_uid:
                content_hash = hashlib.sha256(
                    (chunk.chunk_text or "").encode("utf-8", errors="replace")
                ).hexdigest()[:8]
                element_uid = f"{chunk.page_number or 0}-{elements_created}-{chunk.modality}-{content_hash}"
            element_uids.append(element_uid)
            elements_created += 1

        # 5. Dual-write Artifact rows with deterministic IDs
        artifact_ids = _persist_extraction_results(db, document_id, result.elements, element_uids=element_uids)
        db.flush()  # Ensure artifact rows visible for FK checks in Core SQL inserts below

        # Build a lookup of image storage keys from Artifact uploads to avoid re-uploading
        _image_storage: dict[int, tuple[str, str]] = {}
        for idx, chunk in enumerate(result.elements):
            if chunk.raw_image_bytes:
                # The Artifact was already uploaded in _persist_extraction_results;
                # query its storage_key to reuse for the DocumentElement row.
                art = db.get(Artifact, artifact_ids[idx])
                if art and art.storage_key:
                    _image_storage[idx] = (art.storage_bucket, art.storage_key)

        # 6. Persist canonical DocumentElement rows with artifact_id linked inline
        elements_created = 0
        for idx, chunk in enumerate(result.elements):
            element_uid = element_uids[idx]

            # Reuse image storage from Artifact upload (no duplicate MinIO I/O)
            if idx in _image_storage:
                storage_bucket, storage_key = _image_storage[idx]
            else:
                storage_bucket = None
                storage_key = None

            element_hash = hashlib.sha256(
                f"{document_id}:{element_uid}:{chunk.chunk_text or ''}".encode()
            ).hexdigest()

            element_values = {
                "document_id": uuid.UUID(document_id),
                "element_uid": element_uid,
                "element_type": chunk.modality,
                "element_order": (chunk.metadata or {}).get("element_order", elements_created),
                "page_number": chunk.page_number,
                "bounding_box": chunk.bounding_box,
                "section_path": (chunk.metadata or {}).get("section_path"),
                "heading_level": (chunk.metadata or {}).get("heading_level"),
                "content_text": _normalize_text(chunk.chunk_text),
                "storage_bucket": storage_bucket,
                "storage_key": storage_key,
                "element_metadata": chunk.metadata or {},
                "element_hash": element_hash,
                "artifact_id": artifact_ids[idx],
            }

            stmt = pg_insert(DocumentElement).values(**element_values)
            stmt = stmt.on_conflict_do_update(
                constraint="document_elements_document_id_element_uid_key",
                set_={
                    "element_type": stmt.excluded.element_type,
                    "element_order": stmt.excluded.element_order,
                    "content_text": stmt.excluded.content_text,
                    "storage_bucket": stmt.excluded.storage_bucket,
                    "storage_key": stmt.excluded.storage_key,
                    "metadata": stmt.excluded.metadata,
                    "element_hash": stmt.excluded.element_hash,
                    "artifact_id": stmt.excluded.artifact_id,
                },
            )
            db.execute(stmt)
            elements_created += 1

        db.commit()

        # Remove stale DocumentElements/Artifacts not in current extraction
        from sqlalchemy import delete as sql_delete
        stale_elems = db.execute(
            sql_delete(DocumentElement).where(
                DocumentElement.document_id == uuid.UUID(document_id),
                ~DocumentElement.element_uid.in_(element_uids),
            )
        )
        stale_elem_count = stale_elems.rowcount

        stale_arts = db.execute(
            sql_delete(Artifact).where(
                Artifact.document_id == uuid.UUID(document_id),
                ~Artifact.id.in_(artifact_ids),
            )
        )
        stale_art_count = stale_arts.rowcount
        if stale_elem_count or stale_art_count:
            db.commit()
            logger.info(
                "prepare_document: cleaned %d stale elements, %d stale artifacts for %s",
                stale_elem_count, stale_art_count, document_id,
            )

        # Persist DoclingDocument markdown and JSON to MinIO for the viewer
        try:
            from app.services.storage import upload_bytes_sync
            _docling_base = f"artifacts/{document_id}"
            if result.markdown:
                upload_bytes_sync(
                    _normalize_text(result.markdown).encode("utf-8"),
                    settings.minio_bucket_derived,
                    f"{_docling_base}/docling_document.md",
                    content_type="text/markdown; charset=utf-8",
                )
            if getattr(result, "document_json", None):
                import json as _json
                doc_dict = result.document_json

                # Build identity map: self_ref -> element_uid
                identity_map: dict[str, str] = {}
                for idx, chunk in enumerate(result.elements):
                    self_ref = (chunk.metadata or {}).get("self_ref")
                    if self_ref:
                        identity_map[self_ref] = element_uids[idx]

                doc_dict["_enrichments"] = {
                    "version": 0,
                    "identity_map": identity_map,
                    "translations": {},
                    "context": {},
                }

                _raw_json = _json.dumps(doc_dict, ensure_ascii=False, default=str)
                upload_bytes_sync(
                    _normalize_text(_raw_json).encode("utf-8"),
                    settings.minio_bucket_derived,
                    f"{_docling_base}/docling_document.json",
                    content_type="application/json; charset=utf-8",
                )
                logger.info("prepare_document: persisted DoclingDocument md+json for %s", document_id)
        except Exception as _doc_err:
            logger.warning("prepare_document: failed to persist DoclingDocument for %s: %s", document_id, _doc_err)

        _update_stage_run(
            db, run_id, "prepare_document", "COMPLETE",
            attempt=self.request.retries + 1,
            metrics={
                "elements": elements_created,
                "num_pages": result.num_pages,
                "processing_time_ms": result.processing_time_ms,
                "stale_elements_removed": stale_elem_count,
                "stale_artifacts_removed": stale_art_count,
            },
        )
        db.commit()

        logger.info(
            "prepare_document: document_id=%s elements=%d pages=%d",
            document_id, elements_created, result.num_pages,
        )
        return document_id

    except CeleryRetry:
        raise
    except SoftTimeLimitExceeded as exc:
        # Task killed mid-wait or mid-conversion — Docling may still be working on
        # this or another document.  Re-queue without consuming the retry budget.
        logger.warning(
            "prepare_document: soft time limit for %s — re-queuing in 180s (attempt %d, not counted)",
            document_id, self.request.retries + 1,
        )
        db.rollback()
        raise self.retry(exc=exc, countdown=180)
    except Exception as exc:
        logger.error("prepare_document failed for %s: %s", document_id, exc)
        db.rollback()

        # Docling 5xx: fall back to legacy extraction if enabled
        # doc/file_bytes are assigned before Docling call, so always in scope for 5xx
        if (
            settings.docling_fallback_enabled
            and isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code >= 500
        ):
            logger.warning(
                "prepare_document: Docling %d for %s — falling back to legacy extraction",
                exc.response.status_code, document_id,
            )
            try:
                _legacy_extract(db, document_id, doc, file_bytes)
                db.commit()
                _update_stage_run(
                    db, run_id, "prepare_document", "COMPLETE",
                    attempt=self.request.retries + 1,
                    metrics={"fallback": True, "reason": f"docling_{exc.response.status_code}"},
                )
                db.commit()
                return document_id
            except Exception as fallback_exc:
                logger.error("prepare_document: legacy fallback also failed for %s: %s", document_id, fallback_exc)
                db.rollback()
                # Fall through to normal retry/fail logic

        # Artifact PK collision — deterministic IDs collided with existing rows.
        # This shouldn't happen after the upsert fix, but if it does, retrying
        # will produce the same collision.  Fail immediately.
        from sqlalchemy.exc import IntegrityError as _IntegrityError
        if isinstance(exc, _IntegrityError) and "artifacts_pkey" in str(exc):
            logger.error(
                "prepare_document: artifact PK collision for %s — failing without retry: %s",
                document_id, exc,
            )
            _update_document_status(
                document_id, STATUS_FAILED, stage="prepare_document", error=str(exc)
            )
            run_id_for_err = run_id or _get_pipeline_run_id(db, document_id)
            if run_id_for_err:
                _update_stage_run(db, run_id_for_err, "prepare_document", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            raise

        # Deterministic Docling errors (VlmPipeline failed, unsupported format)
        # won't resolve on retry — fail immediately.
        _deterministic_markers = ("VlmPipeline failed", "unsupported format", "invalid PDF")
        if isinstance(exc, RuntimeError) and any(m in str(exc) for m in _deterministic_markers):
            logger.warning(
                "prepare_document: deterministic Docling failure for %s — skipping retries: %s",
                document_id, exc,
            )
            _update_document_status(
                document_id, STATUS_FAILED, stage="prepare_document", error=str(exc)
            )
            run_id_for_err = run_id or _get_pipeline_run_id(db, document_id)
            if run_id_for_err:
                _update_stage_run(db, run_id_for_err, "prepare_document", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            raise

        # 503 is now handled by the in-task retry loop above; if we still
        # reach here with a 503 it means all in-task retries were exhausted —
        # treat it as a normal error and let the Celery retry budget handle it.
        countdown = settings.prepare_retry_delay

        if self.request.retries >= self.max_retries:
            _update_document_status(
                document_id, STATUS_FAILED, stage="prepare_document", error=str(exc)
            )
            run_id_for_err = run_id or _get_pipeline_run_id(db, document_id)
            if run_id_for_err:
                _update_stage_run(db, run_id_for_err, "prepare_document", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            raise
        logger.info("prepare_document: retrying %s (attempt %d/%d)", document_id, self.request.retries + 1, self.max_retries)
        raise self.retry(exc=exc, countdown=countdown)
    finally:
        db.close()
        # Release singleflight lock (safe if already released or expired)
        try:
            _singleflight_lock.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Stage: derive_document_metadata — LLM metadata extraction
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="app.workers.pipeline.derive_document_metadata",
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=settings.doc_analysis_timeout + 60,
    time_limit=settings.doc_analysis_timeout + 120,
    queue="ingest",
)
def derive_document_metadata(self, document_id: str, run_id: str | None = None) -> dict:
    """Extract document metadata (summary, date, classification, source) via LLM."""
    import json as json_mod

    logger.info("derive_document_metadata: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_document_metadata")

    if not settings.doc_analysis_enabled:
        logger.info("derive_document_metadata: disabled, skipping for %s", document_id)
        if run_id:
            _db = _get_db()
            try:
                _update_stage_run(_db, run_id, "derive_document_metadata", "COMPLETE",
                                  attempt=self.request.retries + 1,
                                  metrics={"skipped": True, "reason": "disabled"})
                _db.commit()
            finally:
                _db.close()
        return {"stage": "derive_document_metadata", "status": "skipped"}

    db = _get_db()
    try:
        if run_id:
            _update_stage_run(db, run_id, "derive_document_metadata", "RUNNING", attempt=self.request.retries + 1)
            db.commit()

        # Load markdown from MinIO
        from app.services.storage import download_bytes_sync
        base_key = f"artifacts/{document_id}"
        bucket = settings.minio_bucket_derived

        try:
            original_md = download_bytes_sync(bucket, f"{base_key}/docling_document.md").decode("utf-8")
        except Exception:
            logger.info("derive_document_metadata: no markdown available for %s, skipping", document_id)
            if run_id:
                _update_stage_run(db, run_id, "derive_document_metadata", "COMPLETE",
                                  attempt=self.request.retries + 1,
                                  metrics={"skipped": True, "reason": "no_markdown"})
                db.commit()
            return {"stage": "derive_document_metadata", "status": "skipped", "reason": "no_markdown"}

        try:
            translated_md = download_bytes_sync(bucket, f"{base_key}/docling_document_translated.md").decode("utf-8")
        except Exception:
            translated_md = None

        # Use translated for summary/date/source (English); original for classification (detect original markings)
        markdown = translated_md or original_md
        classification_markdown = original_md

        # Extract metadata via LLM
        from app.services.document_analysis import extract_document_metadata
        metadata = extract_document_metadata(markdown, classification_text=classification_markdown)

        # Merge into documents.document_metadata (preserves translation fields set earlier)
        from sqlalchemy import text
        db.execute(
            text("""
                UPDATE ingest.documents
                SET document_metadata = COALESCE(document_metadata, '{}'::jsonb) || cast(:meta AS jsonb)
                WHERE id = cast(:doc_id AS uuid)
            """),
            {"meta": json_mod.dumps(metadata), "doc_id": document_id},
        )
        db.commit()

        if run_id:
            _update_stage_run(
                db, run_id, "derive_document_metadata", "COMPLETE",
                attempt=self.request.retries + 1,
                metrics={"summary_length": len(metadata.get("document_summary", ""))},
            )
            db.commit()

        logger.info(
            "derive_document_metadata: document_id=%s classification=%s",
            document_id, metadata.get("classification"),
        )
        return {"stage": "derive_document_metadata", "status": "ok"}

    except Exception as exc:
        logger.error("derive_document_metadata failed for %s: %s", document_id, exc)
        if run_id:
            try:
                _update_stage_run(db, run_id, "derive_document_metadata", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            except Exception:
                pass
        raise self.retry(exc=exc)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Stage: detect_and_translate — per-element language detection and translation
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="app.workers.pipeline.detect_and_translate",
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=settings.translation_soft_time_limit,
    time_limit=settings.translation_time_limit,
    queue="ingest",
)
def detect_and_translate(self, document_id: str, run_id: str | None = None) -> dict:
    """Detect non-English elements and translate them via Ollama."""
    import json as json_mod

    from app.models.ingest import Document, DocumentElement
    from app.services.translation import detect_element_languages, translate_elements
    from app.services.storage import download_bytes_sync, upload_bytes_sync
    from sqlalchemy import select

    logger.info("detect_and_translate: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="detect_and_translate")

    if not settings.translation_enabled:
        logger.info("detect_and_translate: disabled, skipping for %s", document_id)
        if run_id:
            _db = _get_db()
            try:
                _update_stage_run(_db, run_id, "detect_and_translate", "COMPLETE",
                                  attempt=self.request.retries + 1,
                                  metrics={"skipped": True, "reason": "disabled"})
                _db.commit()
            finally:
                _db.close()
        return {"stage": "detect_and_translate", "status": "skipped", "reason": "disabled"}

    db = _get_db()
    try:
        if run_id:
            _update_stage_run(db, run_id, "detect_and_translate", "RUNNING", attempt=self.request.retries + 1)
            db.commit()

        # Query elements eligible for language detection / translation
        elements = db.execute(
            select(DocumentElement)
            .where(
                DocumentElement.document_id == uuid.UUID(document_id),
                DocumentElement.content_text.isnot(None),
                DocumentElement.element_type.in_(["text", "heading", "table", "equation"]),
            )
            .order_by(DocumentElement.element_order)
        ).scalars().all()

        if not elements:
            logger.info("detect_and_translate: no eligible elements for %s", document_id)
            if run_id:
                _update_stage_run(db, run_id, "detect_and_translate", "COMPLETE",
                                  attempt=self.request.retries + 1,
                                  metrics={"skipped": True, "reason": "no_elements"})
                db.commit()
            return {"stage": "detect_and_translate", "status": "skipped", "reason": "no_elements"}

        # Build list of dicts for translation service
        elem_dicts = [
            {"content_text": e.content_text, "element_type": e.element_type}
            for e in elements
        ]

        # Detect languages
        detection = detect_element_languages(elem_dicts)
        detected_language = detection["document_language"]
        non_english_indices = detection["non_english_indices"]

        total_elements = len(elem_dicts)
        non_english_count = len(non_english_indices)

        # Load existing document_metadata (may be None if derive_document_metadata hasn't run)
        from sqlalchemy import text as sa_text

        def _merge_doc_metadata(updates: dict):
            """Atomically merge keys into document_metadata JSONB (no clobber)."""
            db.execute(
                sa_text("""
                    UPDATE ingest.documents
                    SET document_metadata = COALESCE(document_metadata, '{}'::jsonb) || cast(:updates AS jsonb)
                    WHERE id = cast(:doc_id AS uuid)
                """),
                {"updates": json_mod.dumps(updates), "doc_id": document_id},
            )

        if not non_english_indices:
            # Nothing to translate — just persist language detection result
            _merge_doc_metadata({"detected_language": detected_language, "has_translation": False})
            db.commit()
            if run_id:
                _update_stage_run(db, run_id, "detect_and_translate", "COMPLETE",
                                  attempt=self.request.retries + 1,
                                  metrics={
                                      "detected_language": detected_language,
                                      "total_elements": total_elements,
                                      "non_english_elements": 0,
                                      "elements_translated": 0,
                                  })
                db.commit()
            logger.info("detect_and_translate: document_id=%s all-English (%s), nothing to translate",
                        document_id, detected_language)
            return {"stage": "detect_and_translate", "status": "ok", "translated": 0}

        # Translate non-English elements
        translated_texts = translate_elements(elem_dicts, non_english_indices)

        # Update DocumentElement rows with translated content
        elements_translated = 0
        for idx in non_english_indices:
            new_text = translated_texts[idx]
            if new_text and new_text != elements[idx].content_text:
                elements[idx].translated_text = new_text
                elements_translated += 1

        if elements_translated:
            db.commit()

        # ------------------------------------------------------------------
        # Enrich DoclingDocument JSON with translations
        # ------------------------------------------------------------------
        _base_key = f"artifacts/{document_id}"
        try:
            _raw = download_bytes_sync(settings.minio_bucket_derived, f"{_base_key}/docling_document.json")
            _doc_dict = json_mod.loads(_raw)
        except Exception:
            _doc_dict = None

        if _doc_dict is not None:
            enrichments = _doc_dict.setdefault("_enrichments", {"version": 0, "identity_map": {}, "translations": {}, "context": {}})
            # Build reverse map: element_uid → self_ref
            identity_map = enrichments.get("identity_map", {})
            reverse_map = {v: k for k, v in identity_map.items()}

            for elem in elements:
                euid = str(elem.element_uid) if hasattr(elem, "element_uid") else ""
                self_ref = reverse_map.get(euid)
                if self_ref and elem.translated_text:
                    enrichments["translations"][self_ref] = {
                        "original_text": elem.content_text or "",
                        "translated_text": elem.translated_text,
                        "language": detected_language or "unknown",
                    }

            enrichments["version"] = enrichments.get("version", 0) + 1

            # Re-persist enriched JSON
            upload_bytes_sync(
                json_mod.dumps(_doc_dict, ensure_ascii=False, default=str).encode("utf-8"),
                settings.minio_bucket_derived,
                f"{_base_key}/docling_document.json",
                content_type="application/json; charset=utf-8",
            )

            # Regenerate translated markdown from enriched DoclingDocument
            from app.services.docling_enrichment import _regenerate_translated_markdown
            try:
                translated_md = _regenerate_translated_markdown(_doc_dict)
                upload_bytes_sync(
                    translated_md.encode("utf-8"),
                    settings.minio_bucket_derived,
                    f"{_base_key}/docling_document_translated.md",
                    content_type="text/markdown; charset=utf-8",
                )
            except Exception as _md_err:
                logger.warning("Failed to regenerate translated markdown: %s", _md_err)
        else:
            # Fallback: reassemble element texts into markdown (no DoclingDocument JSON available)
            md_parts = []
            for elem in elements:
                text = elem.translated_text or elem.content_text or ""
                if elem.element_type == "heading":
                    md_parts.append(f"## {text}")
                else:
                    md_parts.append(text)
            translated_md = "\n\n".join(md_parts)
            upload_bytes_sync(
                translated_md.encode("utf-8"),
                settings.minio_bucket_derived,
                f"{_base_key}/docling_document_translated.md",
                content_type="text/markdown; charset=utf-8",
            )

        # Persist translation flags to document_metadata (atomic merge, no clobber)
        _merge_doc_metadata({"detected_language": detected_language, "has_translation": True})
        db.commit()

        if run_id:
            _update_stage_run(db, run_id, "detect_and_translate", "COMPLETE",
                              attempt=self.request.retries + 1,
                              metrics={
                                  "detected_language": detected_language,
                                  "total_elements": total_elements,
                                  "non_english_elements": non_english_count,
                                  "elements_translated": elements_translated,
                              })
            db.commit()

        logger.info(
            "detect_and_translate: document_id=%s language=%s non_english=%d translated=%d",
            document_id, detected_language, non_english_count, elements_translated,
        )
        return {
            "stage": "detect_and_translate",
            "status": "ok",
            "detected_language": detected_language,
            "translated": elements_translated,
        }

    except CeleryRetry:
        raise
    except SoftTimeLimitExceeded:
        logger.warning(
            "detect_and_translate: soft time limit for %s — marking FAILED",
            document_id,
        )
        if run_id:
            try:
                _update_stage_run(db, run_id, "detect_and_translate", "FAILED",
                                  attempt=self.request.retries + 1, error="soft time limit exceeded")
                db.commit()
            except Exception:
                pass
        _update_document_status(document_id, STATUS_PARTIAL_COMPLETE, stage="detect_and_translate")
        return {"stage": "detect_and_translate", "status": "timeout"}
    except Exception as exc:
        logger.error("detect_and_translate failed for %s: %s", document_id, exc)
        if run_id:
            try:
                _update_stage_run(db, run_id, "detect_and_translate", "FAILED",
                                  attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            except Exception:
                pass
        raise self.retry(exc=exc)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Stage: derive_picture_descriptions — LLM image enrichment
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="app.workers.pipeline.derive_picture_descriptions",
    max_retries=settings.picture_desc_max_retries,
    default_retry_delay=settings.picture_desc_retry_delay,
    soft_time_limit=settings.picture_desc_soft_time_limit,
    time_limit=settings.picture_desc_time_limit,
    queue="ingest",
)
def derive_picture_descriptions(self, document_id: str, run_id: str | None = None) -> dict:
    """Enrich picture items with LLM-generated descriptions using document summary context."""
    import json as json_mod

    logger.info("derive_picture_descriptions: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_picture_descriptions")

    db = _get_db()
    try:
        if run_id:
            _update_stage_run(db, run_id, "derive_picture_descriptions", "RUNNING", attempt=self.request.retries + 1)
            db.commit()

        # Load document metadata for summary
        from sqlalchemy import text as sa_text
        row = db.execute(
            sa_text("SELECT document_metadata FROM ingest.documents WHERE id = cast(:doc_id AS uuid)"),
            {"doc_id": document_id},
        ).first()
        document_summary = ""
        if row and row[0]:
            meta = row[0] if isinstance(row[0], dict) else json_mod.loads(row[0])
            document_summary = meta.get("document_summary", "")

        # Load Docling JSON from MinIO
        from app.services.storage import download_bytes_sync, upload_bytes_sync
        base_key = f"artifacts/{document_id}"
        bucket = settings.minio_bucket_derived
        try:
            json_bytes = download_bytes_sync(bucket, f"{base_key}/docling_document.json")
            docling_json = json_mod.loads(json_bytes)
        except Exception:
            logger.info("derive_picture_descriptions: no Docling JSON for %s, skipping", document_id)
            if run_id:
                _update_stage_run(db, run_id, "derive_picture_descriptions", "COMPLETE",
                                  attempt=self.request.retries + 1,
                                  metrics={"pictures_updated": 0, "skipped": True})
                db.commit()
            return {"stage": "derive_picture_descriptions", "status": "skipped"}

        # Describe images from the STORED artifacts (MinIO), not from the
        # Docling JSON pictures array.  The Docling service returns images in
        # two places (elements[].image_base64 and document_json.pictures[].image)
        # and they may not correspond 1:1.  Describing the stored artifacts
        # guarantees the description matches the image that will be served.
        from app.models.ingest import DocumentElement
        from app.services.document_analysis import _describe_single_image
        from sqlalchemy import select as sa_select
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import base64

        pic_elements = db.execute(
            sa_select(DocumentElement).where(
                DocumentElement.document_id == uuid.UUID(document_id),
                DocumentElement.element_type == "image",
                DocumentElement.storage_key.isnot(None),
            ).order_by(DocumentElement.element_order)
        ).scalars().all()

        if not pic_elements:
            logger.info("derive_picture_descriptions: no image elements with storage for %s", document_id)
            if run_id:
                _update_stage_run(db, run_id, "derive_picture_descriptions", "COMPLETE",
                                  attempt=self.request.retries + 1,
                                  metrics={"pictures_updated": 0})
                db.commit()
            return {"stage": "derive_picture_descriptions", "status": "ok", "pictures_updated": 0}

        prompt_template = settings.picture_description_prompt.replace("\\n", "\n")
        prompt = prompt_template.replace("{document_summary}", document_summary)
        model = settings.picture_description_model
        timeout = settings.picture_description_timeout

        # Load each stored image and describe it
        describable: list[tuple[int, str]] = []  # (element index, base64)
        for idx, elem in enumerate(pic_elements):
            try:
                img_bytes = download_bytes_sync(elem.storage_bucket, elem.storage_key)
                b64 = base64.b64encode(img_bytes).decode("ascii")
                describable.append((idx, b64))
            except Exception as e:
                logger.warning("Could not load image for element %s: %s", elem.element_uid, e)

        descriptions: dict[int, str] = {}
        pictures_updated = 0

        if describable:
            max_workers = min(3, len(describable))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_describe_single_image, b64, prompt, model, timeout, settings): idx
                    for idx, b64 in describable
                }
                for future in as_completed(futures):
                    elem_idx = futures[future]
                    try:
                        desc = future.result()
                        if desc:
                            descriptions[elem_idx] = desc
                    except Exception as e:
                        logger.warning("Picture description failed for element %d: %s", elem_idx, e)

            # Write descriptions to DocumentElement rows
            for idx, desc in descriptions.items():
                elem = pic_elements[idx]
                if desc != elem.content_text:
                    elem.content_text = desc
                    pictures_updated += 1

            logger.info("derive_picture_descriptions: described=%d, updated=%d elements",
                        len(descriptions), pictures_updated)
        if pictures_updated:
            db.commit()
            logger.info("derive_picture_descriptions: updated %d DocumentElement rows", pictures_updated)

        # Also update the markdown in MinIO to include picture descriptions
        # so derive_document_metadata and graph extraction see enriched content
        if descriptions:
            try:
                md_bytes = download_bytes_sync(bucket, f"{base_key}/docling_document.md")
                markdown = md_bytes.decode("utf-8")
                appendix_parts = [f"[Image Description]: {desc}" for desc in descriptions.values()]
                enriched_md = markdown + "\n\n## Image Descriptions\n\n" + "\n\n".join(appendix_parts)
                upload_bytes_sync(
                    enriched_md.encode("utf-8"),
                    bucket,
                    f"{base_key}/docling_document.md",
                    content_type="text/markdown; charset=utf-8",
                )
            except Exception as md_err:
                logger.debug("derive_picture_descriptions: could not update markdown: %s", md_err)

        # ------------------------------------------------------------------
        # Enrich DoclingDocument JSON with picture descriptions
        # ------------------------------------------------------------------
        if descriptions and docling_json is not None:
            for pic_idx, description in descriptions.items():
                if pic_idx < len(docling_json.get("pictures", [])):
                    pic_item = docling_json["pictures"][pic_idx]
                    # Native canonical field
                    pic_item.setdefault("meta", {})["description"] = {
                        "text": description,
                        "created_by": f"llm:{model}",
                    }
                    # Legacy annotation for viewer hover contract
                    if "annotations" not in pic_item:
                        pic_item["annotations"] = []
                    pic_item["annotations"].append({
                        "kind": "description",
                        "text": description,
                        "provenance": f"llm:{model}",
                        "source": "llm",
                        "model": model,
                    })

            enrichments = docling_json.setdefault("_enrichments", {})
            enrichments["version"] = enrichments.get("version", 0) + 1
            upload_bytes_sync(
                json_mod.dumps(docling_json, ensure_ascii=False, default=str).encode("utf-8"),
                bucket,
                f"{base_key}/docling_document.json",
                content_type="application/json; charset=utf-8",
            )

            # Regenerate translated markdown (includes picture descriptions in appendix)
            from app.services.docling_enrichment import _regenerate_translated_markdown
            try:
                translated_md = _regenerate_translated_markdown(docling_json)
                upload_bytes_sync(
                    translated_md.encode("utf-8"),
                    bucket,
                    f"{base_key}/docling_document_translated.md",
                    content_type="text/markdown; charset=utf-8",
                )
            except Exception:
                pass

        if run_id:
            _update_stage_run(
                db, run_id, "derive_picture_descriptions", "COMPLETE",
                attempt=self.request.retries + 1,
                metrics={"pictures_updated": pictures_updated},
            )
            db.commit()

        logger.info("derive_picture_descriptions: document_id=%s updated=%d", document_id, pictures_updated)
        return {"stage": "derive_picture_descriptions", "status": "ok", "pictures_updated": pictures_updated}

    except CeleryRetry:
        raise
    except SoftTimeLimitExceeded as exc:
        logger.warning(
            "derive_picture_descriptions: soft time limit for %s — marking PARTIAL_COMPLETE",
            document_id,
        )
        if run_id:
            try:
                _update_stage_run(db, run_id, "derive_picture_descriptions", "FAILED", attempt=self.request.retries + 1, error="soft time limit exceeded")
                db.commit()
            except Exception:
                pass
        _update_document_status(document_id, STATUS_PARTIAL_COMPLETE, stage="derive_picture_descriptions")
        return {"stage": "derive_picture_descriptions", "status": "timeout", "pictures_updated": 0}
    except Exception as exc:
        logger.error("derive_picture_descriptions failed for %s: %s", document_id, exc)
        if run_id:
            try:
                _update_stage_run(db, run_id, "derive_picture_descriptions", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            except Exception:
                pass
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, soft_time_limit=settings.finalize_soft_time_limit,
                 time_limit=settings.finalize_time_limit, queue="ingest")
def purge_document_derivations(self, document_id: str, run_id: str | None = None) -> str:
    """Delete stale derived data for a document before re-deriving.

    Purges: TextChunks, ImageChunks, ChunkLinks (Postgres),
    ArcadeDB graph vertices + vector embeddings.
    Idempotent — safe to call on first ingest (no-op if nothing exists).
    """
    from app.models.retrieval import TextChunk, ImageChunk, ChunkLink
    from app.models.ingest import DocumentGraphExtraction
    from app.db.session import get_graph_store
    from sqlalchemy import delete as sql_delete

    logger.info("purge_document_derivations: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="purge_document_derivations")

    db = _get_db()
    metrics: dict = {}
    try:
        if run_id:
            _update_stage_run(db, run_id, "purge_document_derivations", "RUNNING", attempt=1)
            db.commit()

        doc_uuid = uuid.UUID(document_id)

        # 1. Postgres derived tables
        for model, label in [
            (ChunkLink, "chunk_links"),
            (TextChunk, "text_chunks"),
            (ImageChunk, "image_chunks"),
        ]:
            result = db.execute(
                sql_delete(model).where(model.document_id == doc_uuid)
            )
            metrics[f"{label}_deleted"] = result.rowcount

        result = db.execute(
            sql_delete(DocumentGraphExtraction).where(
                DocumentGraphExtraction.document_id == doc_uuid
            )
        )
        metrics["graph_extractions_deleted"] = result.rowcount
        db.commit()

        # 2. ArcadeDB graph — delete document structural subgraph (includes vectors)
        try:
            graph_store = get_graph_store()
            deleted = graph_store.delete_document_graph_sync(document_id)
            metrics["graph_elements_deleted"] = deleted
        except Exception as exc:
            logger.warning("purge: graph cleanup failed for %s: %s", document_id, exc)
            metrics["graph_purge_error"] = str(exc)

        if run_id:
            _update_stage_run(db, run_id, "purge_document_derivations", "COMPLETE", attempt=1, metrics=metrics)
            db.commit()

        logger.info("purge_document_derivations: document_id=%s metrics=%s", document_id, metrics)
        return document_id

    except Exception as exc:
        logger.error("purge_document_derivations failed for %s: %s", document_id, exc)
        db.rollback()
        if run_id:
            _update_stage_run(db, run_id, "purge_document_derivations", "FAILED", attempt=1, error=str(exc))
            db.commit()
        raise
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60, queue="embed",
                 soft_time_limit=settings.embed_soft_time_limit,
                 time_limit=settings.embed_time_limit)
def derive_text_chunks_and_embeddings(self, document_id: str, run_id: str | None = None) -> dict:
    """Read text/table/heading document_elements → chunk → BGE embed → upsert text_chunks.

    Uses deterministic chunk keys for idempotent retries.
    """
    from app.models.ingest import Document, DocumentElement
    from app.models.retrieval import TextChunk
    from app.services.chunking import structure_aware_chunk
    from app.services.embedding import embed_texts
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    self.max_retries = settings.embed_max_retries
    self.default_retry_delay = settings.embed_retry_delay
    self.soft_time_limit = settings.embed_soft_time_limit
    self.time_limit = settings.embed_time_limit

    logger.info("derive_text_chunks_and_embeddings: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_text_embeddings")

    db = _get_db()
    try:
        if not run_id:
            run_id = _get_pipeline_run_id(db, document_id)
        if run_id:
            _update_stage_run(db, run_id, "derive_text_embeddings", "RUNNING", attempt=self.request.retries + 1)
            db.commit()

        # Advisory lock to prevent concurrent runs for same document
        db.execute(
            __import__("sqlalchemy").text(
                "SELECT pg_advisory_xact_lock(hashtext(:doc_id || '_text_embed'))"
            ),
            {"doc_id": document_id},
        )

        # Resolve classification from document metadata (fallback: UNCLASSIFIED)
        doc_obj = db.get(Document, uuid.UUID(document_id))
        doc_classification = "UNCLASSIFIED"
        if doc_obj and doc_obj.document_metadata:
            doc_classification = doc_obj.document_metadata.get("classification", "UNCLASSIFIED")

        # ── Primary path: native HybridChunker on enriched DoclingDocument ──
        import json as _json_mod
        from app.services.storage import download_bytes_sync
        use_native_chunking = False
        native_chunks = []

        try:
            _raw = download_bytes_sync(
                settings.minio_bucket_derived,
                f"artifacts/{document_id}/docling_document.json",
            )
            doc_dict = _json_mod.loads(_raw)
            enrichments = doc_dict.get("_enrichments", {})

            if enrichments.get("version") is not None:
                try:
                    from app.services.docling_enrichment import _build_enriched_copy_for_chunking

                    enriched = _build_enriched_copy_for_chunking(doc_dict)
                    from docling.datamodel.document import DoclingDocument as _DLDoc
                    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
                    from transformers import AutoTokenizer
                    from docling.chunking import HybridChunker

                    tok = AutoTokenizer.from_pretrained(settings.chunk_tokenizer_model)
                    hf_tok = HuggingFaceTokenizer(tokenizer=tok, max_tokens=settings.chunk_max_tokens)
                    chunker = HybridChunker(tokenizer=hf_tok)
                    doc_obj_dl = _DLDoc.model_validate(enriched)
                    native_chunks = list(chunker.chunk(doc_obj_dl))
                    use_native_chunking = True
                except Exception as exc:
                    logger.warning("Native HybridChunker failed for %s: %s, falling back", document_id, exc)
                    use_native_chunking = False
            else:
                use_native_chunking = False
        except Exception:
            # docling_document.json not available — fall through to legacy path
            use_native_chunking = False

        chunks_created = 0
        _embed_batch = settings.embed_text_batch_size
        model_version = settings.text_embedding_model

        from app.db.session import get_graph_store
        graph_store = get_graph_store()
        graph_store.ensure_ready_sync()

        if use_native_chunking:
            # ── Native HybridChunker path ──
            all_texts = []
            all_chunk_metas: list[dict] = []
            _seen_chunk_texts: set[str] = set()

            for chunk_idx, chunk in enumerate(native_chunks):
                chunk_text = chunk.text
                if not chunk_text or chunk_text in _seen_chunk_texts:
                    continue
                _seen_chunk_texts.add(chunk_text)

                # Extract page number from chunk metadata if available
                page_number = None
                try:
                    page_number = chunk.meta.doc_items[0].prov[0].page_no
                except (AttributeError, IndexError, TypeError):
                    pass

                # Deterministic chunk_id using same hash pattern as legacy code
                chunk_key = hashlib.sha256(
                    f"{document_id}:native:{chunk_idx}:{model_version}".encode()
                ).hexdigest()
                chunk_id = uuid.UUID(hashlib.md5(chunk_key.encode()).hexdigest())

                all_texts.append(chunk_text)
                all_chunk_metas.append({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "page_number": page_number,
                    "modality": "text",
                })

            if all_texts:
                embeddings: list[list[float]] = []
                for _eb_start in range(0, len(all_texts), _embed_batch):
                    embeddings.extend(embed_texts(all_texts[_eb_start:_eb_start + _embed_batch]))

                from app.services.graph_store import TextChunkRecord as _TCR
                text_chunk_records: list[_TCR] = []
                for meta, text, embedding in zip(all_chunk_metas, all_texts, embeddings):
                    chunk_values = {
                        "id": meta["chunk_id"],
                        "artifact_id": None,
                        "document_id": uuid.UUID(document_id),
                        "chunk_index": meta["chunk_index"],
                        "chunk_text": text,
                        "modality": meta["modality"],
                        "page_number": meta["page_number"],
                        "bounding_box": None,
                    }

                    stmt = pg_insert(TextChunk).values(**chunk_values).on_conflict_do_update(
                        index_elements=["id"],
                        set_={
                            "chunk_text": chunk_values["chunk_text"],
                            "modality": chunk_values["modality"],
                        },
                    )
                    db.execute(stmt)

                    text_chunk_records.append(_TCR(
                        chunk_id=str(meta["chunk_id"]),
                        text=text,
                        document_id=document_id,
                        properties={
                            "artifact_id": None,
                            "modality": meta["modality"],
                            "page_number": meta["page_number"],
                            "classification": doc_classification,
                        },
                        embedding=embedding,
                    ))
                    chunks_created += 1

                if text_chunk_records:
                    graph_store.create_text_chunks_batch_sync(text_chunk_records)

            # Need elements query for Pass 2 (image descriptions)
            elements = db.execute(
                select(DocumentElement).where(
                    DocumentElement.document_id == uuid.UUID(document_id),
                    DocumentElement.element_type.in_(["text", "table", "heading", "equation", "schematic"]),
                    DocumentElement.content_text.isnot(None),
                ).order_by(DocumentElement.element_order)
            ).scalars().all()

            db.commit()

        if not use_native_chunking:
            # ── Legacy path: structure_aware_chunk from DocumentElement rows ──
            elements = db.execute(
                select(DocumentElement).where(
                    DocumentElement.document_id == uuid.UUID(document_id),
                    DocumentElement.element_type.in_(["text", "table", "heading", "equation", "schematic"]),
                    DocumentElement.content_text.isnot(None),
                ).order_by(DocumentElement.element_order)
            ).scalars().all()

            # Convert ORM objects to dicts for structure-aware chunker
            element_dicts = [
                {
                    "element_type": elem.element_type,
                    "content_text": elem.translated_text or elem.content_text,
                    "page_number": elem.page_number,
                    "section_path": elem.section_path,
                    "element_uid": str(elem.element_uid) if elem.element_uid else "",
                    "element_order": elem.element_order,
                    "heading_level": elem.heading_level,
                }
                for elem in elements
                if (elem.translated_text or elem.content_text)
            ]
            structured_chunks = structure_aware_chunk(
                element_dicts,
                max_chunk_tokens=settings.chunk_max_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            )

            # Build a lookup from element_uid to the ORM element for artifact_id / bounding_box
            elem_by_uid: dict[str, DocumentElement] = {
                str(e.element_uid): e for e in elements if e.element_uid
            }

            all_texts = []
            all_chunk_refs = []
            _seen_chunk_texts: set[str] = set()

            for sc in structured_chunks:
                if sc.text in _seen_chunk_texts:
                    continue
                _seen_chunk_texts.add(sc.text)
                all_texts.append(sc.text)
                all_chunk_refs.append(sc)

            if all_texts:
                # Batch embedding to limit memory for very large documents
                embeddings: list[list[float]] = []
                for _eb_start in range(0, len(all_texts), _embed_batch):
                    embeddings.extend(embed_texts(all_texts[_eb_start:_eb_start + _embed_batch]))

                from app.services.graph_store import TextChunkRecord as _TCR
                text_chunk_records: list[_TCR] = []
                for sc, text, embedding in zip(all_chunk_refs, all_texts, embeddings):
                    # Resolve artifact_id from the first element_uid in this chunk
                    first_uid = sc.element_uids[0] if sc.element_uids else ""
                    ref_elem = elem_by_uid.get(first_uid)
                    artifact_id = ref_elem.artifact_id if ref_elem else None
                    bounding_box = ref_elem.bounding_box if ref_elem else None

                    # Deterministic chunk key using element_uids for stability
                    uid_key = "|".join(sc.element_uids)
                    chunk_key = hashlib.sha256(
                        f"{document_id}:{uid_key}:{sc.chunk_index}:{model_version}".encode()
                    ).hexdigest()

                    chunk_id = uuid.UUID(hashlib.md5(chunk_key.encode()).hexdigest())

                    chunk_values = {
                        "id": chunk_id,
                        "artifact_id": artifact_id,
                        "document_id": uuid.UUID(document_id),
                        "chunk_index": sc.chunk_index,
                        "chunk_text": text,
                        "modality": sc.modality,
                        "page_number": sc.page_number,
                        "bounding_box": bounding_box,
                    }

                    stmt = pg_insert(TextChunk).values(**chunk_values).on_conflict_do_update(
                        index_elements=["id"],
                        set_={
                            "chunk_text": chunk_values["chunk_text"],
                            "modality": chunk_values["modality"],
                        },
                    )
                    db.execute(stmt)

                    text_chunk_records.append(_TCR(
                        chunk_id=str(chunk_id),
                        text=text,
                        document_id=document_id,
                        properties={
                            "artifact_id": str(artifact_id) if artifact_id else None,
                            "modality": sc.modality,
                            "page_number": sc.page_number,
                            "classification": doc_classification,
                        },
                        embedding=embedding,
                    ))
                    chunks_created += 1

                # Batch-create all TextChunk vertices (with embeddings) in one HTTP call
                if text_chunk_records:
                    graph_store.create_text_chunks_batch_sync(text_chunk_records)

            db.commit()

        # ── Pass 2: Image description sections ──────────────────────────
        from app.services.chunking import split_description_sections
        from app.models.retrieval import ChunkLink

        img_elements = db.execute(
            select(DocumentElement).where(
                DocumentElement.document_id == uuid.UUID(document_id),
                DocumentElement.element_type == "image",
                DocumentElement.content_text.isnot(None),
                DocumentElement.content_text != "",
                DocumentElement.artifact_id.isnot(None),
            ).order_by(DocumentElement.element_order)
        ).scalars().all()

        img_desc_chunks_created = 0
        img_desc_texts: list[str] = []
        img_desc_chunk_metas: list[dict] = []

        for img_elem in img_elements:
            # Normalize Unicode to prevent NaN embeddings (same pattern as text chunk pass)
            desc_text = _normalize_text(img_elem.content_text)
            sections = split_description_sections(desc_text)
            if not sections:
                continue

            for sec_idx, section_text in enumerate(sections):
                chunk_index = 100000 + img_elem.element_order * 100 + sec_idx
                uid_str = str(img_elem.element_uid) if img_elem.element_uid else str(img_elem.id)
                chunk_key = hashlib.sha256(
                    f"{document_id}:{uid_str}:{sec_idx}:{model_version}".encode()
                ).hexdigest()
                chunk_id = uuid.UUID(hashlib.md5(chunk_key.encode()).hexdigest())

                img_desc_texts.append(section_text)
                img_desc_chunk_metas.append({
                    "chunk_id": chunk_id,
                    "artifact_id": img_elem.artifact_id,
                    "document_id": uuid.UUID(document_id),
                    "chunk_index": chunk_index,
                    "page_number": img_elem.page_number,
                    "section_text": section_text,
                    "element_order": img_elem.element_order,
                    "sec_idx": sec_idx,
                })

        # Batch embed all image description sections
        if img_desc_texts:
            img_desc_embeddings: list[list[float]] = []
            for _eb_start in range(0, len(img_desc_texts), _embed_batch):
                img_desc_embeddings.extend(
                    embed_texts(img_desc_texts[_eb_start:_eb_start + _embed_batch])
                )

            # Create TextChunk rows in Postgres + ArcadeDB vertices with embeddings
            from app.services.graph_store import TextChunkRecord as _TCR2
            img_desc_records: list[_TCR2] = []
            for meta, emb in zip(img_desc_chunk_metas, img_desc_embeddings):
                chunk_values = {
                    "id": meta["chunk_id"],
                    "artifact_id": meta["artifact_id"],
                    "document_id": meta["document_id"],
                    "chunk_index": meta["chunk_index"],
                    "chunk_text": meta["section_text"],
                    "modality": "image_description",
                    "page_number": meta["page_number"],
                    "bounding_box": None,
                    "classification": doc_classification,
                }

                stmt = pg_insert(TextChunk).values(**chunk_values).on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "chunk_text": chunk_values["chunk_text"],
                        "modality": chunk_values["modality"],
                    },
                )
                db.execute(stmt)

                img_desc_records.append(_TCR2(
                    chunk_id=str(meta["chunk_id"]),
                    text=meta["section_text"],
                    document_id=document_id,
                    properties={
                        "artifact_id": str(meta["artifact_id"]),
                        "modality": "image_description",
                        "page_number": meta["page_number"],
                        "classification": doc_classification,
                    },
                    embedding=emb,
                ))
                img_desc_chunks_created += 1

            # Batch-create all image-description TextChunk vertices in one HTTP call
            if img_desc_records:
                graph_store.create_text_chunks_batch_sync(img_desc_records)

            # SAME_ARTIFACT chunk_links (neighbor-only) between consecutive sections
            from collections import defaultdict
            artifact_section_chunks: dict[str, list[uuid.UUID]] = defaultdict(list)
            for meta in img_desc_chunk_metas:
                artifact_section_chunks[str(meta["artifact_id"])].append(meta["chunk_id"])

            for art_id, chunk_ids_list in artifact_section_chunks.items():
                for i in range(len(chunk_ids_list) - 1):
                    for src, tgt in [(chunk_ids_list[i], chunk_ids_list[i + 1]),
                                     (chunk_ids_list[i + 1], chunk_ids_list[i])]:
                        link_vals = {
                            "source_chunk_id": src,
                            "target_chunk_id": tgt,
                            "document_id": uuid.UUID(document_id),
                            "link_type": "SAME_ARTIFACT",
                            "hop": 1,
                            "weight": settings.retrieval_weight_same_artifact,
                        }
                        link_stmt = pg_insert(ChunkLink).values(**link_vals).on_conflict_do_update(
                            constraint="chunk_links_pkey",
                            set_={"weight": link_vals["weight"]},
                        )
                        db.execute(link_stmt)

            db.commit()

        chunks_created += img_desc_chunks_created

        if run_id:
            _update_stage_run(
                db, run_id, "derive_text_embeddings", "COMPLETE",
                attempt=self.request.retries + 1,
                metrics={"chunks": chunks_created, "elements": len(elements), "img_desc_chunks": img_desc_chunks_created},
            )
            db.commit()

        logger.info(
            "derive_text_chunks_and_embeddings: document_id=%s chunks=%d img_desc_chunks=%d",
            document_id, chunks_created, img_desc_chunks_created,
        )
        return {"stage": "derive_text_embeddings", "status": "ok", "chunks": chunks_created}

    except CeleryRetry:
        raise
    except SoftTimeLimitExceeded:
        logger.warning("derive_text_chunks_and_embeddings: soft time limit for %s", document_id)
        db.rollback()
        if run_id:
            _update_stage_run(db, run_id, "derive_text_embeddings", "FAILED",
                              attempt=self.request.retries + 1, error="soft time limit exceeded")
            db.commit()
        return {"stage": "derive_text_embeddings", "status": "failed", "error": "soft time limit exceeded"}
    except Exception as exc:
        logger.error("derive_text_chunks_and_embeddings failed for %s: %s", document_id, exc)
        db.rollback()
        if self.request.retries >= self.max_retries:
            _update_document_status(
                document_id, STATUS_PARTIAL_COMPLETE,
                stage="derive_text_embeddings", error=str(exc),
            )
            if run_id:
                _update_stage_run(db, run_id, "derive_text_embeddings", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            return {"stage": "derive_text_embeddings", "status": "failed", "error": str(exc)}
        if run_id:
            _update_stage_run(db, run_id, "derive_text_embeddings", "FAILED", attempt=self.request.retries + 1, error=str(exc))
            db.commit()
        logger.info("derive_text_chunks_and_embeddings: retrying %s (attempt %d/%d)", document_id, self.request.retries + 1, self.max_retries)
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60, queue="embed",
                 soft_time_limit=settings.embed_soft_time_limit,
                 time_limit=settings.embed_time_limit)
def derive_image_embeddings(self, document_id: str, run_id: str | None = None) -> dict:
    """Read image document_elements → CLIP embed → upsert image_chunks.

    Uses deterministic chunk keys for idempotent retries.
    """
    import io
    from app.models.ingest import Document, DocumentElement
    from app.models.retrieval import ImageChunk
    from app.services.embedding import embed_images
    from app.services.storage import download_bytes_sync
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    self.max_retries = settings.embed_max_retries
    self.default_retry_delay = settings.embed_retry_delay
    self.soft_time_limit = settings.embed_soft_time_limit
    self.time_limit = settings.embed_time_limit

    logger.info("derive_image_embeddings: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_image_embeddings")

    db = _get_db()
    try:
        if not run_id:
            run_id = _get_pipeline_run_id(db, document_id)
        if run_id:
            _update_stage_run(db, run_id, "derive_image_embeddings", "RUNNING", attempt=self.request.retries + 1)
            db.commit()

        # Resolve classification from document metadata (fallback: UNCLASSIFIED)
        doc_obj = db.get(Document, uuid.UUID(document_id))
        doc_classification = "UNCLASSIFIED"
        if doc_obj and doc_obj.document_metadata:
            doc_classification = doc_obj.document_metadata.get("classification", "UNCLASSIFIED")

        db.execute(
            __import__("sqlalchemy").text(
                "SELECT pg_advisory_xact_lock(hashtext(:doc_id || '_image_embed'))"
            ),
            {"doc_id": document_id},
        )

        elements = db.execute(
            select(DocumentElement).where(
                DocumentElement.document_id == uuid.UUID(document_id),
                DocumentElement.element_type == "image",
                DocumentElement.storage_key.isnot(None),
            ).order_by(DocumentElement.element_order)
        ).scalars().all()

        chunks_created = 0
        if elements:
            from PIL import Image

            pil_images = []
            valid_elements = []
            for elem in elements:
                try:
                    img_bytes = download_bytes_sync(elem.storage_bucket, elem.storage_key)
                    pil_images.append(Image.open(io.BytesIO(img_bytes)))
                    valid_elements.append(elem)
                except Exception as e:
                    logger.warning("Could not load image element %s: %s", elem.element_uid, e)

            if pil_images:
                image_embeddings = embed_images(pil_images)
                model_version = settings.image_embedding_model

                from app.db.session import get_graph_store
                graph_store = get_graph_store()
                graph_store.ensure_ready_sync()

                from app.services.graph_store import ImageChunkRecord as _ICR
                image_chunk_records: list[_ICR] = []
                for elem, img_embedding in zip(valid_elements, image_embeddings):
                    chunk_key = hashlib.sha256(
                        f"{document_id}:{elem.element_uid}:{model_version}".encode()
                    ).hexdigest()

                    chunk_id = uuid.UUID(hashlib.md5(chunk_key.encode()).hexdigest())

                    chunk_values = {
                        "id": chunk_id,
                        "artifact_id": elem.artifact_id,
                        "document_id": uuid.UUID(document_id),
                        "chunk_index": 0,
                        "chunk_text": elem.content_text or None,
                        "modality": "image",
                        "page_number": elem.page_number,
                        "bounding_box": elem.bounding_box,
                    }

                    stmt = pg_insert(ImageChunk).values(**chunk_values).on_conflict_do_update(
                        index_elements=["id"],
                        set_={
                            "chunk_text": chunk_values["chunk_text"],
                        },
                    )
                    db.execute(stmt)

                    image_chunk_records.append(_ICR(
                        chunk_id=str(chunk_id),
                        document_id=document_id,
                        properties={
                            "artifact_id": str(elem.artifact_id),
                            "modality": "image",
                            "page_number": elem.page_number,
                            "classification": doc_classification,
                            "chunk_text": elem.content_text or "",
                        },
                        embedding=img_embedding,
                    ))
                    chunks_created += 1

                # Batch-create all ImageChunk vertices (with embeddings) in one HTTP call
                if image_chunk_records:
                    graph_store.create_image_chunks_batch_sync(image_chunk_records)

        db.commit()

        if run_id:
            _update_stage_run(
                db, run_id, "derive_image_embeddings", "COMPLETE",
                attempt=self.request.retries + 1,
                metrics={"chunks": chunks_created, "elements": len(elements)},
            )
            db.commit()

        logger.info(
            "derive_image_embeddings: document_id=%s chunks=%d",
            document_id, chunks_created,
        )
        return {"stage": "derive_image_embeddings", "status": "ok", "chunks": chunks_created}

    except CeleryRetry:
        raise
    except SoftTimeLimitExceeded:
        logger.warning("derive_image_embeddings: soft time limit for %s", document_id)
        db.rollback()
        if run_id:
            _update_stage_run(db, run_id, "derive_image_embeddings", "FAILED",
                              attempt=self.request.retries + 1, error="soft time limit exceeded")
            db.commit()
        return {"stage": "derive_image_embeddings", "status": "failed", "error": "soft time limit exceeded"}
    except Exception as exc:
        logger.error("derive_image_embeddings failed for %s: %s", document_id, exc)
        db.rollback()
        if self.request.retries >= self.max_retries:
            _update_document_status(
                document_id, STATUS_PARTIAL_COMPLETE,
                stage="derive_image_embeddings", error=str(exc),
            )
            if run_id:
                _update_stage_run(db, run_id, "derive_image_embeddings", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            return {"stage": "derive_image_embeddings", "status": "failed", "error": str(exc)}
        logger.info("derive_image_embeddings: retrying %s (attempt %d/%d)", document_id, self.request.retries + 1, self.max_retries)
        raise self.retry(exc=exc)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# New bundle_passes branch — spec §5.4 orchestrator. Task 4.6.
# ---------------------------------------------------------------------------

def _derive_ontology_graph_bundle_passes(self, pipeline_run_id: str, document_id: str) -> dict:
    """New path: fixed per-pass templates, merge, import, rollback. Spec §5.4."""
    from app.models.ingest import PipelineRun, StageRun
    from datetime import datetime
    from types import SimpleNamespace

    # 1. Stage-summary row
    db = _get_db()
    try:
        run = db.get(PipelineRun, uuid.UUID(pipeline_run_id))
        stage_summary = StageRun(
            pipeline_run_id=uuid.UUID(pipeline_run_id),
            stage_name="derive_ontology_graph",
            pass_name=None,
            attempt=self.request.retries + 1,
            status="RUNNING",
            started_at=datetime.utcnow(),
        )
        db.add(stage_summary)
        db.flush()
        stage_summary_id = stage_summary.id
        run_document_id = str(run.document_id)
        run_mode = run.mode
        bundle_key = run.ontology_bundle_key
        db.commit()
    finally:
        db.close()

    tracker = GraphWriteTracker()

    def _terminalize_failure(exc_type, error_msg, should_rollback):
        rollback_note = _attempt_rollback(run_document_id) if should_rollback else ""
        db2 = _get_db()
        try:
            from datetime import datetime as dt
            row = db2.get(StageRun, stage_summary_id)
            if row:
                row.status = "FAILED"
                row.execution_status = "FAILED"
                row.rollback_executed = should_rollback
                row.error_message = f"{exc_type}: {error_msg}{rollback_note}"
                row.finished_at = dt.utcnow()
            run_row = db2.get(PipelineRun, uuid.UUID(pipeline_run_id))
            if run_row:
                run_row.status = "FAILED"
                run_row.finished_at = dt.utcnow()
            db2.commit()
        except Exception as bookkeeping_exc:
            db2.rollback()
            logger.error(
                "derive_ontology_graph: bookkeeping also failed: %s", bookkeeping_exc
            )
        finally:
            db2.close()
        _update_document_pipeline_status(run_document_id, "PARTIAL_COMPLETE")

    try:
        manifest = load_bundle_manifest(bundle_key)
        ontology = load_ontology(bundle_key=bundle_key)
        doc_json = _build_docling_document_json(run_document_id)

        pass_results: dict = {}
        upstream_refs: dict = {}

        for pass_def in manifest.passes:
            _run_single_pass(
                pipeline_run_id=pipeline_run_id,
                pass_def=pass_def,
                manifest=manifest,
                ontology=ontology,
                bundle_key=bundle_key,
                doc_json=doc_json,
                pass_results=pass_results,
                upstream_refs=upstream_refs,
                document_id=run_document_id,
            )

        gate = check_required_pass_gate(pipeline_run_id)
        if not gate.passed:
            raise IngestFailed(f"Required passes failed: {gate.failures}")

        merged = merge_and_resolve(
            pass_results=pass_results,
            manifest=manifest,
            ontology=ontology,
            document_id=run_document_id,
            pipeline_run_id=str(pipeline_run_id),
        )

        _apply_post_merge_yield_updates(pipeline_run_id, merged)
        _write_pipeline_run_metrics(pipeline_run_id, merged, manifest)

        identity_to_rid = _import_graph_phase_nodes(
            merged, ontology, run_document_id, tracker,
        )
        _import_graph_phase_domain_edges(merged, ontology, tracker)
        _import_graph_phase_structural_edges(
            merged, identity_to_rid, run_document_id, str(pipeline_run_id), tracker,
        )

        # Build a detachment-safe snapshot so _upsert_document_graph_extraction
        # can access run metadata after the original DB session was closed.
        run_snapshot = SimpleNamespace(
            ontology_bundle_key=bundle_key,
            ontology_name=getattr(manifest, "ontology_name", None),
            ontology_version=getattr(manifest, "ontology_version", None),
            use_case_key=None,
            extraction_profile_version=getattr(manifest, "extraction_profile_version", None),
        )

        _upsert_document_graph_extraction(
            document_id=run_document_id,
            pipeline_run_id=pipeline_run_id,
            run=run_snapshot,
            merged=merged,
            manifest=manifest,
        )

        # Success terminalization
        db3 = _get_db()
        try:
            from datetime import datetime as dt
            row = db3.get(StageRun, stage_summary_id)
            if row:
                row.status = "COMPLETE"
                row.execution_status = "COMPLETE"
                row.rollback_executed = False
                row.finished_at = dt.utcnow()

            run_row = db3.get(PipelineRun, uuid.UUID(pipeline_run_id))
            if run_row and run_mode == "graph_only":
                run_row.status = "COMPLETE"
                run_row.finished_at = dt.utcnow()
            db3.commit()
        finally:
            db3.close()

        if run_mode == "graph_only":
            _update_document_pipeline_status(run_document_id, "COMPLETE")

        return {
            "stage": "derive_ontology_graph",
            "status": "ok",
            "entities": len(merged.entities),
            "edges": len(merged.edges),
        }

    except IngestFailed as exc:
        _terminalize_failure("gate_failed", str(exc), should_rollback=False)
        raise
    except Exception as exc:
        logger.exception("derive_ontology_graph bundle_passes failure")
        _terminalize_failure(
            "unexpected_failure", str(exc),
            should_rollback=tracker.any_mutation_attempted,
        )
        raise


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60, queue="graph",
                 soft_time_limit=settings.graph_soft_time_limit,
                 time_limit=settings.graph_time_limit)
def derive_ontology_graph(self, document_id: str, run_id: str | None = None) -> dict:
    """Dispatch graph extraction via the bundle-passes orchestrator.

    Callers always dispatch as derive_ontology_graph.si(document_id, run_id),
    where run_id IS the pipeline_run_id. The legacy path was removed in Task 5.2.
    """
    return _derive_ontology_graph_bundle_passes(self, run_id, document_id)


@celery_app.task(bind=True, max_retries=1, default_retry_delay=30, queue="graph",
                 soft_time_limit=settings.finalize_soft_time_limit,
                 time_limit=settings.finalize_time_limit)
def derive_structure_links(self, document_id: str, run_id: str | None = None) -> dict:
    """Generate chunk_links and structural ArcadeDB edges.

    Creates:
    - NEXT_CHUNK links between consecutive text_chunks
    - SAME_PAGE links between text and image chunks on same page
    - SAME_SECTION links for chunks sharing section_path
    - DOCUMENT node, TextChunk/ImageChunk vertices, CONTAINS/SAME_PAGE ArcadeDB edges
    - EXTRACTED_FROM edges linking ontology entities to chunk vertices
    """
    from app.models.ingest import Document, DocumentElement, Artifact
    from app.models.retrieval import TextChunk, ImageChunk, ChunkLink
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    self.max_retries = settings.finalize_max_retries
    self.default_retry_delay = settings.finalize_retry_delay
    self.soft_time_limit = settings.finalize_soft_time_limit
    self.time_limit = settings.finalize_time_limit

    logger.info("derive_structure_links: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_structure_links")

    db = _get_db()
    try:
        if not run_id:
            run_id = _get_pipeline_run_id(db, document_id)
        if run_id:
            _update_stage_run(db, run_id, "derive_structure_links", "RUNNING", attempt=self.request.retries + 1)
            db.commit()

        db.execute(
            __import__("sqlalchemy").text(
                "SELECT pg_advisory_xact_lock(hashtext(:doc_id || '_structure_links'))"
            ),
            {"doc_id": document_id},
        )

        doc = db.get(Document, uuid.UUID(document_id))
        if not doc:
            logger.warning("derive_structure_links: document %s not found", document_id)
            return {"stage": "derive_structure_links", "status": "skipped"}

        # Fetch chunks
        text_chunks = db.execute(
            select(TextChunk)
            .where(TextChunk.document_id == uuid.UUID(document_id))
            .order_by(TextChunk.page_number.nullslast(), TextChunk.chunk_index)
        ).scalars().all()

        image_chunks = db.execute(
            select(ImageChunk)
            .where(ImageChunk.document_id == uuid.UUID(document_id))
        ).scalars().all()

        # Fetch document_elements for section_path metadata
        elements = db.execute(
            select(DocumentElement)
            .where(DocumentElement.document_id == uuid.UUID(document_id))
            .order_by(DocumentElement.element_order)
        ).scalars().all()

        # Build element_uid → section_path map
        element_section_map = {}
        for elem in elements:
            if elem.element_uid and elem.section_path:
                element_section_map[elem.element_uid] = elem.section_path

        # Build artifact_id → element map for section lookups
        artifact_element_map = {}
        for elem in elements:
            if elem.artifact_id:
                artifact_element_map[str(elem.artifact_id)] = elem

        links_created = 0

        def _upsert_link(source_id, target_id, link_type, hop, weight):
            nonlocal links_created
            vals = {
                "source_chunk_id": uuid.UUID(str(source_id)),
                "target_chunk_id": uuid.UUID(str(target_id)),
                "document_id": uuid.UUID(document_id),
                "link_type": link_type,
                "hop": hop,
                "weight": weight,
            }
            stmt = pg_insert(ChunkLink).values(**vals).on_conflict_do_update(
                constraint="chunk_links_pkey",
                set_={"weight": weight},
            )
            db.execute(stmt)
            links_created += 1

        # NEXT_CHUNK links (consecutive text chunks)
        for i in range(len(text_chunks) - 1):
            _upsert_link(
                text_chunks[i].id, text_chunks[i + 1].id,
                "NEXT_CHUNK", 1, settings.retrieval_weight_next_chunk,
            )
            # Bidirectional
            _upsert_link(
                text_chunks[i + 1].id, text_chunks[i].id,
                "NEXT_CHUNK", 1, settings.retrieval_weight_next_chunk,
            )

        # SAME_PAGE links
        page_text_map: dict[int, list] = {}
        for tc in text_chunks:
            if tc.page_number is not None:
                page_text_map.setdefault(tc.page_number, []).append(tc)

        page_image_map: dict[int, list] = {}
        for ic in image_chunks:
            if ic.page_number is not None:
                page_image_map.setdefault(ic.page_number, []).append(ic)

        for page_num, ics in page_image_map.items():
            tcs = page_text_map.get(page_num, [])
            for ic in ics:
                for tc in tcs:
                    _upsert_link(
                        tc.id, ic.id, "SAME_PAGE", 1,
                        settings.retrieval_weight_same_page,
                    )
                    _upsert_link(
                        ic.id, tc.id, "SAME_PAGE", 1,
                        settings.retrieval_weight_same_page,
                    )

        # SAME_SECTION links — neighbor-only (prev/next by position) to avoid O(n²)
        section_chunks: dict[str, list] = {}
        for tc in text_chunks:
            if tc.artifact_id and str(tc.artifact_id) in artifact_element_map:
                elem = artifact_element_map[str(tc.artifact_id)]
                if elem.section_path:
                    section_chunks.setdefault(elem.section_path, []).append(tc)

        for section, chunks in section_chunks.items():
            for i in range(len(chunks) - 1):
                _upsert_link(
                    chunks[i].id, chunks[i + 1].id, "SAME_SECTION", 1,
                    settings.retrieval_weight_same_section,
                )
                _upsert_link(
                    chunks[i + 1].id, chunks[i].id, "SAME_SECTION", 1,
                    settings.retrieval_weight_same_section,
                )

        # SAME_ARTIFACT links — neighbor-only (prev/next by position) to avoid O(n²)
        artifact_chunks: dict[str, list] = {}
        for tc in text_chunks:
            if tc.artifact_id:
                artifact_chunks.setdefault(str(tc.artifact_id), []).append(tc)

        for art_id, chunks in artifact_chunks.items():
            for i in range(len(chunks) - 1):
                _upsert_link(
                    chunks[i].id, chunks[i + 1].id, "SAME_ARTIFACT", 1,
                    settings.retrieval_weight_same_artifact,
                )
                _upsert_link(
                    chunks[i + 1].id, chunks[i].id, "SAME_ARTIFACT", 1,
                    settings.retrieval_weight_same_artifact,
                )

        db.commit()

        # Create graph structural edges via GraphStore
        from app.db.session import get_graph_store
        graph_store = get_graph_store()
        graph_store.ensure_ready_sync()

        # Include document metadata as properties
        doc_node_props: dict[str, Any] = {"source_id": str(doc.source_id), "title": doc.filename}
        if doc.document_metadata and isinstance(doc.document_metadata, dict):
            if doc.document_metadata.get("document_summary"):
                doc_node_props["summary"] = doc.document_metadata["document_summary"]
            if doc.document_metadata.get("classification"):
                doc_node_props["classification"] = doc.document_metadata["classification"]
            if doc.document_metadata.get("date_of_information"):
                doc_node_props["date_of_information"] = doc.document_metadata["date_of_information"]
            if doc.document_metadata.get("source_characterization"):
                doc_node_props["source_characterization"] = doc.document_metadata["source_characterization"]

        # Upsert Document vertex (sync for Celery)
        from app.services.graph_store import NodeRecord as _NR
        doc_rid = graph_store.upsert_node_sync(_NR(
            entity_type="Document",
            identity_fields={"document_id": document_id},
            name=doc.filename,
            properties=doc_node_props,
        ))

        # Connect existing TextChunk/ImageChunk vertices (created in embedding stages) to Document
        # Use get_chunk_rid_sync to look up already-created vertices — do NOT re-create them
        tc_rid_map: dict[str, str] = {}
        for tc in text_chunks:
            tc_rid = graph_store.get_chunk_rid_sync(str(tc.id), "TextChunk")
            if tc_rid:
                tc_rid_map[str(tc.id)] = tc_rid
                graph_store.create_structural_edge_sync(doc_rid, tc_rid, "CONTAINS_TEXT")
            else:
                logger.warning("derive_structure_links: TextChunk vertex not found for chunk %s", tc.id)

        ic_rid_map: dict[str, str] = {}
        for ic in image_chunks:
            ic_rid = graph_store.get_chunk_rid_sync(str(ic.id), "ImageChunk")
            if ic_rid:
                ic_rid_map[str(ic.id)] = ic_rid
                graph_store.create_structural_edge_sync(doc_rid, ic_rid, "CONTAINS_IMAGE")
            else:
                logger.warning("derive_structure_links: ImageChunk vertex not found for chunk %s", ic.id)

        for page_num, ics in page_image_map.items():
            tcs = page_text_map.get(page_num, [])
            for ic in ics:
                ic_rid = ic_rid_map.get(str(ic.id))
                if not ic_rid:
                    continue
                for tc in tcs:
                    tc_rid = tc_rid_map.get(str(tc.id))
                    if not tc_rid:
                        continue
                    # SAME_PAGE edges between text and image chunks using resolved RIDs
                    try:
                        graph_store.create_structural_edge_sync(tc_rid, ic_rid, "SAME_PAGE")
                    except Exception:
                        pass  # Best-effort for cross-chunk edges

        # Create doc-structure edges in ArcadeDB (NEXT_CHUNK, SAME_SECTION,
        # SAME_ARTIFACT) with weight properties, mirroring the Postgres
        # chunk_links. These enable ArcadeDB-native retrieval expansion.
        structural_edge_sql = []
        structural_params: dict[str, Any] = {}
        edge_idx = 0

        def _add_structural_edge(from_rid: str, to_rid: str, link_type: str, weight: float) -> None:
            nonlocal edge_idx
            structural_edge_sql.append(
                f"CREATE EDGE {link_type} FROM {from_rid} TO {to_rid} "
                f"SET weight = :w_{edge_idx}, created_at = sysdate()"
            )
            structural_params[f"w_{edge_idx}"] = weight
            edge_idx += 1

        # NEXT_CHUNK (consecutive text chunks)
        for i in range(len(text_chunks) - 1):
            src_rid = tc_rid_map.get(str(text_chunks[i].id))
            tgt_rid = tc_rid_map.get(str(text_chunks[i + 1].id))
            if src_rid and tgt_rid:
                _add_structural_edge(src_rid, tgt_rid, "NEXT_CHUNK", settings.retrieval_weight_next_chunk)
                _add_structural_edge(tgt_rid, src_rid, "NEXT_CHUNK", settings.retrieval_weight_next_chunk)

        # SAME_SECTION (neighbor-only within section)
        for section, chunks in section_chunks.items():
            sec_rids = [(str(c.id), tc_rid_map.get(str(c.id))) for c in chunks]
            for i in range(len(sec_rids) - 1):
                _, r1 = sec_rids[i]
                _, r2 = sec_rids[i + 1]
                if r1 and r2:
                    _add_structural_edge(r1, r2, "SAME_SECTION", settings.retrieval_weight_same_section)
                    _add_structural_edge(r2, r1, "SAME_SECTION", settings.retrieval_weight_same_section)

        # SAME_ARTIFACT (neighbor-only within artifact)
        for art_id, chunks in artifact_chunks.items():
            art_rids = [(str(c.id), tc_rid_map.get(str(c.id))) for c in chunks]
            for i in range(len(art_rids) - 1):
                _, r1 = art_rids[i]
                _, r2 = art_rids[i + 1]
                if r1 and r2:
                    _add_structural_edge(r1, r2, "SAME_ARTIFACT", settings.retrieval_weight_same_artifact)
                    _add_structural_edge(r2, r1, "SAME_ARTIFACT", settings.retrieval_weight_same_artifact)

        if structural_edge_sql:
            try:
                graph_store._client.command_sync(
                    graph_store._database, "sqlscript",
                    ";\n".join(structural_edge_sql), structural_params,
                )
                logger.info(
                    "derive_structure_links: created %d ArcadeDB structural edges for %s",
                    len(structural_edge_sql), document_id,
                )
            except Exception as exc:
                logger.warning(
                    "derive_structure_links: ArcadeDB structural edges failed for %s: %s",
                    document_id, exc,
                )

        # Entity-chunk EXTRACTED_FROM edges
        entity_links = 0

        # Build element_uid → chunk_ids map (via artifact_id).
        # Include BOTH text_chunks AND image_chunks so entities grounded
        # in images/schematics get linked to the corresponding ImageChunk.
        element_uid_chunk_map: dict[str, list[str]] = {}
        artifact_id_to_element_uid: dict[str, str] = {}
        for elem in elements:
            if elem.artifact_id and elem.element_uid:
                artifact_id_to_element_uid[str(elem.artifact_id)] = elem.element_uid
        for tc in text_chunks:
            if tc.artifact_id:
                euid = artifact_id_to_element_uid.get(str(tc.artifact_id))
                if euid:
                    element_uid_chunk_map.setdefault(euid, []).append(str(tc.id))
        for ic in image_chunks:
            if ic.artifact_id:
                euid = artifact_id_to_element_uid.get(str(ic.artifact_id))
                if euid:
                    element_uid_chunk_map.setdefault(euid, []).append(str(ic.id))

        # Try graph_json mentions path first (new pipeline)
        from app.models.ingest import DocumentGraphExtraction
        graph_extraction = db.execute(
            select(DocumentGraphExtraction).where(
                DocumentGraphExtraction.document_id == uuid.UUID(document_id),
            )
        ).scalars().first()

        # Collect all entity-chunk edges, then batch-create in one call
        edge_tuples: list[tuple[str, str, str]] = []  # (name, type, chunk_id)

        # Track which entities got at least one mention via the primary path
        mentioned_entities: set[str] = set()
        all_extracted_entities: list[tuple[str, str]] = []

        if graph_extraction and graph_extraction.graph_json:
            mentions = graph_extraction.graph_json.get("mentions", [])
            for mention in mentions:
                name = mention.get("entity_name", "")
                etype = mention.get("entity_type", "UNKNOWN")
                euid = mention.get("element_uid", "")
                for chunk_id in element_uid_chunk_map.get(euid, []):
                    edge_tuples.append((name, etype, chunk_id))
                    mentioned_entities.add(name)

            # Collect all entity names from the extraction for fallback
            for node in graph_extraction.graph_json.get("nodes", []):
                n = node.get("name", node.get("id", ""))
                t = node.get("entity_type", "UNKNOWN")
                if n:
                    all_extracted_entities.append((n, t))

        # Fallback: for entities with ZERO mentions from the primary path
        # (or all entities when no mentions at all), use the artifact-wide
        # linking. This catches partial misses, not just complete failure.
        entities_needing_fallback = [
            (n, t) for n, t in all_extracted_entities if n not in mentioned_entities
        ]
        if entities_needing_fallback or not mentioned_entities:
            # Include BOTH text and image chunks in the fallback artifact map
            artifact_chunk_map: dict[str, list[str]] = {}
            for tc in text_chunks:
                artifact_chunk_map.setdefault(str(tc.artifact_id), []).append(str(tc.id))
            for ic in image_chunks:
                artifact_chunk_map.setdefault(str(ic.artifact_id), []).append(str(ic.id))

            artifacts_with_entities = db.execute(
                select(Artifact).where(
                    Artifact.document_id == uuid.UUID(document_id),
                    Artifact.content_metadata.isnot(None),
                )
            ).scalars().all()

            for artifact in artifacts_with_entities:
                metadata = artifact.content_metadata or {}
                chunk_ids = artifact_chunk_map.get(str(artifact.id), [])
                if not chunk_ids:
                    continue

                entities_list: list[tuple[str, str]] = []
                graph_data = metadata.get("docling_graph_data")
                if graph_data:
                    for node in graph_data.get("nodes", []):
                        entities_list.append((
                            node.get("name", node.get("id", "")),
                            node.get("entity_type", "UNKNOWN"),
                        ))
                else:
                    for ent in metadata.get("extracted_entities", []):
                        entities_list.append((ent["name"], ent["entity_type"]))

                for (name, etype) in entities_list:
                    for chunk_id in chunk_ids:
                        edge_tuples.append((name, etype, chunk_id))

        # Batch-create EXTRACTED_FROM edges in one sqlscript call.
        # Look up chunk RIDs from both text and image maps.
        from app.services.graph_store import EntityChunkEdge as _ECE
        entity_edge_records: list[_ECE] = []
        for (ent_name, ent_type, chunk_id) in edge_tuples:
            chunk_rid = tc_rid_map.get(chunk_id) or ic_rid_map.get(chunk_id)
            if not chunk_rid:
                continue
            entity_edge_records.append(_ECE(
                entity_name=ent_name,
                entity_type=ent_type,
                chunk_rid=chunk_rid,
            ))

        entity_links = 0
        if entity_edge_records:
            try:
                entity_links = graph_store.batch_create_entity_chunk_edges_sync(
                    entity_edge_records,
                )
            except Exception as exc:
                logger.warning(
                    "derive_structure_links: batch entity-chunk edge creation failed: %s",
                    exc,
                )

        db.commit()

        if run_id:
            _update_stage_run(
                db, run_id, "derive_structure_links", "COMPLETE",
                attempt=self.request.retries + 1,
                metrics={
                    "chunk_links": links_created,
                    "entity_links": entity_links,
                    "text_chunks": len(text_chunks),
                    "image_chunks": len(image_chunks),
                },
            )
            db.commit()

        logger.info(
            "derive_structure_links: document_id=%s chunk_links=%d entity_links=%d",
            document_id, links_created, entity_links,
        )
        return {"stage": "derive_structure_links", "status": "ok", "links": links_created}

    except CeleryRetry:
        raise
    except Exception as exc:
        logger.error("derive_structure_links failed for %s: %s", document_id, exc)
        db.rollback()
        if self.request.retries >= self.max_retries:
            _update_document_status(
                document_id, STATUS_PARTIAL_COMPLETE,
                stage="derive_structure_links", error=str(exc),
            )
            if run_id:
                _update_stage_run(db, run_id, "derive_structure_links", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            raise
        logger.info("derive_structure_links: retrying %s (attempt %d/%d)", document_id, self.request.retries + 1, self.max_retries)
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True)
def collect_derivations(self, document_id: str, run_id: str | None = None) -> None:
    """Post-derivation checkpoint: mark document as past derivation stages."""
    try:
        logger.info("collect_derivations: document_id=%s", document_id)
        _update_document_status(document_id, STATUS_PROCESSING, stage="collect_derivations")
    except Exception as exc:
        logger.error("collect_derivations failed for %s: %s", document_id, exc)
        _update_document_status(
            document_id, STATUS_PARTIAL_COMPLETE,
            stage="collect_derivations", error=str(exc),
        )


@celery_app.task(bind=True, max_retries=1, default_retry_delay=30, queue="graph",
                 soft_time_limit=settings.finalize_soft_time_limit,
                 time_limit=settings.finalize_time_limit)
def derive_canonicalization(self, document_id: str, run_id: str | None = None) -> dict:
    """Post-extraction entity canonicalization pass.

    Resolves entity aliases to canonical names via GraphStore fulltext search
    and creates HAS_ALIAS edges for discovered matches.
    """
    from app.db.session import get_graph_store
    from app.services.canonicalization import canonicalize_document_entities

    self.max_retries = settings.finalize_max_retries
    self.default_retry_delay = settings.finalize_retry_delay
    self.soft_time_limit = settings.finalize_soft_time_limit
    self.time_limit = settings.finalize_time_limit

    logger.info("derive_canonicalization: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_canonicalization")

    db = _get_db()
    try:
        if not run_id:
            run_id = _get_pipeline_run_id(db, document_id)
        if run_id:
            _update_stage_run(db, run_id, "derive_canonicalization", "RUNNING", attempt=self.request.retries + 1)
            db.commit()

        graph_store = get_graph_store()
        graph_store.ensure_ready_sync()
        stats = canonicalize_document_entities(graph_store, document_id)

        if run_id:
            _update_stage_run(
                db, run_id, "derive_canonicalization", "COMPLETE",
                attempt=self.request.retries + 1,
                metrics=stats,
            )
            db.commit()

        logger.info(
            "derive_canonicalization: document_id=%s resolved=%d/%d",
            document_id, stats["resolved"], stats["total"],
        )
        return {"stage": "derive_canonicalization", "status": "ok", **stats}

    except CeleryRetry:
        raise
    except Exception as exc:
        logger.error("derive_canonicalization failed for %s: %s", document_id, exc)
        db.rollback()
        if self.request.retries >= self.max_retries:
            if run_id:
                _update_stage_run(db, run_id, "derive_canonicalization", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            raise
        logger.info("derive_canonicalization: retrying %s (attempt %d/%d)", document_id, self.request.retries + 1, self.max_retries)
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, soft_time_limit=settings.finalize_soft_time_limit,
                 time_limit=settings.finalize_time_limit)
def finalize_document(self, document_id: str, run_id: str | None = None) -> None:
    """Mark pipeline COMPLETE if all required stages succeeded."""
    from app.models.ingest import PipelineRun, StageRun
    from sqlalchemy import select, update as sql_update
    import datetime

    self.max_retries = settings.finalize_max_retries
    self.default_retry_delay = settings.finalize_retry_delay
    self.soft_time_limit = settings.finalize_soft_time_limit
    self.time_limit = settings.finalize_time_limit

    logger.info("finalize_document: document_id=%s run_id=%s", document_id, run_id)
    db = _get_db()
    try:
        if not run_id:
            run_id = _get_pipeline_run_id(db, document_id)
        if not run_id:
            _update_document_status(document_id, STATUS_COMPLETE, stage=None)
            return

        # Check for failed, missing, or stuck stages
        REQUIRED_STAGES = {
            "prepare_document",
            "detect_and_translate",
            "derive_document_metadata",
            "derive_picture_descriptions",
            "purge_document_derivations",
            "derive_text_embeddings",
            "derive_image_embeddings",
            "derive_ontology_graph",
            "derive_structure_links",
            "derive_canonicalization",
        }

        all_stages = db.execute(
            select(StageRun).where(StageRun.pipeline_run_id == uuid.UUID(run_id))
        ).scalars().all()
        stage_statuses = {s.stage_name: s.status for s in all_stages}

        failed = [n for n, s in stage_statuses.items() if s == "FAILED"]
        missing = REQUIRED_STAGES - set(stage_statuses.keys())
        stuck = [n for n, s in stage_statuses.items() if s in ("RUNNING", "PENDING")]

        if failed or missing or stuck:
            final_status = STATUS_PARTIAL_COMPLETE
            logger.warning(
                "finalize_document: document_id=%s failed=%s missing=%s stuck=%s",
                document_id, failed, list(missing), stuck,
            )
        else:
            # Check if any artifacts need human review
            from app.models.ingest import Artifact as _Artifact
            review_artifacts = db.execute(
                select(_Artifact).where(
                    _Artifact.document_id == uuid.UUID(document_id),
                    _Artifact.requires_human_review == True,  # noqa: E712
                )
            ).scalars().all()
            if review_artifacts:
                final_status = STATUS_PENDING_REVIEW
            else:
                final_status = STATUS_COMPLETE

        _update_document_status(document_id, final_status, stage=None)

        # Update PipelineRun
        db.execute(
            sql_update(PipelineRun)
            .where(PipelineRun.id == uuid.UUID(run_id))
            .values(
                status=final_status,
                finished_at=datetime.datetime.now(datetime.timezone.utc),
            )
        )
        db.commit()

        # Post-ingest community-detection trigger — increment counter and
        # fire an incremental run when the threshold is reached.
        if final_status == STATUS_COMPLETE:
            _maybe_trigger_post_ingest_community_detection(document_id)

        logger.info(
            "finalize_document: document_id=%s — pipeline %s",
            document_id, final_status,
        )
    except CeleryRetry:
        raise
    except Exception as exc:
        logger.error("finalize_document failed for %s: %s", document_id, exc)
        db.rollback()
        # Ensure PipelineRun doesn't get stuck in PROCESSING
        if run_id:
            _update_stage_run(db, run_id, "finalize_document", "FAILED", attempt=self.request.retries + 1, error=str(exc))
            db.commit()
        _update_document_status(document_id, STATUS_PARTIAL_COMPLETE, stage="finalize_document", error=str(exc))
    finally:
        db.close()


_POST_INGEST_COUNTER_KEY = "community:pending_ingest_count"


def _maybe_trigger_post_ingest_community_detection(document_id: str) -> None:
    """Increment the post-ingest counter and trigger detection at threshold.

    Controlled by ``community_detection_post_ingest_enabled`` and
    ``community_detection_post_ingest_threshold``. When the counter reaches
    the threshold, it is reset to zero and an incremental community detection
    task is dispatched. Errors are logged and swallowed — post-ingest detection
    is best-effort and must not fail document ingestion.

    Settings are fetched via ``get_settings()`` inline so tests can override
    them by clearing the cache and setting env vars.
    """
    try:
        _s = get_settings()
        if not _s.community_detection_post_ingest_enabled:
            return

        from app.services.redis_utils import get_redis as _get_redis
        r = _get_redis()
        try:
            count = int(r.incr(_POST_INGEST_COUNTER_KEY))
            threshold = _s.community_detection_post_ingest_threshold
            logger.info(
                "post-ingest community counter: document_id=%s count=%d threshold=%d",
                document_id, count, threshold,
            )
            if count >= threshold:
                r.set(_POST_INGEST_COUNTER_KEY, 0)
                from app.workers.community_tasks import run_community_detection_task
                run_community_detection_task.delay(mode="incremental")
                logger.info(
                    "post-ingest community detection dispatched (count reached %d)",
                    count,
                )
        finally:
            # Shared pool-owned client — do not close here.
            pass
    except Exception as exc:
        logger.warning(
            "post-ingest community detection trigger failed for %s: %s",
            document_id, exc,
        )


