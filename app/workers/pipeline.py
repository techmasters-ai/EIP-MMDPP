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
import json
import logging
import uuid
from typing import Any, Literal, Optional

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
from sqlalchemy import text  # noqa: E402 — used by dispatch-ledger helpers
from dataclasses import dataclass as _dataclass  # noqa: E402
from dataclasses import dataclass  # noqa: E402

from app.services.extraction_merge import classify_yield  # noqa: E402
from app.services.extraction_merge import classify_yield_from_counts  # noqa: E402
from app.services.extraction_merge import build_display_label  # noqa: E402
from app.services.extraction_merge import YieldStatus  # noqa: E402
from app.services.extraction_merge import merge_and_resolve  # noqa: E402
from app.services.extraction_merge import PreMergeWalkSummary  # noqa: E402
from app.services.extraction_merge import walk_entity_graph  # noqa: E402
from app.services.ontology_bundles import load_bundle_manifest  # noqa: E402
from app.services.ontology_templates import load_ontology  # noqa: E402
from app.db.session import get_graph_store  # noqa: E402
from app.db.session import get_sync_session  # noqa: E402
from ontology_bundles.air_defense_v3.validators import (  # noqa: E402
    canonicalize_identity_text,
)


# --- Custom exception types for the single-pass dispatcher (spec §5.5 + §6.5) ---

class PassRetryable(Exception):
    """Raised by _call_extract_pass for HTTP 5xx, partial response parse errors,
    TransientOllamaBusyError, and service-level pipeline_error stubs.
    _run_single_pass retries up to pass_max_retries with exponential backoff."""


class PassTransportError(PassRetryable):
    """Subclass for connection-level failures: ``httpx.TimeoutException``,
    ``httpx.TransportError`` (server disconnect mid-stream, connection refused,
    DNS failure). These are infra-level and should NOT consume the business
    ``pass_max_retries`` budget — ``_run_single_pass`` retries them up to
    ``pass_max_transport_retries`` without incrementing ``attempt``."""


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


@dataclass
class PassAttemptOutcome:
    """Rich result of one attempt at one pass.

    Returned by ``_execute_pass_attempt``; consumed by ``_run_single_pass``
    (in-process retry loop) and by Task 5's per-pass Celery task (Celery as
    retry boundary).  The caller decides retry/persistence — the helper is
    stateless.
    """
    execution_status: Literal["COMPLETE", "FAILED", "SKIPPED"]
    skip_reason: str | None        # "NO_UPSTREAM_ENDPOINTS" etc.; set iff SKIPPED
    yield_status: str | None       # "HIT"/"EMPTY"/"DEGRADED"/"BRIDGES_ONLY"; set iff COMPLETE
    pass_result: "PassResult | None"  # populated iff COMPLETE; forward-stringed to avoid eager import
    raw_response_payload: dict | None  # literal /extract-pass JSON; set when HTTP call succeeded
    counts: dict | None            # _count_pass_output result; set iff COMPLETE
    error: Exception | None        # PassRetryable/PassTransportError/PassTerminal; set iff FAILED


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


def _serialize_for_audit(
    merged,
    manifest,
    identity_to_rid: dict | None = None,
    element_uid_to_artifact_id: dict | None = None,
) -> dict:
    """Build the DocumentGraphExtraction.graph_json audit blob.

    Distinguishes primary vs bridge entities using the manifest's
    bridge_entity_types lists (a type is a bridge iff it appears in
    ANY pass's bridge_entity_types).

    Phase 8 Task 53: emits ``nodes[]``, ``mentions[]`` and the
    ``element_to_artifact`` map alongside the legacy count summaries.

      * ``nodes[]`` — one entry per MergedEntityRecord with
        ``{name, entity_type, entity_id, rid, artifact_ids}``.
        ``entity_id`` = ``record.identity.serialize_as_entity_id()``.
        ``rid`` comes from the ``identity_to_rid`` map (passed by the
        caller once phase-2 vertex upserts have resolved identities).
        ``artifact_ids`` derive from walking ``record.provenance``
        element_uids through ``element_uid_to_artifact_id``; deduped +
        sorted for determinism; empty list when no provenance resolves.

      * ``mentions[]`` — one entry per ``ExtractionProvenance`` across
        every merged record with ``{entity_name, entity_type,
        entity_id, rid, element_uid, page, chunk_index}``.
        ``derive_structure_links`` reads this to emit EXTRACTED_FROM
        chunk-link edges without a second entity_id → rid join.

      * ``element_to_artifact`` — the caller's map, persisted so
        ``derive_structure_links`` can consume it from the snapshot
        blob instead of re-querying Postgres. Snapshot-consistency:
        later reads see the state-at-ingestion, not the live DB.

    Backward-compatible default: when callers don't pass
    ``identity_to_rid`` / ``element_uid_to_artifact_id`` (legacy call
    sites, test fixtures), the new keys are still emitted but with
    empty / None entries — the blob shape stays consistent so
    downstream readers never hit KeyError.
    """
    identity_to_rid = identity_to_rid or {}
    element_uid_to_artifact_id = element_uid_to_artifact_id or {}

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

    rejection_reasons: dict[str, int] = {}
    for tup in merged.rejected_edges:
        reason = tup[2]
        reason_val = reason.value if hasattr(reason, "value") else str(reason)
        rejection_reasons[reason_val] = rejection_reasons.get(reason_val, 0) + 1

    # --- Phase 8 Task 53: nodes[] + mentions[] ---
    nodes: list[dict] = []
    mentions: list[dict] = []
    for record in merged.entities:
        entity_id = record.identity.serialize_as_entity_id()
        rid = identity_to_rid.get(record.identity)

        artifact_id_set: set[str] = set()
        for prov in record.provenance:
            aid = element_uid_to_artifact_id.get(prov.element_uid)
            if aid:
                artifact_id_set.add(aid)
        artifact_ids = sorted(artifact_id_set)

        nodes.append({
            "name": record.display_label,
            "entity_type": record.identity.entity_type,
            "entity_id": entity_id,
            "rid": rid,
            "artifact_ids": artifact_ids,
        })

        for prov in record.provenance:
            mentions.append({
                "entity_name": record.display_label,
                "entity_type": record.identity.entity_type,
                "entity_id": entity_id,
                "rid": rid,
                "element_uid": prov.element_uid,
                "page": prov.page,
                "chunk_index": prov.chunk_index,
                "instance_id": prov.instance_id,
            })

    audit_blob_size_hint = (
        len(nodes), len(mentions), len(element_uid_to_artifact_id),
    )
    logger.info(
        "audit_blob_size doc_id=%s nodes=%d mentions=%d element_to_artifact=%d",
        getattr(merged, "document_id", "?"),
        audit_blob_size_hint[0], audit_blob_size_hint[1], audit_blob_size_hint[2],
    )

    return {
        "primary_entities_total": primary_total,
        "bridge_entities_total": bridge_total,
        "entity_count_by_type": entity_count_by_type,
        "edges_accepted": edges_accepted,
        "edges_rejected": edges_rejected,
        "rejection_reasons": rejection_reasons,
        # Phase 8 additions:
        "nodes": nodes,
        "mentions": mentions,
        "element_to_artifact": dict(element_uid_to_artifact_id),
    }


def _build_element_uid_to_artifact_id(db, document_id: str) -> dict[str, str]:
    """Map every DoclingDocument element_uid to its owning artifact_id.

    DocumentElement is the only model that carries both. Used by
    _serialize_for_audit to derive ``artifact_ids`` on each node from
    its provenance element_uids, and by derive_structure_links as a
    fallback when the audit blob is pre-Phase-8 (legacy).

    Index: DocumentElement carries a UniqueConstraint on
    (document_id, element_uid) — the filter is a B-tree range scan.
    """
    from sqlalchemy import select
    from app.models.ingest import DocumentElement

    rows = db.execute(
        select(DocumentElement.element_uid, DocumentElement.artifact_id)
        .where(DocumentElement.document_id == uuid.UUID(document_id))
    ).all()
    return {uid: str(aid) for uid, aid in rows if uid and aid}


_DOMAIN_PASS_NAMES: frozenset[str] = frozenset({
    # radar (post-radar-cutover)
    "radar_identity", "radar_power_rf", "radar_antenna",
    "radar_timing", "radar_modulation",
    # missile (post-missile-cutover — 6 sub-passes replace missile_domain)
    "missile_identity", "missile_kinematics", "missile_guidance",
    "missile_airframe", "missile_speed_timing", "missile_propulsion",
    # system_links — preserved from radar cutover; do NOT drop
    "system_links",
})


def _classify_extraction_quality(
    pass_outcomes: dict,
    section_count: int,
    text_chunk_count: int,
) -> str:
    """Three-state extraction-quality aggregate (spec §6.8).

    Post-C-1/C-2 rewrite: the LLM `reference` pass was deleted and the
    deterministic Docling anchor walker (D-3/D-4) now emits SECTION
    vertices directly. "degraded" is therefore anchored on the *graph*
    signal (SECTION vertices + TextChunks exist) rather than a
    pass-level reference HIT.

    States:
      - ``"ok"``       — at least one domain pass (radar / missile /
                         other_systems / system_links) achieved HIT.
      - ``"degraded"`` — no domain HIT, but the document produced at
                         least one SECTION vertex AND at least one
                         TextChunk. Signals "real document processed
                         all the way through anchor derivation and
                         chunking, but nothing matched the SAM/radar
                         ontology".
      - ``"anomaly"``  — no domain HIT AND either no SECTION vertices
                         or no TextChunks. Processing broke somewhere
                         upstream of the ontology passes, or the
                         document is entirely unprocessable.
    """
    domain_hit = any(
        v.get("yield_status") == "HIT"
        for k, v in pass_outcomes.items()
        if k in _DOMAIN_PASS_NAMES
    )
    if domain_hit:
        return "ok"
    if section_count > 0 and text_chunk_count > 0:
        return "degraded"
    return "anomaly"


def _write_pipeline_run_metrics(pipeline_run_id, merged, manifest) -> None:
    """Populates PipelineRun.metrics with the quality-signal blob.

    Spec §6.6 + §6.8. Queries v_latest_pass_attempts for per-pass
    outcomes, fetches SECTION vertex count from the graph store and
    TextChunk count from Postgres for the new degraded/anomaly split,
    and computes overall_relationship_rejection_ratio + a rejection
    sample.
    """
    from sqlalchemy import func, select
    from app.models.ingest import PipelineRun
    from app.models.retrieval import TextChunk

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

        # Graph + text_chunk signals for the new degraded/anomaly split.
        # Fetch the run's document_id (required for document-scoped SECTION
        # lookup). Errors are swallowed because the classifier must not
        # block metrics writes; defaults treat the missing signal as zero,
        # which pushes borderline cases toward "anomaly" conservatively.
        document_uuid = str(getattr(run, "document_id", "") or "")
        section_count = 0
        text_chunk_count = 0
        if document_uuid:
            try:
                graph_store = get_graph_store()
                section_count = graph_store.count_ontology_nodes_sync(
                    "SECTION", document_id=document_uuid,
                )
            except Exception as exc:
                logger.debug(
                    "_write_pipeline_run_metrics: SECTION count failed for %s: %s",
                    document_uuid, exc,
                )
            try:
                text_chunk_count = session.execute(
                    select(func.count()).select_from(TextChunk).where(
                        TextChunk.document_id == uuid.UUID(document_uuid),
                    )
                ).scalar_one()
            except Exception as exc:
                logger.debug(
                    "_write_pipeline_run_metrics: TextChunk count failed for %s: %s",
                    document_uuid, exc,
                )

        run.metrics = {
            "pass_outcomes": pass_outcomes,
            "document_extraction_anomaly": document_extraction_anomaly,
            "pass_degraded_count": pass_degraded_count,
            "overall_relationship_rejection_ratio": overall_rejection_ratio,
            "rejected_relationships_sample": rejected_sample,
            "extraction_quality": _classify_extraction_quality(
                pass_outcomes, section_count, text_chunk_count,
            ),
            "section_count": section_count,
            "text_chunk_count": text_chunk_count,
            "bundle_legacy": False,
            "bundle_key_display": bundle_key or "",
        }
        session.commit()


def _build_pre_merge_walk_summary(
    pass_result,
    pass_def,
    ontology: dict,
    document_id: str,
) -> PreMergeWalkSummary:
    """Build the shared pre-merge carrier for one PassResult (plan Task 34b).

    Typed-edge passes: run walk_entity_graph once over the template_instance
    with both on_entity and on_edge callbacks. entities gets every emitted
    entity (nested children included); raw_edge_count counts every edge
    emission pre-validation.

    relationships_only passes (system_links, Decision 4 exception): entities=[];
    raw_edge_count=len(template_instance.relationships). The DTO-list length
    feeds classify_yield the same provisional-edge signal a typed-edge pass
    would get from walker emissions, so system_links pre-merge HIT/EMPTY
    classification matches entity-bearing passes.
    """
    if getattr(pass_def, "kind", None) == "relationships_only":
        relationships = getattr(pass_result.template_instance, "relationships", None) or []
        return PreMergeWalkSummary(entities=[], raw_edge_count=len(relationships))

    entities: list = []
    edge_count = 0

    def _on_edge(_parent_identity, _label, _child):
        nonlocal edge_count
        edge_count += 1

    walk_entity_graph(
        pass_result.template_instance,
        on_entity=entities.append,
        ontology=ontology,
        document_id=document_id,
        on_edge=_on_edge,
        visited_objects=set(),
        at_pass_root=True,
    )
    return PreMergeWalkSummary(entities=entities, raw_edge_count=edge_count)


def _execute_pass_attempt(
    *,
    pipeline_run_id,
    pass_def,
    manifest,
    ontology: dict,
    bundle_key: str,
    doc_json: dict,
    upstream_refs: dict,
    document_id: str,
) -> "PassAttemptOutcome":
    """One attempt at one pass.  Does NOT retry — the caller decides retry.
    Does NOT write StageRun or pipeline_pass_outputs — the caller persists.
    Returns rich metadata so the caller can decide what to do.

    r4: introduced to allow Task 5's per-pass Celery task to invoke this
    helper directly, with Celery as the retry boundary instead of the
    in-process ``while True`` loop in ``_run_single_pass``.

    Note: ``pipeline_run_id`` is currently accepted but unused inside this
    helper. It's reserved for Task 5's Celery task, which will use it to
    correlate StageRun and pipeline_pass_outputs writes after the helper
    returns.
    """
    # 1. Skip check
    if _should_skip(pass_def, upstream_refs, ontology):
        return PassAttemptOutcome(
            execution_status="SKIPPED",
            skip_reason="NO_UPSTREAM_ENDPOINTS",
            yield_status=None,
            pass_result=None,
            raw_response_payload=None,
            counts=None,
            error=None,
        )

    # 2. Compute selected_refs ONCE — reused for both the request body and the
    #    upstream-refs attachment below, eliminating any drift surface.
    selected_refs = (
        _select_upstream_refs_for_pass(pass_def, upstream_refs, ontology)
        if pass_def.input_mode == "document_plus_entity_refs"
        else None
    )

    # 3. Build request + call HTTP
    request_body = _build_extract_pass_request(
        bundle_key=bundle_key,
        pass_def=pass_def,
        doc_json=doc_json,
        upstream_refs=selected_refs,
        document_id=document_id,
    )
    try:
        raw_payload = _call_extract_pass(request_body, timeout=settings.docling_graph_timeout)
    except (PassRetryable, PassTransportError, PassTerminal) as exc:
        # PassTransportError is a subclass of PassRetryable, so the tuple catches all
        # three; the caller (_run_single_pass) uses order-dependent isinstance checks
        # to distinguish them.
        return PassAttemptOutcome(
            execution_status="FAILED",
            skip_reason=None,
            yield_status=None,
            pass_result=None,
            raw_response_payload=None,
            counts=None,
            error=exc,
        )

    # 4. Parse response
    try:
        pass_result = _parse_pass_response(raw_payload, pass_def, manifest)
    except PassTerminal as exc:
        return PassAttemptOutcome(
            execution_status="FAILED",
            skip_reason=None,
            yield_status=None,
            pass_result=None,
            raw_response_payload=raw_payload,  # captured — useful forensic data
            counts=None,
            error=exc,
        )

    # 5. Attach upstream refs as LogicalIdentity objects so merge_and_resolve
    #    can resolve from_ref_id / to_ref_id (extraction_merge.py:384).
    #    Only document_plus_entity_refs passes use this — document_only passes
    #    do not consume upstream refs.
    if pass_def.input_mode == "document_plus_entity_refs":
        from app.services.extraction_merge import logical_identity_from_dict
        # Reuse the same selection that built the request body above —
        # ensures the merge side sees exactly the refs the LLM was
        # told about, and removes a drift surface where future edits
        # could cause the two sites to disagree.
        selected = selected_refs or {}
        pass_result.upstream_refs = {}
        for ref_id, ref in selected.items():
            identity = logical_identity_from_dict(
                ref.entity_type,
                ref.identity_values or {},
                ontology,
                document_id,
            )
            if identity is not None:
                pass_result.upstream_refs[ref_id] = identity

    # 6. Compute pre_merge_walk + yield_status + counts
    # Plan Task 34b: build the single shared pre-merge carrier and
    # attach it to PassResult. classify_yield and _count_pass_output
    # consume pass_result.pre_merge_walk — the walker runs ONCE per
    # PassResult for the whole pre-merge phase.
    pass_result.pre_merge_walk = _build_pre_merge_walk_summary(
        pass_result, pass_def, ontology, document_id,
    )
    yield_status_val = classify_yield(pass_result, pass_def, ontology)
    # classify_yield returns a YieldStatus enum; normalise to string
    yield_str = (
        yield_status_val.value
        if hasattr(yield_status_val, "value")
        else str(yield_status_val)
    )
    counts = _count_pass_output(pass_result, pass_def, ontology)

    # 7. Return COMPLETE outcome
    return PassAttemptOutcome(
        execution_status="COMPLETE",
        skip_reason=None,
        yield_status=yield_str,
        pass_result=pass_result,
        raw_response_payload=raw_payload,
        counts=counts,
        error=None,
    )


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

    r4: now wraps _execute_pass_attempt for the per-attempt logic. The retry
    loop, StageRun writes, and pass_results population stay here. The helper
    can be invoked independently by Task 5's Celery task without going through
    this wrapper.
    """
    max_retries = getattr(settings, "pass_max_retries", 3)
    max_transport_retries = getattr(settings, "pass_max_transport_retries", 3)
    attempt = 1
    transport_attempt = 0

    while True:
        outcome = _execute_pass_attempt(
            pipeline_run_id=pipeline_run_id,
            pass_def=pass_def,
            manifest=manifest,
            ontology=ontology,
            bundle_key=bundle_key,
            doc_json=doc_json,
            upstream_refs=upstream_refs,
            document_id=document_id,
        )

        if outcome.execution_status == "SKIPPED":
            _write_stage_run(
                pipeline_run_id=pipeline_run_id,
                pass_def=pass_def,
                attempt=attempt,
                execution_status="SKIPPED",
                yield_status=None,
                skip_reason=outcome.skip_reason,
                counts=None,
                error=None,
            )
            return

        if outcome.execution_status == "FAILED":
            # Order-dependent: PassTransportError is a subclass of PassRetryable;
            # check the more specific class first so transport errors use their own
            # counter and do not burn the business-retry budget.
            if isinstance(outcome.error, PassTransportError):
                transport_attempt += 1
                _write_stage_run(
                    pipeline_run_id=pipeline_run_id,
                    pass_def=pass_def,
                    attempt=attempt,
                    execution_status="FAILED",
                    yield_status=None,
                    skip_reason=None,
                    counts=None,
                    error=f"[transport-retry {transport_attempt}/{max_transport_retries}] {outcome.error}",
                )
                if transport_attempt >= max_transport_retries:
                    if pass_def.required:
                        raise IngestFailed(
                            f"Required pass {pass_def.name} exhausted transport retries"
                        ) from outcome.error
                    return
                _backoff(transport_attempt)
                continue
            elif isinstance(outcome.error, PassRetryable):
                _write_stage_run(
                    pipeline_run_id=pipeline_run_id,
                    pass_def=pass_def,
                    attempt=attempt,
                    execution_status="FAILED",
                    yield_status=None,
                    skip_reason=None,
                    counts=None,
                    error=str(outcome.error),
                )
                if attempt >= max_retries:
                    if pass_def.required:
                        raise IngestFailed(
                            f"Required pass {pass_def.name} exhausted retries"
                        ) from outcome.error
                    return
                _backoff(attempt)
                attempt += 1
                continue
            elif isinstance(outcome.error, PassTerminal):
                _write_stage_run(
                    pipeline_run_id=pipeline_run_id,
                    pass_def=pass_def,
                    attempt=attempt,
                    execution_status="FAILED",
                    yield_status=None,
                    skip_reason=None,
                    counts=None,
                    error=str(outcome.error),
                )
                if pass_def.required:
                    raise IngestFailed(
                        f"Required pass {pass_def.name} terminal failure"
                    ) from outcome.error
                return
            else:
                # Defensive: _execute_pass_attempt should always set outcome.error for
                # FAILED outcomes. If it didn't, surface a diagnostic instead of raising
                # None (which would produce TypeError: exceptions must derive from BaseException).
                raise outcome.error or RuntimeError(
                    f"_execute_pass_attempt returned FAILED with error=None for pass {pass_def.name}"
                )

        # COMPLETE outcome — write StageRun and populate pass_results
        # Plan Task 36 pre-merge JSONB shape: all 5 authoritative-shape keys
        # are present on every write. counts_authoritative=False so readers
        # know the values are provisional; _apply_post_merge_yield_updates
        # overwrites all 5 keys + flips counts_authoritative=True post-merge.
        # Top-level StageRun columns (relationships_extracted / _rejected)
        # are mirrored into the JSONB block so the two projections never
        # drift — lockstep contract pinned by test_counts_authoritative_lifecycle.
        # Build the StageRun-bound counts dict on a shallow copy so we don't
        # inject "metrics" into outcome.counts (which Task 5's Celery caller
        # inspects directly without expecting that key).
        counts = dict(outcome.counts)
        counts["metrics"] = {
            "counts_authoritative": False,
            "relationships_extracted": counts["relationships_extracted"],
            "relationships_rejected": counts["relationships_rejected"],
            "rejection_sample": [],
            "rejections_by_reason": _build_rejections_by_reason(
                getattr(outcome.pass_result, "pre_merge_rejections", None),
            ),
        }
        _write_stage_run(
            pipeline_run_id=pipeline_run_id,
            pass_def=pass_def,
            attempt=attempt,
            execution_status="COMPLETE",
            yield_status=outcome.yield_status,
            skip_reason=None,
            counts=counts,
            error=None,
        )
        pass_results[pass_def.name] = outcome.pass_result

        if _any_downstream_pass_depends_on(manifest, pass_def.name):
            _extend_upstream_refs(upstream_refs, outcome.pass_result, pass_def, ontology)
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


def _apply_post_merge_yield_updates(pipeline_run_id, merged, manifest) -> None:
    """Authoritative per-pass yield + metrics update after merge (plan Task 36).

    Reads ``merged.per_pass_edge_metrics[pass_name]`` (populated uniformly
    for typed-edge AND system_links passes by ``merge_and_resolve``) and
    writes the full post-merge picture in lockstep across BOTH the
    ``StageRun`` top-level columns and the ``StageRun.metrics`` JSONB:
    ``relationships_extracted``, ``relationships_rejected``,
    ``counts_authoritative=True``, ``rejection_sample``, and
    ``rejections_by_reason``. An XOR of the two surfaces is a regression.

    Yield dispatch by ``manifest.find_pass(pass_name).kind``:
    - ``relationships_only`` (e.g. system_links): ``yield_status``
      overwritten unconditionally via ``classify_yield_from_counts(primary=0,
      bridge=0, extracted_rels=accepted, rejected_rels=rejected)``.
      Promotes HIT→EMPTY (0 accepted, <4 total) and HIT→DEGRADED (≥4 total
      with ≥75% rejected) without hardcoded values.
    - Otherwise (entity-bearing passes): existing HIT → DEGRADED rule
      (guarded on ``yield_status == "HIT"``) preserved; EMPTY/BRIDGES_ONLY
      are never promoted upward.

    Fallback when ``merged.per_pass_edge_metrics`` is not yet populated
    (interim builds / test fixtures): derive ``accepted``/``rejected``
    from ``merged.edges``/``merged.rejected_edges`` grouped by pass name,
    ``rejection_sample=[]`` (populated only when the carrier is present),
    ``rejections_by_reason`` via ``_build_rejections_by_reason``.
    """
    from app.models.ingest import StageRun

    # Fallback path (carrier empty) — per-pass counts + rejections from the
    # edge/rejected_edges lists. Typed-edge and system_links alike.
    accepted_fallback: dict[str, int] = {}
    for edge in merged.edges:
        for pass_name in edge.pass_origins:
            accepted_fallback[pass_name] = accepted_fallback.get(pass_name, 0) + 1
    rejected_fallback: dict[str, int] = {}
    rejections_by_pass_fallback: dict[str, list] = {}
    for tup in merged.rejected_edges:
        pass_name = tup[0]
        rejected_fallback[pass_name] = rejected_fallback.get(pass_name, 0) + 1
        rejections_by_pass_fallback.setdefault(pass_name, []).append(tup)

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
            metrics_entry = merged.per_pass_edge_metrics.get(pass_name)
            if metrics_entry is not None:
                accepted = metrics_entry.accepted
                rejected = metrics_entry.rejected
                rejection_sample = list(metrics_entry.rejection_sample)
                rejections_by_reason_post = dict(metrics_entry.rejections_by_reason)
            else:
                accepted = accepted_fallback.get(pass_name, 0)
                rejected = rejected_fallback.get(pass_name, 0)
                rejection_sample = []
                rejections_by_reason_post = _build_rejections_by_reason(
                    rejections_by_pass_fallback.get(pass_name, []),
                )

            # Lockstep update: top-level columns and JSONB metrics move together.
            row.relationships_extracted = accepted
            row.relationships_rejected = rejected

            # Yield-status authority — dispatch by manifest pass kind.
            try:
                pass_def = manifest.find_pass(pass_name)
                pass_kind = pass_def.kind
            except (KeyError, AttributeError):
                pass_kind = None

            if pass_kind == "relationships_only":
                # system_links et al: overwrite unconditionally from the
                # canonical classifier so EMPTY/DEGRADED/HIT transitions
                # match pre-merge semantics across the DTO/typed-edge split.
                authoritative = classify_yield_from_counts(
                    primary=0,
                    bridge=0,
                    extracted_rels=accepted,
                    rejected_rels=rejected,
                )
                row.yield_status = authoritative.value
            else:
                # Entity-bearing passes: existing HIT→DEGRADED rule only.
                if row.yield_status == "HIT":
                    new_yield = classify_yield_from_counts(
                        primary=row.primary_entities_extracted or 0,
                        bridge=row.bridge_entities_extracted or 0,
                        extracted_rels=accepted,
                        rejected_rels=rejected,
                    )
                    if new_yield == YieldStatus.DEGRADED:
                        row.yield_status = "DEGRADED"

            # JSONB: overwrite all 5 authoritative-shape keys. Flip
            # counts_authoritative=True so readers can filter for post-merge.
            merged_metrics = dict(row.metrics or {})
            merged_metrics["counts_authoritative"] = True
            merged_metrics["relationships_extracted"] = accepted
            merged_metrics["relationships_rejected"] = rejected
            merged_metrics["rejection_sample"] = rejection_sample
            existing_reasons = dict(merged_metrics.get("rejections_by_reason") or {})
            existing_reasons.update(rejections_by_reason_post)  # post-merge wins
            merged_metrics["rejections_by_reason"] = existing_reasons
            row.metrics = merged_metrics

        session.commit()


def _build_provenance_envelope(
    document_id: str,
    pipeline_run_id: str | None,
    entities,
    db_session,
) -> "ProvenanceMetadata":  # noqa: F821 — forward ref resolved at call time
    """Assemble a fully-populated ProvenanceMetadata for a graph-write batch.

    Unions ``page`` values across every ``ExtractionProvenance`` row hanging
    off the batch's MergedEntityRecord instances, and fetches Document
    ``created_at`` + ``document_metadata['date_of_information']`` once so the
    envelope's ``upload_datetime`` + ``document_datetime`` land on the
    written records instead of staying null. See arcadedb_schema.py for the
    columns these feed.
    """
    from app.models.ingest import Document
    from app.services.graph_store import ProvenanceMetadata

    pages = sorted({
        p.page
        for rec in entities
        for p in (rec.provenance or [])
        if p.page is not None
    })

    upload_dt: str | None = None
    document_dt: str | None = None
    try:
        doc_uuid = uuid.UUID(str(document_id))
    except ValueError:
        doc_uuid = None
    if doc_uuid is not None:
        doc_row = db_session.query(Document).filter(Document.id == doc_uuid).first()
        if doc_row is not None:
            if doc_row.created_at is not None:
                upload_dt = doc_row.created_at.isoformat()
            meta = doc_row.document_metadata or {}
            if isinstance(meta, dict):
                raw_dt = meta.get("date_of_information")
                if isinstance(raw_dt, str) and raw_dt and raw_dt.lower() != "unknown":
                    document_dt = raw_dt

    return ProvenanceMetadata(
        document_id=str(document_id),
        page_numbers=pages,
        upload_datetime=upload_dt,
        document_datetime=document_dt,
        pipeline_run_id=pipeline_run_id,
    )


def _import_graph_phase_nodes(merged, ontology, document_id, tracker, provenance):
    """Spec §5.6 phase 2 — node upsert.

    Builds the full NodeRecord list in pure Python FIRST so that any
    pre-mutation failure (e.g. build_display_label raising) leaves
    tracker.any_mutation_attempted == False and the rollback gate
    correctly skips.  tracker.mark() is called AFTER the list is built
    and IMMEDIATELY before the first graph_store mutation.

    Task 4.4.
    """
    from app.services.graph_store import NodeRecord

    # Build all records in pure Python first. If this raises, tracker
    # stays False and the rollback gate correctly skips.
    def _build_node_record(e):
        # Phase 3 task 33: persist per-field evidence on the entity vertex.
        # Attach as _field_evidence (JSON-serializable) when populated.
        props = dict(e.properties)
        if getattr(e, "field_evidence", None):
            props["_field_evidence"] = {
                field_name: [
                    {
                        "chunk_id": row.chunk_id,
                        "snippet": row.snippet,
                        "element_uid": row.element_uid,
                        "value": row.value,
                        # Fix B: surface evidence metadata added in Task 12.5.
                        "evidence_id": row.evidence_id,
                        "page": row.page,
                        "document_id": row.document_id,
                    }
                    for row in rows
                ]
                for field_name, rows in e.field_evidence.items()
            }
        # Flat per-entity provenance (Task 12.5 wire shape).
        # Aggregate evidence_ids/page_numbers/evidence_text across all
        # ExtractionProvenance rows for this entity so the entity vertex
        # carries the data inline — no second-hop traversal required.
        provenance_rows = getattr(e, "provenance", None) or []
        if provenance_rows:
            agg_evidence_ids: list[str] = []
            agg_page_numbers: list[int] = []
            agg_evidence_texts: list[str] = []
            seen_eids: set[str] = set()
            seen_pages: set[int] = set()
            for prov_row in provenance_rows:
                for eid in getattr(prov_row, "evidence_ids", None) or []:
                    if eid not in seen_eids:
                        seen_eids.add(eid)
                        agg_evidence_ids.append(eid)
                for pg in getattr(prov_row, "page_numbers", None) or []:
                    if pg not in seen_pages:
                        seen_pages.add(pg)
                        agg_page_numbers.append(pg)
                ev_text = getattr(prov_row, "evidence_text", None)
                if ev_text:
                    agg_evidence_texts.append(ev_text)
            if agg_evidence_ids:
                props["_evidence_ids"] = agg_evidence_ids
            if agg_page_numbers:
                props["_page_numbers"] = sorted(agg_page_numbers)
            if agg_evidence_texts:
                # Store first non-empty evidence text as representative snippet.
                props["_evidence_text"] = agg_evidence_texts[0]
        return NodeRecord(
            entity_type=e.identity.entity_type,
            identity_fields=e.identity.as_upsert_identity_dict(),
            name=build_display_label(
                e.identity.entity_type,
                e.identity.identity_values_dict(),
                e.properties,
            ),
            properties=props,
            extraction_confidence=e.confidence,
        )

    node_records = [_build_node_record(e) for e in merged.entities]

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


def _instance_to_identity_map(
    entity_provenance_rows,
) -> "dict[str, Any]":
    """Build {instance_id: LogicalIdentity} from ExtractionProvenance rows.

    ExtractionProvenance carries BOTH ``instance_id`` (the per-instance UUID
    used in ExtractionRelationshipProvenance.source/target_instance_id) AND
    ``ontology_name`` + ``identity_values`` (the components that define a
    LogicalIdentity).  This map lets the relationship-provenance join step
    convert per-instance UUIDs into LogicalIdentity keys that match
    MergedEdgeRecord.from_identity / to_identity.

    Rows where ``instance_id`` or ``ontology_name`` is absent are silently
    skipped — they cannot be matched and will fall back to the rel_type-only
    bucket in the caller.  Rows where LogicalIdentity construction fails
    (e.g. identity_values not a dict) are also skipped.
    """
    from app.services.extraction_merge import LogicalIdentity

    out: dict[str, Any] = {}
    for p in entity_provenance_rows or []:
        instance_id = getattr(p, "instance_id", None)
        ontology_name = getattr(p, "ontology_name", None)
        identity_values = getattr(p, "identity_values", None) or {}
        if not instance_id or not ontology_name:
            continue
        if not isinstance(identity_values, dict):
            continue
        try:
            # LogicalIdentity is frozen; build from identity_values dict.
            # We don't have the canonical field order from the ontology here,
            # so we sort keys to get a stable (deterministic) ordering.
            # The resulting identity compares equal to MergedEdgeRecord
            # identities built from the same values, because
            # merge_and_resolve sorts keys the same way when it lacks an
            # explicit ordering override.
            sorted_items = sorted(identity_values.items())
            identity = LogicalIdentity(
                entity_type=str(ontology_name),
                identity_field_names=tuple(str(k) for k, _ in sorted_items),
                identity_tuple=tuple(v for _, v in sorted_items),
                scope="global",      # provenance rows don't carry scope;
                document_id=None,    # default to global/no-doc for matching.
            )
            out[str(instance_id)] = identity
        except Exception:
            continue
    return out


def _import_graph_phase_domain_edges(
    merged, ontology, tracker, provenance,
    relationship_provenance_rows=None,
    entity_provenance_rows=None,
) -> None:
    """Spec §5.6 phase 3 — domain edge upsert (identity-based).

    Builds RelationshipRecord list in pure Python, calls tracker.mark()
    defensively (idempotent — phase 2 likely already marked), then
    upserts.  An empty edges list still calls upsert_relationships_batch_sync
    with an empty list to match graph_store semantics.

    Fix A (code-review): relationship_provenance_rows (collected from all
    rehydrated pass results) is plumbed here so RelationshipRecord.provenance
    can be populated.

    Composite-key matching: ``entity_provenance_rows`` (ExtractionProvenance
    list from the same passes) is used to build an instance_id →
    LogicalIdentity map via ``_instance_to_identity_map``.  Each
    relationship_provenance row's source/target instance_ids are resolved to
    LogicalIdentity values that can be matched directly against
    MergedEdgeRecord.from_identity / to_identity, giving a precise
    (from_identity, rel_type, to_identity) composite key.

    Rows whose source or target instance_id cannot be resolved (e.g. entity
    provenance was not provided, or the instance was not retained after
    merge) fall back to a ``__rel_type_fallback__`` bucket keyed by
    (sentinel, rel_type).  MergedEdgeRecord edges that find no composite
    match also try this fallback, so behavior degrades gracefully rather
    than silently dropping provenance.

    Task 4.4.
    """
    from app.services.graph_store import RelationshipRecord

    # Build instance_id → LogicalIdentity map from entity provenance rows.
    id_to_identity = _instance_to_identity_map(entity_provenance_rows)

    # Sentinel used as the "from_identity" slot in the fallback bucket key.
    _FALLBACK = "__rel_type_fallback__"

    # Build provenance buckets keyed by composite (from_identity, rel_type,
    # to_identity) where resolvable, or (_FALLBACK, rel_type) otherwise.
    provenance_by_triple: dict[tuple, dict] = {}

    for row in (relationship_provenance_rows or []):
        rt = getattr(row, "relationship_type", None)
        if not rt:
            continue
        src_id_str = str(row.source_instance_id) if getattr(row, "source_instance_id", None) else ""
        tgt_id_str = str(row.target_instance_id) if getattr(row, "target_instance_id", None) else ""
        src_identity = id_to_identity.get(src_id_str)
        tgt_identity = id_to_identity.get(tgt_id_str)

        if src_identity is not None and tgt_identity is not None:
            key: tuple = (src_identity, rt, tgt_identity)
        else:
            key = (_FALLBACK, rt)

        bucket = provenance_by_triple.setdefault(key, {
            "evidence_ids": [],
            "self_refs": [],
            "page_numbers": [],
        })
        bucket["evidence_ids"] = sorted(set(
            bucket["evidence_ids"] + list(getattr(row, "evidence_ids", []) or [])
        ))
        bucket["self_refs"] = sorted(set(
            bucket["self_refs"] + list(getattr(row, "self_refs", []) or [])
        ))
        bucket["page_numbers"] = sorted(set(
            bucket["page_numbers"] + list(getattr(row, "page_numbers", []) or [])
        ))

    rel_records = []
    for e in merged.edges:
        triple_key = (e.from_identity, e.rel_type, e.to_identity)
        fallback_key = (_FALLBACK, e.rel_type)
        rel_prov = (
            provenance_by_triple.get(triple_key)
            or provenance_by_triple.get(fallback_key)
        )
        rel_records.append(RelationshipRecord(
            from_type=e.from_identity.entity_type,
            from_identity=e.from_identity.as_upsert_identity_dict(),
            to_type=e.to_identity.entity_type,
            to_identity=e.to_identity.as_upsert_identity_dict(),
            rel_type=e.rel_type,
            extraction_confidence=e.confidence,
            provenance=rel_prov or None,
        ))

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


# Terminal statuses the guard must NOT overwrite. String literals (not the
# STATUS_* constants) because those are defined later in the module.
# PENDING_HUMAN_REVIEW is terminal-ish — the doc is awaiting operator action;
# guard must not downgrade it to PARTIAL_COMPLETE.
_TERMINAL_DOC_STATUSES = {
    "COMPLETE",
    "FAILED",
    "PARTIAL_COMPLETE",
    "PENDING_HUMAN_REVIEW",
}


def _terminalize_doc_and_run(document_id: str, run_id: str | None, doc_status: str) -> None:
    """Flip a document and its owning PipelineRun to terminal states.

    Preserves existing terminal document statuses — if the doc is already
    FAILED / COMPLETE / PARTIAL_COMPLETE / PENDING_HUMAN_REVIEW, do NOT
    overwrite with a softer value.

    The pipeline_run always moves to FAILED when it was PROCESSING, regardless
    of the doc_status argument (a PARTIAL_COMPLETE document still corresponds
    to a FAILED run — the chain didn't reach finalize_document).

    All failure paths are swallowed — the caller is usually the guard handling
    an already-failing task; we don't want the terminalization itself to mask
    the original exception.
    """
    from datetime import datetime as dt
    from app.models.ingest import Document, PipelineRun

    try:
        db = _get_db()
    except Exception:
        logger.exception(
            "_terminalize_doc_and_run: failed to open DB session for document=%s",
            document_id,
        )
        return

    try:
        try:
            doc = db.get(Document, uuid.UUID(str(document_id)))
            if doc is not None and doc.pipeline_status not in _TERMINAL_DOC_STATUSES:
                doc.pipeline_status = doc_status

            if run_id:
                run = db.get(PipelineRun, uuid.UUID(str(run_id)))
                if run is not None and run.status == "PROCESSING":
                    run.status = "FAILED"
                    if run.finished_at is None:
                        run.finished_at = dt.utcnow()

            db.commit()
        except Exception:
            logger.exception(
                "_terminalize_doc_and_run: failed for document=%s run_id=%s",
                document_id, run_id,
            )
            try:
                db.rollback()
            except Exception:
                logger.exception("_terminalize_doc_and_run: rollback also failed")
    finally:
        try:
            db.close()
        except Exception:
            pass


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
                # EMPTY_ANCHOR_SET: synthesized by derive_ontology_graph when
                # the prior anchors stage produced 0 text_blocks/tables/figures.
                # Authorized so docs with no extractable graph content (image-
                # only / picture-description-only) can finalize as COMPLETE
                # rather than falling out the FAILED branch of this gate.
                if latest.skip_reason in {"NO_UPSTREAM_ENDPOINTS", "EMPTY_ANCHOR_SET"}:
                    continue
                failures.append(
                    (pass_name, f"unauthorized skip: {latest.skip_reason}")
                )
                continue
    finally:
        db.close()

    return GateResult(passed=(not failures), failures=failures)


def _build_legacy_docling_document_json(document_id: str, text: str) -> dict:
    """Construct a minimal schema-valid DoclingDocument dict for legacy fallback.

    Triggered for non-Docling-supported mimes (text/plain, etc.) in
    prepare_document. Downstream stages like derive_document_anchors call
    DoclingDocument.model_validate(...) on the stored JSON artifact, so a
    hand-crafted dict fails validation. This uses the DoclingDocument API
    to guarantee schema validity.

    Empty text is OK — the stub just omits the text item.
    """
    from docling_core.types.doc import DoclingDocument, DocItemLabel

    doc = DoclingDocument(name=str(document_id))
    if text:
        doc.add_text(label=DocItemLabel.TEXT, text=text)
    return doc.export_to_dict()


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

    Callers must have previously invoked ``_ensure_structural_document_vertex``
    (derive_ontology_graph does this just before phase 4). A missing vertex at
    this point is a worker-invariant violation, not a pass failure.
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


def _ensure_structural_document_vertex(document_id: str) -> str:
    """Idempotently upsert the structural Document vertex and return its @rid.

    Why: derive_ontology_graph's phase-4 structural edges (MENTIONED_IN, etc.)
    reference a Document vertex, but the full metadata-rich upsert in
    ``derive_structure_links`` runs LATER in the chain. Calling this earlier
    guarantees the vertex exists. ``derive_structure_links`` performs the same
    upsert with richer properties; both calls merge via identity on
    ``document_id``.
    """
    from app.models.ingest import Document as _Document
    from app.services.graph_store import NodeRecord as _NR

    db = _get_db()
    try:
        doc = db.get(_Document, uuid.UUID(document_id))
        filename = doc.filename if doc else document_id
        source_id = str(doc.source_id) if doc and doc.source_id else None
    finally:
        db.close()

    graph_store = get_graph_store()
    graph_store.ensure_ready_sync()
    props: dict[str, Any] = {"title": filename}
    if source_id:
        props["source_id"] = source_id
    return graph_store.upsert_node_sync(_NR(
        entity_type="Document",
        identity_fields={"document_id": document_id},
        name=filename,
        properties=props,
    ))


def _upsert_document_graph_extraction(
    *,
    document_id: str,
    pipeline_run_id: str,
    run,
    merged,
    manifest,
    identity_to_rid: dict | None = None,
    element_uid_to_artifact_id: dict | None = None,
) -> None:
    """Writes the DocumentGraphExtraction snapshot row per spec §5.7.

    graph_json carries the audit blob (per Phase 8 Task 53: counts,
    rejection reasons, ``nodes[]`` with entity_id/rid/artifact_ids,
    ``mentions[]`` with element_uid+entity_id, ``element_to_artifact``
    map). ``identity_to_rid`` + ``element_uid_to_artifact_id`` are
    passed from the caller — the serializer uses them to stamp rid +
    artifact_ids on each node/mention. Default None keeps older test
    call sites working (they get empty-rid/empty-artifact-id blobs).

    Upserts: queries by document_id; inserts a new row if none exists,
    otherwise updates the existing row in place.
    """
    import datetime
    from app.models.ingest import DocumentGraphExtraction

    audit_blob = _serialize_for_audit(
        merged, manifest, identity_to_rid, element_uid_to_artifact_id,
    )

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


def _sweep_stale_runs() -> int:
    """Sweep stale RUNNING stage_runs and auto-restart their documents.

    For each stage_run at status='RUNNING' older than
    settings.stale_stage_run_threshold_seconds:

      1. Mark the stage_run FAILED + pipeline_run FAILED + bump retry_count.
         **Commit these writes before calling start_ingest_pipeline** — the
         dispatch guard in start_ingest_pipeline runs in its own DB session
         and won't see uncommitted changes.
      2. If new retry_count <= settings.max_doc_retry_count, call
         start_ingest_pipeline(doc_id). On success it creates a fresh
         pipeline_run + chain in its own transaction. On exception we run
         a compensating transaction: revert retry_count, mark doc FAILED.
      3. If new retry_count > cap, the initial transaction sets
         pipeline_status='FAILED' and we do NOT dispatch.

    The failure → dispatch handoff is split across two transactions by
    design; required for the dispatch guard to see the FAILED row.

    Returns the number of stage_runs swept.
    """
    from sqlalchemy import text

    threshold = settings.stale_stage_run_threshold_seconds
    max_retry = settings.max_doc_retry_count

    db = _get_db()
    try:
        # ── stale DISPATCHED reset (ledger v1, spec 2026-05-10) ──────────
        # The dispatcher published a Celery task but a worker did not pick it up
        # within stale_dispatched_threshold_seconds. Reset to PENDING so the next
        # tick republishes. dispatch_attempt is unchanged — the stage didn't run.
        db.execute(
            text(
                """
                UPDATE ingest.stage_runs
                SET status        = 'PENDING',
                    dispatched_at = NULL,
                    error_message = COALESCE(error_message, '')
                                    || ' stale; reset by dispatcher sweeper'
                WHERE status        = 'DISPATCHED'
                  AND pass_name     IS NULL
                  AND task_name     IS NOT NULL
                  AND dispatched_at < NOW() - make_interval(secs => :dispatched_threshold)
                """
            ),
            {"dispatched_threshold": settings.stale_dispatched_threshold_seconds},
        )
        db.commit()

        # ── ledger stale-RUNNING (spec 2026-05-10) ──────────────────────────
        # Sequential stages 1–8 (LEDGER_SEQUENTIAL_STAGES): under cap → reset
        # to PENDING with attempt bump; at/over cap → terminalize stage_run +
        # pipeline_run. Stage 9 (derive_ontology_graph) is intentionally
        # excluded — `reconcile_ontology_graph_runs` is its sole owner during
        # fan-in and its summary row legitimately stays RUNNING.
        db.execute(
            text(
                """
                WITH stale AS (
                    SELECT sr.id, sr.pipeline_run_id,
                           sr.dispatch_attempt + 1 AS next_attempt
                    FROM ingest.stage_runs sr
                    JOIN ingest.pipeline_runs pr ON pr.id = sr.pipeline_run_id
                    WHERE sr.status     = 'RUNNING'
                      AND sr.pass_name  IS NULL
                      AND sr.stage_name = ANY(:ledger_sequential_stages)
                      AND sr.started_at < NOW() - make_interval(secs => :threshold)
                      AND pr.status     = 'PROCESSING'
                ),
                retryable AS (
                    UPDATE ingest.stage_runs sr
                    SET status           = 'PENDING',
                        dispatch_attempt = s.next_attempt,
                        started_at       = NULL,
                        dispatched_at    = NULL,
                        available_at     = NOW(),
                        error_message    = COALESCE(sr.error_message, '')
                                           || ' stale; reset by sweeper'
                    FROM stale s
                    WHERE sr.id = s.id AND s.next_attempt <= :max_dispatches
                    RETURNING sr.id
                ),
                terminal AS (
                    UPDATE ingest.stage_runs sr
                    SET status           = 'FAILED',
                        finished_at      = NOW(),
                        dispatch_attempt = s.next_attempt,
                        error_message    = COALESCE(sr.error_message, '')
                                           || ' stale; max dispatches reached'
                    FROM stale s
                    WHERE sr.id = s.id AND s.next_attempt > :max_dispatches
                    RETURNING sr.pipeline_run_id
                )
                UPDATE ingest.pipeline_runs pr
                SET status        = 'FAILED',
                    finished_at   = NOW(),
                    error_message = COALESCE(pr.error_message, '')
                                    || ' stage exceeded max dispatches'
                FROM terminal t
                WHERE pr.id = t.pipeline_run_id AND pr.status = 'PROCESSING'
                """
            ),
            {
                "ledger_sequential_stages": LEDGER_SEQUENTIAL_STAGES,
                "threshold": threshold,
                "max_dispatches": settings.max_stage_dispatches,
            },
        )
        db.commit()

        # Legacy stale-RUNNING handler. Excludes every ledger-owned stage:
        #   • LEDGER_SEQUENTIAL_STAGES (1–8) — owned by the CTE above.
        #   • LEDGER_FANOUT_STAGES (stage 9, derive_ontology_graph) — owned
        #     by reconcile_ontology_graph_runs during per-pass fan-in.
        stale_rows = db.execute(
            text(
                """
                SELECT sr.id, sr.pipeline_run_id, pr.document_id, sr.stage_name
                FROM ingest.stage_runs sr
                JOIN ingest.pipeline_runs pr ON pr.id = sr.pipeline_run_id
                WHERE sr.status = 'RUNNING'
                  AND sr.started_at < NOW() - make_interval(secs => :threshold)
                  AND pr.status = 'PROCESSING'
                  AND sr.stage_name <> ALL(:ledger_excluded_stages)
                """
            ),
            {
                "threshold": threshold,
                "ledger_excluded_stages": (
                    LEDGER_SEQUENTIAL_STAGES + LEDGER_FANOUT_STAGES
                ),
            },
        ).fetchall()

        if not stale_rows:
            return 0

        to_dispatch: list[tuple] = []  # (document_id, stage_name, new_retry_count)

        for stage_run_id, pipeline_run_id, document_id, stage_name in stale_rows:
            # 1a. Mark stage_run FAILED
            db.execute(
                text(
                    """
                    UPDATE ingest.stage_runs
                    SET status = 'FAILED',
                        finished_at = NOW(),
                        error_message = COALESCE(error_message, '') || 'stale; swept by periodic_stale_run_sweep'
                    WHERE id = :id
                    """
                ),
                {"id": stage_run_id},
            )

            # 1b. Atomically flip pipeline_run PROCESSING -> FAILED.
            pr_update = db.execute(
                text(
                    """
                    UPDATE ingest.pipeline_runs
                    SET status = 'FAILED',
                        finished_at = COALESCE(finished_at, NOW()),
                        error_message = COALESCE(error_message, '') || 'stale; swept by periodic_stale_run_sweep'
                    WHERE id = :id AND status = 'PROCESSING'
                    """
                ),
                {"id": pipeline_run_id},
            )
            if pr_update.rowcount == 0:
                # Already FAILED from a prior sweep; don't double-dispatch.
                continue

            # 1c. Bump retry_count atomically.
            bump = db.execute(
                text(
                    """
                    UPDATE ingest.documents
                    SET retry_count = retry_count + 1
                    WHERE id = :doc_id
                    RETURNING retry_count
                    """
                ),
                {"doc_id": document_id},
            ).scalar()

            if bump is None:
                logger.warning(
                    "sweeper: document %s disappeared before retry bump; skipping",
                    document_id,
                )
                continue

            # 1d. If over cap, mark document permanently FAILED in this same tx.
            #     failed_stages is ARRAY(String) — see app/models/ingest.py:67.
            if bump > max_retry:
                db.execute(
                    text(
                        """
                        UPDATE ingest.documents
                        SET pipeline_status = 'FAILED',
                            pipeline_stage = :stage,
                            failed_stages =
                                COALESCE(failed_stages, ARRAY[]::text[])
                                || ARRAY[:stage]::text[]
                        WHERE id = :doc_id
                        """
                    ),
                    {"doc_id": document_id, "stage": stage_name},
                )
                logger.error(
                    "sweeper: document=%s exhausted retries (%d > %d) — permanently FAILED",
                    document_id, bump, max_retry,
                )
            else:
                # Under cap: defer dispatch until after this tx commits.
                to_dispatch.append((document_id, stage_name, bump))

        # Commit failure bookkeeping. After this, start_ingest_pipeline can see
        # pipeline_run.status='FAILED' from its own session.
        db.commit()

        swept = len(stale_rows)

        for document_id, stage_name, bump in to_dispatch:
            try:
                start_ingest_pipeline(str(document_id))
                logger.warning(
                    "sweeper: redispatched document=%s stage_failed=%s retry=%d/%d",
                    document_id, stage_name, bump, max_retry,
                )
            except Exception:
                # Compensating transaction in a fresh session so the failed
                # dispatch doesn't leave a doc with bumped retry_count + no
                # running chain (invisible to next sweep).
                logger.exception(
                    "sweeper: redispatch failed for document=%s; "
                    "marking FAILED and reverting retry_count",
                    document_id,
                )
                comp = _get_db()
                try:
                    comp.execute(
                        text(
                            """
                            UPDATE ingest.documents
                            SET retry_count = GREATEST(retry_count - 1, 0),
                                pipeline_status = 'FAILED',
                                pipeline_stage = :stage,
                                failed_stages =
                                    COALESCE(failed_stages, ARRAY[]::text[])
                                    || ARRAY[:stage]::text[]
                            WHERE id = :doc_id
                            """
                        ),
                        {"doc_id": document_id, "stage": stage_name},
                    )
                    comp.commit()
                except Exception:
                    logger.exception(
                        "sweeper: compensation write also failed for %s; operator must triage",
                        document_id,
                    )
                    comp.rollback()
                finally:
                    comp.close()

        return swept
    except Exception:
        logger.exception("_sweep_stale_runs: rollback due to error")
        db.rollback()
        return 0
    finally:
        db.close()


@celery_app.task(bind=True)
def periodic_stale_run_sweep(self) -> int:
    """Beat-scheduled wrapper around _sweep_stale_runs.

    Scheduled in `app/workers/celery_app.py::beat_schedule`. Runs on any worker
    that consumes the `celery` queue. Safe to run concurrently — the UPDATE
    statements are idempotent and scoped to RUNNING rows older than the
    threshold.
    """
    return _sweep_stale_runs()


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


def _get_markdown_chars(db, run_id, document_id: str) -> int:
    """Read markdown size from prepare_document.metrics, falling back to a
    MinIO read when the metric is missing (older runs / 5xx-fallback path).
    """
    row = db.execute(
        sa.text(
            "SELECT metrics FROM ingest.stage_runs "
            "WHERE pipeline_run_id = :run_id "
            "  AND stage_name = 'prepare_document' "
            "  AND status = 'COMPLETE' "
            "  AND pass_name IS NULL "
            "ORDER BY started_at DESC NULLS LAST LIMIT 1"
        ),
        {"run_id": str(run_id)},
    ).scalar()
    if isinstance(row, dict) and "markdown_chars" in row:
        return int(row.get("markdown_chars") or 0)
    try:
        from app.services.storage import download_bytes_sync
        return len(download_bytes_sync(
            settings.minio_bucket_derived,
            f"artifacts/{document_id}/docling_document.md",
        ))
    except Exception:
        return 0


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


# ── dispatch ledger v1 (spec 2026-05-10) ──────────────────────────────────


@dataclass(frozen=True)
class StageEdge:
    """Edge in the sequential pipeline graph.

    next_stage is the persisted stage_name (matches @guard_stage_run argument).
    next_task is the fully-qualified Celery task path. These can differ —
    e.g. derive_text_embeddings (persisted) ↔ derive_text_chunks_and_embeddings (task).
    """
    next_stage: str | None
    next_task:  str | None


STAGE_SUCCESSORS: dict[str, StageEdge] = {
    "prepare_document":            StageEdge("detect_and_translate",        "app.workers.pipeline.detect_and_translate"),
    "detect_and_translate":        StageEdge("derive_document_metadata",    "app.workers.pipeline.derive_document_metadata"),
    "derive_document_metadata":    StageEdge("purge_document_derivations",  "app.workers.pipeline.purge_document_derivations"),
    "purge_document_derivations":  StageEdge("derive_picture_descriptions", "app.workers.pipeline.derive_picture_descriptions"),
    "derive_picture_descriptions": StageEdge("derive_text_embeddings",      "app.workers.pipeline.derive_text_chunks_and_embeddings"),
    "derive_text_embeddings":      StageEdge("derive_image_embeddings",     "app.workers.pipeline.derive_image_embeddings"),
    "derive_image_embeddings":     StageEdge("derive_document_anchors",     "app.workers.pipeline.derive_document_anchors"),
    "derive_document_anchors":     StageEdge("derive_ontology_graph",       "app.workers.pipeline.derive_ontology_graph"),
    "derive_ontology_graph":       StageEdge(None, None),
}

# Stages 1–8 (sequential). Stage 9 (derive_ontology_graph) is in STAGE_SUCCESSORS
# but excluded here because its summary row legitimately stays RUNNING for the
# entire per-pass fan-in window; the existing reconcile_ontology_graph_runs
# reconciler owns its stale-RUNNING handling.
LEDGER_SEQUENTIAL_STAGES = [s for s in STAGE_SUCCESSORS if s != "derive_ontology_graph"]
LEDGER_FANOUT_STAGES     = ["derive_ontology_graph"]


def _resolve_queue(task_name: str) -> str:
    """Return the queue Celery will actually route a task to.

    3-tier precedence (matches Celery's own lookup order):
    1. Explicit `task_routes[task_name]["queue"]` from celery_app.conf
    2. The task's decorator `queue=` argument (via celery_app.tasks[name].queue,
       which Celery exposes through Task._get_exec_options() and apply_async()
       honors at send time)
    3. Broker default (celery_app.conf.task_default_queue, "celery" unless overridden)

    All current ledger stages resolve via tiers 1–2; tier 3 is the safety net
    for any task that is neither in task_routes nor decorated with queue=.
    The helper unifies the lookup so the ledger's queue_name column matches
    the runtime destination.
    """
    routes = celery_app.conf.task_routes or {}
    entry = routes.get(task_name)
    if entry and entry.get("queue"):
        return entry["queue"]
    task = celery_app.tasks.get(task_name)
    if task is not None:
        decorator_queue = getattr(task, "queue", None)
        if decorator_queue:
            return decorator_queue
    return celery_app.conf.task_default_queue or "celery"


def _seed_first_stage(
    db,
    *,
    pipeline_run_id: str,
    stage_name: str,
    task_name: str,
) -> None:
    """Insert the initial PENDING ledger row for a pipeline_run.

    Idempotent on the partial unique index (pipeline_run_id, stage_name, attempt)
    WHERE pass_name IS NULL — a second call is a no-op.

    Caller is responsible for db.commit().
    """
    queue = _resolve_queue(task_name)
    db.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             queue_name, task_name, available_at, dispatch_attempt)
        VALUES (gen_random_uuid(), :run_id, :stage, 1, 'PENDING',
                :queue, :task, NOW(), 1)
        ON CONFLICT (pipeline_run_id, stage_name, attempt)
        WHERE pass_name IS NULL
        DO NOTHING
    """), {
        "run_id": pipeline_run_id,
        "stage":  stage_name,
        "queue":  queue,
        "task":   task_name,
    })


@dataclass(frozen=True)
class _ClaimResult:
    """Outcome of a Tx-1 CLAIM attempt.

    outcome ∈ {proceed, legacy, already_complete, concurrent_running,
               stale_pending, terminal_failed}
    early_result is what the wrapper returns to Celery (None or dict).
    dispatch_attempt is populated only when outcome == "proceed".
    """
    outcome: str
    early_result: dict | None = None
    dispatch_attempt: int = 0


def _claim_tx1(
    db,
    pipeline_run_id: str,
    stage_name: str,
    *,
    celery_task_id: str,
    is_celery_retry: bool,
) -> _ClaimResult:
    """Tx-1 CLAIM: atomically transition a ledger row to RUNNING.

    Returns a _ClaimResult describing one of 6 outcomes. The caller
    (guard_stage_run wrapper) reads `outcome` to decide whether to set
    `_CTX` (only on `proceed`), return early (the 4 zero-row outcomes
    with dict/None payload), or run the body inline (legacy).
    """
    update = db.execute(text("""
        UPDATE ingest.stage_runs
        SET status         = 'RUNNING',
            started_at     = COALESCE(started_at, NOW()),
            celery_task_id = :celery_task_id
        WHERE pipeline_run_id = :run_id
          AND stage_name      = :stage_name
          AND pass_name       IS NULL
          AND (
                status IN ('DISPATCHED', 'PENDING')
             OR (status = 'RUNNING' AND :is_celery_retry)
          )
        RETURNING id, attempt, dispatch_attempt
    """), {
        "run_id": pipeline_run_id,
        "stage_name": stage_name,
        "celery_task_id": celery_task_id,
        "is_celery_retry": is_celery_retry,
    }).first()

    if update is not None:
        return _ClaimResult(outcome="proceed", dispatch_attempt=update.dispatch_attempt)

    # 0 rows updated — follow-up SELECT to disambiguate.
    current = db.execute(text("""
        SELECT status FROM ingest.stage_runs
        WHERE pipeline_run_id = :run_id
          AND stage_name      = :stage_name
          AND pass_name       IS NULL
    """), {"run_id": pipeline_run_id, "stage_name": stage_name}).first()

    if current is None:
        return _ClaimResult(outcome="legacy", early_result=None)

    if current.status == "COMPLETE":
        return _ClaimResult(
            outcome="already_complete",
            early_result={
                "stage": stage_name,
                "status": "skipped",
                "reason": "already_complete",
            },
        )
    if current.status == "RUNNING":
        return _ClaimResult(outcome="concurrent_running", early_result=None)
    if current.status == "PENDING":
        return _ClaimResult(outcome="stale_pending", early_result=None)
    if current.status == "FAILED":
        return _ClaimResult(
            outcome="terminal_failed",
            early_result={
                "stage": stage_name,
                "status": "terminal_failed",
                "reason": "stage_previously_failed",
            },
        )

    # Defensive: unexpected status (e.g. DISPATCHED visible to follow-up means
    # CLAIM raced — treat as concurrent and skip).
    return _ClaimResult(outcome="concurrent_running", early_result=None)


from app.workers._stage_lifecycle import _LifecycleCtx, _CTX  # noqa: E402


def _tx3_complete_and_enqueue_next(ctx: _LifecycleCtx) -> None:
    """Tx-3: insert successor PENDING + flip self → COMPLETE, single transaction.

    The central durability guarantee. Both writes commit together; any exception
    inside the `with db.begin():` block rolls both back atomically, leaving the
    ledger row in RUNNING for the stale-RUNNING sweeper to recover.
    """
    db = _get_db()
    try:
        with db.begin():
            # 3a: insert successor PENDING (idempotent via partial unique index)
            db.execute(text("""
                INSERT INTO ingest.stage_runs
                    (id, pipeline_run_id, stage_name, attempt, status,
                     queue_name, task_name, available_at, dispatch_attempt)
                VALUES (gen_random_uuid(), :run_id, :next_stage, 1, 'PENDING',
                        :next_queue, :next_task, NOW(), 1)
                ON CONFLICT (pipeline_run_id, stage_name, attempt)
                WHERE pass_name IS NULL
                DO NOTHING
            """), {
                "run_id":     ctx.pipeline_run_id,
                "next_stage": ctx.next_stage,
                "next_queue": _resolve_queue(ctx.next_task),
                "next_task":  ctx.next_task,
            })
            # 3b: flip self → COMPLETE with metrics stashed by interception.
            # CAST(:metrics AS jsonb) is required because psycopg2 cannot
            # adapt a Python dict directly when bound via raw text(); we
            # serialize to JSON text and let Postgres parse it back.
            db.execute(text("""
                UPDATE ingest.stage_runs
                SET status      = 'COMPLETE',
                    finished_at = NOW(),
                    metrics     = CAST(:metrics AS jsonb)
                WHERE pipeline_run_id = :run_id
                  AND stage_name      = :stage_name
                  AND pass_name       IS NULL
            """), {
                "run_id":     ctx.pipeline_run_id,
                "stage_name": ctx.stage_name,
                "metrics":    json.dumps(ctx.pending_metrics)
                              if ctx.pending_metrics is not None else None,
            })
    finally:
        db.close()


def _tx4_finalize_failure(
    ctx: _LifecycleCtx,
    *,
    error: str,
    celery_retries: int,
    max_retries: int,
    backoff_seconds: int = 60,
) -> None:
    """Tx-4: handle failure of a lifecycle-wrapped stage.

    Retryable if ``ctx.dispatch_attempt + 1 <= settings.max_stage_dispatches``:
    status → PENDING, dispatch_attempt += 1, available_at advances by backoff,
    and started_at / dispatched_at are cleared so the next CLAIM is clean.

    Terminal otherwise: stage_run → FAILED and pipeline_run → FAILED in a
    single ``with db.begin():`` block. The pipeline_run UPDATE is gated on
    ``status = 'PROCESSING'`` so an earlier FAILED message is preserved.

    ``max_stage_dispatches`` is read inside the function via ``get_settings()``
    so tests can monkeypatch the live cached `Settings` instance.
    """
    _settings = get_settings()

    next_dispatch_attempt = ctx.dispatch_attempt + 1
    db = _get_db()
    try:
        if next_dispatch_attempt <= _settings.max_stage_dispatches:
            db.execute(text("""
                UPDATE ingest.stage_runs
                SET status           = 'PENDING',
                    dispatch_attempt = :next_da,
                    available_at     = NOW() + (:backoff || ' seconds')::interval,
                    started_at       = NULL,
                    dispatched_at    = NULL,
                    error_message    = :err
                WHERE pipeline_run_id = :run_id
                  AND stage_name      = :stage_name
                  AND pass_name       IS NULL
            """), {
                "run_id":     ctx.pipeline_run_id,
                "stage_name": ctx.stage_name,
                "next_da":    next_dispatch_attempt,
                "backoff":    str(backoff_seconds),
                "err":        error,
            })
            db.commit()
            return

        # Terminal — atomic stage_run FAILED + pipeline_run FAILED.
        with db.begin():
            db.execute(text("""
                UPDATE ingest.stage_runs
                SET status           = 'FAILED',
                    dispatch_attempt = :next_da,
                    finished_at      = NOW(),
                    error_message    = :err
                WHERE pipeline_run_id = :run_id
                  AND stage_name      = :stage_name
                  AND pass_name       IS NULL
            """), {
                "run_id":     ctx.pipeline_run_id,
                "stage_name": ctx.stage_name,
                "next_da":    next_dispatch_attempt,
                "err":        error,
            })
            db.execute(text("""
                UPDATE ingest.pipeline_runs
                SET status        = 'FAILED',
                    finished_at   = NOW(),
                    error_message = :err
                WHERE id = :run_id AND status = 'PROCESSING'
            """), {"run_id": ctx.pipeline_run_id, "err": error})
    finally:
        db.close()


def _finalize_after_body(ctx: _LifecycleCtx, result) -> None:
    """Body-return contract: Tx-3 on success, Tx-4 on failure dict / pending FAILED.

    skipped is treated as success — detect_and_translate, derive_document_metadata,
    derive_picture_descriptions all return {"status":"skipped",...} on legitimate
    no-op completions and the pipeline must advance.
    """
    if not ctx.intercept_terminal:
        return  # stage 9: merge owns finalization

    failed = (
        ctx.pending_status == "FAILED"
        or (isinstance(result, dict) and result.get("status") in ("FAILED", "failed"))
    )
    if failed:
        _tx4_finalize_failure(
            ctx,
            error=ctx.pending_error or "stage returned failure status",
            celery_retries=0,
            max_retries=0,
        )
        return

    _tx3_complete_and_enqueue_next(ctx)


def _assert_ledger_wiring() -> None:
    """Module-load check: every STAGE_SUCCESSORS key has a registered task
    whose wrapper carries ``_lifecycle=True``. Raises RuntimeError on mismatch.

    NOTE: Until Task 20 wires lifecycle=True on all 9 stage decorators, this
    function will raise. That's expected; the check is called only after
    Task 22 enables ``_post_register_ledger_checks`` in celery_app.py.
    """
    for stage_name in STAGE_SUCCESSORS:
        match = None
        for task_name, task in celery_app.tasks.items():
            run = getattr(task, "run", None)
            if run is None:
                continue
            if getattr(run, "stage_name", None) == stage_name:
                match = task
                break
        if match is None:
            raise RuntimeError(
                f"_assert_ledger_wiring: STAGE_SUCCESSORS lists {stage_name!r} "
                f"but no registered Celery task has stage_name={stage_name!r}"
            )
        if not getattr(match.run, "_lifecycle", False):
            raise RuntimeError(
                f"_assert_ledger_wiring: task for stage {stage_name!r} is missing "
                f"@guard_stage_run(..., lifecycle=True)"
            )


def _assert_threshold_envelope() -> None:
    """Module-load check: ``stale_stage_run_threshold_seconds`` exceeds every
    ledger stage's ``time_limit + max_retries * default_retry_delay``.

    Raises RuntimeError on misconfiguration.
    """
    _settings = get_settings()
    threshold = _settings.stale_stage_run_threshold_seconds
    for stage_name in STAGE_SUCCESSORS:
        for task_name, task in celery_app.tasks.items():
            if getattr(getattr(task, "run", None), "stage_name", None) != stage_name:
                continue
            time_limit = task.time_limit or 0
            max_retries = task.max_retries or 0
            retry_delay = task.default_retry_delay or 0
            envelope = time_limit + max_retries * retry_delay
            if threshold < envelope:
                raise RuntimeError(
                    f"_assert_threshold_envelope: "
                    f"stale_stage_run_threshold_seconds ({threshold}) "
                    f"must exceed envelope ({envelope}) "
                    f"for ledger stage {stage_name!r}"
                )


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

        # ── seed first ledger row (spec 2026-05-10) ───────────────────────
        # The dispatcher will pick up this PENDING row within 5s and publish
        # the prepare_document task. From there, each stage's lifecycle wrapper
        # commits the next stage's PENDING row in the same transaction as its
        # own COMPLETE — no chain to lose.
        _seed_first_stage(
            db,
            pipeline_run_id=run_id,
            stage_name="prepare_document",
            task_name="app.workers.pipeline.prepare_document",
        )
        db.commit()
    finally:
        db.close()

    logger.info(
        "start_ingest_pipeline: document_id=%s pipeline_run_id=%s bundle=%s "
        "(ledger seed; dispatcher will publish within 5s)",
        document_id, run_id, resolved_key,
    )
    return IngestDispatchResult(
        pipeline_run_id=run_id,
        celery_task_id="",
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
    snapshot, then dispatches a 2-stage chain ending at
    derive_ontology_graph. The downstream chain
    (derive_structure_links → finalize_document) is dispatched by
    derive_ontology_graph_merge after the per-pass fan-in completes.

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

    # CHANGED 2026-05-06 (Task 7 of per-pass-celery-fanin): outer chain trimmed
    # from 4 → 2 stages. Downstream stages (derive_structure_links →
    # finalize_document) are now dispatched by derive_ontology_graph_merge in
    # graph_only mode after the per-pass fan-in completes.
    result = celery_chain(
        derive_document_anchors.si(doc_id_str, run_id),
        derive_ontology_graph.si(doc_id_str, run_id),
    ).apply_async()

    return {
        "pipeline_run_id": run_id,
        "celery_task_id": result.id,
        "ontology_bundle_key": resolved_key,
    }


import functools


def guard_stage_run(
    stage_name: str,
    *,
    lifecycle: bool = False,
    next_stage: str | None = None,
    next_task: str | None = None,
    intercept_terminal: bool = True,
):
    """Wrap a pipeline task with FAILED-on-uncaught-exception safety net.

    Lifecycle additions (v1, spec 2026-05-10):
    - `lifecycle=True` enables Tx-1 CLAIM, _CTX, Tx-3 / Tx-4.
    - `next_stage` / `next_task` describe the successor (for Tx-3a).
    - `intercept_terminal=False` is for stage 9 (derive_ontology_graph), whose
      summary row's COMPLETE is owned by derive_ontology_graph_merge.

    Without `lifecycle=True`, behaves exactly as before: pass CeleryRetry /
    SoftTimeLimitExceeded through, terminalize other exceptions.

    Original safety-net rationale: silent-orphan case observed on 2026-04-23 —
    a task marked RUNNING then died with no log entry and no status update.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, document_id, run_id=None, *args, **kwargs):
            # Non-lifecycle invocation — preserve existing behavior unchanged.
            if not (lifecycle and run_id):
                return _guard_existing_body(
                    self, fn, stage_name, document_id, run_id, *args, **kwargs
                )

            # Tx-1 CLAIM
            db = _get_db()
            try:
                claim = _claim_tx1(
                    db, run_id, stage_name,
                    celery_task_id=str(self.request.id),
                    is_celery_retry=(self.request.retries > 0),
                )
                db.commit()
            finally:
                db.close()

            if claim.outcome == "legacy":
                return _guard_existing_body(
                    self, fn, stage_name, document_id, run_id, *args, **kwargs
                )
            if claim.outcome != "proceed":
                return claim.early_result

            ctx = _LifecycleCtx(
                pipeline_run_id=str(run_id),
                stage_name=stage_name,
                dispatch_attempt=claim.dispatch_attempt,
                intercept_terminal=intercept_terminal,
                next_stage=next_stage,
                next_task=next_task,
            )
            token = _CTX.set(ctx)
            try:
                result = fn(self, document_id, run_id, *args, **kwargs)
                if intercept_terminal:
                    _finalize_after_body(ctx, result)
                return result
            except CeleryRetry:
                # Re-raise without Tx-4; row stays RUNNING for Celery to republish.
                raise
            except Exception as exc:
                if intercept_terminal:
                    _tx4_finalize_failure(
                        ctx,
                        error=f"{type(exc).__name__}: {exc!r}",
                        celery_retries=self.request.retries,
                        max_retries=self.max_retries,
                    )
                raise
            finally:
                _CTX.reset(token)

        wrapper.stage_name = stage_name                  # pre-existing marker
        wrapper._lifecycle = lifecycle                   # NEW
        wrapper._intercept_terminal = intercept_terminal # NEW
        return wrapper
    return decorator


def _guard_existing_body(self, fn, stage_name, document_id, run_id, *args, **kwargs):
    """Pre-design guard_stage_run body — preserved for non-lifecycle invocations.

    CeleryRetry and SoftTimeLimitExceeded are passed through untouched — those
    are Celery's own control-flow exceptions and the task's existing except
    branches handle them. Any other exception triggers a defensive FAILED
    status write (scoped to the current run_id, if any) and a full traceback
    log, then re-raises so Celery's retry / failure machinery still runs.
    """
    try:
        return fn(self, document_id, run_id, *args, **kwargs)
    except CeleryRetry:
        raise
    # NOTE: SoftTimeLimitExceeded is intentionally NOT passed through here.
    # When a task's body calls self.retry(exc=SoftTimeLimitExceeded) on retry
    # exhaustion, Celery 5 re-raises the provided `exc` itself (not
    # MaxRetriesExceededError), so SoftTimeLimitExceeded reaches this wrapper
    # on the final attempt. Letting it fall through to `except Exception`
    # ensures terminalization on exhaustion.
    except Exception as exc:
        logger.exception(
            "guard_stage_run: %s raised unhandled exception "
            "(document_id=%s run_id=%s)",
            stage_name, document_id, run_id,
        )
        if run_id:
            try:
                db = _get_db()
                try:
                    _update_stage_run(
                        db, run_id, stage_name, "FAILED",
                        attempt=self.request.retries + 1,
                        error=f"unhandled exception: {exc!r}",
                    )
                    db.commit()
                finally:
                    db.close()
            except Exception:
                logger.exception(
                    "guard_stage_run: FAILED-status write also failed "
                    "for run_id=%s stage=%s",
                    run_id, stage_name,
                )
        # Unconditional terminalization: reaching this branch means the task
        # did not convert the exception via self.retry (which raises
        # CeleryRetry and is pass-through). The helper preserves existing
        # terminal statuses.
        _terminalize_doc_and_run(document_id, run_id, "PARTIAL_COMPLETE")
        raise


def _update_stage_run(
    db, pipeline_run_id: str, stage_name: str, status: str,
    attempt: int = 1, metrics: dict | None = None, error: str | None = None,
) -> None:
    """Upsert a StageRun record."""
    # ── lifecycle interception (spec 2026-05-10) ────────────────────────
    # When a lifecycle-wrapped stage is mid-body, defer terminal writes to
    # the wrapper's Tx-3 / Tx-4 so the body's COMPLETE commit cannot escape
    # before the successor row is inserted in the same transaction.
    ctx = _CTX.get()
    if (
        ctx is not None
        and str(ctx.pipeline_run_id) == str(pipeline_run_id)
        and ctx.stage_name == stage_name
        and ctx.intercept_terminal
    ):
        if status == "RUNNING":
            return                          # wrapper already wrote RUNNING via CLAIM
        if status in ("COMPLETE", "FAILED"):
            ctx.pending_status = status
            ctx.pending_metrics = metrics
            ctx.pending_error = error
            return
        # Other statuses fall through (defensive)
    # ─────────────────────────────────────────────────────────────────────

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
) -> uuid.UUID | None:
    """Upsert a per-pass StageRun row targeting the partial unique index
    uq_stage_runs_run_pass_attempt (WHERE pass_name IS NOT NULL) added in
    migration 0015.

    Returns the UUID of the inserted or updated StageRun row so callers can
    populate pipeline_pass_outputs.stage_run_id for direct FK linkage.
    Returns None only if the upsert fails (stale run_id / FK violation) and
    the exception is swallowed.

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
        if counts.get("metrics"):
            values["metrics"] = counts["metrics"]
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
        .returning(StageRun.id)
    )
    db = _get_db()
    try:
        row = db.execute(stmt).fetchone()
        db.commit()
        return row[0] if row else None
    except Exception as exc:
        db.rollback()
        logger.debug(
            "_write_stage_run skipped (stale run_id %s, pass %s): %s",
            pipeline_run_id, pass_def.name, exc,
        )
        return None
    finally:
        db.close()


def _build_extract_pass_request(
    *, bundle_key: str, pass_def, doc_json: dict,
    upstream_refs: dict | None, document_id: str,
) -> dict:
    """Assemble the POST body for /extract-pass.

    document_id is always included so the service can log and attribute
    extraction runs to a specific document (useful when correlating
    salvage warnings and timeout retries across the batch).
    """
    body: dict = {
        "bundle_key": bundle_key,
        "pass_name": pass_def.name,
        "document_id": document_id,
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
        raise PassTransportError(f"transport error: {exc}") from exc

    if response.status_code >= 500:
        raise PassRetryable(f"HTTP {response.status_code}: {response.text[:200]}")
    if response.status_code >= 400:
        raise PassTerminal(f"HTTP {response.status_code}: {response.text[:200]}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise PassRetryable(f"partial/malformed response: {exc}") from exc

    # Worker-side handling of service-level soft-fails. The docling-graph
    # service returns 200 even when run_pipeline raised internally (it stubs
    # the context with empty output and a pipeline_error diagnostic marker).
    # We treat that stub as PassRetryable: a real internal exception inside
    # the library is a transient failure mode (LLM JSON-parse drift,
    # connection blip to Ollama mid-extraction), not a clean "no entities
    # found." Empty results from a clean run_pipeline path do NOT set
    # pipeline_error and are handled below as ZERO_YIELD (logged but not
    # retried — could be legitimate empty for off-domain passes).
    diagnostics = (payload or {}).get("diagnostics") or {}
    pipeline_err = diagnostics.get("pipeline_error") if isinstance(diagnostics, dict) else None
    metadata = (payload or {}).get("metadata") or {}
    node_count = metadata.get("node_count", 0) or 0
    edge_count = metadata.get("edge_count", 0) or 0
    bundle_key = request_body.get("bundle_key", "?")
    pass_name = request_body.get("pass_name", "?")
    document_id = request_body.get("document_id", "?")
    if pipeline_err:
        logger.error(
            "EXTRACT_PASS_PIPELINE_ERROR bundle=%s pass=%s document_id=%s "
            "error_type=%s error_msg=%s — service stubbed the response; "
            "raising PassRetryable so the worker can retry instead of "
            "silently treating it as success.",
            bundle_key, pass_name, document_id,
            pipeline_err.get("type", "?"), pipeline_err.get("message", "?"),
        )
        raise PassRetryable(
            f"service pipeline_error: type={pipeline_err.get('type', '?')} "
            f"msg={pipeline_err.get('message', '?')[:200]}"
        )
    if node_count == 0 and edge_count == 0:
        logger.error(
            "EXTRACT_PASS_ZERO_YIELD bundle=%s pass=%s document_id=%s "
            "node_count=0 edge_count=0 — pass succeeded but produced no "
            "entities/edges; review docling-graph logs for library warnings.",
            bundle_key, pass_name, document_id,
        )

    return payload


def _parse_pass_response(response_json: dict, pass_def, manifest) -> "object":
    """Validate the /extract-pass response dict into a PassResult with an
    instantiated pass template class.

    Pydantic validation errors are terminal — a malformed response won't
    heal on retry (spec §6.5).

    Phase 8 Task 52: any ``provenance`` list in the response is parsed
    into ``ExtractionProvenance`` rows and attached to PassResult.
    Malformed rows (missing instance_id / ontology_name / element_uid)
    are dropped with a WARNING; the pass still succeeds with the rows
    that DID parse, because losing some mention rows is a quality
    degradation, not a correctness failure.
    """
    import importlib
    from pydantic import ValidationError
    from app.services.extraction_merge import (
        ExtractionMetadata,
        ExtractionProvenance,
        ExtractionRelationshipProvenance,
        FieldEvidenceRow,
        PassResult,
    )

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

    provenance_rows: list[ExtractionProvenance] = []
    for raw in response_json.get("provenance") or []:
        if not isinstance(raw, dict):
            logger.warning("_parse_pass_response: dropping non-dict provenance row: %r", raw)
            continue
        instance_id = raw.get("instance_id")
        ontology_name = raw.get("ontology_name")
        element_uid = raw.get("element_uid")
        if not (isinstance(instance_id, str) and instance_id
                and isinstance(ontology_name, str) and ontology_name
                and isinstance(element_uid, str) and element_uid):
            logger.warning(
                "_parse_pass_response: dropping provenance row missing required fields: %r",
                raw,
            )
            continue
        identity_values = raw.get("identity_values") or {}
        if not isinstance(identity_values, dict):
            identity_values = {}
        page = raw.get("page")
        if page is not None and not isinstance(page, int):
            page = None
        chunk_index = raw.get("chunk_index")
        if chunk_index is not None and not isinstance(chunk_index, int):
            chunk_index = None
        evidence_ids = raw.get("evidence_ids") or []
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        page_numbers = raw.get("page_numbers") or []
        if not isinstance(page_numbers, list):
            page_numbers = []
        evidence_text = raw.get("evidence_text")
        if evidence_text is not None and not isinstance(evidence_text, str):
            evidence_text = None
        provenance_rows.append(ExtractionProvenance(
            instance_id=instance_id,
            ontology_name=ontology_name,
            identity_values=identity_values,
            element_uid=element_uid,
            page=page,
            chunk_index=chunk_index,
            evidence_ids=evidence_ids,
            page_numbers=page_numbers,
            evidence_text=evidence_text,
        ))

    # Phase 3 task 32: parse field_provenance rows from the response
    # and group by (instance_id, field_name) for the merger to attach.
    # element_uid → chunk_id resolution happens lazily downstream
    # (chunk_id is the worker-side handle); we keep element_uid here.
    field_evidence: dict[str, dict[str, list[FieldEvidenceRow]]] = {}
    for raw in response_json.get("field_provenance") or []:
        if not isinstance(raw, dict):
            logger.warning(
                "_parse_pass_response: dropping non-dict field_provenance row: %r",
                raw,
            )
            continue
        instance_id = raw.get("instance_id")
        field_name = raw.get("field_name")
        snippet = raw.get("supporting_snippet") or ""
        if not (isinstance(instance_id, str) and instance_id
                and isinstance(field_name, str) and field_name
                and snippet):
            logger.warning(
                "_parse_pass_response: dropping field_provenance row missing required fields: %r",
                raw,
            )
            continue
        element_uid = raw.get("element_uid")
        if element_uid is not None and not isinstance(element_uid, str):
            element_uid = None
        evidence_id = raw.get("evidence_id")
        if evidence_id is not None and not isinstance(evidence_id, str):
            evidence_id = None
        fe_page = raw.get("page")
        if fe_page is not None and not isinstance(fe_page, int):
            fe_page = None
        fe_document_id = raw.get("document_id")
        if fe_document_id is not None and not isinstance(fe_document_id, str):
            fe_document_id = None
        row = FieldEvidenceRow(
            chunk_id=None,  # resolved later from element_uid → chunk vertex
            snippet=snippet,
            element_uid=element_uid,
            value=raw.get("value"),
            evidence_id=evidence_id,
            page=fe_page,
            document_id=fe_document_id,
        )
        field_evidence.setdefault(instance_id, {}).setdefault(field_name, []).append(row)

    # Mechanism A1 (spec §4.4 + §5.5, plan Task 9): parse the doc-level
    # table_overlay payload into the worker-side TableOverlay so the
    # downstream merge_and_resolve._extract_doc_overlay (Task 8) finds
    # real data instead of falling through the kill-switch path.
    # Malformed payloads do NOT terminate the pass — the LLM extraction
    # itself succeeded; losing the table overlay is a quality
    # degradation, so log WARNING and continue with table_overlay=None.
    overlay_dict = response_json.get("table_overlay")
    table_overlay_obj = None
    if isinstance(overlay_dict, dict):
        try:
            from app.services.table_overlay import TableOverlay
            table_overlay_obj = TableOverlay.model_validate(overlay_dict)
        except Exception as exc:
            logger.warning(
                "_parse_pass_response: dropping malformed table_overlay: %s", exc,
            )
            table_overlay_obj = None

    relationship_provenance_rows: list[ExtractionRelationshipProvenance] = []
    for raw in response_json.get("relationship_provenance") or []:
        if not isinstance(raw, dict):
            logger.warning(
                "_parse_pass_response: dropping non-dict relationship_provenance row: %r", raw,
            )
            continue
        rel_type = raw.get("relationship_type")
        if not (isinstance(rel_type, str) and rel_type):
            logger.warning(
                "_parse_pass_response: dropping relationship_provenance row missing "
                "relationship_type: %r", raw,
            )
            continue
        rp_evidence_ids = raw.get("evidence_ids") or []
        if not isinstance(rp_evidence_ids, list):
            rp_evidence_ids = []
        rp_self_refs = raw.get("self_refs") or []
        if not isinstance(rp_self_refs, list):
            rp_self_refs = []
        rp_page_numbers = raw.get("page_numbers") or []
        if not isinstance(rp_page_numbers, list):
            rp_page_numbers = []
        rp_snippet = raw.get("supporting_snippet")
        if rp_snippet is not None and not isinstance(rp_snippet, str):
            rp_snippet = None
        relationship_provenance_rows.append(ExtractionRelationshipProvenance(
            relationship_type=rel_type,
            source_instance_id=raw.get("source_instance_id"),
            target_instance_id=raw.get("target_instance_id"),
            evidence_ids=rp_evidence_ids,
            self_refs=rp_self_refs,
            page_numbers=rp_page_numbers,
            supporting_snippet=rp_snippet,
        ))

    return PassResult(
        pass_name=pass_def.name,
        template_instance=template_instance,
        metadata=ExtractionMetadata(
            schema_size_chars=metadata_dict.get("schema_size_chars", 0),
            structured_output_mode=metadata_dict.get("structured_output_mode", "strict"),
        ),
        pre_merge_rejections=[],
        provenance=provenance_rows,
        field_evidence=field_evidence,
        table_overlay=table_overlay_obj,
        relationship_provenance=relationship_provenance_rows,
    )


def _backoff(attempt: int) -> None:
    """Exponential backoff per spec §6.5: 30s × 2^(attempt-1), capped at 300s."""
    import time
    delay = min(30 * (2 ** (attempt - 1)), 300)
    time.sleep(delay)


def _count_pass_output(pass_result, pass_def, ontology) -> dict:
    """Pre-merge per-pass counts (plan Task 35c).

    Entity counts come from ``iter_entities_of_type`` — which now reads
    from ``pre_merge_walk.entities`` (via ``_cached_entities``) when the
    pass loop built the shared summary (Task 34b), so nested children
    behind typed-edge fields are included. ``relationships_extracted``
    is the walker's ``raw_edge_count`` for typed-edge passes or the DTO
    list length for ``system_links`` — both carried uniformly on
    ``pre_merge_walk.raw_edge_count``.

    ``relationships_rejected`` is FORCED to 0 at pre-merge.
    ``_apply_post_merge_yield_updates`` (Task 36) is the single
    authority for rejected counts — merge-time VALIDATION_MATRIX
    triple-checks haven't run yet when this row is written.

    Fallback for test-built ``PassResult``s without ``pre_merge_walk``:
    ``relationships_extracted`` falls back to ``len(relationships)``
    (legacy DTO-list count) but ``relationships_rejected`` stays 0.
    """
    primary = sum(
        len(list(pass_result.iter_entities_of_type(t)))
        for t in pass_def.primary_entity_types
    ) if hasattr(pass_result, "iter_entities_of_type") else 0
    bridge = sum(
        len(list(pass_result.iter_entities_of_type(t)))
        for t in pass_def.bridge_entity_types
    ) if hasattr(pass_result, "iter_entities_of_type") else 0
    pmw = getattr(pass_result, "pre_merge_walk", None)
    if pmw is not None:
        extracted_rels = pmw.raw_edge_count
    else:
        extracted_rels = len(getattr(pass_result, "relationships", []) or [])
    rejected_rels = 0  # Forced at pre-merge; post-merge is authoritative.
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


def _build_rejections_by_reason(
    rejections: list | None,
) -> dict[str, int]:
    """Bucket rejection tuples by the reason enum's ``.value``
    (lowercase, e.g. 'unknown_ref_id'). Accepts both tuple shapes:

    * ``(rel, reason)`` — ``pass_result.pre_merge_rejections``
    * ``(source_pass, raw_rel, reason)`` — ``MergedExtraction.rejected_edges``
      (see extraction_merge.py:159)

    The helper treats the **last** element of each tuple as the reason,
    which works for both shapes without the caller needing a conditional.
    Used to persist per-reason counts into ``StageRun.metrics`` JSONB so
    UNKNOWN_REF_ID trends are queryable from the DB without reprocessing
    passes."""
    result: dict[str, int] = {}
    for tup in rejections or []:
        if not tup:
            continue
        reason = tup[-1]
        key = reason.value if hasattr(reason, "value") else str(reason)
        result[key] = result.get(key, 0) + 1
    return result


def _is_valid_upstream_ref(ref, ontology: dict) -> bool:
    """Single shared validity rule used at ref emission, request build, and
    merge attachment. A ref is valid iff:
      (a) its entity_type is in the ontology,
      (b) every ontology identity_field for that type is present as a key
          in identity_values,
      (c) every such value is truthy after ``str.strip()`` for strings
          (None, "", "   " all reject).

    Applied at three sites so invalid refs cannot leak into the request
    body, the prompt preamble, or ``PassResult.upstream_refs``. A ref
    that fails this check simply never existed as far as the rest of the
    pipeline is concerned — no UNKNOWN_REF_ID rejection, no polluted
    LogicalIdentity.
    """
    entity_type = getattr(ref, "entity_type", None)
    identity_values = getattr(ref, "identity_values", None) or {}
    entity_def = next(
        (e for e in ontology.get("entity_types", []) if e["name"] == entity_type),
        None,
    )
    if entity_def is None:
        return False
    identity_fields = list(entity_def.get("identity_fields") or ())
    if not identity_fields:
        # Rule (b): no anchors → not usable as an upstream ref.
        return False
    for field in identity_fields:
        if field not in identity_values:
            return False
        val = identity_values[field]
        if val is None:
            return False
        if isinstance(val, str) and not val.strip():
            return False
    return True


def _normalized_identity_key(entity_type: str, identity_values: dict) -> tuple:
    """Build the dedupe key matching the merge layer's identity contract.

    Spec §4.5: dedupe by (entity_type, normalized identity_values). Uses
    canonicalize_identity_text() (collapses whitespace, drops null
    sentinels) plus .casefold() for case-insensitive matching — same
    canonicalizer the merge layer applies. Non-string values pass through
    unchanged so non-text identity fields don't get mangled.
    """
    normalized: dict = {}
    for k, v in identity_values.items():
        if isinstance(v, str):
            canon = canonicalize_identity_text(v)
            normalized[k] = canon.casefold() if canon else None
        else:
            normalized[k] = v
    return (entity_type, tuple(sorted(normalized.items())))


def _extend_upstream_refs(
    upstream_refs: dict, pass_result, pass_def, ontology
) -> None:
    """Add ref_id → ref entries to upstream_refs for every primary entity
    produced by this pass so downstream passes can reference them.

    **Merge-preserving scratch-dict dedup (plan Task 35a):** once
    ``iter_entities_of_type`` yields nested entities, the same logical
    entity can be reached via multiple graph paths with complementary
    non-null fields. A detached scratch accumulator keyed on
    ``(entity_type, ontology-ordered identity tuple)`` unions non-identity
    fields across duplicates (first-non-null wins) WITHOUT mutating the
    live Python instances — merge stages later consume those instances.
    ``display_label`` is built from the merged scratch so prompts and UIs
    see the richest name even when no single instance had everything.

    **Cross-pass dedupe (spec §4.5, plan Task 13):** after the radar
    field-group cutover, 5 sub-passes each emit a partial RADAR_SYSTEM
    with the same system_name. A ``seen`` set built from existing
    ``upstream_refs`` at function entry — keyed on
    ``(entity_type, normalized identity_values)`` via
    ``_normalized_identity_key`` — collapses the duplicates so the
    relationship pass receives a single ref-id per logical entity.

    Ref ids follow a SINGLE monotonic counter across all
    primary_entity_types (no per-type restart) and are only allocated
    AFTER ``_is_valid_upstream_ref`` accepts the ref — so invalid
    identities don't leave gaps in the ref-id sequence.
    """
    from types import SimpleNamespace

    ontology_by_type = {
        e["name"]: e for e in ontology.get("entity_types", [])
    }

    if not hasattr(pass_result, "iter_entities_of_type"):
        return

    # Build dedupe set from refs already collected by previous passes so
    # cross-pass duplicates (e.g. all 5 radar sub-passes naming "Fan Song")
    # collapse to one ref. The same set is updated as this pass emits new
    # refs to also catch within-pass case/whitespace variants.
    seen: set[tuple] = {
        _normalized_identity_key(r.entity_type, r.identity_values)
        for r in upstream_refs.values()
        if hasattr(r, "entity_type") and hasattr(r, "identity_values")
    }

    # Phase 1: collect all yielded instances into scratch accumulators
    # keyed on (entity_type, ontology-ordered identity tuple). Dedup key is
    # ontology-field order, not dict-iteration order — prevents drift on
    # key tuples across Python versions / insertion paths.
    accumulators: dict[tuple, dict] = {}
    allocation_order: list[tuple] = []

    for entity_type in pass_def.primary_entity_types:
        entity_def = ontology_by_type.get(entity_type)
        if entity_def is None:
            # Not in ontology — skip rather than emit a malformed ref.
            continue
        identity_fields = list(entity_def.get("identity_fields") or ())

        for instance in pass_result.iter_entities_of_type(entity_type):
            instance_dict = (
                instance.__dict__
                if hasattr(instance, "__dict__")
                else {}
            )
            identity_values = {
                k: instance_dict.get(k) for k in identity_fields
            }
            identity_tuple = tuple(identity_values[k] for k in identity_fields)
            key = (entity_type, identity_tuple)

            if key not in accumulators:
                scratch = {
                    k: v
                    for k, v in instance_dict.items()
                    if not k.startswith("_") and k not in identity_values and v is not None
                }
                accumulators[key] = {
                    "entity_type": entity_type,
                    "identity_values": identity_values,
                    "scratch": scratch,
                }
                allocation_order.append(key)
            else:
                scratch = accumulators[key]["scratch"]
                for k, v in instance_dict.items():
                    if k.startswith("_") or k in identity_values or v is None:
                        continue
                    if k not in scratch:
                        scratch[k] = v  # first-non-null wins

    # Phase 2: emit refs in allocation order; ref id counter advances only
    # after a successful _is_valid_upstream_ref gate, preserving the
    # no-gaps contract. Cross-pass / within-pass case+whitespace duplicates
    # are skipped via the ``seen`` set (spec §4.5).
    counter = len(upstream_refs) + 1
    for key in allocation_order:
        acc = accumulators[key]
        entity_type = acc["entity_type"]
        identity_values = acc["identity_values"]
        scratch = acc["scratch"]
        dedupe_key = _normalized_identity_key(entity_type, identity_values)
        if dedupe_key in seen:
            continue  # Already represented by a prior pass / earlier in this pass.
        display_label = build_display_label(
            entity_type, identity_values, scratch,
        )
        ref = SimpleNamespace(
            pass_origin=pass_def.name,
            entity_type=entity_type,
            identity_values=identity_values,
            display_label=display_label,
        )
        if not _is_valid_upstream_ref(ref, ontology):
            continue  # Drop refs with missing/empty identity.
        upstream_refs[f"E{counter:03d}"] = ref
        seen.add(dedupe_key)
        counter += 1


def _endpoint_types_for_rel_types(
    ontology: dict, rel_types: list[str],
) -> set[str]:
    """Return the set of entity types that appear as source or target for
    any of the given relationship types in the ontology validation_matrix.

    Used by _select_upstream_refs_for_pass to drop upstream refs whose
    entity_type cannot legally participate in any relationship the
    downstream pass extracts. For system_links (ASSOCIATED_WITH, CUES)
    this resolves to the system-level entity types only."""
    if not rel_types:
        return set()
    wanted = set(rel_types)
    endpoint_types: set[str] = set()
    for row in ontology.get("validation_matrix", []):
        if row.get("relationship") in wanted:
            src = row.get("source")
            tgt = row.get("target")
            if src:
                endpoint_types.add(src)
            if tgt:
                endpoint_types.add(tgt)
    return endpoint_types


def _select_upstream_refs_for_pass(
    pass_def, upstream_refs: dict, ontology: dict,
) -> dict:
    """Filter upstream_refs so the downstream pass only sees refs it can
    legally use: (1) pass_origin in pass_def.depends_on, (2) the ref is
    valid (see _is_valid_upstream_ref), and (3) the ref's entity_type is
    a valid source or target for at least one of
    pass_def.extracted_relationship_types in the ontology validation_matrix.
    Returns a dict ordered by (pass_origin, entity_type, identity) so
    repeat runs produce the same preamble."""
    depends_on = set(getattr(pass_def, "depends_on", None) or [])
    if not depends_on:
        return {}

    rel_types = list(getattr(pass_def, "extracted_relationship_types", None) or [])
    endpoint_types = _endpoint_types_for_rel_types(ontology, rel_types)

    # Precompute the ontology-declared identity_fields order per type so
    # the sort key matches LogicalIdentity's canonical ordering
    # (extraction_merge.py:43-ish: identity_field_names comes straight
    # from entity_def["identity_fields"]). Sorting by sorted(dict.keys())
    # would diverge from that canonical order on any multi-field
    # identity, which means the LLM preamble and the merge identity
    # tuple could disagree on which value goes first.
    identity_fields_by_type = {
        e["name"]: tuple(e.get("identity_fields") or ())
        for e in ontology.get("entity_types", [])
    }

    eligible = []
    for ref_id, ref in upstream_refs.items():
        if getattr(ref, "pass_origin", None) not in depends_on:
            continue
        if not _is_valid_upstream_ref(ref, ontology):
            continue
        # When the downstream pass extracts relationships, the ref's type
        # must be legal for at least one of them. If the pass declares
        # no extracted_relationship_types, keep all depends_on refs.
        if endpoint_types and ref.entity_type not in endpoint_types:
            continue
        eligible.append((ref_id, ref))

    def _sort_key(item):
        _ref_id, ref = item
        identity_values = getattr(ref, "identity_values", {}) or {}
        # Use ontology-declared identity_fields order (same as
        # LogicalIdentity.identity_tuple), NOT sorted(dict.keys()).
        fields = identity_fields_by_type.get(ref.entity_type, ())
        identity_tuple = tuple(identity_values.get(k) for k in fields)
        return (ref.pass_origin, ref.entity_type, identity_tuple)

    eligible.sort(key=_sort_key)
    return {ref_id: ref for ref_id, ref in eligible}


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
@guard_stage_run("prepare_document",
    lifecycle=True,
    next_stage="detect_and_translate",
    next_task="app.workers.pipeline.detect_and_translate")
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
            "text/plain",
        }
        if mime_type not in _DOCLING_MIMES:
            logger.info("prepare_document: %s not supported by Docling (mime=%s), using legacy extraction", document_id, mime_type)
            _legacy_extract(db, document_id, doc, file_bytes)
            db.commit()

            # Aggregate extracted text once — used for BOTH the markdown artifact
            # and the stub docling_document.json below.
            fallback_md = ""
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
            except Exception:
                logger.exception(
                    "prepare_document: failed to aggregate legacy text for %s",
                    document_id,
                )

            _fb_base = f"artifacts/{document_id}"
            from app.services.storage import upload_bytes_sync

            # Markdown is best-effort — keep the swallow so a MinIO blip doesn't
            # fail an otherwise-successful legacy ingest.
            if fallback_md:
                try:
                    upload_bytes_sync(
                        fallback_md.encode("utf-8"),
                        settings.minio_bucket_derived,
                        f"{_fb_base}/docling_document.md",
                        content_type="text/markdown; charset=utf-8",
                    )
                    logger.info(
                        "prepare_document: persisted legacy markdown for %s (%d chars)",
                        document_id, len(fallback_md),
                    )
                except Exception as _fb_err:
                    logger.warning(
                        "prepare_document: failed to persist legacy markdown for %s: %s",
                        document_id, _fb_err,
                    )

            # Stub docling_document.json is REQUIRED by derive_document_anchors
            # and other downstream stages (they call DoclingDocument.model_validate
            # on the fetched JSON). Always write, even when fallback_md is empty.
            # Do NOT swallow failures — let them propagate to guard_stage_run.
            import json as _json
            stub = _build_legacy_docling_document_json(document_id, fallback_md or "")
            upload_bytes_sync(
                _json.dumps(stub, ensure_ascii=False).encode("utf-8"),
                settings.minio_bucket_derived,
                f"{_fb_base}/docling_document.json",
                content_type="application/json; charset=utf-8",
            )
            logger.info(
                "prepare_document: persisted legacy docling_document.json stub for %s",
                document_id,
            )

            _update_stage_run(
                db, run_id, "prepare_document", "COMPLETE",
                attempt=self.request.retries + 1,
                metrics={
                    "fallback": True,
                    "reason": "unsupported_format",
                    "markdown_chars": len(fallback_md.encode("utf-8")) if fallback_md else 0,
                },
            )
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
        markdown_chars = 0
        try:
            from app.services.storage import upload_bytes_sync
            _docling_base = f"artifacts/{document_id}"
            if result.markdown:
                _md_bytes = _normalize_text(result.markdown).encode("utf-8")
                markdown_chars = len(_md_bytes)
                upload_bytes_sync(
                    _md_bytes,
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
                "markdown_chars": markdown_chars,
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
    soft_time_limit=settings.doc_analysis_soft_time_limit,
    time_limit=settings.doc_analysis_time_limit,
    queue="ingest",
)
@guard_stage_run("derive_document_metadata",
    lifecycle=True,
    next_stage="purge_document_derivations",
    next_task="app.workers.pipeline.purge_document_derivations")
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

        # Image-only / handwritten-only docs have no markdown to summarize;
        # skip without attempting MinIO download.
        from app.models.ingest import DocumentElement as _DE
        from sqlalchemy import select as _sa_sel, func as _sa_func
        text_element_count = db.execute(
            _sa_sel(_sa_func.count(_DE.id)).where(
                _DE.document_id == uuid.UUID(document_id),
                _DE.element_type.in_(("text", "heading", "table", "equation")),
                _DE.content_text.isnot(None),
            )
        ).scalar() or 0
        if text_element_count == 0:
            logger.info(
                "derive_document_metadata: no text elements for %s — "
                "skipping (image-only / handwritten-only doc; no markdown "
                "to summarize)",
                document_id,
            )
            if run_id:
                _update_stage_run(db, run_id, "derive_document_metadata", "COMPLETE",
                                  attempt=self.request.retries + 1,
                                  metrics={"skipped": True, "reason": "no_text_elements",
                                           "text_element_count": 0})
                db.commit()
            return {"stage": "derive_document_metadata", "status": "skipped",
                    "reason": "no_text_elements"}

        # Load markdown from MinIO
        from app.services.storage import download_bytes_sync
        base_key = f"artifacts/{document_id}"
        bucket = settings.minio_bucket_derived

        # Retry the markdown read — observed transient swallow on a freshly-
        # written object. Exception type is captured so a recurring failure
        # mode (NoSuchKey vs ConnectError) is visible.
        original_md = None
        _md_last_exc: Exception | None = None
        for _md_attempt in range(3):
            try:
                original_md = download_bytes_sync(bucket, f"{base_key}/docling_document.md").decode("utf-8")
                break
            except Exception as _md_exc:
                _md_last_exc = _md_exc
                if _md_attempt < 2:
                    import time as _time
                    _time.sleep(2.0)
        if original_md is None:
            logger.warning(
                "derive_document_metadata: markdown download failed for %s "
                "after 3 attempts (last_exc=%s) — doc has %d text elements; "
                "escalating to PARTIAL_COMPLETE",
                document_id,
                type(_md_last_exc).__name__ if _md_last_exc else "None",
                text_element_count,
            )
            _update_document_status(
                document_id, STATUS_PARTIAL_COMPLETE,
                stage="derive_document_metadata",
                error="no_markdown_with_text_elements",
            )
            if run_id:
                _update_stage_run(db, run_id, "derive_document_metadata", "COMPLETE",
                                  attempt=self.request.retries + 1,
                                  metrics={"skipped": True,
                                           "reason": "no_markdown_with_text_elements",
                                           "text_element_count": text_element_count})
                db.commit()
            return {"stage": "derive_document_metadata", "status": "skipped",
                    "reason": "no_markdown_with_text_elements"}

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
@guard_stage_run("detect_and_translate",
    lifecycle=True,
    next_stage="derive_document_metadata",
    next_task="app.workers.pipeline.derive_document_metadata")
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
            # Distinguish image-only / no-text docs (legitimate skip) from
            # docs that should have text but don't (real bug). Image-only
            # docs have element_type='image' rows but no text-eligible ones,
            # and that's the correct shape for them — they go through picture
            # description + image embedding instead. Only escalate when the
            # doc has NO elements of any kind (truly empty extraction).
            from app.models.ingest import DocumentElement as _DE
            from sqlalchemy import select as _sa_sel, func as _sa_func
            any_element_count = db.execute(
                _sa_sel(_sa_func.count(_DE.id)).where(
                    _DE.document_id == uuid.UUID(document_id),
                )
            ).scalar() or 0
            if any_element_count == 0:
                logger.warning(
                    "detect_and_translate: no elements at all for %s — "
                    "escalating to PARTIAL_COMPLETE (extraction produced nothing)",
                    document_id,
                )
                _update_document_status(
                    document_id, STATUS_PARTIAL_COMPLETE,
                    stage="detect_and_translate", error="no_elements_at_all",
                )
            else:
                logger.info(
                    "detect_and_translate: no text-eligible elements for %s "
                    "(has %d non-text elements) — expected for image-only docs",
                    document_id, any_element_count,
                )
            if run_id:
                _update_stage_run(db, run_id, "detect_and_translate", "COMPLETE",
                                  attempt=self.request.retries + 1,
                                  metrics={"skipped": True, "reason": "no_elements",
                                           "any_element_count": any_element_count})
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
        language_confidences = detection.get("language_confidences", {})

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

        # Guard against langdetect-noise: if every flagged element failed langdetect
        # classification, the non-Latin regex almost certainly false-fired on OCR
        # artefacts. Sending that garbage to the LLM wastes a full translation
        # timeout per element with no useful result.
        unknown_count = language_confidences.get("unknown", 0)
        if unknown_count and unknown_count == non_english_count:
            _merge_doc_metadata({
                "detected_language": "unknown",
                "has_translation": False,
                "translation_skipped_reason": "langdetect_no_match",
            })
            db.commit()
            if run_id:
                _update_stage_run(db, run_id, "detect_and_translate", "COMPLETE",
                                  attempt=self.request.retries + 1,
                                  metrics={
                                      "detected_language": "unknown",
                                      "total_elements": total_elements,
                                      "non_english_elements": non_english_count,
                                      "elements_translated": 0,
                                      "skipped_reason": "langdetect_no_match",
                                  })
                db.commit()
            logger.warning(
                "detect_and_translate: document_id=%s %d elements flagged by non-Latin "
                "regex but none classified by langdetect — skipping as probable OCR noise",
                document_id, non_english_count,
            )
            return {
                "stage": "detect_and_translate",
                "status": "skipped",
                "reason": "langdetect_no_match",
                "flagged": non_english_count,
            }

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
    except SoftTimeLimitExceeded as exc:
        logger.warning(
            "detect_and_translate: soft time limit for %s — retrying via Celery",
            document_id,
        )
        if run_id:
            try:
                _update_stage_run(db, run_id, "detect_and_translate", "FAILED",
                                  attempt=self.request.retries + 1, error="soft time limit exceeded")
                db.commit()
            except Exception:
                logger.exception("detect_and_translate: stage_run FAILED write also failed")
        raise self.retry(exc=exc)
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
@guard_stage_run("derive_picture_descriptions",
    lifecycle=True,
    next_stage="derive_text_embeddings",
    next_task="app.workers.pipeline.derive_text_chunks_and_embeddings")
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
            max_workers = min(settings.picture_desc_concurrency, len(describable))
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
            "derive_picture_descriptions: soft time limit for %s — retrying via Celery",
            document_id,
        )
        if run_id:
            try:
                _update_stage_run(db, run_id, "derive_picture_descriptions", "FAILED", attempt=self.request.retries + 1, error="soft time limit exceeded")
                db.commit()
            except Exception:
                logger.exception("derive_picture_descriptions: stage_run FAILED write also failed")
        raise self.retry(exc=exc)
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
@guard_stage_run("purge_document_derivations",
    lifecycle=True,
    next_stage="derive_picture_descriptions",
    next_task="app.workers.pipeline.derive_picture_descriptions")
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


def _build_native_chunk_meta(
    chunk_idx: int,
    chunk,
    document_id: str,
    model_version: str,
) -> dict:
    """Build per-chunk metadata for the native HybridChunker path.

    Carries evidence_ids/self_refs/page_numbers so embedding chunks share
    the same source-unit lineage as graph-extraction chunks (independent
    boundaries, identical lineage shape).
    """
    self_refs: list[str] = []
    page_numbers: set[int] = set()
    for item in (getattr(getattr(chunk, "meta", None), "doc_items", None) or []):
        ref = getattr(item, "self_ref", None)
        if isinstance(ref, str) and ref:
            self_refs.append(ref)
        for p in (getattr(item, "prov", None) or []):
            pn = getattr(p, "page_no", None)
            if pn is not None:
                page_numbers.add(pn)
    # HybridChunker exposes the heading hierarchy for each chunk via
    # chunk.meta.headings (list of section header strings ordered outermost
    # → innermost). Joining with " / " yields the section_path form used
    # by retrieval (section filter / section ranking / display crumbs).
    headings: list[str] = []
    for h in (getattr(getattr(chunk, "meta", None), "headings", None) or []):
        if isinstance(h, str) and h.strip():
            headings.append(h.strip())
    from app.services.docling_anchors import _build_section_path_string
    section_path = _build_section_path_string(tuple(headings))
    chunk_key = hashlib.sha256(
        f"{document_id}:native:{chunk_idx}:{model_version}".encode()
    ).hexdigest()
    chunk_id = uuid.UUID(hashlib.md5(chunk_key.encode()).hexdigest())
    return {
        "chunk_id": chunk_id,
        "chunk_index": chunk_idx,
        "page_number": min(page_numbers) if page_numbers else None,
        "page_numbers": sorted(page_numbers),
        "modality": "text",
        "self_refs": self_refs,
        # evidence_ids is an alias for self_refs at chunk-creation time; the LLM
        # (per ir_normalizer._attach_evidence_to_prov) filters this down per-node.
        "evidence_ids": list(self_refs),
        "document_id": document_id,
        "section_path": section_path,
        "headings": headings,
    }


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60, queue="embed",
                 soft_time_limit=settings.embed_soft_time_limit,
                 time_limit=settings.embed_time_limit)
@guard_stage_run("derive_text_embeddings",
    lifecycle=True,
    next_stage="derive_image_embeddings",
    next_task="app.workers.pipeline.derive_image_embeddings")
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

                    tok = AutoTokenizer.from_pretrained(settings.embedding_chunk_tokenizer_model)
                    hf_tok = HuggingFaceTokenizer(tokenizer=tok, max_tokens=settings.embedding_chunk_max_tokens)
                    chunker = HybridChunker(
                        tokenizer=hf_tok,
                        merge_peers=True,
                        repeat_table_header=True,
                        omit_header_on_overflow=False,
                        always_emit_headings=False,
                    )
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

                all_texts.append(chunk_text)
                all_chunk_metas.append(
                    _build_native_chunk_meta(
                        chunk_idx=chunk_idx,
                        chunk=chunk,
                        document_id=document_id,
                        model_version=model_version,
                    )
                )

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
                            "page_numbers": meta["page_numbers"],
                            "self_refs": meta["self_refs"],
                            "evidence_ids": meta["evidence_ids"],
                            "section_path": meta.get("section_path"),
                            "headings": meta.get("headings", []),
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
                max_chunk_tokens=settings.embedding_chunk_max_tokens,
                overlap_tokens=settings.embedding_chunk_overlap_tokens,
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

        # ── Pass 2: Image descriptions ──────────────────────────────────
        # One chunk per image element. The prior behavior split descriptions
        # into sections via split_description_sections and embedded each
        # section as a separate TextChunk — that matched the old multi-section
        # VLM prompt format. The current prompt produces a single-blob
        # description per image, so we embed the full text as one unit.
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
            if not desc_text or not desc_text.strip():
                continue

            # Single chunk per image description. sec_idx=0 kept in the key so
            # existing chunk_id derivation for already-ingested docs stays
            # backward-compatible.
            sec_idx = 0
            chunk_index = 100000 + img_elem.element_order * 100 + sec_idx
            uid_str = str(img_elem.element_uid) if img_elem.element_uid else str(img_elem.id)
            chunk_key = hashlib.sha256(
                f"{document_id}:{uid_str}:{sec_idx}:{model_version}".encode()
            ).hexdigest()
            chunk_id = uuid.UUID(hashlib.md5(chunk_key.encode()).hexdigest())

            # Synthesize a page-scoped section_path so image-description
            # chunks are filterable by document region. Uses the canonical
            # `" > "` separator from _build_section_path_string so all
            # TextChunk section_path values share one convention.
            from app.services.docling_anchors import _build_section_path_string
            if img_elem.section_path:
                img_section_path = img_elem.section_path
            elif img_elem.page_number is not None:
                img_section_path = _build_section_path_string(
                    ("Image Descriptions", f"page {img_elem.page_number}")
                )
            else:
                img_section_path = "Image Descriptions"
            img_desc_texts.append(desc_text)
            img_desc_chunk_metas.append({
                "chunk_id": chunk_id,
                "artifact_id": img_elem.artifact_id,
                "document_id": uuid.UUID(document_id),
                "chunk_index": chunk_index,
                "page_number": img_elem.page_number,
                "section_text": desc_text,
                "section_path": img_section_path,
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
                        "section_path": meta["section_path"],
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
    except SoftTimeLimitExceeded as exc:
        logger.warning("derive_text_chunks_and_embeddings: soft time limit for %s — retrying via Celery", document_id)
        db.rollback()
        if run_id:
            _update_stage_run(db, run_id, "derive_text_embeddings", "FAILED",
                              attempt=self.request.retries + 1, error="soft time limit exceeded")
            db.commit()
        raise self.retry(exc=exc)
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
@guard_stage_run("derive_image_embeddings",
    lifecycle=True,
    next_stage="derive_document_anchors",
    next_task="app.workers.pipeline.derive_document_anchors")
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

        # Source-bytes backstop: when an image-MIME doc has no
        # element_type='image' row (Docling parsed only text fragments),
        # the source JPEG never reaches CLIP. Synthesize an Artifact +
        # element pointing at the source bytes so the per-element loop
        # below embeds it.
        if not elements and doc_obj is not None \
                and (doc_obj.mime_type or "").startswith("image/") \
                and doc_obj.storage_key:
            from app.models.ingest import Artifact
            from types import SimpleNamespace
            backstop_artifact_id = _deterministic_artifact_id(
                document_id, "source-image"
            )
            db.execute(
                pg_insert(Artifact).values(
                    id=backstop_artifact_id,
                    document_id=uuid.UUID(document_id),
                    artifact_type="image",
                    storage_bucket=doc_obj.storage_bucket,
                    storage_key=doc_obj.storage_key,
                    classification=doc_classification,
                ).on_conflict_do_nothing(index_elements=["id"])
            )
            db.flush()
            elements = [SimpleNamespace(
                element_uid=f"source-image-{document_id}",
                element_order=0,
                element_type="image",
                page_number=None,
                bounding_box=None,
                content_text=None,
                storage_bucket=doc_obj.storage_bucket,
                storage_key=doc_obj.storage_key,
                artifact_id=backstop_artifact_id,
            )]
            logger.info(
                "derive_image_embeddings: source-bytes backstop for %s "
                "(image-MIME=%s, artifact_id=%s)",
                document_id, doc_obj.mime_type, backstop_artifact_id,
            )

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
                            "artifact_id": str(elem.artifact_id) if elem.artifact_id is not None else "",
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
    except SoftTimeLimitExceeded as exc:
        logger.warning("derive_image_embeddings: soft time limit for %s — retrying via Celery", document_id)
        db.rollback()
        if run_id:
            _update_stage_run(db, run_id, "derive_image_embeddings", "FAILED",
                              attempt=self.request.retries + 1, error="soft time limit exceeded")
            db.commit()
        raise self.retry(exc=exc)
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
# derive_document_anchors (D-4) — deterministic DOCUMENT/SECTION/FIGURE/
# TABLE emission via the Docling anchor walker (spec §3.3). Runs between
# derive_image_embeddings and derive_ontology_graph.
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True, max_retries=1, default_retry_delay=30, queue="graph",
    soft_time_limit=settings.finalize_soft_time_limit,
    time_limit=settings.finalize_time_limit,
)
@guard_stage_run("derive_document_anchors",
    lifecycle=True,
    next_stage="derive_ontology_graph",
    next_task="app.workers.pipeline.derive_ontology_graph")
def derive_document_anchors(self, document_id: str, run_id: str | None = None) -> dict:
    """Emit ontology DOCUMENT / SECTION / FIGURE / TABLE vertices and
    their structural edges (HAS_SECTION / HAS_FIGURE / HAS_TABLE /
    CHILD_OF) from the persisted DoclingDocument.

    Writes go through ``upsert_nodes_batch_sync`` for vertices and
    ``create_structural_edge_sync`` for edges — NOT the ontology
    relationship path. Spec §3.1–§3.5 and §5.6 Phase 2/3.
    """
    # Import walker lazily — docling_anchors imports heavyweight
    # ontology_bundles.air_defense_v3.entities at module load time.
    from app.services import docling_anchors as _docling_anchors
    from app.services.graph_store import NodeRecord, ProvenanceMetadata

    logger.info(
        "derive_document_anchors: document_id=%s run_id=%s",
        document_id, run_id,
    )
    _update_document_status(
        document_id, STATUS_PROCESSING, stage="derive_document_anchors",
    )

    db = _get_db()
    try:
        if not run_id:
            run_id = _get_pipeline_run_id(db, document_id)
        # Orphaned-task guard: if the document was hard-deleted while this
        # task was queued, MinIO artefacts and the parent pipeline_run are
        # gone too. Exit cleanly to avoid a NoSuchKey on the docling JSON
        # download and a follow-on FK violation when writing stage_runs.
        # Match the string-comparison pattern used at the existing
        # Document lookup below so SQLAlchemy handles UUID coercion.
        from app.models.ingest import Document as _DocModel
        _existing_doc = db.query(_DocModel).filter(
            _DocModel.id == document_id
        ).first()
        if _existing_doc is None:
            logger.warning(
                "derive_document_anchors: document %s not found "
                "(likely deleted); skipping orphaned task",
                document_id,
            )
            return {
                "stage": "derive_document_anchors",
                "status": "skipped",
                "reason": "orphaned_document",
            }
        if run_id:
            _update_stage_run(
                db, run_id, "derive_document_anchors", "RUNNING",
                attempt=self.request.retries + 1,
            )
            db.commit()

        doc_json = _build_docling_document_json(document_id)

        # Ontology dict is a no-op for walker-sourced passes (model_config
        # carries graph_id_fields). We still load + pass it through so the
        # component-branch inside _build_logical_identity stays compatible
        # if a future component entity enters the anchor set.
        # NOTE: load_ontology lives in ontology_templates, not ontology_bundles.
        # The previous try/except-Exception swallowed the ImportError from the
        # wrong import path and silently fell back to {} for every doc — dormant
        # because the walker doesn't actually need ontology fields today, but
        # real enough that future component entities would misbehave.
        try:
            from app.services.ontology_templates import load_ontology
            ontology = load_ontology()
        except Exception:
            ontology = {}

        # §4.4 — propagate Document SQL row's storage_key into the walker
        # so DocumentEntity.storage_key lands on the graph without a
        # downstream SQL join.
        from app.models.ingest import Document
        document_row = db.query(Document).filter(Document.id == document_id).first()
        source_storage_key = document_row.storage_key if document_row is not None else None

        merged = _docling_anchors.walk(
            doc_json, document_id, run_id, ontology,
            source_storage_key=source_storage_key,
        )

        # --- Vertex upserts ------------------------------------------------
        node_records = [
            NodeRecord(
                entity_type=e.identity.entity_type,
                identity_fields=e.identity.as_upsert_identity_dict(),
                name=e.display_label,
                properties=e.properties,
                extraction_confidence=e.confidence,
            )
            for e in merged.entities
        ]
        provenance = _build_provenance_envelope(
            document_id, run_id, merged.entities, db,
        )
        graph_store = get_graph_store()
        rids = graph_store.upsert_nodes_batch_sync(node_records, provenance)

        # --- Identity → RID bridge ----------------------------------------
        identity_to_rid = dict(zip(
            (e.identity for e in merged.entities), rids, strict=True,
        ))

        # --- Structural edges ---------------------------------------------
        # Guard against empty RIDs. _extract_rids pads missing upsert
        # results with "" (arcadedb_graph.py:287-300) so a silent upsert
        # failure on one entity would otherwise produce a malformed
        # CREATE EDGE ... FROM {rid} TO  SET ... that ArcadeDB rejects
        # with 500 and kills the whole task. Skip the bad edge with a
        # warning instead so the rest of the document still lands; the
        # underlying identity corruption (usually empty self_ref from a
        # malformed DoclingDocument) surfaces in the log.
        edges_skipped_empty_rid = 0
        for edge in merged.edges:
            from_rid = identity_to_rid[edge.from_identity]
            to_rid = identity_to_rid[edge.to_identity]
            if not from_rid or not to_rid:
                edges_skipped_empty_rid += 1
                logger.warning(
                    "derive_document_anchors: skipping %s edge with empty "
                    "RID (from=%r to=%r). Source identity=%s target identity=%s.",
                    edge.rel_type, from_rid, to_rid,
                    edge.from_identity, edge.to_identity,
                )
                continue
            graph_store.create_structural_edge_sync(
                from_rid, to_rid, edge.rel_type,
                properties={
                    "document_id": str(document_id),
                    "pipeline_run_id": str(run_id) if run_id is not None else None,
                    "extraction_confidence": edge.confidence,
                    "source": "docling_anchors",
                },
            )

        # --- Metrics ------------------------------------------------------
        section_count = sum(
            1 for e in merged.entities if e.identity.entity_type == "SECTION"
        )
        figure_count = sum(
            1 for e in merged.entities if e.identity.entity_type == "FIGURE"
        )
        table_count = sum(
            1 for e in merged.entities if e.identity.entity_type == "TABLE"
        )
        image_count = sum(
            1 for e in merged.entities if e.identity.entity_type == "IMAGE"
        )
        text_block_count = sum(
            1 for e in merged.entities if e.identity.entity_type == "TEXT_BLOCK"
        )
        document_ontology_emitted = any(
            e.identity.entity_type == "DOCUMENT" for e in merged.entities
        )
        fallback_fired = any(
            e.identity.entity_type == "SECTION"
            and e.identity.identity_tuple == ("0",)
            for e in merged.entities
        )
        metrics = {
            "section_count": section_count,
            "figure_count": figure_count,
            "table_count": table_count,
            "image_count": image_count,
            "text_block_count": text_block_count,
            "document_ontology_emitted": document_ontology_emitted,
            "fallback_fired": fallback_fired,
            "edge_count": len(merged.edges),
            "edges_skipped_empty_rid": edges_skipped_empty_rid,
        }

        if run_id:
            _update_stage_run(
                db, run_id, "derive_document_anchors", "COMPLETE",
                attempt=self.request.retries + 1, metrics=metrics,
            )
            db.commit()

        # TODO #87: when the synthetic-section fallback fires AND every other
        # per-type count is zero, the document has no usable anchored content.
        # Graph extraction will run against an empty anchor set and any LLM-
        # produced entities will reference null anchors. Flag the doc as
        # degraded so the UI surfaces the regression and downstream operators
        # can filter.
        empty_anchor_set = (
            fallback_fired
            and figure_count == 0
            and table_count == 0
            and image_count == 0
            and text_block_count == 0
        )
        if empty_anchor_set:
            # No PARTIAL escalation here — the empty_anchor_set short-circuit
            # in derive_ontology_graph handles graf cleanly, and
            # finalize_document derives the final status from stage outcomes.
            logger.warning(
                "derive_document_anchors: empty-anchor-set fallback fired "
                "for %s (counts all 0) — derive_ontology_graph short-"
                "circuit will skip extraction; doc may still finalize "
                "COMPLETE if upstream stages succeeded",
                document_id,
            )

        logger.info(
            "derive_document_anchors: document_id=%s sections=%d figures=%d "
            "tables=%d images=%d text_blocks=%d document_emitted=%s edges=%d "
            "fallback_fired=%s",
            document_id, section_count, figure_count, table_count,
            image_count, text_block_count, document_ontology_emitted, len(merged.edges),
            fallback_fired,
        )

        return {"stage": "derive_document_anchors", "status": "ok", **metrics}

    except CeleryRetry:
        raise
    except SoftTimeLimitExceeded as exc:
        logger.warning(
            "derive_document_anchors: soft time limit for %s — retrying via Celery", document_id,
        )
        db.rollback()
        if run_id:
            _update_stage_run(
                db, run_id, "derive_document_anchors", "FAILED",
                attempt=self.request.retries + 1,
                error="soft time limit exceeded",
            )
            db.commit()
        raise self.retry(exc=exc)
    except Exception as exc:
        logger.error(
            "derive_document_anchors failed for %s: %s", document_id, exc,
        )
        db.rollback()
        if run_id:
            _update_stage_run(
                db, run_id, "derive_document_anchors", "FAILED",
                attempt=self.request.retries + 1, error=str(exc),
            )
            db.commit()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Per-pass Celery fan-in — Task 5 helpers + derive_ontology_graph_pass task
# ---------------------------------------------------------------------------
# Import note: pass_outputs_store and run_phase_dispatch are imported here at
# module level so the task and helpers can reference them directly. Placing the
# imports here (after the main pipeline body) avoids circular-import risk from
# placing them at the top of a very large module.
from app.services.pass_outputs_store import (  # noqa: E402
    save_pass_output,
    is_pass_already_resolved,
    count_terminal_passes,
    load_pass_output,
    load_completed_pass_outputs,
)
from app.services.run_phase_dispatch import (  # noqa: E402
    claim_phase,
    mark_phase_dispatched,
    mark_phase_terminal,
    read_phase_state,
    is_run_cancelled,
)


def _phase_key(pass_name: str) -> str:
    """Map pass_name to the dispatched_phases JSONB key.

    Entity passes use the ``entity_pass_<name>`` prefix; ``system_links`` and
    ``merge`` are stored as top-level keys (no prefix) because they aren't
    per-entity extraction passes.
    """
    if pass_name in ("system_links", "merge"):
        return pass_name
    return f"entity_pass_{pass_name}"


def _retry_delay(retries: int) -> int:
    """Celery countdown-compatible delay: 30s × 2^retries, capped at 300s.

    Sequence: retries=0 → 30s; retries=1 → 60s; retries=2 → 120s;
    retries=3 → 240s; retries=4+ → 300s (cap).

    Note: Celery's ``self.request.retries`` starts at 0 on the first retry,
    so we use 2^retries directly (not 2^(retries-1) like _backoff).
    """
    return min(30 * (2 ** retries), 300)


def _save_terminal_pass_output(
    db,
    *,
    run_id: str,
    stage_run_id,
    pass_name: str,
    attempt: int,
    outcome: "PassAttemptOutcome",
    override_status: str | None = None,
    override_diagnostics_extra: dict | None = None,
) -> None:
    """Write the single terminal ``pipeline_pass_outputs`` row for this pass.

    Upserts by (run_id, pass_name) — overwrites any prior terminal write
    (defensive; a pass should only terminalize once in normal operation, but
    the upsert is safe if it does).

    ``override_status`` lets the caller force ``FAILED`` even when
    ``outcome.execution_status`` says otherwise (used for retry-exhaustion).
    ``override_diagnostics_extra`` is merged into the diagnostics dict pulled
    from ``raw_response_payload``; used to add ``{"retry_exhausted": True}``.

    IMPORTANT: counts keys from ``_count_pass_output`` are:
      ``primary_entities_extracted``, ``bridge_entities_extracted``,
      ``relationships_extracted``, ``relationships_rejected``
    NOT the shorter forms ``primary_entities`` / ``bridge_entities``.
    """
    diagnostics = (outcome.raw_response_payload or {}).get("diagnostics", {}) or {}
    if override_diagnostics_extra:
        diagnostics = {**diagnostics, **override_diagnostics_extra}
    save_pass_output(
        db,
        pipeline_run_id=run_id,
        stage_run_id=stage_run_id,
        pass_name=pass_name,
        attempt=attempt,
        execution_status=override_status or outcome.execution_status,
        skip_reason=outcome.skip_reason,
        yield_status=outcome.yield_status,
        extract_pass_response=outcome.raw_response_payload or {},
        primary_entities_extracted=(outcome.counts or {}).get("primary_entities_extracted", 0),
        bridge_entities_extracted=(outcome.counts or {}).get("bridge_entities_extracted", 0),
        relationships_extracted=(outcome.counts or {}).get("relationships_extracted", 0),
        relationships_rejected=(outcome.counts or {}).get("relationships_rejected", 0),
        diagnostics=diagnostics,
        field_provenance=(outcome.raw_response_payload or {}).get("field_provenance", []),
    )


def _rehydrate_upstream_refs_from_persisted_passes(
    db,
    run_id: str,
    pass_def,
    manifest,
    ontology: dict,
    document_id: str,
) -> dict:
    """Build the upstream_refs dict for a ``document_plus_entity_refs`` pass.

    The per-pass Celery task runs in isolation — it has no in-memory
    ``pass_results`` dict from prior passes.  For passes like ``system_links``
    that declare ``input_mode='document_plus_entity_refs'`` and non-empty
    ``depends_on``, we must reconstruct upstream_refs by loading each
    dependency's persisted row from ``pipeline_pass_outputs`` and re-parsing
    the stored response JSON through ``_parse_pass_response``.

    Algorithm:
    1. Early-exit if the pass doesn't need upstream refs.
    2. For each dependency pass name in ``pass_def.depends_on``:
       a. Load the persisted row.  Skip if missing or not COMPLETE.
       b. Re-parse the stored ``extract_pass_response_json`` through
          ``_parse_pass_response`` to get a live PassResult object.
       c. Attach ``pre_merge_walk`` via ``_build_pre_merge_walk_summary`` so
          ``_extend_upstream_refs`` sees entity counts.
       d. Call ``_extend_upstream_refs`` to merge the dependency's entities
          into the accumulating ``upstream_refs`` dict.
    3. Return the accumulated dict (may be empty if all deps were missing/FAILED).

    Contract: a ``system_links`` task running at time T sees the same upstream
    refs it would have seen if dependencies A, B, C had run in-process at T-1.
    """
    upstream_refs: dict = {}

    # Early-exit: only document_plus_entity_refs passes with dependencies need this.
    if getattr(pass_def, "input_mode", None) != "document_plus_entity_refs":
        return upstream_refs
    depends_on = list(getattr(pass_def, "depends_on", None) or [])
    if not depends_on:
        return upstream_refs

    # Build a fast lookup of dep_pass_name → dep_pass_def from the manifest.
    manifest_by_name = {p.name: p for p in manifest.passes}

    for dep_pass_name in depends_on:
        dep_row = load_pass_output(db, run_id, dep_pass_name)
        if dep_row is None:
            logger.warning(
                "_rehydrate_upstream_refs: dependency '%s' has no terminal row "
                "(run_id=%s) — skipping",
                dep_pass_name, run_id,
            )
            continue
        if dep_row.execution_status != "COMPLETE":
            logger.warning(
                "_rehydrate_upstream_refs: dependency '%s' is %s, not COMPLETE "
                "(run_id=%s) — skipping",
                dep_pass_name, dep_row.execution_status, run_id,
            )
            continue

        dep_pass_def = manifest_by_name.get(dep_pass_name)
        if dep_pass_def is None:
            logger.warning(
                "_rehydrate_upstream_refs: dependency '%s' not in manifest "
                "(run_id=%s) — skipping",
                dep_pass_name, run_id,
            )
            continue

        try:
            dep_pass_result = _parse_pass_response(
                dep_row.extract_pass_response_json, dep_pass_def, manifest
            )
        except PassTerminal as exc:
            logger.warning(
                "_rehydrate_upstream_refs: re-parsing dependency '%s' raised "
                "PassTerminal: %s (run_id=%s) — skipping",
                dep_pass_name, exc, run_id,
            )
            continue

        # Attach pre_merge_walk so _extend_upstream_refs can walk entities.
        dep_pass_result.pre_merge_walk = _build_pre_merge_walk_summary(
            dep_pass_result, dep_pass_def, ontology, document_id,
        )
        _extend_upstream_refs(upstream_refs, dep_pass_result, dep_pass_def, ontology)

    return upstream_refs


def _update_summary_stage_run(
    db,
    run_id: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    """Update the summary StageRun for ``derive_ontology_graph`` (pass_name IS NULL).

    The summary row is created by the Task-8 dispatcher with status=RUNNING.
    This helper advances it to a terminal status (COMPLETE or FAILED) at the
    end of the fan-in lifecycle — called by ``derive_ontology_graph_merge``
    on success (Task 6) or by ``derive_ontology_graph_pass`` on required-pass
    terminal failure (this task).

    Design choice: resilient to a missing summary row (logs a warning and
    returns) so that tests that don't pre-seed the summary row still pass.
    Task 8's dispatcher will always create the row in production; the
    resilience here is defensive and avoids coupling Task 5 tests to Task 8
    fixtures.
    """
    import datetime
    from app.models.ingest import StageRun
    from sqlalchemy import select

    row = db.execute(
        select(StageRun).where(
            StageRun.pipeline_run_id == uuid.UUID(str(run_id)),
            StageRun.stage_name == "derive_ontology_graph",
            StageRun.pass_name.is_(None),
        )
    ).scalars().first()

    if row is None:
        logger.warning(
            "_update_summary_stage_run: no summary StageRun found "
            "(run_id=%s stage=derive_ontology_graph pass_name=NULL) — skipping update",
            run_id,
        )
        return

    row.status = status
    if hasattr(row, "execution_status"):
        row.execution_status = status
    row.finished_at = datetime.datetime.now(datetime.timezone.utc)
    if error and status in ("FAILED", "COMPLETE"):
        row.error_message = error


def _try_advance_phase(db, document_id: str, run_id: str) -> None:
    """Decide whether to dispatch the next entity pass, system_links, or merge.

    Called by each finishing pass (COMPLETE / SKIPPED / FAILED-optional) so
    the fan-in automatically advances to the next stage without a separate
    coordinator task.

    Three mutually exclusive branches (return after the first successful
    dispatch so we do exactly one dispatch per finisher):

    1. If in-flight entity passes < concurrency cap → dispatch the next
       not-yet-dispatched entity pass.
    2. If all entity passes resolved AND the bundle defines system_links AND
       system_links not yet dispatched → dispatch system_links.
    3. If system_links resolved (or bundle has no system_links and all entity
       passes are terminal) AND merge not yet dispatched → dispatch merge.

    Forward reference: ``derive_ontology_graph_merge`` is defined later in
    this module (Task 6).  We use ``celery_app.send_task(...)`` with the
    registered task name rather than a direct ``.delay()`` call to avoid a
    Python forward-reference at function-definition time.  The name is
    resolved at call time via Celery's task registry — so as long as the
    module is fully loaded before this branch runs, the dispatch succeeds.

    Note: ``mark_phase_dispatched`` and friends use raw SQL (``jsonb_set``)
    that bypasses the ORM identity map.  We expire ``dispatched_phases``
    before reading it so the in_flight count reflects the latest DB state and
    cannot underreport, which would allow dispatching above the concurrency cap.
    """
    from app.models.ingest import PipelineRun
    run = db.get(PipelineRun, uuid.UUID(str(run_id)))
    if run is None:
        logger.warning("_try_advance_phase: run_id=%s not found — skipping", run_id)
        return

    # Expire the identity-map cache so the dispatched_phases JSONB read
    # reflects the latest state from raw-SQL UPDATEs (jsonb_set bypasses
    # the ORM identity map). Without this, in_flight counts can be stale
    # and dispatch may exceed the concurrency cap.
    db.expire(run, ["dispatched_phases"])
    db.refresh(run, ["dispatched_phases"])

    manifest = load_bundle_manifest(run.ontology_bundle_key)
    entity_passes = [p.name for p in manifest.passes if not p.depends_on]

    # Branch 1: dispatch next entity pass if cap allows.
    in_flight = sum(
        1 for k, v in (run.dispatched_phases or {}).items()
        if k.startswith("entity_pass_") and v.get("state") in ("claimed", "dispatched")
    )
    if in_flight < settings.pass_concurrency_per_document:
        completed_or_terminal = {
            k.removeprefix("entity_pass_") for k, v in (run.dispatched_phases or {}).items()
            if k.startswith("entity_pass_") and v.get("state") == "completed"
        }
        in_flight_names = {
            k.removeprefix("entity_pass_") for k, v in (run.dispatched_phases or {}).items()
            if k.startswith("entity_pass_") and v.get("state") in ("claimed", "dispatched")
        }
        next_pass = next(
            (p for p in entity_passes
             if p not in completed_or_terminal and p not in in_flight_names),
            None,
        )
        if next_pass is not None:
            _claim_and_dispatch_pass(db, document_id, run_id, next_pass)
            return  # one dispatch per finisher

    # Branch 2: dispatch system_links if all entity passes resolved AND
    # the bundle defines a system_links pass (not all bundles do — guard
    # against StopIteration in the per-pass task body).
    n_resolved = count_terminal_passes(db, run_id, entity_passes)
    has_system_links = any(p.name == "system_links" for p in manifest.passes)
    if n_resolved >= len(entity_passes) and has_system_links:
        sl_state = read_phase_state(db, run_id, "system_links")
        if sl_state is None:
            _claim_and_dispatch_pass(db, document_id, run_id, "system_links")
            return

    # Branch 3: dispatch merge if (a) system_links is resolved, OR
    # (b) the bundle has no system_links and all entity passes are terminal.
    # send_task is used (not .delay) because derive_ontology_graph_merge is
    # defined later in this same file (forward reference). queue="graph" is
    # MANDATORY: without it the message routes to the default "celery" queue
    # where any subscribed worker may grab it. Stale celery processes (e.g.
    # workers started before per-pass-fanin commits) ack-drop with KeyError
    # on the unregistered task name, silently losing the merge dispatch.
    if has_system_links:
        sl_pass = load_pass_output(db, run_id, "system_links")
        sl_resolved = sl_pass is not None and sl_pass.execution_status in (
            "COMPLETE", "SKIPPED", "FAILED"
        )
    else:
        sl_resolved = (n_resolved >= len(entity_passes))

    if sl_resolved:
        merge_state = read_phase_state(db, run_id, "merge")
        if merge_state is None:
            if claim_phase(db, run_id, "merge"):
                celery_app.send_task(
                    "app.workers.pipeline.derive_ontology_graph_merge",
                    args=[document_id, run_id],
                    queue="graph",
                )
                mark_phase_dispatched(db, run_id, "merge", "<send_task>")
                db.commit()


def _claim_and_dispatch_pass(db, document_id: str, run_id: str, pass_name: str) -> None:
    """Claim a phase slot and dispatch the corresponding Celery task.

    Used by both the initial dispatcher (Task 8) and the follow-up dispatch in
    ``_try_advance_phase``.  Pattern: claim_phase → .delay() → mark_phase_dispatched.
    If the claim fails (another worker won), returns without dispatching.

    A crash between claim and mark_phase_dispatched leaves the phase in
    'claimed' state — the reconciler (Task 9) will reclaim it after the
    stale-claim threshold (``phase_claim_stale_seconds``).
    """
    phase_key = _phase_key(pass_name)
    if not claim_phase(db, run_id, phase_key):
        return  # another worker won the claim
    async_result = derive_ontology_graph_pass.delay(document_id, run_id, pass_name)
    mark_phase_dispatched(db, run_id, phase_key, async_result.id)
    db.commit()


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    queue="graph",
    soft_time_limit=settings.pass_soft_time_limit,
    name="app.workers.pipeline.derive_ontology_graph_pass",
)
@guard_stage_run("derive_ontology_graph_pass")
def derive_ontology_graph_pass(
    self, document_id: str, run_id: str, pass_name: str,
) -> dict:
    """One Celery task per pass attempt. Celery is the retry boundary.

    Pass-output write semantics (r4): ``pipeline_pass_outputs`` has at most ONE
    row per (run_id, pass_name) — the terminal one. Intermediate retry attempts
    only update ``StageRun``. The fan-in counter (``count_terminal_passes``)
    therefore counts pass-resolved passes, not failed attempts that may still retry.

    Order of operations:
    1. Cancel check — bail early if run is terminal/cancelled.
    2. Already-resolved check — idempotency guard; advances phase + returns.
    3. Mark phase dispatched (compare-and-reset; benign if reconciler reset us).
    4. Execute one attempt via ``_execute_pass_attempt``.
    5. Write per-attempt StageRun audit row (always, regardless of outcome).
    6. Branch on outcome:
       - COMPLETE / SKIPPED → write terminal pass-output, advance phase.
       - FAILED + retryable + retries remain → commit StageRun; raise self.retry().
       - FAILED + exhausted OR non-retryable terminal → write terminal pass-output,
         mark phase failed, terminalize run if pass required.
    """
    db = _get_db()
    try:
        # 1. Cancel check
        if is_run_cancelled(db, run_id):
            return {"pass_name": pass_name, "skipped": "cancelled"}

        # 2. Already-resolved idempotency guard
        if is_pass_already_resolved(db, run_id, pass_name):
            mark_phase_terminal(db, run_id, _phase_key(pass_name), result="succeeded")
            db.commit()
            _try_advance_phase(db, document_id, run_id)
            return {"pass_name": pass_name, "skipped": "already_resolved"}

        # 3. Compare-and-reset advance to 'dispatched'. Returns False if the
        # reconciler reclaimed the phase under us — we proceed anyway because any
        # new dispatch will see our terminal write and skip.
        mark_phase_dispatched(db, run_id, _phase_key(pass_name), self.request.id)
        db.commit()

        # 4. Execute one attempt
        from app.models.ingest import PipelineRun
        run = db.get(PipelineRun, uuid.UUID(str(run_id)))
        if run is None:
            # Run hard-deleted between cancel-check and here (edge case)
            return {"pass_name": pass_name, "skipped": "run_missing"}

        bundle_key = run.ontology_bundle_key
        manifest = load_bundle_manifest(bundle_key)
        ontology = load_ontology(bundle_key=bundle_key)
        pass_def = next(p for p in manifest.passes if p.name == pass_name)
        doc_json = _build_docling_document_json(document_id)
        upstream_refs = _rehydrate_upstream_refs_from_persisted_passes(
            db, run_id, pass_def, manifest, ontology, document_id,
        )

        attempt_n = self.request.retries + 1
        outcome = _execute_pass_attempt(
            pipeline_run_id=run_id,
            pass_def=pass_def,
            manifest=manifest,
            ontology=ontology,
            bundle_key=bundle_key,
            doc_json=doc_json,
            upstream_refs=upstream_refs,
            document_id=document_id,
        )

        # 5. ALWAYS write StageRun (per-attempt audit; matches existing shape).
        # _write_stage_run manages its own DB session and commit — returns the
        # UUID of the inserted/upserted row so pipeline_pass_outputs.stage_run_id
        # can be populated for direct FK linkage (Issue #2 fix).
        stage_run_id = _write_stage_run(
            pipeline_run_id=run_id,
            pass_def=pass_def,
            attempt=attempt_n,
            execution_status=outcome.execution_status,
            yield_status=outcome.yield_status,
            skip_reason=outcome.skip_reason,
            counts=outcome.counts,
            error=str(outcome.error) if outcome.error else None,
        )

        # 6. Branch: COMPLETE / SKIPPED → terminal write + advance.
        #    FAILED with retry pending → no pass-output write; self.retry().
        #    FAILED with retry exhausted → terminal write + terminalize if required.
        if outcome.execution_status in ("COMPLETE", "SKIPPED"):
            # Second cancel-check: narrow the race window before save.  The targeted
            # FK swallow in save_pass_output catches the residual race where
            # cancel_document fires AFTER this check but BEFORE db.commit().
            if is_run_cancelled(db, run_id):
                return {"pass_name": pass_name, "skipped": "cancelled_mid_extraction"}
            _save_terminal_pass_output(
                db,
                run_id=run_id,
                stage_run_id=stage_run_id,
                pass_name=pass_name,
                attempt=attempt_n,
                outcome=outcome,
            )
            db.commit()
            mark_phase_terminal(
                db, run_id, _phase_key(pass_name),
                result="succeeded" if outcome.execution_status == "COMPLETE" else "skipped",
            )
            db.commit()
            _try_advance_phase(db, document_id, run_id)
            return {"pass_name": pass_name, "execution_status": outcome.execution_status}

        # FAILED branch
        is_retryable = isinstance(outcome.error, (PassRetryable, PassTransportError))
        retries_left = self.request.retries < self.max_retries

        if is_retryable and retries_left:
            # Pending Celery retry. Phase stays in 'dispatched' for the next
            # attempt. NO pipeline_pass_outputs write — fan-in counter must not
            # count this attempt as resolved.
            # Defensive: any pending writes on this session (none expected in the
            # retry branch since _write_stage_run uses its own session, but the
            # flush is harmless insurance) flush before Celery raises.
            db.commit()
            raise self.retry(
                exc=outcome.error,
                countdown=_retry_delay(self.request.retries),
            )

        # Terminal failure: non-retryable PassTerminal OR retryable after exhausting
        # Celery retries. r4: do NOT rely on Celery's MaxRetriesExceededError — it
        # would re-raise without running this cleanup.
        # Second cancel-check: narrow the race window before save.  The targeted
        # FK swallow in save_pass_output catches the residual race where
        # cancel_document fires AFTER this check but BEFORE db.commit().
        if is_run_cancelled(db, run_id):
            return {"pass_name": pass_name, "skipped": "cancelled_mid_extraction"}
        _save_terminal_pass_output(
            db,
            run_id=run_id,
            stage_run_id=stage_run_id,
            pass_name=pass_name,
            attempt=attempt_n,
            outcome=outcome,
            override_status="FAILED",
            override_diagnostics_extra={"retry_exhausted": is_retryable},
        )
        db.commit()
        mark_phase_terminal(db, run_id, _phase_key(pass_name), result="failed")
        db.commit()

        if pass_def.required:
            _update_summary_stage_run(
                db, run_id, "FAILED",
                error=(
                    f"required pass {pass_name} "
                    f"{'retry-exhausted' if is_retryable else 'terminal failure'}"
                ),
            )
            db.commit()
            _terminalize_doc_and_run(document_id, run_id, "PARTIAL_COMPLETE")
            raise IngestFailed(f"Required pass {pass_name} terminal failure")

        # Optional terminal — phase done with result=failed; run continues.
        _try_advance_phase(db, document_id, run_id)
        return {
            "pass_name": pass_name,
            "execution_status": "FAILED",
            "reason": "retry_exhausted" if is_retryable else "terminal",
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Per-pass Celery fan-in — Task 6 helpers + derive_ontology_graph_merge task
# ---------------------------------------------------------------------------


def _assert_stage_run_pass_output_consistency(db, run_id) -> None:
    """Raise WorkerInvariantError if any pass has a COMPLETE StageRun row
    but no matching COMPLETE pipeline_pass_outputs row.

    Detects the crash window between _write_stage_run() and
    _save_terminal_pass_output() in derive_ontology_graph_pass.
    """
    from sqlalchemy import select
    from app.models.ingest import StageRun, PipelinePassOutput

    # Find passes with at least one COMPLETE StageRun row
    complete_stage_run_passes = set(
        row[0] for row in db.execute(
            select(StageRun.pass_name)
            .where(StageRun.pipeline_run_id == uuid.UUID(str(run_id)))
            .where(StageRun.execution_status == "COMPLETE")
            .where(StageRun.pass_name.is_not(None))
            .distinct()
        )
    )
    # Find passes with a COMPLETE pipeline_pass_outputs row
    complete_pass_output_passes = set(
        row[0] for row in db.execute(
            select(PipelinePassOutput.pass_name)
            .where(PipelinePassOutput.pipeline_run_id == uuid.UUID(str(run_id)))
            .where(PipelinePassOutput.execution_status == "COMPLETE")
        )
    )
    missing = complete_stage_run_passes - complete_pass_output_passes
    if missing:
        raise WorkerInvariantError(
            f"COMPLETE StageRun rows exist without matching pipeline_pass_outputs "
            f"rows for run_id={run_id}, passes={sorted(missing)}. "
            f"This indicates a crash between _write_stage_run and "
            f"_save_terminal_pass_output in derive_ontology_graph_pass."
        )


def _rehydrate_pass_result(
    row: "PipelinePassOutput", manifest, ontology, document_id: str,
):
    """Rebuild a PassResult from a persisted pipeline_pass_outputs row.

    Reuses _parse_pass_response (the live-request parsing path) so the
    rehydrated PassResult is structurally identical to one built in-process.
    Then attaches pre_merge_walk via _build_pre_merge_walk_summary so
    classify_yield and merge_and_resolve work as if the pass had just run.

    If _parse_pass_response raises PassTerminal (corrupt persisted JSON), let
    it propagate — the merge task's outer except handler catches it before
    re-raising.
    """
    pass_def = next(p for p in manifest.passes if p.name == row.pass_name)
    pass_result = _parse_pass_response(row.extract_pass_response_json, pass_def, manifest)
    pass_result.pre_merge_walk = _build_pre_merge_walk_summary(
        pass_result, pass_def, ontology, document_id,
    )
    # Note: for document_plus_entity_refs passes, upstream_refs is intentionally
    # not re-attached here — merge_and_resolve does not require it (it resolves
    # refs by identity_dict / ref_id lookup against pass_result entities). If a
    # future merge path needs upstream_refs to be present on rehydrated PassResults,
    # rehydrate them via _select_upstream_refs_for_pass + logical_identity_from_dict
    # the same way _execute_pass_attempt does.
    return pass_result


@celery_app.task(
    bind=True, max_retries=1, default_retry_delay=30, queue="graph",
    soft_time_limit=settings.graph_soft_time_limit,
    name="app.workers.pipeline.derive_ontology_graph_merge",
)
@guard_stage_run("derive_ontology_graph_merge")
def derive_ontology_graph_merge(self, document_id: str, run_id: str) -> dict:
    """Fan-in. Loads COMPLETE pass outputs from pipeline_pass_outputs,
    rehydrates via _parse_pass_response, runs merge_and_resolve + graph
    imports + downstream chain dispatch.

    Rollback contract (preserves pipeline.py:5478-5483 behavior): if
    `tracker.any_mutation_attempted` is True at exception time, call
    `_attempt_rollback(document_id)` BEFORE re-raising, so a Celery
    retry (or operator-driven graph_only reingest) starts from a clean
    graph state.
    """
    db = _get_db()
    tracker = GraphWriteTracker()
    try:
        if is_run_cancelled(db, run_id):
            return {"merge": "skipped_cancelled"}

        _assert_stage_run_pass_output_consistency(db, run_id)

        from app.models.ingest import PipelineRun
        run = db.get(PipelineRun, uuid.UUID(str(run_id)))
        bundle_key = run.ontology_bundle_key
        run_mode = run.mode  # "full" or "graph_only"
        manifest = load_bundle_manifest(bundle_key)
        ontology = load_ontology(bundle_key=bundle_key)

        # Note: check_required_pass_gate opens and closes its own DB session
        # internally (see pipeline.py:1316). This is intentional — the gate's
        # query is independent of the merge task's session lifetime.
        gate = check_required_pass_gate(run_id)
        if not gate.passed:
            _update_summary_stage_run(
                db, run_id, "FAILED",
                error=f"Required passes failed: {gate.failures}",
            )
            mark_phase_terminal(db, run_id, "merge", result="failed")
            db.commit()
            _terminalize_doc_and_run(document_id, run_id, "PARTIAL_COMPLETE")
            raise IngestFailed(f"Required passes failed: {gate.failures}")

        completed_outputs = load_completed_pass_outputs(db, run_id)
        rehydrated = {
            row.pass_name: _rehydrate_pass_result(row, manifest, ontology, document_id)
            for row in completed_outputs.values()
        }

        merged = merge_and_resolve(
            pass_results=rehydrated, manifest=manifest, ontology=ontology,
            document_id=document_id, pipeline_run_id=run_id,
        )
        _apply_post_merge_yield_updates(run_id, merged, manifest)
        _write_pipeline_run_metrics(run_id, merged, manifest)

        provenance_envelope = _build_provenance_envelope(
            document_id, run_id, merged.entities, db,
        )
        identity_to_rid = _import_graph_phase_nodes(
            merged, ontology, document_id, tracker, provenance_envelope,
        )
        # Collect relationship + entity provenance from all rehydrated pass results.
        # entity_provenance_rows is used by _import_graph_phase_domain_edges to build
        # an instance_id → LogicalIdentity map, enabling composite-key matching of
        # relationship provenance rows (Fix A upgrade: per-edge instead of per-rel_type).
        all_rel_provenance = [
            row
            for pr in rehydrated.values()
            for row in (getattr(pr, "relationship_provenance", None) or [])
        ]
        all_entity_provenance = [
            row
            for pr in rehydrated.values()
            for row in (getattr(pr, "provenance", None) or [])
        ]
        _import_graph_phase_domain_edges(
            merged, ontology, tracker, provenance_envelope,
            relationship_provenance_rows=all_rel_provenance,
            entity_provenance_rows=all_entity_provenance,
        )
        _ensure_structural_document_vertex(document_id)
        _import_graph_phase_structural_edges(
            merged, identity_to_rid, document_id, run_id, tracker,
        )

        # Build a detachment-safe snapshot so _upsert_document_graph_extraction
        # can access run metadata after the original DB session was closed.
        from types import SimpleNamespace
        run_snapshot = SimpleNamespace(
            ontology_bundle_key=bundle_key,
            ontology_name=getattr(manifest, "ontology_name", None),
            ontology_version=getattr(manifest, "ontology_version", None),
            use_case_key=None,
            extraction_profile_version=getattr(manifest, "extraction_profile_version", None),
        )

        # Build element_uid → artifact_id map; persist into audit blob so
        # derive_structure_links can read from the snapshot.
        element_uid_to_artifact_id: dict[str, str] = {}
        try:
            db_elem = _get_db()
            try:
                element_uid_to_artifact_id = _build_element_uid_to_artifact_id(
                    db_elem, document_id,
                )
            finally:
                db_elem.close()
        except Exception as exc:
            logger.warning(
                "derive_ontology_graph_merge: element_uid_to_artifact_id build "
                "failed for %s: %s", document_id, exc,
            )

        _upsert_document_graph_extraction(
            document_id=document_id,
            pipeline_run_id=run_id,
            run=run_snapshot,
            merged=merged,
            manifest=manifest,
            identity_to_rid=identity_to_rid,
            element_uid_to_artifact_id=element_uid_to_artifact_id,
        )

        _update_summary_stage_run(db, run_id, "COMPLETE")
        mark_phase_terminal(db, run_id, "merge", result="succeeded")

        db.commit()

        # CHANGED 2026-05-06 (Task 7): dispatch the downstream chain for both
        # modes. After the outer-chain trim, the merge task is the single
        # dispatcher for downstream stages. finalize_document (the last stage
        # in each chain) sets run.status=COMPLETE for both modes — the merge
        # task no longer sets run.status directly in graph_only mode.
        if run_mode == "graph_only":
            # graph_only is a shorter chain — no collect_derivations, no
            # derive_canonicalization (matches the legacy reingest_graph_only
            # chain shape pre-Task-7).
            celery_chain(
                derive_structure_links.si(document_id, run_id),
                finalize_document.si(document_id, run_id),
            ).apply_async()
        else:
            # full mode — 4-stage downstream chain.
            celery_chain(
                collect_derivations.si(document_id, run_id),
                derive_structure_links.si(document_id, run_id),
                derive_canonicalization.si(document_id, run_id),
                finalize_document.si(document_id, run_id),
            ).apply_async()

        return {
            "merge": "ok",
            "entities": len(merged.entities),
            "edges": len(merged.edges),
        }
    except Exception as exc:
        if tracker.any_mutation_attempted:
            rollback_note = _attempt_rollback(document_id)
            logger.info(
                "derive_ontology_graph_merge: rolled back partial graph state "
                "for doc=%s run=%s before re-raising %s%s",
                document_id, run_id, type(exc).__name__, rollback_note,
            )
        try:
            mark_phase_terminal(db, run_id, "merge", result="failed")
            db.commit()
        except Exception:
            logger.exception("merge: mark_phase_terminal failed in error path")
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Task 9: reconcile_ontology_graph_runs — beat-scheduled safety net
# ---------------------------------------------------------------------------


def _get_processing_ontology_graph_runs(db) -> list:
    """Query all PipelineRun rows where status='PROCESSING' AND mode IN ('full', 'graph_only')
    AND the derive_ontology_graph stage has been past initial dispatch (i.e.,
    dispatched_phases is non-empty OR there is a RUNNING summary StageRun).

    The mode filter avoids scanning runs from other unrelated pipeline stages
    that happen to be in PROCESSING.
    """
    from app.models.ingest import PipelineRun
    from sqlalchemy import select, text as sa_text

    # Use a CTE to find run IDs that have a RUNNING derive_ontology_graph summary StageRun.
    # Combined with the dispatched_phases non-empty check using Postgres JSONB operator.
    stmt = select(PipelineRun).where(
        PipelineRun.status == "PROCESSING",
        PipelineRun.mode.in_(["full", "graph_only"]),
        sa_text(
            "("
            "  dispatched_phases != '{}'::jsonb"
            "  OR id IN ("
            "    SELECT pipeline_run_id FROM ingest.stage_runs"
            "    WHERE stage_name = 'derive_ontology_graph'"
            "      AND pass_name IS NULL"
            "      AND status = 'RUNNING'"
            "  )"
            ")"
        ),
    )
    return db.execute(stmt).scalars().all()


def _has_pending_retry_for_pass(
    db,
    pipeline_run_id,
    pass_name: str,
    *,
    countdown_buffer_seconds: int = 600,
) -> bool:
    """True if the latest StageRun for this pass indicates a pending Celery retry.

    Returns True when ALL of:
    - Latest StageRun has execution_status='FAILED'
    - attempt < (pass_max_retries + pass_max_transport_retries)
    - finished_at > now - countdown_buffer_seconds (the Celery countdown is
      still pending in the broker's countdown queue)

    The check uses ``attempt < (pass_max_retries + pass_max_transport_retries)``
    as a conservative upper bound.  In practice, only ``pass_max_retries``
    contributes to Celery's ``self.request.retries`` counter — transport-class
    retries loop internally inside ``_execute_pass_attempt`` without going
    through Celery.  Using the combined sum errs on the side of waiting longer
    rather than racing a retry.  Branch 3 of the reconciler (promote-when-
    pass-output-exists) handles the exhausted-but-recently-finished case
    first, so this conservative bound has no functional impact.

    The countdown_buffer_seconds defaults to 10 min — covers the worst case of
    _retry_delay (300s capped) + a generous safety margin.  Used by the
    reconciler to avoid revoking a task whose retry is genuinely pending.
    """
    from app.models.ingest import StageRun
    from sqlalchemy import select
    from datetime import datetime, timedelta, timezone

    max_attempts = settings.pass_max_retries + settings.pass_max_transport_retries
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=countdown_buffer_seconds)

    row = db.execute(
        select(StageRun)
        .where(
            StageRun.pipeline_run_id == uuid.UUID(str(pipeline_run_id)),
            StageRun.stage_name == "derive_ontology_graph",
            StageRun.pass_name == pass_name,
        )
        .order_by(StageRun.attempt.desc())
        .limit(1)
    ).scalars().first()

    if row is None:
        return False

    if row.execution_status != "FAILED":
        return False

    if row.attempt >= max_attempts:
        return False

    if row.finished_at is None:
        return False

    # finished_at within the countdown buffer → retry is pending
    return row.finished_at > cutoff


@celery_app.task(
    bind=True, queue="graph",
    soft_time_limit=120,
    name="app.workers.pipeline.reconcile_ontology_graph_runs",
)
def reconcile_ontology_graph_runs(self) -> dict:
    """Beat-scheduled safety net for the per-pass fan-in.

    Scans PROCESSING runs every reconciler_period_seconds (default 60s); repairs:
      - Stale claimed phases (>30s, dispatcher crashed before .delay)
      - Stale dispatched phases (>2h, task crashed; pending-retry-aware)
      - Completed-but-not-marked-terminal (task wrote output but didn't mark)
      - Stuck-without-advance (finisher crashed between save and _try_advance_phase)

    Returns a dict summarizing actions taken: {
        "scanned_runs": N,
        "stale_claimed_reclaimed": [...phase keys...],
        "stale_dispatched_reclaimed": [...],
        "promoted_to_terminal": [...],
        "stuck_advances": [...],
        "skipped_pending_retry": [...],
    }
    """
    from datetime import datetime, timezone, timedelta  # local; matches file's style
    from app.models.ingest import PipelineRun
    from app.services.run_phase_dispatch import reclaim_stale_phase

    stale_claimed_threshold_s = settings.phase_claim_stale_seconds
    stale_dispatched_threshold_s = 2 * settings.pass_soft_time_limit

    summary: dict = {
        "scanned_runs": 0,
        "stale_claimed_reclaimed": [],
        "stale_dispatched_reclaimed": [],
        "promoted_to_terminal": [],
        "stuck_advances": [],
        "skipped_pending_retry": [],
    }

    db = _get_db()
    try:
        runs = _get_processing_ontology_graph_runs(db)
        summary["scanned_runs"] = len(runs)

        for run in runs:
            run_id = str(run.id)
            document_id = str(run.document_id)

            # Load the manifest to know which passes are entity passes.
            if not run.ontology_bundle_key:
                logger.debug(
                    "reconcile_ontology_graph_runs: run_id=%s has no ontology_bundle_key, skipping",
                    run_id,
                )
                continue

            try:
                manifest = load_bundle_manifest(run.ontology_bundle_key)
            except Exception:
                logger.warning(
                    "reconcile_ontology_graph_runs: could not load manifest for run_id=%s "
                    "bundle_key=%s; skipping",
                    run_id, run.ontology_bundle_key, exc_info=True,
                )
                continue

            entity_passes = [p.name for p in manifest.passes if not p.depends_on]

            dispatched_phases = run.dispatched_phases or {}

            # ----------------------------------------------------------------
            # Branch 4: stuck-without-advance check (fast path)
            # All entity passes have terminal pass-output rows but no system_links
            # or merge phase has been started yet (finisher crashed between
            # saving the pass-output and calling _try_advance_phase).
            # ----------------------------------------------------------------
            n_terminal = count_terminal_passes(db, run_id, entity_passes)
            if n_terminal >= len(entity_passes) and entity_passes:
                # Check if merge or system_links has not been started / is still running.
                merge_absent = "merge" not in dispatched_phases
                sl_entry = dispatched_phases.get("system_links")
                # sl_blocking is True only when system_links is in-progress (claimed
                # or dispatched). A completed system_links is NOT blocking — it means
                # we should proceed to dispatch merge.  An absent system_links is also
                # not blocking — _try_advance_phase will dispatch it first.
                sl_blocking = sl_entry is not None and sl_entry.get("state") != "completed"
                # Only advance if no follow-up is already in progress.
                # _try_advance_phase has its own idempotency via claim_phase.
                if merge_absent and not sl_blocking:
                    logger.info(
                        "reconcile_ontology_graph_runs: stuck-without-advance "
                        "run_id=%s — all %d entity passes terminal, no follow-up; advancing",
                        run_id, len(entity_passes),
                    )
                    try:
                        _try_advance_phase(db, document_id, run_id)
                        db.commit()
                        summary["stuck_advances"].append(run_id)
                    except Exception:
                        logger.warning(
                            "reconcile_ontology_graph_runs: _try_advance_phase failed for "
                            "run_id=%s", run_id, exc_info=True,
                        )
                        db.rollback()
                    continue

            # ----------------------------------------------------------------
            # Per-phase inspection: iterate every entry in dispatched_phases.
            # ----------------------------------------------------------------
            for phase_key, phase_entry in dispatched_phases.items():
                state = (phase_entry or {}).get("state")
                if state not in ("claimed", "dispatched"):
                    # Completed or missing state — no action needed.
                    continue

                # Map phase_key back to pass_name for pass-output queries.
                if phase_key.startswith("entity_pass_"):
                    pass_name = phase_key[len("entity_pass_"):]
                else:
                    pass_name = phase_key  # "system_links" or "merge"

                if state == "claimed":
                    # --------------------------------------------------------
                    # Branch 1: stale claimed repair
                    # --------------------------------------------------------
                    claimed_at_raw = phase_entry.get("claimed_at")
                    if claimed_at_raw is None:
                        continue
                    claimed_at = datetime.fromisoformat(claimed_at_raw)
                    age_s = (datetime.now(timezone.utc) - claimed_at).total_seconds()
                    if age_s < stale_claimed_threshold_s:
                        continue  # still fresh

                    logger.info(
                        "reconcile_ontology_graph_runs: stale claimed phase "
                        "run_id=%s phase=%s age=%.0fs — reclaiming",
                        run_id, phase_key, age_s,
                    )
                    try:
                        reclaimed = reclaim_stale_phase(
                            db, run_id, phase_key,
                            claim_threshold_s=stale_claimed_threshold_s,
                            dispatch_threshold_s=stale_dispatched_threshold_s,
                        )
                        if reclaimed:
                            db.commit()
                            summary["stale_claimed_reclaimed"].append(phase_key)
                            _try_advance_phase(db, document_id, run_id)
                            db.commit()
                        else:
                            db.rollback()
                    except Exception:
                        logger.warning(
                            "reconcile_ontology_graph_runs: reclaim failed for "
                            "run_id=%s phase=%s", run_id, phase_key, exc_info=True,
                        )
                        db.rollback()

                elif state == "dispatched":
                    dispatched_at_raw = phase_entry.get("dispatched_at")
                    if dispatched_at_raw is None:
                        continue
                    dispatched_at = datetime.fromisoformat(dispatched_at_raw)
                    age_s = (datetime.now(timezone.utc) - dispatched_at).total_seconds()

                    # --------------------------------------------------------
                    # Branch 3: promote completed-but-not-marked-terminal
                    # Check BEFORE the stale-dispatched threshold — a task
                    # that wrote a pass-output row but crashed before
                    # mark_phase_terminal must be promoted regardless of age.
                    # --------------------------------------------------------
                    try:
                        pass_output = load_pass_output(db, run_id, pass_name)
                    except Exception:
                        logger.warning(
                            "reconcile_ontology_graph_runs: load_pass_output failed for "
                            "run_id=%s pass=%s", run_id, pass_name, exc_info=True,
                        )
                        continue

                    if pass_output is not None:
                        # Task completed AND wrote the pass-output row, but
                        # crashed before mark_phase_terminal. Promote it.
                        exec_status = pass_output.execution_status
                        result = (
                            "succeeded" if exec_status == "COMPLETE"
                            else "skipped" if exec_status == "SKIPPED"
                            else "failed"
                        )
                        logger.info(
                            "reconcile_ontology_graph_runs: promote dispatched-with-output "
                            "run_id=%s phase=%s exec_status=%s → result=%s",
                            run_id, phase_key, exec_status, result,
                        )
                        try:
                            mark_phase_terminal(db, run_id, phase_key, result=result)
                            db.commit()
                            summary["promoted_to_terminal"].append(phase_key)
                            _try_advance_phase(db, document_id, run_id)
                            db.commit()
                        except Exception:
                            logger.warning(
                                "reconcile_ontology_graph_runs: promote failed for "
                                "run_id=%s phase=%s", run_id, phase_key, exc_info=True,
                            )
                            db.rollback()
                        continue

                    # --------------------------------------------------------
                    # Branch 2: stale dispatched repair (pending-retry-aware)
                    # --------------------------------------------------------
                    if age_s < stale_dispatched_threshold_s:
                        continue  # still within the expected completion window

                    # Check for pending Celery retry before revoking.
                    try:
                        pending = _has_pending_retry_for_pass(db, run_id, pass_name)
                    except Exception:
                        logger.warning(
                            "reconcile_ontology_graph_runs: _has_pending_retry_for_pass "
                            "failed for run_id=%s pass=%s; skipping reclaim",
                            run_id, pass_name, exc_info=True,
                        )
                        continue

                    if pending:
                        logger.debug(
                            "reconcile_ontology_graph_runs: skipping stale dispatched "
                            "run_id=%s phase=%s — pending Celery retry detected",
                            run_id, phase_key,
                        )
                        summary["skipped_pending_retry"].append(phase_key)
                        continue

                    logger.info(
                        "reconcile_ontology_graph_runs: stale dispatched phase "
                        "run_id=%s phase=%s age=%.0fs — revoking + reclaiming",
                        run_id, phase_key, age_s,
                    )
                    try:
                        reclaimed = reclaim_stale_phase(
                            db, run_id, phase_key,
                            claim_threshold_s=stale_claimed_threshold_s,
                            dispatch_threshold_s=stale_dispatched_threshold_s,
                        )
                        if reclaimed:
                            db.commit()
                            summary["stale_dispatched_reclaimed"].append(phase_key)
                            _try_advance_phase(db, document_id, run_id)
                            db.commit()
                        else:
                            db.rollback()
                    except Exception:
                        logger.warning(
                            "reconcile_ontology_graph_runs: reclaim (dispatched) failed for "
                            "run_id=%s phase=%s", run_id, phase_key, exc_info=True,
                        )
                        db.rollback()

        no_actions = (
            summary["scanned_runs"] == 0
            or not any([
                summary["stale_claimed_reclaimed"],
                summary["stale_dispatched_reclaimed"],
                summary["promoted_to_terminal"],
                summary["stuck_advances"],
                summary["skipped_pending_retry"],
            ])
        )
        log_method = logger.debug if no_actions else logger.info
        log_method(
            "reconcile_ontology_graph_runs: scan complete — "
            "scanned=%d stale_claimed=%d stale_dispatched=%d promoted=%d "
            "stuck=%d skipped=%d",
            summary["scanned_runs"],
            len(summary["stale_claimed_reclaimed"]),
            len(summary["stale_dispatched_reclaimed"]),
            len(summary["promoted_to_terminal"]),
            len(summary["stuck_advances"]),
            len(summary["skipped_pending_retry"]),
        )
        return summary
    except Exception:
        logger.exception("reconcile_ontology_graph_runs: unexpected error in reconciler scan")
        db.rollback()
        raise
    finally:
        db.close()


# CHANGED 2026-05-06 (Task 8): replaced monolithic ~225-line helper
# (_derive_ontology_graph_bundle_passes) with a thin ~50-line dispatcher.
# soft_time_limit dropped from 8 h (settings.graph_soft_time_limit) to 10 min
# because this task only does manifest-load + initial pass dispatch.
@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    queue="graph",
    soft_time_limit=600,
    name="app.workers.pipeline.derive_ontology_graph",
)
@guard_stage_run("derive_ontology_graph",
    lifecycle=True,
    next_stage=None,
    next_task=None,
    intercept_terminal=False)
def derive_ontology_graph(self, document_id: str, run_id: str | None = None) -> dict:
    """Thin dispatcher: create summary StageRun then fan-out to per-pass tasks.

    Callers always dispatch as derive_ontology_graph.si(document_id, run_id),
    where run_id IS the pipeline_run_id.

    Steps:
    1. Resolve run_id + PipelineRun (orphaned-run safety net).
    2. Create the derive_ontology_graph summary StageRun with status=RUNNING
       so finalize_document's REQUIRED_STAGES gate sees this stage as in-flight.
    3. Load the manifest; identify entity passes (p.depends_on empty).
    4. Dispatch the first pass_concurrency_per_document entity passes via
       claim_phase → .delay → mark_phase_dispatched (same flow used by
       _try_advance_phase follow-ups in derive_ontology_graph_pass).
    5. Return immediately — per-pass tasks handle extraction.
    """
    from app.models.ingest import PipelineRun, StageRun
    from datetime import datetime, timezone
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    db = _get_db()
    try:
        if not run_id:
            run_id = _get_pipeline_run_id(db, document_id)
        run = db.get(PipelineRun, uuid.UUID(str(run_id))) if run_id else None
        if run is None:
            logger.warning(
                "derive_ontology_graph: pipeline_run %s not found "
                "(document likely deleted); skipping orphaned task",
                run_id,
            )
            return {
                "stage": "derive_ontology_graph",
                "status": "skipped",
                "reason": "orphaned_run",
            }

        # Idempotent insert — when two dispatcher copies race (e.g., Celery
        # redelivery after worker crash), both can safely fire this; the partial
        # unique index uq_stage_runs_summary_row (WHERE pass_name IS NULL) guards
        # against duplicates without raising IntegrityError on the second copy.
        stmt = (
            pg_insert(StageRun)
            .values(
                id=uuid.uuid4(),
                pipeline_run_id=uuid.UUID(str(run_id)),
                stage_name="derive_ontology_graph",
                pass_name=None,
                attempt=self.request.retries + 1,
                status="RUNNING",
                execution_status="RUNNING",
                started_at=datetime.now(timezone.utc),
            )
            .on_conflict_do_nothing(
                index_elements=["pipeline_run_id", "stage_name", "attempt"],
                index_where=sa.text("pass_name IS NULL"),
            )
        )
        db.execute(stmt)

        bundle_key = run.ontology_bundle_key

        # Skip extraction dispatch entirely when the doc has no
        # extractable content. Every pass would fail the structured-output
        # quality gate against tiny picture-description blurbs and burn
        # LLM time. Synthesize SKIPPED rows for required passes so the
        # merge gate is satisfied and the chain finalizes COMPLETE.
        anchors_metrics_row = db.execute(
            sa.text(
                "SELECT metrics FROM ingest.stage_runs "
                "WHERE pipeline_run_id = :run_id "
                "  AND stage_name = 'derive_document_anchors' "
                "  AND status = 'COMPLETE' "
                "  AND pass_name IS NULL "
                "ORDER BY started_at DESC NULLS LAST LIMIT 1"
            ),
            {"run_id": str(run_id)},
        ).scalar()
        anchors_metrics = anchors_metrics_row if isinstance(anchors_metrics_row, dict) else {}
        text_blocks = int(anchors_metrics.get("text_block_count") or 0)
        tables = int(anchors_metrics.get("table_count") or 0)
        figures = int(anchors_metrics.get("figure_count") or 0)
        # The anchor walker only emits TEXT_BLOCK for text near pictures, so
        # pure-text docs report all-zero anchor counts. Use markdown size
        # (cached on prepare_document.metrics) as the actual "is this doc
        # empty?" signal. Falls back to a MinIO read if the metric is
        # missing (legacy 5xx-fallback path).
        if anchors_metrics and text_blocks == 0 and tables == 0 and figures == 0:
            markdown_chars = _get_markdown_chars(db, run_id, document_id)
        else:
            markdown_chars = -1  # not consulted
        if (
            anchors_metrics
            and text_blocks == 0 and tables == 0 and figures == 0
            and markdown_chars < 5000
        ):
            logger.warning(
                "derive_ontology_graph: skipping extraction dispatch for %s "
                "(text_blocks=0, tables=0, figures=0, markdown=%d chars) — "
                "synthesizing required pass StageRuns and dispatching merge",
                document_id, markdown_chars,
            )
            # Required passes per the manifest gate. Each needs ONE StageRun
            # row whose execution_status='SKIPPED' and skip_reason is
            # authorized so the gate accepts it.
            # Synthesize one StageRun per required pass so the merge gate
            # (check_required_pass_gate) is satisfied. SKIPPED + skip_reason
            # avoids the COMPLETE-with-no-pass_output consistency assertion.
            manifest_skip = load_bundle_manifest(bundle_key)
            required_passes = [p for p in manifest_skip.passes if p.required]
            skip_metrics = {
                "skipped": True,
                "reason": "empty_anchor_set",
                "synthetic": True,
            }
            for pass_def in required_passes:
                _write_stage_run(
                    pipeline_run_id=run_id,
                    pass_def=pass_def,
                    attempt=1,
                    execution_status="SKIPPED",
                    yield_status=None,
                    skip_reason="EMPTY_ANCHOR_SET",
                    counts={"metrics": skip_metrics},
                    error=None,
                )
            db.execute(
                sa.text(
                    "UPDATE ingest.stage_runs "
                    "SET status = 'COMPLETE', execution_status = 'COMPLETE', "
                    "    finished_at = NOW(), "
                    "    metrics = COALESCE(metrics, '{}'::jsonb) || CAST(:metrics AS jsonb) "
                    "WHERE pipeline_run_id = :run_id "
                    "  AND stage_name = 'derive_ontology_graph' "
                    "  AND pass_name IS NULL"
                ),
                {
                    "run_id": str(run_id),
                    "metrics": json.dumps({
                        "skipped": True,
                        "reason": "empty_anchor_set",
                    }),
                },
            )
            db.commit()
            derive_ontology_graph_merge.si(document_id, str(run_id)).apply_async()
            return {
                "stage": "derive_ontology_graph",
                "status": "skipped",
                "reason": "empty_anchor_set",
                "required_passes_synthesized": len(required_passes),
            }

        db.commit()
    finally:
        db.close()

    manifest = load_bundle_manifest(bundle_key)
    entity_passes = [p.name for p in manifest.passes if not p.depends_on]

    # Second session: _claim_and_dispatch_pass commits phase records
    # independently. Using a single session would interleave the RUNNING
    # StageRun commit with phase claim commits and risk dirty reads.
    db2 = _get_db()
    try:
        for pass_name in entity_passes[: settings.pass_concurrency_per_document]:
            _claim_and_dispatch_pass(db2, document_id, str(run_id), pass_name)
    finally:
        db2.close()

    return {
        "stage": "derive_ontology_graph",
        "status": "dispatched",
        "entity_passes_dispatched": min(
            len(entity_passes), settings.pass_concurrency_per_document
        ),
    }


@celery_app.task(bind=True, max_retries=1, default_retry_delay=30, queue="graph",
                 soft_time_limit=settings.finalize_soft_time_limit,
                 time_limit=settings.finalize_time_limit)
@guard_stage_run("derive_structure_links")
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

        # SAME_SECTION is chunk-to-chunk; SECTION vertices are written by
        # derive_document_anchors (spec §3.4). Here we only link TextChunks
        # that share the same `section_path` string — no SECTION-vertex
        # attachment is produced (that would require a new structural edge).
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

        # Provenance envelope re-used across every structural edge below so
        # get_document_edges_sync (which filters on the edge's document_id
        # property) can reach them. Without this each CONTAINS_TEXT /
        # CONTAINS_IMAGE / SAME_PAGE row lands with document_id=None and
        # becomes invisible to per-document queries.
        _edge_props = {
            "document_id": str(document_id),
            "pipeline_run_id": str(run_id) if run_id is not None else None,
            "source": "derive_structure_links",
        }

        # Connect existing TextChunk/ImageChunk vertices (created in embedding stages) to Document
        # Use get_chunk_rid_sync to look up already-created vertices — do NOT re-create them
        tc_rid_map: dict[str, str] = {}
        for tc in text_chunks:
            tc_rid = graph_store.get_chunk_rid_sync(str(tc.id), "TextChunk")
            if tc_rid:
                tc_rid_map[str(tc.id)] = tc_rid
                graph_store.create_structural_edge_sync(
                    doc_rid, tc_rid, "CONTAINS_TEXT", properties=_edge_props,
                )
            else:
                logger.warning("derive_structure_links: TextChunk vertex not found for chunk %s", tc.id)

        ic_rid_map: dict[str, str] = {}
        for ic in image_chunks:
            ic_rid = graph_store.get_chunk_rid_sync(str(ic.id), "ImageChunk")
            if ic_rid:
                ic_rid_map[str(ic.id)] = ic_rid
                graph_store.create_structural_edge_sync(
                    doc_rid, ic_rid, "CONTAINS_IMAGE", properties=_edge_props,
                )
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
                        graph_store.create_structural_edge_sync(
                            tc_rid, ic_rid, "SAME_PAGE", properties=_edge_props,
                        )
                    except Exception:
                        pass  # Best-effort for cross-chunk edges

        # Create doc-structure edges in ArcadeDB (NEXT_CHUNK, SAME_SECTION,
        # SAME_ARTIFACT) with weight properties, mirroring the Postgres
        # chunk_links. These enable ArcadeDB-native retrieval expansion.
        structural_edge_sql = []
        structural_params: dict[str, Any] = {
            "_doc_id": str(document_id),
            "_pipeline_run_id": str(run_id) if run_id is not None else None,
        }
        edge_idx = 0

        def _add_structural_edge(from_rid: str, to_rid: str, link_type: str, weight: float) -> None:
            nonlocal edge_idx
            structural_edge_sql.append(
                f"CREATE EDGE {link_type} FROM {from_rid} TO {to_rid} "
                f"SET weight = :w_{edge_idx}, document_id = :_doc_id, "
                f"pipeline_run_id = :_pipeline_run_id, created_at = sysdate()"
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

        # Phase 8 Task 53b: collect edges via (entity_id, source_rid) from
        # the audit blob's mentions[] / nodes[] entries. entity_id is the
        # canonical LogicalIdentity serialization (Task 52b) so fallback
        # suppression now distinguishes same-name same-type siblings that
        # the old name-only set conflated. source_rid is the pre-resolved
        # entity vertex RID — the batch writer uses it directly instead
        # of the old name+type LIMIT-1 subquery.
        #
        # edge_records carries (entity_name, entity_type, chunk_id,
        # entity_id, source_rid) tuples — built from both the mentions
        # path and the fallback path below.
        edge_records: list[tuple[str, str, str, str | None, str | None]] = []

        # entity_id ↔ (name, type, source_rid) lookup built from nodes[]
        # so the fallback path can emit the same tuple shape as the
        # primary path even when no mention row resolved.
        node_by_entity_id: dict[str, tuple[str, str, str | None]] = {}
        mentioned_entity_ids: set[str] = set()

        if graph_extraction and graph_extraction.graph_json:
            for node in graph_extraction.graph_json.get("nodes", []):
                eid = node.get("entity_id")
                if not eid:
                    continue
                node_by_entity_id[eid] = (
                    node.get("name", ""),
                    node.get("entity_type", "UNKNOWN"),
                    node.get("rid"),
                )

            # Fallback pool for mentions whose element_uid doesn't resolve
            # through element_uid_chunk_map. The docling-graph service's
            # provenance synthesizer emits Docling-internal self_refs
            # (e.g. "#/pictures/1") as element_uid when the library's
            # salvage path strips the per-node element-tracking attrs,
            # but DocumentElement.element_uid is stored as
            # "{page}-{order}-{type}-{hash}" — the two namespaces don't
            # overlap. Without a fallback, those mentions produce zero
            # EXTRACTED_FROM edges and the entity becomes unreachable
            # from Document via the chunk traversal. Fan out across all
            # TextChunks of this document as a coarse-but-valid anchor.
            all_text_chunk_ids = [str(tc.id) for tc in text_chunks]

            for mention in graph_extraction.graph_json.get("mentions", []):
                eid = mention.get("entity_id")
                name = mention.get("entity_name", "")
                etype = mention.get("entity_type", "UNKNOWN")
                euid = mention.get("element_uid", "")
                src_rid = mention.get("rid")
                resolved_chunks = element_uid_chunk_map.get(euid, [])
                if not resolved_chunks and isinstance(euid, str) and euid.startswith("#/"):
                    # Synthesizer-anchored self_ref couldn't resolve to a
                    # concrete DocumentElement. Attach to every text chunk
                    # in the document.
                    resolved_chunks = all_text_chunk_ids
                for chunk_id in resolved_chunks:
                    edge_records.append((name, etype, chunk_id, eid, src_rid))
                    if eid:
                        mentioned_entity_ids.add(eid)

        # Fallback — fan the entity out across its artifact's chunks when
        # the primary mention path yielded zero links. Keyed by entity_id
        # so same-name same-type siblings with different identity tuples
        # are tracked independently (T53b correctness fix).
        entity_ids_needing_fallback = [
            eid for eid in node_by_entity_id if eid not in mentioned_entity_ids
        ]
        if entity_ids_needing_fallback or not mentioned_entity_ids:
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

            # Legacy fallback: artifact_metadata-derived entities (no
            # entity_id / source_rid available from that path — writer
            # falls back to the name+type subquery with a WARNING).
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
                        edge_records.append((name, etype, chunk_id, None, None))

        # Batch-create EXTRACTED_FROM edges in one sqlscript call.
        from app.services.graph_store import EntityChunkEdge as _ECE
        entity_edge_records: list[_ECE] = []
        for (ent_name, ent_type, chunk_id, entity_id, source_rid) in edge_records:
            chunk_rid = tc_rid_map.get(chunk_id) or ic_rid_map.get(chunk_id)
            if not chunk_rid:
                continue
            entity_edge_records.append(_ECE(
                entity_name=ent_name,
                entity_type=ent_type,
                chunk_rid=chunk_rid,
                entity_id=entity_id,
                source_rid=source_rid,
            ))

        entity_links = 0
        if entity_edge_records:
            # NOTE: failures are NOT swallowed here. batch_create_entity_chunk_edges_sync
            # retries on ArcadeDB's RecordNotFound race internally (up to 3 attempts).
            # Any exception that escapes the retry helper is a genuine failure and
            # should propagate to guard_stage_run, which will terminalize the doc
            # rather than silently complete with missing edges. (Pre-2026-04-24
            # behavior swallowed these and the 2026-04-24 run lost 5+ edges silently.)
            entity_links = graph_store.batch_create_entity_chunk_edges_sync(
                entity_edge_records,
                document_id=str(document_id),
                pipeline_run_id=str(run_id) if run_id is not None else None,
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
@guard_stage_run("collect_derivations")
def collect_derivations(self, document_id: str, run_id: str | None = None) -> None:
    """Post-derivation checkpoint: mark document as past derivation stages.

    Also writes a COMPLETE stage_run row so finalize_document's
    REQUIRED_STAGES check (pipeline.py ~line 5749) sees this stage as
    complete. Previously the task only updated documents.pipeline_stage
    without writing stage_runs, which caused finalize to report the stage
    missing in edge cases.
    """
    logger.info("collect_derivations: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="collect_derivations")

    if run_id:
        db = _get_db()
        try:
            _update_stage_run(
                db, run_id, "collect_derivations", "RUNNING",
                attempt=self.request.retries + 1,
            )
            db.commit()
        finally:
            db.close()

    # No-op body beyond the status/stage_run bookkeeping; this task is a
    # Celery join point that runs after derivation stages and before the
    # structure-link / canonicalize / finalize tail.

    if run_id:
        db = _get_db()
        try:
            _update_stage_run(
                db, run_id, "collect_derivations", "COMPLETE",
                attempt=self.request.retries + 1,
            )
            db.commit()
        finally:
            db.close()


@celery_app.task(bind=True, max_retries=1, default_retry_delay=30, queue="graph",
                 soft_time_limit=settings.finalize_soft_time_limit,
                 time_limit=settings.finalize_time_limit)
@guard_stage_run("derive_canonicalization")
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
@guard_stage_run("finalize_document")
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

        # CHANGED 2026-05-06 (Task 8): mode-scoped required stages. graph_only runs
        # (re-extraction without re-converting the source PDF) skip the
        # prepare/translate/metadata/picture/embedding/canonicalization stages —
        # only the graph extraction path runs. Using the full set for graph_only
        # caused those runs to always resolve to PARTIAL_COMPLETE (the bug fixed here).
        _FULL_REQUIRED_STAGES = {
            "prepare_document",
            "detect_and_translate",
            "derive_document_metadata",
            "derive_picture_descriptions",
            "purge_document_derivations",
            "derive_text_embeddings",
            "derive_image_embeddings",
            "derive_document_anchors",
            "derive_ontology_graph",
            "derive_structure_links",
            "derive_canonicalization",
        }
        _GRAPH_ONLY_REQUIRED_STAGES = {
            "derive_document_anchors",
            "derive_ontology_graph",
            "derive_structure_links",
        }
        run_row = db.get(PipelineRun, uuid.UUID(run_id))
        run_mode = run_row.mode if run_row is not None else "full"
        REQUIRED_STAGES = (
            _GRAPH_ONLY_REQUIRED_STAGES if run_mode == "graph_only" else _FULL_REQUIRED_STAGES
        )

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
