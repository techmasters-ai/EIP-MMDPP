# Per-Pass Celery Tasks for `derive_ontology_graph` Implementation Plan (r4)

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic `derive_ontology_graph` Celery task — which runs all 12 ontology passes inline inside one 8-hour-time-limited task — with per-pass Celery tasks plus a DB-backed fan-in driven by a recoverable phase state machine. This bounds blast radius, lets independent entity passes run with bounded concurrency, prevents already-completed passes from being redone on retry, and persists per-pass outputs as the source of truth for resume.

**Prerequisites (already shipped 2026-05-06):**

- `PassTransportError(PassRetryable)` exception class in `app/workers/pipeline.py` — distinguishes infra-level transport failures (`httpx.TransportError`/`TimeoutException`) from business-logic retryables.
- `pass_max_transport_retries` setting (default 3) — separate retry budget from `pass_max_retries`.
- `_call_extract_pass` raises `PassRetryable` for service stub-on-error responses (`diagnostics.pipeline_error` set).

This plan builds on those.

**Architecture (r4 — diff from r3):**

| r3 | r4 |
|---|---|
| Pass-output row written every attempt (incl. intermediate failures) | **Pass-output row written only on terminal resolution.** Intermediate retry-FAILED attempts only update `StageRun`. Resolves the "FAILED-attempt counted as resolved" race the reviewer flagged. |
| Phase entry has `state` only | **Phase entry has `state` + `result`** (`succeeded`/`failed`/`skipped`). Operators and reconciler can read failure-vs-success without joining `pipeline_pass_outputs`. |
| `claim_phase` / `reclaim_stale_phase` use simple presence check | **Compare-and-reset semantics.** UPDATE predicate includes current state AND timestamp; concurrent fresh claims by another worker can't be clobbered by a stale reconciler read. |
| Retry exhaustion implicit (Celery raises `MaxRetriesExceededError`) | **Explicit exhaustion handling before `self.retry()`.** When `self.request.retries >= self.max_retries`, the task writes a terminal pass-output row, marks phase terminal, terminalizes run if pass is required, and returns — bypassing the implicit `MaxRetriesExceededError` path which doesn't run cleanup. |
| Reconciler decides `dispatched` is stale by age + missing-output check | **Reconciler also checks `pipeline_pass_outputs` for the latest attempt's status.** If the latest attempt is `FAILED` and the task scheduled a Celery retry (via Celery's countdown queue), the reconciler waits — does not revoke. |
| Merge had `GraphWriteTracker` referenced but rollback path under retry not specified | **Explicit merge rollback contract.** On any exception in `derive_ontology_graph_merge`, before raising: if `GraphWriteTracker.first_mutation_recorded`, call `_attempt_rollback(document_id)` to clear the partial graph state. Matches the existing pipeline.py:5191 contract. |

**Other r3 architecture decisions retained:**

- `ingest.pipeline_pass_outputs` table; raw `/extract-pass` response payload stored; completion keyed on `execution_status='COMPLETE'`; SKIPPED with `skip_reason` recorded distinctly.
- `pipeline_runs.dispatched_phases` JSONB state machine. Updated shape:

  ```json
  {
    "entity_pass_radar_identity": {
      "state": "completed",
      "result": "succeeded",
      "task_id": "9b1c…",
      "claimed_at": "2026-05-06T15:31:02Z",
      "dispatched_at": "2026-05-06T15:31:03Z",
      "completed_at": "2026-05-06T15:42:11Z"
    },
    "entity_pass_radar_modulation": {
      "state": "completed", "result": "failed",
      "task_id": "ab12…", "claimed_at": "…", "dispatched_at": "…", "completed_at": "…"
    },
    "system_links": { "state": "claimed", "result": null, "task_id": null, "claimed_at": "…" }
  }
  ```

- DB-backed fan-in (no Celery chord); explicit `derive_ontology_graph` summary StageRun lifecycle (RUNNING in dispatcher → COMPLETE in merge / FAILED in required-pass terminalization); outer chain ends at `derive_ontology_graph`; `pass_concurrency_per_document=2` cap shared by initial + follow-up dispatch; reconciler beat schedule.

**Tech Stack:** Celery 5.x (group, no chord), SQLAlchemy + Alembic, PostgreSQL JSONB + `SELECT FOR UPDATE` + compare-and-reset UPDATE predicates, pydantic-settings, pytest.

---

## File Structure

| File | Purpose |
|---|---|
| `app/models/ingest.py` | Add `PipelinePassOutput` model; add `dispatched_phases JSONB DEFAULT '{}'` column on `PipelineRun` |
| `alembic/versions/<hash>_add_pipeline_pass_outputs.py` | Migration: table + column |
| `app/services/pass_outputs_store.py` | `save_pass_output`, `load_pass_output`, `load_completed_pass_outputs`, `count_completed_passes`, `count_terminal_passes`, `is_pass_already_resolved` |
| `app/services/run_phase_dispatch.py` | State-machine helpers: `claim_phase`, `mark_phase_dispatched`, `mark_phase_terminal`, `reclaim_stale_phase`, `is_run_cancelled`, `read_phase_state` |
| `app/workers/pipeline.py` | Refactor: extract `_execute_pass_attempt` helper from `_run_single_pass`; new `derive_ontology_graph_pass`, `derive_ontology_graph_merge`, `reconcile_ontology_graph_runs` tasks; rewrite `derive_ontology_graph` as dispatcher; trim outer chain in `start_ingest_pipeline` and `reingest_graph_only` |
| `app/workers/celery_app.py` | Beat entry for the reconciler |
| `app/config.py` | New `pass_soft_time_limit: int = 3600`, `pass_concurrency_per_document: int = 2`, `reconciler_period_seconds: int = 60`, `phase_claim_stale_seconds: int = 30` |
| `tests/unit/test_pass_outputs_store.py` | Persistence (incl. SKIPPED handling) |
| `tests/unit/test_run_phase_dispatch.py` | State machine + reclaim |
| `tests/unit/test_execute_pass_attempt.py` | Extracted helper |
| `tests/unit/test_derive_ontology_graph_pass_task.py` | Per-pass task end-to-end |
| `tests/unit/test_derive_ontology_graph_merge_task.py` | Merge + summary-row + downstream-chain |
| `tests/unit/test_reconcile_ontology_graph_runs.py` | Stale `claimed` and stale `dispatched` recovery |
| `tests/integration/test_per_pass_db_fanin_e2e.py` | End-to-end with mocked LLM |
| `VERIFICATION_CHECKLIST.md`, `README.md`, `docs/operational/per-pass-fanin-rollout.md` | Docs |

---

## Chunk 1: Persistence Layer

### Task 1: Schema Changes — `PipelinePassOutput` + `dispatched_phases`

**Files:** `app/models/ingest.py`, new alembic migration

Model:

```python
# app/models/ingest.py
class PipelinePassOutput(Base):
    """Terminal pass-output record (r4): at most one row per (run_id, pass_name).
    Existence of a row means the pass has been resolved (succeeded/skipped/
    failed-no-more-retries). Intermediate retry attempts only update StageRun.

    Per-attempt audit lives in ``stage_runs``; this table is the SSoT for
    "did this pass resolve" and "what was the final result."
    """
    __tablename__ = "pipeline_pass_outputs"
    __table_args__ = (
        # Terminal-only: unique on (run_id, pass_name). NOT (run, pass, attempt).
        UniqueConstraint("pipeline_run_id", "pass_name",
                         name="uq_pipeline_pass_outputs_run_pass"),
        {"schema": "ingest"},
    )
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pipeline_run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ingest.pipeline_runs.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    stage_run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ingest.stage_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    pass_name = Column(String(64), nullable=False)
    attempt = Column(Integer, nullable=False, default=1)  # the attempt that resolved
    execution_status = Column(String(16), nullable=False)  # COMPLETE / FAILED / SKIPPED
    skip_reason = Column(String(64), nullable=True)        # NO_UPSTREAM_ENDPOINTS, etc.
    yield_status = Column(String(16), nullable=True)
    extract_pass_response_json = Column(JSONB, nullable=False)  # raw /extract-pass payload
    primary_entities_extracted = Column(Integer, nullable=True)
    bridge_entities_extracted = Column(Integer, nullable=True)
    relationships_extracted = Column(Integer, nullable=True)
    relationships_rejected = Column(Integer, nullable=True)
    diagnostics_json = Column(JSONB, nullable=False, default=dict)
    field_provenance_json = Column(JSONB, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)


# Add to existing PipelineRun:
dispatched_phases = Column(
    JSONB, nullable=False,
    default=dict, server_default=text("'{}'::jsonb"),
)
```

Steps:
- [ ] **1.1** Add model + column above
- [ ] **1.2** `alembic revision --autogenerate -m "add pipeline_pass_outputs and dispatched_phases"`; verify schema qualifier preserved
- [ ] **1.3** `alembic upgrade head`; `\d ingest.pipeline_pass_outputs`; verify `pipeline_runs.dispatched_phases` defaults to `'{}'::jsonb`
- [ ] **1.4** Commit: `feat(extraction): pipeline_pass_outputs + dispatched_phases state-machine column`

### Task 2: `pass_outputs_store` Service

**Files:** `app/services/pass_outputs_store.py`, `tests/unit/test_pass_outputs_store.py`

Critical contracts:
- `is_pass_already_resolved`: returns true iff a terminal row exists in ANY `execution_status` (`COMPLETE`, `FAILED`, or `SKIPPED`). Used by the per-pass Celery task's skip-if-done check — covers successful (`COMPLETE`), terminally-failed (`FAILED`, retries exhausted), and authorized-skipped (`SKIPPED`) passes. EMPTY/DEGRADED yield_status both count as COMPLETE for this check.
- `count_completed_passes`: counts distinct passes with `execution_status='COMPLETE'`.
- `count_terminal_passes(run_id, pass_names)`: counts distinct passes in `pass_names` with `execution_status` ∈ {`COMPLETE`, `FAILED`, `SKIPPED`}. Used by fan-in to know "all entity passes resolved." **A `FAILED` optional pass and an authorized `SKIPPED` pass both count as resolved.**

Tests must include:
- `test_empty_yield_is_completed` — `execution_status=COMPLETE, yield_status=EMPTY` → `is_pass_already_resolved=True`
- `test_skipped_pass_with_reason_persists` — round-trip `skip_reason='NO_UPSTREAM_ENDPOINTS'`
- `test_count_terminal_includes_failed_and_skipped` — verifies fan-in semantics
- `test_load_completed_excludes_failed_and_skipped` — verifies merge-input filtering
- `test_save_overwrites_same_run_pass` — upsert semantics

Steps:
- [ ] **2.1** Write tests
- [ ] **2.2** Implement service
- [ ] **2.3** Run tests
- [ ] **2.4** Commit: `feat(extraction): pass_outputs_store with SKIPPED + execution_status semantics`

---

## Chunk 2: Phase State Machine

### Task 3: `run_phase_dispatch` Service

**Files:** `app/services/run_phase_dispatch.py`, `tests/unit/test_run_phase_dispatch.py`

Phase entry shape (r4 — adds `result`):

```python
@dataclass
class PhaseEntry:
    state: Literal["claimed", "dispatched", "completed"]
    result: Literal["succeeded", "failed", "skipped"] | None  # null until completed
    task_id: str | None
    claimed_at: datetime
    dispatched_at: datetime | None
    completed_at: datetime | None
```

Helper contracts (r4 — all writes use compare-and-reset):

```python
def claim_phase(db, run_id, phase_name) -> bool:
    """Atomic INSERT-if-absent. UPDATE predicate:
        WHERE id = :run_id AND NOT (dispatched_phases ? :phase)
    Returns True iff this caller won. Single UPDATE statement; concurrent
    callers can't both win because PG row lock serializes."""

def mark_phase_dispatched(db, run_id, phase_name, task_id, expected_state="claimed") -> bool:
    """Compare-and-reset advance from claimed → dispatched. UPDATE predicate:
        WHERE id = :run_id
          AND dispatched_phases->:phase->>'state' = :expected_state
    Returns True if state advanced. Returns False (logs warning) if state
    has changed under us — reconciler may have re-claimed; a fresh dispatch
    is on the way."""

def mark_phase_terminal(db, run_id, phase_name, *, result: Literal["succeeded","failed","skipped"]) -> bool:
    """Compare-and-reset advance to completed. UPDATE predicate:
        WHERE id = :run_id
          AND dispatched_phases->:phase->>'state' IN ('claimed', 'dispatched')
    Sets state=completed, result=:result, completed_at=now. Idempotent on
    re-call: returns False if already completed (predicate excludes it)."""

def read_phase_state(db, run_id, phase_name) -> PhaseEntry | None: ...

def reclaim_stale_phase(
    db, run_id, phase_name, *,
    claim_threshold_s: int, dispatch_threshold_s: int,
) -> bool:
    """Compare-and-reset reclaim. The UPDATE predicate must match BOTH
    current state AND current claimed_at/dispatched_at — so a fresh claim
    by another worker since the reconciler read can't be clobbered.

    For stale 'claimed':
        UPDATE pipeline_runs
        SET dispatched_phases = dispatched_phases - :phase
        WHERE id = :run_id
          AND dispatched_phases->:phase->>'state' = 'claimed'
          AND (dispatched_phases->:phase->>'claimed_at')::timestamptz <
              :now - INTERVAL ':claim_threshold_s seconds'

    For stale 'dispatched': the reconciler's caller (not this helper) first
    inspects pipeline_pass_outputs for the latest attempt:
      - If latest attempt is FAILED with retries available: do NOT reclaim
        (Celery's countdown queue holds the retry).
      - If no pass-output row AND dispatched_at older than threshold: best-
        effort celery revoke(task_id, terminate=True), then UPDATE with
        same compare-and-reset semantics on dispatched_at.
    Returns True if reclaim happened."""

def is_run_cancelled(db, run_id) -> bool:
    """True if status ∈ {FAILED, COMPLETE, PARTIAL_COMPLETE, CANCELLED} OR
    row missing entirely (cancel_document hard-deletes)."""
```

Tests:
- `test_claim_first_caller_wins` — concurrent claims; only one returns True
- `test_claim_returns_false_if_already_claimed`
- `test_mark_dispatched_advances_state` — claimed → dispatched; verify timestamps + task_id
- `test_mark_dispatched_no_op_when_state_changed` — entry was already advanced under us; helper returns False, doesn't clobber
- `test_mark_terminal_records_result_succeeded` — verify `result='succeeded'`
- `test_mark_terminal_records_result_failed` — verify `result='failed'`
- `test_mark_terminal_records_result_skipped` — verify `result='skipped'`
- `test_mark_terminal_idempotent`
- `test_reclaim_stale_claimed_compare_and_reset` — pre-seed `claimed_at` 60s ago, threshold 30s, expect reset; THEN re-run with `claimed_at` refreshed to 0s ago, expect NO reset (predicate matches stale timestamp only)
- `test_reclaim_stale_dispatched_revokes_and_resets` — pre-seed `dispatched_at` 2× soft_time_limit ago; expect Celery revoke called + reset; THEN re-seed with fresh `dispatched_at` (10s ago), verify no revoke, no reset (compare-and-reset negative case)
- `test_is_run_cancelled_handles_missing_run`

> **Note:** Tests requiring pass_outputs awareness (e.g., "reconciler waits when a Celery retry is pending" or "reconciler promotes a dispatched phase to terminal when pass_outputs row exists") are deferred to Task 9 — the helper itself does not couple to pass_outputs (per the reclaim_stale_phase contract).

Steps:
- [ ] **3.1** Write tests
- [ ] **3.2** Implement
- [ ] **3.3** Run tests
- [ ] **3.4** Commit: `feat(extraction): run_phase_dispatch state machine with reclaim_stale_phase`

---

## Chunk 3: Per-Pass Execution Helper

### Task 4: Extract `_execute_pass_attempt` from `_run_single_pass`

The current `_run_single_pass` does six things in one function: skip-check, request-build, HTTP call, response-parse, retry, StageRun-write. The new architecture separates the retry boundary (Celery) from the attempt body. Extract a helper that does ONE attempt and returns rich metadata; preserve `_run_single_pass` as a thin wrapper for legacy/test use.

**Files:** `app/workers/pipeline.py`, `tests/unit/test_execute_pass_attempt.py`

Helper contract:

```python
@dataclass
class PassAttemptOutcome:
    execution_status: Literal["COMPLETE", "FAILED", "SKIPPED"]
    skip_reason: str | None
    yield_status: str | None
    pass_result: PassResult | None      # populated iff COMPLETE
    raw_response_payload: dict | None   # the literal /extract-pass JSON
    counts: dict | None                 # _count_pass_output result
    error: Exception | None             # PassRetryable / PassTransportError / PassTerminal


def _execute_pass_attempt(
    *, pipeline_run_id, pass_def, manifest, ontology, bundle_key, doc_json,
    upstream_refs, document_id,
) -> PassAttemptOutcome:
    """One attempt at one pass. Does NOT retry; the caller (Celery task or
    legacy retry loop) decides retry. Does NOT write StageRun or
    pipeline_pass_outputs — the caller persists. Returns rich metadata
    including the raw /extract-pass payload (for pass-output persistence)
    and the parsed PassResult (for merge inputs)."""
    if _should_skip(pass_def, upstream_refs, ontology):
        return PassAttemptOutcome(
            execution_status="SKIPPED",
            skip_reason="NO_UPSTREAM_ENDPOINTS",
            yield_status=None, pass_result=None, raw_response_payload=None,
            counts=None, error=None,
        )

    request_body = _build_extract_pass_request(
        bundle_key=bundle_key, pass_def=pass_def, doc_json=doc_json,
        upstream_refs=_select_upstream_refs_for_pass(pass_def, upstream_refs, ontology)
            if pass_def.input_mode == "document_plus_entity_refs" else None,
        document_id=document_id,
    )
    try:
        raw_payload = _call_extract_pass(request_body, timeout=settings.docling_graph_timeout)
    except (PassRetryable, PassTransportError, PassTerminal) as exc:
        return PassAttemptOutcome(
            execution_status="FAILED", skip_reason=None, yield_status=None,
            pass_result=None, raw_response_payload=None, counts=None, error=exc,
        )

    try:
        pass_result = _parse_pass_response(raw_payload, pass_def, manifest)
    except PassTerminal as exc:
        return PassAttemptOutcome(
            execution_status="FAILED", skip_reason=None, yield_status=None,
            pass_result=None, raw_response_payload=raw_payload, counts=None, error=exc,
        )

    # Attach upstream refs for document_plus_entity_refs passes — same logic
    # as _run_single_pass:628-644
    if pass_def.input_mode == "document_plus_entity_refs":
        # ... existing _run_single_pass:628-644 logic, copied verbatim ...
        pass

    pass_result.pre_merge_walk = _build_pre_merge_walk_summary(
        pass_result, pass_def, ontology, document_id,
    )
    yield_status_val = classify_yield(pass_result, pass_def, ontology)
    yield_str = (yield_status_val.value if hasattr(yield_status_val, "value")
                 else str(yield_status_val))
    counts = _count_pass_output(pass_result, pass_def, ontology)

    return PassAttemptOutcome(
        execution_status="COMPLETE", skip_reason=None, yield_status=yield_str,
        pass_result=pass_result, raw_response_payload=raw_payload,
        counts=counts, error=None,
    )
```

Refactor `_run_single_pass` to be a thin retry loop around `_execute_pass_attempt` — preserves all its tests, just replaces inline logic with the helper.

Tests:
- `test_skip_outcome_when_no_upstream_endpoints` — SKIPPED with skip_reason
- `test_complete_outcome_carries_raw_payload` — verifies the raw `/extract-pass` JSON is captured
- `test_complete_outcome_carries_pass_result` — PassResult populated with template_instance, walk, etc.
- `test_failed_outcome_on_pass_retryable` — error is PassRetryable, no pass_result, raw_payload may be None
- `test_failed_outcome_on_pass_transport_error` — error is PassTransportError (already in codebase from 2026-05-06 transport-retry work); same shape as PassRetryable but classifiable downstream
- `test_failed_outcome_on_pass_terminal_during_call` — terminal HTTP 4xx error from `_call_extract_pass`; raw_payload may be None; error is PassTerminal
- `test_failed_outcome_on_pass_terminal_during_parse` — successful HTTP response but malformed JSON; raw_payload IS captured (post-call, pre-parse failure); error is PassTerminal
- `test_existing_run_single_pass_tests_still_pass` — refactor must not break existing test suite

Steps:
- [ ] **4.1** Write tests
- [ ] **4.2** Extract `_execute_pass_attempt`; refactor `_run_single_pass` to use it
- [ ] **4.3** Run all unit tests including `test_run_single_pass.py`
- [ ] **4.4** Commit: `refactor(pipeline): extract _execute_pass_attempt — single-attempt helper for per-pass tasks`

---

## Chunk 4: Per-Pass Celery Task

### Task 5: `derive_ontology_graph_pass`

**Files:** `app/workers/pipeline.py`, `app/config.py`, `tests/unit/test_derive_ontology_graph_pass_task.py`

Config additions:

```python
pass_soft_time_limit: int = 3600
pass_concurrency_per_document: int = 2
phase_claim_stale_seconds: int = 30
reconciler_period_seconds: int = 60
```

Task body (r4 — explicit retry-exhaustion handling, terminal-only pass-output writes):

```python
@celery_app.task(
    bind=True, max_retries=3, default_retry_delay=60, queue="graph",
    soft_time_limit=settings.pass_soft_time_limit,
    name="app.workers.pipeline.derive_ontology_graph_pass",
)
@guard_stage_run("derive_ontology_graph_pass")
def derive_ontology_graph_pass(self, document_id: str, run_id: str, pass_name: str) -> dict:
    """One Celery task per pass attempt. Celery is the retry boundary.

    Pass-output write semantics (r4): pipeline_pass_outputs has at most ONE
    row per (run_id, pass_name) — the terminal one. Intermediate retry
    attempts only update StageRun. The fan-in counter
    (count_terminal_passes) thus counts pass-resolved passes, not failed
    attempts that may still retry.
    """
    db = _get_db()
    try:
        # 1. Cancel check
        if is_run_cancelled(db, run_id):
            return {"pass_name": pass_name, "skipped": "cancelled"}

        # 2. Already-resolved check
        if is_pass_already_resolved(db, run_id, pass_name):
            mark_phase_terminal(db, run_id, _phase_key(pass_name), result="succeeded")
            _try_advance_phase(db, document_id, run_id)
            return {"pass_name": pass_name, "skipped": "already_resolved"}

        # 3. Compare-and-reset advance to 'dispatched'. Returns False if
        # state changed under us (reconciler reclaimed); we proceed anyway —
        # any new dispatch will see our terminal write and skip.
        mark_phase_dispatched(db, run_id, _phase_key(pass_name), self.request.id)

        # 4. Execute one attempt
        run = db.get(PipelineRun, uuid.UUID(run_id))
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
            pipeline_run_id=run_id, pass_def=pass_def, manifest=manifest,
            ontology=ontology, bundle_key=bundle_key, doc_json=doc_json,
            upstream_refs=upstream_refs, document_id=document_id,
        )

        # 5. ALWAYS write StageRun (per-attempt audit; matches existing shape)
        stage_run_id = _write_stage_run(
            pipeline_run_id=run_id, pass_def=pass_def, attempt=attempt_n,
            execution_status=outcome.execution_status,
            yield_status=outcome.yield_status,
            skip_reason=outcome.skip_reason,
            counts=outcome.counts,
            error=str(outcome.error) if outcome.error else None,
        )

        # 6. Branch: COMPLETE / SKIPPED → terminal write + advance.
        #    FAILED with retry pending → no pass-output write; self.retry().
        #    FAILED with retry exhausted → terminal write + (terminalize if required).
        if outcome.execution_status in ("COMPLETE", "SKIPPED"):
            _save_terminal_pass_output(
                db, run_id=run_id, stage_run_id=stage_run_id, pass_name=pass_name,
                attempt=attempt_n, outcome=outcome,
            )
            db.commit()
            mark_phase_terminal(
                db, run_id, _phase_key(pass_name),
                result="succeeded" if outcome.execution_status == "COMPLETE" else "skipped",
            )
            _try_advance_phase(db, document_id, run_id)
            return {"pass_name": pass_name, "execution_status": outcome.execution_status}

        # FAILED branch
        is_retryable = isinstance(outcome.error, (PassRetryable, PassTransportError))
        retries_left = self.request.retries < self.max_retries

        if is_retryable and retries_left:
            # Pending Celery retry. Phase stays in 'dispatched' for the next
            # attempt. NO pipeline_pass_outputs write — fan-in counter must
            # not see this attempt as resolved.
            db.commit()  # commit StageRun audit
            raise self.retry(
                exc=outcome.error,
                countdown=_retry_delay(self.request.retries),
            )

        # Terminal failure path: either a non-retryable PassTerminal, OR a
        # retryable error after exhausting Celery retries. Both must record
        # a terminal pass-output row, mark phase terminal=failed, and
        # terminalize the run if the pass is required. r4: do NOT rely on
        # Celery's MaxRetriesExceededError — it would re-raise without
        # running this cleanup.
        _save_terminal_pass_output(
            db, run_id=run_id, stage_run_id=stage_run_id, pass_name=pass_name,
            attempt=attempt_n, outcome=outcome,
            override_status="FAILED",
            override_diagnostics_extra={"retry_exhausted": is_retryable},
        )
        db.commit()
        mark_phase_terminal(db, run_id, _phase_key(pass_name), result="failed")

        if pass_def.required:
            _update_summary_stage_run(
                db, run_id, "FAILED",
                error=(f"required pass {pass_name} "
                       f"{'retry-exhausted' if is_retryable else 'terminal failure'}"),
            )
            _terminalize_doc_and_run(document_id, run_id, "PARTIAL_COMPLETE")
            raise IngestFailed(f"Required pass {pass_name} terminal failure")

        # Optional terminal — phase done with result=failed; run continues
        _try_advance_phase(db, document_id, run_id)
        return {"pass_name": pass_name, "execution_status": "FAILED",
                "reason": "retry_exhausted" if is_retryable else "terminal"}
    finally:
        db.close()


def _save_terminal_pass_output(
    db, *, run_id, stage_run_id, pass_name, attempt, outcome,
    override_status: str | None = None,
    override_diagnostics_extra: dict | None = None,
) -> None:
    """Write the single terminal pipeline_pass_outputs row for this pass.
    Upsert by (run_id, pass_name) — overwrites any prior terminal write
    (defensive; same-pass should only terminalize once)."""
    diagnostics = (outcome.raw_response_payload or {}).get("diagnostics", {}) or {}
    if override_diagnostics_extra:
        diagnostics = {**diagnostics, **override_diagnostics_extra}
    save_pass_output(
        db, pipeline_run_id=run_id, stage_run_id=stage_run_id,
        pass_name=pass_name, attempt=attempt,
        execution_status=override_status or outcome.execution_status,
        skip_reason=outcome.skip_reason,
        yield_status=outcome.yield_status,
        extract_pass_response=outcome.raw_response_payload or {},
        primary_entities_extracted=(outcome.counts or {}).get("primary_entities", 0),
        bridge_entities_extracted=(outcome.counts or {}).get("bridge_entities", 0),
        relationships_extracted=(outcome.counts or {}).get("relationships_extracted", 0),
        relationships_rejected=(outcome.counts or {}).get("relationships_rejected", 0),
        diagnostics=diagnostics,
        field_provenance=(outcome.raw_response_payload or {}).get("field_provenance", []),
    )


def _phase_key(pass_name: str) -> str:
    """Map pass_name to the dispatched_phases key. Entity passes use
    'entity_pass_<name>' prefix; system_links and merge are top-level."""
    if pass_name == "system_links":
        return "system_links"
    return f"entity_pass_{pass_name}"


def _try_advance_phase(db, document_id, run_id) -> None:
    """Decide whether to dispatch the next pass / system_links / merge.
    All entity passes and system_links resolved → claim merge phase + dispatch
    derive_ontology_graph_merge. Atomic via claim_phase."""
    run = db.get(PipelineRun, uuid.UUID(run_id))
    manifest = load_bundle_manifest(run.ontology_bundle_key)
    entity_passes = [p.name for p in manifest.passes if not p.depends_on]

    # First: dispatch the NEXT entity pass if cap allows
    in_flight = sum(
        1 for k, v in run.dispatched_phases.items()
        if k.startswith("entity_pass_") and v.get("state") in ("claimed", "dispatched")
    )
    if in_flight < settings.pass_concurrency_per_document:
        completed_or_terminal = {
            k.removeprefix("entity_pass_") for k, v in run.dispatched_phases.items()
            if k.startswith("entity_pass_") and v.get("state") == "completed"
        }
        in_flight_names = {
            k.removeprefix("entity_pass_") for k, v in run.dispatched_phases.items()
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

    # All entity passes resolved → dispatch system_links if not yet
    n_resolved = count_terminal_passes(db, run_id, entity_passes)
    if n_resolved >= len(entity_passes):
        sl_state = read_phase_state(db, run_id, "system_links")
        if sl_state is None:
            _claim_and_dispatch_pass(db, document_id, run_id, "system_links")
            return

    # system_links resolved → dispatch merge
    sl_pass = load_pass_output(db, run_id, "system_links")
    if sl_pass and sl_pass.execution_status in ("COMPLETE", "SKIPPED", "FAILED"):
        merge_state = read_phase_state(db, run_id, "merge")
        if merge_state is None:
            if claim_phase(db, run_id, "merge"):
                async_result = derive_ontology_graph_merge.delay(document_id, run_id)
                mark_phase_dispatched(db, run_id, "merge", async_result.id)


def _claim_and_dispatch_pass(db, document_id, run_id, pass_name) -> None:
    """Single code path for initial dispatch and follow-up dispatch.
    Claims phase first; only dispatches if claim won. Marks dispatched after
    .delay() returns (so a crash between claim and delay leaves the phase
    in 'claimed' state — recoverable by reconciler)."""
    phase_key = _phase_key(pass_name)
    if not claim_phase(db, run_id, phase_key):
        return  # someone else won
    async_result = derive_ontology_graph_pass.delay(document_id, run_id, pass_name)
    mark_phase_dispatched(db, run_id, phase_key, async_result.id)
```

Tests (each its own pytest function — r4 expanded for retry-exhaustion + terminal-only-write semantics):

Cancel / skip / already-resolved:
- `test_skip_when_run_cancelled` — pre-flag run FAILED; verify `_execute_pass_attempt` not called
- `test_skip_when_run_missing` — cancel hard-deletes; task tolerates
- `test_skip_when_already_resolved_advances_phase` — pre-seed terminal pass-output row; verify `mark_phase_terminal` + `_try_advance_phase` called

Successful completion path:
- `test_complete_writes_terminal_pass_output_and_marks_phase_succeeded` — verifies single row written + phase result='succeeded'
- `test_persists_full_extract_pass_response` — `extract_pass_response_json` carries raw payload
- `test_skipped_pass_records_skip_reason_and_phase_result_skipped` — `_should_skip` true; pass-output execution_status=SKIPPED, skip_reason=NO_UPSTREAM_ENDPOINTS, phase result='skipped'
- `test_skipped_pass_counts_as_phase_terminal_for_fanin` — fan-in advance fires

Retryable failure path (r4 — explicit terminal-only write):
- `test_pass_retryable_with_retries_left_does_not_write_pass_output` — error is PassRetryable, attempt 1 of 3; verify NO `pipeline_pass_outputs` row written; verify `self.retry()` raised
- `test_pass_transport_error_with_retries_left_does_not_write_pass_output` — same for PassTransportError
- `test_pass_retryable_writes_stagerun_audit_per_attempt` — even when no pass-output row, every attempt has its own StageRun row

Retry exhaustion (r4 — primary new test class):
- `test_retry_exhausted_writes_terminal_pass_output_failed` — 3 PassRetryable failures; verify pass-output row written AFTER exhaustion with execution_status=FAILED, diagnostics.retry_exhausted=True
- `test_retry_exhausted_marks_phase_result_failed` — verify phase entry shows `state=completed, result=failed`
- `test_required_pass_retry_exhausted_terminalizes_run` — required=True, retries exhausted; verify `_terminalize_doc_and_run` called, summary StageRun marked FAILED, IngestFailed raised. **Critical**: this is the path Celery's `MaxRetriesExceededError` would otherwise short-circuit.
- `test_optional_pass_retry_exhausted_does_not_terminalize` — required=False, retries exhausted; verify run stays PROCESSING, phase result='failed', `_try_advance_phase` called

Terminal failure path (non-retryable):
- `test_pass_terminal_writes_terminal_pass_output_immediately` — error is PassTerminal; verify pass-output written without retries
- `test_required_pass_terminal_terminalizes_run`
- `test_optional_pass_terminal_does_not_terminalize`
- `test_pass_terminal_during_parse_carries_raw_payload` — parse-stage failure; pass-output has raw_response_payload populated even though pass_result is None

Attempt counter:
- `test_attempt_field_reflects_celery_retry_counter` — terminal write's `attempt` column = `self.request.retries + 1` of the terminal attempt

Steps:
- [ ] **5.1** Add config settings
- [ ] **5.2** Write tests
- [ ] **5.3** Implement task + `_phase_key`, `_try_advance_phase`, `_claim_and_dispatch_pass`, `_update_summary_stage_run` helpers
- [ ] **5.4** Run tests
- [ ] **5.5** Commit: `feat(extraction): derive_ontology_graph_pass — Celery-retry boundary, full state-machine integration`

---

## Chunk 5: Merge + Summary Row Lifecycle

### Task 6: `derive_ontology_graph_merge`

**Files:** `app/workers/pipeline.py`, `tests/unit/test_derive_ontology_graph_merge_task.py`

Body:

```python
@celery_app.task(
    bind=True, max_retries=1, default_retry_delay=30, queue="graph",
    soft_time_limit=settings.graph_soft_time_limit,
    name="app.workers.pipeline.derive_ontology_graph_merge",
)
@guard_stage_run("derive_ontology_graph_merge")
def derive_ontology_graph_merge(self, document_id: str, run_id: str) -> dict:
    """Fan-in. Loads COMPLETE pass outputs from pipeline_pass_outputs, rehydrates
    via _parse_pass_response, runs merge_and_resolve + graph imports.

    Rollback contract (r4 — preserves the existing pipeline.py:5191 behavior):
    `GraphWriteTracker` records whether a graph mutation has happened. On any
    exception inside the try block, before raising, we call
    `_attempt_rollback(document_id)` IF `tracker.first_mutation_recorded` —
    so the next merge attempt (Celery retry, or operator-driven graph_only
    reingest) starts from a clean graph state. Without this, partial node
    imports from attempt 1 + full imports from attempt 2 would double-write.
    """
    db = _get_db()
    tracker = GraphWriteTracker()
    try:
        if is_run_cancelled(db, run_id):
            return {"merge": "skipped_cancelled"}

        # COMPLETE-without-output detection
        _assert_stage_run_pass_output_consistency(db, run_id)

        run = db.get(PipelineRun, uuid.UUID(run_id))
        bundle_key = run.ontology_bundle_key
        manifest = load_bundle_manifest(bundle_key)
        ontology = load_ontology(bundle_key=bundle_key)

        gate = check_required_pass_gate(run_id)
        if not gate.passed:
            _update_summary_stage_run(db, run_id, "FAILED",
                                      error=f"Required passes failed: {gate.failures}")
            mark_phase_terminal(db, run_id, "merge", result="failed")
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
        # Phases 1-4 graph import — each marks tracker on first mutation
        identity_to_rid = _import_graph_phase_nodes(
            merged, ontology, document_id, tracker, provenance_envelope,
        )
        _import_graph_phase_domain_edges(merged, ontology, tracker, provenance_envelope)
        _ensure_structural_document_vertex(document_id)
        _import_graph_phase_structural_edges(
            merged, identity_to_rid, document_id, run_id, tracker,
        )
        # ... existing _upsert_document_graph_extraction call ...

        _update_summary_stage_run(db, run_id, "COMPLETE")
        mark_phase_terminal(db, run_id, "merge", result="succeeded")

        from celery import chain as celery_chain
        celery_chain(
            collect_derivations.si(document_id, run_id),
            derive_structure_links.si(document_id, run_id),
            derive_canonicalization.si(document_id, run_id),
            finalize_document.si(document_id, run_id),
        ).apply_async()

        return {"merge": "ok"}
    except Exception as exc:
        # r4: rollback any partial graph state BEFORE raising, so
        # Celery retry (max_retries=1) starts from a clean slate. Match
        # the existing pipeline.py:5191 contract: rollback gated by
        # tracker.first_mutation_recorded so we don't delete data from
        # a prior successful run when the failure happened pre-mutation.
        if tracker.first_mutation_recorded:
            try:
                _attempt_rollback(document_id)
                logger.info(
                    "derive_ontology_graph_merge: rolled back partial graph "
                    "state for doc=%s run=%s before re-raising %s",
                    document_id, run_id, type(exc).__name__,
                )
            except Exception as rollback_exc:
                logger.exception(
                    "derive_ontology_graph_merge: rollback ALSO failed for "
                    "doc=%s run=%s — original exc=%r rollback exc=%r",
                    document_id, run_id, exc, rollback_exc,
                )
        # Mark merge phase as failed (compare-and-reset is idempotent if a
        # parallel attempt already ran)
        try:
            mark_phase_terminal(db, run_id, "merge", result="failed")
        except Exception:
            logger.exception("merge: mark_phase_terminal failed in error path")
        raise
    finally:
        db.close()


def _update_summary_stage_run(db, run_id, status: str, *, error: str | None = None) -> None:
    """Update the derive_ontology_graph stage SUMMARY row (the one created
    as RUNNING in the dispatcher). Match by (run_id, stage_name='derive_ontology_graph',
    pass_name IS NULL). Sets finished_at on terminal status."""
    ...
```

Tests:
- `test_merge_loads_only_complete_pass_outputs` — FAILED + SKIPPED rows excluded from rehydration
- `test_merge_marks_summary_stagerun_complete` — RUNNING row updated to COMPLETE
- `test_merge_marks_summary_stagerun_failed_on_gate_failure`
- `test_merge_dispatches_downstream_chain` — verify celery_chain with the 4 stages
- `test_complete_without_output_raises_worker_invariant_error` — drift detection
- `test_skipped_passes_excluded_from_merge_inputs`
- **`test_merge_failure_post_mutation_calls_rollback_before_raising`** (r4) — mock `_import_graph_phase_domain_edges` to raise; verify `tracker.first_mutation_recorded` triggers `_attempt_rollback(document_id)` BEFORE the exception propagates
- **`test_merge_failure_pre_mutation_skips_rollback`** (r4) — exception before any `tracker.mark()`; verify `_attempt_rollback` NOT called (matches existing pipeline.py:5191 gated behavior)
- **`test_merge_retry_after_rollback_starts_from_clean_state`** (r4) — simulate retry: verify second attempt's import phases see no leftover nodes from attempt 1
- **`test_merge_phase_marked_failed_on_rollback_path`** (r4) — phase entry shows `state=completed, result=failed` after error path

Steps:
- [ ] **6.1** Write tests
- [ ] **6.2** Implement merge + `_assert_stage_run_pass_output_consistency` + `_update_summary_stage_run` + `_rehydrate_pass_result`
- [ ] **6.3** Run tests
- [ ] **6.4** Commit: `feat(extraction): derive_ontology_graph_merge with summary-row lifecycle + drift detection`

### Task 7: Refactor `start_ingest_pipeline` and `reingest_graph_only`

**Files:** `app/workers/pipeline.py`

Outer chain ends at `derive_ontology_graph`. Signatures preserved as `(document_id, run_id)` for every stage.

```python
# start_ingest_pipeline — was ~16-stage chain; now ends at derive_ontology_graph
result = celery_chain(
    prepare_document.si(doc_id, run_id),
    detect_and_translate.si(doc_id, run_id),
    derive_document_metadata.si(doc_id, run_id),
    purge_document_derivations.si(doc_id, run_id),
    derive_picture_descriptions.si(doc_id, run_id),
    derive_text_chunks_and_embeddings.si(doc_id, run_id),
    derive_image_embeddings.si(doc_id, run_id),
    derive_document_anchors.si(doc_id, run_id),
    derive_ontology_graph.si(doc_id, run_id),  # CHAIN ENDS HERE — merge dispatches the rest
).apply_async()
```

Same shape for `reingest_graph_only` (skip the embeddings stages it already skips today).

Tests:
- `test_start_ingest_pipeline_chain_ends_at_derive_ontology_graph` — verify task names in chain
- `test_reingest_graph_only_chain_ends_at_derive_ontology_graph`
- `test_all_chain_stages_use_doc_id_run_id_signature` — guards against the r2-style regression

Steps:
- [ ] **7.1** Write tests
- [ ] **7.2** Trim outer chains; preserve signatures; add `# CHANGED 2026-05-06:` markers for grep-rollback
- [ ] **7.3** Run tests
- [ ] **7.4** Commit: `refactor(pipeline): outer chain ends at derive_ontology_graph; merge dispatches downstream`

---

## Chunk 6: Dispatcher + Summary Row Creation

### Task 8: `derive_ontology_graph` Becomes Dispatcher

**Files:** `app/workers/pipeline.py:5366`, `tests/unit/test_derive_ontology_graph_dispatch.py`

Body:

```python
@celery_app.task(
    bind=True, max_retries=2, default_retry_delay=60, queue="graph",
    soft_time_limit=600, name="app.workers.pipeline.derive_ontology_graph",
)
@guard_stage_run("derive_ontology_graph")
def derive_ontology_graph(self, document_id: str, run_id: str | None = None) -> dict:
    """Dispatcher. Creates the derive_ontology_graph summary StageRun as
    RUNNING (preserves the existing finalize_document REQUIRED_STAGES gate
    — pipeline.py:6029). Kicks off the first ``pass_concurrency_per_document``
    entity passes via the SAME claim/dispatch flow used for follow-ups."""
    pipeline_run_id, run_document_id = _resolve_run_and_document(run_id, document_id)

    # Orphaned-run safety net (matches existing pipeline.py:5158)
    db = _get_db()
    try:
        run = db.get(PipelineRun, uuid.UUID(pipeline_run_id))
        if run is None:
            return {"stage": "derive_ontology_graph", "status": "skipped",
                    "reason": "orphaned_run"}

        # Create RUNNING summary StageRun (matches pipeline.py:5171)
        stage_summary = StageRun(
            pipeline_run_id=uuid.UUID(pipeline_run_id),
            stage_name="derive_ontology_graph",
            pass_name=None,
            attempt=self.request.retries + 1,
            status="RUNNING",
            execution_status="RUNNING",
            started_at=datetime.utcnow(),
        )
        db.add(stage_summary)
        db.commit()

        bundle_key = run.ontology_bundle_key
        manifest = load_bundle_manifest(bundle_key)
        entity_passes = [p.name for p in manifest.passes if not p.depends_on]

        # Initial dispatch — same claim/dispatch flow used by follow-ups.
        dispatched: list[str] = []
        for pass_name in entity_passes[:settings.pass_concurrency_per_document]:
            if claim_phase(db, pipeline_run_id, _phase_key(pass_name)):
                async_result = derive_ontology_graph_pass.delay(
                    run_document_id, pipeline_run_id, pass_name,
                )
                mark_phase_dispatched(db, pipeline_run_id, _phase_key(pass_name),
                                      async_result.id)
                dispatched.append(pass_name)
    finally:
        db.close()

    return {
        "stage": "derive_ontology_graph",
        "status": "dispatched",
        "summary_stage_run_status": "RUNNING",
        "passes_dispatched": dispatched,
        "passes_pending": [p for p in entity_passes if p not in dispatched],
    }
```

Tests:
- `test_dispatcher_creates_summary_stagerun_as_running` — verify RUNNING row exists post-dispatch
- `test_dispatcher_uses_same_claim_dispatch_flow_as_followups` — same code path; mock claim_phase, expect calls
- `test_dispatcher_does_not_dispatch_system_links_directly` — system_links waits for fan-in
- `test_dispatcher_handles_orphaned_run` — missing PipelineRun → returns skipped
- `test_dispatcher_dispatches_only_concurrency_cap_initial_passes` — verify call count == cap

Steps:
- [ ] **8.1** Write tests
- [ ] **8.2** Implement
- [ ] **8.3** Run tests
- [ ] **8.4** Commit: `refactor(extraction): derive_ontology_graph dispatcher — RUNNING summary + atomic initial dispatch`

---

## Chunk 7: Reconciler

### Task 9: `reconcile_ontology_graph_runs`

**Files:** `app/workers/pipeline.py`, `app/workers/celery_app.py`, `tests/unit/test_reconcile_ontology_graph_runs.py`

The reconciler scans `PROCESSING` runs every 60s. For each, it inspects `dispatched_phases` and `pipeline_pass_outputs` together (r4 — must not race Celery's countdown queue):

1. **Stale `claimed` repair**: phase entry with `state=claimed` and `claimed_at < now - phase_claim_stale_seconds` → call `reclaim_stale_phase` (compare-and-reset on state + timestamp), then `_try_advance_phase` to re-dispatch.

2. **Stale `dispatched` repair (r4 — pending-retry-aware)**: phase entry with `state=dispatched` and `dispatched_at < now - 2*pass_soft_time_limit`, BUT only reclaim when **all** of:
   - No terminal `pipeline_pass_outputs` row exists for this pass (would indicate the task DID complete and just failed to mark phase terminal — handle that separately, see #4)
   - No StageRun row from a pending retry attempt: query the latest StageRun for this pass; if `execution_status=FAILED` AND `attempt < pass_max_retries + pass_max_transport_retries` AND `finished_at < now - countdown_buffer`, the task scheduled a Celery retry that hasn't fired yet — wait
   - Otherwise: best-effort `celery.revoke(task_id, terminate=True)`, then compare-and-reset

3. **Promote completed-but-not-marked-terminal**: phase entry with `state=dispatched` AND a terminal `pipeline_pass_outputs` row exists → `mark_phase_terminal` directly (the task crashed between the save and the mark; the work is done).

4. **Stuck-without-advance repair**: `count_terminal_passes` meets phase threshold but no follow-up dispatched (finisher crashed between save and `_try_advance_phase`) → `_try_advance_phase`.

Tests:
- `test_repairs_stale_claimed_compare_and_reset_dispatches`
- `test_does_not_touch_recent_claimed` — `claimed_at` within threshold; no action
- `test_repairs_stale_dispatched_revokes_and_redispatches` — no pass-output, no pending retry, > threshold
- `test_does_not_touch_recent_dispatched`
- **`test_does_not_reclaim_dispatched_with_pending_retry`** (r4) — pre-seed a FAILED StageRun with `attempt < max_retries` and `finished_at` within Celery's countdown window; verify NO revoke, NO reclaim (also covers `test_reclaim_dispatched_with_failed_attempt_pending_retry_does_not_reset` intent from Task 3)
- **`test_does_not_reclaim_dispatched_when_attempts_exhausted_but_pass_output_exists`** (r4) — pre-seed terminal pass-output row; verify reconciler calls `mark_phase_terminal` instead of reclaim (also covers `test_reclaim_dispatched_with_pass_output_promotes_to_terminal` intent from Task 3)
- `test_promotes_completed_but_not_marked_terminal` — task wrote pass-output but didn't mark phase terminal; reconciler does it
- `test_dispatches_missing_follow_up` — all entity passes resolved, no `system_links` phase entry; reconciler dispatches it
- `test_skips_terminalized_runs` — run.status=FAILED; no action
- **`test_compare_and_reset_does_not_clobber_fresh_claim_under_us`** (r4) — race scenario: reconciler reads stale entry, but a fresh `claim_phase` runs before reconciler's UPDATE; verify reconciler's UPDATE no-ops because predicate matches stale-only timestamps

Beat schedule:

```python
# app/workers/celery_app.py
celery_app.conf.beat_schedule["reconcile-ontology-graph-runs"] = {
    "task": "app.workers.pipeline.reconcile_ontology_graph_runs",
    "schedule": settings.reconciler_period_seconds,
}
```

Steps:
- [ ] **9.1** Write tests
- [ ] **9.2** Implement
- [ ] **9.3** Run tests
- [ ] **9.4** Commit: `feat(extraction): reconcile_ontology_graph_runs — repair stale claimed/dispatched + missed advance`

---

## Chunk 8: Cancellation Hardening

### Task 10: Cancel-Tolerance with Targeted FK Swallow

**Files:** `app/services/pass_outputs_store.py`, `app/workers/pipeline.py`

`save_pass_output` must distinguish "run was hard-deleted by cancel" from other integrity errors. Specific FK-violation match:

```python
from psycopg2.errors import ForeignKeyViolation

try:
    db.execute(stmt)
    db.commit()
except IntegrityError as exc:
    db.rollback()
    # Only swallow the specific "pipeline_run_id FK to pipeline_runs" violation
    # (cancel_document hard-deleted the run). Unique-constraint violations,
    # JSON serialization errors, enum/length errors, and other integrity
    # failures must still raise.
    inner = getattr(exc.orig, "diag", None)
    if isinstance(exc.orig, ForeignKeyViolation) and \
       inner is not None and \
       inner.constraint_name == "pipeline_pass_outputs_pipeline_run_id_fkey":
        logger.warning(
            "save_pass_output: run %s no longer exists (cancelled/deleted) — "
            "discarding pass output for %s",
            pipeline_run_id, pass_name,
        )
        return
    raise  # any other integrity error must surface
```

Plus: pass task adds a second cancel-check immediately before save (catches mid-extraction cancels with low race-window):

```python
# In derive_ontology_graph_pass, between _execute_pass_attempt and save:
if is_run_cancelled(db, run_id):
    return {"pass_name": pass_name, "skipped": "cancelled_mid_extraction"}
```

Tests:
- `test_save_swallows_specific_pipeline_run_fk_violation` — pre-delete the run; verify save returns cleanly
- `test_save_raises_on_unique_constraint_violation` — terminal-only schema is unique on `(run_id, pass_name)`; ship two rows for same `(run, pass)` via direct INSERT bypassing upsert; verify raise
- `test_save_raises_on_json_serialization_error` — pass non-JSON-serializable diagnostics; verify raise
- `test_save_raises_on_other_fk_violation` — invalid stage_run_id FK (not the swallowed one); verify raise
- `test_pass_task_aborts_when_cancel_lands_mid_extraction` — flip status mid-test; verify pass-output not written

Steps:
- [ ] **10.1** Write tests
- [ ] **10.2** Implement targeted FK match
- [ ] **10.3** Run tests
- [ ] **10.4** Commit: `feat(extraction): targeted FK-violation swallow in save_pass_output (cancel-only)`

---

## Chunk 9: Integration + Smoke

### Task 11: End-to-End Integration Tests

**File:** `tests/integration/test_per_pass_db_fanin_e2e.py`

Each test its own pytest function:

1. `test_full_run_completes_through_all_phases` — happy path; all 12 passes COMPLETE/SKIPPED/EMPTY; merge runs; downstream chain runs; doc=COMPLETE; summary StageRun for `derive_ontology_graph` ends as COMPLETE
2. `test_optional_pass_retry_exhaustion_does_not_block_run` — one optional pass exhausts Celery retries; run still completes; merge excludes the failed pass; phase entry `state=completed, result=failed`
3. `test_required_pass_retry_exhaustion_terminalizes_with_failed_summary` — `system_links` exhausts retries; verify (a) `_terminalize_doc_and_run` called, (b) summary StageRun marked FAILED, (c) finalize_document gate sees consistent state. **r4: confirms the explicit retry-exhaustion path runs cleanup, not Celery's `MaxRetriesExceededError` short-circuit.**
4. `test_authorized_skip_treated_as_terminal_for_fanin` — `system_links` has no upstream endpoints; pass task records SKIPPED with skip_reason; phase result='skipped'; fan-in proceeds to merge; merge runs without system_links inputs
5. `test_skip_already_resolved_passes_on_retry` — pre-seed 5 terminal pass-output rows; dispatch run; verify those 5 pass tasks short-circuit and advance phase
6. `test_complete_without_output_fails_loud` — pre-seed StageRun=COMPLETE for radar_identity without pass-output row; merge raises `WorkerInvariantError`
7. `test_cancel_during_active_pass_aborts_gracefully` — dispatch run; mid-extraction call `cancel_document`; verify no crash; pass task discards output via targeted FK guard
8. `test_downstream_stages_do_not_run_before_merge` — verify `collect_derivations` does NOT run until `derive_ontology_graph_merge` returns
9. **`test_merge_failure_post_mutation_rolls_back_partial_graph_state`** (r4) — mock graph import to fail after first mutation; verify `_attempt_rollback` called before Celery retry; verify retry imports cleanly (no doubled nodes)
10. `test_reconciler_recovers_from_stale_claimed` — claim phase with `claimed_at` 60s ago; reconciler reclaims + re-dispatches
11. `test_reconciler_recovers_from_stale_dispatched_no_pending_retry` — `dispatched_at` past 2× soft_time_limit, no pending retry; revoke + re-dispatch
12. **`test_reconciler_waits_when_pending_retry_in_countdown`** (r4) — `dispatched` stale BUT latest StageRun shows FAILED with attempt < max + recent finished_at (Celery countdown queue); verify NO revoke
13. `test_finalize_document_sees_derive_ontology_graph_stage_complete` — summary StageRun lifecycle correct end-to-end

Steps:
- [ ] **11.1** Implement test fixtures (mocked `_call_extract_pass`, seed helpers)
- [ ] **11.2** Implement all 11 test cases
- [ ] **11.3** Run; expect all pass
- [ ] **11.4** Commit: `test(extraction): E2E DB-fan-in tests — happy path + 10 failure modes`

### Task 12: Live Smoke

- [ ] Pick a small completed doc (e.g., `Radar Basics.pdf`)
- [ ] Graph-only reingest
- [ ] Watch `worker-graph` + `worker` + `beat` logs in parallel
- [ ] Verify:
  - First 2 entity passes start at near-identical timestamps (concurrency cap)
  - As each pass finishes, exactly one new pass dispatches
  - `system_links` starts only after all 11 entity passes have terminal `pipeline_pass_outputs` rows
  - `derive_ontology_graph_merge` runs once
  - Summary StageRun for `derive_ontology_graph` transitions RUNNING → COMPLETE
  - `collect_derivations` runs after merge
  - Run reaches COMPLETE
  - Reconciler logs no actions (everything advanced cleanly)
- [ ] Compare entity counts to a pre-change baseline (>10% deviation = investigate)

---

## Chunk 10: Documentation + Cutover

### Task 13: Docs

**Files:** `VERIFICATION_CHECKLIST.md`, `README.md`, `docs/operational/per-pass-fanin-rollout.md`

VERIFICATION_CHECKLIST rows (additions to § 1.2 + § 2.1):

| Row | What breaks | Verify | Phase |
|---|---|---|---|
| Per-pass Celery task isolation | One bad pass blew the entire 8h ontology-graph stage | Each pass runs in its own Celery task with `pass_soft_time_limit=3600s`; per-pass `StageRun` + `pipeline_pass_outputs` row | 4.x |
| `dispatched_phases` state machine (claimed/dispatched/completed) | Dispatcher crash between DB claim and broker accept silently strands the run | Phase entries record state + timestamps + task_id; reconciler repairs stale `claimed` (>30s) and stale `dispatched` (>2× pass_soft_time_limit + no pass-output row) | 4.x |
| `derive_ontology_graph` summary StageRun lifecycle | finalize_document gate fails because the stage row is missing or stuck RUNNING | Dispatcher writes RUNNING; merge writes COMPLETE; required-pass terminalization writes FAILED. Same stage_name as before so finalize_document's REQUIRED_STAGES check is satisfied | 4.x |
| Authorized SKIPPED preserved through fan-in | system_links with no upstream endpoints recorded as FAILED — semantic regression | Pass task writes execution_status=SKIPPED with skip_reason; fan-in counts SKIPPED as terminal but excludes from merge inputs | 4.x |
| COMPLETE-without-output detection | Crash between StageRun COMPLETE write and pass-output save lets merge run with missing inputs | `_assert_stage_run_pass_output_consistency` cross-checks before merge; raises WorkerInvariantError on drift | 4.x |
| Targeted FK swallow on cancel | Broad IntegrityError swallow hides unique-constraint or JSON errors | save_pass_output only swallows the specific `pipeline_pass_outputs_pipeline_run_id_fkey` violation; all other integrity errors raise | 4.x |
| Per-document pass concurrency cap | 11 entity passes × 4 LLM connections = 44 concurrent calls per doc swamps Ollama | `pass_concurrency_per_document` (default 2) caps in-flight entity passes via shared claim/dispatch flow for initial + follow-up | 4.x |

README env vars:

```
PASS_SOFT_TIME_LIMIT=3600
PASS_CONCURRENCY_PER_DOCUMENT=2
RECONCILER_PERIOD_SECONDS=60
PHASE_CLAIM_STALE_SECONDS=30
```

Rollout doc: schema migration is additive (safe ahead of code); code deploy bounces worker-graph + worker + beat (the reconciler runs there); in-flight runs at deploy time will need graph-only reingest; rollback is revert + redeploy (`pipeline_pass_outputs` orphaned-but-harmless via CASCADE).

Steps:
- [ ] **13.1** Update VERIFICATION_CHECKLIST.md
- [ ] **13.2** Update README.md
- [ ] **13.3** Write rollout doc
- [ ] **13.4** Commit: `docs(extraction): per-pass DB fan-in r3 — checklist, env vars, rollout`

---

## Open Questions (r4 — narrower than r3)

1. **`dispatched` stale threshold** — 2× `pass_soft_time_limit` is conservative. The pending-retry awareness in r4 reduces the false-positive risk; threshold may be tunable down to 1.5× once measured.
2. **Reconciler beat host** — `beat` service runs the schedule; reconciler task runs on the `graph` queue. Verify worker-graph load.
3. **JSONB size for large `extract_pass_response_json`** — typical <100KB; large docs may push 1–5MB. PG TOAST handles transparently. Defer until measured.
4. **Per-document concurrency cap interaction with per-bundle parallelism** — currently no cap on concurrent documents. Total worker-graph load = `pass_concurrency_per_document × concurrent_documents`.
5. **Retry-countdown awareness window** — the reconciler's pending-retry check (Task 9) compares the latest StageRun's `finished_at` against `now - countdown_buffer`. The `countdown_buffer` value matches the `_retry_delay()` used by `self.retry()` (currently `30 * 2^retries` capped at 300s). Worth pinning these together with a shared helper.

---

## Estimated Effort (r4)

| Chunk | Tasks | r3 Effort | r4 Effort | Note |
|---|---|---|---|---|
| 1. Schema + persistence | 1, 2 | 1 day | 1 day | terminal-only schema simpler |
| 2. Phase state machine | 3 | 1 day | 1.25 days | compare-and-reset adds complexity |
| 3. `_execute_pass_attempt` extraction | 4 | 1 day | 1 day | + PassTerminal classification tests |
| 4. Per-pass task | 5 | 2 days | 2.25 days | + retry-exhaustion explicit handling + tests |
| 5. Merge + chain | 6, 7 | 1.5 days | 1.75 days | + GraphWriteTracker rollback path + tests |
| 6. Dispatcher + summary row | 8 | 0.75 day | 0.75 day | unchanged |
| 7. Reconciler | 9 | 1 day | 1.5 days | + pending-retry awareness + compare-and-reset race tests |
| 8. Cancel hardening | 10 | 0.5 day | 0.5 day | + other-FK test |
| 9. Integration tests | 11, 12 | 2 days | 2 days | shifted: more unit coverage, similar integration |
| 10. Docs + cutover | 13 | 0.5 day | 0.5 day | unchanged |
| **Total** | **13 tasks** | **~11.25 days** | **~12.5 days** | |

Practical estimate **13–15 working days** with buffer for the r4 specifics (compare-and-reset SQL, pending-retry timing window, rollback-path test fixtures).

---

## Changelog

- **r4 (2026-05-06):** Address review of r3.
  - Phase entries gain `result` field (`succeeded`/`failed`/`skipped`).
  - Phase helpers use compare-and-reset SQL predicates (state + timestamp matching) so a fresh claim can't be clobbered by a stale reconciler read.
  - Pass-output writes are terminal-only — at most one row per `(run_id, pass_name)`. Intermediate retry attempts only update StageRun. Resolves the "FAILED-attempt counted as resolved" race.
  - Per-pass task explicitly handles retry exhaustion before `self.retry()`. Required-pass cleanup runs even on Celery `max_retries` reached.
  - `_execute_pass_attempt` test list adds explicit PassTerminal classification (during call vs during parse).
  - Merge task gets explicit `GraphWriteTracker` + `_attempt_rollback` contract on exception (preserves pipeline.py:5191 behavior).
  - Reconciler gains pending-retry awareness — does not revoke a `dispatched` task whose latest StageRun shows a FAILED attempt with retries remaining and a recent `finished_at`.
  - Schema unique constraint changes from `(run_id, pass_name, attempt)` to `(run_id, pass_name)` — terminal-only.
  - Prerequisites section added: `PassTransportError` and `pass_max_transport_retries` were shipped 2026-05-06; this plan builds on them.
- **r3 (2026-05-06):** First state-machine version. Replaced flat task-id map with state-object; added `derive_ontology_graph` summary StageRun lifecycle; switched persistence from invented `PassResult` attributes to raw `_call_extract_pass` payload via new `_execute_pass_attempt` helper; preserved authorized SKIPPED; signatures restored.
- **r2 (2026-05-06):** Replaced Celery chord with DB-backed fan-in. Did not adequately address dispatch-state recoverability (caught by review).
- **r1 (2026-05-06):** Initial draft using Celery chord (rejected — chord+Redis unreliable per pipeline.py:2052).
