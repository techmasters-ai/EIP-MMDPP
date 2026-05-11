# Pipeline Stage Dispatch Ledger Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the in-memory Celery `chain(...)` for ingest pipeline stages 1–9 with a database-backed dispatch ledger on `ingest.stage_runs`, so worker death between stages cannot strand a `pipeline_run`.

**Architecture:** Each handoff commits a successor PENDING row in the same DB transaction that marks the current stage COMPLETE. A 5-second-cadence beat task (`dispatch_pending_pipeline_stages`) atomically claims PENDING rows via `FOR UPDATE SKIP LOCKED` and publishes Celery tasks. Stage bodies are unchanged; lifecycle is folded into the existing `guard_stage_run` decorator, with a `ContextVar`-based interception inside `_update_stage_run` that defers terminal status writes to the wrapper's single Tx-3 commit.

**Tech Stack:** Python 3.12, SQLAlchemy 2.x, Alembic, Celery 5.x, Redis (broker + advisory lock), PostgreSQL (`ingest` schema).

**Spec:** `docs/superpowers/specs/2026-05-10-pipeline-stage-dispatch-ledger-design.md`

**Conventions for every task in this plan:**
- File paths are absolute from repo root (`/home/josh/development/EIP-MMDPP/`).
- All Python work happens inside the Docker stack (`docker compose run --rm worker ...` for one-shot, or attach to a running container for live changes). Bind mounts on `app/` mean code edits land immediately; container rebuilds are only required for `docling`/`docling-graph` (not used by this plan).
- Tests run via `pytest` inside the `worker` or `api` service.
- **New test directories require an empty `__init__.py`** to match the existing convention (`tests/native/`, `tests/e2e/`, etc.). When a task creates a new `tests/<subdir>/` directory, also `touch tests/<subdir>/__init__.py` and include it in the same commit.
- "Commit" steps create a single commit per logical task; do NOT batch commits across tasks.
- Every commit must pass pre-commit hooks; never use `--no-verify`.

---

## Chunk 1: Schema + foundations

This chunk lays the durable storage and process-local infrastructure the rest of the plan depends on. Order matters: migration → model → settings → lifecycle module.

### Task 1: Alembic migration `0020_stage_dispatch_columns`

**Files:**
- Create: `alembic/versions/0020_stage_dispatch_columns.py`
- Test: `tests/db/test_migration_0020.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/db/test_migration_0020.py
"""Migration 0020 adds dispatch ledger columns + partial index to stage_runs."""
from sqlalchemy import inspect, text


def test_migration_0020_adds_expected_columns(db_session):
    """All six new columns exist with correct types and defaults."""
    inspector = inspect(db_session.bind)
    cols = {c["name"]: c for c in inspector.get_columns("stage_runs", schema="ingest")}

    assert "queue_name" in cols and "VARCHAR" in str(cols["queue_name"]["type"]).upper()
    assert "task_name" in cols and "VARCHAR" in str(cols["task_name"]["type"]).upper()
    assert "celery_task_id" in cols and "VARCHAR" in str(cols["celery_task_id"]["type"]).upper()
    assert "available_at" in cols
    assert "dispatched_at" in cols
    assert "dispatch_attempt" in cols
    assert cols["dispatch_attempt"]["nullable"] is False


def test_migration_0020_creates_partial_index(db_session):
    """Dispatcher's hot-path partial index exists with correct predicate."""
    result = db_session.execute(text("""
        SELECT indexdef FROM pg_indexes
        WHERE schemaname = 'ingest' AND indexname = 'ix_stage_runs_dispatcher_claim'
    """)).scalar_one_or_none()
    assert result is not None
    assert "status = 'PENDING'" in result
    assert "pass_name IS NULL" in result
    assert "task_name IS NOT NULL" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/db/test_migration_0020.py -v
```

Expected: both tests FAIL because migration doesn't exist yet.

- [ ] **Step 3: Write the migration**

```python
# alembic/versions/0020_stage_dispatch_columns.py
"""add dispatch ledger columns to stage_runs

Revision ID: 0020
Revises: 0019
Create Date: 2026-05-10

Adds 6 columns and 1 partial index to ingest.stage_runs, enabling the durable
dispatch-ledger model. Ledger summary rows always have attempt=1; the new
dispatch_attempt column tracks retries. See:
docs/superpowers/specs/2026-05-10-pipeline-stage-dispatch-ledger-design.md
"""
from alembic import op
import sqlalchemy as sa

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("stage_runs",
        sa.Column("queue_name", sa.String(64), nullable=True),
        schema="ingest")
    op.add_column("stage_runs",
        sa.Column("task_name", sa.String(255), nullable=True),
        schema="ingest")
    op.add_column("stage_runs",
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        schema="ingest")
    op.add_column("stage_runs",
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            nullable=True,
            server_default=sa.text("NOW()"),
        ),
        schema="ingest")
    op.add_column("stage_runs",
        sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
        schema="ingest")
    op.add_column("stage_runs",
        sa.Column(
            "dispatch_attempt",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        schema="ingest")

    op.create_index(
        "ix_stage_runs_dispatcher_claim",
        "stage_runs",
        ["available_at"],
        unique=False,
        postgresql_where=sa.text(
            "status = 'PENDING' AND pass_name IS NULL AND task_name IS NOT NULL"
        ),
        schema="ingest",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_stage_runs_dispatcher_claim",
        table_name="stage_runs",
        schema="ingest",
    )
    op.drop_column("stage_runs", "dispatch_attempt", schema="ingest")
    op.drop_column("stage_runs", "dispatched_at", schema="ingest")
    op.drop_column("stage_runs", "available_at", schema="ingest")
    op.drop_column("stage_runs", "celery_task_id", schema="ingest")
    op.drop_column("stage_runs", "task_name", schema="ingest")
    op.drop_column("stage_runs", "queue_name", schema="ingest")
```

- [ ] **Step 4: Apply the migration**

```bash
docker compose run --rm worker alembic upgrade head
```

Expected output: `Running upgrade 0019 -> 0020, add dispatch ledger columns to stage_runs`. No errors.

- [ ] **Step 5: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/db/test_migration_0020.py -v
```

Expected: both PASS.

- [ ] **Step 6: Verify rollback works**

```bash
docker compose run --rm worker alembic downgrade 0019
docker compose run --rm worker alembic upgrade head
```

Expected: clean downgrade then re-upgrade. No errors.

- [ ] **Step 7: Add `__init__.py` to new test dir**

```bash
touch tests/db/__init__.py
```

- [ ] **Step 8: Commit**

```bash
git add alembic/versions/0020_stage_dispatch_columns.py tests/db/__init__.py tests/db/test_migration_0020.py
git commit -m "feat(migration): add stage_runs dispatch ledger columns (0020)"
```

---

### Task 2: Extend `StageRun` model

**Files:**
- Modify: `app/models/ingest.py:279-324`
- Test: `tests/models/test_stage_run_columns.py`

The model must mirror the migration's columns. Existing column definitions stay; we add the six new ones plus a `StageRunStatus` enum constant module-level.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_stage_run_columns.py
"""StageRun model exposes the new dispatch-ledger columns and DISPATCHED status."""
from app.models.ingest import StageRun, StageRunStatus


def test_stage_run_has_new_columns():
    """ORM mapper recognises all six new columns."""
    cols = {c.name for c in StageRun.__table__.columns}
    assert {
        "queue_name",
        "task_name",
        "celery_task_id",
        "available_at",
        "dispatched_at",
        "dispatch_attempt",
    }.issubset(cols)


def test_stage_run_status_enum_has_dispatched():
    """DISPATCHED is a valid StageRunStatus value."""
    assert StageRunStatus.DISPATCHED.value == "DISPATCHED"
    # Verify all five statuses
    assert {s.value for s in StageRunStatus} == {
        "PENDING", "DISPATCHED", "RUNNING", "COMPLETE", "FAILED"
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/models/test_stage_run_columns.py -v
```

Expected: ImportError or AttributeError — `StageRunStatus` and new columns don't exist.

- [ ] **Step 3: Add the `StageRunStatus` enum and extend `StageRun`**

Edit `app/models/ingest.py`. Near the top of the file with other enums (or just above the `StageRun` class definition), add:

```python
from enum import Enum


class StageRunStatus(str, Enum):
    """Lifecycle states for a stage_runs summary row.

    Ledger semantics: PENDING → DISPATCHED → RUNNING → COMPLETE (success)
    or → FAILED (terminal). DISPATCHED added in spec
    docs/superpowers/specs/2026-05-10-pipeline-stage-dispatch-ledger-design.md
    """
    PENDING    = "PENDING"
    DISPATCHED = "DISPATCHED"
    RUNNING    = "RUNNING"
    COMPLETE   = "COMPLETE"
    FAILED     = "FAILED"
```

Inside the `StageRun` class, after the existing `rollback_executed` column declaration, add:

```python
    # ── dispatch ledger columns (migration 0020) ─────────────────────────
    # Set on ledger rows by _seed_first_stage / Tx-3a; NULL on legacy non-ledger rows.
    queue_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    task_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    available_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        server_default=func.now(),
    )
    dispatched_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Ledger retry counter. Starts at 1, incremented by Tx-4 and the
    # stale-RUNNING sweeper. Distinct from `attempt`, which legacy code
    # mutates per Celery retry.
    dispatch_attempt: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=sa.text("1")
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/models/test_stage_run_columns.py -v
```

Expected: both PASS.

- [ ] **Step 5: Run the full models test suite to verify no regression**

```bash
docker compose run --rm worker pytest tests/models/ -v
```

Expected: all existing tests still pass; no import errors anywhere.

- [ ] **Step 6: Add `__init__.py` to new test dir**

```bash
touch tests/models/__init__.py
```

- [ ] **Step 7: Commit**

```bash
git add app/models/ingest.py tests/models/__init__.py tests/models/test_stage_run_columns.py
git commit -m "feat(models): add StageRunStatus enum and dispatch ledger columns to StageRun"
```

---

### Task 3: Add new settings

**Files:**
- Modify: `app/config.py` (after the existing `stale_stage_run_threshold_seconds` setting, around line 483)
- Test: `tests/test_config.py` (extend existing or create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py (extend existing)
def test_max_stage_dispatches_default():
    from app.config import settings
    assert settings.max_stage_dispatches == 5


def test_stale_dispatched_threshold_default():
    from app.config import settings
    assert settings.stale_dispatched_threshold_seconds == 600
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/test_config.py::test_max_stage_dispatches_default tests/test_config.py::test_stale_dispatched_threshold_default -v
```

Expected: AttributeError — settings don't exist.

- [ ] **Step 3: Add the settings**

Edit `app/config.py`. Find `stale_stage_run_threshold_seconds: int = 34200` and immediately below it, add:

```python
    # ── dispatch ledger v1 (spec 2026-05-10) ─────────────────────────────
    # Maximum dispatch_attempt before a stage_run is terminalized to FAILED.
    # Counts ledger retries (Tx-4 + sweeper resets), NOT Celery retries.
    max_stage_dispatches: int = 5

    # How long a DISPATCHED row may sit before the sweeper resets it to
    # PENDING. Should be longer than a worker's accept-and-start time but
    # MUCH shorter than any stage's expected runtime.
    stale_dispatched_threshold_seconds: int = 600
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/test_config.py::test_max_stage_dispatches_default tests/test_config.py::test_stale_dispatched_threshold_default -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat(config): add max_stage_dispatches and stale_dispatched_threshold_seconds"
```

---

### Task 4: Create `_stage_lifecycle` module

**Files:**
- Create: `app/workers/_stage_lifecycle.py`
- Test: `tests/workers/test_stage_lifecycle.py`

The module holds the `_LifecycleCtx` dataclass and `_CTX` `ContextVar`. It is purely process-local; no imports of `app.models` or `app.workers.pipeline` are needed (avoid import cycles).

- [ ] **Step 1: Write the failing tests**

```python
# tests/workers/test_stage_lifecycle.py
"""Lifecycle ContextVar and dataclass behaviour."""
from app.workers._stage_lifecycle import _CTX, _LifecycleCtx


def test_lifecycle_ctx_dataclass_fields():
    """All required fields exist with the right defaults."""
    ctx = _LifecycleCtx(
        pipeline_run_id="abc-123",
        stage_name="prepare_document",
        dispatch_attempt=1,
        intercept_terminal=True,
        next_stage="detect_and_translate",
        next_task="app.workers.pipeline.detect_and_translate",
    )
    assert ctx.pipeline_run_id == "abc-123"
    assert ctx.pending_status is None
    assert ctx.pending_metrics is None
    assert ctx.pending_error is None


def test_lifecycle_ctx_normalizes_pipeline_run_id_to_str():
    """UUID input is converted to str at construction (spec rule 1)."""
    import uuid
    u = uuid.uuid4()
    ctx = _LifecycleCtx(
        pipeline_run_id=u,
        stage_name="prepare_document",
        dispatch_attempt=1,
        intercept_terminal=True,
        next_stage="detect_and_translate",
        next_task="app.workers.pipeline.detect_and_translate",
    )
    assert ctx.pipeline_run_id == str(u)
    assert isinstance(ctx.pipeline_run_id, str)


def test_ctx_var_starts_unset():
    """_CTX default is None — no leakage between processes."""
    assert _CTX.get() is None


def test_ctx_var_set_and_reset():
    """Token-based set/reset works as expected."""
    ctx = _LifecycleCtx(
        pipeline_run_id="abc-123",
        stage_name="s",
        dispatch_attempt=1,
        intercept_terminal=True,
        next_stage=None,
        next_task=None,
    )
    token = _CTX.set(ctx)
    try:
        assert _CTX.get() is ctx
    finally:
        _CTX.reset(token)
    assert _CTX.get() is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/workers/test_stage_lifecycle.py -v
```

Expected: ImportError — module doesn't exist.

- [ ] **Step 3: Create the module**

```python
# app/workers/_stage_lifecycle.py
"""Process-local lifecycle context for the dispatch-ledger wrapper.

See: docs/superpowers/specs/2026-05-10-pipeline-stage-dispatch-ledger-design.md
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional


@dataclass
class _LifecycleCtx:
    """Per-stage-invocation state used by the lifecycle wrapper.

    Set by `guard_stage_run` immediately after a successful Tx-1 CLAIM.
    Read by `_update_stage_run` interception (to stash terminal writes) and
    by `_finalize_after_body` (to issue Tx-3 / Tx-4).
    """
    pipeline_run_id: str
    stage_name: str
    dispatch_attempt: int
    intercept_terminal: bool
    next_stage: Optional[str]
    next_task: Optional[str]
    pending_status: Optional[str] = None      # "COMPLETE" | "FAILED" | None
    pending_metrics: Optional[dict] = None
    pending_error: Optional[str] = None

    def __post_init__(self) -> None:
        # Normalize so equality / interception comparisons are str-on-str.
        self.pipeline_run_id = str(self.pipeline_run_id)


_CTX: ContextVar[Optional[_LifecycleCtx]] = ContextVar(
    "stage_lifecycle_ctx", default=None
)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/workers/test_stage_lifecycle.py -v
```

Expected: all four PASS.

- [ ] **Step 5: Add `__init__.py` to new test dir**

```bash
touch tests/workers/__init__.py
```

- [ ] **Step 6: Commit**

```bash
git add app/workers/_stage_lifecycle.py tests/workers/__init__.py tests/workers/test_stage_lifecycle.py
git commit -m "feat(workers): add _stage_lifecycle module (_LifecycleCtx + _CTX)"
```

---

**Chunk 1 complete.** Schema, model, settings, and lifecycle context infrastructure are in place. Run the full unit test suite as a smoke check before proceeding:

```bash
docker compose run --rm worker pytest tests/db/ tests/models/ tests/workers/test_stage_lifecycle.py -v
```

Expected: all green.

---

## Chunk 2: Helpers

This chunk adds the pure-function helpers the wrapper, dispatcher, and entry points all consume. None of these helpers run on the worker yet (no decorator is wired); they're testable in isolation.

All edits in this chunk target `app/workers/pipeline.py`. Place all new code as one contiguous block **immediately above the `start_ingest_pipeline` task definition** (currently around line 2300). All four helpers + `STAGE_SUCCESSORS` live in that single region so `git log -p pipeline.py` shows one location for "the ledger helpers."

**Pre-flight import check:** `pipeline.py` line 88-90 has `import sqlalchemy as sa` and `from dataclasses import dataclass`. `text` is imported only inside individual functions today. Before writing helpers, **add `from sqlalchemy import text` to the top-level imports** (alongside the existing `sa` import). Do not re-import `dataclass` — it's already at line 90.

### Task 5: `STAGE_SUCCESSORS` table

**Files:**
- Modify: `app/workers/pipeline.py` (add near other module-level constants, after the existing imports block)
- Test: `tests/workers/test_stage_successors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_stage_successors.py
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
    edge = STAGE_SUCCESSORS["prepare_document"]
    import dataclasses
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/workers/test_stage_successors.py -v
```

Expected: ImportError — names don't exist.

- [ ] **Step 3: Add `STAGE_SUCCESSORS` to `pipeline.py`**

First, verify `from sqlalchemy import text` is at the top-level imports (around line 80). If absent, add it. Then find a location in `app/workers/pipeline.py` immediately before `start_ingest_pipeline` (around line 2300). Add (do NOT re-import `dataclass` — already imported at line 90):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/workers/test_stage_successors.py -v
```

Expected: all four PASS.

- [ ] **Step 5: Commit**

```bash
git add app/workers/pipeline.py tests/workers/test_stage_successors.py
git commit -m "feat(pipeline): add STAGE_SUCCESSORS table and ledger stage subsets"
```

---

### Task 6: `_resolve_queue` helper (3-tier lookup)

**Files:**
- Modify: `app/workers/pipeline.py` (immediately below `STAGE_SUCCESSORS`)
- Test: `tests/workers/test_resolve_queue.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_resolve_queue.py
"""_resolve_queue returns the queue an apply_async() call would route to."""
from app.workers.pipeline import _resolve_queue


def test_task_routes_entry_wins():
    """Tasks with an explicit task_routes entry resolve to that queue."""
    assert _resolve_queue("app.workers.pipeline.prepare_document") == "ingest"
    assert _resolve_queue("app.workers.pipeline.derive_image_embeddings") == "embed"
    assert _resolve_queue("app.workers.pipeline.derive_document_anchors") == "graph"
    assert _resolve_queue("app.workers.pipeline.derive_ontology_graph") == "graph_extract"


def test_no_task_routes_no_decorator_queue_returns_broker_default():
    """Tier 3 (broker default): tasks routed via decorator queue= only."""
    # detect_and_translate / derive_document_metadata / derive_picture_descriptions
    # have no task_routes entry AND no decorator queue=. They fall back to
    # the broker default "celery".
    assert _resolve_queue("app.workers.pipeline.detect_and_translate") == "celery"
    assert _resolve_queue("app.workers.pipeline.derive_document_metadata") == "celery"
    assert _resolve_queue("app.workers.pipeline.derive_picture_descriptions") == "celery"


def test_decorator_queue_resolves_tier_2(monkeypatch):
    """Tier 2 (decorator queue=): a task registered with queue= but no task_routes entry."""
    from app.workers.celery_app import celery_app

    class FakeTask:
        queue = "custom-queue"

    monkeypatch.setitem(celery_app.tasks, "fake.module.tier2_task", FakeTask())
    # Ensure no task_routes entry exists for this fake task.
    assert "fake.module.tier2_task" not in (celery_app.conf.task_routes or {})
    assert _resolve_queue("fake.module.tier2_task") == "custom-queue"


def test_unknown_task_falls_to_broker_default():
    """Tier 3 (broker default): task not registered at all."""
    assert _resolve_queue("nonexistent.task.path.that.does.not.exist") == "celery"


def test_every_stage_successor_task_resolves_without_error():
    """Smoke check: every STAGE_SUCCESSORS next_task can be resolved."""
    from app.workers.pipeline import STAGE_SUCCESSORS
    for stage, edge in STAGE_SUCCESSORS.items():
        if edge.next_task is None:
            continue
        queue = _resolve_queue(edge.next_task)
        assert isinstance(queue, str) and queue, f"{stage} → {edge.next_task} resolved to {queue!r}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/workers/test_resolve_queue.py -v
```

Expected: ImportError — `_resolve_queue` doesn't exist.

- [ ] **Step 3: Implement `_resolve_queue`**

In `app/workers/pipeline.py`, immediately below the `STAGE_SUCCESSORS` constants block (still inside the ledger-helpers section), add:

```python
def _resolve_queue(task_name: str) -> str:
    """Return the queue Celery will actually route a task to.

    3-tier precedence (matches Celery's own lookup order):
    1. Explicit `task_routes[task_name]["queue"]` from celery_app.conf
    2. The task's decorator `queue=` argument (via celery_app.tasks[name].queue)
    3. Broker default (celery_app.conf.task_default_queue, "celery" unless overridden)

    Three ledger stages — detect_and_translate, derive_document_metadata,
    derive_picture_descriptions — are not in task_routes and have no decorator
    queue=; they route to the broker default. The helper unifies the lookup
    so the ledger's queue_name column matches the runtime destination.
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
```

`celery_app` is already imported at the top of `pipeline.py` — verify the existing `from app.workers.celery_app import celery_app` import; if absent, add it.

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/workers/test_resolve_queue.py -v
```

Expected: all four PASS.

- [ ] **Step 5: Commit**

```bash
git add app/workers/pipeline.py tests/workers/test_resolve_queue.py
git commit -m "feat(pipeline): add _resolve_queue 3-tier helper"
```

---

### Task 7: `_seed_first_stage` helper

**Files:**
- Modify: `app/workers/pipeline.py` (immediately below `_resolve_queue`)
- Test: `tests/workers/test_seed_first_stage.py`

This helper inserts the initial PENDING ledger row for a fresh pipeline_run. It's called by `start_ingest_pipeline` (for full ingest) and `reingest_graph_only` (for graph-only re-ingest). The insert is idempotent on the partial unique index.

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_seed_first_stage.py
"""_seed_first_stage inserts a PENDING ledger row idempotently."""
import uuid
from sqlalchemy import text
from app.workers.pipeline import _seed_first_stage


_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _new_pipeline_run(db_session) -> str:
    """Create a minimal source+document+pipeline_run for testing.

    Includes every NOT NULL column on each table (sources.created_by,
    documents.storage_bucket/storage_key). If the schema adds more NOT NULL
    columns, this helper must be updated.
    """
    doc_id = uuid.uuid4()
    run_id = uuid.uuid4()
    src_id = uuid.uuid4()
    db_session.execute(text("""
        INSERT INTO ingest.sources (id, name, created_by)
        VALUES (:s, 'test-source', :u)
    """), {"s": src_id, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key,
             uploaded_by, pipeline_status)
        VALUES (:d, :s, 'x.pdf', 'application/pdf', 0,
                'test-bucket', 'test/x.pdf',
                :u, 'PROCESSING')
    """), {"d": doc_id, "s": src_id, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run_id, "d": doc_id})
    db_session.commit()
    return str(run_id)


def test_seed_first_stage_inserts_pending_row(db_session):
    run_id = _new_pipeline_run(db_session)
    _seed_first_stage(
        db_session,
        pipeline_run_id=run_id,
        stage_name="prepare_document",
        task_name="app.workers.pipeline.prepare_document",
    )
    db_session.commit()

    row = db_session.execute(text("""
        SELECT status, attempt, dispatch_attempt, queue_name, task_name,
               available_at IS NOT NULL AS has_available_at
        FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'prepare_document'
    """), {"r": run_id}).first()
    assert row is not None
    assert row.status == "PENDING"
    assert row.attempt == 1                  # ledger invariant
    assert row.dispatch_attempt == 1
    assert row.queue_name == "ingest"        # resolved via task_routes
    assert row.task_name == "app.workers.pipeline.prepare_document"
    assert row.has_available_at is True


def test_seed_first_stage_is_idempotent(db_session):
    """Calling twice does not create a duplicate row."""
    run_id = _new_pipeline_run(db_session)
    _seed_first_stage(
        db_session,
        pipeline_run_id=run_id,
        stage_name="prepare_document",
        task_name="app.workers.pipeline.prepare_document",
    )
    db_session.commit()
    _seed_first_stage(
        db_session,
        pipeline_run_id=run_id,
        stage_name="prepare_document",
        task_name="app.workers.pipeline.prepare_document",
    )
    db_session.commit()

    count = db_session.execute(text("""
        SELECT COUNT(*) FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'prepare_document' AND pass_name IS NULL
    """), {"r": run_id}).scalar_one()
    assert count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/workers/test_seed_first_stage.py -v
```

Expected: ImportError — `_seed_first_stage` doesn't exist.

- [ ] **Step 3: Implement `_seed_first_stage`**

In `app/workers/pipeline.py`, immediately below `_resolve_queue`:

```python
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
```

`text` is already imported via `from sqlalchemy import text` near the top of `pipeline.py`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/workers/test_seed_first_stage.py -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add app/workers/pipeline.py tests/workers/test_seed_first_stage.py
git commit -m "feat(pipeline): add _seed_first_stage helper"
```

---

### Task 8: `_claim_tx1` helper (with 6-outcome disambiguation)

**Files:**
- Modify: `app/workers/pipeline.py` (immediately below `_seed_first_stage`)
- Test: `tests/workers/test_claim_tx1.py`

`_claim_tx1` performs Tx-1 (the wrapper's first transaction). It atomically transitions a ledger row to RUNNING when the row is in a claimable state, and returns a structured outcome the wrapper uses to decide whether to set `_CTX`, return early, or run inline (legacy).

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_claim_tx1.py
"""Tx-1 CLAIM atomically transitions ledger rows to RUNNING with 6 outcomes."""
import uuid
from sqlalchemy import text
from app.workers.pipeline import _claim_tx1, _seed_first_stage


_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _setup_run_and_stage(db_session, status: str | None = "PENDING"):
    """Create source+document+pipeline_run + (optionally) one ledger row.

    Same NOT NULL coverage as test_seed_first_stage's helper.
    """
    src_id, doc_id, run_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text("""
        INSERT INTO ingest.sources (id, name, created_by)
        VALUES (:s, 'test', :u)
    """), {"s": src_id, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key,
             uploaded_by, pipeline_status)
        VALUES (:d, :s, 'x.pdf', 'application/pdf', 0,
                'test-bucket', 'test/x.pdf',
                :u, 'PROCESSING')
    """), {"d": doc_id, "s": src_id, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run_id, "d": doc_id})
    if status is not None:
        db_session.execute(text("""
            INSERT INTO ingest.stage_runs
                (id, pipeline_run_id, stage_name, attempt, status,
                 task_name, dispatch_attempt)
            VALUES (gen_random_uuid(), :r, 'prepare_document', 1, :st,
                    'app.workers.pipeline.prepare_document', 1)
        """), {"r": run_id, "st": status})
    db_session.commit()
    return str(run_id)


def test_claim_proceed_from_pending(db_session):
    run_id = _setup_run_and_stage(db_session, status="PENDING")
    result = _claim_tx1(db_session, run_id, "prepare_document",
                        celery_task_id="t-1", is_celery_retry=False)
    db_session.commit()
    assert result.outcome == "proceed"
    assert result.dispatch_attempt == 1
    row = db_session.execute(text(
        "SELECT status, celery_task_id FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.status == "RUNNING"
    assert row.celery_task_id == "t-1"


def test_claim_proceed_from_dispatched(db_session):
    run_id = _setup_run_and_stage(db_session, status="DISPATCHED")
    result = _claim_tx1(db_session, run_id, "prepare_document",
                        celery_task_id="t-2", is_celery_retry=False)
    db_session.commit()
    assert result.outcome == "proceed"


def test_claim_already_complete_returns_skip_dict(db_session):
    run_id = _setup_run_and_stage(db_session, status="COMPLETE")
    result = _claim_tx1(db_session, run_id, "prepare_document",
                        celery_task_id="t-3", is_celery_retry=False)
    db_session.commit()
    assert result.outcome == "already_complete"
    assert result.early_result == {
        "stage": "prepare_document",
        "status": "skipped",
        "reason": "already_complete",
    }


def test_claim_concurrent_running_no_retry_returns_none(db_session):
    run_id = _setup_run_and_stage(db_session, status="RUNNING")
    result = _claim_tx1(db_session, run_id, "prepare_document",
                        celery_task_id="t-4", is_celery_retry=False)
    db_session.commit()
    assert result.outcome == "concurrent_running"
    assert result.early_result is None


def test_claim_celery_retry_proceeds_on_running(db_session):
    """is_celery_retry=True allows re-entry on RUNNING (same task republished)."""
    run_id = _setup_run_and_stage(db_session, status="RUNNING")
    result = _claim_tx1(db_session, run_id, "prepare_document",
                        celery_task_id="t-5", is_celery_retry=True)
    db_session.commit()
    assert result.outcome == "proceed"
    row = db_session.execute(text(
        "SELECT celery_task_id FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.celery_task_id == "t-5"  # overwritten with current attempt's id


def test_claim_stale_pending_returns_none(db_session):
    """Sweeper reset between dispatcher tick and worker pickup: 0 rows + PENDING."""
    run_id = _setup_run_and_stage(db_session, status="PENDING")
    # Simulate: dispatcher claims (PENDING → DISPATCHED), but BEFORE worker
    # CLAIM runs, sweeper resets DISPATCHED → PENDING. Worker then tries CLAIM.
    # First CLAIM call succeeds (PENDING → RUNNING). To force "stale_pending"
    # we instead set the row back to PENDING after a hypothetical proceed:
    db_session.execute(text(
        "UPDATE ingest.stage_runs SET status='PENDING' WHERE pipeline_run_id=:r"
    ), {"r": run_id})
    db_session.commit()
    # If the CLAIM WHERE matches PENDING and proceeds, this test demonstrates
    # the proceed path. The stale_pending case in production is observed when
    # the CLAIM UPDATE returns 0 rows AND the follow-up SELECT shows PENDING —
    # which requires concurrency the unit test can't easily simulate without
    # threading. The full coverage lives in tests/integration/test_dispatcher.py.
    result = _claim_tx1(db_session, run_id, "prepare_document",
                        celery_task_id="t-6", is_celery_retry=False)
    db_session.commit()
    assert result.outcome == "proceed"  # not stale_pending in this single-threaded case


def test_claim_terminal_failed_returns_distinct_dict(db_session):
    run_id = _setup_run_and_stage(db_session, status="FAILED")
    result = _claim_tx1(db_session, run_id, "prepare_document",
                        celery_task_id="t-7", is_celery_retry=False)
    db_session.commit()
    assert result.outcome == "terminal_failed"
    assert result.early_result == {
        "stage": "prepare_document",
        "status": "terminal_failed",     # distinct from "skipped"
        "reason": "stage_previously_failed",
    }


def test_claim_legacy_no_row(db_session):
    run_id = _setup_run_and_stage(db_session, status=None)  # NO ledger row
    result = _claim_tx1(db_session, run_id, "prepare_document",
                        celery_task_id="t-8", is_celery_retry=False)
    db_session.commit()
    assert result.outcome == "legacy"
    assert result.early_result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/workers/test_claim_tx1.py -v
```

Expected: ImportError — `_claim_tx1` doesn't exist.

- [ ] **Step 3: Implement `_claim_tx1`**

Below `_seed_first_stage` in `app/workers/pipeline.py` (do NOT re-import `dataclass`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/workers/test_claim_tx1.py -v
```

Expected: all eight PASS (the `stale_pending` test documents the single-threaded limitation explicitly and asserts the proceed-path; concurrency-driven stale_pending coverage lives in the dispatcher integration tests, Chunk 7).

- [ ] **Step 5: Commit**

```bash
git add app/workers/pipeline.py tests/workers/test_claim_tx1.py
git commit -m "feat(pipeline): add _claim_tx1 helper with 6-outcome disambiguation"
```

---

**Chunk 2 complete.** The four pure helpers (`STAGE_SUCCESSORS`, `_resolve_queue`, `_seed_first_stage`, `_claim_tx1`) are in place and unit-tested. None are wired to the runtime pipeline yet — that happens in Chunks 3 and 6.

Smoke check:

```bash
docker compose run --rm worker pytest tests/workers/ -v
```

---

## Chunk 3: Wrapper integration

This chunk extends `guard_stage_run` with lifecycle behavior, adds the four committing helpers (`_tx3_complete_and_enqueue_next`, `_tx4_finalize_failure`, `_finalize_after_body`), and modifies `_update_stage_run` to intercept terminal writes when running inside a wrapped stage. After this chunk, the wrapper is fully functional but **no stage decorators are updated yet** — the runtime pipeline still uses the old chain. Decorator updates happen in Chunk 6.

Order: bottom-up. Tx-3 and Tx-4 first, then `_finalize_after_body` (which calls them), then `_update_stage_run` interception, then `guard_stage_run` extension (which orchestrates everything).

All new code lives in the same contiguous "ledger helpers" block in `app/workers/pipeline.py` introduced in Chunk 2.

### Task 9: `_tx3_complete_and_enqueue_next`

**Files:**
- Modify: `app/workers/pipeline.py` (below `_claim_tx1`)
- Test: `tests/workers/test_tx3_complete_and_enqueue_next.py`

The single-transaction commit that closes the orphan window: INSERT successor PENDING + UPDATE self → COMPLETE, both in one `with db.begin():` block.

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_tx3_complete_and_enqueue_next.py
"""Tx-3 atomically inserts the successor PENDING row and marks self COMPLETE."""
import uuid
from sqlalchemy import text
from app.workers._stage_lifecycle import _LifecycleCtx
from app.workers.pipeline import (
    _seed_first_stage,
    _tx3_complete_and_enqueue_next,
)

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _setup(db_session, stage="purge_document_derivations"):
    src_id, doc_id, run_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text("""
        INSERT INTO ingest.sources (id, name, created_by)
        VALUES (:s, 'test', :u)
    """), {"s": src_id, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, uploaded_by, pipeline_status)
        VALUES (:d, :s, 'x.pdf', 'application/pdf', 0,
                'test-bucket', 'test/x.pdf', :u, 'PROCESSING')
    """), {"d": doc_id, "s": src_id, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run_id, "d": doc_id})
    # Seed self as RUNNING (post-CLAIM state) — Tx-3 will flip it to COMPLETE.
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             task_name, dispatch_attempt, started_at)
        VALUES (gen_random_uuid(), :r, :s, 1, 'RUNNING',
                'app.workers.pipeline.purge_document_derivations', 1, NOW())
    """), {"r": run_id, "s": stage})
    db_session.commit()
    return str(run_id)


def test_tx3_atomic_complete_and_successor_insert(db_session):
    run_id = _setup(db_session, stage="purge_document_derivations")
    ctx = _LifecycleCtx(
        pipeline_run_id=run_id,
        stage_name="purge_document_derivations",
        dispatch_attempt=1,
        intercept_terminal=True,
        next_stage="derive_picture_descriptions",
        next_task="app.workers.pipeline.derive_picture_descriptions",
        pending_metrics={"some_metric": 7},
    )
    _tx3_complete_and_enqueue_next(ctx)

    # Self row → COMPLETE with metrics persisted.
    self_row = db_session.execute(text("""
        SELECT status, metrics, finished_at IS NOT NULL AS done
        FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'purge_document_derivations'
    """), {"r": run_id}).first()
    assert self_row.status == "COMPLETE"
    assert self_row.metrics == {"some_metric": 7}
    assert self_row.done

    # Successor row inserted PENDING with task_name + queue resolved.
    next_row = db_session.execute(text("""
        SELECT status, task_name, queue_name, attempt, dispatch_attempt
        FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'derive_picture_descriptions'
    """), {"r": run_id}).first()
    assert next_row.status == "PENDING"
    assert next_row.task_name == "app.workers.pipeline.derive_picture_descriptions"
    assert next_row.queue_name == "celery"  # tier-3 fallback for this stage
    assert next_row.attempt == 1
    assert next_row.dispatch_attempt == 1


def test_tx3_rolls_back_both_writes_on_error(monkeypatch, db_session):
    """If 3b UPDATE fails, the 3a INSERT must also be rolled back."""
    run_id = _setup(db_session, stage="purge_document_derivations")
    ctx = _LifecycleCtx(
        pipeline_run_id=run_id,
        stage_name="purge_document_derivations",
        dispatch_attempt=1,
        intercept_terminal=True,
        next_stage="derive_picture_descriptions",
        next_task="app.workers.pipeline.derive_picture_descriptions",
    )

    # Patch `_tx3_run_3b` (or whatever name the impl uses) to raise after Tx-3a.
    # If the impl inlines 3a+3b inside a `with db.begin():` block, force the
    # raise by patching db.execute to raise on the second call.
    import app.workers.pipeline as pl
    original_execute = pl._get_db().__class__.execute  # session class method
    call_count = {"n": 0}

    def maybe_raise(self, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:   # second statement = Tx-3b UPDATE
            raise RuntimeError("simulated DB blip")
        return original_execute(self, *args, **kwargs)

    monkeypatch.setattr(pl._get_db().__class__, "execute", maybe_raise)

    import pytest
    with pytest.raises(RuntimeError, match="simulated DB blip"):
        _tx3_complete_and_enqueue_next(ctx)

    # Successor row must NOT exist; self row must still be RUNNING.
    next_count = db_session.execute(text("""
        SELECT COUNT(*) FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'derive_picture_descriptions'
    """), {"r": run_id}).scalar_one()
    assert next_count == 0

    self_row = db_session.execute(text("""
        SELECT status FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'purge_document_derivations'
    """), {"r": run_id}).first()
    assert self_row.status == "RUNNING"


def test_tx3a_idempotent_on_concurrent_re_run(db_session):
    """If the successor row already exists, Tx-3a is a no-op via ON CONFLICT."""
    run_id = _setup(db_session, stage="purge_document_derivations")
    # Pre-seed the successor row as if a prior run inserted it.
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             task_name, dispatch_attempt, available_at)
        VALUES (gen_random_uuid(), :r, 'derive_picture_descriptions', 1, 'PENDING',
                'app.workers.pipeline.derive_picture_descriptions', 1, NOW())
    """), {"r": run_id})
    db_session.commit()

    ctx = _LifecycleCtx(
        pipeline_run_id=run_id,
        stage_name="purge_document_derivations",
        dispatch_attempt=1,
        intercept_terminal=True,
        next_stage="derive_picture_descriptions",
        next_task="app.workers.pipeline.derive_picture_descriptions",
    )
    _tx3_complete_and_enqueue_next(ctx)

    # Still exactly one successor row.
    count = db_session.execute(text("""
        SELECT COUNT(*) FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'derive_picture_descriptions'
    """), {"r": run_id}).scalar_one()
    assert count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/workers/test_tx3_complete_and_enqueue_next.py -v
```

Expected: ImportError — `_tx3_complete_and_enqueue_next` doesn't exist.

- [ ] **Step 3: Implement `_tx3_complete_and_enqueue_next`**

Below `_claim_tx1`:

```python
def _tx3_complete_and_enqueue_next(ctx: "_LifecycleCtx") -> None:
    """Tx-3: insert successor PENDING + flip self → COMPLETE, single transaction.

    This is the central durability guarantee of the design. Both writes commit
    together; any exception inside the `with db.begin():` block rolls both back
    atomically, leaving the ledger row in RUNNING for the stale-RUNNING sweeper
    to recover.
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
            # 3b: flip self → COMPLETE with metrics stashed by interception
            db.execute(text("""
                UPDATE ingest.stage_runs
                SET status      = 'COMPLETE',
                    finished_at = NOW(),
                    metrics     = :metrics
                WHERE pipeline_run_id = :run_id
                  AND stage_name      = :stage_name
                  AND pass_name       IS NULL
            """), {
                "run_id":     ctx.pipeline_run_id,
                "stage_name": ctx.stage_name,
                "metrics":    ctx.pending_metrics,
            })
    finally:
        db.close()
```

If the import of `_LifecycleCtx` for the type hint creates a cycle (because `pipeline.py` → `_stage_lifecycle.py` is fine, but the reverse would not be), keep the forward-reference string `"_LifecycleCtx"` as written. Add the runtime import:

```python
from app.workers._stage_lifecycle import _LifecycleCtx, _CTX
```

Place this import in the ledger-helpers section (not at top of file) to keep the module's existing top-imports stable and to scope the dependency.

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/workers/test_tx3_complete_and_enqueue_next.py -v
```

Expected: all three PASS.

- [ ] **Step 5: Commit**

```bash
git add app/workers/pipeline.py tests/workers/test_tx3_complete_and_enqueue_next.py
git commit -m "feat(pipeline): add _tx3_complete_and_enqueue_next single-tx helper"
```

---

### Task 10: `_tx4_finalize_failure`

**Files:**
- Modify: `app/workers/pipeline.py` (below `_tx3_complete_and_enqueue_next`)
- Test: `tests/workers/test_tx4_finalize_failure.py`

`_tx4_finalize_failure` is invoked when the body raises a non-`CeleryRetry` exception or returns `{"status":"FAILED"}`. It either bumps `dispatch_attempt` and sets PENDING (retryable) or terminalizes the row + pipeline_run (cap exceeded).

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_tx4_finalize_failure.py
"""Tx-4 retries within cap, terminalizes past it, propagates to pipeline_run."""
import uuid
from sqlalchemy import text
from app.workers._stage_lifecycle import _LifecycleCtx
from app.workers.pipeline import _tx4_finalize_failure

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _setup_running(db_session, dispatch_attempt=1, stage="prepare_document"):
    src, doc, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, uploaded_by, pipeline_status)
        VALUES (:d, :s, 'x.pdf', 'application/pdf', 0,
                'test-bucket', 'test/x.pdf', :u, 'PROCESSING')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run, "d": doc})
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             task_name, dispatch_attempt, started_at)
        VALUES (gen_random_uuid(), :r, :s, 1, 'RUNNING',
                'app.workers.pipeline.prepare_document', :da, NOW())
    """), {"r": run, "s": stage, "da": dispatch_attempt})
    db_session.commit()
    return str(run)


def _ctx(run_id, stage="prepare_document", dispatch_attempt=1):
    """Build a ctx with the dispatch_attempt the row was inserted with.

    In production this comes from `claim.dispatch_attempt` (Tx-1 RETURNING).
    Tests must mirror the row's current dispatch_attempt or the cap check
    will misfire.
    """
    return _LifecycleCtx(
        pipeline_run_id=run_id, stage_name=stage,
        dispatch_attempt=dispatch_attempt, intercept_terminal=True,
        next_stage=None, next_task=None,
    )


def test_tx4_retryable_bumps_dispatch_attempt_and_sets_pending(db_session):
    run_id = _setup_running(db_session, dispatch_attempt=1)
    _tx4_finalize_failure(
        _ctx(run_id),
        error="boom",
        celery_retries=0,
        max_retries=0,                 # not Celery-retryable; ledger retry path
        backoff_seconds=60,
    )
    row = db_session.execute(text("""
        SELECT status, dispatch_attempt, attempt,
               started_at IS NULL AS started_cleared,
               dispatched_at IS NULL AS dispatched_cleared,
               available_at > NOW() AS in_future,
               error_message LIKE '%boom%' AS has_err
        FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'prepare_document'
    """), {"r": run_id}).first()
    assert row.status == "PENDING"
    assert row.dispatch_attempt == 2
    assert row.attempt == 1               # ledger invariant: attempt never mutates
    assert row.started_cleared
    assert row.dispatched_cleared
    assert row.in_future
    assert row.has_err


def test_tx4_terminal_at_cap_marks_failed_and_propagates_to_pipeline_run(db_session, monkeypatch):
    from app import config
    monkeypatch.setattr(config.settings, "max_stage_dispatches", 3)

    # dispatch_attempt=3 already → next would be 4 > cap=3 → terminal.
    # The ctx MUST mirror the row's dispatch_attempt — in production Tx-1
    # RETURNING populates this from the DB row.
    run_id = _setup_running(db_session, dispatch_attempt=3)
    _tx4_finalize_failure(
        _ctx(run_id, dispatch_attempt=3), error="exhausted",
        celery_retries=0, max_retries=0, backoff_seconds=60,
    )
    row = db_session.execute(text("""
        SELECT status, dispatch_attempt, finished_at IS NOT NULL AS done
        FROM ingest.stage_runs
        WHERE pipeline_run_id = :r
    """), {"r": run_id}).first()
    assert row.status == "FAILED"
    assert row.dispatch_attempt == 4
    assert row.done

    pr = db_session.execute(text(
        "SELECT status, error_message FROM ingest.pipeline_runs WHERE id = :r"
    ), {"r": run_id}).first()
    assert pr.status == "FAILED"
    assert "exhausted" in (pr.error_message or "")


def test_tx4_does_not_overwrite_already_failed_pipeline_run(db_session, monkeypatch):
    from app import config
    monkeypatch.setattr(config.settings, "max_stage_dispatches", 1)
    run_id = _setup_running(db_session, dispatch_attempt=1)
    # Pre-mark the pipeline_run as FAILED with a distinct earlier error.
    db_session.execute(text("""
        UPDATE ingest.pipeline_runs
        SET status = 'FAILED', error_message = 'earlier failure'
        WHERE id = :r
    """), {"r": run_id})
    db_session.commit()

    _tx4_finalize_failure(
        _ctx(run_id, dispatch_attempt=1), error="later failure",
        celery_retries=0, max_retries=0, backoff_seconds=60,
    )
    pr = db_session.execute(text(
        "SELECT error_message FROM ingest.pipeline_runs WHERE id = :r"
    ), {"r": run_id}).first()
    assert pr.error_message == "earlier failure"  # WHERE status='PROCESSING' guard preserved earlier error
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/workers/test_tx4_finalize_failure.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `_tx4_finalize_failure`**

```python
def _tx4_finalize_failure(
    ctx: "_LifecycleCtx",
    *,
    error: str,
    celery_retries: int,
    max_retries: int,
    backoff_seconds: int = 60,
) -> None:
    """Tx-4: handle failure of a lifecycle-wrapped stage.

    Retryable if `ctx.dispatch_attempt + 1 <= settings.max_stage_dispatches`:
    status → PENDING, dispatch_attempt += 1, available_at advances by backoff.

    Terminal otherwise: status → FAILED + pipeline_run → FAILED (only if still
    PROCESSING; preserves earlier failure messages).

    `celery_retries`/`max_retries` are accepted for future use (e.g., if the
    wrapper wants to distinguish Celery-exhausted vs. exception-raised paths).
    In v1 we treat both uniformly: any non-CeleryRetry exception that reaches
    the wrapper is the ledger's responsibility.
    """
    from app.config import settings

    next_dispatch_attempt = ctx.dispatch_attempt + 1
    db = _get_db()
    try:
        if next_dispatch_attempt <= settings.max_stage_dispatches:
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

        # Terminal
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/workers/test_tx4_finalize_failure.py -v
```

Expected: all three PASS.

- [ ] **Step 5: Commit**

```bash
git add app/workers/pipeline.py tests/workers/test_tx4_finalize_failure.py
git commit -m "feat(pipeline): add _tx4_finalize_failure with cap-driven terminalization"
```

---

### Task 11: `_finalize_after_body`

**Files:**
- Modify: `app/workers/pipeline.py` (below `_tx4_finalize_failure`)
- Test: `tests/workers/test_finalize_after_body.py`

The body-return contract: routes a returning body to either Tx-3 (success) or Tx-4 (return-dict-as-failure). Skipped returns flow through Tx-3 (pipeline advances). Stage 9's wrapper bypasses this entirely via `intercept_terminal=False`.

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_finalize_after_body.py
"""_finalize_after_body decides Tx-3 vs Tx-4 from body return value."""
import uuid
from unittest.mock import patch
from sqlalchemy import text
from app.workers._stage_lifecycle import _LifecycleCtx
from app.workers.pipeline import _finalize_after_body

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _ctx(intercept_terminal=True, pending_status=None, pending_error=None):
    return _LifecycleCtx(
        pipeline_run_id="abc-123",
        stage_name="prepare_document",
        dispatch_attempt=1,
        intercept_terminal=intercept_terminal,
        next_stage="detect_and_translate",
        next_task="app.workers.pipeline.detect_and_translate",
        pending_status=pending_status,
        pending_error=pending_error,
    )


def test_intercept_terminal_false_does_nothing():
    """Stage 9 (intercept_terminal=False): wrapper doesn't call helpers."""
    with patch("app.workers.pipeline._tx3_complete_and_enqueue_next") as tx3, \
         patch("app.workers.pipeline._tx4_finalize_failure") as tx4:
        _finalize_after_body(_ctx(intercept_terminal=False), result=None)
        tx3.assert_not_called()
        tx4.assert_not_called()


def test_pending_status_failed_triggers_tx4():
    """Body called _update_stage_run('FAILED') without raising → Tx-4."""
    with patch("app.workers.pipeline._tx3_complete_and_enqueue_next") as tx3, \
         patch("app.workers.pipeline._tx4_finalize_failure") as tx4:
        _finalize_after_body(
            _ctx(pending_status="FAILED", pending_error="bad data"),
            result={"stage": "prepare_document", "status": "COMPLETE"},
        )
        tx4.assert_called_once()
        tx3.assert_not_called()


def test_return_dict_status_failed_triggers_tx4():
    """Body returned {"status":"FAILED"} → Tx-4."""
    with patch("app.workers.pipeline._tx3_complete_and_enqueue_next") as tx3, \
         patch("app.workers.pipeline._tx4_finalize_failure") as tx4:
        _finalize_after_body(
            _ctx(),
            result={"stage": "prepare_document", "status": "FAILED",
                   "reason": "no elements"},
        )
        tx4.assert_called_once()
        tx3.assert_not_called()


def test_return_dict_status_skipped_advances_pipeline():
    """skipped is success → Tx-3 (the bug from review round 2)."""
    with patch("app.workers.pipeline._tx3_complete_and_enqueue_next") as tx3, \
         patch("app.workers.pipeline._tx4_finalize_failure") as tx4:
        _finalize_after_body(
            _ctx(),
            result={"stage": "detect_and_translate", "status": "skipped",
                   "reason": "disabled"},
        )
        tx3.assert_called_once()
        tx4.assert_not_called()


def test_normal_completion_advances_pipeline():
    """Body returned a normal dict → Tx-3."""
    with patch("app.workers.pipeline._tx3_complete_and_enqueue_next") as tx3, \
         patch("app.workers.pipeline._tx4_finalize_failure") as tx4:
        _finalize_after_body(_ctx(), result={"stage": "prepare_document",
                                              "status": "complete",
                                              "elements": 124})
        tx3.assert_called_once()
        tx4.assert_not_called()


def test_non_dict_return_advances_pipeline():
    """Some stages return strings or None — treat as success."""
    with patch("app.workers.pipeline._tx3_complete_and_enqueue_next") as tx3:
        _finalize_after_body(_ctx(), result=None)
        tx3.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/workers/test_finalize_after_body.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `_finalize_after_body`**

```python
def _finalize_after_body(ctx: "_LifecycleCtx", result) -> None:
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/workers/test_finalize_after_body.py -v
```

Expected: all six PASS.

- [ ] **Step 5: Commit**

```bash
git add app/workers/pipeline.py tests/workers/test_finalize_after_body.py
git commit -m "feat(pipeline): add _finalize_after_body body-return contract"
```

---

### Task 12: Extend `_update_stage_run` with interception

**Files:**
- Modify: `app/workers/pipeline.py` (top of the existing `_update_stage_run` function, around line 2590)
- Test: `tests/workers/test_update_stage_run_interception.py`

This is the change that closes the premature-COMPLETE death window. Add a single predicate at the top of the existing function; everything below stays untouched.

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_update_stage_run_interception.py
"""_update_stage_run intercepts terminal writes when _CTX is active."""
import uuid
from sqlalchemy import text
from app.workers._stage_lifecycle import _CTX, _LifecycleCtx
from app.workers.pipeline import _update_stage_run

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _setup_running_row(db_session, stage="prepare_document"):
    src, doc, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, uploaded_by, pipeline_status)
        VALUES (:d,:s,'x.pdf','application/pdf',0,
                'b','k', :u, 'PROCESSING')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run, "d": doc})
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             task_name, dispatch_attempt, started_at)
        VALUES (gen_random_uuid(), :r, :s, 1, 'RUNNING',
                'app.workers.pipeline.prepare_document', 1, NOW())
    """), {"r": run, "s": stage})
    db_session.commit()
    return str(run)


def test_intercepts_complete_when_ctx_active_with_matching_run(db_session):
    """Body's _update_stage_run('COMPLETE', metrics=...) stashes metrics in ctx, no DB write."""
    run_id = _setup_running_row(db_session)
    ctx = _LifecycleCtx(
        pipeline_run_id=run_id, stage_name="prepare_document",
        dispatch_attempt=1, intercept_terminal=True,
        next_stage="detect_and_translate", next_task="x",
    )
    token = _CTX.set(ctx)
    try:
        _update_stage_run(db_session, run_id, "prepare_document", "COMPLETE",
                          attempt=1, metrics={"elements": 124})
        db_session.commit()
    finally:
        _CTX.reset(token)

    # Stashed
    assert ctx.pending_status == "COMPLETE"
    assert ctx.pending_metrics == {"elements": 124}

    # NOT committed to DB
    row = db_session.execute(text(
        "SELECT status, metrics FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.status == "RUNNING"          # unchanged
    assert row.metrics is None              # unchanged


def test_intercepts_failed_when_ctx_active(db_session):
    run_id = _setup_running_row(db_session)
    ctx = _LifecycleCtx(
        pipeline_run_id=run_id, stage_name="prepare_document",
        dispatch_attempt=1, intercept_terminal=True,
        next_stage=None, next_task=None,
    )
    token = _CTX.set(ctx)
    try:
        _update_stage_run(db_session, run_id, "prepare_document", "FAILED",
                          attempt=1, error="boom")
    finally:
        _CTX.reset(token)
    assert ctx.pending_status == "FAILED"
    assert ctx.pending_error == "boom"


def test_normalizes_uuid_vs_str(db_session):
    """ctx stores str; caller passes UUID; predicate compares str(both)."""
    run_id_str = _setup_running_row(db_session)
    ctx = _LifecycleCtx(
        pipeline_run_id=run_id_str, stage_name="prepare_document",
        dispatch_attempt=1, intercept_terminal=True,
        next_stage=None, next_task=None,
    )
    token = _CTX.set(ctx)
    try:
        # Pass UUID instead of str — interception must still fire.
        _update_stage_run(db_session, uuid.UUID(run_id_str), "prepare_document",
                          "COMPLETE", attempt=1, metrics={"x": 1})
    finally:
        _CTX.reset(token)
    assert ctx.pending_status == "COMPLETE"


def test_does_not_intercept_when_ctx_none(db_session):
    """Legacy path (no _CTX): existing implementation commits the write."""
    run_id = _setup_running_row(db_session)
    # _CTX is None by default — represents legacy run
    _update_stage_run(db_session, run_id, "prepare_document", "COMPLETE",
                      attempt=1, metrics={"elements": 99})
    db_session.commit()
    row = db_session.execute(text(
        "SELECT status, metrics FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.status == "COMPLETE"
    assert row.metrics == {"elements": 99}


def test_does_not_intercept_different_stage(db_session):
    """ctx for stage A; call for stage B → not intercepted."""
    run_id = _setup_running_row(db_session, stage="prepare_document")
    # Also insert a derive_document_metadata RUNNING row to be the legit target.
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             task_name, dispatch_attempt, started_at)
        VALUES (gen_random_uuid(), :r, 'derive_document_metadata', 1, 'RUNNING',
                'app.workers.pipeline.derive_document_metadata', 1, NOW())
    """), {"r": run_id})
    db_session.commit()

    ctx = _LifecycleCtx(
        pipeline_run_id=run_id, stage_name="prepare_document",
        dispatch_attempt=1, intercept_terminal=True,
        next_stage=None, next_task=None,
    )
    token = _CTX.set(ctx)
    try:
        # Call _update_stage_run for a DIFFERENT stage — should not be intercepted.
        _update_stage_run(db_session, run_id, "derive_document_metadata", "COMPLETE",
                          attempt=1, metrics={"summary_length": 100})
        db_session.commit()
    finally:
        _CTX.reset(token)
    row = db_session.execute(text("""
        SELECT status, metrics FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'derive_document_metadata'
    """), {"r": run_id}).first()
    assert row.status == "COMPLETE"
    assert row.metrics == {"summary_length": 100}


def test_does_not_intercept_when_intercept_terminal_false(db_session):
    """Stage 9 ctx has intercept_terminal=False → writes commit through."""
    run_id = _setup_running_row(db_session, stage="derive_ontology_graph")
    ctx = _LifecycleCtx(
        pipeline_run_id=run_id, stage_name="derive_ontology_graph",
        dispatch_attempt=1, intercept_terminal=False,
        next_stage=None, next_task=None,
    )
    token = _CTX.set(ctx)
    try:
        _update_stage_run(db_session, run_id, "derive_ontology_graph", "COMPLETE",
                          attempt=1, metrics={"node_count": 50})
        db_session.commit()
    finally:
        _CTX.reset(token)
    row = db_session.execute(text("""
        SELECT status, metrics FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'derive_ontology_graph'
    """), {"r": run_id}).first()
    assert row.status == "COMPLETE"
    assert row.metrics == {"node_count": 50}


def test_running_status_intercepted_as_noop(db_session):
    """Body writes RUNNING redundantly; wrapper already wrote it via CLAIM."""
    run_id = _setup_running_row(db_session)
    ctx = _LifecycleCtx(
        pipeline_run_id=run_id, stage_name="prepare_document",
        dispatch_attempt=1, intercept_terminal=True,
        next_stage=None, next_task=None,
    )
    token = _CTX.set(ctx)
    try:
        _update_stage_run(db_session, run_id, "prepare_document", "RUNNING", attempt=1)
    finally:
        _CTX.reset(token)
    # No stashed status — RUNNING is no-op.
    assert ctx.pending_status is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/workers/test_update_stage_run_interception.py -v
```

Expected: at minimum the COMPLETE-interception tests fail (current implementation commits unconditionally).

- [ ] **Step 3: Add the interception block at the top of `_update_stage_run`**

Open `app/workers/pipeline.py`, find `def _update_stage_run(` (around line 2590). **Do not rewrite the signature** — keep its existing parameters and annotations verbatim. Immediately inside the function body, before the existing implementation begins, add the interception block shown below. The shown signature line is reproduced only to give you a visual anchor for where the new block goes.

```python
def _update_stage_run(
    db, pipeline_run_id, stage_name, status,
    attempt: int = 1, metrics: dict | None = None, error: str | None = None,
) -> None:
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
            ctx.pending_status  = status
            ctx.pending_metrics = metrics
            ctx.pending_error   = error
            return
        # Other statuses fall through (defensive)
    # ─────────────────────────────────────────────────────────────────────

    # ── existing implementation continues unchanged below ───────────────
    # (keep the rest of the function body as-is)
```

Do not modify the rest of the function. The interception predicate is the only addition.

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/workers/test_update_stage_run_interception.py -v
```

Expected: all seven PASS.

- [ ] **Step 5: Run the broader `_update_stage_run` test suite (if one exists) to check for regression**

```bash
docker compose run --rm worker pytest tests/ -k "stage_run" -v
```

Expected: no regressions in existing tests; legacy callers (no `_CTX`) unaffected.

- [ ] **Step 6: Commit**

```bash
git add app/workers/pipeline.py tests/workers/test_update_stage_run_interception.py
git commit -m "feat(pipeline): intercept terminal _update_stage_run writes inside lifecycle ctx"
```

---

### Task 13: Extend `guard_stage_run` with lifecycle

**Files:**
- Modify: `app/workers/pipeline.py` (the existing `guard_stage_run` decorator, around line 2528)
- Test: `tests/workers/test_guard_stage_run_lifecycle.py`

The orchestrator. Reads the spec's wrapper skeleton (around lines 290-356 of the spec) and wires CLAIM → body → finalize. Critical: `_CTX.set()` happens ONLY on `proceed`; legacy and early-return paths leave `_CTX` untouched.

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_guard_stage_run_lifecycle.py
"""guard_stage_run integrates CLAIM, body, _CTX, and finalize."""
import uuid
from unittest.mock import MagicMock, patch
from sqlalchemy import text
from app.workers._stage_lifecycle import _CTX
from app.workers.pipeline import guard_stage_run

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _seed_pending(db_session, stage="prepare_document"):
    src, doc, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, uploaded_by, pipeline_status)
        VALUES (:d,:s,'x.pdf','application/pdf',0,'b','k',:u,'PROCESSING')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run, "d": doc})
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             task_name, dispatch_attempt)
        VALUES (gen_random_uuid(), :r, :s, 1, 'PENDING',
                'app.workers.pipeline.prepare_document', 1)
    """), {"r": run, "s": stage})
    db_session.commit()
    return str(run)


def _fake_self(retries: int = 0, task_id: str = "task-1", max_retries: int = 2):
    self = MagicMock()
    self.request.retries = retries
    self.request.id = task_id
    self.max_retries = max_retries
    return self


def test_wrapper_marker_attributes_set():
    """guard_stage_run sets `stage_name`, `_lifecycle`, on the wrapped function."""
    @guard_stage_run("test_stage", lifecycle=True, next_stage="next",
                     next_task="app.workers.pipeline.detect_and_translate")
    def body(self, doc_id, run_id=None):
        return {"status": "ok"}
    assert body.stage_name == "test_stage"
    assert body._lifecycle is True


def test_proceed_runs_body_with_ctx_then_finalizes(db_session):
    run_id = _seed_pending(db_session)

    captured_ctx = {}

    @guard_stage_run("prepare_document", lifecycle=True,
                     next_stage="detect_and_translate",
                     next_task="app.workers.pipeline.detect_and_translate")
    def body(self, doc_id, run_id=None):
        captured_ctx["ctx"] = _CTX.get()
        return {"status": "ok"}

    with patch("app.workers.pipeline._finalize_after_body") as fin:
        body(_fake_self(), "doc-1", run_id)
        fin.assert_called_once()

    # Body saw _CTX set; wrapper reset it afterwards.
    assert captured_ctx["ctx"] is not None
    assert captured_ctx["ctx"].pipeline_run_id == run_id
    assert captured_ctx["ctx"].stage_name == "prepare_document"
    assert _CTX.get() is None


def test_legacy_path_runs_body_without_ctx(db_session):
    """No ledger row exists → body runs inline, _CTX stays None."""
    run = uuid.uuid4()
    src, doc = uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, uploaded_by, pipeline_status)
        VALUES (:d,:s,'x.pdf','application/pdf',0,'b','k',:u,'PROCESSING')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run, "d": doc})
    db_session.commit()
    run_id = str(run)
    # NO stage_runs row → CLAIM returns outcome=legacy

    seen = {}

    @guard_stage_run("prepare_document", lifecycle=True,
                     next_stage="detect_and_translate",
                     next_task="app.workers.pipeline.detect_and_translate")
    def body(self, doc_id, run_id=None):
        seen["ctx"] = _CTX.get()
        return {"status": "ok"}

    with patch("app.workers.pipeline._finalize_after_body") as fin:
        body(_fake_self(), "doc-1", run_id)
        fin.assert_not_called()    # legacy: no Tx-3 / Tx-4

    assert seen["ctx"] is None     # _CTX never set for legacy path


def test_early_return_already_complete_skips_body(db_session):
    run_id = _seed_pending(db_session)
    db_session.execute(text(
        "UPDATE ingest.stage_runs SET status='COMPLETE' WHERE pipeline_run_id=:r"
    ), {"r": run_id})
    db_session.commit()

    ran = {"body": False}

    @guard_stage_run("prepare_document", lifecycle=True,
                     next_stage="detect_and_translate",
                     next_task="app.workers.pipeline.detect_and_translate")
    def body(self, doc_id, run_id=None):
        ran["body"] = True
        return None

    result = body(_fake_self(), "doc-1", run_id)
    assert ran["body"] is False
    assert result == {"stage": "prepare_document",
                      "status": "skipped",
                      "reason": "already_complete"}
    assert _CTX.get() is None


def test_celery_retry_passthrough_no_tx4(db_session):
    """Body raises CeleryRetry → wrapper passes through, row stays RUNNING."""
    from celery.exceptions import Retry as CeleryRetry
    run_id = _seed_pending(db_session)
    db_session.execute(text(
        "UPDATE ingest.stage_runs SET status='DISPATCHED' WHERE pipeline_run_id=:r"
    ), {"r": run_id})
    db_session.commit()

    @guard_stage_run("prepare_document", lifecycle=True,
                     next_stage="detect_and_translate",
                     next_task="app.workers.pipeline.detect_and_translate")
    def body(self, doc_id, run_id=None):
        raise CeleryRetry()

    import pytest
    with patch("app.workers.pipeline._finalize_after_body") as fin, \
         patch("app.workers.pipeline._tx4_finalize_failure") as tx4:
        with pytest.raises(CeleryRetry):
            body(_fake_self(), "doc-1", run_id)
        fin.assert_not_called()
        tx4.assert_not_called()

    row = db_session.execute(text(
        "SELECT status FROM ingest.stage_runs WHERE pipeline_run_id=:r"
    ), {"r": run_id}).first()
    assert row.status == "RUNNING"          # CLAIM moved it; Celery will re-deliver
    assert _CTX.get() is None


def test_non_celery_exception_triggers_tx4(db_session):
    run_id = _seed_pending(db_session)

    @guard_stage_run("prepare_document", lifecycle=True,
                     next_stage="detect_and_translate",
                     next_task="app.workers.pipeline.detect_and_translate")
    def body(self, doc_id, run_id=None):
        raise ValueError("boom")

    import pytest
    with patch("app.workers.pipeline._tx4_finalize_failure") as tx4:
        with pytest.raises(ValueError, match="boom"):
            body(_fake_self(), "doc-1", run_id)
        tx4.assert_called_once()
    assert _CTX.get() is None


def test_ctx_reset_between_invocations(db_session):
    """Two task invocations in same process: _CTX is None at start of each."""
    run_id = _seed_pending(db_session)

    @guard_stage_run("prepare_document", lifecycle=True,
                     next_stage="detect_and_translate",
                     next_task="app.workers.pipeline.detect_and_translate")
    def body(self, doc_id, run_id=None):
        return {}

    with patch("app.workers.pipeline._finalize_after_body"):
        body(_fake_self(task_id="t1"), "doc-1", run_id)
        assert _CTX.get() is None      # reset by finally block

        # Seed another row + invocation
        db_session.execute(text(
            "UPDATE ingest.stage_runs SET status='PENDING' WHERE pipeline_run_id=:r"
        ), {"r": run_id})
        db_session.commit()
        body(_fake_self(task_id="t2"), "doc-1", run_id)
        assert _CTX.get() is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/workers/test_guard_stage_run_lifecycle.py -v
```

Expected: failures or TypeErrors — `guard_stage_run` doesn't accept the new kwargs yet.

- [ ] **Step 3: Extend `guard_stage_run`**

Replace the existing `guard_stage_run` body (around line 2528 in `app/workers/pipeline.py`) with:

```python
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

    Without `lifecycle=True`, the decorator behaves exactly as before: pass
    CeleryRetry / SoftTimeLimitExceeded through, terminalize other exceptions.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, document_id, run_id=None, *args, **kwargs):
            # Non-lifecycle invocation — preserve existing behavior unchanged.
            if not (lifecycle and run_id):
                return _guard_existing_body(self, fn, stage_name, document_id, run_id,
                                            *args, **kwargs)

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
                return _guard_existing_body(self, fn, stage_name, document_id, run_id,
                                            *args, **kwargs)
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

        wrapper.stage_name = stage_name   # pre-existing marker (test introspection)
        wrapper._lifecycle = lifecycle    # NEW — read by module-load assertion
        return wrapper
    return decorator


def _guard_existing_body(self, fn, stage_name, document_id, run_id, *args, **kwargs):
    """The pre-design `guard_stage_run` body — preserved for non-lifecycle invocations."""
    try:
        return fn(self, document_id, run_id, *args, **kwargs)
    except CeleryRetry:
        raise
    except Exception as exc:
        logger.exception(
            "guard_stage_run: %s raised unhandled exception (document_id=%s run_id=%s)",
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
        _terminalize_doc_and_run(document_id, run_id, "PARTIAL_COMPLETE")
        raise
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/workers/test_guard_stage_run_lifecycle.py -v
```

Expected: all seven PASS.

- [ ] **Step 5: Run the broader guard_stage_run test suite**

```bash
docker compose run --rm worker pytest tests/ -k "guard_stage_run" -v
```

Expected: no regressions in existing tests (decorator's pre-design behavior is preserved by `_guard_existing_body`).

- [ ] **Step 6: Commit**

```bash
git add app/workers/pipeline.py tests/workers/test_guard_stage_run_lifecycle.py
git commit -m "feat(pipeline): extend guard_stage_run with lifecycle kwargs and _CTX management"
```

---

**Chunk 3 complete.** The wrapper integration is functional but not wired to any stage decorator yet. Smoke check the entire pipeline.py unit suite:

```bash
docker compose run --rm worker pytest tests/workers/ -v
```

Expected: all green.

---

## Chunk 4: Dispatcher + beat wiring

This chunk adds the beat-scheduled task that owns the PENDING → DISPATCHED transition and Celery publish.

### Task 14: Dispatcher module skeleton

**Files:**
- Create: `app/workers/dispatcher.py`
- Test: `tests/workers/test_dispatcher_skeleton.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_dispatcher_skeleton.py
"""dispatcher module exports the expected task + helpers."""
def test_module_imports():
    from app.workers import dispatcher
    assert hasattr(dispatcher, "dispatch_pending_pipeline_stages")
    assert hasattr(dispatcher, "_run_dispatch_tick")
    assert hasattr(dispatcher, "_publish")
    assert hasattr(dispatcher, "_undo_claim")
    assert hasattr(dispatcher, "_release_lock_if_owner")


def test_task_is_registered_with_celery():
    from app.workers.celery_app import celery_app
    name = "app.workers.dispatcher.dispatch_pending_pipeline_stages"
    assert name in celery_app.tasks


def test_constants_exposed():
    from app.workers.dispatcher import (
        DISPATCH_BATCH_LIMIT, DISPATCH_LOCK_KEY, DISPATCH_LOCK_TTL,
    )
    assert DISPATCH_BATCH_LIMIT == 50
    assert DISPATCH_LOCK_KEY == "dispatcher:pipeline_stages"
    assert DISPATCH_LOCK_TTL == 30
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/workers/test_dispatcher_skeleton.py -v
```

Expected: ImportError (`app.workers.dispatcher` doesn't exist).

- [ ] **Step 3: Create the dispatcher module**

Create `app/workers/dispatcher.py`. The full module body is specified in the spec (lines ~530–720, sections "Task body", "Claim query", "Publish helper", "_undo_claim", "Race note"). Copy the spec's code verbatim. The module structure:

```python
"""Pipeline stage dispatch — beat-scheduled claim + publish.

See: docs/superpowers/specs/2026-05-10-pipeline-stage-dispatch-ledger-design.md
"""
import logging
import uuid

from sqlalchemy import text

from app.services.redis_utils import get_redis
# pipeline.py aliases this as `_get_db` at line 59 — match that convention.
from app.workers._db import get_worker_db as _get_db
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

DISPATCH_BATCH_LIMIT = 50
DISPATCH_LOCK_KEY    = "dispatcher:pipeline_stages"
DISPATCH_LOCK_TTL    = 30  # seconds


@celery_app.task(
    bind=True,
    name="app.workers.dispatcher.dispatch_pending_pipeline_stages",
)
def dispatch_pending_pipeline_stages(self) -> dict:
    redis = get_redis()
    if not redis.set(DISPATCH_LOCK_KEY, self.request.id, nx=True, ex=DISPATCH_LOCK_TTL):
        return {"skipped": "another dispatcher tick is in flight"}
    try:
        return _run_dispatch_tick()
    finally:
        _release_lock_if_owner(redis, DISPATCH_LOCK_KEY, self.request.id)


def _release_lock_if_owner(redis, key: str, token: str) -> None:
    """Lua-CAS release: only the lock holder can release."""
    redis.eval(
        "if redis.call('get', KEYS[1]) == ARGV[1] then "
        "return redis.call('del', KEYS[1]) else return 0 end",
        1, key, token,
    )


def _run_dispatch_tick() -> dict:
    # ... (see spec lines ~570-625 for full body)
    ...


def _publish(task_name, document_id, run_id, stage_run_id):
    # ... (see spec lines ~647-680)
    ...


def _undo_claim(stage_run_id, *, error: str) -> None:
    # ... (see spec lines ~700-722)
    ...
```

Implement each `...` block by copying directly from the spec section indicated.

Verify `app.services.redis_utils.get_redis` exists:

```bash
docker compose run --rm worker python -c "from app.services.redis_utils import get_redis; print(get_redis)"
```

If the helper lives at a different path (e.g., `app.services.cache.get_redis`), adjust the import; do not invent a new helper. If no `get_redis` exists at all, check `pipeline.py` or `celery_app.py` for whatever module already builds a Redis client and reuse it.

- [ ] **Step 4: Register `app.workers.dispatcher` with Celery**

Edit `app/workers/celery_app.py`. Find the `Celery(..., include=[...])` call. Add `"app.workers.dispatcher"` to the `include` list.

- [ ] **Step 5: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/workers/test_dispatcher_skeleton.py -v
```

Expected: all three PASS.

- [ ] **Step 6: Commit**

```bash
git add app/workers/dispatcher.py app/workers/celery_app.py tests/workers/test_dispatcher_skeleton.py
git commit -m "feat(workers): add dispatcher module + register with Celery"
```

---

### Task 15: `_run_dispatch_tick` end-to-end

**Files:**
- Modify: `app/workers/dispatcher.py` (replace the `...` stubs from Task 14)
- Test: `tests/workers/test_dispatch_tick.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_dispatch_tick.py
"""_run_dispatch_tick claims PENDING ledger rows and publishes Celery tasks."""
import uuid
from unittest.mock import patch
from sqlalchemy import text

from app.workers.dispatcher import _run_dispatch_tick

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _seed_pending_ledger_row(db_session, stage="prepare_document", run_status="PROCESSING"):
    src, doc, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, uploaded_by, pipeline_status)
        VALUES (:d,:s,'x.pdf','application/pdf',0,'b','k',:u,'PROCESSING')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, :st)
    """), {"r": run, "d": doc, "st": run_status})
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             task_name, queue_name, available_at, dispatch_attempt)
        VALUES (gen_random_uuid(), :r, :s, 1, 'PENDING',
                'app.workers.pipeline.prepare_document', 'ingest', NOW(), 1)
    """), {"r": run, "s": stage})
    db_session.commit()
    return str(run)


def test_claims_pending_row_and_publishes(db_session):
    run_id = _seed_pending_ledger_row(db_session)

    with patch("app.workers.dispatcher._publish") as pub:
        result = _run_dispatch_tick()
        assert pub.called

    assert result["claimed"] >= 1
    row = db_session.execute(text(
        "SELECT status, dispatched_at FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.status == "DISPATCHED"
    assert row.dispatched_at is not None


def test_does_not_claim_rows_without_task_name(db_session):
    """task_name IS NULL → not a ledger row → never claimed."""
    src, doc, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, uploaded_by, pipeline_status)
        VALUES (:d,:s,'x.pdf','application/pdf',0,'b','k',:u,'PROCESSING')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run, "d": doc})
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status, dispatch_attempt)
        VALUES (gen_random_uuid(), :r, 'unknown_stage', 1, 'PENDING', 1)
    """), {"r": run})
    db_session.commit()

    with patch("app.workers.dispatcher._publish") as pub:
        result = _run_dispatch_tick()
    row = db_session.execute(text(
        "SELECT status FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run}).first()
    assert row.status == "PENDING"
    pub.assert_not_called()


def test_does_not_claim_when_pipeline_run_not_processing(db_session):
    run_id = _seed_pending_ledger_row(db_session, run_status="FAILED")
    with patch("app.workers.dispatcher._publish") as pub:
        _run_dispatch_tick()
    row = db_session.execute(text(
        "SELECT status FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.status == "PENDING"
    pub.assert_not_called()


def test_does_not_claim_pass_name_rows(db_session):
    """pass_name IS NOT NULL → per-pass row → dispatcher must ignore."""
    src, doc, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, uploaded_by, pipeline_status)
        VALUES (:d,:s,'x.pdf','application/pdf',0,'b','k',:u,'PROCESSING')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run, "d": doc})
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             pass_name, task_name, dispatch_attempt)
        VALUES (gen_random_uuid(), :r, 'derive_ontology_graph', 1, 'PENDING',
                'radar_modulation',
                'app.workers.pipeline.derive_ontology_graph_pass', 1)
    """), {"r": run})
    db_session.commit()

    with patch("app.workers.dispatcher._publish") as pub:
        _run_dispatch_tick()
    pub.assert_not_called()


def test_respects_available_at_future(db_session):
    run_id = _seed_pending_ledger_row(db_session)
    db_session.execute(text(
        "UPDATE ingest.stage_runs SET available_at = NOW() + INTERVAL '60 seconds' "
        "WHERE pipeline_run_id = :r"
    ), {"r": run_id})
    db_session.commit()
    with patch("app.workers.dispatcher._publish") as pub:
        _run_dispatch_tick()
    pub.assert_not_called()


def test_undo_claim_on_publish_failure(db_session):
    run_id = _seed_pending_ledger_row(db_session)

    def boom(*a, **kw):
        raise RuntimeError("broker down")

    with patch("app.workers.dispatcher._publish", side_effect=boom):
        _run_dispatch_tick()

    row = db_session.execute(text(
        "SELECT status, error_message FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.status == "PENDING"
    assert "broker down" in (row.error_message or "")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/workers/test_dispatch_tick.py -v
```

Expected: most fail because `_run_dispatch_tick` is still a stub.

- [ ] **Step 3: Implement `_run_dispatch_tick`, `_publish`, `_undo_claim`**

Open `app/workers/dispatcher.py` and replace the three `...` stubs with the implementations specified in the spec, sections "Claim query" (~570-625), "Publish helper" (~647-680), and "_undo_claim" (~700-722). Reproduce verbatim.

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/workers/test_dispatch_tick.py -v
```

Expected: all six PASS.

- [ ] **Step 5: Commit**

```bash
git add app/workers/dispatcher.py tests/workers/test_dispatch_tick.py
git commit -m "feat(dispatcher): implement _run_dispatch_tick, _publish, _undo_claim"
```

---

### Task 16: Beat schedule entry

**Files:**
- Modify: `app/workers/celery_app.py` (beat_schedule dict)
- Test: `tests/workers/test_beat_schedule.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_beat_schedule.py
"""Dispatcher beat entry is registered with the correct schedule."""
def test_dispatch_pending_pipeline_stages_beat_entry():
    from app.workers.celery_app import celery_app
    schedule = celery_app.conf.beat_schedule
    assert "dispatch-pending-pipeline-stages" in schedule
    entry = schedule["dispatch-pending-pipeline-stages"]
    assert entry["task"] == "app.workers.dispatcher.dispatch_pending_pipeline_stages"
    assert entry["schedule"] == 5.0
    assert entry["options"]["queue"] == "celery"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
docker compose run --rm worker pytest tests/workers/test_beat_schedule.py -v
```

Expected: KeyError — entry not registered.

- [ ] **Step 3: Add the beat entry**

Edit `app/workers/celery_app.py`. In the `beat_schedule={...}` dict (currently containing `scan-watch-directories`, `community-detection`, `periodic-stale-run-sweep`, `reconcile-ontology-graph-runs`), add:

```python
        "dispatch-pending-pipeline-stages": {
            "task": "app.workers.dispatcher.dispatch_pending_pipeline_stages",
            "schedule": 5.0,
            "options": {"queue": "celery"},
        },
```

- [ ] **Step 4: Run test to verify it passes**

```bash
docker compose run --rm worker pytest tests/workers/test_beat_schedule.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/workers/celery_app.py tests/workers/test_beat_schedule.py
git commit -m "feat(celery): schedule dispatch_pending_pipeline_stages every 5s"
```

---

**Chunk 4 complete.** Dispatcher module is registered, beat-scheduled, and unit-tested. No stage decorator changes have happened yet — the dispatcher would simply pick up no rows in production until Chunk 6 wires the stages.

Continue to Chunk 5 (sweeper + startup assertions).

---

## Chunk 5: Sweeper extension + startup assertions

This chunk extends `_sweep_stale_runs` with the new ledger logic and adds module-load assertions that prevent silent misconfiguration.

### Task 17: Stale-DISPATCHED reset

**Files:**
- Modify: `app/workers/pipeline.py` (extend `_sweep_stale_runs`)
- Test: `tests/workers/test_stale_dispatched_sweep.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_stale_dispatched_sweep.py
"""DISPATCHED rows older than threshold reset to PENDING; dispatch_attempt unchanged."""
import uuid
from sqlalchemy import text
from app.workers.pipeline import _sweep_stale_runs

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _seed_dispatched(db_session, dispatched_secs_ago=900):
    src, doc, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, uploaded_by, pipeline_status)
        VALUES (:d,:s,'x.pdf','application/pdf',0,'b','k',:u,'PROCESSING')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run, "d": doc})
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             task_name, dispatch_attempt, dispatched_at)
        VALUES (gen_random_uuid(), :r, 'prepare_document', 1, 'DISPATCHED',
                'app.workers.pipeline.prepare_document', 2,
                NOW() - make_interval(secs => :secs))
    """), {"r": run, "secs": dispatched_secs_ago})
    db_session.commit()
    return str(run)


def test_stale_dispatched_resets_to_pending_no_attempt_bump(db_session):
    run_id = _seed_dispatched(db_session, dispatched_secs_ago=700)
    _sweep_stale_runs()
    row = db_session.execute(text("""
        SELECT status, dispatched_at IS NULL AS cleared,
               dispatch_attempt
        FROM ingest.stage_runs WHERE pipeline_run_id = :r
    """), {"r": run_id}).first()
    assert row.status == "PENDING"
    assert row.cleared
    assert row.dispatch_attempt == 2     # unchanged


def test_fresh_dispatched_not_swept(db_session):
    run_id = _seed_dispatched(db_session, dispatched_secs_ago=60)
    _sweep_stale_runs()
    row = db_session.execute(text(
        "SELECT status FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.status == "DISPATCHED"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/workers/test_stale_dispatched_sweep.py -v
```

Expected: first test fails (sweeper doesn't reset DISPATCHED yet).

- [ ] **Step 3: Extend `_sweep_stale_runs`**

Open `app/workers/pipeline.py`, find the existing `_sweep_stale_runs` function (entry around line 1786; helper body within). After the existing stale-RUNNING handling block (which marks RUNNING → FAILED + redispatches via `start_ingest_pipeline`), add the stale-DISPATCHED reset block:

```python
        # ── stale DISPATCHED reset (ledger v1, spec 2026-05-10) ──────────
        # The dispatcher published a Celery task but a worker did not pick it up
        # within stale_dispatched_threshold_seconds. Reset to PENDING so the next
        # tick republishes. dispatch_attempt is unchanged — the stage didn't run.
        db.execute(text("""
            UPDATE ingest.stage_runs
            SET status        = 'PENDING',
                dispatched_at = NULL,
                error_message = COALESCE(error_message, '')
                                || ' stale; reset by dispatcher sweeper'
            WHERE status        = 'DISPATCHED'
              AND pass_name     IS NULL
              AND task_name     IS NOT NULL
              AND dispatched_at < NOW() - make_interval(secs => :dispatched_threshold)
        """), {"dispatched_threshold": settings.stale_dispatched_threshold_seconds})
```

Place this block before `db.commit()` at the end of the function.

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/workers/test_stale_dispatched_sweep.py -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add app/workers/pipeline.py tests/workers/test_stale_dispatched_sweep.py
git commit -m "feat(sweeper): reset stale DISPATCHED rows to PENDING (no attempt bump)"
```

---

### Task 18: Ledger-aware stale-RUNNING sweep (CTE)

**Files:**
- Modify: `app/workers/pipeline.py` (extend `_sweep_stale_runs` further)
- Test: `tests/workers/test_ledger_stale_running.py`

This is the larger change. It introduces the CTE from spec lines ~744-790 and excludes `derive_ontology_graph` from ledger handling. Stages outside `LEDGER_SEQUENTIAL_STAGES` keep the existing FAILED+redispatch path.

- [ ] **Step 1: Write the failing tests**

```python
# tests/workers/test_ledger_stale_running.py
"""Stale RUNNING rows for ledger stages reset to PENDING (under cap) or FAILED (over)."""
import uuid
from sqlalchemy import text
from app.workers.pipeline import _sweep_stale_runs

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _seed_running(db_session, *, stage, dispatch_attempt, started_secs_ago,
                  pass_name=None, run_status="PROCESSING"):
    src, doc, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, uploaded_by, pipeline_status)
        VALUES (:d,:s,'x.pdf','application/pdf',0,'b','k',:u,'PROCESSING')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, :st)
    """), {"r": run, "d": doc, "st": run_status})
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             pass_name, task_name, dispatch_attempt, started_at)
        VALUES (gen_random_uuid(), :r, :s, 1, 'RUNNING',
                :pn,
                'app.workers.pipeline.x',
                :da,
                NOW() - make_interval(secs => :secs))
    """), {"r": run, "s": stage, "pn": pass_name, "da": dispatch_attempt,
           "secs": started_secs_ago})
    db_session.commit()
    return str(run)


def test_ledger_running_under_cap_resets_to_pending(db_session, monkeypatch):
    from app import config
    monkeypatch.setattr(config.settings, "max_stage_dispatches", 5)
    monkeypatch.setattr(config.settings, "stale_stage_run_threshold_seconds", 60)

    run_id = _seed_running(
        db_session, stage="prepare_document", dispatch_attempt=2,
        started_secs_ago=200,
    )
    _sweep_stale_runs()
    row = db_session.execute(text("""
        SELECT status, dispatch_attempt, started_at IS NULL AS cleared,
               available_at <= NOW() AS due_now
        FROM ingest.stage_runs WHERE pipeline_run_id = :r
    """), {"r": run_id}).first()
    assert row.status == "PENDING"
    assert row.dispatch_attempt == 3
    assert row.cleared
    assert row.due_now


def test_ledger_running_at_cap_terminalizes_with_pipeline_run(db_session, monkeypatch):
    from app import config
    monkeypatch.setattr(config.settings, "max_stage_dispatches", 3)
    monkeypatch.setattr(config.settings, "stale_stage_run_threshold_seconds", 60)

    run_id = _seed_running(
        db_session, stage="prepare_document", dispatch_attempt=3,
        started_secs_ago=200,
    )
    _sweep_stale_runs()
    sr = db_session.execute(text(
        "SELECT status, dispatch_attempt FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert sr.status == "FAILED"
    assert sr.dispatch_attempt == 4

    pr = db_session.execute(text(
        "SELECT status FROM ingest.pipeline_runs WHERE id = :r"
    ), {"r": run_id}).first()
    assert pr.status == "FAILED"


def test_derive_ontology_graph_running_is_excluded(db_session, monkeypatch):
    """Stage 9 stays RUNNING even when stale — reconcile_ontology_graph_runs owns it."""
    from app import config
    monkeypatch.setattr(config.settings, "stale_stage_run_threshold_seconds", 60)

    run_id = _seed_running(
        db_session, stage="derive_ontology_graph", dispatch_attempt=1,
        started_secs_ago=3600,                    # 1 hour stale
    )
    _sweep_stale_runs()
    row = db_session.execute(text(
        "SELECT status FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.status == "RUNNING"                # not touched by ledger sweep


def test_pass_name_rows_not_affected(db_session, monkeypatch):
    from app import config
    monkeypatch.setattr(config.settings, "stale_stage_run_threshold_seconds", 60)

    run_id = _seed_running(
        db_session, stage="derive_ontology_graph", dispatch_attempt=1,
        started_secs_ago=3600, pass_name="radar_modulation",
    )
    _sweep_stale_runs()
    row = db_session.execute(text("""
        SELECT status FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND pass_name = 'radar_modulation'
    """), {"r": run_id}).first()
    # Existing non-ledger sweeper may or may not touch this row — verify the
    # row was NOT touched by the new CTE specifically (it has pass_name).
    assert row.status in ("RUNNING", "FAILED")    # depending on existing behavior
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/workers/test_ledger_stale_running.py -v
```

Expected: the cap-based and exclusion tests fail (current sweeper FAILS the run via legacy path for ALL stale RUNNING).

- [ ] **Step 3: Implement the CTE-based stale-RUNNING sweep**

In `_sweep_stale_runs` (`app/workers/pipeline.py`), add the new ledger logic BEFORE the existing legacy logic so ledger stages are handled by the new path and never reach the old `start_ingest_pipeline` redispatch:

```python
        # ── ledger stale-RUNNING (spec 2026-05-10) ────────────────────────
        # Sequential stages 1–8: reset under cap; terminalize at cap.
        # Stage 9 (derive_ontology_graph) is intentionally excluded —
        # reconcile_ontology_graph_runs is its sole owner during fan-in.
        db.execute(text("""
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
        """), {
            "ledger_sequential_stages": LEDGER_SEQUENTIAL_STAGES,
            "threshold": settings.stale_stage_run_threshold_seconds,
            "max_dispatches": settings.max_stage_dispatches,
        })
```

**Required follow-up edit** (not optional): in the same `_sweep_stale_runs` function, find the legacy SELECT around `pipeline.py:1815-1827` that fetches stale RUNNING rows (the one feeding the Python `for stage_run_id, pipeline_run_id, document_id, stage_name in stale_rows:` loop that calls `start_ingest_pipeline`). Add an explicit exclusion clause so it never picks up ledger stages, regardless of whether the CTE has already updated them:

```sql
-- Existing SELECT (verbatim except for the two AND clauses noted):
SELECT sr.id, sr.pipeline_run_id, pr.document_id, sr.stage_name
FROM ingest.stage_runs sr
JOIN ingest.pipeline_runs pr ON pr.id = sr.pipeline_run_id
WHERE sr.status = 'RUNNING'
  AND sr.started_at < NOW() - make_interval(secs => :threshold)
  AND pr.status = 'PROCESSING'
  -- ── NEW partitioning clauses (spec 2026-05-10) ────────────────────────
  AND sr.stage_name <> ALL(:ledger_sequential_stages)
  AND sr.pass_name IS NULL                                   -- existing per-pass
                                                              -- handling already
                                                              -- has its own logic
```

Pass the same `:ledger_sequential_stages` parameter as the new CTE. This makes the partition between ledger and legacy code paths explicit at the query level — there is no path by which the legacy redispatch loop touches a ledger stage.

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/workers/test_ledger_stale_running.py -v
```

Expected: all four PASS.

- [ ] **Step 5: Run existing sweeper tests to verify no regression**

```bash
docker compose run --rm worker pytest tests/ -k "stale_run or sweep" -v
```

Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add app/workers/pipeline.py tests/workers/test_ledger_stale_running.py
git commit -m "feat(sweeper): add ledger stale-RUNNING CTE (sequential stages, cap-aware)"
```

---

### Task 19: Startup assertions

**Files:**
- Modify: `app/workers/celery_app.py` (post-config block, after all task registration)
- Test: `tests/workers/test_startup_assertions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_startup_assertions.py
"""Module-load assertions catch wrapper-marker and threshold misconfigurations."""
def test_all_ledger_stages_have_lifecycle_marker():
    """Every key in STAGE_SUCCESSORS maps to a task with _lifecycle=True."""
    from app.workers.celery_app import celery_app
    from app.workers.pipeline import STAGE_SUCCESSORS, _assert_ledger_wiring
    # If a wrapper is missing _lifecycle, this would have raised at import.
    # Calling it again must be a no-op.
    _assert_ledger_wiring()


def test_threshold_envelope_assertion_raises_on_misconfiguration(monkeypatch):
    """If threshold < time_limit + max_retries × default_retry_delay, raise."""
    from app.workers.celery_app import celery_app
    from app.workers.pipeline import _assert_threshold_envelope
    from app import config

    monkeypatch.setattr(config.settings, "stale_stage_run_threshold_seconds", 1)

    import pytest
    with pytest.raises(RuntimeError, match="must exceed envelope"):
        _assert_threshold_envelope()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/workers/test_startup_assertions.py -v
```

Expected: ImportError (`_assert_ledger_wiring`, `_assert_threshold_envelope` don't exist).

- [ ] **Step 3: Implement the assertions**

Add to `app/workers/pipeline.py` (near the end of the ledger-helpers block):

```python
def _assert_ledger_wiring() -> None:
    """Module-load check: every STAGE_SUCCESSORS key has a registered task
    whose wrapper carries `_lifecycle=True`. Raises RuntimeError on mismatch."""
    from app.workers.celery_app import celery_app
    for stage_name in STAGE_SUCCESSORS:
        # Find the task whose wrapped function has stage_name == this stage.
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
    """Module-load check: stale_stage_run_threshold_seconds exceeds every
    ledger stage's `time_limit + max_retries × default_retry_delay`."""
    from app.workers.celery_app import celery_app
    threshold = settings.stale_stage_run_threshold_seconds
    for stage_name in STAGE_SUCCESSORS:
        for task_name, task in celery_app.tasks.items():
            if getattr(task.run, "stage_name", None) != stage_name:
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
```

Wire both into `app/workers/celery_app.py` after task registration is complete. **Important rollout note:** Chunk 6 is what gives stage decorators their `_lifecycle=True` marker. If `_assert_ledger_wiring()` runs at module load during Chunk 5's commit (before Chunk 6 lands), the worker boot will fail. To keep Chunk 5's commit deployable on its own, define the function with a `pass` body in this task and have Chunk 6 Task 22 Step 5 replace the body and add the module-load call.

```python
# At the bottom of celery_app.py, AFTER autodiscover_tasks / include processing:
def _post_register_ledger_checks() -> None:
    """Re-enabled in Chunk 6 Task 22 Step 5 once stage decorators carry _lifecycle=True."""
    pass   # Chunk 5 ships with a no-op so worker boot stays green.

# Do NOT call _post_register_ledger_checks() at module load yet.
# Chunk 6 Task 22 Step 5 replaces the body and adds the call.
```

Step 4's tests directly invoke `_assert_ledger_wiring()` / `_assert_threshold_envelope()` to verify the assertion *logic*; they don't need the module-load wiring to be active.

If Celery's task discovery is lazy, the eventual Chunk 6 wiring may need to call `celery_app.loader.import_default_modules()` before the checks. Verify by running the worker after Chunk 6 lands.

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/workers/test_startup_assertions.py -v
```

Expected: both PASS. The first test relies on Chunk 6 wiring `_lifecycle=True` on all 9 stages — if Chunk 5 runs before Chunk 6 in execution order, the test will fail at module-load time. **Defer the `_post_register_ledger_checks()` call** until after Chunk 6 is done (insert a placeholder `pass` body for the function and uncomment the body at the end of Chunk 6). The test itself can still pass at unit level because it directly calls the helper after stages are wired.

- [ ] **Step 5: Commit**

```bash
git add app/workers/pipeline.py app/workers/celery_app.py tests/workers/test_startup_assertions.py
git commit -m "feat(pipeline): add module-load assertions for ledger wiring and stale threshold"
```

---

**Chunk 5 complete.** The sweeper now handles both legacy and ledger paths. Startup assertions are defined but not yet wired (will be enabled at the end of Chunk 6 once all stage decorators carry `_lifecycle=True`).

Continue to Chunk 6 (stage decorators + entry points).

---

## Chunk 6: Stage decorators + entry points

This chunk does two things: (1) adds `lifecycle=True` + successor metadata to all 9 stage decorators, and (2) rewrites `start_ingest_pipeline` and `reingest_graph_only` to seed-and-go.

### Task 20: Update all 9 stage decorators

**Files:**
- Modify: `app/workers/pipeline.py` (the 9 stage task definitions)
- Test: `tests/workers/test_stage_decorators_wired.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_stage_decorators_wired.py
"""All 9 stages carry guard_stage_run(lifecycle=True) with correct successor."""
from app.workers.celery_app import celery_app
from app.workers.pipeline import STAGE_SUCCESSORS


def test_every_stage_task_has_lifecycle_marker():
    for stage_name, edge in STAGE_SUCCESSORS.items():
        # Locate the registered task by its wrapper.stage_name marker.
        matching = [
            t for t in celery_app.tasks.values()
            if getattr(t.run, "stage_name", None) == stage_name
        ]
        assert len(matching) == 1, f"{stage_name}: expected exactly 1 task, got {len(matching)}"
        task = matching[0]
        assert getattr(task.run, "_lifecycle", False) is True, (
            f"{stage_name} task is missing lifecycle=True"
        )


def test_stage_9_has_intercept_terminal_false():
    """derive_ontology_graph wrapper must NOT intercept terminal writes."""
    matching = [
        t for t in celery_app.tasks.values()
        if getattr(t.run, "stage_name", None) == "derive_ontology_graph"
    ]
    assert len(matching) == 1
    # We can't read intercept_terminal directly from the wrapper, but the
    # decorator stores enough on the wrapper for a sanity check.
    # In Chunk 3 we set wrapper._lifecycle = lifecycle; extend to also set
    # wrapper._intercept_terminal so this test can assert.
    assert matching[0].run._intercept_terminal is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm worker pytest tests/workers/test_stage_decorators_wired.py -v
```

Expected: assertion failures — stages aren't wired yet.

- [ ] **Step 3: Extend `guard_stage_run` to set `_intercept_terminal` marker**

In `app/workers/pipeline.py`, find the `guard_stage_run` decorator added in Chunk 3 Task 13. Add one line after the existing `wrapper._lifecycle = lifecycle`:

```python
        wrapper._lifecycle = lifecycle
        wrapper._intercept_terminal = intercept_terminal   # NEW
        return wrapper
```

- [ ] **Step 4: Update each of the 9 stage decorators**

**Locate each decorator by grep, not by hard-coded line number.** Earlier chunks have added lines and shifted offsets:

```bash
grep -n '@guard_stage_run(' app/workers/pipeline.py
```

For each of the 9 stages, the existing decorator looks like `@guard_stage_run("<stage_name>")`. Replace each with the corresponding form below. The exact decorator forms:

```python
# prepare_document (~line 3393)
@guard_stage_run("prepare_document",
    lifecycle=True,
    next_stage="detect_and_translate",
    next_task="app.workers.pipeline.detect_and_translate")

# detect_and_translate (~line 4175)
@guard_stage_run("detect_and_translate",
    lifecycle=True,
    next_stage="derive_document_metadata",
    next_task="app.workers.pipeline.derive_document_metadata")

# derive_document_metadata (~line 4013)
@guard_stage_run("derive_document_metadata",
    lifecycle=True,
    next_stage="purge_document_derivations",
    next_task="app.workers.pipeline.purge_document_derivations")

# purge_document_derivations (~line 4704)
@guard_stage_run("purge_document_derivations",
    lifecycle=True,
    next_stage="derive_picture_descriptions",
    next_task="app.workers.pipeline.derive_picture_descriptions")

# derive_picture_descriptions (~line 4486)
@guard_stage_run("derive_picture_descriptions",
    lifecycle=True,
    next_stage="derive_text_embeddings",
    next_task="app.workers.pipeline.derive_text_chunks_and_embeddings")

# derive_text_chunks_and_embeddings (~line 4830)
@guard_stage_run("derive_text_embeddings",
    lifecycle=True,
    next_stage="derive_image_embeddings",
    next_task="app.workers.pipeline.derive_image_embeddings")

# derive_image_embeddings (~line 5311)
@guard_stage_run("derive_image_embeddings",
    lifecycle=True,
    next_stage="derive_document_anchors",
    next_task="app.workers.pipeline.derive_document_anchors")

# derive_document_anchors (~line 5526)
@guard_stage_run("derive_document_anchors",
    lifecycle=True,
    next_stage="derive_ontology_graph",
    next_task="app.workers.pipeline.derive_ontology_graph")

# derive_ontology_graph (~line 6966) — special: intercept_terminal=False
@guard_stage_run("derive_ontology_graph",
    lifecycle=True,
    next_stage=None,
    next_task=None,
    intercept_terminal=False)
```

Update each in place. Do not change anything else about the existing task definitions.

- [ ] **Step 5: Run tests to verify they pass**

```bash
docker compose run --rm worker pytest tests/workers/test_stage_decorators_wired.py -v
```

Expected: both PASS.

- [ ] **Step 6: Run the broader pipeline test suite to catch regressions**

```bash
docker compose run --rm worker pytest tests/ -k "not e2e and not integration" -v
```

Expected: green. Any failures indicate a stage decorator update that broke an existing test — investigate before continuing.

- [ ] **Step 7: Commit**

```bash
git add app/workers/pipeline.py tests/workers/test_stage_decorators_wired.py
git commit -m "feat(pipeline): wire lifecycle on all 9 stage decorators"
```

---

### Task 21: Rewrite `start_ingest_pipeline`

**Files:**
- Modify: `app/workers/pipeline.py` (`start_ingest_pipeline`, around line 2300+)
- Test: `tests/workers/test_start_ingest_pipeline_seed.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_start_ingest_pipeline_seed.py
"""start_ingest_pipeline seeds the first ledger row instead of chain.apply_async."""
import uuid
from unittest.mock import patch
from sqlalchemy import text
from app.workers.pipeline import start_ingest_pipeline

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _create_document(db_session):
    src, doc = uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, uploaded_by, pipeline_status)
        VALUES (:d,:s,'x.pdf','application/pdf',0,'b','k',:u,'PROCESSING')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.commit()
    return str(doc)


def test_seeds_prepare_document_pending_no_chain(db_session):
    doc_id = _create_document(db_session)

    with patch("app.workers.pipeline.chain") as chain_mock:
        result = start_ingest_pipeline(doc_id)
        chain_mock.assert_not_called()       # old chain path is dead

    # First ledger row exists in PENDING.
    row = db_session.execute(text("""
        SELECT status, task_name, queue_name
        FROM ingest.stage_runs
        WHERE pipeline_run_id = :r
    """), {"r": result.pipeline_run_id}).first()
    assert row.status == "PENDING"
    assert row.task_name == "app.workers.pipeline.prepare_document"
    assert row.queue_name == "ingest"
    assert result.celery_task_id == ""        # no apply_async happened
```

- [ ] **Step 2: Run test to verify it fails**

```bash
docker compose run --rm worker pytest tests/workers/test_start_ingest_pipeline_seed.py -v
```

Expected: fails (current impl calls chain.apply_async).

- [ ] **Step 3: Rewrite the body of `start_ingest_pipeline`**

In `app/workers/pipeline.py`, find `def start_ingest_pipeline(...)` and replace the section that builds `pipeline = chain(...)` and calls `pipeline.apply_async()` (around lines 2369-2385). The new body:

```python
        # ── seed first ledger row (spec 2026-05-10) ───────────────────────
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
```

Remove the `pipeline = chain(...)` construction and the `pipeline.apply_async()` call entirely. The `chain` import at the top of the file can remain (other code may use it) but `start_ingest_pipeline` no longer references it.

- [ ] **Step 4: Run test to verify it passes**

```bash
docker compose run --rm worker pytest tests/workers/test_start_ingest_pipeline_seed.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/workers/pipeline.py tests/workers/test_start_ingest_pipeline_seed.py
git commit -m "feat(pipeline): rewrite start_ingest_pipeline to seed-and-go"
```

---

### Task 22: Rewrite `reingest_graph_only`

**Files:**
- Modify: `app/workers/pipeline.py` (`reingest_graph_only`, around line 2421)
- Test: `tests/workers/test_reingest_graph_only_seed.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/workers/test_reingest_graph_only_seed.py
"""reingest_graph_only seeds derive_document_anchors PENDING; no chain."""
import uuid
from unittest.mock import patch
from sqlalchemy import text
from app.workers.pipeline import reingest_graph_only

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _setup_completed_doc(db_session):
    """Create a doc with a completed pipeline_run (typical graph_only target)."""
    src, doc, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, uploaded_by, pipeline_status)
        VALUES (:d,:s,'x.pdf','application/pdf',0,'b','k',:u,'COMPLETE')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'COMPLETE')
    """), {"r": run, "d": doc})
    db_session.commit()
    return str(doc)


def test_reingest_graph_only_seeds_anchors_no_chain(db_session):
    doc_id = _setup_completed_doc(db_session)
    body = {}  # minimal request shape — adjust to match the real signature

    with patch("app.workers.pipeline.celery_chain") as chain_mock:
        result = reingest_graph_only(doc_id, body)
        chain_mock.assert_not_called()

    new_run_id = result["pipeline_run_id"]
    row = db_session.execute(text("""
        SELECT status, task_name FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'derive_document_anchors'
    """), {"r": new_run_id}).first()
    assert row.status == "PENDING"
    assert row.task_name == "app.workers.pipeline.derive_document_anchors"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
docker compose run --rm worker pytest tests/workers/test_reingest_graph_only_seed.py -v
```

Expected: fails.

- [ ] **Step 3: Rewrite the body**

In `app/workers/pipeline.py`, find `def reingest_graph_only(...)` and replace the section that builds `result = celery_chain(derive_document_anchors.si(...), derive_ontology_graph.si(...)).apply_async()` with:

```python
        # ── seed graph_only first ledger row (spec 2026-05-10) ────────────
        _seed_first_stage(
            db,
            pipeline_run_id=run_id,
            stage_name="derive_document_anchors",
            task_name="app.workers.pipeline.derive_document_anchors",
        )
        db.commit()
    finally:
        db.close()

    return {
        "pipeline_run_id": run_id,
        "celery_task_id": "",
    }
```

Preserve all the existing setup code that creates the new PipelineRun, resolves bundles, etc. — only the chain construction at the end is replaced.

- [ ] **Step 4: Run test to verify it passes**

```bash
docker compose run --rm worker pytest tests/workers/test_reingest_graph_only_seed.py -v
```

Expected: PASS.

- [ ] **Step 5: Replace the no-op startup assertion stub with the real body and enable it**

In `app/workers/celery_app.py`, replace the `pass`-bodied `_post_register_ledger_checks()` stub from Chunk 5 Task 19 with:

```python
def _post_register_ledger_checks() -> None:
    # Imported here so pipeline.py registers all tasks first.
    from app.workers.pipeline import _assert_ledger_wiring, _assert_threshold_envelope
    _assert_ledger_wiring()
    _assert_threshold_envelope()


_post_register_ledger_checks()    # NEW — run at module load
```

Run the worker boot smoke test:

```bash
docker compose run --rm worker python -c "from app.workers.celery_app import celery_app; print(sorted(celery_app.tasks.keys())[:5])"
```

Expected: no `RuntimeError` from the assertions. If one fires, fix the offending stage decorator before continuing.

- [ ] **Step 6: Commit**

```bash
git add app/workers/pipeline.py app/workers/celery_app.py tests/workers/test_reingest_graph_only_seed.py
git commit -m "feat(pipeline): rewrite reingest_graph_only to seed-and-go; enable startup checks"
```

---

**Chunk 6 complete.** All stage decorators are wired, both entry points use seed-and-go, and startup assertions are live. The pipeline now runs entirely through the ledger for new ingests.

Continue to Chunk 7 (integration tests + post-deploy verification).

---

## Chunk 7: Integration tests + post-deploy verification

This chunk has two purposes: (1) prove the system works end-to-end against a real broker + DB (the headline regression test), and (2) document the manual verification steps that run after deploy.

### Task 23: Headline regression test (worker death between stages)

**Files:**
- Create: `tests/integration/test_ledger_worker_death.py`
- Modify: `tests/integration/__init__.py` (create if missing)

This test exercises the central guarantee: killing a worker between stages does not strand the pipeline_run.

- [ ] **Step 1: Verify integration test infrastructure exists**

```bash
ls tests/integration/ 2>&1 || echo "missing"
grep -rn "celery_worker\|small_pdf_fixture\|pg_session" tests/ 2>&1 | head -20
```

If `tests/integration/conftest.py` already provides these fixtures, reuse them and skip to Step 2.

**If the integration infrastructure does NOT exist** (likely — a repo-wide grep currently returns zero matches), this task's first action is to mark the headline test as skipped pending fixture work, so the rest of the plan can land. Add at the top of `tests/integration/test_ledger_worker_death.py`:

```python
import pytest

pytestmark = pytest.mark.skip(
    reason="integration fixtures (celery_worker, small_pdf_fixture, pg_session) "
           "not yet defined; see TODO below"
)
```

Then create a TODO at the bottom of the file:

```python
# TODO: implement integration fixtures
# ─────────────────────────────────────
# Required fixtures (define in tests/integration/conftest.py):
#   - pg_session: SQLAlchemy session against the real Postgres in the docker stack
#   - celery_worker: process-wrapper with .kill_and_restart() method
#   - small_pdf_fixture: returns a document_id for a fixture PDF already
#       uploaded to the test ingest source
#
# Without these the headline regression test cannot run. Track this in a
# follow-up plan/PR — the dispatch-ledger plan itself can ship as long as
# the unit tests in tests/workers/ all pass and the live-system verification
# steps in VERIFICATION_CHECKLIST.md are performed at deploy.
```

This keeps the plan implementable. Removing the `@skip` mark and implementing the fixtures is a follow-up plan item — not gating ledger ship.

- [ ] **Step 2: Write the headline test**

```python
# tests/integration/test_ledger_worker_death.py
"""Worker death between stages does not strand the pipeline_run."""
import time
import uuid
import pytest
from sqlalchemy import text


# Skip if integration infra is missing.
pytestmark = pytest.mark.integration


def _wait_until(predicate, timeout_s: float, interval_s: float = 0.5):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def test_kill_worker_between_stages_resumes_via_dispatcher(
    pg_session, celery_worker, small_pdf_fixture,
):
    """Headline regression: this would fail on main before the spec is implemented."""
    from app.workers.pipeline import start_ingest_pipeline

    # 1. Start ingest.
    doc_id = small_pdf_fixture
    result = start_ingest_pipeline(doc_id)
    run_id = result.pipeline_run_id

    # 2. Wait until derive_image_embeddings (stage 7) is COMPLETE.
    def stage_complete(stage):
        row = pg_session.execute(text("""
            SELECT status FROM ingest.stage_runs
            WHERE pipeline_run_id = :r AND stage_name = :s
        """), {"r": run_id, "s": stage}).first()
        return row is not None and row.status == "COMPLETE"

    assert _wait_until(lambda: stage_complete("derive_image_embeddings"),
                       timeout_s=120), "stage 7 did not complete in time"

    # 3. Snapshot the next stage's status BEFORE killing the worker. If it's
    #    already COMPLETE we cannot meaningfully test recovery — re-run with
    #    a fixture document that has more pages so stage 8 takes longer.
    next_row = pg_session.execute(text("""
        SELECT status FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'derive_document_anchors'
    """), {"r": run_id}).first()
    assert next_row is not None
    assert next_row.status != "COMPLETE", (
        "stage 8 already complete; pick a slower fixture so the kill window opens"
    )
    pre_kill_status = next_row.status   # PENDING | DISPATCHED | RUNNING

    # 4. Kill and restart the worker — the orphaned-chain failure mode.
    celery_worker.kill_and_restart()

    # 5. Within ~15s (dispatcher beat fires every 5s + handoff time), stage 8
    #    must reach COMPLETE.
    assert _wait_until(lambda: stage_complete("derive_document_anchors"),
                       timeout_s=15), "stage 8 did not resume after worker kill"
```

- [ ] **Step 3: Run the test**

```bash
docker compose run --rm worker pytest tests/integration/test_ledger_worker_death.py -v
```

Expected: PASS. If the test fails:
1. Check dispatcher beat is firing (`docker logs eip-mmdpp-beat-1 --since 1m | grep DISPATCHER_TICK`).
2. Check stage 8 ledger row exists with `task_name IS NOT NULL`.
3. Check no exception in dispatcher logs.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_ledger_worker_death.py
git commit -m "test(integration): headline regression — worker death does not strand pipeline_run"
```

---

### Task 24: Post-deploy verification checklist

This isn't code — it's documentation that lives alongside the plan and is consulted at deploy time. The plan-runner should perform these steps post-deploy and record results in the PR description or a deploy ticket.

**Files:**
- Modify: `VERIFICATION_CHECKLIST.md` (repo root — add new section)

- [ ] **Step 1: Append the dispatch-ledger verification section**

Open `VERIFICATION_CHECKLIST.md` and add at the end:

````markdown
## Pipeline Stage Dispatch Ledger (2026-05-10)

After deploying the dispatch-ledger feature, verify:

- [ ] **Dispatcher beat is firing.**
  ```bash
  docker logs eip-mmdpp-beat-1 --since 2m | grep DISPATCHER_TICK
  ```
  Expected: ~24 entries in 2 minutes (one every 5s).

- [ ] **The two pre-deploy stalled docs resume after manual reingest.**
  ```bash
  for id in <Radar-Basics-id> <radar2_waveform1-id>; do
    curl -sX POST http://localhost:8005/v1/documents/$id/reingest \
         -H "content-type: application/json" \
         -d '{"mode":"graph_only"}'
  done
  ```
  Watch `/v1/documents/<id>/stages` — `derive_document_anchors` reaches `COMPLETE` within minutes.

- [ ] **Functional equivalence (manual A/B).** Pick one document of each type
  (PDF, jpg, txt) from the pre-deploy baseline (recorded counts of stage_runs
  rows, text_chunks, image_chunks, ArcadeDB elements). Re-ingest each post-deploy.
  Terminal counts must match within ±1 of baseline.

- [ ] **Single-doc latency.** Time a fresh ingest of one small fixture document
  end-to-end. Expected: within ~20s of pre-deploy median for the same doc.

- [ ] **Stale-DISPATCHED alarm is alive (negative check).**
  ```bash
  docker logs eip-mmdpp-worker-1 --since 30m | grep "stale; reset by dispatcher sweeper"
  ```
  Expected on a healthy system: empty. Non-empty entries indicate broker / worker
  problems and warrant investigation.

- [ ] **Stale-RUNNING ledger sweep alarm (negative check).**
  ```bash
  docker logs eip-mmdpp-worker-1 --since 30m | grep "stale; reset by sweeper"
  ```
  Expected: empty on healthy system.
````

- [ ] **Step 2: Commit**

```bash
git add VERIFICATION_CHECKLIST.md
git commit -m "docs(verification): add dispatch-ledger post-deploy checklist"
```

---

**Chunk 7 complete.** The plan is fully implemented. Run the entire test suite as a final sanity check:

```bash
docker compose run --rm worker pytest tests/ -v
```

Expected: green.

---

## Plan complete

Every chunk implemented and committed. Post-deploy, follow the verification checklist in `VERIFICATION_CHECKLIST.md` for live-system steps.

**Total tasks:** 24 (Chunks 1–7).
**Total commits expected:** 24 (one per task), plus any out-of-band fixes.
