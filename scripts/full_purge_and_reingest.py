#!/usr/bin/env python3
"""Full-corpus purge + re-ingest migration script.

Executes the 12-step migration described in spec §5.1 + §5.2:

  1.  Safety-flag check (``--i-understand-this-deletes-derived-data``).
  2.  Stop worker + beat containers.
  3.  Truncate Postgres tables per §5.1.
  4.  Drop + recreate the ArcadeDB schema.
  5.  Empty MinIO ``derived/`` bucket.
  6.  Redis FLUSHALL.
  7.  alembic upgrade head.
  8.  Restart worker + beat.
  9.  Reset ``ingest.documents`` pipeline_status to PENDING.
  10. Enqueue the full pipeline chain for every document.
  11. Poll until all pipeline_runs reach a terminal status.
  12. Emit ``/tmp/migration-report-{timestamp}.md``.

Usage::

    python scripts/full_purge_and_reingest.py --dry-run
    python scripts/full_purge_and_reingest.py --i-understand-this-deletes-derived-data

``--dry-run`` prints the truncate plan without executing.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


# --- Constants (spec §5.1) --------------------------------------------------

POSTGRES_TRUNCATE_TABLES: list[str] = [
    # Ordered by FK dependency (children first).
    "ingest.stage_runs",
    "ingest.pipeline_runs",
    "ingest.document_graph_extractions",
    "ingest.document_elements",
    "ingest.artifacts",
    "retrieval.chunk_links",
    "retrieval.text_chunks",
    "retrieval.image_chunks",
    "retrieval.chunks_legacy",
    "retrieval.community_runs",
    "governance.feedback",
    "governance.patch_approvals",
    "governance.patch_events",
    "governance.patches",
    "governance.query_profile_registries",
    "governance.trusted_data_submissions",
    "ontology.entity_types",
    "ontology.relationship_types",
    "ontology.versions",
    "ingest.watch_logs",
]

# ingest.documents: rows preserved; pipeline_status reset to PENDING.
DOCUMENTS_RESET_SQL = """
    UPDATE ingest.documents
    SET pipeline_status = 'PENDING',
        pipeline_stage = NULL,
        failed_stages = NULL,
        error_message = NULL,
        celery_task_id = NULL
"""

PRESERVE_TABLES = {
    "ingest.documents",
    "ingest.sources",
    "ingest.watch_dirs",
    "auth.users",
    "auth.user_roles",
    "public.alembic_version",
}

WORKER_CONTAINERS = [
    "eip-mmdpp-worker-1",
    "eip-mmdpp-beat-1",
    "eip-mmdpp-docling-graph-1",
]

POLL_INTERVAL_SECONDS = 30
DEFAULT_POLL_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours


# --- Logging ----------------------------------------------------------------

logger = logging.getLogger("full_purge_and_reingest")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(message)s")
)
logger.addHandler(_handler)


# --- Step helpers -----------------------------------------------------------

def stop_workers(dry_run: bool) -> None:
    """Step 2 — stop worker + beat + docling-graph containers."""
    for container in WORKER_CONTAINERS:
        if dry_run:
            logger.info("[dry-run] would stop container %s", container)
            continue
        try:
            subprocess.run(
                ["docker", "stop", container],
                check=False, timeout=60,
            )
            logger.info("stopped %s", container)
        except Exception as exc:
            logger.warning("stop %s failed (may not be running): %s", container, exc)


def truncate_postgres(dry_run: bool) -> None:
    """Step 3 — truncate all tables per spec §5.1 + reset documents."""
    logger.info("Postgres truncate plan (%d tables):", len(POSTGRES_TRUNCATE_TABLES))
    for t in POSTGRES_TRUNCATE_TABLES:
        logger.info("  TRUNCATE %s", t)
    logger.info("  + UPDATE ingest.documents SET pipeline_status='PENDING' ...")
    logger.info("Preserved: %s", ", ".join(sorted(PRESERVE_TABLES)))

    if dry_run:
        logger.info("[dry-run] skipping postgres truncate")
        return

    from sqlalchemy import create_engine, text
    from app.config import get_settings

    settings = get_settings()
    engine = create_engine(settings.sync_database_url)
    with engine.begin() as conn:
        for table in POSTGRES_TRUNCATE_TABLES:
            try:
                conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
                logger.info("truncated %s", table)
            except Exception as exc:
                logger.warning("truncate %s failed: %s", table, exc)
        conn.execute(text(DOCUMENTS_RESET_SQL))
        logger.info("reset ingest.documents pipeline_status to PENDING")


def reset_arcadedb(dry_run: bool) -> None:
    """Step 4 — drop + recreate ArcadeDB schema via ensure_schema_sync."""
    if dry_run:
        logger.info("[dry-run] would truncate every vertex/edge class + re-run ensure_schema_sync")
        return

    from app.services.arcadedb_client import ArcadeDBClient
    from app.services.arcadedb_schema import ensure_schema_sync
    from app.config import get_settings

    settings = get_settings()
    client = ArcadeDBClient(
        base_url=settings.arcadedb_base_url,
        username=settings.arcadedb_username,
        password=settings.arcadedb_password,
    )
    try:
        # Best-effort: fetch class list, truncate each. Missing classes OK.
        try:
            schema_rows = client.query_sync(
                settings.arcadedb_database, "sql", "SELECT FROM schema:types",
            )
            class_names = [r.get("name") for r in schema_rows if r.get("name")]
        except Exception as exc:
            logger.warning("could not list ArcadeDB classes: %s", exc)
            class_names = []
        for name in class_names:
            if name in ("V", "E") or name.startswith("_"):
                continue
            try:
                client.command_sync(
                    settings.arcadedb_database, "sql",
                    f"TRUNCATE TYPE `{name}`",
                )
                logger.info("truncated ArcadeDB class %s", name)
            except Exception as exc:
                logger.warning("truncate %s failed: %s", name, exc)
        report = ensure_schema_sync(client, settings.arcadedb_database)
        logger.info(
            "ArcadeDB schema re-ensured — types_created=%d properties_added=%d errors=%d",
            report.types_created, report.properties_added, len(report.errors),
        )
    finally:
        client.close_sync()


def empty_minio_derived(dry_run: bool) -> None:
    """Step 5 — empty the MinIO derived bucket."""
    if dry_run:
        logger.info("[dry-run] would empty MinIO derived/ bucket")
        return

    from app.services.storage import get_sync_s3_client
    from app.config import get_settings

    settings = get_settings()
    bucket = settings.minio_bucket_derived
    client = get_sync_s3_client()

    deleted = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        contents = page.get("Contents") or []
        if not contents:
            continue
        keys = [{"Key": obj["Key"]} for obj in contents]
        client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
        deleted += len(keys)
    logger.info("deleted %d objects from MinIO bucket %s", deleted, bucket)


def flush_redis(dry_run: bool) -> None:
    """Step 6 — Redis FLUSHALL."""
    if dry_run:
        logger.info("[dry-run] would FLUSHALL redis")
        return

    import redis
    from app.config import get_settings

    settings = get_settings()
    client = redis.Redis.from_url(settings.celery_broker_url)
    client.flushall()
    logger.info("redis FLUSHALL complete")


def apply_migrations(dry_run: bool) -> None:
    """Step 7 — alembic upgrade head."""
    if dry_run:
        logger.info("[dry-run] would run `alembic upgrade head`")
        return
    subprocess.run(["alembic", "upgrade", "head"], check=True)
    logger.info("alembic migrations applied")


def restart_workers(dry_run: bool) -> None:
    """Step 8 — restart worker + beat + docling-graph containers."""
    for container in WORKER_CONTAINERS:
        if dry_run:
            logger.info("[dry-run] would start container %s", container)
            continue
        try:
            subprocess.run(
                ["docker", "start", container],
                check=True, timeout=60,
            )
            logger.info("started %s", container)
        except Exception as exc:
            logger.error("start %s failed: %s", container, exc)


def reset_document_statuses(dry_run: bool) -> list[str]:
    """Step 9 — return the document_id list after status reset.

    The UPDATE itself already ran in step 3; this step collects the IDs
    to enqueue and confirms the reset took effect.
    """
    if dry_run:
        logger.info(
            "[dry-run] would SELECT id FROM ingest.documents WHERE pipeline_status='PENDING' "
            "then enqueue pipeline for each",
        )
        return []

    from sqlalchemy import create_engine, text
    from app.config import get_settings

    settings = get_settings()
    engine = create_engine(settings.sync_database_url)
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id FROM ingest.documents WHERE pipeline_status = 'PENDING'")
        ).all()
    doc_ids = [str(row[0]) for row in rows]
    logger.info("%d documents pending re-ingest", len(doc_ids))
    return doc_ids


def enqueue_pipeline(document_ids: list[str], dry_run: bool) -> list[str]:
    """Step 10 — enqueue start_ingest_pipeline for each document.

    Returns the list of pipeline_run_ids created (or an empty list in
    dry-run mode).
    """
    if dry_run:
        return []

    from app.workers.pipeline import start_ingest_pipeline

    run_ids: list[str] = []
    for doc_id in document_ids:
        try:
            result = start_ingest_pipeline(doc_id)
            run_ids.append(result.pipeline_run_id)
            logger.info(
                "enqueued document_id=%s pipeline_run_id=%s",
                doc_id, result.pipeline_run_id,
            )
        except Exception as exc:
            logger.error("enqueue failed for %s: %s", doc_id, exc)
    return run_ids


def poll_until_complete(
    document_ids: list[str],
    timeout_seconds: int = DEFAULT_POLL_TIMEOUT_SECONDS,
    poll_interval: int = POLL_INTERVAL_SECONDS,
) -> dict[str, str]:
    """Step 11 — poll ``ingest.documents.pipeline_status`` per document
    until every doc reaches a terminal state (COMPLETE, PARTIAL_COMPLETE,
    FAILED, PENDING_HUMAN_REVIEW) or the timeout elapses. Returns a
    ``{document_id: final_status}`` mapping.
    """
    from sqlalchemy import create_engine, text
    from app.config import get_settings

    settings = get_settings()
    engine = create_engine(settings.sync_database_url)
    terminal = {"COMPLETE", "PARTIAL_COMPLETE", "FAILED", "PENDING_HUMAN_REVIEW"}
    deadline = time.monotonic() + timeout_seconds
    final: dict[str, str] = {}

    while time.monotonic() < deadline:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id::text, pipeline_status FROM ingest.documents "
                    "WHERE id = ANY(:ids)"
                ),
                {"ids": document_ids},
            ).all()
        pending = []
        for doc_id, status in rows:
            if status in terminal:
                final[doc_id] = status
            else:
                pending.append((doc_id, status))
        logger.info(
            "poll: %d/%d complete; %d still PROCESSING",
            len(final), len(document_ids), len(pending),
        )
        if len(final) >= len(document_ids):
            return final
        time.sleep(poll_interval)

    # Timeout path — mark still-pending docs as TIMEOUT.
    for doc_id in document_ids:
        final.setdefault(doc_id, "TIMEOUT")
    return final


def emit_report(
    status_by_doc: dict[str, str],
    path: Path,
    *,
    started_at: _dt.datetime,
    finished_at: _dt.datetime,
) -> None:
    """Step 12 — write the migration report to ``path``."""
    from sqlalchemy import create_engine, text
    from app.config import get_settings

    settings = get_settings()
    engine = create_engine(settings.sync_database_url)

    outcome_counts: dict[str, int] = {}
    for status in status_by_doc.values():
        outcome_counts[status] = outcome_counts.get(status, 0) + 1

    # Collect the latest pipeline_run.metrics per document.
    run_metrics: dict[str, dict[str, Any]] = {}
    with engine.connect() as conn:
        for doc_id in status_by_doc:
            row = conn.execute(
                text(
                    "SELECT metrics FROM ingest.pipeline_runs "
                    "WHERE document_id = :doc_id::uuid "
                    "ORDER BY started_at DESC LIMIT 1"
                ),
                {"doc_id": doc_id},
            ).first()
            run_metrics[doc_id] = row[0] if row and row[0] else {}

    lines: list[str] = []
    lines.append(f"# Migration Report — {started_at.isoformat()}")
    lines.append("")
    lines.append(f"Started:  {started_at.isoformat()}")
    lines.append(f"Finished: {finished_at.isoformat()}")
    lines.append(f"Elapsed:  {(finished_at - started_at)}")
    lines.append(f"Docs:     {len(status_by_doc)}")
    lines.append("")
    lines.append("## Outcome counts")
    for status, count in sorted(outcome_counts.items()):
        lines.append(f"  - {status}: {count}")
    lines.append("")
    lines.append("## Per-document detail")
    for doc_id in sorted(status_by_doc):
        status = status_by_doc[doc_id]
        metrics = run_metrics.get(doc_id, {}) or {}
        quality = metrics.get("extraction_quality", "?")
        sections = metrics.get("section_count", "?")
        chunks = metrics.get("text_chunk_count", "?")
        lines.append(
            f"  - `{doc_id}` → {status} | quality={quality} | "
            f"sections={sections} | chunks={chunks}"
        )
    report_blob = "\n".join(lines) + "\n"
    path.write_text(report_blob)
    logger.info("report written to %s (%d bytes)", path, len(report_blob))


# --- Main -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the truncate plan without executing any destructive step.",
    )
    parser.add_argument(
        "--i-understand-this-deletes-derived-data",
        action="store_true",
        dest="confirm",
        help="Required acknowledgement for real execution.",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=DEFAULT_POLL_TIMEOUT_SECONDS,
        help="Seconds to wait for pipeline completion before declaring TIMEOUT.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Override report output path (default: /tmp/migration-report-{timestamp}.md).",
    )
    args = parser.parse_args(argv)

    if not args.dry_run and not args.confirm:
        logger.error(
            "Real execution requires --i-understand-this-deletes-derived-data. "
            "Use --dry-run to preview the truncate plan.",
        )
        return 2

    started_at = _dt.datetime.now(_dt.timezone.utc)
    logger.info("=== %s ===", "DRY RUN" if args.dry_run else "REAL RUN")
    logger.info("started_at = %s", started_at.isoformat())

    stop_workers(args.dry_run)
    truncate_postgres(args.dry_run)
    reset_arcadedb(args.dry_run)
    empty_minio_derived(args.dry_run)
    flush_redis(args.dry_run)
    apply_migrations(args.dry_run)
    restart_workers(args.dry_run)
    doc_ids = reset_document_statuses(args.dry_run)

    if args.dry_run:
        logger.info("[dry-run] complete — would enqueue %d pipelines next", len(doc_ids))
        return 0

    enqueue_pipeline(doc_ids, args.dry_run)
    status_by_doc = poll_until_complete(doc_ids, timeout_seconds=args.poll_timeout)
    finished_at = _dt.datetime.now(_dt.timezone.utc)

    report_path = args.report_path or Path(
        f"/tmp/migration-report-{started_at.strftime('%Y%m%d-%H%M%S')}.md"
    )
    emit_report(
        status_by_doc, report_path,
        started_at=started_at, finished_at=finished_at,
    )

    failed = [d for d, s in status_by_doc.items() if s in ("FAILED", "TIMEOUT")]
    if failed:
        logger.error("%d doc(s) failed or timed out — see report", len(failed))
        return 1
    logger.info("migration complete — all docs reached terminal state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
