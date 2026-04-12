"""Worker-facing dispatch return types.

Spec §5.2 + Task 3.6: ``start_ingest_pipeline`` returns an
``IngestDispatchResult`` so callers can access both the new
``pipeline_run_id`` and the Celery task id in one call, without
depending on the order they became available in the old code path.

A standalone module keeps both ``app/workers/pipeline.py`` and
``app/api/v1/sources.py`` able to import it without creating a
circular dependency between workers and API.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IngestDispatchResult:
    """Return value of :func:`app.workers.pipeline.start_ingest_pipeline`.

    Attributes:
        pipeline_run_id: UUID (as string) of the newly-created
            ``PipelineRun`` row. Stable across retries.
        celery_task_id: The Celery task id returned by the dispatched
            chain's ``apply_async()`` call. Visible in Celery's
            inspector and in the reingest endpoint response.
    """

    pipeline_run_id: str
    celery_task_id: str
