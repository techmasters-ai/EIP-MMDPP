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
