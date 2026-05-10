"""_finalize_after_body decides Tx-3 vs Tx-4 from body return value."""
from unittest.mock import patch
from app.workers._stage_lifecycle import _LifecycleCtx
from app.workers.pipeline import _finalize_after_body


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
    with patch("app.workers.pipeline._tx3_complete_and_enqueue_next") as tx3, \
         patch("app.workers.pipeline._tx4_finalize_failure") as tx4:
        _finalize_after_body(
            _ctx(pending_status="FAILED", pending_error="bad data"),
            result={"stage": "prepare_document", "status": "COMPLETE"},
        )
        tx4.assert_called_once()
        tx3.assert_not_called()


def test_return_dict_status_failed_triggers_tx4():
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
    with patch("app.workers.pipeline._tx3_complete_and_enqueue_next") as tx3, \
         patch("app.workers.pipeline._tx4_finalize_failure") as tx4:
        _finalize_after_body(_ctx(), result={"stage": "prepare_document",
                                              "status": "complete",
                                              "elements": 124})
        tx3.assert_called_once()
        tx4.assert_not_called()


def test_non_dict_return_advances_pipeline():
    with patch("app.workers.pipeline._tx3_complete_and_enqueue_next") as tx3:
        _finalize_after_body(_ctx(), result=None)
        tx3.assert_called_once()
