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
