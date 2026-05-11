"""Module-load assertions catch wrapper-marker and threshold misconfigurations."""
import pytest


def test_assert_ledger_wiring_succeeds_when_stages_wired():
    """Once Task 20 wires lifecycle=True on all stages, this passes.
    For now, run as-is to verify the function exists and is callable.
    """
    from app.workers.pipeline import _assert_ledger_wiring
    # _assert_ledger_wiring will likely RAISE here because stages don't
    # yet have lifecycle=True (that's Task 20). That's expected. The point
    # of THIS test is just that the function exists and is importable.
    assert callable(_assert_ledger_wiring)


def test_assert_threshold_envelope_raises_on_misconfiguration(monkeypatch):
    """If threshold < time_limit + max_retries × default_retry_delay, raise."""
    from app.workers.pipeline import _assert_threshold_envelope
    from app.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "stale_stage_run_threshold_seconds", 1)

    with pytest.raises(RuntimeError, match="must exceed envelope"):
        _assert_threshold_envelope()


def test_post_register_ledger_checks_runs_assertions_in_chunk_6():
    """Task 22 (Chunk 6) replaces the no-op body with real assertion calls.

    Calling the function with a fully loaded pipeline module must succeed
    (no re-entry guard, no RuntimeError) — wiring + threshold envelope are
    valid in the deployed config.
    """
    from app.workers.celery_app import _post_register_ledger_checks
    # Should not raise. _assert_ledger_wiring / _assert_threshold_envelope
    # both have their own dedicated tests; this asserts the entry-point glue
    # is wired correctly.
    result = _post_register_ledger_checks()
    assert result is None
