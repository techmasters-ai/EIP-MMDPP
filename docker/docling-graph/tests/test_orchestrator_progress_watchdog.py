"""Host unit test for R1 — orchestrator no-progress (inter-arrival) watchdog.

The watchdog helper ``_drain_futures_with_progress_watchdog`` is added to
``docling_graph/core/extractors/contracts/delta/orchestrator.py`` exclusively by
``docker/docling-graph/patches/0001-orchestrator-batch-hard-timeout.patch`` (it
is NOT in the vendored base, and the base module's relative imports
``from ...gleaning import`` prevent a plain import on the host). So this test
extracts the helper's source *directly from the patch file* (the `+` lines that
define the function), execs just that function in an isolated namespace, and
exercises its inter-arrival semantics with a real ThreadPoolExecutor and short
sleeps. This pins the actual shipped patch behavior, host-side, with no
docling_graph dependency.

R1 contract under test (replaces the buggy total-elapsed BATCH_HARD ceiling):
  * steady completions (each within the window)      -> all drained, no stall
  * one slow-but-progressing future (gap < window)   -> all drained, no stall
  * total stall (no completion within a full window) -> trips after ~1 window,
                                                        returns pending as hung
  * partial progress then stall                       -> drains the fast ones,
                                                        then trips on the gap
"""
import ast
import concurrent.futures as cf
import time
from pathlib import Path

_PATCH_PATH = (
    Path(__file__).resolve().parent.parent
    / "patches"
    / "0001-orchestrator-batch-hard-timeout.patch"
)
_FUNC_NAME = "_drain_futures_with_progress_watchdog"


def _extract_helper_from_patch():
    """Pull the helper's added source out of the unified diff and exec it.

    We collect the contiguous run of `+` (added) lines that defines the helper,
    strip the leading `+`, dedent, AST-parse to confirm it's exactly that one
    function, then exec it in a fresh namespace.
    """
    added_lines = []
    capturing = False
    for raw in _PATCH_PATH.read_text().splitlines():
        # Only consider added lines inside a hunk; skip the +++ file header.
        if raw.startswith("+++"):
            continue
        if raw.startswith("+"):
            body = raw[1:]
            if not capturing:
                if body.startswith(f"def {_FUNC_NAME}("):
                    capturing = True
                    added_lines.append(body)
                continue
            added_lines.append(body)
        else:
            if capturing:
                # First non-added line after the function block ends capture.
                break
    assert added_lines, f"{_FUNC_NAME} definition not found in {_PATCH_PATH}"
    src = "\n".join(added_lines).rstrip()
    tree = ast.parse(src)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert len(funcs) == 1 and funcs[0].name == _FUNC_NAME, (
        f"expected exactly one function {_FUNC_NAME}, got "
        f"{[getattr(n, 'name', type(n).__name__) for n in tree.body]}"
    )
    ns: dict = {}
    exec(compile(tree, str(_PATCH_PATH), "exec"), ns)
    return ns[_FUNC_NAME]


_drain = _extract_helper_from_patch()


def _run(sleeps, no_progress_seconds):
    """Submit one job per sleep duration; drain via the watchdog.

    Returns (completed_results, hung_idxs, drain_seconds).
      completed_results = list of (original_idx, returned_value)
      drain_seconds = wall time for the watchdog to drain-or-trip, measured
        BEFORE executor teardown so it reflects the detector, not
        ThreadPoolExecutor.shutdown(wait=True) joining still-running cancelled
        threads (cancel() cannot stop an already-started future).
    """
    results = []
    with cf.ThreadPoolExecutor(max_workers=len(sleeps)) as pool:

        def job(idx, secs):
            time.sleep(secs)
            return idx * 10  # distinguishable payload

        futures = {pool.submit(job, i, s): i for i, s in enumerate(sleeps)}
        t0 = time.time()
        gen = _drain(futures, no_progress_seconds)
        hung = set()
        while True:
            try:
                _future, idx, value = next(gen)
            except StopIteration as stop:
                hung = stop.value or set()
                break
            results.append((idx, value))
        drain_seconds = time.time() - t0
        hung_idxs = sorted(futures[f] for f in hung)
        for f in hung:
            f.cancel()
    return results, hung_idxs, drain_seconds


def test_steady_completions_no_stall():
    results, hung, drain_s = _run([0.05, 0.10, 0.15, 0.20], no_progress_seconds=0.5)
    assert sorted(i for i, _ in results) == [0, 1, 2, 3]
    assert dict(results) == {0: 0, 1: 10, 2: 20, 3: 30}
    assert hung == []
    assert drain_s < 1.0  # finished well within a single window after the last gap


def test_one_slow_but_progressing_no_stall():
    # Inter-arrival gaps (0.10, +0.10, +0.10, +0.25) each stay under the 0.5s
    # window even though the last future is "slow" in absolute terms.
    results, hung, _ = _run([0.10, 0.20, 0.30, 0.55], no_progress_seconds=0.5)
    assert sorted(i for i, _ in results) == [0, 1, 2, 3]
    assert hung == []


def test_total_stall_trips_after_one_window():
    results, hung, drain_s = _run([5.0, 5.0, 5.0], no_progress_seconds=0.5)
    assert results == []
    assert hung == [0, 1, 2]
    # Tripped on the inter-arrival window, NOT after the 5s sleeps.
    assert drain_s < 2.0, f"expected stall ~one window, took {drain_s:.2f}s"


def test_partial_progress_then_stall():
    results, hung, drain_s = _run([0.10, 0.20, 5.0, 5.0], no_progress_seconds=0.5)
    assert sorted(i for i, _ in results) == [0, 1]
    assert hung == [2, 3]
    assert drain_s < 2.0, f"expected stall after draining 2, took {drain_s:.2f}s"


# ---------------------------------------------------------------------------
# R2 — progress-registry hooks fed FROM the drain loop.
#
# The patch wraps the drain loop with registry calls:
#   _pr_safe(_pr_start, len(futures))   before the drain
#   _pr_safe(_pr_advance)               on each completed batch
#   _pr_safe(_pr_touch, "merging")      in a finally, then
#   _pr_safe(_pr_clear)                 (always — normal drain, stall, or raise)
# where _pr_* resolve from a guarded `from app...` import (None stubs if app is
# not importable) and _pr_safe is a best-effort wrapper that no-ops unless both
# run_id and pass_name are set, swallowing any exception.
#
# `app` is not importable in this standalone lib test, so we (a) reproduce the
# exact shipped loop+guard+_pr_safe contract here and exercise it against fake
# registry hooks to prove the call sequence, AND (b) source-inspect the patch
# text to pin that the real shipped lines call the hooks in the right places.
# ---------------------------------------------------------------------------


def _patched_drain_with_registry(sleeps, no_progress_seconds, registry, run_id, pass_name):
    """Reproduce the shipped R2 drain loop (guard + _pr_safe + try/finally).

    `registry` is either a module-like object exposing start/advance/touch/clear
    that record calls, or None to simulate the import-guard fallback (app not
    importable -> _pr_* are None stubs). Returns (completed_idxs, hung_idxs).
    This is byte-for-byte the control flow the patch inserts; the registry-call
    surface (the thing under test) is identical.
    """
    # --- guarded import + _pr_safe shim (mirrors the patched lines exactly) ---
    if registry is not None:
        _pr_start, _pr_advance = registry.start, registry.advance
        _pr_touch, _pr_clear = registry.touch, registry.clear
        _pr_run, _pr_pass = run_id, pass_name
    else:  # standalone lib test / app not importable
        _pr_start = _pr_advance = _pr_touch = _pr_clear = None
        _pr_run = _pr_pass = None

    def _pr_safe(_fn, *_a):
        if _fn is not None and _pr_run and _pr_pass:
            try:
                _fn(_pr_run, _pr_pass, *_a)
            except Exception:
                pass

    completed = []
    with cf.ThreadPoolExecutor(max_workers=len(sleeps)) as pool:

        def job(idx, secs):
            time.sleep(secs)
            return idx * 10

        futures = {pool.submit(job, i, s): i for i, s in enumerate(sleeps)}
        _pr_safe(_pr_start, len(futures))
        gen = _drain(futures, no_progress_seconds)
        hung = set()
        try:
            while True:
                try:
                    _future, idx, _value = next(gen)
                except StopIteration as stop:
                    hung = stop.value or set()
                    break
                _pr_safe(_pr_advance)
                completed.append(idx)
        finally:
            _pr_safe(_pr_touch, "merging")
            _pr_safe(_pr_clear)
        hung_idxs = sorted(futures[f] for f in hung)
        for f in hung:
            f.cancel()
    return sorted(completed), hung_idxs


class _RecordingRegistry:
    """Fake app._progress_registry recording (fn, run_id, pass_name, *args)."""

    def __init__(self):
        self.calls = []

    def start(self, run_id, pass_name, total):
        self.calls.append(("start", run_id, pass_name, total))

    def advance(self, run_id, pass_name):
        self.calls.append(("advance", run_id, pass_name))

    def touch(self, run_id, pass_name, phase):
        self.calls.append(("touch", run_id, pass_name, phase))

    def clear(self, run_id, pass_name):
        self.calls.append(("clear", run_id, pass_name))


def test_registry_advance_once_per_batch_and_clear_once():
    reg = _RecordingRegistry()
    completed, hung = _patched_drain_with_registry(
        [0.05, 0.10, 0.15, 0.20], no_progress_seconds=0.5,
        registry=reg, run_id="r", pass_name="p",
    )
    assert completed == [0, 1, 2, 3]
    assert hung == []
    kinds = [c[0] for c in reg.calls]
    # start exactly once, with total == number of futures
    assert kinds.count("start") == 1
    assert reg.calls[0] == ("start", "r", "p", 4)
    # advance exactly once per completed batch
    assert kinds.count("advance") == 4
    # touch("merging") then clear, exactly once each, in the finally
    assert kinds.count("touch") == 1
    assert kinds.count("clear") == 1
    assert ("touch", "r", "p", "merging") in reg.calls
    assert ("clear", "r", "p") in reg.calls
    # clear is the last recorded call (finally runs after the drain)
    assert reg.calls[-1] == ("clear", "r", "p")
    # every recorded call carried the thread-local run_id/pass_name
    assert all(c[1] == "r" and c[2] == "p" for c in reg.calls)


def test_registry_clear_still_runs_on_stall():
    # A total stall drains nothing, returns the futures as hung; the finally
    # must STILL advance through touch->clear so no stale entry is left behind.
    reg = _RecordingRegistry()
    completed, hung = _patched_drain_with_registry(
        [5.0, 5.0, 5.0], no_progress_seconds=0.5,
        registry=reg, run_id="r", pass_name="p",
    )
    assert completed == []
    assert hung == [0, 1, 2]
    kinds = [c[0] for c in reg.calls]
    assert kinds.count("start") == 1
    assert kinds.count("advance") == 0  # nothing completed
    assert kinds.count("touch") == 1
    assert kinds.count("clear") == 1
    assert reg.calls[-1] == ("clear", "r", "p")


def test_import_guard_no_app_still_drains():
    # registry=None simulates the guarded `from app...` import failing (the
    # standalone-lib / app-not-importable path): _pr_* become None stubs and the
    # loop must still drain every batch without raising.
    completed, hung = _patched_drain_with_registry(
        [0.05, 0.10, 0.15], no_progress_seconds=0.5,
        registry=None, run_id=None, pass_name=None,
    )
    assert completed == [0, 1, 2]
    assert hung == []


def test_pr_safe_noop_when_run_or_pass_missing():
    # Even with a live registry, _pr_safe must no-op when run_id/pass_name are
    # unset (the getattr(..., None) guard) so a pass with no thread-local bridge
    # never touches the registry.
    reg = _RecordingRegistry()
    completed, _ = _patched_drain_with_registry(
        [0.05, 0.10], no_progress_seconds=0.5,
        registry=reg, run_id=None, pass_name=None,
    )
    assert completed == [0, 1]
    assert reg.calls == []  # nothing recorded -> registry untouched


def test_patch_text_wires_hooks_in_right_places():
    """Pin the actual shipped patch lines (source inspection, build-safety)."""
    added = []
    for raw in _PATCH_PATH.read_text().splitlines():
        if raw.startswith("+") and not raw.startswith("+++"):
            added.append(raw[1:])
    blob = "\n".join(added)
    # guarded import + None-stub fallback
    assert "from app._progress_registry import (" in blob
    assert "from app.main import _pass_progress_overrides as _pr_ovr" in blob
    assert "except Exception:  # standalone lib test / app not importable" in blob
    assert "_pr_start = _pr_advance = _pr_touch = _pr_clear = None" in blob
    # best-effort shim guarded on run_id/pass_name
    assert "def _pr_safe(_fn, *_a):" in blob
    assert "if _fn is not None and _pr_run and _pr_pass:" in blob
    # the four hook calls
    assert "_pr_safe(_pr_start, len(futures))" in blob
    assert "_pr_safe(_pr_advance)" in blob
    assert '_pr_safe(_pr_touch, "merging")' in blob
    assert "_pr_safe(_pr_clear)" in blob
    # start before drain; advance after the next(_drainer) consume; clear in a
    # finally (always runs). Check relative ordering in the added text.
    i_start = blob.index("_pr_safe(_pr_start, len(futures))")
    i_drainer = blob.index("_drainer = _drain_futures_with_progress_watchdog")
    i_next = blob.index("future, original_batch_idx, result = next(_drainer)")
    i_advance = blob.index("_pr_safe(_pr_advance)")
    i_finally = blob.index("finally:")
    i_touch = blob.index('_pr_safe(_pr_touch, "merging")')
    i_clear = blob.index("_pr_safe(_pr_clear)")
    assert i_start < i_drainer < i_next < i_advance, "start must precede drain; advance after consume"
    assert i_finally < i_touch < i_clear, "touch+clear must live in the finally"


if __name__ == "__main__":  # allow `python test_...py` outside pytest
    test_steady_completions_no_stall()
    test_one_slow_but_progressing_no_stall()
    test_total_stall_trips_after_one_window()
    test_partial_progress_then_stall()
    test_registry_advance_once_per_batch_and_clear_once()
    test_registry_clear_still_runs_on_stall()
    test_import_guard_no_app_still_drains()
    test_pr_safe_noop_when_run_or_pass_missing()
    test_patch_text_wires_hooks_in_right_places()
    print("ALL R1+R2 watchdog tests passed")
