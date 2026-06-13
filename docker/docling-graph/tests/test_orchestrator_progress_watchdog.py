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


if __name__ == "__main__":  # allow `python test_...py` outside pytest
    test_steady_completions_no_stall()
    test_one_slow_but_progressing_no_stall()
    test_total_stall_trips_after_one_window()
    test_partial_progress_then_stall()
    print("ALL R1 watchdog tests passed")
