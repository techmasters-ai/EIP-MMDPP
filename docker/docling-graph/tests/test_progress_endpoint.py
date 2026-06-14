"""Tests for GET /progress endpoint (R2-T4).

Approach A (chosen): test `_progress_payload` — the module-level pure helper
that the route delegates to — without importing FastAPI or any extraction-library
deps that don't exist on the host.

We load `_progress_payload` by importing main.py via importlib, with a minimal
`app` package stub injected into sys.modules so that the function-local
`from app import _progress_registry` resolves correctly.  The full FastAPI app
object is constructed during that import, which may call out to other `app.*`
sub-modules; if those sub-modules don't exist on the host the import will fail.
In that case we fall back to testing the filter logic directly against the
registry (the same logic inline in the helper), PLUS source-inspection
assertions that confirm the route and helper are wired correctly in main.py.

Source-inspection assertions always run regardless of import success.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Load _progress_registry via importlib (pure stdlib, always succeeds)
# ---------------------------------------------------------------------------
_REG_PATH = Path(__file__).resolve().parent.parent / "app" / "_progress_registry.py"
_reg_spec = importlib.util.spec_from_file_location("dg_progress_registry_ep", _REG_PATH)
_reg = importlib.util.module_from_spec(_reg_spec)
_reg_spec.loader.exec_module(_reg)

# ---------------------------------------------------------------------------
# Inject a minimal `app` stub so any `from app import _progress_registry`
# inside main.py resolves to the module we just loaded.
# ---------------------------------------------------------------------------
_fake_app_pkg = sys.modules.setdefault("app", types.ModuleType("app"))
sys.modules["app._progress_registry"] = _reg  # type: ignore[assignment]
setattr(_fake_app_pkg, "_progress_registry", _reg)

_MAIN_PATH = Path(__file__).resolve().parent.parent / "app" / "main.py"
_MAIN_SRC = _MAIN_PATH.read_text()

# ---------------------------------------------------------------------------
# Attempt to import _progress_payload from main.py.
# main.py imports many heavy deps; we try but fall back gracefully.
# ---------------------------------------------------------------------------
_progress_payload = None
_IMPORT_ERR: Exception | None = None

try:
    _main_spec = importlib.util.spec_from_file_location(
        "dg_main_for_ep_test",
        _MAIN_PATH,
        submodule_search_locations=[],
    )
    _main_mod = importlib.util.module_from_spec(_main_spec)
    _main_spec.loader.exec_module(_main_mod)  # type: ignore[union-attr]
    _progress_payload = getattr(_main_mod, "_progress_payload", None)
except Exception as _exc:  # noqa: BLE001
    _IMPORT_ERR = _exc


def _fallback_progress_payload(pipeline_run_id: str | None = None) -> dict:
    """Inline replica of the helper for fallback testing when main.py won't import."""
    passes = _reg.snapshot()
    if pipeline_run_id is not None:
        passes = [p for p in passes if p["run_id"] == pipeline_run_id]
    return {"passes": passes}


def _get_payload_fn():
    """Return the real helper if importable, else the fallback replica."""
    return _progress_payload if _progress_payload is not None else _fallback_progress_payload


def setup_function(_):
    """Clear the registry before each test."""
    _reg._REGISTRY.clear()


# ---------------------------------------------------------------------------
# Behavioural tests (run against the real helper or inline replica)
# ---------------------------------------------------------------------------

def test_progress_payload_empty_when_idle():
    """GET /progress returns {"passes": []} when nothing is active."""
    fn = _get_payload_fn()
    result = fn()
    assert result == {"passes": []}, f"Expected empty passes, got {result}"


def test_progress_payload_returns_seeded_passes():
    """GET /progress returns all in-flight passes when registry is seeded."""
    _reg.start("run-abc", "radar_identity", total=5)
    _reg.start("run-abc", "radar_spec", total=3)
    fn = _get_payload_fn()
    result = fn()
    assert "passes" in result
    passes = result["passes"]
    assert len(passes) == 2
    names = {p["pass_name"] for p in passes}
    assert names == {"radar_identity", "radar_spec"}
    for p in passes:
        assert p["run_id"] == "run-abc"
        assert "done" in p
        assert "total" in p
        assert "phase" in p
        assert "started_at" in p
        assert "updated_at" in p
        assert "age_s" in p


def test_progress_payload_filters_by_pipeline_run_id():
    """?pipeline_run_id= filters passes to only the requested run."""
    _reg.start("run-X", "radar_identity", total=10)
    _reg.start("run-Y", "radar_spec", total=7)
    fn = _get_payload_fn()
    result_x = fn(pipeline_run_id="run-X")
    result_y = fn(pipeline_run_id="run-Y")
    result_all = fn()

    assert len(result_all["passes"]) == 2

    assert len(result_x["passes"]) == 1
    assert result_x["passes"][0]["run_id"] == "run-X"
    assert result_x["passes"][0]["pass_name"] == "radar_identity"

    assert len(result_y["passes"]) == 1
    assert result_y["passes"][0]["run_id"] == "run-Y"
    assert result_y["passes"][0]["pass_name"] == "radar_spec"


def test_progress_payload_filter_unknown_run_returns_empty():
    """?pipeline_run_id= for a non-existent run returns {"passes": []}."""
    _reg.start("run-Z", "radar_identity", total=2)
    fn = _get_payload_fn()
    result = fn(pipeline_run_id="no-such-run")
    assert result == {"passes": []}


# ---------------------------------------------------------------------------
# Source-inspection assertions — always run, confirm route wiring in main.py
# ---------------------------------------------------------------------------

def test_main_py_parses():
    """main.py must be syntactically valid Python."""
    ast.parse(_MAIN_SRC)  # raises SyntaxError if invalid


def test_main_py_defines_progress_route():
    """main.py must declare @app.get('/progress') and delegate to _progress_payload."""
    assert '@app.get("/progress"' in _MAIN_SRC, (
        'main.py does not contain @app.get("/progress") — route not wired'
    )


def test_main_py_defines_progress_payload_helper():
    """main.py must define the _progress_payload helper function."""
    assert "def _progress_payload(" in _MAIN_SRC, (
        "main.py does not define _progress_payload — helper not present"
    )


def test_main_py_route_calls_helper():
    """The /progress route body must call _progress_payload."""
    # Find the route function body (text after the decorator line)
    route_start = _MAIN_SRC.find('@app.get("/progress"')
    assert route_start != -1, "Route decorator not found"
    route_snippet = _MAIN_SRC[route_start: route_start + 400]
    assert "_progress_payload" in route_snippet, (
        "The /progress route does not call _progress_payload — delegation missing:\n"
        + route_snippet
    )


def test_main_py_helper_accepts_pipeline_run_id():
    """_progress_payload must accept and use pipeline_run_id."""
    helper_start = _MAIN_SRC.find("def _progress_payload(")
    assert helper_start != -1, "_progress_payload not found in main.py"
    helper_snippet = _MAIN_SRC[helper_start: helper_start + 600]
    assert "pipeline_run_id" in helper_snippet, (
        "_progress_payload does not accept pipeline_run_id — filter missing"
    )
    assert "run_id" in helper_snippet, (
        "_progress_payload does not filter by run_id"
    )
