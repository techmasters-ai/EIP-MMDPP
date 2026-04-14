# docker/docling-graph/tests/conftest.py
#
# Makes `from app.main import app` work when pytest is invoked from the
# repo root. We APPEND (not prepend) so we don't shadow the repo-root
# `app/` package for tests that need `app.services.*`.
# The repo-root `app/` is inserted at index 0 by pytest automatically
# (rootdir is always first), so appending here is safe.
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.append(str(_SERVICE_ROOT))

# ---------------------------------------------------------------------------
# Shared shim: loads the docling-graph app/main.py under a stable module
# name so tests can patch it regardless of sys.path ordering.
# ---------------------------------------------------------------------------
_DG_MODULE_NAME = "docling_graph_service_main"
_DG_SERVICE_ROOT = _HERE.parent


def _ensure_dg_app_package() -> None:
    """Ensure the docling-graph `app.*` sub-modules are importable as `app.*`.

    When the combined test suite runs from repo root, the repo-root `app/`
    package is already in sys.modules['app']. We temporarily swap it out
    for the docling-graph `app/` package so that `from app.config_builder
    import ...` in main.py resolves to the docling-graph package.

    This is called once; subsequent calls are no-ops.
    """
    import importlib
    import importlib.util

    if _DG_MODULE_NAME in sys.modules:
        return  # already done

    service_root = _DG_SERVICE_ROOT
    dg_app_path = service_root / "app"  # noqa: F841

    # Save state
    saved = {k: v for k, v in sys.modules.items() if k == "app" or k.startswith("app.")}
    saved_path = list(sys.path)

    # Swap: point sys.path[0] at service root so `import app` finds DG's app/
    sys.path.insert(0, str(service_root))
    # Remove any cached repo-root `app.*` modules so fresh imports find DG's
    for key in list(saved.keys()):
        del sys.modules[key]

    try:
        spec = importlib.util.spec_from_file_location(
            _DG_MODULE_NAME,
            service_root / "app" / "main.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_DG_MODULE_NAME] = mod
        spec.loader.exec_module(mod)
    finally:
        # Restore: remove DG's app.* from cache, re-insert repo-root's
        for key in list(sys.modules.keys()):
            if key == "app" or key.startswith("app."):
                del sys.modules[key]
        sys.modules.update(saved)
        # Restore sys.path
        sys.path[:] = saved_path


@pytest.fixture(scope="module")
def dg_app_module():
    """Load the docling-graph app module once per test module."""
    _ensure_dg_app_package()
    return sys.modules[_DG_MODULE_NAME]
