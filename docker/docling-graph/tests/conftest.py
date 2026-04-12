# docker/docling-graph/tests/conftest.py
#
# Makes `from app.main import app` work when pytest is invoked from the
# repo root. We APPEND (not prepend) so we don't shadow the repo-root
# `app/` package for tests that need `app.services.*`.
# The repo-root `app/` is inserted at index 0 by pytest automatically
# (rootdir is always first), so appending here is safe.
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.append(str(_SERVICE_ROOT))
