"""Pins docker/docling-graph/app/provenance.py value-grounding helpers to the
shared fixture (tests/fixtures/unit_matcher_cases.json) so the mirror can never
drift from app/services/field_value_grounding.py. Update BOTH copies + the
fixture together, always."""
# Import approach: the project pyproject.toml sets pythonpath=["."]; pytest
# therefore adds the worktree root to sys.path BEFORE this file runs, so
# `from app.xxx import` would resolve to the worktree's app/ rather than
# docker/docling-graph/app.  We work around this by loading docling-graph/app
# as an importlib package under a private alias "dg_app", which avoids the
# `app` namespace collision entirely.  The three helpers we need
# (_vg_nfc, _vg_has_unit_token, _vg_value_in_chunk) are then imported
# directly from that private module object.
# PYTHONPATH="docker/docling-graph/repo:$PYTHONPATH" is still required so that
# the patched docling-graph library imports inside provenance.py resolve.
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_DG_ROOT = _HERE.parent           # docker/docling-graph/
_APP_PKG = _DG_ROOT / "app"       # docker/docling-graph/app/


def _load_dg_provenance():
    """Load docker/docling-graph/app/provenance.py under the alias 'dg_app'
    so it never collides with the worktree's top-level 'app' package."""
    # Step 1: load the dg_app package (docker/docling-graph/app/__init__.py)
    #         under a collision-free name.
    pkg_spec = importlib.util.spec_from_file_location(
        "dg_app", _APP_PKG / "__init__.py",
        submodule_search_locations=[str(_APP_PKG)],
    )
    pkg_mod = importlib.util.module_from_spec(pkg_spec)
    sys.modules["dg_app"] = pkg_mod
    pkg_spec.loader.exec_module(pkg_mod)

    # Step 2: for each submodule that provenance.py imports from 'app.*',
    #         pre-register it under 'dg_app.*' AND under 'app.*' in a way
    #         that won't shadow the real app package permanently.
    #
    # Simpler approach: temporarily replace sys.modules['app'] so that
    # `from app._numeric_evidence import ...` inside provenance.py finds the
    # right package, then restore it afterwards.
    real_app = sys.modules.get("app")
    sys.modules["app"] = pkg_mod

    try:
        mod_spec = importlib.util.spec_from_file_location(
            "dg_app.provenance", _APP_PKG / "provenance.py",
        )
        mod = importlib.util.module_from_spec(mod_spec)
        mod.__package__ = "dg_app"
        sys.modules["dg_app.provenance"] = mod
        mod_spec.loader.exec_module(mod)
    finally:
        # Restore the original app (or remove if there wasn't one)
        if real_app is None:
            sys.modules.pop("app", None)
        else:
            sys.modules["app"] = real_app
        # Clean up residual module entries registered during loading
        sys.modules.pop("app._numeric_evidence", None)
        sys.modules.pop("dg_app.provenance", None)
        sys.modules.pop("dg_app", None)

    return mod


prov = _load_dg_provenance()

_CASES = json.loads((_HERE / "fixtures" / "unit_matcher_cases.json").read_text())


@pytest.mark.parametrize("case", _CASES["unit_token_cases"])
def test_mirror_unit_token(case):
    text = prov._vg_nfc(case["text"])
    assert prov._vg_has_unit_token(text, case["units"]) is case["expect"]


@pytest.mark.parametrize("case", _CASES["value_in_chunk_cases"])
def test_mirror_value_in_chunk(case):
    text = prov._vg_nfc(case["text"])
    got = prov._vg_value_in_chunk(set(case["num_strs"]), case["units"], text)
    assert got is (case["expect"] is not None)


def test_mirror_num_variants_equivalence():
    # pyproject pythonpath=["."] puts the worktree root on sys.path, so this
    # resolves to the REAL app/services (not dg_app — see _load_dg_provenance).
    from app.services.field_value_grounding import num_variants as app_nv
    for v in (10.6, 2391, 2391.0, 0.5, "1,250", float("inf")):
        assert prov._vg_num_variants(v) == app_nv(v), v
