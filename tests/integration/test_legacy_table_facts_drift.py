"""§14 experimental-path drift guard.

Ensures _table_facts.py:synthesize_table_facts still produces non-empty
output on a doc with at least one normalizable column-major/hybrid
table. This guards against `_table_facts.py` rotting on disk now that
it's gated off by default — if a future refactor breaks it, the
DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS=true A/B path silently
becomes useless.

Pass criterion: count-based assertion, not text-content snapshot.
- TextItems emitted > 0  (catches catastrophic rot)
- Count within ±10% of recorded baseline (catches large drift)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Use the real SA-2 doc's table 0 (23x13 hybrid variants table) which both
# the new detect.py heuristic AND the older _table_facts.py heuristic
# classify as a normalizable shape. The smaller sa2_sample_table.json
# fixture used by the unit tests is too compact for _table_facts.py's
# stricter shape detection.
SA2_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "sa2_doc_table_0.json"

# Empirically captured on first add. Update intentionally when
# _table_facts.py has a deliberate behavior change.
EXPECTED_TEXT_ITEMS = 9   # captured 2026-05-12 against SA-2 doc table 0
                          # (23x13 hybrid, classifies HYBRID, 2 sections detected,
                          # 9 facts emitted for missile_propulsion pass).
TOLERANCE_PCT = 0.10


@pytest.fixture(scope="module")
def synthesize_table_facts():
    """Import synthesize_table_facts from docling-graph (sys.path swap)."""
    dg_app = REPO_ROOT / "docker" / "docling-graph"
    if str(dg_app) not in sys.path:
        sys.path.insert(0, str(dg_app))
    try:
        from app._table_facts import synthesize_table_facts  # type: ignore
    except ImportError as exc:
        pytest.skip(f"_table_facts.py unavailable ({exc})")
    return synthesize_table_facts


@pytest.fixture(scope="module")
def sa2_sample_doc() -> dict:
    if not SA2_FIXTURE_PATH.exists():
        pytest.skip(f"SA-2 sample fixture missing at {SA2_FIXTURE_PATH}")
    fixture = json.loads(SA2_FIXTURE_PATH.read_text())
    return {
        "tables": [fixture],
        "texts": [],
        "body": {"children": []},
    }


def test_synthesize_emits_nonempty(synthesize_table_facts, sa2_sample_doc):
    """Catastrophic rot guard: function emits > 0 TextItems."""
    doc = json.loads(json.dumps(sa2_sample_doc))
    pre = len(doc.get("texts") or [])
    _, stats = synthesize_table_facts(doc, active_pass="missile_propulsion")
    post = len(doc.get("texts") or [])
    new_items = post - pre
    assert new_items > 0, (
        f"synthesize_table_facts emitted 0 TextItems. stats={stats}"
    )


def test_synthesize_count_within_tolerance(synthesize_table_facts, sa2_sample_doc):
    """Drift guard. Skips if EXPECTED_TEXT_ITEMS not yet pinned."""
    doc = json.loads(json.dumps(sa2_sample_doc))
    pre = len(doc.get("texts") or [])
    _, _stats = synthesize_table_facts(doc, active_pass="missile_propulsion")
    post = len(doc.get("texts") or [])
    new_items = post - pre

    if EXPECTED_TEXT_ITEMS is None:
        pytest.skip(
            f"EXPECTED_TEXT_ITEMS not pinned yet; observed count is "
            f"{new_items}. Set EXPECTED_TEXT_ITEMS = {new_items} at the top "
            f"of this file to enable the tolerance check."
        )

    drift = abs(new_items - EXPECTED_TEXT_ITEMS) / EXPECTED_TEXT_ITEMS
    assert drift <= TOLERANCE_PCT, (
        f"TextItem count drifted: observed={new_items}, expected="
        f"{EXPECTED_TEXT_ITEMS}, drift={drift:.1%} > {TOLERANCE_PCT:.0%}. "
        f"If intentional, update EXPECTED_TEXT_ITEMS."
    )


def test_idempotence_flag_set(synthesize_table_facts, sa2_sample_doc):
    """Second call short-circuits via _IDEMPOTENCE_FLAG."""
    doc = json.loads(json.dumps(sa2_sample_doc))
    synthesize_table_facts(doc, active_pass="missile_propulsion")
    _, stats2 = synthesize_table_facts(doc, active_pass="missile_propulsion")
    assert stats2.idempotent_skip is True
