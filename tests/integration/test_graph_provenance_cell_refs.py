"""§15.2 provenance gate.

Asserts the spike's saved response (captured with master switches ON
on the real SA-2 doc) has the channel-A cell_refs flow working:
- field_provenance rows exist
- a meaningful fraction (≥10%) carry non-empty cell_refs
- all cell_refs match '#/tables/N/data/table_cells/M' shape
- evidence_id on cell_refs-bearing rows points at synthesized #/texts/N

This is the formalized gate version of scripts/spike_provenance_e2e.py.
It runs against the saved fixture so CI doesn't need a live stack.

Spec: docs/superpowers/specs/2026-05-11-table-aware-chunking-design.md §15.2.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPIKE_RESPONSE = REPO_ROOT / "tests" / "fixtures" / "spike" / "missile_kinematics_response_with_normalization.json"
CELL_REF_RE = re.compile(r"^#/tables/\d+/data/table_cells/\d+$")
TEXTS_REF_RE = re.compile(r"^#/texts/\d+$")


@pytest.fixture(scope="module")
def spike_response() -> dict:
    if not SPIKE_RESPONSE.exists():
        pytest.skip(
            f"spike response fixture missing at {SPIKE_RESPONSE}; "
            "run scripts/spike_provenance_e2e.py first"
        )
    return json.loads(SPIKE_RESPONSE.read_text())


def test_field_provenance_present(spike_response):
    """The pass produced field_provenance rows at all."""
    rows = spike_response.get("field_provenance") or []
    assert len(rows) > 0, "field_provenance is empty — extraction may have failed"


def test_some_rows_have_cell_refs(spike_response):
    """At least one field_provenance row has populated cell_refs.

    This is the binary gate: the channel-A flow is working at all.
    """
    rows = spike_response.get("field_provenance") or []
    with_cells = [r for r in rows if r.get("cell_refs")]
    assert len(with_cells) > 0, (
        f"NO field_provenance rows carry cell_refs out of {len(rows)} total. "
        "Channel-A flow is broken — see spec §11.6 for the two channels and "
        "main.py enrichment wrapper for the lookup."
    )


def test_meaningful_fraction_have_cell_refs(spike_response):
    """At least 10% of field_provenance rows have cell_refs.

    A small fraction is OK (prose chunks won't have cell_refs), but if
    almost none do, normalization isn't reaching the LLM input.
    """
    rows = spike_response.get("field_provenance") or []
    with_cells = [r for r in rows if r.get("cell_refs")]
    fraction = len(with_cells) / max(len(rows), 1)
    assert fraction >= 0.10, (
        f"Only {len(with_cells)}/{len(rows)} ({fraction:.1%}) rows have cell_refs. "
        "Expected ≥10% — normalization may not be feeding the LLM input adequately."
    )


def test_all_cell_refs_match_expected_shape(spike_response):
    """Every cell_ref matches '#/tables/N/data/table_cells/M' shape."""
    rows = spike_response.get("field_provenance") or []
    bad_refs: list[str] = []
    for r in rows:
        for ref in (r.get("cell_refs") or []):
            if not CELL_REF_RE.match(ref):
                bad_refs.append(ref)
    assert not bad_refs, f"cell_refs with unexpected shape: {bad_refs[:5]}"


def test_evidence_id_on_cell_refs_rows_points_at_synthesized_texts(spike_response):
    """Rows that carry cell_refs should have evidence_id matching '#/texts/N'.

    The enrichment wrapper looks up cell_refs by parsing evidence_id as
    #/texts/N. If evidence_id isn't in that shape, the bridge map lookup
    skips it — yet we have cell_refs, which means the row reached the
    wrapper. So evidence_id must already have been #/texts/N.
    """
    rows = spike_response.get("field_provenance") or []
    bad: list[dict] = []
    for r in rows:
        if not r.get("cell_refs"):
            continue
        eid = r.get("evidence_id") or r.get("element_uid")
        if not eid or not TEXTS_REF_RE.match(eid):
            bad.append({"evidence_id": eid, "cell_refs": r.get("cell_refs")[:2]})
    assert not bad, (
        f"rows with cell_refs but non-#/texts/N evidence_id: {bad[:3]}"
    )


def test_self_refs_shape_unchanged(spike_response):
    """CRITICAL: spec §11.6 requires cell_refs to NOT pollute self_refs.

    field_provenance.evidence_id (== chunk's self_ref) should stay
    today-shape: #/texts/N or #/tables/N. Never a cell ref.
    """
    rows = spike_response.get("field_provenance") or []
    polluted: list[str] = []
    for r in rows:
        eid = r.get("evidence_id") or ""
        if "/data/table_cells/" in eid:
            polluted.append(eid)
    assert not polluted, (
        f"evidence_id contains cell refs — spec §11.6 invariant violated: {polluted[:3]}"
    )
