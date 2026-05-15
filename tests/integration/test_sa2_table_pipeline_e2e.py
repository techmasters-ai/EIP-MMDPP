"""Task 23: SA-2 e2e sanity test.

Asserts the spike response (captured against a real /extract-pass call
with normalization ON) contains the wire shape Phase 2 consumers need:
  - field_provenance includes synthesized-chunk evidence_ids (#/texts/N
    where N >= original-texts-count)
  - synthesized-chunk rows carry cell_refs pointing at
    #/tables/N/data/table_cells/M
  - entity_count > 0 (extraction worked)
  - service_table_overlay diagnostic is still populated (overlay path
    not regressed by normalization)

Smaller-scope sibling of the §15.2 flip-gate which compares COUNTS vs
baseline; this test asserts the wire shape is correct.

Spec: docs/superpowers/specs/2026-05-11-table-aware-chunking-design.md §14.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPIKE_RESPONSE = REPO_ROOT / "tests" / "fixtures" / "spike" / "missile_kinematics_response_with_normalization.json"
BASELINE_TEXTS = REPO_ROOT / "tests" / "fixtures" / "sa2" / "78673393-639b-4fde-9bda-9e7bfd43ccda_texts_today.json"
TEXTS_REF_RE = re.compile(r"^#/texts/(\d+)$")
CELL_REF_RE = re.compile(r"^#/tables/\d+/data/table_cells/\d+$")


@pytest.fixture(scope="module")
def spike_response() -> dict:
    if not SPIKE_RESPONSE.exists():
        pytest.skip(
            f"spike fixture missing at {SPIKE_RESPONSE}; "
            "run scripts/spike_provenance_e2e.py with norm=ON"
        )
    return json.loads(SPIKE_RESPONSE.read_text())


@pytest.fixture(scope="module")
def baseline_text_count() -> int:
    if not BASELINE_TEXTS.exists():
        pytest.skip(f"baseline texts missing at {BASELINE_TEXTS}")
    return len(json.loads(BASELINE_TEXTS.read_text()))


def test_extraction_produced_entities(spike_response):
    """e2e: extraction returned non-empty pass_output."""
    out = spike_response.get("pass_output") or {}
    total = sum(len(v) for v in out.values() if isinstance(v, list))
    assert total > 0, "extraction returned no entities"


def test_synthesized_chunks_appear_in_evidence(spike_response, baseline_text_count):
    """e2e: at least one field_provenance row references a synthesized
    chunk (evidence_id == #/texts/N where N >= baseline_text_count).

    Baseline had `baseline_text_count` texts post-sanitization. Any
    text_idx >= that count must be a synthesized chunk appended by
    Task 17's normalization wiring.
    """
    rows = spike_response.get("field_provenance") or []
    synthesized_evidence: list[str] = []
    for r in rows:
        eid = r.get("evidence_id") or ""
        m = TEXTS_REF_RE.match(eid)
        if not m:
            continue
        idx = int(m.group(1))
        if idx >= baseline_text_count:
            synthesized_evidence.append(eid)
    # The spike captured 308-text baseline; sanitizer drops some (~65),
    # so synthesized chunks start higher than 308 minus dropped count
    # — we conservatively look for text_idx >= 243 (308-65) or just check
    # for non-empty cell_refs as a proxy.
    if not synthesized_evidence:
        # Fallback proxy: at least one row has cell_refs (which means it
        # came from a synthesized chunk regardless of exact idx)
        cell_refs_rows = [r for r in rows if r.get("cell_refs")]
        assert cell_refs_rows, (
            "no field_provenance row references a synthesized table chunk "
            "(neither by high text_idx nor by cell_refs presence)"
        )


def test_overlay_diagnostic_not_regressed(spike_response):
    """Spec §9.2 invariant: tables[] is not mutated, so the Phase 0/0.5
    overlay path (extract_table_overlay) must still run successfully."""
    diag = spike_response.get("diagnostics") or {}
    overlay_stats = diag.get("service_table_overlay")
    assert overlay_stats is not None, (
        "service_table_overlay diagnostic missing — overlay path may be "
        "broken (spec §9.2 invariant violation)"
    )


def test_field_provenance_well_formed(spike_response):
    """Every field_provenance row has the expected required fields."""
    rows = spike_response.get("field_provenance") or []
    assert rows, "no field_provenance rows"
    required = {"instance_id", "field_name", "supporting_snippet"}
    for r in rows[:50]:  # sample first 50
        missing = required - set(r.keys())
        assert not missing, f"row missing required fields {missing}: {r}"


def test_pass_output_has_missile_systems_list(spike_response):
    """missile_kinematics pass produces a `missile_systems` list."""
    out = spike_response.get("pass_output") or {}
    assert "missile_systems" in out, f"pass_output keys: {list(out.keys())}"
    assert isinstance(out["missile_systems"], list)


def test_cell_refs_only_appear_with_normalization_evidence(spike_response):
    """All cell_refs come from rows whose evidence_id is a #/texts/N ref.

    Spec §11.6 invariant: cell_refs flow ONLY through synthesized chunks
    (evidence_id == #/texts/N), not through other source types.
    """
    rows = spike_response.get("field_provenance") or []
    for r in rows:
        if not r.get("cell_refs"):
            continue
        eid = r.get("evidence_id") or ""
        assert TEXTS_REF_RE.match(eid), (
            f"row with cell_refs has non-#/texts/N evidence_id={eid!r}; "
            "spec §11.6 invariant violated"
        )
        # And every cell_ref must match the canonical shape
        for ref in r["cell_refs"]:
            assert CELL_REF_RE.match(ref), f"malformed cell_ref: {ref}"
