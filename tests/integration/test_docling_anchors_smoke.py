"""D-6: Real-document smoke test for the Docling anchor walker.

Loads a real ``docling_document.json`` from
``tests/fixtures/real_docs/`` (if present — otherwise skips) and runs
``walk()`` end-to-end. Asserts the walker survives real-world shape and
produces a non-zero SECTION count (plus non-zero FIGURE count when the
fixture contains pictures).

Marked ``@pytest.mark.integration`` so the default unit-test run skips
it without the fixture dir. Add real Docling outputs under
``tests/fixtures/real_docs/*.json`` to exercise.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.docling_anchors import walk

REAL_DOCS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "real_docs"


def _real_doc_fixtures() -> list[Path]:
    if not REAL_DOCS_DIR.is_dir():
        return []
    return sorted(REAL_DOCS_DIR.glob("*.json"))


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not _real_doc_fixtures(),
    reason="no tests/fixtures/real_docs/*.json fixtures present",
)
@pytest.mark.parametrize(
    "fixture_path",
    _real_doc_fixtures(),
    ids=lambda p: p.name,
)
def test_walk_real_docling_doc_produces_sections(fixture_path: Path):
    docling_json = json.loads(fixture_path.read_text())
    merged = walk(
        docling_json,
        document_uuid="real-doc-smoke",
        pipeline_run_id="real-doc-smoke-run",
        ontology={},
    )

    sections = [
        e for e in merged.entities if e.identity.entity_type == "SECTION"
    ]
    assert sections, f"walker emitted no SECTIONs for {fixture_path.name}"

    # Non-zero figure count when fixture carries pictures.
    fixture_pictures = docling_json.get("pictures") or []
    figures = [
        e for e in merged.entities if e.identity.entity_type == "FIGURE"
    ]
    if fixture_pictures:
        assert figures, (
            f"{fixture_path.name} has {len(fixture_pictures)} pictures but "
            "walker emitted 0 FIGUREs"
        )
    assert len(figures) == len(fixture_pictures)

    fixture_tables = docling_json.get("tables") or []
    tables = [
        e for e in merged.entities if e.identity.entity_type == "TABLE"
    ]
    assert len(tables) == len(fixture_tables)

    # Every emitted entity carries pass_origins={"document_anchors"}
    for e in merged.entities:
        assert e.pass_origins == {"document_anchors"}
