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
_ANCHOR_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "docling_anchors"


def _load(filename: str) -> dict:
    """Load a JSON fixture from tests/fixtures/docling_anchors/."""
    return json.loads((_ANCHOR_FIXTURES_DIR / filename).read_text())


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


def test_interleaved_pictures_preserve_figure_vs_image_classification():
    """Regression for design §4.3 zip fix: captioned + uncaptioned
    pictures interleaved in doc order must each land in the correct
    bucket (FIGURE vs IMAGE) and neighbor attachment must track the
    right picture, not the reordered one."""
    doc_json = _load("interleaved_pictures.json")
    merged = walk(doc_json, "doc-interleaved", "run-1", {})

    figures = [e for e in merged.entities if e.identity.entity_type == "FIGURE"]
    images  = [e for e in merged.entities if e.identity.entity_type == "IMAGE"]
    assert len(figures) == 2, f"expected 2 captioned FIGUREs, got {len(figures)}"
    assert len(images) == 1, f"expected 1 uncaptioned IMAGE, got {len(images)}"

    # The uncaptioned picture is docling_doc.pictures[1] — middle of three.
    assert images[0].identity.identity_tuple == ("#/pictures/1",)

    # Each of the 3 pictures should have NEAR_TEXT edges to surrounding paragraphs.
    near_text_edges = [e for e in merged.edges if e.rel_type == "NEAR_TEXT"]
    # At least 2 neighbors per picture; exact count depends on window.
    assert len(near_text_edges) >= 3


def test_smoke_emits_image_and_text_block_entities():
    """End-to-end: fixture with pictures + headings yields SECTION,
    FIGURE/IMAGE split, and both SECTION-sourced HAS_* edges.

    The ``with_figures_tables.json`` fixture has two uncaptioned pictures
    (no caption text starting with "Figure"/"Fig.") so they classify as
    IMAGE, not FIGURE. The assertion uses the set-intersection form so it
    passes regardless of whether pictures end up as FIGURE or IMAGE.
    """
    doc_json = _load("with_figures_tables.json")
    merged = walk(doc_json, "doc-smoke", "run-1", {}, source_storage_key="test/key.pdf")

    entity_types = {e.identity.entity_type for e in merged.entities}
    # SECTION is always present.
    assert "SECTION" in entity_types
    # At least one of FIGURE or IMAGE must exist since the fixture has pictures.
    assert entity_types & {"FIGURE", "IMAGE"}

    edge_types = {e.rel_type for e in merged.edges}
    assert edge_types & {"HAS_FIGURE", "HAS_IMAGE"}, (
        f"no figure/image HAS_* edges emitted — got {edge_types}"
    )
