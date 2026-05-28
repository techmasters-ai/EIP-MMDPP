"""Unit tests for ``app.services.hybrid_chunking``.

Tests per plan ``docs/superpowers/plans/2026-05-27-merged-chunk-routing.md``
Task 2 Steps 1a-1e:

* 1a determinism — two calls return identical chunk count + identical
  ``(chunk_index, source_refs)`` tuples
* 1b provenance — every ``MergedChunk.source_refs`` is non-empty and refs
  resolve to real ``texts[]/tables[]/pictures[]`` indices
* 1c heading prefix — ``MergedChunk.text`` contains the parent heading
  string (HybridChunker's ``contextualize`` prepends title+headings)
* 1d empty doc — empty ``texts/tables/pictures`` returns ``[]`` (no crash)
* 1e chunk_index density — ``[c.chunk_index for c in chunks]`` is
  ``[0, 1, ..., len-1]``

The tests use a synthetic ``DoclingDocument`` constructed via the public
docling API (``add_title``/``add_heading``/``add_text``) and exported to
JSON via ``export_to_dict``.  This sidesteps schema-evolution churn in
the on-disk ``tests/fixtures/docling_anchors/*.json`` snapshots — some of
which predate the current ``DocItemLabel`` constraints in docling 2.x.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from docling_core.types.doc import DoclingDocument

from app.services.hybrid_chunking import (
    HybridChunkConfig,
    MergedChunk,
    build_hybrid_chunks_for_extraction,
)


# ---------------------------------------------------------------------------
# Synthetic doc builders
# ---------------------------------------------------------------------------


def _build_dvina_like_doc_json() -> dict:
    """Construct a Dvina-shaped DoclingDocument as a dict.

    Mirrors the structure of a real military-system technical manual:
    title + multiple section headers + technical prose paragraphs across
    several sections.  Yields multiple HybridChunker chunks (one per
    leaf section's text body, post-merge).
    """
    doc = DoclingDocument(name="dvina_like_synthetic")
    doc.add_title("S-75 Dvina System Manual")
    doc.add_heading("Chapter 1: Overview", level=1)
    doc.add_text(
        label="text",
        text=(
            "The S-75 Dvina is a Soviet-era surface-to-air missile system. "
            "It entered service with the Soviet armed forces in 1957 and "
            "remained the backbone of the Soviet Air Defence Forces."
        ),
    )
    doc.add_heading("Chapter 2: Kinematics", level=1)
    doc.add_text(
        label="text",
        text=(
            "Maximum engagement range is approximately 43 kilometres. "
            "Minimum engagement range is 8 kilometres. The missile reaches "
            "Mach 3.5 at altitude. Operational weight is 2300 kilograms."
        ),
    )
    doc.add_heading("Section 2.1: Tracking", level=2)
    doc.add_text(
        label="text",
        text=(
            "Radar tracking accuracy is plus or minus 5 metres at 50 "
            "kilometre range under standard atmospheric conditions."
        ),
    )
    doc.add_heading("Chapter 3: Components", level=1)
    doc.add_text(
        label="text",
        text=(
            "The system comprises the Fan Song radar, the SM-90 launcher, "
            "and the PR-11 transporter/transloader. Each launcher carries "
            "one V-750 missile."
        ),
    )
    return doc.export_to_dict()


def _build_empty_doc_json() -> dict:
    """A DoclingDocument with no texts/tables/pictures."""
    doc = DoclingDocument(name="empty_synthetic")
    return doc.export_to_dict()


# ---------------------------------------------------------------------------
# Step 1a — determinism
# ---------------------------------------------------------------------------


def test_chunker_determinism_chunk_count_and_source_refs() -> None:
    """Two calls on identical input return identical chunk count + identical
    ``(chunk_index, source_refs)`` tuples.

    HybridChunker walks ``body.children`` deterministically and has no RNG,
    so output must be byte-identical across calls.
    """
    doc_json = _build_dvina_like_doc_json()

    first = build_hybrid_chunks_for_extraction(doc_json)
    second = build_hybrid_chunks_for_extraction(doc_json)

    assert len(first) == len(second), (
        f"Chunk count drifted between two calls on identical input: "
        f"{len(first)} vs {len(second)}"
    )
    first_tuples = [(c.chunk_index, tuple(c.source_refs)) for c in first]
    second_tuples = [(c.chunk_index, tuple(c.source_refs)) for c in second]
    assert first_tuples == second_tuples, (
        "Determinism violated: (chunk_index, source_refs) tuples differ "
        "across two calls on the same doc_json."
    )


# ---------------------------------------------------------------------------
# Step 1b — provenance
# ---------------------------------------------------------------------------


_REF_RE = re.compile(r"^#/(texts|tables|pictures)/(\d+)$")


def test_source_refs_non_empty_and_resolve_to_valid_indices() -> None:
    """Every MergedChunk has at least one source_ref, and every ref
    matches a real ``texts[i]/tables[i]/pictures[i]`` index in doc_json.
    """
    doc_json = _build_dvina_like_doc_json()
    chunks = build_hybrid_chunks_for_extraction(doc_json)

    assert chunks, "Synthetic Dvina-like doc must produce >=1 merged chunk"

    n_texts = len(doc_json.get("texts", []))
    n_tables = len(doc_json.get("tables", []))
    n_pictures = len(doc_json.get("pictures", []))

    for c in chunks:
        assert isinstance(c.source_refs, list)
        assert c.source_refs, (
            f"chunk_index={c.chunk_index} has empty source_refs — "
            "every merged chunk must trace back to >=1 element"
        )
        for ref in c.source_refs:
            m = _REF_RE.match(ref)
            assert m, f"source_ref {ref!r} not in #/(texts|tables|pictures)/N form"
            kind, idx_str = m.group(1), m.group(2)
            idx = int(idx_str)
            cap = {"texts": n_texts, "tables": n_tables, "pictures": n_pictures}[kind]
            assert 0 <= idx < cap, (
                f"source_ref {ref!r} index {idx} out of range "
                f"(doc has {cap} {kind})"
            )


# ---------------------------------------------------------------------------
# Step 1c — heading prefix
# ---------------------------------------------------------------------------


def test_chunk_text_contains_parent_heading() -> None:
    """``MergedChunk.text`` (output of ``chunker.contextualize``) prepends
    the title + nearest heading(s) for chunks that descend from a heading.
    """
    doc_json = _build_dvina_like_doc_json()
    chunks = build_hybrid_chunks_for_extraction(doc_json)

    assert chunks, "doc must produce >=1 chunk"

    # The Dvina-like doc has these headings; each text body's chunk must
    # contain the nearest heading text in its contextualized output.
    expected_heading_per_text = {
        "Soviet-era surface-to-air": "Chapter 1: Overview",
        "Maximum engagement range": "Chapter 2: Kinematics",
        "Radar tracking accuracy": "Section 2.1: Tracking",
        "comprises the Fan Song": "Chapter 3: Components",
    }

    matched: dict[str, str] = {}
    for c in chunks:
        for body_fragment, heading in expected_heading_per_text.items():
            if body_fragment in c.text:
                matched[body_fragment] = c.text
                assert heading in c.text, (
                    f"chunk containing {body_fragment!r} did not contain "
                    f"expected heading {heading!r}; text was:\n{c.text!r}"
                )
                # The title should also appear (HybridChunker prepends it).
                assert "S-75 Dvina System Manual" in c.text

    assert matched, (
        "No chunk matched any expected body fragment — HybridChunker output "
        "diverged from synthetic input"
    )


# ---------------------------------------------------------------------------
# Step 1d — empty doc
# ---------------------------------------------------------------------------


def test_empty_doc_returns_empty_list_not_crash() -> None:
    """A DoclingDocument with no texts/tables/pictures returns ``[]``
    rather than raising — per plan caller contract.
    """
    doc_json = _build_empty_doc_json()
    out = build_hybrid_chunks_for_extraction(doc_json)
    assert out == [], (
        f"Empty doc_json must return [], not crash; got {out!r}"
    )


# ---------------------------------------------------------------------------
# Step 1e — chunk_index density
# ---------------------------------------------------------------------------


def test_chunk_index_dense_from_zero() -> None:
    """``[c.chunk_index for c in chunks]`` is ``[0, 1, ..., len-1]`` —
    no gaps, no off-by-one.
    """
    doc_json = _build_dvina_like_doc_json()
    chunks = build_hybrid_chunks_for_extraction(doc_json)
    assert chunks
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# Additional invariants — types + failure semantics
# ---------------------------------------------------------------------------


def test_merged_chunk_dataclass_is_frozen() -> None:
    """``MergedChunk`` is a frozen dataclass — instances cannot be
    mutated post-construction.  Guards against accidental in-place edits
    in callers (e.g. the indexer).
    """
    c = MergedChunk(
        chunk_index=0,
        text="hello",
        source_refs=["#/texts/0"],
        page_no=None,
        token_count=1,
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        c.text = "mutated"  # type: ignore[misc]


def test_hybrid_chunk_config_defaults_match_docling_graph() -> None:
    """Defaults mirror docling-graph's ``DocumentChunker`` (post-Task 0a):

    - tokenizer_model_name="BAAI/bge-m3"  (Task 0a pinned the override)
    - max_tokens=512                        (DocumentChunker default)
    - merge_peers=True                      (DocumentChunker default)
    """
    cfg = HybridChunkConfig()
    assert cfg.tokenizer_model_name == "BAAI/bge-m3"
    assert cfg.max_tokens == 512
    assert cfg.merge_peers is True


def test_malformed_doc_json_raises_value_error() -> None:
    """``DoclingDocument.model_validate`` failure must propagate as
    ``ValueError`` (which pydantic raises as ``ValidationError`` — a
    subclass of ``ValueError``).  Caller (the C.4 dispatcher's
    try/except in Task 3) converts to RUN_FULL fallback.
    """
    bad = {"this": "is not a docling document"}
    with pytest.raises(ValueError):
        build_hybrid_chunks_for_extraction(bad)


def test_token_count_is_positive_for_non_empty_text() -> None:
    """Every emitted ``MergedChunk`` has a positive ``token_count``."""
    doc_json = _build_dvina_like_doc_json()
    chunks = build_hybrid_chunks_for_extraction(doc_json)
    assert chunks
    for c in chunks:
        assert c.token_count > 0, (
            f"chunk_index={c.chunk_index} has token_count={c.token_count}"
        )


def test_page_no_is_none_for_doc_without_prov() -> None:
    """Synthetic docs have no ``prov`` entries; ``page_no`` resolves to
    ``None`` rather than raising.
    """
    doc_json = _build_dvina_like_doc_json()
    chunks = build_hybrid_chunks_for_extraction(doc_json)
    assert chunks
    assert all(c.page_no is None for c in chunks), (
        "Doc has no prov entries; every page_no must be None"
    )
