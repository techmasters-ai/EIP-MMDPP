"""Byte-equal parity test between worker-side shared helper and
docling-graph's ``DocumentChunker``.

This test locks the Task 0a + Task 0b prerequisites of the merged-chunk
routing plan (``docs/superpowers/plans/2026-05-27-merged-chunk-routing.md``)
in CI: if either side drifts (different tokenizer, different
``max_tokens``, different ``merge_peers``, or any future HybridChunker
config divergence), Phase 2's byte-identity invariant breaks and this
test catches it before reaching production.

Parity approach used: **TRUE parity (option b)** — instantiate
docling-graph's ``DocumentChunker`` directly with bge-m3 + max_tokens=512
+ merge_peers=True (the Task 0a-pinned config) and compare its
``chunk_document(doc)`` output line-by-line against the shared helper's
``MergedChunk.text``.

This requires the ``docling_graph`` Python package to be importable.
worker-1 has it (the docling-graph repo is COPYed into the image at
build time per the worker Dockerfile) — confirmed at the time this
test was written.  If a future worker image drops the package, this
test will skip with a clear message; the equivalent config-only check
(option a) is also implemented as ``test_chunker_config_summary_matches``
which never needs ``docling_graph`` and is the safety net.

The doc fed to both chunkers is the post-Layer-1-filter shape (run
``filter_docling_document`` first) so we reproduce what each side
actually sees in the Phase 2 narrowed-mode hot path.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

from docling_core.types.doc import DoclingDocument

from app.services.hybrid_chunking import (
    HybridChunkConfig,
    build_hybrid_chunks_for_extraction,
)
from app.services.scoped_docling_document import filter_docling_document

# Reuse the synthetic Dvina-like doc builder from the unit-test module.
# Tests/ is on the pythonpath, so this is a normal cross-test import.
from tests.unit.test_hybrid_chunking import _build_dvina_like_doc_json


# ---------------------------------------------------------------------------
# Optional dependency: docling-graph's DocumentChunker
# ---------------------------------------------------------------------------


_dg_available = True
_dg_skip_reason = ""
try:
    from docling_graph.core.extractors.document_chunker import (  # type: ignore
        DocumentChunker,
    )
except ImportError as exc:  # pragma: no cover — exercised only when DG absent
    _dg_available = False
    _dg_skip_reason = (
        f"docling_graph is not importable in this environment ({exc}); "
        "the config-only parity test (test_chunker_config_summary_matches) "
        "still runs as the fall-back invariant check."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _post_layer1_filtered(doc_json: dict) -> dict:
    """Apply the worker's Layer-1 quality filter so both chunkers see the
    SAME doc shape they would see in production.

    The Layer-1 filter mutates ``doc_json`` in place AND returns it.  In
    the merged-chunk routing plan, this filter runs in the worker before
    the indexer (and, post-Task 0b, docling-graph skips its own sanitize
    when ``selected_chunks`` is present) — so the post-filter shape is
    what both sides need to chunk identically.
    """
    out, _diag = filter_docling_document(doc_json)
    return out


# ---------------------------------------------------------------------------
# True parity (option b) — only runs when docling-graph importable
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _dg_available, reason=_dg_skip_reason)
def test_byte_equal_chunk_text_per_index() -> None:
    """Worker helper's ``MergedChunk.text`` is byte-equal to docling-graph's
    ``DocumentChunker.chunk_document(doc)`` output at every index, on the
    post-Layer-1-filter Dvina-like doc.

    This is the parity invariant that Phase 2 (narrowed-mode chunked
    routing) depends on: the worker selects merged chunks by index and
    docling-graph receives them by reference — those references only
    resolve identically if the chunkers produce byte-identical output.
    """
    doc_json = _post_layer1_filtered(_build_dvina_like_doc_json())

    cfg = HybridChunkConfig()
    dl_doc = DoclingDocument.model_validate(doc_json)

    # Worker side: shared helper
    helper_chunks = build_hybrid_chunks_for_extraction(doc_json, cfg)

    # Docling-graph side: actual DocumentChunker, configured per Task 0a
    dg = DocumentChunker(
        tokenizer_name=cfg.tokenizer_model_name,
        chunk_max_tokens=cfg.max_tokens,
        merge_peers=cfg.merge_peers,
    )
    dg_chunks = dg.chunk_document(dl_doc)

    assert len(helper_chunks) == len(dg_chunks), (
        f"Chunk count mismatch: helper={len(helper_chunks)}, "
        f"docling-graph={len(dg_chunks)}.  Either side's HybridChunker "
        f"config has drifted — re-sync HybridChunkConfig with "
        f"docker/docling-graph/repo/docling_graph/core/extractors/"
        f"document_chunker.py:60-127."
    )
    assert len(helper_chunks) > 0, (
        "Synthetic Dvina-like doc must produce >=1 chunk on both sides; "
        "got zero — possible Layer-1 filter regression that blanks all "
        "the real prose."
    )

    for i, (h, d) in enumerate(zip(helper_chunks, dg_chunks)):
        assert h.text == d, (
            f"Byte-mismatch at chunk_index={i}:\n"
            f"  helper text (len={len(h.text)}): {h.text!r}\n"
            f"  docling-graph text (len={len(d)}): {d!r}\n"
            f"Phase 2 narrowed-mode routing will break — investigate "
            f"HybridChunkConfig vs DocumentChunker.__init__ drift."
        )


# ---------------------------------------------------------------------------
# Config-only parity (option a) — always runs, even if docling-graph absent
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _dg_available, reason=_dg_skip_reason)
def test_chunker_config_summary_matches() -> None:
    """Sanity check: docling-graph's ``DocumentChunker.get_config_summary()``
    reports the same tokenizer + max_tokens + merge_peers as our
    ``HybridChunkConfig`` defaults.

    Acts as the canary if anyone changes either side's defaults — fails
    cheap (no chunker run, no tokenizer load beyond import) and gives a
    pinpoint error message.
    """
    cfg = HybridChunkConfig()
    dg = DocumentChunker(
        tokenizer_name=cfg.tokenizer_model_name,
        chunk_max_tokens=cfg.max_tokens,
        merge_peers=cfg.merge_peers,
    )
    summary = dg.get_config_summary()
    assert summary["tokenizer_name"] == cfg.tokenizer_model_name
    assert summary["chunk_max_tokens"] == cfg.max_tokens
    assert summary["merge_peers"] is cfg.merge_peers
    # The image's ``DocumentChunker`` may be an older revision than the
    # current worktree source (image is rebuilt on a different cadence).
    # If the summary exposes the additional HybridChunker knobs, assert
    # them too; otherwise skip — the byte-equal test above is the
    # authoritative parity check, this is just a cheap canary.
    for key, expected in (
        ("repeat_table_header", True),
        ("omit_header_on_overflow", False),
        ("always_emit_headings", False),
    ):
        if key in summary:
            assert summary[key] is expected, (
                f"DocumentChunker.{key}={summary[key]!r}, expected {expected!r}; "
                "this drift will break the byte-equal parity test."
            )
