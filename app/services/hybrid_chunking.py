"""Shared HybridChunker helper for the extraction pipeline.

This module centralizes the HybridChunker configuration used by the
worker-side extraction indexer (``app/services/extraction_chunk_index``,
Task 3) and docling-graph's per-pass ``DocumentChunker`` (``docker/
docling-graph/repo/docling_graph/core/extractors/document_chunker.py``).

Centralization is required for Phase 2's byte-identity invariant
(merged-chunk routing plan, Task 2): the worker's vector router and
docling-graph's per-pass extractor MUST chunk the same document into
the same merged-text chunks; that's only true if the tokenizer +
HybridChunker config are identical on both sides.

The chunker setup mirrors ``document_chunker.py:60-127`` exactly —
same tokenizer wrap pattern (``AutoTokenizer.from_pretrained`` → mutate
``model_max_length`` → wrap as ``HuggingFaceTokenizer``), and the
HybridChunker uses its default ``repeat_table_header=True``,
``omit_header_on_overflow=False``, ``always_emit_headings=False``.

Note: ``HuggingFaceTokenizer.from_pretrained`` DOES exist as a
classmethod in docling_core, but is intentionally NOT used here because
``_raise_tokenizer_max_length`` needs to mutate the underlying HF
tokenizer's ``model_max_length`` directly before the
``HuggingFaceTokenizer`` wrapper caches its ``count_tokens`` budget.

Failure semantics (fail-loud):
* ``DoclingDocument.model_validate(doc_json)`` failure → propagates
  ``ValueError``. Caller (the C.4 dispatcher in Task 3) catches and
  falls back to RUN_FULL mode.
* ``HybridChunker.chunk()`` returning zero chunks → returns ``[]``
  (NOT an error). Caller's ``BuildIndexDiagnostics.chunks_inserted=0``
  signals an empty-doc / chunker-ineffective case.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument


# Large tokenizer max length used only for counting/splitting operations.
# Mirrors ``docker/docling-graph/repo/docling_graph/core/extractors/
# document_chunker.py:29``.  Suppresses HF "Token indices sequence length
# is longer than the specified maximum sequence length" warnings when the
# chunker encodes a merged chunk that legitimately exceeds chunk_max_tokens
# (HybridChunker's structure-preserving emit can produce slight overshoots).
_TOKENIZER_COUNTING_MAX_LENGTH = 1_000_000


@functools.lru_cache(maxsize=4)
def _get_tokenizer(model_name: str, max_tokens: int) -> HuggingFaceTokenizer:
    """Return a cached ``HuggingFaceTokenizer`` keyed by ``(model_name, max_tokens)``.

    Without this cache, every call to ``build_hybrid_chunks_for_extraction``
    paid the ``AutoTokenizer.from_pretrained`` cost — fine for a one-shot
    smoke test, but a hot-path perf regression once Task 3 calls this
    helper once per pipeline run.  Cold first call still hits the HF
    disk cache; subsequent calls return the SAME wrapper instance.

    ``maxsize=4`` is generous — we expect at most one or two distinct
    (model_name, max_tokens) keys in practice, but allow headroom for
    A/B testing of tokenizer overrides without evicting the production
    entry.

    The cached tokenizer goes through ``_raise_tokenizer_max_length``
    exactly once (here, on the cache miss).  Mutating ``model_max_length``
    on a shared instance is safe because the helper only ever RAISES the
    limit and never lowers it — concurrent callers see a monotonically
    growing budget.
    """
    raw_tok = AutoTokenizer.from_pretrained(model_name)
    _raise_tokenizer_max_length(raw_tok, max_tokens)
    return HuggingFaceTokenizer(tokenizer=raw_tok, max_tokens=max_tokens)


def _raise_tokenizer_max_length(
    hf_tokenizer: PreTrainedTokenizerBase, chunk_max_tokens: int
) -> None:
    """Raise the tokenizer's ``model_max_length`` so encoding long text
    for counting doesn't warn.

    Mirrors ``document_chunker.py:30-46`` exactly — same guard against
    non-int ``model_max_length`` and same "only raise, never lower"
    semantics.

    Args:
        hf_tokenizer: The raw HuggingFace tokenizer (NOT the docling-core
            ``HuggingFaceTokenizer`` wrapper).
        chunk_max_tokens: The intended chunk size in tokens.  Final
            ``model_max_length`` = ``max(chunk_max_tokens, _COUNTING_MAX)``.
    """
    current = getattr(hf_tokenizer, "model_max_length", None)
    if not isinstance(current, int):
        return
    new_max = max(chunk_max_tokens, _TOKENIZER_COUNTING_MAX_LENGTH)
    if current < new_max:
        hf_tokenizer.model_max_length = new_max


@dataclass(frozen=True)
class HybridChunkConfig:
    """Canonical HybridChunker config — mirror docling-graph's DocumentChunker.

    Sync requirement: MUST match ``docker/docling-graph/repo/docling_graph/
    core/extractors/document_chunker.py`` constructor params.  Parity is
    locked in CI by ``tests/integration/test_hybrid_chunker_parity.py``.

    Defaults are post-Task 0a (docling-graph's tokenizer_name override
    pins ``BAAI/bge-m3``).
    """

    tokenizer_model_name: str = "BAAI/bge-m3"
    max_tokens: int = 512
    merge_peers: bool = True


@dataclass(frozen=True)
class MergedChunk:
    """In-memory representation of one HybridChunker-merged chunk.

    Named ``MergedChunk`` (not ``HybridExtractionChunk``) to avoid
    namespace collision with the ArcadeDB ``ExtractionChunk`` vertex.
    ``MergedChunk`` is the value object; ``ExtractionChunk`` is the row.

    Attributes:
        chunk_index: Position of this chunk in HybridChunker output,
            dense from 0.  Becomes the ExtractionChunk vertex's
            ``chunk_index`` column in Task 3.
        text: Output of ``chunker.contextualize(chunk)``.  Includes
            doc title + section heading prefix.  This is what the LLM
            actually consumes downstream.
        source_refs: ``[item.self_ref for item in chunk.meta.doc_items]``
            — the per-element refs that contributed to this merged chunk.
            Used by ``apply_chunk_scope`` to expand the union of refs
            across selected chunks (Glossary entry ``self_refs``).
        page_no: First ``prov[0].page_no`` if resolvable; else ``None``.
        token_count: ``tokenizer.count_tokens(text)``.  Used for
            ``BuildIndexDiagnostics.mean_token_count`` in Task 3.
    """

    chunk_index: int
    text: str
    source_refs: list[str]
    page_no: str | None
    token_count: int


def build_hybrid_chunks_for_extraction(
    doc_json: dict,
    config: HybridChunkConfig | None = None,
) -> list[MergedChunk]:
    """Run HybridChunker against ``doc_json`` using the canonical config.

    CALLER CONTRACT:
        ``doc_json`` MUST be the post-Layer-1-filter shape.  Doc-shape
        parity with docling-graph (after Task 0b's sanitize-skip) is
        asserted by the parity test, not by this helper.

    Failure semantics (fail-loud, mirrors ``build_extraction_index``
    strict mode):
        * ``DoclingDocument.model_validate(doc_json)`` failure →
          ``ValueError`` raised (caller's try/except converts to
          RUN_FULL fallback per the C.4 wrapper).
        * ``HybridChunker.chunk()`` returning zero chunks → returns
          ``[]`` (NOT an error).

    Args:
        doc_json: The exported DoclingDocument as a dict.
        config: Optional override.  Defaults to ``HybridChunkConfig()``
            (BAAI/bge-m3, max_tokens=512, merge_peers=True).

    Returns:
        Ordered list of ``MergedChunk``; ``chunk_index`` is dense from 0.
    """
    cfg = config or HybridChunkConfig()

    # Cached per (model_name, max_tokens) — see ``_get_tokenizer`` docstring.
    # The wrap-via-AutoTokenizer pattern is preserved; only the per-call
    # ``from_pretrained`` + wrap cost is eliminated.
    tokenizer = _get_tokenizer(cfg.tokenizer_model_name, cfg.max_tokens)

    # HybridChunker takes its token budget from ``tokenizer.get_max_tokens()``
    # — no ``chunk_max_tokens`` kwarg.  ``repeat_table_header``,
    # ``omit_header_on_overflow``, ``always_emit_headings`` retain their
    # docling-graph-equivalent defaults (True / False / False).
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=cfg.merge_peers,
    )

    # ``model_validate`` may raise pydantic ``ValidationError`` (a
    # ``ValueError`` subclass) on bad shape — propagate to caller.
    dl_doc = DoclingDocument.model_validate(doc_json)

    out: list[MergedChunk] = []
    for idx, chunk in enumerate(chunker.chunk(dl_doc=dl_doc)):
        text = chunker.contextualize(chunk=chunk)
        source_refs = [item.self_ref for item in chunk.meta.doc_items]
        page_no = _resolve_first_page_no(chunk)
        token_count = tokenizer.count_tokens(text=text)
        out.append(
            MergedChunk(
                chunk_index=idx,
                text=text,
                source_refs=source_refs,
                page_no=page_no,
                token_count=token_count,
            )
        )
    return out


def _resolve_first_page_no(chunk: Any) -> str | None:
    """Walk ``chunk.meta.doc_items[0].prov[0].page_no``.

    Returns the integer page number as a string (matches the existing
    ``ExtractionChunk.page_no`` column type), or ``None`` if any link
    in the chain is missing or the page number is not an int.
    """
    items = getattr(chunk.meta, "doc_items", None) or []
    if not items:
        return None
    prov = getattr(items[0], "prov", None) or []
    if not prov:
        return None
    page_no = getattr(prov[0], "page_no", None)
    return str(page_no) if isinstance(page_no, int) else None
