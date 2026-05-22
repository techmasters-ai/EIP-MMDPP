"""ExtractionChunk index build + cleanup helpers (VR Phase C.2).

build_extraction_index(doc_json, pipeline_run_id, document_id, store, ...):
    Walk the docling DoclingDocument JSON, extract one chunk per
    text/table/picture-caption element, embed via embed_texts(), and bulk-
    INSERT ExtractionChunk vertices into ArcadeDB. Idempotent: deletes any
    existing chunks for the pipeline_run_id before inserting.

cleanup_extraction_index(pipeline_run_id, store):
    DELETE FROM ExtractionChunk WHERE pipeline_run_id=X. Best-effort: logs
    WARNING on failure (does NOT raise).

CHUNK RENDERING CONTRACT (rev 8 M9 + rev 10):
    - text elements: raw ``text`` field (section_header / title labels are
      NOT indexed as chunks — they are used as parent-heading context only)
    - table elements: caption + markdown-rendered cells, joined as:
          {caption}\\n\\n| col1 | col2 |\\n|---|---|\\n| cell | cell |
      (caption omitted when missing or when include_table_caption=False)
    - picture elements: caption text only (NO inferred description; no OCR)
    - context expansion (defaults): prepend parent-section heading to text
      elements; include table caption (default in table render)
    - elements with empty text after rendering (e.g. sanitized-blank or
      no-caption pictures) are SKIPPED (no embedding, no insert)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from app.services.arcadedb_graph import ArcadeDBGraphStore

logger = logging.getLogger(__name__)

# Lazy import — pulled at module scope after settings are available.
# Tests patch this name directly via:
#   patch("app.services.extraction_chunk_index.embed_texts", ...)
try:
    from app.services.embedding import embed_texts  # noqa: F401
except Exception:  # pragma: no cover
    embed_texts = None  # type: ignore[assignment]

# Labels that are treated as section headings (for parent context expansion).
# These are NOT indexed as extraction chunks themselves — they provide context
# to adjacent text elements via include_parent_section_heading expansion.
_HEADING_LABELS = frozenset({"section_header", "section-header", "title"})

# NOTE: walker policy (rev 14 code-quality review Important #1):
#   _walk_docling_elements indexes EVERY texts[]/tables[]/pictures[] element
#   EXCEPT those whose label is in _HEADING_LABELS. There is no positive
#   allowlist — a future docling version that introduces a new label will be
#   indexed by default. If we ever need explicit allowlisting, reintroduce a
#   _TEXT_CHUNK_LABELS frozenset HERE and wire it into the walk filter.


# ---------------------------------------------------------------------------
# Diagnostics dataclass
# ---------------------------------------------------------------------------


@dataclass
class BuildIndexDiagnostics:
    """Diagnostics from a build_extraction_index() call.

    Ready to merge into the VR router diagnostics block (rev 8 M10) at C.4
    worker-wiring time.

    Fields
    ------
    chunks_inserted:
        Number of ExtractionChunk vertices successfully inserted.
    chunks_skipped:
        Number of elements skipped because their rendered text was empty
        (blank text elements, no-caption pictures, etc.).
    embed_calls:
        Number of calls made to embed_texts(). Typically 1 (single batch).
    embed_ms:
        Wall-clock milliseconds spent in embed_texts() calls.
    insert_ms:
        Wall-clock milliseconds spent in ArcadeDB INSERT calls.
    modality_counts:
        Per-modality breakdown: {"text": N, "table": M, "picture_caption": K}.
    """

    chunks_inserted: int
    chunks_skipped: int
    embed_calls: int
    embed_ms: int
    insert_ms: int
    modality_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Private rendering helpers
# ---------------------------------------------------------------------------


def _resolve_caption_ref(ref_item: dict, doc_json: dict) -> str:
    """Resolve a caption ref dict (``{"$ref": "#/texts/N"}``) to its text.

    Strategy:
      1. Parse the index N from the $ref path.
      2. Try doc_json["texts"][N] first (O(1) array access).
      3. If that element has a different self_ref (or the array is smaller),
         fall back to a linear self_ref scan of the texts array.

    Returns empty string on any resolution failure.
    """
    # Standard docling JSON uses "$ref"; fallbacks ("cref"/"$cref") only catch
    # non-standard test fixtures. Both names are tried for parity with
    # _resolve_parent_section_heading (rev 14 review Minor #3).
    cref = ref_item.get("$ref") or ref_item.get("$cref") or ref_item.get("cref", "")
    if not cref or not cref.startswith("#/texts/"):
        return ""

    texts = doc_json.get("texts") or []

    # Try direct array index first
    try:
        idx = int(cref.split("/")[-1])
        if 0 <= idx < len(texts):
            elem = texts[idx]
            if isinstance(elem, dict):
                # Verify the element's own self_ref matches (guards against
                # sparse arrays in synthetic test docs)
                elem_ref = elem.get("self_ref") or f"#/texts/{idx}"
                if elem_ref == cref:
                    return (elem.get("text") or "").strip()
    except (ValueError, IndexError, TypeError):
        pass

    # Fallback: linear scan by self_ref (handles synthetic / sparse test data)
    for elem in texts:
        if not isinstance(elem, dict):
            continue
        if elem.get("self_ref") == cref:
            return (elem.get("text") or "").strip()

    return ""


def _resolve_first_caption(elem: dict, doc_json: dict) -> str:
    """Return the first non-empty caption text for a table or picture element.

    Captions are stored as a list of ref-dicts (``{"$ref": "#/texts/N"}``)
    pointing into doc_json["texts"]. Returns empty string when absent.
    """
    for cap in (elem.get("captions") or []):
        if isinstance(cap, dict):
            text = _resolve_caption_ref(cap, doc_json)
            if text:
                return text
        elif isinstance(cap, str) and cap.strip():
            return cap.strip()
    return ""


def _resolve_parent_section_heading(elem: dict, doc_json: dict) -> str:
    """Return the section heading text of this element's parent, if any.

    The parent ref is a dict ``{"$ref": "#/texts/N"}``. We look up that
    text element and return its text only if its label is a heading label.
    Returns empty string when parent is absent, unresolvable, or non-heading.
    """
    parent = elem.get("parent")
    if not isinstance(parent, dict):
        return ""
    ref = parent.get("$ref") or parent.get("$cref", "")
    if not ref or not ref.startswith("#/texts/"):
        return ""
    try:
        idx = int(ref.split("/")[-1])
        texts = doc_json.get("texts") or []
        if 0 <= idx < len(texts):
            parent_elem = texts[idx]
            label = (parent_elem.get("label") or "").lower().replace(" ", "_")
            if label in _HEADING_LABELS:
                return (parent_elem.get("text") or "").strip()
    except (ValueError, IndexError, TypeError):
        pass
    return ""


def _render_text_chunk(
    elem: dict,
    doc_json: dict,
    *,
    include_parent_section_heading: bool,
) -> str:
    """Return rendered chunk text for a docling text element.

    Rev 8 M9 contract:
      - raw ``text`` field
      - optionally prepend parent section heading (default on)

    Returns empty string if the element has no text after sanitization.
    """
    raw_text = (elem.get("text") or "").strip()
    if not raw_text:
        return ""

    if include_parent_section_heading:
        heading = _resolve_parent_section_heading(elem, doc_json)
        if heading and heading != raw_text:
            return f"{heading}\n\n{raw_text}"

    return raw_text


def _render_table_chunk(
    elem: dict,
    doc_json: dict,
    *,
    include_caption: bool,
) -> str:
    """Return markdown rendering of a docling table element.

    Rev 8 M9 contract:
      - caption + markdown-rendered cells (caption omitted when missing or
        when include_caption=False)
      - ``{caption}\\n\\n| col1 | col2 |\\n|---|---|\\n| cell | cell |``

    Returns empty string if the table has no caption AND no cells.
    """
    parts: list[str] = []

    # --- Caption ---
    if include_caption:
        caption = _resolve_first_caption(elem, doc_json)
        if caption:
            parts.append(caption)

    # --- Cells → markdown table ---
    cells = (elem.get("data") or {}).get("table_cells") or elem.get("table_cells") or []
    if cells:
        # Determine grid dimensions
        max_row = max((c.get("end_row_offset_idx", 0) for c in cells), default=0)
        max_col = max((c.get("end_col_offset_idx", 0) for c in cells), default=0)
        n_rows = max_row + 1
        n_cols = max_col + 1

        # Build 2-D grid (last writer wins for spans)
        grid: list[list[str]] = [[""] * n_cols for _ in range(n_rows)]
        for cell in cells:
            r = cell.get("start_row_offset_idx", 0)
            c = cell.get("start_col_offset_idx", 0)
            if 0 <= r < n_rows and 0 <= c < n_cols:
                grid[r][c] = (cell.get("text") or "").strip()

        if n_rows > 0 and n_cols > 0:
            # Header row
            header = "| " + " | ".join(grid[0]) + " |"
            separator = "| " + " | ".join(["---"] * n_cols) + " |"
            rows_md = [header, separator]
            for row in grid[1:]:
                rows_md.append("| " + " | ".join(row) + " |")
            parts.append("\n".join(rows_md))

    if not parts:
        return ""
    return "\n\n".join(parts)


def _render_picture_chunk(elem: dict, doc_json: dict) -> str:
    """Return picture caption text only.

    Rev 8 M9 contract:
      - caption text ONLY (NO inferred description, NO OCR fallback)

    Returns empty string if no caption is present.
    """
    return _resolve_first_caption(elem, doc_json)


# ---------------------------------------------------------------------------
# Element walker
# ---------------------------------------------------------------------------


def _walk_docling_elements(
    doc_json: dict,
) -> Iterator[tuple[str, str, dict]]:
    """Yield (self_ref, modality, element_dict) for every indexable element.

    Modalities:
      - "text" for doc_json["texts"] entries (excluding headings/titles,
        which are skipped as indexable chunks — they serve only as parent
        context for neighboring text elements)
      - "table" for doc_json["tables"] entries
      - "picture_caption" for doc_json["pictures"] entries

    self_ref follows the canonical docling format:
      '#/texts/N', '#/tables/N', '#/pictures/N'
    """
    # --- Text elements ---
    for idx, elem in enumerate(doc_json.get("texts") or []):
        if not isinstance(elem, dict):
            continue
        label = (elem.get("label") or "").lower().replace(" ", "_")
        # Skip heading/title elements — they are parent-context providers,
        # not independently useful retrieval targets.
        if label in _HEADING_LABELS:
            continue
        # Use the element's own self_ref if present (canonical docling form),
        # otherwise synthesize from the array index.
        self_ref = elem.get("self_ref") or f"#/texts/{idx}"
        yield self_ref, "text", elem

    # --- Table elements ---
    for idx, elem in enumerate(doc_json.get("tables") or []):
        if not isinstance(elem, dict):
            continue
        self_ref = elem.get("self_ref") or f"#/tables/{idx}"
        yield self_ref, "table", elem

    # --- Picture elements ---
    for idx, elem in enumerate(doc_json.get("pictures") or []):
        if not isinstance(elem, dict):
            continue
        self_ref = elem.get("self_ref") or f"#/pictures/{idx}"
        yield self_ref, "picture_caption", elem


# ---------------------------------------------------------------------------
# Public API: build
# ---------------------------------------------------------------------------


def build_extraction_index(
    doc_json: dict,
    pipeline_run_id: str,
    document_id: str,
    *,
    store: "ArcadeDBGraphStore",
    include_parent_section_heading: bool = True,
    include_table_caption: bool = True,
) -> BuildIndexDiagnostics:
    """Walk docling JSON, embed, and bulk-INSERT ExtractionChunk vertices.

    Idempotent: deletes any existing chunks for pipeline_run_id before
    inserting, so re-running for the same run produces the same final state.

    Parameters
    ----------
    doc_json:
        JSON-parsed DoclingDocument dict (as returned by
        ``_build_docling_document_json``). NOT a pydantic DoclingDocument
        instance.
    pipeline_run_id:
        The pipeline run UUID. Used as the primary filter dimension for VR
        vector queries and the janitor age-sweep.
    document_id:
        The document UUID. Stored on each vertex for provenance.
    store:
        ArcadeDBGraphStore instance (sync path via ``_client.command_sync``).
    include_parent_section_heading:
        When True (default), text element chunks are prefixed with their
        parent section heading (rev 8 M9 context expansion).
    include_table_caption:
        When True (default), table chunks include the caption above the
        markdown-rendered cells (rev 8 M9 context expansion).

    Returns
    -------
    BuildIndexDiagnostics
        Counters ready to merge into the rev 8 M10 router diagnostics block
        at C.4 worker-wiring time.
    """
    # Use the module-level embed_texts (imported at top; tests patch that name).
    _embed = embed_texts

    # ------------------------------------------------------------------
    # Step 1: Idempotent delete — clear any existing chunks for this run.
    # ------------------------------------------------------------------
    # NOTE (rev 14 code-quality review Important #2 + #3): failure modes
    # of this DELETE and the subsequent batch INSERT loop are documented
    # here as the caller contract for VR Phase C.4:
    #
    #   1. _delete_by_run_id silently returns 0 on failure (shared with
    #      cleanup_extraction_index, which deliberately swallows exceptions).
    #      If the DELETE fails (network blip, ArcadeDB restart), the
    #      subsequent INSERT loop may collide with existing rows on the
    #      UNIQUE vertex_id index.
    #   2. The INSERT loop below is NOT transaction-wrapped — each vertex
    #      is its own auto-committed `command_sync` call. A failure at
    #      vertex N leaves N-1 chunks in the database.
    #
    # Both failure modes are RECOVERABLE BY RETRY: the next call to
    # build_extraction_index begins with a fresh DELETE, which idempotently
    # clears whatever state was left behind.
    #
    # CALLER CONTRACT (C.4 dispatcher wiring): wrap build_extraction_index()
    # in try/except inside derive_ontology_graph. On ANY exception, log
    # WARNING and fall back to RUN_FULL for all field-group passes — do
    # NOT call the /v1/extraction/chunk-scope endpoint against a partial
    # index. This satisfies the rev 12 H2 + rev 13 fail-open requirement.
    _delete_by_run_id(store, pipeline_run_id)

    # ------------------------------------------------------------------
    # Step 2: Walk elements and render chunk text.
    # ------------------------------------------------------------------
    pending: list[tuple[str, str, str | None, str]] = []
    # Each tuple: (self_ref, modality, page_number_or_None, rendered_text)

    chunks_skipped = 0
    modality_counts: dict[str, int] = {}

    for self_ref, modality, elem in _walk_docling_elements(doc_json):
        # Render
        if modality == "text":
            rendered = _render_text_chunk(
                elem, doc_json,
                include_parent_section_heading=include_parent_section_heading,
            )
        elif modality == "table":
            rendered = _render_table_chunk(
                elem, doc_json,
                include_caption=include_table_caption,
            )
        else:  # picture_caption
            rendered = _render_picture_chunk(elem, doc_json)

        if not rendered.strip():
            chunks_skipped += 1
            continue

        # Page number (first prov entry)
        page_no: str | None = None
        prov_list = elem.get("prov") or []
        if prov_list and isinstance(prov_list[0], dict):
            pn = prov_list[0].get("page_no")
            if isinstance(pn, int):
                page_no = str(pn)

        pending.append((self_ref, modality, page_no, rendered))
        modality_counts[modality] = modality_counts.get(modality, 0) + 1

    if not pending:
        return BuildIndexDiagnostics(
            chunks_inserted=0,
            chunks_skipped=chunks_skipped,
            embed_calls=0,
            embed_ms=0,
            insert_ms=0,
            modality_counts=modality_counts,
        )

    # ------------------------------------------------------------------
    # Step 3: Embed all non-empty texts as a SINGLE batch.
    # ------------------------------------------------------------------
    rendered_texts = [p[3] for p in pending]

    embed_t0 = time.monotonic()
    embeddings = _embed(rendered_texts, query=False)
    embed_ms = int((time.monotonic() - embed_t0) * 1000)
    # embed_calls counts INVOCATIONS of embed_texts() from this builder,
    # NOT the number of Ollama HTTP requests. embed_texts() internally
    # chunks the input list into sub-batches (default batch_size=64), so a
    # single embed_calls=1 here may translate to multiple Ollama calls.
    # (rev 14 code-quality review Minor #6.)
    embed_calls = 1

    # ------------------------------------------------------------------
    # Step 4: Bulk INSERT ExtractionChunk vertices.
    # ------------------------------------------------------------------
    _INSERT_SQL = (
        "INSERT INTO ExtractionChunk SET "
        "vertex_id = :vertex_id, "
        "pipeline_run_id = :pipeline_run_id, "
        "document_id = :document_id, "
        "self_ref = :self_ref, "
        "chunk_text = :chunk_text, "
        "embedding = :embedding, "
        "page_number = :page_number, "
        "modality = :modality, "
        "created_at = sysdate()"
    )

    insert_t0 = time.monotonic()
    for (self_ref, modality, page_no, rendered), embedding in zip(pending, embeddings):
        # vertex_id assumes pipeline_run_id contains no ":" — production
        # run_ids are UUIDs (e.g. 550e8400-e29b-41d4-a716-446655440000) which
        # never contain ":". self_ref contains "/" and "#" but no ":". So the
        # composite key is unambiguous. (rev 14 code-quality review Minor #5.)
        vertex_id = f"{pipeline_run_id}:{self_ref}"
        params: dict = {
            "vertex_id": vertex_id,
            "pipeline_run_id": pipeline_run_id,
            "document_id": document_id,
            "self_ref": self_ref,
            "chunk_text": rendered,
            "embedding": embedding,
            "page_number": int(page_no) if page_no is not None else None,
            "modality": modality,
        }
        store._client.command_sync(
            store._database,
            "sql",
            _INSERT_SQL,
            params,
        )
    insert_ms = int((time.monotonic() - insert_t0) * 1000)

    chunks_inserted = len(pending)

    logger.info(
        "build_extraction_index: pipeline_run_id=%r document_id=%r "
        "inserted=%d skipped=%d embed_ms=%d insert_ms=%d modalities=%s",
        pipeline_run_id, document_id,
        chunks_inserted, chunks_skipped,
        embed_ms, insert_ms, modality_counts,
    )

    return BuildIndexDiagnostics(
        chunks_inserted=chunks_inserted,
        chunks_skipped=chunks_skipped,
        embed_calls=embed_calls,
        embed_ms=embed_ms,
        insert_ms=insert_ms,
        modality_counts=modality_counts,
    )


# ---------------------------------------------------------------------------
# Public API: cleanup
# ---------------------------------------------------------------------------

_DELETE_SQL = (
    "DELETE FROM ExtractionChunk WHERE pipeline_run_id = :run_id"
)


def _delete_by_run_id(store: "ArcadeDBGraphStore", pipeline_run_id: str) -> int:
    """Delete all ExtractionChunk rows for a given pipeline_run_id.

    Returns the count of deleted rows, or 0 on error (callers treat this as
    best-effort).
    """
    try:
        result = store._client.command_sync(
            store._database,
            "sql",
            _DELETE_SQL,
            {"run_id": pipeline_run_id},
        )
        if result and isinstance(result, list) and isinstance(result[0], dict):
            return int(result[0].get("count", 0))
        return 0
    except Exception:
        return 0


def cleanup_extraction_index(
    pipeline_run_id: str,
    *,
    store: "ArcadeDBGraphStore",
) -> int:
    """Delete all ExtractionChunk rows for a pipeline_run_id.

    Best-effort: logs WARNING on failure and returns 0 instead of raising.
    The hourly janitor task (C.4) provides defense-in-depth.

    Parameters
    ----------
    pipeline_run_id:
        The pipeline run UUID whose ExtractionChunk rows should be deleted.
    store:
        ArcadeDBGraphStore instance (sync path via ``_client.command_sync``).

    Returns
    -------
    int
        Number of rows deleted, or 0 on failure.
    """
    try:
        result = store._client.command_sync(
            store._database,
            "sql",
            _DELETE_SQL,
            {"run_id": pipeline_run_id},
        )
        if result and isinstance(result, list) and isinstance(result[0], dict):
            count = int(result[0].get("count", 0))
            logger.debug(
                "cleanup_extraction_index: pipeline_run_id=%r deleted=%d",
                pipeline_run_id, count,
            )
            return count
        return 0
    except Exception as exc:
        logger.warning(
            "cleanup_extraction_index: failed to delete ExtractionChunk rows "
            "for pipeline_run_id=%r — %s: %s. Janitor will retry.",
            pipeline_run_id,
            type(exc).__name__,
            exc,
        )
        return 0
