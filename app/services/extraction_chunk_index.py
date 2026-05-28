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

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

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

# ---------------------------------------------------------------------------
# C.10: v2 quality-filter primitives moved to app.services.chunk_quality.
# Re-exported with leading-underscore aliases for internal call-site stability.
# ---------------------------------------------------------------------------
from app.services.chunk_quality import (
    MIN_CHUNK_TEXT_CHARS as _MIN_CHUNK_TEXT_CHARS,
    MIN_RESIDUAL_CHARS as _MIN_RESIDUAL_CHARS,
    normalize_for_dedup as _normalize_for_dedup,
    strip_chrome_lines as _strip_chrome_lines,
)

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
    # C.9d / C.9d-v2: quality-filter counts (subset of chunks_skipped).
    #
    #   chunks_skipped_short:        before-strip; rendered text < _MIN_CHUNK_TEXT_CHARS.
    #   chunks_stripped_web_chrome:  v2 — chunk was KEPT but leading/trailing
    #                                chrome lines were stripped before insert.
    #   chunks_skipped_after_strip:  v2 — chunk DROPPED because residue < _MIN_RESIDUAL_CHARS.
    #   chunks_skipped_duplicate:    v2 — dedup operates on POST-STRIP normalized text
    #                                (so different chrome prefixes collapse to the same key).
    #   chunks_skipped_web_chrome:   DEPRECATED — v1 conservative drop. Always 0 in v2.
    chunks_skipped_short: int = 0
    chunks_skipped_duplicate: int = 0
    chunks_skipped_web_chrome: int = 0
    chunks_stripped_web_chrome: int = 0
    chunks_skipped_after_strip: int = 0


# ---------------------------------------------------------------------------
# Merged-chunk row accessors (Phase 1 Task 1 — merged-chunk routing)
# ---------------------------------------------------------------------------
#
# These three helpers provide None-coalescing reads for the merged-mode
# columns added to ``ExtractionChunk``. They MUST be used in place of
# direct ``row["chunk_index"]`` / ``row["source_refs"]`` /
# ``row["token_count"]`` reads, because legacy per-element rows ingested
# before this change have those columns as ``NULL`` (or absent entirely
# if the row was projected from a SELECT that pre-dates Task 5).
#
# Carve-out (per the plan's "belt-and-suspenders rule"): columns that
# have always been mandatory on ``ExtractionChunk`` (``chunk_text``,
# ``embedding``, ``page_number``, ``modality``, ``pipeline_run_id``,
# ``self_ref``) are NEVER ``None`` on any row, legacy or merged — direct
# dict access is allowed for those. The accessors below are only for the
# three legacy-coalesced columns.


def _row_get(row: dict | Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a row that may be a dict or an object with attributes.

    ArcadeDB HTTP responses are dicts, but some test scaffolds and lower-level
    ORM layers expose rows as objects with attribute access. Both paths must
    work without raising.
    """
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def read_chunk_source_refs(row: dict | Any) -> list[str]:
    """Return ``source_refs`` as ``list[str]`` regardless of storage shape.

    Handles three storage paths:

    1. native ArcadeDB ``LIST`` property (``row["source_refs"]``)
    2. JSON-encoded string fallback (``row["source_refs_json"]``)
    3. legacy / missing → ``[]``

    NEVER returns ``None``. An empty list means "no constituent refs known"
    — the caller decides whether that's an error or a legacy row.
    """
    raw = _row_get(row, "source_refs", None)
    if isinstance(raw, list):
        # M3: filter Nones so a stray null element doesn't become the literal
        # string "None" — that's almost certainly worse than dropping it.
        return [str(x) for x in raw if x is not None]
    if raw is not None:
        # Defensive: not a list, not None — try to coerce or fall through.
        if isinstance(raw, str):
            try:
                decoded = json.loads(raw)
                if isinstance(decoded, list):
                    # M3: filter Nones in the JSON-string-in-source_refs path too.
                    return [str(x) for x in decoded if x is not None]
            except (ValueError, TypeError):
                pass
        else:
            # M2: raw is non-None and not a list/string (e.g. an int from a
            # buggy upstream insert). Return [] (safe), but log a WARNING so
            # the upstream data-shape bug is OBSERVABLE — silently coercing
            # to [] masks the corruption.
            logger.warning(
                "read_chunk_source_refs: source_refs is unexpected type %s "
                "for row; coercing to []",
                type(raw).__name__,
            )
    # JSON-string fallback path
    json_raw = _row_get(row, "source_refs_json", None)
    if isinstance(json_raw, str):
        try:
            decoded = json.loads(json_raw)
            if isinstance(decoded, list):
                # M3: filter Nones from the JSON fallback path.
                return [str(x) for x in decoded if x is not None]
        except (ValueError, TypeError):
            return []
    return []


def read_chunk_token_count(row: dict | Any) -> int:
    """Return ``token_count`` as ``int``. Legacy / missing / ``None`` → ``0``.

    Coerces stringified ints (``"42"``) since ArcadeDB HTTP serialisation
    occasionally returns numeric columns as JSON strings. Garbage → ``0``.
    """
    raw = _row_get(row, "token_count", None)
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def read_chunk_index(row: dict | Any) -> int:
    """Return ``chunk_index`` as ``int``. Legacy / missing / ``None`` → ``-1``.

    Note: ``chunk_index == 0`` is a valid merged-chunk position and must NOT
    be treated as a legacy marker — the function only coalesces ``None`` /
    absent / un-coercible values, not falsy values in general.
    """
    raw = _row_get(row, "chunk_index", None)
    if raw is None:
        return -1
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


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


def _resolve_parent_section_heading(self_ref_or_elem, doc_json: dict) -> str:
    """Find the most recent section_header/title BEFORE this element in
    body.children walk order, recursively walking into #/groups/N children.
    Returns its text, or empty string if no heading precedes it.

    Algorithm:
      1. Walk doc_json["body"]["children"] in document order.
      2. When a child cref points at "#/texts/N", resolve the element:
           - If its label is in _HEADING_LABELS, update ``current_heading``.
           - If it IS the target, return current_heading.
      3. When a child cref points at "#/groups/N", recursively walk
         groups[N].children with the same heading state.  Heading state
         persists ACROSS group boundaries in both directions:
           - A heading set in body.children BEFORE a group ref applies to
             all texts inside that group.
           - If a heading appears INSIDE a group, it updates the running
             heading for subsequent siblings and parent-level elements.
      4. Other crefs (#/tables/N, #/pictures/N) are skipped — they carry
         no heading content.
      5. If the target self_ref is not found anywhere in the walk, return "".

    Defensive guards:
      - Recursion depth is capped at ``_MAX_GROUP_DEPTH`` (10) to handle
        pathological deeply-nested groups without stack overflow.
      - Visited group indices are tracked to prevent infinite cycles.

    The self_ref_or_elem argument may be either:
      - A string self_ref (e.g. ``"#/texts/3"``), OR
      - A dict element with a ``"self_ref"`` key.

    Supports both "cref" and "$ref" as the ref key for robustness across
    docling serialisation variants.
    """
    # Accept either a self_ref string or a dict element
    if isinstance(self_ref_or_elem, dict):
        target_self_ref = self_ref_or_elem.get("self_ref", "")
    else:
        target_self_ref = str(self_ref_or_elem)

    if not target_self_ref:
        return ""

    body = doc_json.get("body")
    if not isinstance(body, dict):
        return ""

    texts = doc_json.get("texts") or []
    groups = doc_json.get("groups") or []

    # Build a fast lookup: self_ref → texts element (O(n) once)
    texts_by_ref: dict[str, dict] = {}
    for idx, t in enumerate(texts):
        if not isinstance(t, dict):
            continue
        ref = t.get("self_ref") or f"#/texts/{idx}"
        texts_by_ref[ref] = t

    # Mutable walk state shared across the recursive helper.
    # Using a dict so the nested function can mutate without nonlocal in
    # Python 2-compatible style (and for clarity).
    state: dict = {"heading": "", "found": False, "result": ""}

    _MAX_GROUP_DEPTH = 10

    def _walk(refs: list, depth: int = 0) -> None:
        """Walk a list of ref-dicts, updating state in place.

        Stops as soon as state["found"] is True (short-circuit once target
        is located).  Groups are recursed into up to _MAX_GROUP_DEPTH levels.
        """
        if depth > _MAX_GROUP_DEPTH:
            logger.warning(
                "_resolve_parent_section_heading: group nesting depth %d "
                "exceeds maximum %d — aborting deep walk (groups below this "
                "level will not be searched for heading context).",
                depth, _MAX_GROUP_DEPTH,
            )
            return

        for ref in refs:
            if state["found"]:
                return
            if not isinstance(ref, dict):
                continue

            # Support "cref" (real docling) and "$ref" / "$cref" (synthetic)
            cref = ref.get("cref") or ref.get("$ref") or ref.get("$cref", "")
            if not cref:
                continue

            if cref.startswith("#/texts/"):
                # Resolve to texts element — prefer fast lookup, fall back to index
                elem = texts_by_ref.get(cref)
                if elem is None:
                    try:
                        idx = int(cref.split("/")[-1])
                        if 0 <= idx < len(texts):
                            elem = texts[idx]
                    except (ValueError, IndexError, TypeError):
                        pass

                if elem is None or not isinstance(elem, dict):
                    continue

                # Use elem's own self_ref for the equality check; fall back to cref
                elem_ref = elem.get("self_ref") or cref

                # Is this the target element?
                if elem_ref == target_self_ref or cref == target_self_ref:
                    state["found"] = True
                    state["result"] = state["heading"]
                    return

                # Not the target — update heading state if applicable
                label = (elem.get("label") or "").lower().replace(" ", "_")
                if label in _HEADING_LABELS:
                    state["heading"] = (elem.get("text") or "").strip()

            elif cref.startswith("#/groups/"):
                try:
                    idx = int(cref.split("/")[-1])
                except (ValueError, TypeError):
                    continue
                if idx < 0 or idx >= len(groups):
                    continue
                grp = groups[idx]
                if not isinstance(grp, dict):
                    continue
                grp_children = grp.get("children") or []
                # Recurse; heading state and found flag persist through the call
                _walk(grp_children, depth + 1)

            # Other ref types (#/tables/N, #/pictures/N) carry no heading content
            # and are intentionally skipped.

    _walk(body.get("children") or [])
    return state["result"]


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
    data = elem.get("data") or {}
    cells = data.get("table_cells") or elem.get("table_cells") or []
    if cells:
        # Use data.num_rows / data.num_cols as the authoritative grid size.
        # These are set by docling and reflect the true table dimensions.
        # Do NOT derive n_rows/n_cols from max(end_*_offset_idx) — docling uses
        # EXCLUSIVE (Python-slice) end indices, so end=1 means occupies only
        # index 0, and max(end)+1 would produce an off-by-one over-sized grid.
        n_rows = int(data.get("num_rows") or 0)
        n_cols = int(data.get("num_cols") or 0)

        # Fallback: if num_rows/num_cols absent, derive from start indices only
        # (still correct because start is always inclusive).
        if n_rows == 0 or n_cols == 0:
            n_rows = max((c.get("start_row_offset_idx", 0) + 1 for c in cells), default=0)
            n_cols = max((c.get("start_col_offset_idx", 0) + 1 for c in cells), default=0)

        # Build 2-D grid (last writer wins for spans).
        # Cell placement uses range(start, end) — Python EXCLUSIVE slice semantics.
        # A cell with start_row=0, end_row=1 occupies ONLY row 0.
        grid: list[list[str]] = [[""] * n_cols for _ in range(n_rows)]
        for cell in cells:
            r_start = cell.get("start_row_offset_idx", 0)
            r_end = cell.get("end_row_offset_idx", r_start + 1)
            c_start = cell.get("start_col_offset_idx", 0)
            c_end = cell.get("end_col_offset_idx", c_start + 1)
            cell_text = (cell.get("text") or "").strip()
            for r in range(r_start, min(r_end, n_rows)):
                for c in range(c_start, min(c_end, n_cols)):
                    grid[r][c] = cell_text

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
    #
    # strict=True: DELETE failures propagate to caller (C.4 try/except).
    # If DELETE fails and we proceed to INSERT, we risk UNIQUE constraint
    # violations on vertex_id that produce noisy partial-insert errors and
    # break the idempotency contract. Raising here lets C.4 fall back cleanly.
    _delete_by_run_id(store, pipeline_run_id, strict=True)

    # ------------------------------------------------------------------
    # Step 2: Walk elements and render chunk text.
    # ------------------------------------------------------------------
    pending: list[tuple[str, str, str | None, str]] = []
    # Each tuple: (self_ref, modality, page_number_or_None, rendered_text)

    chunks_skipped = 0
    chunks_skipped_short = 0
    chunks_skipped_duplicate = 0
    chunks_stripped_web_chrome = 0
    chunks_skipped_after_strip = 0
    seen_norms: set[str] = set()
    # C.9d-v2: per-category examples. Each entry is a dict with self_ref,
    # original (first 120 chars), stripped (first 120 chars), reason.
    examples_leading_chrome: list[dict] = []
    examples_trailing_chrome: list[dict] = []
    examples_residual_too_short: list[dict] = []
    examples_duplicate_after_strip: list[dict] = []
    short_examples: list[str] = []
    modality_counts: dict[str, int] = {}

    def _add_example(bucket: list[dict], self_ref: str, original: str, stripped: str, reason: str) -> None:
        if len(bucket) < 5:
            bucket.append({
                "self_ref": self_ref,
                "original": original[:120],
                "stripped": stripped[:120],
                "reason": reason,
            })

    for self_ref, modality, elem in _walk_docling_elements(doc_json):
        # 1. Render
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

        # 2. Quick reject — obviously useless before bothering with strip.
        #    Catches "™", "11", single nav fragments.
        rendered_normalized = _normalize_for_dedup(rendered)
        if len(rendered_normalized) < _MIN_CHUNK_TEXT_CHARS:
            chunks_skipped += 1
            chunks_skipped_short += 1
            if len(short_examples) < 5:
                short_examples.append(rendered[:60])
            continue

        # 3. Strip leading/trailing chrome lines. Middle lines preserved.
        stripped, leading_removed, trailing_removed = _strip_chrome_lines(rendered)
        had_chrome_stripped = (leading_removed > 0) or (trailing_removed > 0)

        # 4. Normalize stripped for residue check + dedup key.
        stripped_normalized = _normalize_for_dedup(stripped)

        # 5. Drop if residue too short — only when chrome was actually
        #    stripped. Short non-chrome content (e.g. spec-table fragments
        #    like "Length:" rendered with a short heading prefix) is preserved
        #    here so its self_ref enters the index and can be picked up by the
        #    vector router for narrowed passes. Mirrors the Layer-1 gate in
        #    filter_docling_document (gate_after_strip_on_chrome=True).
        if had_chrome_stripped and len(stripped_normalized) < _MIN_RESIDUAL_CHARS:
            chunks_skipped += 1
            chunks_skipped_after_strip += 1
            _add_example(
                examples_residual_too_short, self_ref, rendered, stripped,
                "residual_too_short",
            )
            continue

        # 6. Dedup on STRIPPED normalized text. Two chunks like
        #    "Audio Coming Soon\n\nDonald Trump's Golf" and
        #    "SUBSCRIBE NOW\n\nDonald Trump's Golf" both reduce to
        #    "Donald Trump's Golf" and dedup as identical.
        if stripped_normalized in seen_norms:
            chunks_skipped += 1
            chunks_skipped_duplicate += 1
            _add_example(
                examples_duplicate_after_strip, self_ref, rendered, stripped,
                "duplicate_after_strip",
            )
            continue
        seen_norms.add(stripped_normalized)

        # 7. Record stripping diagnostics (kept-and-stripped chunks).
        if had_chrome_stripped:
            chunks_stripped_web_chrome += 1
            if leading_removed > 0:
                _add_example(
                    examples_leading_chrome, self_ref, rendered, stripped,
                    "leading_chrome",
                )
            if trailing_removed > 0:
                _add_example(
                    examples_trailing_chrome, self_ref, rendered, stripped,
                    "trailing_chrome",
                )

        # Page number (first prov entry)
        page_no: str | None = None
        prov_list = elem.get("prov") or []
        if prov_list and isinstance(prov_list[0], dict):
            pn = prov_list[0].get("page_no")
            if isinstance(pn, int):
                page_no = str(pn)

        # 8. Insert the STRIPPED text (NOT the original). The reranker and the
        #    LLM both see the chrome-free version, so chrome can't poison
        #    reranker scores and can't waste downstream LLM context.
        pending.append((self_ref, modality, page_no, stripped))
        modality_counts[modality] = modality_counts.get(modality, 0) + 1

    any_quality_filter_hit = (
        chunks_skipped_short
        or chunks_skipped_duplicate
        or chunks_stripped_web_chrome
        or chunks_skipped_after_strip
    )
    if any_quality_filter_hit:
        logger.debug(
            "build_extraction_index quality filter: short=%d (e.g. %s) "
            "stripped=%d (leading e.g. %s; trailing e.g. %s) "
            "after_strip=%d (e.g. %s) "
            "duplicate_after_strip=%d (e.g. %s)",
            chunks_skipped_short, short_examples,
            chunks_stripped_web_chrome,
            examples_leading_chrome, examples_trailing_chrome,
            chunks_skipped_after_strip, examples_residual_too_short,
            chunks_skipped_duplicate, examples_duplicate_after_strip,
        )

    if not pending:
        return BuildIndexDiagnostics(
            chunks_inserted=0,
            chunks_skipped=chunks_skipped,
            embed_calls=0,
            embed_ms=0,
            insert_ms=0,
            modality_counts=modality_counts,
            chunks_skipped_short=chunks_skipped_short,
            chunks_skipped_duplicate=chunks_skipped_duplicate,
            chunks_skipped_web_chrome=0,  # deprecated in v2
            chunks_stripped_web_chrome=chunks_stripped_web_chrome,
            chunks_skipped_after_strip=chunks_skipped_after_strip,
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

    # Explicit length guard: zip() silently truncates on a short response.
    # If embed_texts returns fewer (or more) vectors than pending elements,
    # the index would be partial — diagnostics.chunks_inserted would report
    # len(pending) but only the truncated count would actually be inserted,
    # breaking C.4's failure detection. Raise here so C.4's try/except can
    # catch and fall back to RUN_FULL.
    if len(embeddings) != len(pending):
        raise RuntimeError(
            f"build_extraction_index: embed_texts returned {len(embeddings)} "
            f"vectors but expected {len(pending)} "
            f"(pipeline_run_id={pipeline_run_id!r}). "
            f"This indicates a partial embedding response; C.4 must catch this "
            f"exception and fall back to RUN_FULL."
        )

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
        "inserted=%d skipped=%d (short=%d dup_after_strip=%d after_strip=%d) "
        "stripped=%d embed_ms=%d insert_ms=%d modalities=%s",
        pipeline_run_id, document_id,
        chunks_inserted, chunks_skipped,
        chunks_skipped_short, chunks_skipped_duplicate, chunks_skipped_after_strip,
        chunks_stripped_web_chrome,
        embed_ms, insert_ms, modality_counts,
    )

    return BuildIndexDiagnostics(
        chunks_inserted=chunks_inserted,
        chunks_skipped=chunks_skipped,
        embed_calls=embed_calls,
        embed_ms=embed_ms,
        insert_ms=insert_ms,
        modality_counts=modality_counts,
        chunks_skipped_short=chunks_skipped_short,
        chunks_skipped_duplicate=chunks_skipped_duplicate,
        chunks_skipped_web_chrome=0,  # deprecated in v2
        chunks_stripped_web_chrome=chunks_stripped_web_chrome,
        chunks_skipped_after_strip=chunks_skipped_after_strip,
    )


# ---------------------------------------------------------------------------
# Public API: cleanup
# ---------------------------------------------------------------------------

_DELETE_SQL = (
    "DELETE FROM ExtractionChunk WHERE pipeline_run_id = :run_id"
)


def _delete_by_run_id(
    store: "ArcadeDBGraphStore",
    pipeline_run_id: str,
    *,
    strict: bool = False,
) -> int:
    """Delete all ExtractionChunk rows for a given pipeline_run_id.

    Parameters
    ----------
    strict:
        When False (default, used by ``cleanup_extraction_index``):
            Swallows exceptions, logs WARNING, returns 0. Best-effort cleanup.
        When True (used by ``build_extraction_index``):
            Lets exceptions propagate. A failed DELETE means the subsequent
            INSERT loop could collide with existing rows, violating the
            idempotency contract. The C.4 dispatcher try/except catches the
            propagated exception and falls back to RUN_FULL.

    Returns
    -------
    int
        Count of deleted rows, or 0 on error in non-strict mode.
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
    except Exception as exc:
        if strict:
            raise
        logger.warning(
            "_delete_by_run_id: failed to delete ExtractionChunk rows "
            "for pipeline_run_id=%r — %s: %s.",
            pipeline_run_id,
            type(exc).__name__,
            exc,
        )
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
