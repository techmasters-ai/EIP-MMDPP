"""Scoped DoclingDocument builder (VR Phase C.4, rev 10 M7).

apply_chunk_scope(doc_json, chunk_scope) builds a scoped DoclingDocument JSON
from a full doc_json and a chunk_scope dict produced by the /v1/extraction/chunk-scope
endpoint.

Design rules (per rev 7 H1 + rev 10 M7 + C4d constraint from rev 3):
  * Preserve top-level arrays (texts[], tables[], pictures[], groups[]) in their
    ORIGINAL positions — never reindex.  Docling refs are index-based
    (#/texts/N, #/tables/N) — removing elements would invalidate every
    cross-reference in the document.
  * Rewrite body.children to ONLY include refs from chunk_scope.self_refs
    PLUS the section_header/title headings that provide context for selected
    content (nearest heading that PRECEDES one of the selected refs in
    document order).
  * Validate every self_ref in chunk_scope against the actual arrays in doc_json.
    An unknown self_ref raises ValueError so the worker can detect its own bugs.
  * mode != "selected_refs" is not this function's responsibility — the caller
    (derive_ontology_graph_pass) gates on chunk_scope.get("mode") == "selected_refs"
    before calling. Called with any other mode, we raise ValueError rather than
    silently returning unchanged (undefined contract → loud error).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Docling labels that mark section headings.  Headings preceding selected
# content are included in body.children to maintain structural context.
_HEADING_LABELS = frozenset({"section_header", "section-header", "title"})


def _is_heading_label(label: str) -> bool:
    return label.lower().replace(" ", "_") in _HEADING_LABELS


def _ref_to_array_key_and_index(self_ref: str) -> tuple[str, int] | None:
    """Parse '#/texts/N', '#/tables/N', '#/pictures/N' → (array_key, N).

    Returns None when the self_ref format is unrecognized.
    """
    # Standard docling format: '#/<array>/<index>'
    if not self_ref.startswith("#/"):
        return None
    parts = self_ref.split("/")
    if len(parts) != 3:
        return None
    _, array_name, index_str = parts
    if array_name not in ("texts", "tables", "pictures", "groups"):
        return None
    try:
        return array_name, int(index_str)
    except ValueError:
        return None


def _resolve_element(doc_json: dict, self_ref: str) -> dict | None:
    """Resolve a self_ref to its element dict in doc_json.

    Supports '#/texts/N', '#/tables/N', '#/pictures/N'.
    Returns None when not found.
    """
    parsed = _ref_to_array_key_and_index(self_ref)
    if parsed is None:
        return None
    array_name, idx = parsed
    arr = doc_json.get(array_name) or []
    if 0 <= idx < len(arr):
        elem = arr[idx]
        if isinstance(elem, dict):
            return elem
    # Fallback: linear scan (handles sparse / non-standard arrays in test fixtures)
    for elem in arr:
        if isinstance(elem, dict) and elem.get("self_ref") == self_ref:
            return elem
    return None


def _validate_self_refs(doc_json: dict, self_refs: list[str]) -> None:
    """Validate every self_ref is resolvable.

    Raises ValueError on the FIRST unknown ref.  This catches a worker-side
    bug class (e.g. stale self_refs from a prior build_extraction_index run).
    """
    for sr in self_refs:
        elem = _resolve_element(doc_json, sr)
        if elem is None:
            raise ValueError(
                f"apply_chunk_scope: self_ref {sr!r} is not resolvable in this "
                f"doc_json. This indicates a mismatch between the ExtractionChunk "
                f"index and the document JSON being processed. "
                f"Ensure build_extraction_index ran on the same doc_json version."
            )


def _walk_body_children_ordered(
    doc_json: dict,
) -> list[tuple[str, dict]]:
    """Walk body.children in document order, yielding (cref, element_dict).

    Groups are recursed into (depth-limited).  Only texts[], tables[], and
    pictures[] are yielded — groups themselves are structural wrappers.

    Supports both "cref" (real docling) and "$ref"/"$cref" (synthetic fixtures).
    """
    results: list[tuple[str, dict]] = []
    visited_groups: set[int] = set()

    def _resolve_cref(ref: dict) -> str:
        return ref.get("cref") or ref.get("$ref") or ref.get("$cref", "")

    def _walk(refs: list, depth: int = 0) -> None:
        if depth > 10:
            return
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            cref = _resolve_cref(ref)
            if not cref:
                continue

            if cref.startswith("#/groups/"):
                try:
                    idx = int(cref.split("/")[-1])
                except (ValueError, TypeError):
                    continue
                if idx in visited_groups:
                    continue
                visited_groups.add(idx)
                grp_arr = doc_json.get("groups") or []
                if 0 <= idx < len(grp_arr) and isinstance(grp_arr[idx], dict):
                    _walk(grp_arr[idx].get("children") or [], depth + 1)
            elif cref.startswith("#/texts/") or cref.startswith("#/tables/") or cref.startswith("#/pictures/"):
                elem = _resolve_element(doc_json, cref)
                if elem is not None:
                    results.append((cref, elem))

    body = doc_json.get("body")
    if isinstance(body, dict):
        _walk(body.get("children") or [])
    return results


def apply_chunk_scope(doc_json: dict, chunk_scope: dict) -> dict:
    """Build a scoped DoclingDocument JSON from a full doc + chunk_scope.

    chunk_scope must have:
        mode: "selected_refs"   — any other mode raises ValueError.
        self_refs: list[str]    — the docling self_refs to include in scope.

    Strategy
    --------
    1. Validate mode == "selected_refs" (loudly, not silently).
    2. Validate every self_ref is resolvable in doc_json (bug-detection gate).
    3. Walk body.children in document order, collecting:
       a. Section heading elements (section_header / title labels) that
          immediately PRECEDE at least one selected ref in document order.
          Heading is included if ANY selected ref follows it before the next
          heading.
       b. All elements whose self_ref appears in chunk_scope["self_refs"].
    4. Return a shallow copy of doc_json with body.children rewritten to the
       ordered list from step 3, using the same ref-dict format as the
       original body.children ({"cref": ...} for real docling docs,
       {"$ref": ...} for synthetic tests).

    Arrays preserved
    ----------------
    texts[], tables[], pictures[], groups[] are NEVER modified.  Removing
    elements would invalidate cross-references. body.children is the ONLY
    field rewritten.

    Returns
    -------
    A new dict (shallow copy of doc_json with body replaced).  The original
    doc_json is NOT modified.
    """
    mode = chunk_scope.get("mode")
    if mode != "selected_refs":
        raise ValueError(
            f"apply_chunk_scope: chunk_scope.mode must be 'selected_refs', "
            f"got {mode!r}. Behavior for other modes is undefined; the caller "
            f"must only invoke apply_chunk_scope when mode='selected_refs'."
        )

    self_refs: list[str] = list(chunk_scope.get("self_refs") or [])
    selected_set: set[str] = set(self_refs)

    # Validate all refs are resolvable before building the scoped doc.
    _validate_self_refs(doc_json, self_refs)

    if not self_refs:
        # Empty scope → empty body.children.  Still valid per the contract:
        # an empty chunk_scope means the router returned no relevant chunks.
        new_body = dict(doc_json.get("body") or {})
        new_body["children"] = []
        return {**doc_json, "body": new_body}

    # Walk the document in order to build the scoped children list.
    # Algorithm:
    #   For each (cref, elem) in document order:
    #     - If it's a heading: remember it as "pending_heading" (not yet emitted).
    #     - If cref is in selected_set: emit any pending_heading first, then emit
    #       the selected elem. Clear pending_heading.
    #     - Non-selected non-heading elements are silently dropped.
    doc_order = _walk_body_children_ordered(doc_json)

    scoped_crefs: list[str] = []
    pending_heading_cref: str | None = None

    for cref, elem in doc_order:
        label = (elem.get("label") or "").lower().replace(" ", "_")
        if _is_heading_label(label):
            # Only headings in texts[] are valid context providers.
            # Table/picture elements with heading-looking labels are rare
            # but safe to skip — they don't serve as structural headings.
            if cref.startswith("#/texts/"):
                pending_heading_cref = cref
        elif cref in selected_set:
            if pending_heading_cref is not None:
                scoped_crefs.append(pending_heading_cref)
                pending_heading_cref = None
            scoped_crefs.append(cref)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_scoped_crefs: list[str] = []
    for c in scoped_crefs:
        if c not in seen:
            seen.add(c)
            unique_scoped_crefs.append(c)

    # Detect the ref-dict format used in the original body.children.
    # Real docling uses "cref"; synthetic test fixtures use "$ref" or "$cref".
    # Infer from the first child entry to preserve round-trip fidelity.
    def _detect_ref_key(body: dict) -> str:
        """Return the ref key used in the original children list."""
        children = body.get("children") or []
        for child in children:
            if not isinstance(child, dict):
                continue
            if "cref" in child:
                return "cref"
            if "$ref" in child:
                return "$ref"
            if "$cref" in child:
                return "$cref"
        return "cref"  # default: real docling format

    original_body = doc_json.get("body") or {}
    ref_key = _detect_ref_key(original_body)
    new_children = [{ref_key: cref} for cref in unique_scoped_crefs]

    new_body = dict(original_body)
    new_body["children"] = new_children

    logger.debug(
        "apply_chunk_scope: selected %d self_refs → %d scoped children "
        "(including %d heading refs for context)",
        len(selected_set),
        len(new_children),
        sum(1 for c in unique_scoped_crefs if c not in selected_set),
    )

    return {**doc_json, "body": new_body}
