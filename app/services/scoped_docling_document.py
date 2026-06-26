"""Scoped DoclingDocument builder (VR Phase C.4, rev 10 M7; C.7r list_item fix).

apply_chunk_scope(doc_json, chunk_scope) builds a scoped DoclingDocument JSON
from a full doc_json and a chunk_scope dict produced by the /v1/extraction/chunk-scope
endpoint.

Design rules (per rev 7 H1 + rev 10 M7 + C4d constraint from rev 3 + C.7r):
  * Preserve top-level arrays (texts[], tables[], pictures[], groups[]) in their
    ORIGINAL positions — never reindex.  Docling refs are index-based
    (#/texts/N, #/tables/N) — removing elements would invalidate every
    cross-reference in the document.
  * Rewrite body.children to include refs from chunk_scope.self_refs PLUS the
    section_header/title headings that provide context for selected content
    (nearest heading that PRECEDES one of the selected refs in document order).
    For retained list_items whose original parent is a ListGroup, body.children
    references the GROUP instead of the list_item (see C.7r below).
  * Validate every self_ref in chunk_scope against the actual arrays in doc_json.
    An unknown self_ref raises ValueError so the worker can detect its own bugs.
  * mode != "selected_refs" is not this function's responsibility — the caller
    (derive_ontology_graph_pass) gates on chunk_scope.get("mode") == "selected_refs"
    before calling. Called with any other mode, we raise ValueError rather than
    silently returning unchanged (undefined contract → loud error).

C.7r list_item fix
------------------
docling-core's DoclingDocument has a ``validate_misplaced_list_items`` model
validator that fires on construction. It finds any ListItem whose parent isn't
a ListGroup and "rescues" it: creates a new ListGroup, DELETES the misplaced
list_items from texts[], then re-adds new list_items at the end of texts[].

The delete-then-readd cycle re-indexes via a breadth-first walk of body.children.
Items unreachable from body.children (e.g. texts that are children of pictures
when the picture isn't in body.children) DON'T get their self_ref updated. The
result: texts at shifted positions keep their old self_refs, while the new
list_items reuse those self_refs. ``ChunkingDocSerializer(doc=dl_doc)`` then
re-runs ``_validate_unique_refs`` (via Pydantic model-validator-on-instance-pass)
and raises ``Duplicate ref: #/texts/N``.

Fix: for retained list_items whose immediate parent is a ListGroup
(group.label ∈ {"list", "ordered_list"}), KEEP the group parent and reference
the GROUP from body.children. The group's children list keeps the retained
list_items. Non-list_item retained refs in the same group still reparent to
#/body (current behavior; they don't trigger the rescue logic).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.services.chunk_quality import classify_chunk

logger = logging.getLogger(__name__)

# Docling labels that mark section headings.  Headings preceding selected
# content are included in body.children to maintain structural context.
_HEADING_LABELS = frozenset({"section_header", "section-header", "title"})

# Group labels that docling treats as ListGroups (subject to the
# ``validate_misplaced_list_items`` rescue validator).
_LIST_GROUP_LABELS = frozenset({"list", "ordered_list"})


def _is_heading_label(label: str) -> bool:
    return label.lower().replace(" ", "_") in _HEADING_LABELS


def _is_list_group_label(label: str) -> bool:
    return label.lower().replace(" ", "_") in _LIST_GROUP_LABELS


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
        text_by_ref: dict[str, str] (optional) — post-filter chunk text
            keyed by selected text self_ref. When present, retained TextItems
            are rendered from this text so live extraction sees the same
            evidence string that retrieval/rerank selected.

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
    raw_text_by_ref = chunk_scope.get("text_by_ref") or {}
    text_by_ref: dict[str, str] = (
        {k: v for k, v in raw_text_by_ref.items() if isinstance(k, str) and isinstance(v, str)}
        if isinstance(raw_text_by_ref, dict)
        else {}
    )

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
    # NOTE: moved before the reachability pass so ref_key is available.
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

    # --- Post-scope parent reachability pass (rev 7 H1 + M7, fix rev 19) ---
    # Validate that retained elements' parent chains resolve through the
    # ORIGINAL document body topology.  Emit WARNING (not ValueError) — narrowing
    # still proceeds.  A future hardening pass can promote to ValueError if
    # production data shows actual mis-navigation.
    #
    # Build reachable_refs by walking the ORIGINAL body.children (not the
    # rewritten unique_scoped_crefs).  A group is "reachable" iff at least one
    # of its (possibly nested) descendant refs is in unique_scoped_crefs.
    # This matches docling's parent-chain semantics: a text with parent
    # #/groups/3 has a semantically valid parent IF #/groups/3 is structurally
    # on the path from #/body to one of the selected refs in the original topology.
    #
    # Rev 19 fix for C.6 BLOCKER: the old code seeded _collect_reachable with
    # unique_scoped_crefs (which never contains #/groups/N refs), so groups were
    # never marked reachable.  SA-2 has ~40/308 group-parented texts — the old
    # code would emit hundreds of spurious WARNINGs per document in C.6.
    _reachable_refs: set[str] = {"#/body"}

    _selected_set_for_reach: set[str] = set(unique_scoped_crefs)
    _groups_array = doc_json.get("groups") or []

    def _resolve_cref_from_dict(d: dict) -> str:
        return d.get("cref") or d.get("$ref") or d.get("$cref", "")

    def _group_idx_from_ref(group_ref: str) -> int | None:
        """Parse '#/groups/N' → int N. Returns None on parse error OR negative N.
        Rev 20 review Minor #2: explicitly reject negative indices to prevent
        Python negative-list-indexing from silently resolving '#/groups/-1'
        to the last group (false-positive reachability on corrupted input).
        """
        try:
            idx = int(group_ref.rsplit("/", 1)[-1])
        except (ValueError, TypeError):
            return None
        if idx < 0:
            return None
        return idx

    def _has_selected_descendant(group_ref: str, visited: set[str], depth: int = 0) -> bool:
        """Return True iff the group contains a selected ref among its descendants."""
        if depth > 10 or group_ref in visited:
            return False
        visited.add(group_ref)
        idx = _group_idx_from_ref(group_ref)
        if idx is None:
            return False
        try:
            grp = _groups_array[idx]
        except IndexError:
            return False
        if not isinstance(grp, dict):
            return False
        for child in (grp.get("children") or []):
            if not isinstance(child, dict):
                continue
            child_ref = _resolve_cref_from_dict(child)
            if not child_ref:
                continue
            if child_ref in _selected_set_for_reach:
                return True
            if child_ref.startswith("#/groups/"):
                if _has_selected_descendant(child_ref, visited, depth + 1):
                    return True
        return False

    def _mark_group_and_nested_reachable(group_ref: str, visited: set[str], depth: int = 0) -> None:
        """Mark group_ref and all nested sub-groups reachable (they're on the path to selected refs)."""
        if depth > 10 or group_ref in visited:
            return
        visited.add(group_ref)
        _reachable_refs.add(group_ref)
        idx = _group_idx_from_ref(group_ref)
        if idx is None:
            return
        try:
            grp = _groups_array[idx]
        except IndexError:
            return
        if not isinstance(grp, dict):
            return
        for child in (grp.get("children") or []):
            if not isinstance(child, dict):
                continue
            child_ref = _resolve_cref_from_dict(child)
            if not child_ref:
                continue
            if child_ref.startswith("#/groups/"):
                # Only recurse into sub-groups that also have selected descendants
                if _has_selected_descendant(child_ref, set(), depth + 1):
                    _mark_group_and_nested_reachable(child_ref, visited, depth + 1)

    # Walk the ORIGINAL body.children to find reachable groups.
    # (Direct non-group children of body don't need explicit handling here —
    # rev 20 review Minor #1: the final `_reachable_refs.update(_selected_set_for_reach)`
    # below covers them unconditionally; an extra branch in the loop would be
    # redundant code that misleads readers about coverage.)
    original_body_children = (doc_json.get("body") or {}).get("children") or []
    for child_dict in original_body_children:
        if not isinstance(child_dict, dict):
            continue
        cref = _resolve_cref_from_dict(child_dict)
        if not cref or not cref.startswith("#/groups/"):
            continue
        if _has_selected_descendant(cref, set()):
            _mark_group_and_nested_reachable(cref, set())

    # Selected refs are reachable by definition; this also covers direct
    # (non-group) children of body that are in scope.
    _reachable_refs.update(_selected_set_for_reach)

    # Check each retained element's parent ref against the reachable set.
    for cref in unique_scoped_crefs:
        elem = _resolve_element(doc_json, cref)
        if elem is None or not isinstance(elem, dict):
            continue
        parent = elem.get("parent")
        if parent is None or not isinstance(parent, dict):
            continue
        parent_ref = parent.get("cref") or parent.get("$ref") or parent.get("$cref")
        if parent_ref and parent_ref not in _reachable_refs:
            logger.warning(
                "apply_chunk_scope: retained element %r has unreachable parent %r "
                "(parent not in rewritten body.children). Docling ref-resolution may "
                "mis-navigate. Narrowing proceeds; promote to ValueError in a "
                "future hardening pass if production data shows mis-navigation.",
                cref, parent_ref,
            )
    # --- end post-scope reachability pass ---

    # --- Hierarchy mutation pass (C.7 + C.7r fix for DoclingDocument validation) ---
    # The reachability walker above is diagnostic-only: it tells us when a
    # retained element's parent was orphaned by the flatten. DoclingDocument's
    # Pydantic validator REJECTS docs whose tree topology is inconsistent:
    #   Value error, Document hierarchy is inconsistent.
    #   #/body has child #/texts/N with parent #/groups/M
    #
    # The scoped body.children is FLAT for everything EXCEPT retained list_items
    # whose immediate parent is a ListGroup. For those, we KEEP the original
    # ListGroup parent and reference the GROUP from body.children (see C.7r in
    # the module docstring for the duplicate-ref bug this prevents).
    #
    # No-mutation invariant: top-level arrays are shallow-copied, and any
    # element dict we modify is itself shallow-copied. The input doc_json is
    # never touched.

    # --- C.7r: identify retained list_items that must keep ListGroup parent ---
    # group_ref -> group.label cache (lowercase, underscored).
    group_label_cache: dict[str, str] = {}
    for grp in (doc_json.get("groups") or []):
        if isinstance(grp, dict):
            gr = grp.get("self_ref")
            if gr:
                group_label_cache[gr] = (grp.get("label") or "").lower().replace(" ", "_")

    # list_item_in_list_group: set of crefs (in unique_scoped_crefs) that are
    # list_items whose immediate parent is a ListGroup. Their original parent
    # must be preserved; body.children references the group, not them.
    list_item_in_list_group: set[str] = set()
    # parent_group_for_list_item: cref -> group_ref (the ListGroup parent).
    parent_group_for_list_item: dict[str, str] = {}
    for cref in unique_scoped_crefs:
        elem = _resolve_element(doc_json, cref)
        if elem is None or not isinstance(elem, dict):
            continue
        label = (elem.get("label") or "").lower().replace(" ", "_")
        if label != "list_item":
            continue
        parent = elem.get("parent")
        if not isinstance(parent, dict):
            continue
        pref = parent.get("cref") or parent.get("$ref") or parent.get("$cref")
        if not pref or not pref.startswith("#/groups/"):
            continue
        if _is_list_group_label(group_label_cache.get(pref, "")):
            list_item_in_list_group.add(cref)
            parent_group_for_list_item[cref] = pref

    # retained_list_groups: groups that own ≥1 retained list_item; they go
    # into body.children in place of those list_items.
    retained_list_groups: set[str] = set(parent_group_for_list_item.values())

    # reparent_to_body_set: refs whose `parent` field should be rewritten to
    # {"cref": "#/body"} — everything in scope EXCEPT list_items kept inside
    # a ListGroup.
    reparent_to_body_set: set[str] = set(unique_scoped_crefs) - list_item_in_list_group

    # body.children build order: walk unique_scoped_crefs (already in document
    # order), substituting list_item refs with their group ref, deduplicating.
    body_child_crefs: list[str] = []
    body_child_seen: set[str] = set()
    for cref in unique_scoped_crefs:
        out_cref = parent_group_for_list_item.get(cref, cref)
        if out_cref not in body_child_seen:
            body_child_seen.add(out_cref)
            body_child_crefs.append(out_cref)

    doc_overlay: dict = {}

    def _rewrite_parent_to_body(elem: dict) -> dict:
        # Preserve the parent dict's existing key style when present so that
        # synthetic test fixtures using "$ref"/"$cref" round-trip; default to
        # "cref" (the docling-core canonical key) otherwise.
        existing_parent = elem.get("parent")
        if isinstance(existing_parent, dict):
            parent_key = next(
                (k for k in ("cref", "$ref", "$cref") if k in existing_parent),
                "cref",
            )
        else:
            parent_key = "cref"
        new_elem = dict(elem)
        new_elem["parent"] = {parent_key: "#/body"}
        return new_elem

    for array_key in ("texts", "tables", "pictures"):
        arr = doc_json.get(array_key)
        if not arr:
            continue
        new_arr = list(arr)
        mutated = False
        for i, elem in enumerate(new_arr):
            if not isinstance(elem, dict):
                continue
            elem_ref = elem.get("self_ref")
            new_elem = elem
            if elem_ref in reparent_to_body_set:
                new_elem = _rewrite_parent_to_body(new_elem)
                mutated = True
            if array_key == "texts" and elem_ref in text_by_ref:
                override_text = text_by_ref[elem_ref].strip()
                if override_text:
                    new_elem = dict(new_elem)
                    new_elem["text"] = override_text
                    new_elem["orig"] = override_text
                    if "hyperlink" in new_elem:
                        new_elem["hyperlink"] = None
                    mutated = True
            if new_elem is not elem:
                new_arr[i] = new_elem
        if mutated:
            doc_overlay[array_key] = new_arr

    groups_arr = doc_json.get("groups")
    if groups_arr:
        new_groups = list(groups_arr)
        mutated_any = False
        for i, grp in enumerate(new_groups):
            if not isinstance(grp, dict):
                continue
            grp_self_ref = grp.get("self_ref")
            children = grp.get("children") or []
            # Strip children whose parent was rewritten to #/body. Retained
            # list_items kept in this group (parent unchanged) STAY in children.
            filtered = [
                c for c in children
                if not (
                    isinstance(c, dict)
                    and (c.get("cref") or c.get("$ref") or c.get("$cref"))
                    in reparent_to_body_set
                )
            ]
            needs_children_update = len(filtered) != len(children)
            # When a retained ListGroup is placed into body.children (C.7r path),
            # its parent must be rewritten to #/body. For a NESTED list group whose
            # original parent is another group (e.g. #/groups/11), the hierarchy
            # validator rejects: "has child #/groups/12 with parent #/groups/11".
            # Invariant: every node referenced in body.children must declare #/body
            # as its parent. This includes groups moved up from nested positions.
            needs_parent_update = bool(grp_self_ref and grp_self_ref in retained_list_groups)
            if needs_children_update or needs_parent_update:
                new_grp = dict(grp)
                if needs_children_update:
                    new_grp["children"] = filtered
                if needs_parent_update:
                    new_grp = _rewrite_parent_to_body(new_grp)
                new_groups[i] = new_grp
                mutated_any = True
        if mutated_any:
            doc_overlay["groups"] = new_groups
    # --- end hierarchy mutation pass ---

    new_children = [{ref_key: cref} for cref in body_child_crefs]

    new_body = dict(original_body)
    new_body["children"] = new_children

    logger.debug(
        "apply_chunk_scope: selected %d self_refs → %d scoped children "
        "(including %d heading refs for context)",
        len(selected_set),
        len(new_children),
        sum(1 for c in unique_scoped_crefs if c not in selected_set),
    )

    return {**doc_json, **doc_overlay, "body": new_body}


# ---------------------------------------------------------------------------
# C.10: filter_docling_document — worker-side v2 quality filter for ALL passes
# ---------------------------------------------------------------------------


#: Label values whose text content is protected from blanking and dedup.
#:
#: - ``"caption"``: matches docling-graph's own sanitizer (main.py:482). Image
#:   captions carry intentional repetition (figure numbers, "see also" notes)
#:   that an aggressive dedup would silently destroy.
#:
#: - ``"section_header"`` / ``"section-header"`` / ``"title"``: heading labels.
#:   The chunker uses these as parent-heading context for adjacent text chunks
#:   (e.g. ``_resolve_parent_section_heading`` reads the heading's ``text``
#:   field). Blanking them leaves downstream chunks without their section
#:   prefix, degrading LLM context. Mirrors the ``_HEADING_LABELS`` exclusion
#:   in ``app/services/extraction_chunk_index.py`` (those labels are also
#:   already skipped from indexing, so protecting them here is a no-op for
#:   the indexed-chunk pool but load-bearing for parent-heading resolution).
_PROTECTED_LABELS: frozenset[str] = frozenset({
    "caption",
    "section_header",
    "section-header",
    "title",
})


@dataclass
class FilterDiagnostics:
    """Per-call counters from filter_docling_document.

    texts_in counts every entry processed (including ones unchanged or skipped
    as non-dict). The four mutation counters are mutually exclusive — any single
    entry contributes to at most one of:

      - blanked_short:        entry was blanked because rendered text < MIN_CHUNK_TEXT_CHARS
      - blanked_dedup:        entry was blanked because its stripped form was already seen
      - blanked_after_strip:  entry was blanked because residue < MIN_RESIDUAL_CHARS after strip
      - stripped_in_place:    entry was KEPT but its text was overwritten with the stripped form

    protected_labels counts label=="caption" entries that bypassed all mutation
    checks (never blanked, never stripped, never deduped).
    """
    texts_in: int = 0
    blanked_short: int = 0
    blanked_dedup: int = 0
    blanked_after_strip: int = 0
    stripped_in_place: int = 0
    protected_labels: int = 0


def filter_docling_document(doc_json: dict) -> tuple[dict, FilterDiagnostics]:
    """Apply v2 quality filter to a DoclingDocument JSON in place.

    For each entry in ``doc_json["texts"]``:
      * Entries with ``label`` in ``_PROTECTED_LABELS`` (e.g. ``"caption"``)
        are NEVER blanked or deduped. The filter records them in
        ``diag.protected_labels`` and moves on.
      * Dropped entries (short / dedup / after_strip) have their ``text`` and
        ``orig`` blanked and ``hyperlink`` cleared. The entry stays in the
        array so $refs from body.children / pictures.children / tables.children
        remain valid.
      * Kept-with-strip entries have their ``text`` and ``orig`` overridden
        with the post-strip residue and their ``hyperlink`` cleared (the
        hyperlink may have pointed at chrome-only context).
      * Kept-as-is entries are untouched.

    The function mutates ``doc_json`` and returns it for chained-call ergonomics.

    Idempotent: running it twice on the same doc produces zero new blanks
    or strips on the second pass.

    Defensive: missing/None/non-list texts and non-dict elements are skipped
    without raising. The worker still wraps the call in try/except for
    catastrophic shapes (see Task 4 / Task 5), but the most common malformed
    cases must not crash this function.
    """
    diag = FilterDiagnostics()
    texts = doc_json.get("texts") or []
    if not isinstance(texts, list):
        return doc_json, diag
    diag.texts_in = len(texts)

    seen_norms: set[str] = set()
    for i, t in enumerate(texts):
        if not isinstance(t, dict):
            continue

        # Caption protection: docling-graph sanitizer parallel (main.py:482).
        # Image-description captions must survive even when their text is
        # short or duplicate across figures.
        label = (t.get("label") or "").lower()
        if label in _PROTECTED_LABELS:
            diag.protected_labels += 1
            continue

        # Defensive: text/orig can be None per observed docling output.
        rendered_raw = t.get("text") if t.get("text") is not None else t.get("orig")
        if rendered_raw is None:
            continue
        rendered = str(rendered_raw).strip()
        if not rendered:
            continue

        decision = classify_chunk(
            rendered,
            seen_norms,
            skip_short_reject=True,
            gate_after_strip_on_chrome=True,
        )
        if not decision.keep:
            new_t = dict(t)
            new_t["text"] = ""
            new_t["orig"] = ""
            if "hyperlink" in new_t:
                new_t["hyperlink"] = None
            texts[i] = new_t
            if decision.reason == "short":
                diag.blanked_short += 1
            elif decision.reason == "after_strip":
                diag.blanked_after_strip += 1
            elif decision.reason == "dedup":
                diag.blanked_dedup += 1
            continue
        if decision.stripped_text is not None:
            new_t = dict(t)
            new_t["text"] = decision.stripped_text
            new_t["orig"] = decision.stripped_text
            if "hyperlink" in new_t:
                new_t["hyperlink"] = None
            texts[i] = new_t
            diag.stripped_in_place += 1

    doc_json["texts"] = texts
    return doc_json, diag
