"""HybridChunker integration: post-process native chunks to substitute
normalized table chunks where appropriate.

See §10.1 of the design spec. The functions here are pure (no I/O);
they are invoked from app/workers/pipeline.py between the native
chunker call and the chunk-iteration loop.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from app.services.table_normalization.models import (
    NormalizedTable,
    Shape,
)
from app.services.table_normalization.tokens import count_bge_m3_tokens
from app.services.table_normalization.render_graph import _render_column_as_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AdapterProv:
    page_no: int


@dataclass(frozen=True)
class _AdapterDocItem:
    self_ref: str
    prov: tuple[_AdapterProv, ...]


@dataclass(frozen=True)
class _AdapterMeta:
    doc_items: tuple[_AdapterDocItem, ...]
    headings: tuple[str, ...]


@dataclass(frozen=True)
class _NormalizedTableChunkAdapter:
    """Ducktypes the HybridChunker chunk interface.

    Read interface that pipeline.py:5559-5623 uses:
    - .text
    - .meta.doc_items[].self_ref
    - .meta.doc_items[].prov[].page_no
    - .meta.headings
    Plus .extra_metadata for chunk_metadata column population.

    CRITICAL: the single synthetic doc_item's self_ref is the TABLE-LEVEL
    ref ("#/tables/{N}"), never a cell ref. Cell refs flow through
    .extra_metadata.cell_refs only — keeps TextChunk.self_refs in today's
    shape (#/texts/N or #/tables/N), so provenance.py:_resolve_element_uid
    and the retrieval response surface stay backwards compatible.
    """
    etc: Any  # EmbeddingTableChunk or GraphTableChunk
    parent_headings: tuple[str, ...]
    parent_table_ref: str

    @property
    def text(self) -> str:
        return self.etc.text

    @property
    def meta(self) -> _AdapterMeta:
        prov_tuple = tuple(_AdapterProv(page_no=p) for p in self.etc.page_numbers)
        item = _AdapterDocItem(self_ref=self.parent_table_ref, prov=prov_tuple)
        return _AdapterMeta(doc_items=(item,), headings=self.parent_headings)

    @property
    def extra_metadata(self) -> dict:
        return {
            "chunk_kind": self.etc.chunk_kind.value,
            "table_ref": self.etc.table_ref,
            "entity_display_name": self.etc.entity_display_name,
            "section": self.etc.section,
            "column_index": self.etc.column_index,
            "cell_refs": list(self.etc.cell_refs),
            "row_labels": list(self.etc.row_labels),
            "page_numbers": list(self.etc.page_numbers),
        }


def _normalized_table_size_tokens(nt: NormalizedTable) -> int:
    """Canonical size function per spec rev. 7 §10.1.

    Sum of bge-m3 tokens across rendered columns. Single contract; no
    cheap fallback (boundary behavior at MIN_TABLE_NORMALIZATION_TOKENS
    must be deterministic).
    """
    return sum(
        count_bge_m3_tokens(_render_column_as_text(col, nt, nt.sections))
        for col in nt.columns
    )


def _classify_native_chunk(
    nc: Any, normalized_by_table_idx: dict[int, NormalizedTable],
) -> tuple[str, int | None]:
    """Classify a native chunk as table_dominant / table_mixed / non_table."""
    items = getattr(getattr(nc, "meta", None), "doc_items", None) or []
    if not items:
        return ("non_table", None)
    table_idx_counts: dict[int, int] = {}
    for item in items:
        ref = getattr(item, "self_ref", None) or ""
        if not ref.startswith("#/tables/"):
            continue
        try:
            idx = int(ref.split("/")[-1])
        except (ValueError, IndexError):
            continue
        if idx in normalized_by_table_idx and normalized_by_table_idx[idx].shape != Shape.OTHER:
            table_idx_counts[idx] = table_idx_counts.get(idx, 0) + 1
    if not table_idx_counts:
        return ("non_table", None)
    dominant_idx = max(table_idx_counts, key=table_idx_counts.get)
    dominant_share = table_idx_counts[dominant_idx] / len(items)
    return (("table_dominant" if dominant_share >= 0.8 else "table_mixed"), dominant_idx)


def _suppress_raw_table_texts(
    doc_json: dict,
    normalized: list[NormalizedTable],
) -> None:
    """Blank the flat-text mirrors of normalized non-OTHER tables in-place.

    Per §9.2 invariant:
    - len(doc_json['texts']) is UNCHANGED. No element is removed; no
      index shifts. This preserves self_ref stability for any code that
      references texts by index (children refs, prov entries, etc.).
    - doc_json['tables'] is NOT touched. The Phase 0/0.5 overlay
      machinery reads tables[] directly and must remain functional.
    - Tables with shape == OTHER keep their flat text (the OTHER
      fallback depends on it).
    """
    non_other = {nt.table_index for nt in normalized if nt.shape != Shape.OTHER}
    if not non_other:
        return
    target_refs = {f"#/tables/{i}" for i in non_other}
    for t in doc_json.get("texts") or []:
        if t.get("self_ref") in target_refs:
            t["text"] = ""
            t["orig"] = ""


# 2026-05-16: per-pass + per-table synth-only policy. Generalization over
# "kinematics-only on SA-2" — the underlying invariant is "the active pass
# extracts numeric/spec values AND this specific table has row labels that
# match what the pass extracts." A doc without a relevant table gets v9
# behavior (raw refs preserved); a doc with multiple tables gets synth-only
# treatment only for the tables that actually match the pass.

SYNTH_ELIGIBLE_PASSES: frozenset[str] = frozenset({
    "missile_kinematics",
    "missile_airframe",
    "missile_speed_timing",
    "missile_propulsion",
    "radar_antenna",
    "radar_timing",
    "radar_modulation",
    "radar_power_rf",
})

# Identity/prose-heavy passes: never synth-only. They quietly benefit from
# the raw flat's over-emission per the V9-POST-FIX-GLOBAL regression analysis.
# system_links additionally short-circuits normalization upstream (v8b gate).
RAW_ONLY_PASSES: frozenset[str] = frozenset({
    "missile_identity",
    "radar_identity",
    "system_links",
})

# Row-label aliases that signal a normalized table is relevant to a pass.
# All comparisons are case-insensitive against the normalized row label.
PASS_TABLE_ROW_ALIASES: dict[str, frozenset[str]] = {
    "missile_kinematics": frozenset({
        "max range", "min range", "range",
        "max alt", "min alt", "max altitude", "min altitude",
        "ceiling", "engagement range", "engagement altitude",
    }),
    "missile_airframe": frozenset({
        "length", "overall length", "missile length", "body length",
        "diameter", "body diameter", "calibre", "caliber",
        "weight", "mass", "launch weight", "launch mass",
        "warhead weight", "warhead mass",
    }),
    "missile_speed_timing": frozenset({
        "speed", "max speed", "maximum speed", "velocity", "max velocity",
        "vmax", "vmax appr tgt", "vmax reced tgt",
        "average speed", "average velocity",
        "time of flight", "flight time", "flyout time",
    }),
    "missile_propulsion": frozenset({
        "booster", "sustainer", "sustain", "ejector",
        "burn time", "thrust", "stage weight", "stage mass",
        "1st stage", "2nd stage", "first stage", "second stage",
    }),
    "radar_antenna": frozenset({
        "antenna", "gain", "antenna gain", "beamwidth",
        "azimuth", "elevation", "antenna width", "antenna height",
        "azimuth beamwidth", "elevation beamwidth",
        "azimuth aperture", "elevation aperture",
    }),
    "radar_timing": frozenset({
        "pri", "pulse repetition interval", "pulse interval",
        "pulse width", "pulse duration", "pw",
        "scan period", "scan time", "rotation period",
        "dwell", "dwell time",
    }),
    "radar_modulation": frozenset({
        "chirp", "chirp bandwidth", "frequency excursion", "sweep width",
        "code length", "chips", "bits", "pulses per dwell",
    }),
    "radar_power_rf": frozenset({
        "frequency", "operating frequency", "carrier frequency", "rf",
        "peak power", "transmitter power", "tx power",
        "erp", "effective radiated power",
    }),
}


def _normalize_row_label(label: str | None) -> str:
    """Lowercased + whitespace-collapsed row label for alias matching."""
    if not label:
        return ""
    return " ".join(label.lower().split())


# 2026-05-16 (Option A): unit-convention detection from caption + adjacent
# prose. Hardcoded SI works for SA-2 (Soviet, metric-native) but breaks on
# US DoD legacy / Cold-War imperial-convention docs. Caption phrasing like
# "(metric)" / "(imperial)" / explicit unit mentions ("range in nautical
# miles", "altitude in feet") gives us per-doc evidence.

_IMPERIAL_MARKERS: tuple[re.Pattern, ...] = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bimperial\b",
    r"\bus[ -]?customary\b",
    r"\(\s*feet\s*\)", r"\(\s*ft\s*\)",
    r"\bin\s+feet\b", r"\bin\s+ft\b",
    r"\(\s*nautical[ -]?miles?\s*\)",
    r"\bin\s+nautical[ -]?miles?\b",
    r"\(\s*nmi?\s*\)",
    r"\bin\s+nmi?\b",
    r"\(\s*pounds?\s*\)", r"\(\s*lbs?\s*\)",
    r"\bin\s+pounds?\b", r"\bin\s+lbs?\b",
    r"\(\s*knots?\s*\)", r"\(\s*kts?\s*\)",
    r"\bin\s+knots?\b",
    r"\(\s*inches?\s*\)", r"\bin\s+inches?\b",
))

_METRIC_MARKERS: tuple[re.Pattern, ...] = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bmetric\b",
    r"\(\s*si\s*\)", r"\bsi\s+units?\b",
    r"\(\s*metres?\s*\)", r"\(\s*meters?\s*\)",
    r"\bin\s+metres?\b", r"\bin\s+meters?\b",
    r"\(\s*kilometres?\s*\)", r"\(\s*kilometers?\s*\)",
    r"\bin\s+kilometres?\b", r"\bin\s+kilometers?\b",
    r"\(\s*kilograms?\s*\)", r"\bin\s+kilograms?\b",
    r"\(\s*kg\s*\)", r"\bin\s+kg\b",
    r"\(\s*km\s*\)", r"\bin\s+km\b",
    r"\(\s*m\s*\)", r"\(\s*m/s\s*\)",
))


def detect_unit_convention(table_idx: int, doc_json: dict) -> str:
    """Return 'imperial' or 'metric' based on the table caption and the prose
    immediately preceding the table's `#/tables/N` ref in body.children.

    `metric` is the default — chosen on no signal AND on equal-evidence cases.
    `imperial` requires unambiguous markers in caption or adjacent prose
    (an imperial marker present + outnumbering or matching metric markers).
    Asymmetric bias is intentional: SI is the modern technical default; we
    only flip when the document declares otherwise.
    """
    haystack_parts: list[str] = []
    tables = doc_json.get("tables") or []
    if 0 <= table_idx < len(tables):
        table = tables[table_idx]
        # Table captions: list of refs to text items
        for cap in (table.get("captions") or []):
            if isinstance(cap, dict) and isinstance(cap.get("$ref"), str):
                ref = cap["$ref"]
                if ref.startswith("#/texts/"):
                    try:
                        idx = int(ref.split("/")[-1])
                    except ValueError:
                        continue
                    texts = doc_json.get("texts") or []
                    if 0 <= idx < len(texts):
                        haystack_parts.append(str(texts[idx].get("text") or ""))

    # Prose preceding the table ref in body.children: walk back up to 3 text
    # refs and aggregate their content.
    table_ref = f"#/tables/{table_idx}"
    body = doc_json.get("body") or {}
    body_children = (body.get("children") or []) if isinstance(body, dict) else []
    target_pos: int | None = None
    for i, child in enumerate(body_children):
        if isinstance(child, dict) and child.get("$ref") == table_ref:
            target_pos = i
            break
    if target_pos is not None:
        looked_back = 0
        for j in range(target_pos - 1, max(target_pos - 8, -1), -1):
            prev = body_children[j]
            if not isinstance(prev, dict):
                continue
            ref = prev.get("$ref", "")
            if not ref.startswith("#/texts/"):
                continue
            try:
                idx = int(ref.split("/")[-1])
            except ValueError:
                continue
            texts = doc_json.get("texts") or []
            if 0 <= idx < len(texts):
                haystack_parts.append(str(texts[idx].get("text") or ""))
                looked_back += 1
                if looked_back >= 3:
                    break

    haystack = "\n".join(haystack_parts)
    if not haystack.strip():
        return "metric"

    imperial_hits = sum(1 for p in _IMPERIAL_MARKERS if p.search(haystack))
    metric_hits = sum(1 for p in _METRIC_MARKERS if p.search(haystack))
    if imperial_hits >= 1 and imperial_hits >= metric_hits:
        return "imperial"
    return "metric"


def is_table_relevant_for_pass(pass_name: str, normalized_table: Any) -> bool:
    """Return True when `normalized_table` has row labels that match the
    active pass's expected row aliases.

    A table is relevant when:
      - the table is normalized non-OTHER (otherwise synth render is just
        a passthrough TABLE_WHOLE chunk and won't help),
      - the pass has an alias set defined,
      - at least one row label in the table appears in that alias set.

    Returns False for any pass not in PASS_TABLE_ROW_ALIASES (conservative —
    new numeric passes need explicit alias entries to opt in).
    """
    if pass_name in RAW_ONLY_PASSES:
        return False
    if pass_name not in SYNTH_ELIGIBLE_PASSES:
        return False
    if getattr(normalized_table, "shape", None) == Shape.OTHER:
        return False
    aliases = PASS_TABLE_ROW_ALIASES.get(pass_name)
    if not aliases:
        return False
    row_labels = {
        _normalize_row_label(c.row_label)
        for c in normalized_table.cells
        if c.row_label
    }
    return bool(row_labels & aliases)


def _replace_raw_table_refs_in_body_children(
    doc_json: dict,
    replacements_by_table_ref: dict[str, list[str]],
) -> int:
    """In-place substitution of `#/tables/N` $refs with synthesized `#/texts/M`
    $refs in body.children, preserving sibling order.

    Why in-place vs append: when the chunker walks body.children it does so in
    document order. Appending synth refs at the END moves the table evidence
    to the tail of the walk, which empirically hurts identity/prose passes
    that depend on page-order proximity between table content and the
    surrounding narrative. In-place replacement preserves that proximity.

    Behavior:
    - Walks `body` recursively, including nested `children`.
    - For each entry whose `$ref` matches a key in `replacements_by_table_ref`,
      replaces it with the list of synthesized `$ref` entries for that table
      (in order).
    - $refs not in the replacement map (raw `#/tables/N` for OTHER-shape
      tables, plain `#/texts/N` refs, picture refs, etc.) are left alone.
    - `tables[]` array is NOT touched — overlay machinery reads it directly.

    Returns the number of raw table $refs that were substituted (for logging
    and test assertions).
    """
    if not replacements_by_table_ref:
        return 0

    substituted = 0

    def _walk(node: Any) -> None:
        nonlocal substituted
        if not isinstance(node, dict):
            return
        children = node.get("children")
        if not isinstance(children, list):
            return
        new_children: list = []
        for c in children:
            ref = c.get("$ref") if isinstance(c, dict) else None
            if ref in replacements_by_table_ref:
                for synth_ref in replacements_by_table_ref[ref]:
                    new_children.append({"$ref": synth_ref})
                substituted += 1
            else:
                new_children.append(c)
                _walk(c)
        node["children"] = new_children

    body = doc_json.get("body")
    if isinstance(body, dict):
        _walk(body)
    return substituted


def _drop_raw_table_refs_from_body_children(
    doc_json: dict,
    normalized: list[NormalizedTable],
) -> int:
    """Remove `#/tables/N` $refs from body.children for normalized non-OTHER
    tables, leaving the entries in `tables[]` intact.

    Why this is necessary: HybridChunker walks `body.children` and, when it
    encounters a `#/tables/N` $ref, dereferences it through `tables[]` and
    emits its own flattened cell-by-cell representation
    ("Min Range, 1 = m. Min Range, 2 = 8000.") regardless of whether the
    text-mirror in `texts[]` was blanked by `_suppress_raw_table_texts`.
    That competes with — and usually wins over — the synthesized per-column
    blocks from `render_for_graph` for the same table.

    What this fix preserves:
    - `tables[]` entries are NOT touched. The Phase 0/0.5 overlay machinery
      (table_overlay, cross_entity_hints, alias_map_by_entity_type) reads
      `tables[]` directly and must remain functional.
    - Synthesized `#/texts/N` $refs that were appended by main.py for this
      same set of normalized tables are NOT removed.
    - Tables with shape == OTHER keep their `#/tables/N` $ref because
      they fall through to the raw rendering anyway.
    - Nested children (recursive walks) are also pruned — see the
      _recursive_drop helper.

    Returns the number of refs removed (for logging/test assertions).
    """
    non_other = {nt.table_index for nt in normalized if nt.shape != Shape.OTHER}
    if not non_other:
        return 0
    target_refs = {f"#/tables/{i}" for i in non_other}

    removed = 0

    def _is_raw_table_ref(child: Any) -> bool:
        return isinstance(child, dict) and child.get("$ref") in target_refs

    def _recursive_drop(node: Any) -> None:
        nonlocal removed
        if not isinstance(node, dict):
            return
        children = node.get("children")
        if isinstance(children, list):
            new_children = [c for c in children if not _is_raw_table_ref(c)]
            removed += len(children) - len(new_children)
            node["children"] = new_children
            for c in new_children:
                _recursive_drop(c)

    body = doc_json.get("body")
    if isinstance(body, dict):
        _recursive_drop(body)
    return removed


def _substitute_table_chunks(
    native_chunks: list[Any],
    normalized_by_table_idx: dict[int, NormalizedTable],
    render_fn: Callable[..., list[Any]],
    *,
    token_limit: int,
    summary_limit: int,
    min_table_tokens: int,
) -> list[Any]:
    """Per §10.1 of the design spec.

    Substitution decision tree:
    - non_table: pass through unchanged.
    - normalized table below min_table_tokens: pass through unchanged.
    - table_dominant: substitute entirely. Subsequent natives for the same
      table_idx are dropped (NormalizedTable.cells covers all content).
    - table_mixed above threshold: emit normalized chunks AND keep native
      (defensive — wide tables shouldn't reach this branch per spike findings).
    """
    seen_table_idx: set[int] = set()
    out: list[Any] = []
    for nc in native_chunks:
        cls, table_idx = _classify_native_chunk(nc, normalized_by_table_idx)

        if cls == "non_table":
            out.append(nc)
            continue

        nt = normalized_by_table_idx[table_idx]
        size = _normalized_table_size_tokens(nt)
        if size < min_table_tokens:
            out.append(nc)
            continue

        parent_headings = tuple(getattr(getattr(nc, "meta", None), "headings", None) or [])
        parent_table_ref = f"#/tables/{table_idx}"

        if cls == "table_dominant":
            if table_idx in seen_table_idx:
                continue
            seen_table_idx.add(table_idx)
            for etc in render_fn(nt, token_limit=token_limit, summary_limit=summary_limit):
                out.append(_NormalizedTableChunkAdapter(
                    etc=etc, parent_headings=parent_headings,
                    parent_table_ref=parent_table_ref,
                ))
            continue

        # cls == "table_mixed"
        logger.warning(
            "_substitute_table_chunks: table_mixed classification fired (table_idx=%d, "
            "dominant_share<0.8). HybridChunker merged this table with prose; "
            "emitting normalized chunks AND keeping native (degraded path).",
            table_idx,
        )
        if table_idx not in seen_table_idx:
            seen_table_idx.add(table_idx)
            for etc in render_fn(nt, token_limit=token_limit, summary_limit=summary_limit):
                out.append(_NormalizedTableChunkAdapter(
                    etc=etc, parent_headings=parent_headings,
                    parent_table_ref=parent_table_ref,
                ))
        out.append(nc)
    return out
