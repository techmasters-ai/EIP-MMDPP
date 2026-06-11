"""Tasks C2+C3 — in-memory lexical alias and regex pattern search over chunk rows.

Both functions are PURE (no DB / no network calls). They operate over
pre-fetched ExtractionChunk row dicts (as returned by
``fetch_extraction_chunks_for_run`` in ``extraction_chunk_search.py``) and a
sequence of ``FieldRetrievalQuery`` objects (from ``extraction_query_builder``).

NFC normalisation
-----------------
All text comparisons NFC-normalise BOTH sides before matching.  This is
required for the SA-2 document, which is in Russian and may contain Cyrillic
characters in either NFC (precomposed) or NFD (decomposed) form depending on
how the PDF was encoded.  Without normalisation, visually identical strings
fail substring tests because their byte representations differ.

Candidate key
-------------
The stable key for each row is ``row.get("vertex_id") or row.get("self_ref")``.
``vertex_id`` is the synthetic PK; when absent or None, ``self_ref`` is the
string identity used by the rest of the pipeline.

These functions are consumed downstream by C4 (merge) and C5 (scoring).
They do NOT wire into the VR endpoint (C6) and do NOT compute final scores.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Sequence

from app.services.extraction_chunk_index import (
    read_chunk_headings,
    read_chunk_section_path,
)
from app.services.extraction_query_builder import FieldRetrievalQuery
from app.services.field_value_grounding import unit_token_regex


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nfc(text: str) -> str:
    """Return the NFC-normalised form of *text*.

    Shared by both C2 and C3 to ensure identical normalisation behaviour.
    """
    return unicodedata.normalize("NFC", text)


def _candidate_key(row: dict) -> str:
    """Return the stable candidate key for a chunk row.

    Prefers ``vertex_id`` (synthetic PK) when present and non-None;
    falls back to ``self_ref``.
    """
    return row.get("vertex_id") or row["self_ref"]


@lru_cache(maxsize=512)
def _compiled_keyword_re(needle_nfc: str) -> re.Pattern[str]:
    """Return a compiled token-bounded pattern for a single-token keyword needle.

    Reuses :func:`~app.services.field_value_grounding.unit_token_regex` — the
    same Unicode-aware boundary builder used by the unit-matcher — so boundary
    semantics are identical across the codebase.

    Called only for needles that contain no whitespace after NFC+casefold
    normalisation (i.e. single-token needles).  Hyphenated tokens such as
    "x-band" count as single tokens: ``unit_token_regex`` applies leading and
    trailing guards at the *outer edges* only, which is correct.

    Cached via ``lru_cache`` so each distinct needle is compiled exactly once
    per process lifetime, mirroring how ``_compiled_unit_re`` works in
    ``extraction_unit_gate``.
    """
    return re.compile(unit_token_regex(needle_nfc))


def _compile_patterns(
    field_queries: Sequence[FieldRetrievalQuery],
) -> list[tuple[FieldRetrievalQuery, list[re.Pattern[str] | str]]]:
    """Pre-compile all evidence_patterns for C3 (fail-fast on bad regexes).

    Each entry in the returned list corresponds to one ``FieldRetrievalQuery``.
    Each inner element is either:
    - a compiled ``re.Pattern`` (for ``re:``-prefixed patterns), or
    - a plain NFC+casefolded ``str`` (for literal-phrase patterns).

    Invalid ``re:`` patterns raise ``re.error`` here — at function entry —
    rather than silently at match time, so unit tests surface them immediately.
    """
    compiled: list[tuple[FieldRetrievalQuery, list[re.Pattern[str] | str]]] = []
    for fq in field_queries:
        fq_compiled: list[re.Pattern[str] | str] = []
        for pattern in fq.evidence_patterns:
            if pattern.startswith("re:"):
                raw = pattern[3:]
                # Raises re.error on invalid regex — intentional (fail-fast).
                fq_compiled.append(re.compile(raw, re.IGNORECASE | re.MULTILINE))
            else:
                # Literal phrase: pre-normalise so we don't repeat per-chunk.
                fq_compiled.append(_nfc(pattern).casefold())
        compiled.append((fq, fq_compiled))
    return compiled


# ---------------------------------------------------------------------------
# C2 — lexical alias search
# ---------------------------------------------------------------------------

def lexical_hit_counts(
    rows: list[dict],
    field_queries: Sequence[FieldRetrievalQuery],
) -> dict[str, dict]:
    """Return per-chunk lexical alias and negative-term hit counts.

    Parameters
    ----------
    rows:
        List of ExtractionChunk row dicts (from ``fetch_extraction_chunks_for_run``).
        Each must have ``self_ref`` (str) and ``chunk_text`` (str).  May also
        carry ``vertex_id`` (str | None) for the candidate key.
    field_queries:
        Sequence of ``FieldRetrievalQuery`` objects.  Each contributes its
        ``.aliases`` and ``.negative_terms`` to the per-chunk counts.

    Returns
    -------
    dict[str, dict]
        Keyed by candidate_key.  Each value has:
        - ``alias_hits``     (int) — total alias substring matches across all
          field_queries.  Negative terms are excluded.
        - ``negative_hits``  (int) — total negative_term substring matches across
          all field_queries.  Tracked SEPARATELY from alias_hits.
        - ``supported_fields`` (set[str]) — field names whose aliases matched at
          least once in this chunk.
    """
    # Pre-normalise aliases and negative terms for all field queries.
    # Structure: list of (field_name, normalised_aliases, normalised_negs)
    prepped: list[tuple[str, list[str], list[str]]] = []
    for fq in field_queries:
        norm_aliases = [_nfc(a).casefold() for a in fq.aliases]
        norm_negs = [_nfc(n).casefold() for n in fq.negative_terms]
        prepped.append((fq.field_name, norm_aliases, norm_negs))

    result: dict[str, dict] = {}

    for row in rows:
        key = _candidate_key(row)
        haystack = _nfc(row.get("chunk_text") or "").casefold()

        alias_hits = 0
        negative_hits = 0
        supported_fields: set[str] = set()

        for field_name, norm_aliases, norm_negs in prepped:
            field_matched = False
            for alias in norm_aliases:
                if alias in haystack:
                    alias_hits += 1
                    field_matched = True
            if field_matched:
                supported_fields.add(field_name)
            for neg in norm_negs:
                if neg in haystack:
                    negative_hits += 1

        result[key] = {
            "alias_hits": alias_hits,
            "negative_hits": negative_hits,
            "supported_fields": supported_fields,
        }

    return result


# ---------------------------------------------------------------------------
# KEYWORD — per-pass lexical_keywords search (decomposed-lexical feature)
# ---------------------------------------------------------------------------

def keyword_hit_counts(
    rows: list[dict],
    keywords: Sequence[str | None],
) -> dict[str, dict]:
    """Return per-chunk per-pass KEYWORD hit counts.

    Normalisation: NFC + casefold (identical to :func:`lexical_hit_counts`).
    Keying: ``vertex_id`` preferred, ``self_ref`` fallback.

    Matching semantics (post-fix):
    - **Single-token needles** (no whitespace after NFC+casefold) are matched
      with Unicode-aware word boundaries via
      :func:`~app.services.field_value_grounding.unit_token_regex`:
      leading guard ``(?<![^\\W\\d_])`` (not preceded by a Unicode letter) and
      trailing guard ``(?!\\w)`` (not followed by a Unicode word char).  This
      prevents "mach" from matching inside "machinery", "fins" from matching
      inside "muffins", etc.  Hyphenated tokens ("x-band") count as single
      tokens; boundary guards apply at the OUTER edges only (the hyphen is
      interior, not a word boundary).  Trailing-digit suppression: "mach" does
      NOT match "mach2" (trailing digit is ``\\w``); "mach 2" matches because
      the space separates "mach" from "2".
    - **Multi-word phrases** (contain whitespace after folding) use the
      original NFC+casefold SUBSTRING semantics — no boundary wrapping.  A
      phrase like "pulse repetition interval" is expected to appear inside a
      larger sentence; adding outer boundaries to phrases would silently drop
      legitimate matches.

    Counting quirks (preserved for backward-compat):
    - Binary presence per needle: each needle that matches the chunk
      contributes 1 regardless of how many times it appears in the text.
    - Duplicate needle entries each contribute independently: a keyword
      listed twice in ``keywords`` will add 2 if it matches (mirrors the
      ``lexical_hit_counts`` per-alias counting convention).

    Scope guard: the ``lexical_hit_counts`` alias channel (production
    final_score, lexical_weight 0.20) is INTENTIONALLY left on plain substring
    semantics.  Only THIS function (per-pass keyword channel) uses word-bounded
    matching.

    Parameters
    ----------
    rows:
        List of ExtractionChunk row dicts. Each must have ``self_ref`` (str)
        and ``chunk_text`` (str). May also carry ``vertex_id`` (str | None).
    keywords:
        Per-pass keyword needles to match against each chunk's body text.
        Blank / ``None`` entries are skipped (they would otherwise match every
        chunk). Empty list → every row scores ``keyword_hits == 0``.

    Returns
    -------
    dict[str, dict]
        Keyed by candidate_key. Each value has:
        - ``keyword_hits`` (int) — number of ``keywords`` found in the chunk's
          body text.

    Pure: no DB / no network. Consumed by C4 ``merge_candidates`` via the
    ``lexical_hits`` payload (``keyword_hits`` key) → ``pass_keyword_hits``.
    """
    # Pre-normalise keywords once (NFC + casefold), dropping blanks / Nones so
    # an empty needle can't substring-match every chunk body.
    #
    # For each needle, pre-decide the matching strategy so we pay the
    # "has whitespace?" check once per keyword rather than once per (keyword, row).
    #
    # Single-token needles → compiled bounded regex (cached by _compiled_keyword_re).
    # Multi-word phrases  → plain substring (original semantics, no boundary change).
    norm_kw_matchers: list[tuple[str, re.Pattern[str] | None]] = []
    for k in keywords:
        if not (k and k.strip()):
            continue
        needle = _nfc(k).casefold()
        if " " in needle:
            # Multi-word phrase: substring semantics unchanged.
            norm_kw_matchers.append((needle, None))
        else:
            # Single-token: bounded match via unit_token_regex.
            norm_kw_matchers.append((needle, _compiled_keyword_re(needle)))

    result: dict[str, dict] = {}

    for row in rows:
        key = _candidate_key(row)
        haystack = _nfc(row.get("chunk_text") or "").casefold()

        keyword_hits = 0
        if haystack:
            for needle, pat in norm_kw_matchers:
                if pat is not None:
                    # Single-token: bounded regex search.
                    if pat.search(haystack):
                        keyword_hits += 1
                else:
                    # Multi-word phrase: plain substring.
                    if needle in haystack:
                        keyword_hits += 1

        result[key] = {"keyword_hits": keyword_hits}

    return result


# ---------------------------------------------------------------------------
# SECTION — section-heading anchor search (router-scoring section signal, v1)
# ---------------------------------------------------------------------------

def section_hit_counts(
    rows: list[dict],
    anchors: Sequence[str],
    likely_sections: Sequence[str] = (),
) -> dict[str, dict]:
    """Return per-chunk SECTION-heading anchor hit counts.

    Mirrors :func:`lexical_hit_counts` exactly in normalisation and keying, but
    the haystack is the chunk's HEADING hierarchy (the ``section_path`` /
    ``headings`` projection added in router-scoring Part 1) rather than the
    chunk body text, and the needles are the ``anchors`` — the already
    type-matched committed entity names supplied by the caller.

    Parameters
    ----------
    rows:
        List of ExtractionChunk row dicts (from
        ``fetch_extraction_chunks_for_run``). Each must have ``self_ref``
        (str); may carry ``vertex_id`` (str | None) for the candidate key and
        ``section_path`` (str | None) / ``headings`` (list) projected in Part
        1. Legacy rows (indexed before Part 1) lack the heading columns — the
        ``read_chunk_*`` accessors None/[]-coalesce, yielding an empty haystack
        and therefore ``section_hits == 0``.
    anchors:
        Committed entity names to match against each chunk's heading haystack.
        Blank / ``None`` entries are skipped (they would otherwise match every
        chunk). Each anchor that appears in the haystack contributes 1 to the
        count; the SAME anchor listed twice contributes twice (mirrors
        ``lexical_hit_counts`` per-alias counting).
    likely_sections:
        Anchor-INDEPENDENT section needles. These are the pass's expected
        section-name strings (schema-typed, instance-free) — the union of each
        field's ``json_schema_extra['retrieval']['likely_sections']``. Unlike
        ``anchors`` (committed entity NAMES, which are empty at field-pass
        routing time), they are available whenever the pass runs, so they give
        the otherwise-constant ``section_norm`` feature a real source. Matched
        against the SAME heading haystack as ``anchors`` (NFC + casefold
        substring). Blank / ``None`` entries are skipped. Each matching term
        contributes 1, ADDED to the anchor matches for that chunk. Defaults to
        ``()`` → byte-identical to the anchor-only behaviour.

    Returns
    -------
    dict[str, dict]
        Keyed by candidate_key (``vertex_id`` preferred, ``self_ref``
        fallback). Each value has:
        - ``section_hits`` (int) — number of ``anchors`` PLUS ``likely_sections``
          terms found in the chunk's heading haystack.

    Pure: no DB / no network. Consumed by C4 ``merge_candidates`` as the
    ``section_meta`` argument.
    """
    # Pre-normalise BOTH needle sources once (NFC + casefold), dropping blanks /
    # Nones so an empty needle can't substring-match every heading. The two
    # contributions are additive but tracked through the same single pass over
    # the haystack — anchor matches + likely_section matches → section_hits.
    norm_anchors = [
        _nfc(a).casefold()
        for a in anchors
        if a and a.strip()
    ]
    norm_likely_sections = [
        _nfc(s).casefold()
        for s in likely_sections
        if s and s.strip()
    ]

    result: dict[str, dict] = {}

    for row in rows:
        key = _candidate_key(row)

        # Build the section haystack from the chunk's heading projection.
        # ``read_chunk_headings`` returns list[str] (outermost -> innermost);
        # ``read_chunk_section_path`` returns the joined breadcrumb or None.
        # Search BOTH so a row carrying only one of the two columns still
        # matches. Join with newlines so distinct heading levels can't form
        # an accidental cross-boundary substring.
        heading_parts = list(read_chunk_headings(row))
        section_path = read_chunk_section_path(row)
        if section_path:
            heading_parts.append(section_path)
        haystack = _nfc("\n".join(heading_parts)).casefold()

        section_hits = 0
        if haystack:
            for anchor in norm_anchors:
                if anchor in haystack:
                    section_hits += 1
            for section in norm_likely_sections:
                if section in haystack:
                    section_hits += 1

        result[key] = {"section_hits": section_hits}

    return result


# ---------------------------------------------------------------------------
# C3 — pattern (regex) search
# ---------------------------------------------------------------------------

def pattern_hit_counts(
    rows: list[dict],
    field_queries: Sequence[FieldRetrievalQuery],
    pattern_hit_limit: int = 50,
) -> dict[str, dict]:
    """Return per-chunk evidence-pattern hit counts.

    Each ``evidence_pattern`` is treated as:
    - A **literal phrase** (NFC+casefold substring match) if it does NOT start
      with ``re:``.
    - A **regex** compiled with ``re.IGNORECASE | re.MULTILINE`` if it DOES
      start with ``re:`` (the prefix is stripped before compilation).

    Invalid ``re:`` patterns raise ``re.error`` at function entry (patterns are
    compiled up-front, not lazily per row).

    Parameters
    ----------
    rows:
        List of ExtractionChunk row dicts.
    field_queries:
        Sequence of ``FieldRetrievalQuery`` objects.
    pattern_hit_limit:
        Maximum number of diagnostic match samples retained internally.
        Does NOT cap the ``pattern_hits`` count or the number of result entries
        — every row still gets its own result dict with the correct count.

    Returns
    -------
    dict[str, dict]
        Keyed by candidate_key.  Each value has:
        - ``pattern_hits``    (int) — total pattern matches across all field_queries.
        - ``supported_fields`` (set[str]) — field names whose patterns matched at
          least once in this chunk.
    """
    # Compile all patterns up-front — raises re.error on invalid regex immediately.
    compiled_fqs = _compile_patterns(field_queries)

    result: dict[str, dict] = {}

    for row in rows:
        key = _candidate_key(row)
        # NFC-normalise the haystack (same step as C2; casefold for literals).
        raw_text = row.get("chunk_text") or ""
        nfc_text = _nfc(raw_text)
        # For literal matches we compare casefold; for regex we use the NFC
        # haystack and rely on re.IGNORECASE for case-insensitivity.
        nfc_casefolded = nfc_text.casefold()

        pattern_hits = 0
        supported_fields: set[str] = set()

        for fq, compiled_patterns in compiled_fqs:
            field_matched = False
            for pat in compiled_patterns:
                if isinstance(pat, re.Pattern):
                    matches = pat.findall(nfc_text)
                    count = len(matches)
                else:
                    # Literal phrase: count non-overlapping occurrences
                    # (str.count is sufficient for our use-case).
                    count = nfc_casefolded.count(pat)
                if count > 0:
                    pattern_hits += count
                    field_matched = True
            if field_matched:
                supported_fields.add(fq.field_name)

        result[key] = {
            "pattern_hits": pattern_hits,
            "supported_fields": supported_fields,
        }

    return result
