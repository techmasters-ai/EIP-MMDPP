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
from typing import Sequence

from app.services.extraction_query_builder import FieldRetrievalQuery


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
