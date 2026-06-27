"""Shared v2 quality-filter primitives.

This module is the single source of truth for the rules used to decide which
DoclingDocument text elements survive into LLM extraction context. Both the
ExtractionChunk indexer (graph-extraction retrieval pool) and the
doc-level filter applied to every pass via the worker boundary consume
from here.

Pure functions — no I/O, no docling/ArcadeDB imports. Easy to unit-test.

The general-retrieval / RAG embedding phase via TextChunk creation does NOT import from this module by design: it preserves every chunk regardless of extraction-time quality.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum normalized length BEFORE stripping. Catches single symbols
#: ("™"), one-word nav fragments ("Log in"), and OCR-spurious tokens.
MIN_CHUNK_TEXT_CHARS = 20

#: Minimum residual length AFTER stripping leading/trailing chrome lines.
#: A chunk that survives the < MIN_CHUNK_TEXT_CHARS check but whose body
#: is mostly chrome (residue < MIN_RESIDUAL_CHARS after strip) is dropped.
MIN_RESIDUAL_CHARS = 40

#: Phrases that constitute "chrome" when they make up the ENTIRE normalized
#: content of a single line. Whole-line equality (not substring) — a real
#: paragraph that embeds "subscribe now" mid-body keeps the paragraph intact.
WEB_CHROME_PHRASES: frozenset[str] = frozenset({
    "audio coming soon",
    "subscribe now",
    "sponsored",
    "advertisement",
    "log in",
    "share this article",
    "historynet",
    "recommended for you",
    "related stories",
    "sign me up",
    "advertise with us",
    "meet our staff",
    "privacy policy terms of service",
    "stay curious",
    "apa",
    "mla",
    "chicago",
})


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def normalize_for_dedup(text: str) -> str:
    """Lowercase + collapse whitespace + strip edges. Returns the dedup key."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_chrome_line(line: str) -> bool:
    """True if `line` is whitespace-only OR its normalized form exactly
    matches a chrome phrase.

    Whitespace-only lines are treated as chrome so the strip pass can
    pull them along when they sit between/adjacent to real chrome lines
    (typical PDF chrome block: ``"Audio Coming Soon\\n\\nSponsored..."``).
    """
    norm = re.sub(r"\s+", " ", line.lower()).strip()
    if not norm:
        return True
    return norm in WEB_CHROME_PHRASES


def strip_chrome_lines(text: str) -> tuple[str, int, int]:
    """Strip leading and trailing chrome (or empty) lines from `text`.

    Middle chrome lines are NEVER stripped — preserves real article body
    where boilerplate is embedded between paragraphs of useful text.

    Returns ``(stripped_text, leading_removed, trailing_removed)``.
    When the entire chunk is chrome/empty, returns ``("", n, 0)``.
    """
    lines = text.split("\n")
    n = len(lines)

    leading = 0
    while leading < n and is_chrome_line(lines[leading]):
        leading += 1
    if leading == n:
        return "", n, 0

    trailing = 0
    # Stop at the first non-chrome line counting from the end. Strictly > leading
    # guards against removing the only kept line in single-line residues.
    while (n - 1 - trailing) > leading and is_chrome_line(lines[n - 1 - trailing]):
        trailing += 1

    return "\n".join(lines[leading:n - trailing]), leading, trailing


# ---------------------------------------------------------------------------
# Unified decision predicate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterDecision:
    """One decision about a single chunk's fate.

    Fields
    ------
    keep:
        True if the chunk should be retained for extraction. False if it
        should be blanked / not indexed.
    stripped_text:
        The post-strip text to insert when ``keep`` is True AND chrome was
        actually stripped. None when no stripping happened OR ``keep`` is False.
    reason:
        One of ``"kept"``, ``"stripped"``, ``"short"``, ``"after_strip"``,
        ``"dedup"``. Used for diagnostic counters and example logging.
    """

    keep: bool
    stripped_text: str | None
    reason: str


def classify_chunk(
    rendered: str,
    seen_norms: set[str],
    *,
    skip_short_reject: bool = False,
    gate_after_strip_on_chrome: bool = False,
) -> FilterDecision:
    """Apply the v2 quality rules to a single rendered chunk.

    Order matches the contract documented in the C.9d-v2 commit:
    1. Quick reject: normalized < MIN_CHUNK_TEXT_CHARS -> "short"
       (skipped when ``skip_short_reject=True``)
    2. Strip leading/trailing chrome lines
    3. Drop if stripped-normalized < MIN_RESIDUAL_CHARS -> "after_strip"
       (when ``gate_after_strip_on_chrome=True`` this only fires if leading
       or trailing chrome was actually stripped; otherwise short legitimate
       content is preserved for downstream merging)
    4. Dedup against ``seen_norms`` (set of post-strip normalized keys);
       returns "dedup" if already present
    5. Otherwise: kept (or "stripped" if chrome was removed)

    SIDE EFFECT: when the chunk is kept, its post-strip normalized form
    is added to ``seen_norms`` so subsequent calls dedup against it.

    Parameters
    ----------
    rendered:
        The chunk's rendered text (may contain newlines).
    seen_norms:
        Caller-managed set of already-seen post-strip normalized keys.
        Mutated when a chunk is kept.
    skip_short_reject:
        When True, Rule 1 (< MIN_CHUNK_TEXT_CHARS) is not applied. Used by
        Layer-1 (``filter_docling_document``) where the chunker downstream
        (HybridChunker in docling-graph) merges peer siblings under a heading
        and short individual entries gain meaning only post-merge.
    gate_after_strip_on_chrome:
        When True, Rule 2 only fires when chrome was actually stripped
        (``leading + trailing > 0``). Used by Layer-1 to avoid blanking
        legitimate short content that has no chrome to strip — these
        fragments gain meaning post-merge in HybridChunker. Pure-chrome
        entries still get blanked because their residue is empty AND chrome
        was stripped.
    """
    normalized = normalize_for_dedup(rendered)
    if not skip_short_reject and len(normalized) < MIN_CHUNK_TEXT_CHARS:
        return FilterDecision(keep=False, stripped_text=None, reason="short")

    stripped, leading, trailing = strip_chrome_lines(rendered)
    stripped_normalized = normalize_for_dedup(stripped)
    chrome_was_stripped = (leading + trailing) > 0
    after_strip_applies = chrome_was_stripped if gate_after_strip_on_chrome else True
    if after_strip_applies and len(stripped_normalized) < MIN_RESIDUAL_CHARS:
        return FilterDecision(keep=False, stripped_text=None, reason="after_strip")

    if stripped_normalized in seen_norms:
        return FilterDecision(keep=False, stripped_text=None, reason="dedup")

    seen_norms.add(stripped_normalized)

    if leading > 0 or trailing > 0:
        return FilterDecision(keep=True, stripped_text=stripped, reason="stripped")
    return FilterDecision(keep=True, stripped_text=None, reason="kept")
