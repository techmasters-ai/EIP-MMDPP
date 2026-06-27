"""G1 recall-floor gate (guarded-ranker spec §3).

The gate mirrors the value-grounding LABEL's form without knowing values:
force-keep a chunk for pass P iff it contains a digit AND a unit token from
P's unit signature. Token matching is imported from field_value_grounding —
the SAME matcher the label uses; gate and label must never drift.
No equipment names, no config: the signature derives from schema field-name
suffixes only.
"""
import re
from collections.abc import Iterable

from app.services.field_value_grounding import _compiled_unit_re, has_unit_token, nfc, units_for

_DIGIT = re.compile(r"\d")


def signature_for_fields(field_names: Iterable[str]) -> tuple[str, ...]:
    """Sorted union of unit synonyms over the pass's field names."""
    units: set[str] = set()
    for name in field_names:
        units.update(units_for(name))
    return tuple(sorted(units))


def chunk_passes_unit_gate(text_nfc: str, signature: Iterable[str]) -> bool:
    """True iff the (already nfc()-folded) chunk text could ground ANY
    numeric+unit value of this pass — digit present + signature unit token."""
    if not signature or not text_nfc:
        return False
    return bool(_DIGIT.search(text_nfc)) and has_unit_token(text_nfc, signature)


def count_unit_tokens(text_nfc: str, signature: Iterable[str]) -> int:
    """Count of DISTINCT unit synonyms from ``signature`` that appear as bounded
    tokens in the already-nfc()-folded chunk text, capped at 20.

    Semantics — DISTINCT synonyms, not occurrences:
      "50 kw and 60 kw" with signature ("kw",) → 1  (one synonym matched)
      "50 kw at 60 mhz" with signature ("kw","mhz") → 2  (two synonyms matched)
    Capping at 20 prevents a pathologically long signature from inflating the
    feature unboundedly.

    Uses the same ``_compiled_unit_re`` cache that ``has_unit_token`` uses, so
    matching semantics are byte-identical across gate, label, and this counter —
    they can never drift.

    ``signature`` is already nfc()-folded (produced by ``signature_for_fields``
    which iterates ``units_for``, which returns literals from SUFFIX_UNITS — all
    ASCII/symbol).  The text is expected to be nfc()-folded by the caller (same
    contract as ``chunk_passes_unit_gate``).

    Returns 0 when ``signature`` is empty or ``text_nfc`` is empty.
    """
    if not signature or not text_nfc:
        return 0
    count = 0
    for syn in signature:
        if count >= 20:
            break
        syn_nfc = nfc(syn)
        if _compiled_unit_re(syn_nfc).search(text_nfc):
            count += 1
    return count
