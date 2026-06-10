"""G1 recall-floor gate (guarded-ranker spec §3).

The gate mirrors the value-grounding LABEL's form without knowing values:
force-keep a chunk for pass P iff it contains a digit AND a unit token from
P's unit signature. Token matching is imported from field_value_grounding —
the SAME matcher the label uses; gate and label must never drift.
No equipment names, no config: the signature derives from schema field-name
suffixes only.
"""
import re

from app.services.field_value_grounding import has_unit_token, units_for

_DIGIT = re.compile(r"\d")


def signature_for_fields(field_names) -> tuple[str, ...]:
    """Sorted union of unit synonyms over the pass's field names."""
    units: set[str] = set()
    for name in field_names:
        units.update(units_for(name))
    return tuple(sorted(units))


def chunk_passes_unit_gate(text_nfc: str, signature) -> bool:
    """True iff the (already nfc()-folded) chunk text could ground ANY
    numeric+unit value of this pass — digit present + signature unit token."""
    if not signature or not text_nfc:
        return False
    return bool(_DIGIT.search(text_nfc)) and has_unit_token(text_nfc, signature)
