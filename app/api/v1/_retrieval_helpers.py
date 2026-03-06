"""Pure helper functions for the retrieval endpoint.

Kept in a separate module so unit tests can import them without triggering
the DB session / asyncpg dependency chain (same pattern as _agent_helpers.py).
"""

import re

from app.schemas.retrieval import QueryResultItem, UnifiedQueryRequest

# ---------------------------------------------------------------------------
# Config-backed constants (lazy-loaded to avoid import-time settings issues)
# ---------------------------------------------------------------------------

_settings_cache = None


def _settings():
    global _settings_cache
    if _settings_cache is None:
        from app.config import get_settings
        _settings_cache = get_settings()
    return _settings_cache


# Legacy decay — read from env, used when chunk_links unavailable (fallback)
def get_cross_modal_decay() -> float:
    return _settings().retrieval_cross_modal_decay


def get_ontology_decay() -> float:
    return _settings().retrieval_ontology_decay


# Keep module-level names for backward compat (read once at first use)
class _LazyFloat:
    def __init__(self, getter):
        self._getter = getter
        self._value = None

    def __float__(self):
        if self._value is None:
            self._value = self._getter()
        return self._value

    def __mul__(self, other):
        return float(self) * other

    def __rmul__(self, other):
        return other * float(self)

    def __repr__(self):
        return repr(float(self))


CROSS_MODAL_DECAY = _LazyFloat(get_cross_modal_decay)
ONTOLOGY_DECAY = _LazyFloat(get_ontology_decay)


# ---------------------------------------------------------------------------
# Fusion weight getters (all from env vars)
# ---------------------------------------------------------------------------

def get_fusion_weights() -> tuple[float, float, float]:
    """Return (semantic, doc_structure, ontology) weights."""
    s = _settings()
    return s.retrieval_semantic_weight, s.retrieval_doc_structure_weight, s.retrieval_ontology_weight


def get_ontology_relation_weights() -> dict[str, float]:
    s = _settings()
    return {
        "IS_VARIANT_OF": s.retrieval_onto_weight_is_variant_of,
        "USES_COMPONENT": s.retrieval_onto_weight_uses_component,
        "IS_SUBSYSTEM_OF": s.retrieval_onto_weight_is_subsystem_of,
        "CONTAINS": s.retrieval_onto_weight_contains,
        "PART_OF": s.retrieval_onto_weight_part_of,
        "INTERFACES_WITH": s.retrieval_onto_weight_interfaces_with,
        "OPERATES_ON": s.retrieval_onto_weight_operates_on,
        "MEETS_STANDARD": s.retrieval_onto_weight_meets_standard,
        "RELATED_TO": s.retrieval_onto_weight_related_to,
    }


def get_doc_link_weights() -> dict[str, float]:
    s = _settings()
    return {
        "NEXT_CHUNK": s.retrieval_weight_next_chunk,
        "SAME_SECTION": s.retrieval_weight_same_section,
        "SAME_ARTIFACT": s.retrieval_weight_same_artifact,
        "SAME_PAGE": s.retrieval_weight_same_page,
    }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_results(results: list[QueryResultItem]) -> list[QueryResultItem]:
    """Deduplicate by chunk_id, keeping the highest-scoring entry."""
    best: dict[str, QueryResultItem] = {}
    for r in results:
        key = str(r.chunk_id) if r.chunk_id else str(id(r))
        if key not in best or r.score > best[key].score:
            best[key] = r
    return list(best.values())


# ---------------------------------------------------------------------------
# Parameterized filter builders (SQL injection safe)
# ---------------------------------------------------------------------------

def build_text_filters(body: UnifiedQueryRequest) -> tuple[str, dict]:
    """Return (WHERE clause suffix, bind params dict).

    Uses parameterized queries to prevent SQL injection.
    """
    clauses = ""
    params: dict = {}
    if body.filters:
        if body.filters.classification:
            clauses += " AND tc.classification = :filter_classification"
            params["filter_classification"] = body.filters.classification
        if body.filters.document_ids:
            clauses += " AND tc.document_id = ANY(:filter_doc_ids)"
            params["filter_doc_ids"] = [str(d) for d in body.filters.document_ids]
        if body.filters.modalities:
            clauses += " AND tc.modality = ANY(:filter_modalities)"
            params["filter_modalities"] = body.filters.modalities
        if body.filters.source_ids:
            clauses += " AND tc.document_id IN (SELECT id FROM ingest.documents WHERE source_id = ANY(:filter_source_ids))"
            params["filter_source_ids"] = [str(s) for s in body.filters.source_ids]
    return clauses, params


def build_image_filters(body: UnifiedQueryRequest) -> tuple[str, dict]:
    """Return (WHERE clause suffix, bind params dict).

    Uses parameterized queries to prevent SQL injection.
    """
    clauses = ""
    params: dict = {}
    if body.filters:
        if body.filters.classification:
            clauses += " AND ic.classification = :filter_classification"
            params["filter_classification"] = body.filters.classification
        if body.filters.document_ids:
            clauses += " AND ic.document_id = ANY(:filter_doc_ids)"
            params["filter_doc_ids"] = [str(d) for d in body.filters.document_ids]
        if body.filters.modalities:
            clauses += " AND ic.modality = ANY(:filter_modalities)"
            params["filter_modalities"] = body.filters.modalities
        if body.filters.source_ids:
            clauses += " AND ic.document_id IN (SELECT id FROM ingest.documents WHERE source_id = ANY(:filter_source_ids))"
            params["filter_source_ids"] = [str(s) for s in body.filters.source_ids]
    return clauses, params


# ---------------------------------------------------------------------------
# Fusion scoring
# ---------------------------------------------------------------------------

MIL_ID_PATTERNS = [
    re.compile(r'\b\d{4}-\d{2}-\d{3}-\d{4}\b'),  # NSN
    re.compile(r'\bMIL-[A-Z]+-\d+[A-Z]?\b'),       # MIL-STD
    re.compile(r'\bPN-\d{4,}\b'),                    # Part number
]


def compute_fusion_score(
    semantic_score: float,
    doc_structure_weight: float = 0.0,
    doc_structure_hops: int = 0,
    ontology_rel_type: str | None = None,
    ontology_hops: int = 0,
    content_text: str | None = None,
    query_text: str | None = None,
) -> float:
    """Compute weighted fusion score for a candidate chunk.

    All weights are read from app.config.settings (env-var configurable).
    """
    s = _settings()
    sem_w, doc_w, onto_w = get_fusion_weights()
    hop_base = s.retrieval_hop_penalty_base

    # Document-structure component
    doc_score = 0.0
    if doc_structure_weight > 0:
        hop_penalty = hop_base ** max(0, doc_structure_hops - 1)
        doc_score = doc_structure_weight * hop_penalty

    # Ontology component
    onto_score = 0.0
    if ontology_rel_type:
        rel_weights = get_ontology_relation_weights()
        rel_weight = rel_weights.get(ontology_rel_type, s.retrieval_onto_weight_default)
        hop_penalty = hop_base ** max(0, ontology_hops - 1)
        onto_score = rel_weight * hop_penalty

    # Weighted fusion
    final = sem_w * semantic_score + doc_w * doc_score + onto_w * onto_score

    # Military identifier bonus
    mil_bonus = s.retrieval_mil_id_bonus
    if content_text and query_text and mil_bonus > 0:
        for pattern in MIL_ID_PATTERNS:
            query_ids = set(pattern.findall(query_text))
            if query_ids:
                content_ids = set(pattern.findall(content_text))
                if query_ids & content_ids:
                    final = min(final + mil_bonus, 1.0)
                    break

    return round(final, 6)
