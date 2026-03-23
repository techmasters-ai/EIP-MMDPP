"""Pure helper functions for the retrieval endpoint.

Kept in a separate module so unit tests can import them without triggering
the DB session / asyncpg dependency chain (same pattern as _agent_helpers.py).
"""

import re
from functools import lru_cache
from pathlib import Path

import yaml

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


# ---------------------------------------------------------------------------
# Ontology relation weights (loaded from ontology YAML)
# ---------------------------------------------------------------------------

_ONTOLOGY_PATH = Path(__file__).resolve().parent.parent.parent.parent / "ontology" / "ontology.yaml"


@lru_cache(maxsize=1)
def _load_scoring_weights() -> dict[str, float]:
    """Load ontology relation scoring weights from ontology.yaml."""
    with open(_ONTOLOGY_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("scoring_weights", {})


def get_ontology_relation_weights() -> dict[str, float]:
    return _load_scoring_weights()


# ---------------------------------------------------------------------------
# Fusion weight getters (all from env vars)
# ---------------------------------------------------------------------------

def get_fusion_weights() -> tuple[float, float, float]:
    """Return (semantic, doc_structure, ontology) weights."""
    s = _settings()
    return s.retrieval_semantic_weight, s.retrieval_doc_structure_weight, s.retrieval_ontology_weight


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


def diversify_results(results: list[QueryResultItem]) -> list[QueryResultItem]:
    """Content-level dedupe: keep highest-scoring entry per unique text within same doc+page.

    Non-text modalities (image, graph_node) and items without content_text pass through.
    """
    best: dict[str, QueryResultItem] = {}
    for r in results:
        if r.modality in ("image", "graph_node") or not r.content_text:
            best[str(id(r))] = r
            continue
        normalized = r.content_text.strip().lower()
        key = f"{r.document_id}:{r.page_number}:{normalized}"
        if key not in best or r.score > best[key].score:
            best[key] = r
    return list(best.values())


# ---------------------------------------------------------------------------
# Parameterized filter builders (SQL injection safe)
# ---------------------------------------------------------------------------

def build_filters(body: UnifiedQueryRequest, alias: str = "tc") -> tuple[str, dict]:
    """Return (WHERE clause suffix, bind params dict).

    Uses parameterized queries to prevent SQL injection.
    """
    clauses = ""
    params: dict = {}
    if body.filters:
        if body.filters.classification:
            clauses += f" AND {alias}.classification = :filter_classification"
            params["filter_classification"] = body.filters.classification
        if body.filters.document_ids:
            clauses += f" AND {alias}.document_id = ANY(:filter_doc_ids)"
            params["filter_doc_ids"] = [str(d) for d in body.filters.document_ids]
        if body.filters.modalities:
            clauses += f" AND {alias}.modality = ANY(:filter_modalities)"
            params["filter_modalities"] = body.filters.modalities
        if body.filters.source_ids:
            clauses += f" AND {alias}.document_id IN (SELECT id FROM ingest.documents WHERE source_id = ANY(:filter_source_ids))"
            params["filter_source_ids"] = [str(s) for s in body.filters.source_ids]
    return clauses, params


# Backward-compat aliases
def build_text_filters(body: UnifiedQueryRequest) -> tuple[str, dict]:
    return build_filters(body, "tc")


def build_image_filters(body: UnifiedQueryRequest) -> tuple[str, dict]:
    return build_filters(body, "ic")


# ---------------------------------------------------------------------------
# Fusion scoring
# ---------------------------------------------------------------------------

MIL_ID_PATTERNS = [
    re.compile(r'\b\d{4}-\d{2}-\d{3}-\d{4}\b'),  # NSN
    re.compile(r'\bMIL-[A-Z]+-\d+[A-Z]?\b'),       # MIL-STD
    re.compile(r'\bPN-\d{4,}\b'),                    # Part number
    re.compile(r'\b[A-Z]{2}\d{4}\b'),                # ELNOT codes
    re.compile(r'\b\d{4}-[A-Z]{2}-\d{3}-\d{4}\b'),  # DIEQP identifiers
    re.compile(r'\bAN/[A-Z]{3}-\d+[A-Z]?(?:\(V\)\d*)?\b'),  # AN/ designators
]


def compute_fusion_score(
    semantic_score: float,
    doc_structure_weight: float = 0.0,
    doc_structure_hops: int = 0,
    ontology_rel_type: str | None = None,
    ontology_hops: int = 0,
    cross_modal_decay: float = 0.0,
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

    # Cross-modal component (fallback legacy path)
    cross_score = 0.0
    if cross_modal_decay > 0:
        cross_score = cross_modal_decay

    # Ontology component
    onto_score = 0.0
    if ontology_rel_type:
        rel_weights = get_ontology_relation_weights()
        rel_weight = rel_weights.get(ontology_rel_type, rel_weights.get("default", 0.70))
        hop_penalty = hop_base ** max(0, ontology_hops - 1)
        onto_score = rel_weight * hop_penalty

    # Weighted fusion
    final = sem_w * semantic_score + doc_w * max(doc_score, cross_score) + onto_w * onto_score

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
