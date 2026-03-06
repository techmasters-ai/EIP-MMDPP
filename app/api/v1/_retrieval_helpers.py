"""Pure helper functions for the retrieval endpoint.

Kept in a separate module so unit tests can import them without triggering
the DB session / asyncpg dependency chain (same pattern as _agent_helpers.py).
"""

from app.schemas.retrieval import QueryResultItem, UnifiedQueryRequest

# Score decay for graph-expanded results
CROSS_MODAL_DECAY = 0.85
ONTOLOGY_DECAY = 0.75


def deduplicate_results(results: list[QueryResultItem]) -> list[QueryResultItem]:
    """Deduplicate by chunk_id, keeping the highest-scoring entry."""
    best: dict[str, QueryResultItem] = {}
    for r in results:
        key = str(r.chunk_id) if r.chunk_id else str(id(r))
        if key not in best or r.score > best[key].score:
            best[key] = r
    return list(best.values())


def build_text_filters(body: UnifiedQueryRequest) -> str:
    clauses = ""
    if body.filters:
        if body.filters.classification:
            clauses += f" AND tc.classification = '{body.filters.classification}'"
        if body.filters.document_ids:
            doc_ids = ",".join(f"'{d}'" for d in body.filters.document_ids)
            clauses += f" AND tc.document_id IN ({doc_ids})"
    return clauses


def build_image_filters(body: UnifiedQueryRequest) -> str:
    clauses = ""
    if body.filters:
        if body.filters.classification:
            clauses += f" AND ic.classification = '{body.filters.classification}'"
        if body.filters.document_ids:
            doc_ids = ",".join(f"'{d}'" for d in body.filters.document_ids)
            clauses += f" AND ic.document_id IN ({doc_ids})"
    return clauses
