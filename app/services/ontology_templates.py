"""Ontology loading and type-name helpers for GraphRAG and retrieval.

Extraction prompt building and Pydantic extraction models have moved to
the Docling-Graph Docker service.
"""

from copy import deepcopy
import logging
from pathlib import Path
from threading import Lock
import time
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)

_ONTOLOGY_PATH = Path(__file__).resolve().parent.parent.parent / "ontology" / "ontology.yaml"
_DEFAULT_CACHE_TTL_SECONDS = 5.0
_cache_lock = Lock()
_cached_default_ontology: dict[str, Any] | None = None
_cached_default_signature: str | None = None
_cached_default_expires_at = 0.0
_invalidation_hooks: list[Callable[[], None]] = []


def register_invalidation_hook(fn: Callable[[], None]) -> None:
    """Register a callback to be invoked when the ontology cache is invalidated."""
    _invalidation_hooks.append(fn)


def invalidate_ontology_cache() -> None:
    """Clear the in-process ontology cache.

    Called after registry create/update/activate operations so backend ontology
    consumers see the newly active ontology immediately.
    """
    global _cached_default_ontology, _cached_default_signature, _cached_default_expires_at
    with _cache_lock:
        _cached_default_ontology = None
        _cached_default_signature = None
        _cached_default_expires_at = 0.0
    for hook in _invalidation_hooks:
        try:
            hook()
        except Exception:
            logger.warning("Ontology invalidation hook failed", exc_info=True)


def load_repository_ontology(path: Path | None = None) -> dict[str, Any]:
    """Load and return the repository ontology YAML as a dict."""
    p = path or _ONTOLOGY_PATH
    with open(p) as f:
        return yaml.safe_load(f)


def _repository_signature(path: Path | None = None) -> str:
    p = (path or _ONTOLOGY_PATH).resolve()
    try:
        mtime_ns = p.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    return f"file:{p}:{mtime_ns}"


def _load_active_registry_ontology() -> tuple[dict[str, Any] | None, str | None]:
    """Return the active registry ontology, if available."""
    try:
        from sqlalchemy import select

        from app.db.session import get_sync_session
        from app.models.query_profiles import QueryProfileRegistry

        with get_sync_session() as session:
            result = session.execute(
                select(QueryProfileRegistry)
                .where(QueryProfileRegistry.is_active.is_(True))
                .order_by(QueryProfileRegistry.updated_at.desc())
                .limit(1)
            )
            registry = result.scalar_one_or_none()

            if (
                registry is None
                or not isinstance(registry.ontology_definition, dict)
                or not registry.ontology_definition
            ):
                return None, None

            ontology = deepcopy(registry.ontology_definition)
            if registry.ontology_version and not ontology.get("version"):
                ontology["version"] = registry.ontology_version

            updated_at = (
                registry.updated_at.isoformat()
                if getattr(registry, "updated_at", None) is not None
                else ""
            )
            reg_id = registry.id

        return ontology, f"registry:{reg_id}:{updated_at}"
    except Exception:
        logger.warning(
            "Falling back to repository ontology because the active registry ontology "
            "could not be loaded",
            exc_info=True,
        )
        return None, None


def _resolve_default_ontology() -> tuple[dict[str, Any], str]:
    ontology, signature = _load_active_registry_ontology()
    if ontology is not None and signature is not None:
        return ontology, signature
    return load_repository_ontology(), _repository_signature()


def _ensure_cache_populated() -> tuple[dict[str, Any], str]:
    """Populate the cache if stale/empty, return (ontology_ref, signature).

    The returned ontology_ref is the cached reference — callers that need
    a mutable copy must deepcopy it themselves.
    """
    global _cached_default_ontology, _cached_default_signature, _cached_default_expires_at

    now = time.monotonic()
    with _cache_lock:
        if (
            _cached_default_ontology is not None
            and _cached_default_signature is not None
            and now < _cached_default_expires_at
        ):
            return _cached_default_ontology, _cached_default_signature

    ontology, signature = _resolve_default_ontology()
    with _cache_lock:
        _cached_default_ontology = ontology
        _cached_default_signature = signature
        _cached_default_expires_at = now + _DEFAULT_CACHE_TTL_SECONDS
    return ontology, signature


def get_ontology_cache_signature(
    path: Path | None = None,
    *,
    prefer_active: bool = True,
) -> str:
    """Return a cache signature for the ontology currently in use."""
    if path is not None or not prefer_active:
        return _repository_signature(path)
    _, signature = _ensure_cache_populated()
    return signature


def load_ontology(
    path: Path | None = None,
    *,
    prefer_active: bool = True,
) -> dict[str, Any]:
    """Load the active registry ontology, falling back to repository YAML."""
    if path is not None or not prefer_active:
        return load_repository_ontology(path)
    ontology_ref, _ = _ensure_cache_populated()
    return deepcopy(ontology_ref)


def load_validation_matrix(
    path: Path | None = None,
    *,
    prefer_active: bool = True,
) -> set[tuple[str, str, str]]:
    """Load the ontology validation matrix as a set of (source, rel, target) triples.

    Returns an empty set if the ontology doesn't define a validation_matrix.
    """
    ontology = load_ontology(path, prefer_active=prefer_active)
    matrix: set[tuple[str, str, str]] = set()
    for entry in ontology.get("validation_matrix", []):
        source = entry.get("source", "") or entry.get("source_type", "")
        rel = entry.get("relationship", "")
        target = entry.get("target", "") or entry.get("target_type", "")
        if source and rel and target:
            matrix.add((source, rel, target))
    return matrix


def _extract_type_names(ontology: dict[str, Any], key: str) -> list[str]:
    """Extract type names from an ontology list (handles both str and dict entries)."""
    names: list[str] = []
    for item in ontology.get(key, []):
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    return names


def build_entity_type_names(ontology: dict[str, Any] | None = None) -> list[str]:
    """Return a list of all entity type names from the ontology."""
    if ontology is None:
        ontology = load_ontology()
    return _extract_type_names(ontology, "entity_types")


def build_relationship_type_names(ontology: dict[str, Any] | None = None) -> list[str]:
    """Return a list of all relationship type names from the ontology."""
    if ontology is None:
        ontology = load_ontology()
    return _extract_type_names(ontology, "relationship_types")
