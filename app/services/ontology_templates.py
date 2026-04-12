"""Ontology loading with bundle-aware resolution.

Extraction schemas, runtime Pydantic templates, and prompt construction
have moved to ontology_bundles/<bundle_key>/extraction_schemas/ and the
docling-graph sidecar service. This module is now a thin loader for
ontology.yaml content, plus a small registry-lookup helper for audit
paths.

Public API (spec §2 + §7.3):
- load_ontology(*, bundle_key=None, path=None) — bundle/path loader
- load_registry_ontology(version_id) — version-pinned registry lookup
- load_validation_matrix(*, bundle_key=None, path=None) — derived helper
- build_entity_type_names / build_relationship_type_names — derived helpers
- invalidate_ontology_cache() — clear the per-bundle cache

Private helpers:
- load_repository_ontology(path) — direct file reader, kept for back-compat
- _ensure_bundle_cached(bundle_key) — per-bundle cache population
"""
from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
from threading import Lock
import time
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_TTL_SECONDS = 5.0
SYSTEM_DEFAULT_BUNDLE_KEY = "air_defense_v3"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BUNDLE_ROOT = _REPO_ROOT / "ontology_bundles"

# Per-bundle cache: bundle_key -> (ontology_dict, signature, expires_at_monotonic)
_cache_lock = Lock()
_bundle_cache: dict[str, tuple[dict[str, Any], str, float]] = {}
_invalidation_hooks: list[Callable[[], None]] = []


class UnknownBundleError(ValueError):
    """Raised when a bundle_key does not resolve to a directory under
    ontology_bundles/."""


def register_invalidation_hook(fn: Callable[[], None]) -> None:
    """Register a callback to be invoked when the ontology cache is invalidated."""
    _invalidation_hooks.append(fn)


def invalidate_ontology_cache() -> None:
    """Clear the per-bundle in-process ontology cache.

    Called after registry create/update/activate operations so backend ontology
    consumers see the newly active ontology immediately.
    """
    with _cache_lock:
        _bundle_cache.clear()
    for hook in _invalidation_hooks:
        try:
            hook()
        except Exception:
            logger.warning("Ontology invalidation hook failed", exc_info=True)


def load_repository_ontology(path: Path | None = None) -> dict[str, Any]:
    """Load and return the repository ontology YAML as a dict.

    Kept for backwards compatibility. When a path is given, reads from it
    directly. Otherwise falls back to load_ontology() (the default bundle),
    because ontology/ontology.yaml symlink was removed in Task 5.2.
    New code should prefer load_ontology(bundle_key=...).
    """
    if path is not None:
        with open(path) as f:
            return yaml.safe_load(f)
    # Legacy symlink removed — fall through to bundle-based loader
    return load_ontology()


def _bundle_ontology_path(bundle_key: str) -> Path:
    p = _BUNDLE_ROOT / bundle_key / "ontology.yaml"
    if not p.exists():
        raise UnknownBundleError(
            f"No ontology.yaml found for bundle_key={bundle_key!r} (expected at {p})"
        )
    return p


def _bundle_signature(bundle_key: str) -> str:
    p = _bundle_ontology_path(bundle_key).resolve()
    try:
        mtime_ns = p.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    return f"bundle:{bundle_key}:{mtime_ns}"


def _ensure_bundle_cached(bundle_key: str) -> tuple[dict[str, Any], str]:
    now = time.monotonic()
    with _cache_lock:
        entry = _bundle_cache.get(bundle_key)
        if entry is not None:
            ontology_ref, sig, expires = entry
            if now < expires:
                return ontology_ref, sig

    # Load outside the lock, then store.
    path = _bundle_ontology_path(bundle_key)
    with open(path) as f:
        ontology = yaml.safe_load(f)
    sig = _bundle_signature(bundle_key)

    with _cache_lock:
        _bundle_cache[bundle_key] = (ontology, sig, now + _DEFAULT_CACHE_TTL_SECONDS)
    return ontology, sig


def load_ontology(
    *,
    bundle_key: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Load an ontology definition.

    Resolution order (exactly one applies):
    1. If `path` is given, load directly from that file.
    2. Else if `bundle_key` is given, load that bundle's ontology.yaml.
    3. Else load the system default bundle's ontology.yaml (air_defense_v3).

    This function never consults the registry/version-pinning store.
    For version-pinned loads, call load_registry_ontology(version_id)
    explicitly.

    Raises UnknownBundleError when bundle_key does not resolve.
    """
    if path is not None:
        with open(path) as f:
            return yaml.safe_load(f)
    resolved_key = bundle_key or SYSTEM_DEFAULT_BUNDLE_KEY
    ontology_ref, _ = _ensure_bundle_cached(resolved_key)
    return deepcopy(ontology_ref)


def load_registry_ontology(version_id: str) -> dict[str, Any]:
    """Load a version-pinned ontology snapshot from the registry.

    Consults app.models.query_profiles.QueryProfileRegistry, looking up
    by id (the version_id). Used by audit/historical-reproduction paths
    and by query-profile-aware readers that need a specific registry row.

    Raises LookupError if no row matches.
    """
    from sqlalchemy import select
    from app.db.session import get_sync_session
    from app.models.query_profiles import QueryProfileRegistry

    with get_sync_session() as session:
        result = session.execute(
            select(QueryProfileRegistry)
            .where(QueryProfileRegistry.id == version_id)
            .limit(1)
        )
        registry = result.scalar_one_or_none()
        if registry is None or not isinstance(registry.ontology_definition, dict):
            raise LookupError(
                f"No registry ontology found for version_id={version_id!r}"
            )
        ontology = deepcopy(registry.ontology_definition)
        if registry.ontology_version and not ontology.get("version"):
            ontology["version"] = registry.ontology_version
        return ontology


def get_ontology_cache_signature(bundle_key: str | None = None) -> str:
    """Return a cache signature for the named bundle (or system default)."""
    resolved_key = bundle_key or SYSTEM_DEFAULT_BUNDLE_KEY
    _, sig = _ensure_bundle_cached(resolved_key)
    return sig


def load_validation_matrix(
    *,
    bundle_key: str | None = None,
    path: Path | None = None,
) -> set[tuple[str, str, str]]:
    """Load the ontology validation matrix as a set of
    (source, rel, target) triples. Returns empty set if absent."""
    ontology = load_ontology(bundle_key=bundle_key, path=path)
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
