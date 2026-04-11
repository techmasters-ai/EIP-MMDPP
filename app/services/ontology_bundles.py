"""Worker-side ontology bundle loader.

Reads manifest.yaml metadata and resolves bundle keys without importing
extraction_schemas modules. The service side (docling-graph) has a
separate loader at docker/docling-graph/app/bundles.py that DOES
pre-import the extraction schema classes for fast dispatch.

Spec §2 Bundle loader API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from app.services.ontology_templates import (
    UnknownBundleError,
    SYSTEM_DEFAULT_BUNDLE_KEY,
    load_ontology as _load_ontology_from_templates,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BUNDLE_ROOT = _REPO_ROOT / "ontology_bundles"

LEGACY_BUNDLE_LABEL = "legacy/unknown"


class BundleResolutionError(ValueError):
    """Raised when resolve_bundle_key has no tier to resolve from."""


class PassManifest(BaseModel):
    """One pass as declared in manifest.yaml."""

    name: str
    required: bool
    kind: Literal["entities", "entities_and_relationships", "relationships_only"]
    input_mode: Literal["document_only", "document_plus_entity_refs"]
    module: str  # e.g. "extraction_schemas.radar_domain"
    template_class: str  # e.g. "RadarDomainPass"
    primary_entity_types: list[str] = Field(default_factory=list)
    bridge_entity_types: list[str] = Field(default_factory=list)
    extracted_relationship_types: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    skip_if_no_upstream_endpoints: bool = False
    skip_justification: str | None = None


class BundleManifest(BaseModel):
    """Top-level manifest.yaml shape."""

    bundle_key: str
    manifest_schema_version: str
    ontology_name: str
    ontology_version: str
    extraction_profile_version: str
    passes: list[PassManifest]

    def find_pass(self, pass_name: str) -> PassManifest:
        """Return the pass with the given name, or raise KeyError."""
        for p in self.passes:
            if p.name == pass_name:
                return p
        raise KeyError(pass_name)


def _bundle_manifest_path(bundle_key: str) -> Path:
    p = _BUNDLE_ROOT / bundle_key / "manifest.yaml"
    if not p.exists():
        raise UnknownBundleError(
            f"No manifest.yaml for bundle_key={bundle_key!r} (expected at {p})"
        )
    return p


def load_bundle_manifest(bundle_key: str) -> BundleManifest:
    """Read ontology_bundles/<bundle_key>/manifest.yaml and return a BundleManifest.

    Raises UnknownBundleError (from app.services.ontology_templates) if the
    bundle directory doesn't exist, or pydantic.ValidationError if the manifest
    doesn't conform.
    """
    path = _bundle_manifest_path(bundle_key)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return BundleManifest.model_validate(raw)


def list_available_bundles() -> list[str]:
    """Return the list of bundle_keys by scanning ontology_bundles/<bundle>/manifest.yaml
    directories. Skips directories starting with '_'."""
    if not _BUNDLE_ROOT.is_dir():
        return []
    keys: list[str] = []
    for entry in sorted(_BUNDLE_ROOT.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        if (entry / "manifest.yaml").exists():
            keys.append(entry.name)
    return keys


def load_bundle_ontology(bundle_key: str) -> dict:
    """Read ontology_bundles/<bundle_key>/ontology.yaml and return the parsed dict.

    Thin wrapper around load_ontology(bundle_key=...) from app.services.ontology_templates.
    """
    return _load_ontology_from_templates(bundle_key=bundle_key)


def resolve_bundle_key(
    *,
    run_key: str | None,
    source_key: str | None,
    system_default: str | None,
) -> str:
    """Resolve bundle_key via the three-tier precedence run → source → system_default.

    Returns the first non-None tier. Raises BundleResolutionError if every tier is
    None (caller passed None as system_default, which is a config bug).
    """
    for tier in (run_key, source_key, system_default):
        if tier is not None:
            return tier
    raise BundleResolutionError(
        "Cannot resolve bundle_key: all tiers (run, source, system_default) are None"
    )


def resolve_bundle_key_for_graph_only(
    *,
    run_key: str | None,
    source_key: str | None,
    system_default: str | None,
    inherited_from_run: str | None = None,
) -> str:
    """Graph-only precedence: run → inherited (from latest run) → source → system_default.

    For PR 1 Task 2.3 we support this signature but the reingest route that consumes
    it lands later. Raises BundleResolutionError if all tiers are None.
    """
    for tier in (run_key, inherited_from_run, source_key, system_default):
        if tier is not None:
            return tier
    raise BundleResolutionError(
        "Cannot resolve graph_only bundle_key: all tiers are None"
    )


def describe_bundle_for_display(bundle_key: str | None) -> str:
    """Return a human-readable label for a bundle_key. None -> LEGACY_BUNDLE_LABEL."""
    return bundle_key if bundle_key else LEGACY_BUNDLE_LABEL
