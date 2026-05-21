"""Worker-side ontology bundle loader.

Reads manifest.yaml metadata and resolves bundle keys without importing
extraction_schemas modules. The service side (docling-graph) has a
separate loader at docker/docling-graph/app/bundles.py that DOES
pre-import the extraction schema classes for fast dispatch.

Spec §2 Bundle loader API.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from app.models.ingest import DocumentGraphExtraction

from app.services.ontology_templates import (
    UnknownBundleError,
    SYSTEM_DEFAULT_BUNDLE_KEY,
    load_ontology as _load_ontology_from_templates,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BUNDLE_ROOT = _REPO_ROOT / "ontology_bundles"

LEGACY_BUNDLE_LABEL = "legacy/unknown"

logger = logging.getLogger(__name__)

# Valid phase values for pass definitions.
PassPhase = Literal["identity", "field_group", "relationship"]


def _infer_pass_phase(name: str, input_mode: str) -> PassPhase:
    """Infer the phase for a pass that doesn't declare one explicitly.

    Rules (in priority order):
    1. Name ends with ``_identity``  → ``identity``
    2. ``input_mode == "document_plus_entity_refs"`` → ``relationship``
    3. Otherwise → ``field_group``
    """
    if name.endswith("_identity"):
        return "identity"
    if input_mode == "document_plus_entity_refs":
        return "relationship"
    return "field_group"


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
    # C1.6: explicit phase — required in new bundles; inferred + INFO-logged when absent
    # for back-compat with existing bundles that pre-date this field.
    phase: PassPhase = Field(default="field_group")  # default overridden by validator below

    @model_validator(mode="before")
    @classmethod
    def _infer_phase_when_missing(cls, data: Any) -> Any:
        """Infer ``phase`` when the manifest doesn't declare it.

        If ``phase`` is present, use it directly (pydantic validates the Literal).
        If absent, derive it from ``name`` + ``input_mode`` and log INFO.
        """
        if not isinstance(data, dict):
            return data
        if "phase" not in data:
            name = data.get("name", "")
            input_mode = data.get("input_mode", "document_only")
            inferred = _infer_pass_phase(name, input_mode)
            logger.info(
                "BundleManifest: inferred phase=%s for pass %s (manifest didn't declare it).",
                inferred, name,
            )
            data = {**data, "phase": inferred}
        return data


class BundleManifest(BaseModel):
    """Top-level manifest.yaml shape."""

    bundle_key: str
    manifest_schema_version: str
    ontology_name: str
    ontology_version: str
    extraction_profile_version: str
    passes: list[PassManifest]

    @model_validator(mode="after")
    def _validate_phase_constraints(self) -> "BundleManifest":
        """Enforce two structural invariants on the pass list.

        Fix 2: phase=relationship is currently only supported for name='system_links'.
        The worker dispatches it by that literal name; a future checkpoint will
        generalise this.

        Fix 3: if any pass has phase=field_group, at least one pass must have
        phase=identity.  The initial dispatcher queues only identity passes; without
        one, no task is ever queued and the run wedges in PROCESSING forever.
        """
        from pydantic import ValidationError as _ValidationError  # noqa: PLC0415 (avoid circular at module top)

        # Fix 2: relationship pass must be named 'system_links'.
        for p in self.passes:
            if p.phase == "relationship" and p.name != "system_links":
                raise ValueError(
                    f"phase='relationship' is currently only supported for name='system_links'. "
                    f"Pass '{p.name}' declared phase=relationship which is not yet supported. "
                    "Either rename to 'system_links' or wait for a future checkpoint that "
                    "generalizes relationship-phase dispatch."
                )

        # Fix 3: field_group passes require at least one identity pass.
        has_field_group = any(p.phase == "field_group" for p in self.passes)
        has_identity = any(p.phase == "identity" for p in self.passes)
        if has_field_group and not has_identity:
            raise ValueError(
                "Bundle has phase=field_group passes but no phase=identity pass. "
                "The initial dispatcher only queues identity passes; without one, "
                "the run will never start. Add at least one identity pass."
            )

        return self

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


@dataclass
class StatusSignals:
    """Status API roll-up per spec §7.10 + Task 3.5.

    Produced by the ``compute_status_signals`` helper (added in Chunk 4
    Task 4.7) and consumed by the document-status endpoint to build the
    three-concept response: document_status, latest_run, graph_snapshot
    (+ top-level graph_queryable).

    Fields:
        snapshot:         The DocumentGraphExtraction row for this document,
                          or None if no row exists (never written, or purged
                          by a full reingest's pre-derive_ontology_graph phase).
        is_stale:         Meaningful iff ``snapshot is not None``. True when
                          the snapshot's pipeline_run_id doesn't match the
                          most recent PipelineRun OR when that run is not
                          COMPLETE. Has no meaning when ``snapshot is None``
                          and the status API omits the nested ``is_stale``
                          field from the response in that case.
        graph_queryable:  ALWAYS meaningful, even when ``snapshot is None``.
                          Computed via the cross-run rollback query from
                          spec §7.10 — True iff a queryable extraction-layer
                          graph exists for this document right now. This is
                          the top-level field in the status response.
    """

    snapshot: "DocumentGraphExtraction | None"
    is_stale: bool
    graph_queryable: bool
