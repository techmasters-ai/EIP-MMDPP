"""Worker-side ontology bundle loader.

Reads manifest.yaml metadata and resolves bundle keys without importing
extraction_schemas modules. The service side (docling-graph) has a
separate loader at docker/docling-graph/app/bundles.py that DOES
pre-import the extraction schema classes for fast dispatch.

Spec §2 Bundle loader API.
"""
from __future__ import annotations

import importlib
import logging
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class ExecutionProfile(BaseModel):
    """Optional per-pass execution knobs declared in manifest.yaml.

    All four fields are optional — omitting a field means "use the service's
    env-var default for that knob."  Injected via
    ``_build_extract_pass_request`` onto the /extract-pass HTTP body so the
    docling-graph service can apply them per call.

    C2 Iter 1: only ``llm_batch_token_size`` is annotated (identity passes get
    2048).  The other three fields (chunk_max_tokens, temperature, max_tokens)
    are available for future iterations.

    Validation constraints:
    - Token sizes must be positive (> 0); zero or negative breaks chunking/LLM.
    - Temperature in [0.0, 2.0] matches Ollama/OpenAI semantics.
    - ``extra="forbid"`` turns misspelled keys (e.g. ``llm_batch_size_token``)
      into ValidationError instead of silently ignoring them.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_max_tokens: int | None = Field(default=None, gt=0)
    llm_batch_token_size: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)


class RetrievalProfile(BaseModel):
    """Per-pass vector-router retrieval config. Optional; ONLY present on
    field_group passes (identity/required/relationship passes are bypassed
    by the router short-circuit per VR section).

    Default values are CONSERVATIVE (rev 12 H1):
      - fallback_to_full=true means empty match → mode=full (worker dispatches
        RUN_FULL), not mode=would_skip. Operators may flip to false ONLY after
        C.6 shadow data shows the pass reliably routes would_skip on off-topic
        chunks without false positives.

    See plan revision history for rev 8 / rev 9 / rev 10 / rev 12 / rev 13
    decisions on each field.
    """

    model_config = ConfigDict(extra="forbid")

    min_similarity: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine similarity threshold below which chunks are dropped "
            "before rerank."
        ),
    )
    top_n_candidates: int = Field(
        default=50,
        gt=0,
        le=2000,
        description="Number of candidates retrieved pre-rerank.",
    )
    top_k: int = Field(
        default=20,
        gt=0,
        le=2000,
        description="Final ChunkScope size post-rerank.",
    )
    fallback_to_full: bool = Field(
        default=True,
        description=(
            "If true, empty retrieval → mode=full. If false, empty → "
            "mode=would_skip."
        ),
    )

    # ------------------------------------------------------------------
    # Channel sizing / limits (B5)
    # ------------------------------------------------------------------
    field_query_top_k: int = Field(
        default=8,
        gt=0,
        le=2000,
        description=(
            "Per-field dense-vector retrieval top-k fed to each field-level "
            "sub-query before merging candidates."
        ),
    )
    lexical_top_k: int = Field(
        default=40,
        gt=0,
        le=2000,
        description="Number of candidates returned from the lexical (BM25/keyword) channel.",
    )
    pattern_hit_limit: int = Field(
        default=50,
        gt=0,
        le=2000,
        description="Maximum chunk hits retained from the pattern-match channel.",
    )
    fallback_min_field_coverage: int = Field(
        default=1,
        gt=0,
        description=(
            "Minimum number of schema fields that must be covered by retrieved "
            "chunks before the fallback-to-full path is considered."
        ),
    )
    fallback_similarity_relaxation: float = Field(
        default=0.07,
        ge=0.0,
        le=1.0,
        description=(
            "Amount by which min_similarity is relaxed on the fallback retry "
            "pass when initial retrieval falls below fallback_min_field_coverage."
        ),
    )

    # ------------------------------------------------------------------
    # POST-rerank precision-boost weights (B5)
    # Applied after the cross-encoder, before the top_k cut.
    # Final score = rerank_weight * norm_ce_score
    #             + lexical_weight * norm_lexical_score
    #             + pattern_weight * pattern_hit
    #             + section_weight * section_match
    #             + table_boost * is_table
    #             - negative_weight * negative_signal
    # ------------------------------------------------------------------
    rerank_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Base weight applied to the normalised cross-encoder score.",
    )
    lexical_weight: float = Field(
        default=0.20,
        ge=0.0,
        description="Boost weight for lexical (BM25/keyword) channel hit score.",
    )
    pattern_weight: float = Field(
        default=0.15,
        ge=0.0,
        description="Boost weight for pattern-match channel hit.",
    )
    section_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Boost weight awarded when the chunk is from a matching section. "
            "Router-scoring section-signal piece: default flipped 0.10 -> 0.0. "
            "Before the section matcher wired real data, ``section_norm`` was "
            "always 0, so ``section_weight * section_norm`` (0.10 * 0) "
            "contributed nothing. Now that ``section_hit_counts`` makes "
            "``section_norm`` non-zero, keeping the term INERT (byte-identical "
            "final_score) REQUIRES the default be 0.0. Calibration raises it "
            "once shadow data justifies the section boost."
        ),
    )
    table_boost: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Additive boost for table chunks (numeric evidence density). "
            "TABLE signal (is_table wiring): default flipped 0.08 -> 0.0 — the "
            "section_weight precedent. Before the persisted ``is_table`` column "
            "wired real data, ``content_type`` was always None, so "
            "``table_boost * is_table`` (0.08 * 0) contributed nothing. Now "
            "that ``table_meta`` / ``merged_candidate_from_row`` make "
            "``is_table`` non-zero for table chunks, keeping the term INERT "
            "(byte-identical final_score) REQUIRES the default be 0.0. "
            "Calibration raises it once shadow data justifies the table boost."
        ),
    )
    negative_weight: float = Field(
        default=0.20,
        ge=0.0,
        description="Penalty weight subtracted for negative-signal indicators.",
    )

    # ------------------------------------------------------------------
    # Decomposed-lexical weights + flag (DECLARED here; CONSUMED in the
    # next router-scoring piece — the 3-weight C5 lexical decomposition).
    #
    # Declaring them now under ``extra="forbid"`` is harmless: a manifest may
    # set them, and the scoring formula does not read them yet. All three
    # weights default to 0.0 and the flag defaults to False, so the future
    # decomposition stays inert until calibration enables it. Splitting the
    # single ``lexical_weight`` term into field-label / anchor-text /
    # anchor-section sub-terms lets the router weight an alias hit differently
    # depending on WHERE it matched (schema field label vs committed-entity
    # name in body text vs that same name in the section heading).
    # ------------------------------------------------------------------
    field_label_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Decomposed-lexical sub-weight for alias hits that matched a "
            "schema field LABEL. Inert until lexical_decomposed=True (next "
            "piece)."
        ),
    )
    anchor_text_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Decomposed-lexical sub-weight for committed-entity ANCHOR names "
            "that matched in the chunk body text. Inert until "
            "lexical_decomposed=True (next piece)."
        ),
    )
    anchor_section_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Decomposed-lexical sub-weight for committed-entity ANCHOR names "
            "that matched in the chunk SECTION heading. Inert until "
            "lexical_decomposed=True."
        ),
    )
    pass_keyword_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Decomposed-lexical sub-weight for per-pass ``lexical_keywords`` "
            "matches in the chunk body text. Enriches the weak field-alias "
            "vocabulary with a per-pass keyword list weighted as its own "
            "feature. Inert until lexical_decomposed=True."
        ),
    )
    lexical_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Per-pass extra keyword needles matched (NFC + casefold substring) "
            "against chunk body text as a SEPARATE feature from schema field "
            "aliases. Counted by ``keyword_hit_counts`` and weighted by "
            "``pass_keyword_weight`` when lexical_decomposed=True. A manifest "
            "``retrieval:`` block may set it; omitting it yields [] (no extra "
            "keywords). Values are operator-supplied — no defaults are baked in."
        ),
    )
    lexical_keyword_weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Optional per-keyword lift weights for the pass-keyword channel. "
            "Keys are keyword strings as written in lexical_keywords (matched "
            "after NFC+casefold normalisation at runtime). Values multiply that "
            "keyword's presence contribution in keyword_hit_counts; an absent "
            "key defaults to 1.0. Offline miners may ship mined lift scores "
            "here. All values must be >= 0. Default {} (every keyword weighs "
            "1.0 — byte-identical to the pre-weight behaviour)."
        ),
    )

    @field_validator("lexical_keyword_weights")
    @classmethod
    def _no_negative_weights(cls, v: dict[str, float]) -> dict[str, float]:
        """Reject any negative weight value (negative lift is not meaningful)."""
        neg = [k for k, w in v.items() if w < 0]
        if neg:
            raise ValueError(
                f"lexical_keyword_weights: negative values are not allowed "
                f"(offending keys: {neg!r})"
            )
        return v

    lexical_decomposed: bool = Field(
        default=False,
        description=(
            "When True, the C5 lexical term is split into field_label / "
            "pass_keyword / anchor_text / anchor_section sub-terms using the "
            "weights above instead of the single ``lexical_weight``. Default "
            "False preserves the legacy single-weight lexical term "
            "(byte-identical final_score)."
        ),
    )
    unit_gate: bool = Field(
        default=False,
        description=(
            "Enable the G1 unit-signature recall gate: chunks containing a digit "
            "+ a pass-unit token join the candidate pool exempt from the "
            "top_n_candidates cap (guarded-ranker spec §3). Default False "
            "(byte-identical legacy pool)."
        ),
    )

    # ------------------------------------------------------------------
    # Task 18 — selection_mode + guarded-quantile cut (guarded-ranker spec).
    #
    # The router's final post-rerank cut is parameterised here. The DEFAULT
    # ("topk") is byte-identical to the historical ``c5_scored[: top_k]`` slice
    # at all selection sites. ``guarded_quantile`` swaps that for a recall-safe
    # OR-gate ∪ quantile-ranker cut (gates kept by construction; the ranker
    # supplies the precision floor with k_min/k_max bounds).
    # ------------------------------------------------------------------
    selection_mode: Literal["topk", "guarded_quantile"] = Field(
        default="topk",
        description=(
            "Final post-rerank candidate cut. 'topk' (DEFAULT) = the legacy "
            "c5_scored[: top_k] slice (byte-identical). 'guarded_quantile' = "
            "dedup(gate-flagged ∪ quantile-ranker keeps) with a k_min floor and "
            "k_max cap (gate-flagged members are exempt from k_max)."
        ),
    )
    quantile_q: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "guarded_quantile only: the quantile of in-pool ranker scores used "
            "as the keep threshold. Members with ranker score >= this quantile "
            "are ranker-keeps. Higher q = stricter (keeps fewer)."
        ),
    )
    k_min: int = Field(
        default=3,
        gt=0,
        description=(
            "guarded_quantile only: minimum number of ranker-keeps (floor) "
            "regardless of how few clear the quantile threshold."
        ),
    )
    k_max: int = Field(
        default=0,
        ge=0,
        description=(
            "guarded_quantile only: maximum number of ranker-keeps (cap); "
            "0 = uncapped. Gate-flagged members are NEVER counted against this "
            "cap — they are always kept."
        ),
    )
    ranker_weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "guarded_quantile only: component-name -> weight for the ranker "
            "score. Empty {} (DEFAULT) = rank by the C5 final_score. When "
            "non-empty, ranker score = Σ ranker_weights[k] * components[k]. "
            "Keys MUST be known scoring-component names (validated at profile "
            "construction time) — an unknown key is a hard error, not a "
            "silent 0 at scoring time."
        ),
    )

    @field_validator("ranker_weights")
    @classmethod
    def _ranker_weight_keys_known(cls, v: dict[str, float]) -> dict[str, float]:
        """Reject ranker_weights keys that are not known scoring components.

        Guarded-ranker spec: an unknown component name must FAIL LOUD at profile
        validation time, never silently contribute 0 during scoring. The valid
        set is the numeric scoring components in
        ``extraction_candidate_scoring.COMPONENT_KEYS`` (the non-numeric
        ``candidate_key`` is excluded — it is an identifier, not a feature).
        """
        # Local import to avoid a circular import at module load
        # (extraction_candidate_scoring imports RetrievalProfile for typing).
        from app.services.extraction_candidate_scoring import (  # noqa: PLC0415
            COMPONENT_KEYS,
        )

        valid = set(COMPONENT_KEYS) - {"candidate_key"}
        unknown = sorted(k for k in v if k not in valid)
        if unknown:
            raise ValueError(
                f"ranker_weights references unknown scoring component(s): "
                f"{unknown!r}. Valid components: {sorted(valid)!r}."
            )
        return v

    # ------------------------------------------------------------------
    # §9 Subset-schema extraction (opt-in; default off) (B5)
    # ------------------------------------------------------------------
    subset_schema_extraction: bool = Field(
        default=False,
        description=(
            "When True, extract only the subset of the schema fields whose "
            "retrieval chunks scored above min_similarity; reduces LLM payload "
            "size. Opt-in until C.9 shadow data confirms recall parity."
        ),
    )


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
    # C2: optional per-pass execution knobs. When present, these values are
    # threaded onto the /extract-pass request body by _build_extract_pass_request
    # and applied by the docling-graph service for that pass only. Omitting the
    # block (or setting it to null) means "use service env-var defaults."
    execution: ExecutionProfile | None = None
    # C.2b: optional per-pass vector-router retrieval config. Only present on
    # field_group passes; identity/required/relationship passes are bypassed by
    # the C.4 router short-circuit and must NOT have a retrieval block.
    retrieval: RetrievalProfile | None = None

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


def iter_routable_pass_fields(bundle_key: str) -> list[tuple[str, list[str]]]:
    """Return (pass_name, field_names) for every routable (field_group) pass in the bundle.

    "Routable" is defined as passes whose ``retrieval`` attribute is not None —
    only field_group passes carry a RetrievalProfile; identity and relationship
    passes are excluded by construction.

    Field names are resolved from the pass's Pydantic template class by
    following the same importlib pattern used in pipeline.py. The template
    class holds a single list field (e.g. ``radar_systems``) whose item type
    is the per-entity record class; the record class's ``model_fields`` yields
    the actual extraction field names (e.g. ``gain_dbi``, ``beamwidth_az_deg``).
    Falls back to the top-level template's own ``model_fields`` if no list
    item with a nested Pydantic model is found.
    """
    manifest = load_bundle_manifest(bundle_key)
    result: list[tuple[str, list[str]]] = []
    for pass_def in manifest.passes:
        if pass_def.retrieval is None:
            continue
        full_module_path = f"ontology_bundles.{manifest.bundle_key}.{pass_def.module}"
        try:
            template_module = importlib.import_module(full_module_path)
            template_cls = getattr(template_module, pass_def.template_class)
        except (ImportError, AttributeError) as exc:
            raise BundleResolutionError(
                f"cannot load template {full_module_path}.{pass_def.template_class}: {exc}"
            ) from exc
        # Unwrap the list-of-records container to get the actual record field names.
        # Pattern: template_cls has one list field; its item type is the record model.
        field_names: list[str] | None = None
        for top_field_info in template_cls.model_fields.values():
            args = typing.get_args(top_field_info.annotation)
            if args:
                item_cls = args[0]
                if hasattr(item_cls, "model_fields"):
                    field_names = list(item_cls.model_fields.keys())
                    break
        if field_names is None:
            logger.warning(
                "iter_routable_pass_fields: no nested record model found for "
                "bundle=%s pass=%s template=%s; falling back to container-level "
                "field names (signal config may be empty for this pass).",
                bundle_key, pass_def.name, pass_def.template_class,
            )
            field_names = list(template_cls.model_fields.keys())
        result.append((pass_def.name, field_names))
    return result


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
