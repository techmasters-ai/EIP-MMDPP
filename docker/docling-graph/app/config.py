"""Docling-graph service configuration.

Introduced in Task 2.5 as part of the extraction-refactor rollout.
Currently scoped to ONE setting only: the structured-output threshold
that was hardcoded as 8000 inside _patched_build_request. Other scattered
config stays where it is for PR 1; later PRs may consolidate more.

Spec §2 / §5.9 / §409: the CI coverage checker reads the same default
via ontology_bundles/_shared/limits.DEFAULT_STRUCTURED_OUTPUT_THRESHOLD_CHARS
so the two cannot drift.
"""
from pydantic_settings import BaseSettings

from ontology_bundles._shared.limits import DEFAULT_STRUCTURED_OUTPUT_THRESHOLD_CHARS


class ServiceSettings(BaseSettings):
    """Docling-graph service runtime settings (DOCLING_GRAPH_* env prefix)."""

    # Char budget for the raw JSON Schema string. Schemas exceeding this
    # fall through to loose `format="json"` on Ollama rather than the
    # constrained grammar, because large schemas degrade constrained-
    # decoding quality.
    structured_output_threshold_chars: int = DEFAULT_STRUCTURED_OUTPUT_THRESHOLD_CHARS

    # Kill switch for Ollama schema-grammar constrained decoding. When
    # true, every extraction request uses `format="json"` regardless of
    # schema size. Set this for mid-size models (<=30B) that fail on the
    # constraint — gemma4:26b in particular truncates string values
    # mid-generation because it can't predict tokens consistent with
    # the grammar, producing unterminated-string JSON parse errors.
    # Pydantic still validates the loose JSON against the template
    # downstream via evidence_gate + apply_bundle_postprocessing, so
    # loose mode doesn't relax schema validation — it only relaxes
    # Ollama's token-level constraint.
    force_json_mode: bool = False

    class Config:
        env_prefix = "DOCLING_GRAPH_"


settings = ServiceSettings()
