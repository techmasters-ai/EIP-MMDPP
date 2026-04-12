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
    """PR 1 scope: only structured_output_threshold_chars.
    Do NOT add more keys in this PR — other scattered config stays in place."""

    structured_output_threshold_chars: int = DEFAULT_STRUCTURED_OUTPUT_THRESHOLD_CHARS

    class Config:
        env_prefix = "DOCLING_GRAPH_"


settings = ServiceSettings()
