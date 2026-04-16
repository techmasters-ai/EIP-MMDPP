"""Shared threshold constants for ontology bundle tooling and services.

The docling-graph service imports DEFAULT_STRUCTURED_OUTPUT_THRESHOLD_CHARS at
runtime; the coverage checker rule 5 reads it via SCHEMA_SIZE_THRESHOLD_CHARS.
Both resolve from this single location so they stay in sync.
"""
import os

# Maximum JSON-schema character count for a structured-output template class.
# If the schema serialised as JSON exceeds this limit the LLM provider's
# context window may reject the schema or produce worse completions.
#
# Phase 5 raised the default from 8000 → 20000 to accommodate typed-edge
# extraction schemas: when relationship targets are nested Pydantic models
# declared via ``edge(label=...)`` the JSON schema expands with nested
# ``$defs`` for each edge target class. A radar_domain pass with 7 typed
# edges (HAS_ANTENNA, HAS_RECEIVER, HAS_TRANSMITTER, HAS_PROCESSING_CHAIN,
# OPERATES_IN_BAND, USES_WAVEFORM, INSTALLED_ON) produces ~14KB schema.
# 20000 chars leaves headroom for future edge additions without exceeding
# modern LLM context limits (still well under 10K tokens).
DEFAULT_STRUCTURED_OUTPUT_THRESHOLD_CHARS: int = int(
    os.environ.get(
        "DOCLING_GRAPH_STRUCTURED_OUTPUT_THRESHOLD_CHARS",
        20000,
    )
)
