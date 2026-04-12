"""Shared threshold constants for ontology bundle tooling and services.

The docling-graph service imports DEFAULT_STRUCTURED_OUTPUT_THRESHOLD_CHARS at
runtime; the coverage checker rule 5 reads it via SCHEMA_SIZE_THRESHOLD_CHARS.
Both resolve from this single location so they stay in sync.
"""
import os

# Maximum JSON-schema character count for a structured-output template class.
# If the schema serialised as JSON exceeds this limit the LLM provider's
# context window may reject the schema or produce worse completions.
DEFAULT_STRUCTURED_OUTPUT_THRESHOLD_CHARS: int = int(
    os.environ.get(
        "DOCLING_GRAPH_STRUCTURED_OUTPUT_THRESHOLD_CHARS",
        8000,
    )
)
