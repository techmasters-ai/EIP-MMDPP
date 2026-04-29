"""Fail loudly if the docling-graph mirror drifts from the canonical client."""
from pathlib import Path

_MARKER = "# === SHARED CODE BELOW THIS LINE ===\n"


def _shared_body(text: str, file_label: str) -> str:
    parts = text.split(_MARKER, 1)
    if len(parts) != 2:
        raise AssertionError(
            f"{file_label} is missing the marker line {_MARKER!r}; the "
            "mirror invariant requires it immediately after the docstring."
        )
    return parts[1]


_MIRROR_PAIRS = [
    ("app/services/ollama_pool_client.py",
     "docker/docling-graph/app/ollama_pool_client.py"),
    ("app/services/llm_json.py",
     "docker/docling-graph/app/llm_json.py"),
]


def test_pool_client_mirror_in_sync():
    for canonical_path, mirror_path in _MIRROR_PAIRS:
        canonical = Path(canonical_path).read_text()
        mirror = Path(mirror_path).read_text()
        canon_body = _shared_body(canonical, canonical_path)
        mirror_body = _shared_body(mirror, mirror_path)
        assert canon_body == mirror_body, (
            f"{mirror_path} drifted from {canonical_path} — copy the "
            "canonical file's shared section."
        )
