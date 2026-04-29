"""Fail loudly if the docling-graph mirror drifts from the canonical client."""
from pathlib import Path

import pytest

_MARKER = "# === SHARED CODE BELOW THIS LINE ===\n"

# Anchor relative to this test file rather than pytest's invocation cwd —
# matches the convention used by other tests in the repo and lets the
# mirror check run from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _shared_body(text: str, file_label: str) -> str:
    parts = text.split(_MARKER, 1)
    if len(parts) != 2:
        raise AssertionError(
            f"{file_label} is missing the marker line {_MARKER!r}; the "
            "mirror invariant requires it immediately after the docstring."
        )
    # Defensive: the marker must appear exactly once. A second copy in a
    # docstring example or comment would silently shift the split point.
    assert text.count(_MARKER) == 1, (
        f"{file_label} contains multiple SHARED CODE markers — the marker "
        "must be unique so split('marker', 1) is unambiguous."
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
        canonical = (_REPO_ROOT / canonical_path).read_text()
        mirror = (_REPO_ROOT / mirror_path).read_text()
        canon_body = _shared_body(canonical, canonical_path)
        mirror_body = _shared_body(mirror, mirror_path)
        assert canon_body == mirror_body, (
            f"{mirror_path} drifted from {canonical_path} — copy the "
            "canonical file's shared section."
        )


def test_shared_body_rejects_text_without_marker():
    """Negative test for _shared_body: a string lacking the marker must
    raise AssertionError so a future refactor that drops the marker
    fails the mirror invariant loudly instead of silently."""
    with pytest.raises(AssertionError, match="missing the marker"):
        _shared_body("no marker here\nstill nothing", "test_input")
