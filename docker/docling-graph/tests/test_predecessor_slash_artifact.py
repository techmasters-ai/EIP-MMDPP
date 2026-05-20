"""Step 6 follow-up: predecessor-context slash-group artifact filter.

When the LLM extracts every named missile mention, it can promote a
malformed slash-group token from predecessor-context prose into a
standalone `system_name`. Example from the SA-2 corpus:

    "...is a direct evolution of the earlier SA-25/S-25 / SA-1 Guild,
     itself evolved from the 1944 German Wasserfall design."

`SA-25` is almost certainly an OCR/typesetting artifact (SA-2 with a
trailing 5 fused). The canonical predecessor entities (S-25, SA-1) are
already present; the standalone `SA-25` row has no attributes filled.

The filter `_is_predecessor_slash_artifact` returns True when the
entity name:
  1. Appears exactly once in evidence_text.
  2. That occurrence is immediately followed by `/`.
  3. Within ~30 chars BEFORE the occurrence there's a predecessor-
     context marker (EARLIER, EVOLVED FROM, PREDECESSOR, PRECURSOR,
     ANCESTOR, SUPERSEDED).

Generic — operates on context language and slash adjacency, not on
equipment names.
"""
import importlib.util
import pathlib
import sys

_SERVICE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "app"


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_load("app._numeric_evidence", _SERVICE_ROOT / "_numeric_evidence.py")
_eg = _load("_dgp_evidence_gate_pred", _SERVICE_ROOT / "evidence_gate.py")

_is_artifact = _eg._is_predecessor_slash_artifact
normalize = _eg.normalize_evidence_text


class TestPredecessorSlashArtifactPositive:
    """SA-25-shaped artifacts MUST be flagged."""

    def test_sa25_in_production_phrase(self):
        ev = normalize(
            "The configuration of the 1D through 5Ya23 missiles is a "
            "direct evolution of the earlier SA-25/S-25 / SA-1 Guild, "
            "itself evolved from the 1944 German Wasserfall design."
        )
        assert _is_artifact("SA-25", ev) is True

    def test_predecessor_with_evolved_from(self):
        ev = normalize("The Nike system evolved from the FAKE-99/Nike-Ajax ancestor.")
        assert _is_artifact("FAKE-99", ev) is True

    def test_precursor_marker(self):
        ev = normalize("Its precursor X-77/X-7 platform was retired.")
        assert _is_artifact("X-77", ev) is True

    def test_ancestor_marker(self):
        ev = normalize("Its ancestor BAD-1/BAD-2 design was unstable.")
        assert _is_artifact("BAD-1", ev) is True

    def test_superseded_marker(self):
        ev = normalize("Eventually superseded NOISE-1/SIGNAL-1 designs.")
        assert _is_artifact("NOISE-1", ev) is True


class TestPredecessorSlashArtifactNegative:
    """Real entities with multiple mentions or no predecessor context
    MUST NOT be flagged."""

    def test_legitimate_slash_group_without_predecessor_context(self):
        """A slash group without 'earlier'/'evolved from'/etc. before it
        is NOT an artifact (e.g. configuration variants listed in tables)."""
        ev = normalize("The 13D/13DM variants share the same airframe.")
        assert _is_artifact("13D", ev) is False

    def test_entity_with_multiple_mentions_not_flagged(self):
        """If the entity appears elsewhere standalone, it's source-
        supported even if it also appears in a slash group."""
        ev = normalize(
            "The earlier 13D/13DM variants were short-range. "
            "Modern 13D rounds achieve 34 km."
        )
        assert _is_artifact("13D", ev) is False

    def test_duplicate_occurrence_all_in_predecessor_context_flagged(self):
        """Docling stores `text` and `orig` fields with overlapping
        content, which puts each prose token in evidence_text twice. As
        long as EVERY occurrence is a predecessor-slash artifact, the
        entity must still be flagged."""
        # Two identical mentions, both predecessor-context + slash
        ev = normalize(
            "Direct evolution of the earlier SA-25/S-25 / SA-1 Guild. "
            "Direct evolution of the earlier SA-25/S-25 / SA-1 Guild."
        )
        assert _is_artifact("SA-25", ev) is True

    def test_at_least_one_legitimate_mention_overrides_artifacts(self):
        """If one occurrence is an artifact but another is a real
        standalone mention, the entity is source-supported. Keep."""
        ev = normalize(
            "The earlier SA-25/S-25 system was retired. "
            "Modern SA-25 batteries deploy worldwide."
        )
        assert _is_artifact("SA-25", ev) is False

    def test_entity_not_in_evidence(self):
        assert _is_artifact("MADE-UP-77", normalize("some text")) is False

    def test_entity_followed_by_space_not_slash(self):
        """`earlier X-99 system` (no slash) → not a slash-group token."""
        ev = normalize("The earlier X-99 system was retired.")
        assert _is_artifact("X-99", ev) is False

    def test_predecessor_context_too_far_away(self):
        """`earlier` >30 chars before the name → not in window."""
        ev = normalize(
            "The earlier system was retired. " * 3
            + "Now we discuss X-77/X-7 variants."
        )
        # X-77 here is preceded by "Now we discuss " — no predecessor markers
        # in the ~30-char window before X-77.
        assert _is_artifact("X-77", ev) is False

    def test_empty_inputs(self):
        assert _is_artifact("", "some text") is False
        assert _is_artifact("X", "") is False
        assert _is_artifact(None, "some text") is False


class TestNormalization:
    """The predicate uppercase-normalizes both the name and evidence;
    case-insensitivity is part of the contract."""

    def test_case_insensitive_match(self):
        ev = normalize("evolved from the earlier sa-25/s-25 system")
        assert _is_artifact("SA-25", ev) is True
        assert _is_artifact("sa-25", ev) is True
