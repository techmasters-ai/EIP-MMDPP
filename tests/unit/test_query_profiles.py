"""Unit tests for query profile schemas, ontology caching, and Cypher compilation."""

import pytest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from app.schemas.query_profiles import (
    QueryProfileStep,
    QueryProfileTraversal,
    QueryProfileDefinition,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------


class TestQueryProfileStep:
    """Validate QueryProfileStep field constraints."""

    def test_requires_at_least_one_rel_type(self):
        with pytest.raises(ValidationError):
            QueryProfileStep(rel_types=[])

    def test_rejects_empty_strings_in_rel_types(self):
        with pytest.raises(ValidationError):
            QueryProfileStep(rel_types=["", "   "])

    def test_strips_whitespace_from_rel_types(self):
        step = QueryProfileStep(rel_types=["  HAS_PART  ", "CONTAINS"])
        assert step.rel_types == ["HAS_PART", "CONTAINS"]

    def test_default_direction_is_out(self):
        step = QueryProfileStep(rel_types=["REL"])
        assert step.direction == "out"

    def test_direction_in_accepted(self):
        step = QueryProfileStep(rel_types=["REL"], direction="in")
        assert step.direction == "in"

    def test_invalid_direction_rejected(self):
        with pytest.raises(ValidationError):
            QueryProfileStep(rel_types=["REL"], direction="both")

    def test_default_hops(self):
        step = QueryProfileStep(rel_types=["REL"])
        assert step.min_hops == 1
        assert step.max_hops == 1

    def test_max_hops_gte_min_hops(self):
        step = QueryProfileStep(rel_types=["REL"], min_hops=1, max_hops=3)
        assert step.max_hops == 3

    def test_max_hops_lt_min_hops_rejected(self):
        with pytest.raises(ValidationError, match="max_hops"):
            QueryProfileStep(rel_types=["REL"], min_hops=3, max_hops=1)

    def test_hops_equal_is_valid(self):
        step = QueryProfileStep(rel_types=["REL"], min_hops=2, max_hops=2)
        assert step.min_hops == step.max_hops == 2

    def test_min_hops_below_lower_bound_rejected(self):
        with pytest.raises(ValidationError):
            QueryProfileStep(rel_types=["REL"], min_hops=0)

    def test_max_hops_above_upper_bound_rejected(self):
        with pytest.raises(ValidationError):
            QueryProfileStep(rel_types=["REL"], max_hops=5)


class TestQueryProfileTraversal:
    """Validate QueryProfileTraversal constraints."""

    def test_requires_at_least_one_step(self):
        with pytest.raises(ValidationError):
            QueryProfileTraversal(steps=[])

    def test_accepts_single_step(self):
        step = QueryProfileStep(rel_types=["REL"])
        traversal = QueryProfileTraversal(steps=[step])
        assert len(traversal.steps) == 1

    def test_rejects_more_than_three_steps(self):
        steps = [QueryProfileStep(rel_types=["REL"]) for _ in range(4)]
        with pytest.raises(ValidationError):
            QueryProfileTraversal(steps=steps)


class TestQueryProfileDefinition:
    """Validate QueryProfileDefinition shape and validators."""

    @staticmethod
    def _section_profile(**overrides):
        defaults = dict(
            id="test_section",
            label="Test Section",
            kind="section",
            traversals=[
                QueryProfileTraversal(
                    steps=[QueryProfileStep(rel_types=["HAS_COMPONENT"])]
                )
            ],
        )
        defaults.update(overrides)
        return QueryProfileDefinition(**defaults)

    @staticmethod
    def _dossier_profile(**overrides):
        defaults = dict(
            id="test_dossier",
            label="Test Dossier",
            kind="dossier",
            section_profile_ids=["sec_a"],
        )
        defaults.update(overrides)
        return QueryProfileDefinition(**defaults)

    def test_section_requires_at_least_one_traversal(self):
        with pytest.raises(ValidationError, match="at least one traversal"):
            QueryProfileDefinition(
                id="bad",
                label="Bad Section",
                kind="section",
                traversals=[],
            )

    def test_dossier_requires_at_least_one_section_profile_id(self):
        with pytest.raises(ValidationError, match="at least one section_profile_id"):
            QueryProfileDefinition(
                id="bad",
                label="Bad Dossier",
                kind="dossier",
                section_profile_ids=[],
            )

    def test_valid_section_profile(self):
        p = self._section_profile()
        assert p.kind == "section"
        assert len(p.traversals) == 1

    def test_valid_dossier_profile(self):
        p = self._dossier_profile()
        assert p.kind == "dossier"
        assert p.section_profile_ids == ["sec_a"]

    def test_strip_string_lists_cleans_whitespace(self):
        p = self._section_profile(
            root_entity_types=["  PLATFORM ", " RADAR_SYSTEM  "],
            target_entity_types=["  ANTENNA", ""],
        )
        assert p.root_entity_types == ["PLATFORM", "RADAR_SYSTEM"]
        assert p.target_entity_types == ["ANTENNA"]

    def test_strip_string_lists_removes_blank_entries(self):
        p = self._dossier_profile(
            section_profile_ids=["  sec_a  ", "", "   ", "sec_b"],
        )
        assert p.section_profile_ids == ["sec_a", "sec_b"]

    def test_default_kind_is_section(self):
        # kind defaults to "section", so providing traversals satisfies shape
        p = QueryProfileDefinition(
            id="x",
            label="X",
            traversals=[
                QueryProfileTraversal(
                    steps=[QueryProfileStep(rel_types=["REL"])]
                )
            ],
        )
        assert p.kind == "section"

    def test_dossier_with_traversals_allowed(self):
        """Dossier may carry traversals, but must have section_profile_ids."""
        p = self._dossier_profile(
            traversals=[
                QueryProfileTraversal(
                    steps=[QueryProfileStep(rel_types=["REL"])]
                )
            ],
        )
        assert p.kind == "dossier"
        assert len(p.traversals) == 1


# ---------------------------------------------------------------------------
# 2. Ontology loader caching
# ---------------------------------------------------------------------------


class TestOntologyCacheInvalidation:
    """Verify cache clear and hook dispatch without touching the database."""

    def test_invalidate_clears_cached_state(self):
        import app.services.ontology_templates as mod

        # Seed the module-level cache manually.
        with mod._cache_lock:
            mod._cached_default_ontology = {"entity_types": []}
            mod._cached_default_signature = "test:sig"
            mod._cached_default_expires_at = 9999999999.0

        mod.invalidate_ontology_cache()

        with mod._cache_lock:
            assert mod._cached_default_ontology is None
            assert mod._cached_default_signature is None
            assert mod._cached_default_expires_at == 0.0

    def test_register_invalidation_hook_fires_on_invalidate(self):
        import app.services.ontology_templates as mod

        callback = MagicMock()
        original_hooks = mod._invalidation_hooks[:]
        try:
            mod.register_invalidation_hook(callback)
            mod.invalidate_ontology_cache()
            callback.assert_called_once()
        finally:
            # Clean up so we do not pollute other tests.
            mod._invalidation_hooks[:] = original_hooks

    def test_invalidation_hook_failure_does_not_raise(self):
        import app.services.ontology_templates as mod

        def bad_hook():
            raise RuntimeError("boom")

        original_hooks = mod._invalidation_hooks[:]
        try:
            mod.register_invalidation_hook(bad_hook)
            # Should not propagate.
            mod.invalidate_ontology_cache()
        finally:
            mod._invalidation_hooks[:] = original_hooks

    def test_load_ontology_prefer_active_false_falls_back_to_yaml(self):
        """With prefer_active=False the loader must skip the DB and return
        the repository YAML directly."""
        from app.services.ontology_templates import load_ontology

        ontology = load_ontology(prefer_active=False)
        assert isinstance(ontology, dict)
        assert "entity_types" in ontology
        assert "relationship_types" in ontology


# ---------------------------------------------------------------------------
# 3. Cypher compilation
# ---------------------------------------------------------------------------


class TestCompileTraversalArm:
    """Test _compile_traversal_arm produces expected Cypher fragments."""

    def test_single_step_out_direction(self):
        from app.services.query_profiles import _compile_traversal_arm

        traversal = QueryProfileTraversal(
            steps=[
                QueryProfileStep(
                    direction="out",
                    rel_types=["HAS_COMPONENT"],
                    min_hops=1,
                    max_hops=2,
                )
            ]
        )
        cypher = _compile_traversal_arm(traversal)

        # Must reference the root alias and produce outward arrow.
        assert "(root)" in cypher
        assert "->(" in cypher
        assert "[:HAS_COMPONENT*1..2]" in cypher
        assert "RETURN n" in cypher

    def test_single_step_in_direction(self):
        from app.services.query_profiles import _compile_traversal_arm

        traversal = QueryProfileTraversal(
            steps=[
                QueryProfileStep(
                    direction="in",
                    rel_types=["PART_OF"],
                    min_hops=1,
                    max_hops=3,
                )
            ]
        )
        cypher = _compile_traversal_arm(traversal)

        # Inward arrow pattern
        assert "<-[" in cypher
        assert "[:PART_OF*1..3]" in cypher

    def test_multiple_rel_types_joined_with_pipe(self):
        from app.services.query_profiles import _compile_traversal_arm

        traversal = QueryProfileTraversal(
            steps=[
                QueryProfileStep(
                    direction="out",
                    rel_types=["HAS_SUBSYSTEM", "HAS_COMPONENT"],
                    min_hops=1,
                    max_hops=1,
                )
            ]
        )
        cypher = _compile_traversal_arm(traversal)
        assert "[:HAS_SUBSYSTEM|HAS_COMPONENT*1..1]" in cypher

    def test_multi_step_traversal(self):
        from app.services.query_profiles import _compile_traversal_arm

        traversal = QueryProfileTraversal(
            steps=[
                QueryProfileStep(
                    direction="out",
                    rel_types=["HAS_SUBSYSTEM"],
                    min_hops=1,
                    max_hops=1,
                ),
                QueryProfileStep(
                    direction="out",
                    rel_types=["EMITS"],
                    min_hops=1,
                    max_hops=2,
                ),
            ]
        )
        cypher = _compile_traversal_arm(traversal)

        # First step starts from root, second step from intermediate node.
        assert "MATCH p1 =" in cypher
        assert "MATCH p2 =" in cypher
        # Intermediate alias n1 should appear in the first MATCH but final
        # target should be just "n".
        assert "(n1" in cypher
        assert "RETURN n," in cypher or "RETURN n " in cypher

    def test_hop_count_aggregation(self):
        from app.services.query_profiles import _compile_traversal_arm

        traversal = QueryProfileTraversal(
            steps=[
                QueryProfileStep(rel_types=["A"], min_hops=1, max_hops=1),
                QueryProfileStep(rel_types=["B"], min_hops=1, max_hops=2),
            ]
        )
        cypher = _compile_traversal_arm(traversal)
        # hop_count should sum lengths of both paths.
        assert "length(p1)" in cypher
        assert "length(p2)" in cypher
        assert "AS hop_count" in cypher


class TestCompileSectionQuery:
    """Test _compile_section_query produces a full Cypher query."""

    @staticmethod
    def _make_section_profile(**overrides):
        defaults = dict(
            id="test_section",
            label="Test",
            kind="section",
            traversals=[
                QueryProfileTraversal(
                    steps=[
                        QueryProfileStep(
                            direction="out",
                            rel_types=["HAS_COMPONENT"],
                            min_hops=1,
                            max_hops=2,
                        )
                    ]
                )
            ],
        )
        defaults.update(overrides)
        return QueryProfileDefinition(**defaults)

    def test_produces_parameterized_query(self):
        from app.services.query_profiles import _compile_section_query

        profile = self._make_section_profile()
        cypher = _compile_section_query(profile)

        assert "$root_id" in cypher
        assert "$target_entity_types" in cypher
        assert "$limit" in cypher

    def test_contains_match_root(self):
        from app.services.query_profiles import _compile_section_query

        profile = self._make_section_profile()
        cypher = _compile_section_query(profile)
        assert "MATCH (root:Entity {id: $root_id})" in cypher

    def test_union_of_multiple_traversals(self):
        from app.services.query_profiles import _compile_section_query

        profile = self._make_section_profile(
            traversals=[
                QueryProfileTraversal(
                    steps=[QueryProfileStep(rel_types=["HAS_SUBSYSTEM"])]
                ),
                QueryProfileTraversal(
                    steps=[QueryProfileStep(rel_types=["PART_OF"], direction="in")]
                ),
            ]
        )
        cypher = _compile_section_query(profile)
        assert "UNION" in cypher

    def test_excludes_root_from_results(self):
        from app.services.query_profiles import _compile_section_query

        profile = self._make_section_profile()
        cypher = _compile_section_query(profile)
        assert "n.id <> $root_id" in cypher

    def test_rejects_non_section_profile(self):
        from app.services.query_profiles import _compile_section_query

        dossier = QueryProfileDefinition(
            id="d",
            label="D",
            kind="dossier",
            section_profile_ids=["x"],
        )
        with pytest.raises(ValueError, match="not a section profile"):
            _compile_section_query(dossier)

    def test_returns_expected_columns(self):
        from app.services.query_profiles import _compile_section_query

        profile = self._make_section_profile()
        cypher = _compile_section_query(profile)
        for col in ("node_id", "name", "canonical_name", "entity_type",
                     "properties", "rel_types", "hop_count"):
            assert col in cypher, f"Expected column '{col}' in compiled Cypher"
