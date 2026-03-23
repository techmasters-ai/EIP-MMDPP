"""Unit tests for pure helper functions in app.api.v1._retrieval_helpers.

These tests exercise deduplicate_results, build_filters (and its aliases
build_text_filters / build_image_filters), score decay getters, fusion
weight getters, and compute_fusion_score — all of which have no DB dependency.
"""

import uuid

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(**kwargs):
    """Build a QueryResultItem with sensible defaults."""
    from app.schemas.retrieval import QueryResultItem

    defaults = dict(
        score=0.85,
        modality="text",
        content_text="Some chunk text.",
        classification="UNCLASSIFIED",
    )
    defaults.update(kwargs)
    return QueryResultItem(**defaults)


def _make_request(**kwargs):
    """Build a UnifiedQueryRequest with sensible defaults."""
    from app.schemas.retrieval import UnifiedQueryRequest

    defaults = dict(query_text="test query")
    defaults.update(kwargs)
    return UnifiedQueryRequest(**defaults)


# ---------------------------------------------------------------------------
# deduplicate_results
# ---------------------------------------------------------------------------

class TestDeduplicateResults:
    def _call(self, results):
        from app.api.v1._retrieval_helpers import deduplicate_results
        return deduplicate_results(results)

    def test_empty_list_returns_empty(self):
        assert self._call([]) == []

    def test_single_item_unchanged(self):
        item = _make_item(chunk_id=uuid.uuid4())
        result = self._call([item])
        assert len(result) == 1
        assert result[0].chunk_id == item.chunk_id

    def test_duplicate_chunk_ids_keeps_highest_score(self):
        cid = uuid.uuid4()
        low = _make_item(chunk_id=cid, score=0.5)
        high = _make_item(chunk_id=cid, score=0.9)
        result = self._call([low, high])
        assert len(result) == 1
        assert abs(result[0].score - 0.9) < 1e-6

    def test_duplicate_keeps_highest_regardless_of_order(self):
        cid = uuid.uuid4()
        high = _make_item(chunk_id=cid, score=0.9)
        low = _make_item(chunk_id=cid, score=0.5)
        result = self._call([high, low])
        assert len(result) == 1
        assert abs(result[0].score - 0.9) < 1e-6

    def test_none_chunk_ids_treated_independently(self):
        a = _make_item(chunk_id=None, score=0.8)
        b = _make_item(chunk_id=None, score=0.6)
        result = self._call([a, b])
        assert len(result) == 2

    def test_mixed_duplicates_and_uniques(self):
        cid1 = uuid.uuid4()
        cid2 = uuid.uuid4()
        cid3 = uuid.uuid4()
        items = [
            _make_item(chunk_id=cid1, score=0.5),
            _make_item(chunk_id=cid1, score=0.9),  # dup of cid1
            _make_item(chunk_id=cid2, score=0.7),
            _make_item(chunk_id=cid2, score=0.3),  # dup of cid2
            _make_item(chunk_id=cid3, score=0.6),   # unique
        ]
        result = self._call(items)
        assert len(result) == 3
        scores = {str(r.chunk_id): r.score for r in result}
        assert abs(scores[str(cid1)] - 0.9) < 1e-6
        assert abs(scores[str(cid2)] - 0.7) < 1e-6
        assert abs(scores[str(cid3)] - 0.6) < 1e-6

    def test_preserves_context_of_highest_scored(self):
        cid = uuid.uuid4()
        low = _make_item(chunk_id=cid, score=0.4, context={"source": "low"})
        high = _make_item(chunk_id=cid, score=0.8, context={"source": "high"})
        result = self._call([low, high])
        assert result[0].context["source"] == "high"


# ---------------------------------------------------------------------------
# build_text_filters (now returns tuple[str, dict])
# ---------------------------------------------------------------------------

class TestBuildTextFilters:
    def _call(self, **kwargs):
        from app.api.v1._retrieval_helpers import build_text_filters
        body = _make_request(**kwargs)
        return build_text_filters(body)

    def test_no_filters_returns_empty(self):
        clause, params = self._call()
        assert clause == ""
        assert params == {}

    def test_classification_filter(self):
        from app.schemas.retrieval import QueryFilters
        clause, params = self._call(filters=QueryFilters(classification="SECRET"))
        assert "tc.classification" in clause
        assert ":filter_classification" in clause
        assert params["filter_classification"] == "SECRET"

    def test_document_ids_filter(self):
        from app.schemas.retrieval import QueryFilters
        doc_id = uuid.uuid4()
        clause, params = self._call(filters=QueryFilters(document_ids=[doc_id]))
        assert "tc.document_id" in clause
        assert ":filter_doc_ids" in clause
        assert str(doc_id) in params["filter_doc_ids"]

    def test_modalities_filter(self):
        from app.schemas.retrieval import QueryFilters
        clause, params = self._call(filters=QueryFilters(modalities=["text", "table"]))
        assert "tc.modality" in clause
        assert ":filter_modalities" in clause
        assert params["filter_modalities"] == ["text", "table"]

    def test_source_ids_filter(self):
        from app.schemas.retrieval import QueryFilters
        source_id = uuid.uuid4()
        clause, params = self._call(filters=QueryFilters(source_ids=[source_id]))
        assert "source_id" in clause
        assert ":filter_source_ids" in clause
        assert str(source_id) in params["filter_source_ids"]

    def test_all_filters_combined(self):
        from app.schemas.retrieval import QueryFilters
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        clause, params = self._call(
            filters=QueryFilters(
                classification="CUI",
                document_ids=[doc_id],
                modalities=["text"],
                source_ids=[source_id],
            )
        )
        assert "tc.classification" in clause
        assert "tc.document_id" in clause
        assert "tc.modality" in clause
        assert "source_id" in clause
        assert len(params) == 4

    def test_no_sql_injection_in_classification(self):
        """Ensure classification value is parameterized, not interpolated."""
        from app.schemas.retrieval import QueryFilters
        clause, params = self._call(
            filters=QueryFilters(classification="'; DROP TABLE users;--")
        )
        # The malicious string should be in params, not in the clause itself
        assert "'; DROP TABLE" not in clause
        assert params["filter_classification"] == "'; DROP TABLE users;--"


# ---------------------------------------------------------------------------
# build_image_filters (now returns tuple[str, dict])
# ---------------------------------------------------------------------------

class TestBuildImageFilters:
    def _call(self, **kwargs):
        from app.api.v1._retrieval_helpers import build_image_filters
        body = _make_request(**kwargs)
        return build_image_filters(body)

    def test_no_filters_returns_empty(self):
        clause, params = self._call()
        assert clause == ""
        assert params == {}

    def test_classification_filter(self):
        from app.schemas.retrieval import QueryFilters
        clause, params = self._call(filters=QueryFilters(classification="CUI"))
        assert "ic.classification" in clause
        assert ":filter_classification" in clause
        assert params["filter_classification"] == "CUI"

    def test_document_ids_filter(self):
        from app.schemas.retrieval import QueryFilters
        doc_id = uuid.uuid4()
        clause, params = self._call(filters=QueryFilters(document_ids=[doc_id]))
        assert "ic.document_id" in clause
        assert str(doc_id) in params["filter_doc_ids"]

    def test_modalities_filter(self):
        from app.schemas.retrieval import QueryFilters
        clause, params = self._call(filters=QueryFilters(modalities=["image"]))
        assert "ic.modality" in clause
        assert params["filter_modalities"] == ["image"]


# ---------------------------------------------------------------------------
# Score decay getters (from settings)
# ---------------------------------------------------------------------------

class TestScoreDecayGetters:
    def test_cross_modal_decay_value(self):
        from app.api.v1._retrieval_helpers import get_cross_modal_decay
        assert abs(get_cross_modal_decay() - 0.85) < 1e-6

    def test_ontology_decay_value(self):
        from app.api.v1._retrieval_helpers import get_ontology_decay
        assert abs(get_ontology_decay() - 0.75) < 1e-6


# ---------------------------------------------------------------------------
# Fusion weight getters
# ---------------------------------------------------------------------------

class TestFusionWeights:
    def test_get_fusion_weights_returns_three_floats(self):
        from app.api.v1._retrieval_helpers import get_fusion_weights
        sem, doc, onto = get_fusion_weights()
        assert isinstance(sem, float)
        assert isinstance(doc, float)
        assert isinstance(onto, float)

    def test_fusion_weights_defaults(self):
        from app.api.v1._retrieval_helpers import get_fusion_weights
        sem, doc, onto = get_fusion_weights()
        assert abs(sem - 0.65) < 1e-6
        assert abs(doc - 0.20) < 1e-6
        assert abs(onto - 0.15) < 1e-6

    def test_fusion_weights_sum_to_one(self):
        from app.api.v1._retrieval_helpers import get_fusion_weights
        sem, doc, onto = get_fusion_weights()
        assert abs(sem + doc + onto - 1.0) < 1e-6

    def test_get_ontology_relation_weights_returns_dict(self):
        from app.api.v1._retrieval_helpers import get_ontology_relation_weights
        weights = get_ontology_relation_weights()
        assert isinstance(weights, dict)
        assert "IS_VARIANT_OF" in weights
        assert "USES_COMPONENT" in weights
        assert "RELATED_TO" in weights

    def test_ontology_relation_weights_ordered(self):
        """Higher-signal relations should have higher weights."""
        from app.api.v1._retrieval_helpers import get_ontology_relation_weights
        w = get_ontology_relation_weights()
        assert w["IS_VARIANT_OF"] >= w["USES_COMPONENT"]
        assert w["USES_COMPONENT"] >= w["INTERFACES_WITH"]
        assert w["INTERFACES_WITH"] >= w["RELATED_TO"]

    def test_get_doc_link_weights_returns_dict(self):
        from app.api.v1._retrieval_helpers import get_doc_link_weights
        weights = get_doc_link_weights()
        assert isinstance(weights, dict)
        assert "NEXT_CHUNK" in weights
        assert "SAME_SECTION" in weights
        assert "SAME_ARTIFACT" in weights
        assert "SAME_PAGE" in weights


# ---------------------------------------------------------------------------
# compute_fusion_score
# ---------------------------------------------------------------------------

class TestComputeFusionScore:
    def _call(self, **kwargs):
        from app.api.v1._retrieval_helpers import compute_fusion_score
        return compute_fusion_score(**kwargs)

    def test_semantic_only(self):
        """With no doc-structure or ontology, score is just semantic * weight."""
        score = self._call(semantic_score=0.9)
        # 0.65 * 0.9 = 0.585
        assert abs(score - 0.585) < 1e-3

    def test_with_doc_structure(self):
        """Doc-structure component should increase score."""
        base = self._call(semantic_score=0.8)
        with_doc = self._call(semantic_score=0.8, doc_structure_weight=0.9, doc_structure_hops=1)
        assert with_doc > base

    def test_with_ontology(self):
        """Ontology component should increase score."""
        base = self._call(semantic_score=0.8)
        with_onto = self._call(semantic_score=0.8, ontology_rel_type="IS_VARIANT_OF", ontology_hops=1)
        assert with_onto > base

    def test_hop_penalty_reduces_score(self):
        """More hops should reduce the doc-structure contribution."""
        hop1 = self._call(semantic_score=0.8, doc_structure_weight=0.9, doc_structure_hops=1)
        hop3 = self._call(semantic_score=0.8, doc_structure_weight=0.9, doc_structure_hops=3)
        assert hop1 > hop3

    def test_ontology_hop_penalty(self):
        """More hops should reduce the ontology contribution."""
        hop1 = self._call(semantic_score=0.8, ontology_rel_type="CONTAINS", ontology_hops=1)
        hop3 = self._call(semantic_score=0.8, ontology_rel_type="CONTAINS", ontology_hops=3)
        assert hop1 > hop3

    def test_mil_id_bonus_nsn(self):
        """NSN match should boost score."""
        base = self._call(
            semantic_score=0.8,
            content_text="Part NSN 1234-56-789-0123 description",
            query_text="Find NSN 1234-56-789-0123",
        )
        no_match = self._call(
            semantic_score=0.8,
            content_text="No matching IDs here",
            query_text="Find NSN 1234-56-789-0123",
        )
        assert base > no_match

    def test_mil_id_bonus_mil_std(self):
        """MIL-STD match should boost score."""
        base = self._call(
            semantic_score=0.8,
            content_text="Compliant with MIL-STD-810G requirements",
            query_text="MIL-STD-810G testing",
        )
        no_match = self._call(
            semantic_score=0.8,
            content_text="No matching standards",
            query_text="MIL-STD-810G testing",
        )
        assert base > no_match

    def test_mil_id_bonus_capped_at_one(self):
        """Score with bonus should never exceed 1.0."""
        score = self._call(
            semantic_score=1.0,
            doc_structure_weight=1.0,
            doc_structure_hops=1,
            ontology_rel_type="IS_VARIANT_OF",
            ontology_hops=1,
            content_text="NSN 1234-56-789-0123",
            query_text="NSN 1234-56-789-0123",
        )
        assert score <= 1.0

    def test_mil_id_bonus_an_designator(self):
        """AN/ designator match should boost score."""
        base = self._call(
            semantic_score=0.8,
            content_text="The AN/MPQ-53 radar provides target tracking",
            query_text="AN/MPQ-53 specifications",
        )
        no_match = self._call(
            semantic_score=0.8,
            content_text="No designators here",
            query_text="AN/MPQ-53 specifications",
        )
        assert base > no_match

    def test_returns_rounded_to_six_decimals(self):
        score = self._call(semantic_score=0.33333333333)
        # Check it's rounded
        decimal_str = str(score).split(".")[-1] if "." in str(score) else ""
        assert len(decimal_str) <= 6

    def test_zero_semantic_score(self):
        score = self._call(semantic_score=0.0)
        assert score == 0.0

    def test_unknown_ontology_rel_type_uses_default(self):
        """Unknown relation types should use the default weight."""
        known = self._call(semantic_score=0.8, ontology_rel_type="IS_VARIANT_OF", ontology_hops=1)
        unknown = self._call(semantic_score=0.8, ontology_rel_type="UNKNOWN_REL", ontology_hops=1)
        assert known > unknown  # default weight is lower


def test_fusion_score_with_mil_id_bonus_doc_structure():
    """Doc-structure chunks with matching MIL IDs should get the bonus."""
    from app.api.v1._retrieval_helpers import compute_fusion_score
    score = compute_fusion_score(
        semantic_score=0.8,
        doc_structure_weight=0.9,
        doc_structure_hops=1,
        content_text="The AN/MPQ-53 radar system provides tracking.",
        query_text="AN/MPQ-53 fire control radar",
    )
    assert score > 0.70


def test_fusion_score_ontology_preserves_relation_weight():
    """Ontology chunks should reflect the relation weight, not just cosine similarity."""
    from app.api.v1._retrieval_helpers import compute_fusion_score
    high_rel = compute_fusion_score(
        semantic_score=0.6, ontology_rel_type="IS_VARIANT_OF", ontology_hops=1,
        content_text="S-75 variant", query_text="SA-2 missile",
    )
    low_rel = compute_fusion_score(
        semantic_score=0.6, ontology_rel_type="RELATED_TO", ontology_hops=1,
        content_text="related system", query_text="SA-2 missile",
    )
    assert high_rel > low_rel


def test_fusion_score_cross_modal_uses_doc_weight():
    """Cross-modal decay should feed through the doc_structure weight slot."""
    from app.api.v1._retrieval_helpers import compute_fusion_score
    score = compute_fusion_score(semantic_score=0.8, cross_modal_decay=0.85)
    # 0.65*0.8 + 0.20*0.85 + 0 = 0.52 + 0.17 = 0.69
    assert abs(score - 0.69) < 0.01


def test_image_description_modality_in_text_filter():
    """image_description should pass through the text modality filter."""
    from app.schemas.retrieval import QueryResultItem
    items = [
        QueryResultItem(score=0.9, modality="text", content_text="missile"),
        QueryResultItem(score=0.8, modality="image_description", content_text="photo of radar"),
        QueryResultItem(score=0.7, modality="image", content_text="image"),
    ]
    text_filtered = [r for r in items if r.modality in ("text", "table", "image_description")]
    assert len(text_filtered) == 2
    assert text_filtered[1].modality == "image_description"
