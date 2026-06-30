from app.config import get_settings


def test_rrf_defaults():
    """Test that all 8 RRF config knobs have correct defaults."""
    s = get_settings()
    assert s.retrieval_rrf_fusion_enabled is True
    assert s.retrieval_rrf_k == 20
    assert s.retrieval_rrf_w_text == 1.0
    assert s.retrieval_rrf_w_visual == 1.0
    assert s.retrieval_rrf_w_ontology == 0.5
    assert s.retrieval_rrf_visual_min_prob == 0.35
    assert s.retrieval_rrf_ontology_min_slots == 3  # set to 3 in e269564 (matches v1 reserved-slots guarantee)
    assert s.retrieval_rrf_expansion_floor_slots == 2
    assert s.retrieval_rrf_display_scale == 0.05
