from types import SimpleNamespace
from app.services.extraction_candidate_scoring import select_candidates, MergedCandidate


def _mc(idx, text, refs, cos):
    return MergedCandidate(
        candidate_key=f"r:chunk_{idx}",
        chunk_index=idx,
        self_ref=f"chunk_{idx}",
        chunk_text=text,
        source_refs=refs,
        token_count=10,
        page_number=1,
        vector_score=0.0,
        field_scores={},
        alias_hits=0,
        pattern_hits=0,
        negative_hits=0,
        section_hits=0,
        content_type=None,
        retrieval_sources=set(),
        supported_field_hints=set(),
        max_field_cosine=cos,
    )


def _cfg(tau=0.55):
    return SimpleNamespace(
        selection_mode="absolute_union",
        cosine_tau=tau,
        top_k=15,
        signal_dimensions={"length", "angle"},
        signal_categorical=set(),
        signal_has_image=False,
    )


def test_keeps_measurement_and_cosine_drops_rest():
    cands = [
        (_mc(0, "max range 2500 km", [], 0.1), 0.9),     # measurement fires
        (_mc(1, "the colonel said hello", [], 0.9), 0.4), # cosine >= tau
        (_mc(2, "unrelated prose", [], 0.1), 0.3),        # nothing fires
    ]
    diag = {}
    out = select_candidates(cands, [{}] * 3, _cfg(), diag_out=diag)
    kept = {c.chunk_index for c, _ in out}
    assert kept == {0, 1}
    assert diag["selection_k"] == 2


def test_empty_when_nothing_fires():
    cands = [
        (_mc(0, "prose", [], 0.1), 0.3),
        (_mc(1, "more prose", [], 0.2), 0.2),
    ]
    out = select_candidates(cands, [{}] * 2, _cfg(), diag_out={})
    assert out == []  # no k_min floor — genuinely 0


def test_diag_per_signal_counts():
    cands = [
        (_mc(0, "range 30 degrees", [], 0.1), 0.9),          # measurement (angle)
        (_mc(1, "the colonel said hello", [], 0.9), 0.4),     # cosine only
        (_mc(2, "unrelated prose", [], 0.1), 0.3),            # nothing
    ]
    diag = {}
    out = select_candidates(cands, [{}] * 3, _cfg(), diag_out=diag)
    assert diag["selection_mode"] == "absolute_union"
    assert diag["selection_k"] == 2
    assert diag["measurement_keeps"] >= 1
    assert diag["cosine_keeps"] >= 1
    assert diag["image_keeps"] == 0


def test_categorical_signal_keeps_candidate():
    cfg = SimpleNamespace(
        selection_mode="absolute_union", cosine_tau=0.55, top_k=15,
        signal_dimensions=set(), signal_categorical={"guidance_type"}, signal_has_image=False,
    )
    cands = [
        (_mc(0, "uses semi-active radar homing", [], 0.1), 0.5),  # categorical fires
        (_mc(1, "unrelated prose", [], 0.1), 0.2),                # nothing fires
    ]
    diag = {}
    out = select_candidates(cands, [{}] * 2, cfg, diag_out=diag)
    assert {c.chunk_index for c, _ in out} == {0}
    assert diag["categorical_keeps"] == 1
    assert diag["measurement_keeps"] == 0


def test_no_crash_with_diag_out_none():
    cands = [(_mc(0, "max range 2500 km", [], 0.1), 0.5)]
    out = select_candidates(cands, [{}], _cfg(), diag_out=None)
    assert isinstance(out, list)


def test_image_signal_keeps_candidate():
    cfg = SimpleNamespace(
        selection_mode="absolute_union",
        cosine_tau=0.55,
        top_k=15,
        signal_dimensions=set(),
        signal_categorical=set(),
        signal_has_image=True,
    )
    cands = [
        (_mc(0, "unrelated prose", ["#/pictures/3"], 0.1), 0.3),  # image fires
        (_mc(1, "more prose", [], 0.1), 0.2),                     # nothing fires
    ]
    diag = {}
    out = select_candidates(cands, [{}] * 2, cfg, diag_out=diag)
    kept = {c.chunk_index for c, _ in out}
    assert kept == {0}
    assert diag["image_keeps"] == 1
    assert diag["cosine_keeps"] == 0
