from app.services.rrf_fusion import (
    assign_ranks, rrf_score, display_score, fuse, apply_expansion_floor, FusedUnit,
)

def test_assign_ranks_contiguous_stable():
    ranks = assign_ranks([("a", 0.9), ("b", 0.5), ("c", 0.9)])
    assert ranks == {"a": 1, "c": 2, "b": 3}

def test_rrf_score_sums_present_signals():
    r = rrf_score({"text": 1, "visual": 1}, {"text": 1.0, "visual": 1.0}, k=20)
    assert abs(r - (1/21 + 1/21)) < 1e-9

def test_display_monotonic_and_anchor():
    assert abs(display_score(1/21, 0.05) - 0.4878) < 1e-3
    assert display_score(0.10, 0.05) > display_score(0.05, 0.05)

def test_fuse_text_wins_tie_over_lone_image():
    units = [
        FusedUnit(id="img", signals={"visual": 1}, text_bearing=False),
        FusedUnit(id="txt", signals={"text": 1}, text_bearing=True),
    ]
    out = fuse(units, {"text": 1.0, "visual": 1.0, "ontology": 0.5}, k=20, c=0.05)
    assert out[0].id == "txt" and out[1].id == "img"

def test_expansion_floor_never_evicts_and_caps_below():
    fused = fuse([FusedUnit(id="t1", signals={"text": 1}, text_bearing=True)],
                 {"text":1.0,"visual":1.0,"ontology":0.5}, k=20, c=0.05)
    floored = apply_expansion_floor(fused_units=fused, expansion_candidates=[("e1", 0.40)],
                                    top_k=20, floor_slots=2, display_scale=0.05)
    ids = [u.id for u in floored]
    assert "t1" in ids and "e1" in ids
    t1 = next(u for u in floored if u.id == "t1"); e1 = next(u for u in floored if u.id == "e1")
    assert e1.display < t1.display

def test_expansion_floor_no_evict_on_full_topk():
    fused = fuse([FusedUnit(id=f"t{i}", signals={"text": i+1}, text_bearing=True) for i in range(20)],
                 {"text":1.0,"visual":1.0,"ontology":0.5}, k=20, c=0.05)
    out = apply_expansion_floor(fused, [("e1", 0.4)], top_k=20, floor_slots=2, display_scale=0.05)
    assert len([u for u in out if u.id.startswith("t")]) == 20
