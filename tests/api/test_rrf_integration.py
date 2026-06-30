import json
import subprocess


def _query(payload):
    out = subprocess.check_output(["curl", "-s", "-X", "POST", "http://localhost:8005/v1/retrieval/query",
        "-H", "Content-Type: application/json", "-d", json.dumps(payload)])
    d = json.loads(out)
    return [(r["chunk_id"], round(r["score"], 6)) for r in d["results"]]


def _query_full(payload):
    out = subprocess.check_output(["curl", "-s", "-X", "POST", "http://localhost:8005/v1/retrieval/query",
        "-H", "Content-Type: application/json", "-d", json.dumps(payload)])
    return json.loads(out)["results"]


_HYBRID = {"strategy": "hybrid", "modality_filter": "all", "top_k": 20,
           "min_confidence": 0.1, "ontology_reserved_slots": 3}


def test_text_basic_unchanged():
    payload = {"query_text": "Fan Song", "strategy": "basic", "modality_filter": "all",
               "top_k": 20, "reranker_top_n": 20, "min_confidence": 0.1, "include_context": True}
    got = _query(payload)
    expected = [tuple(x) for x in json.load(open("tests/api/fixtures/text_basic_fan_song.json"))]
    assert got == expected


def test_hybrid_determinism():
    """RRF over a fixed candidate set is deterministic: same query -> same chunk_id list."""
    payload = {"query_text": "Fan Song", **_HYBRID, "top_k": 15}
    runs = [[r["chunk_id"] for r in _query_full(payload)] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
    assert len(runs[0]) > 0


def test_hybrid_modality_mix_radar_antenna():
    """A visual query interleaves images and text (cross-modal fusion works)."""
    results = _query_full({"query_text": "radar antenna", **_HYBRID})
    mods = {r["modality"] for r in results}
    assert "image" in mods and "text" in mods


def test_hybrid_ontology_floor_present():
    """The ontology floor + S_ontology surface qualifying ontology_relation chunks in top_k."""
    results = _query_full({"query_text": "SNR-75", **_HYBRID})
    n_onto = sum(1 for r in results if (r.get("context") or {}).get("source") == "ontology_relation")
    assert n_onto >= 1


def test_hybrid_merged_image_card_lineage():
    """A collapsed image+description card retains both source chunk_ids."""
    results = _query_full({"query_text": "Fan Song", **_HYBRID})
    merged = [r for r in results
              if len((r.get("context") or {}).get("merged_chunk_ids") or []) >= 2]
    # Fan Song's top hit is the agreement image (image + its description), so at least one merged card exists.
    assert merged, "expected at least one merged image+description card"
    assert set((merged[0]["context"].get("merged_sources") or [])) <= {"visual", "description"}


# NOTE: flag-off byte-identical equality is verified manually (the API container flag
# cannot be toggled from pytest). Captured 2026-06-30:
#   RETRIEVAL_RRF_FUSION_ENABLED=false -> hybrid "Fan Song": total 13 {text:10, image:3},
#   identical to the pre-RRF legacy hybrid (commit 7568ddc). Flag-on -> RRF ordering
#   (agreement image leads). See the plan's Task 8 close notes.
