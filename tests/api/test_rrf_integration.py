import json
import subprocess


def _query(payload):
    out = subprocess.check_output(["curl", "-s", "-X", "POST", "http://localhost:8005/v1/retrieval/query",
        "-H", "Content-Type: application/json", "-d", json.dumps(payload)])
    d = json.loads(out)
    return [(r["chunk_id"], round(r["score"], 6)) for r in d["results"]]


def test_text_basic_unchanged():
    payload = {"query_text": "Fan Song", "strategy": "basic", "modality_filter": "all",
               "top_k": 20, "reranker_top_n": 20, "min_confidence": 0.1, "include_context": True}
    got = _query(payload)
    expected = [tuple(x) for x in json.load(open("tests/api/fixtures/text_basic_fan_song.json"))]
    assert got == expected
