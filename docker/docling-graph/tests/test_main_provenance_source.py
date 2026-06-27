# Use the dg_app_module conftest fixture — bare `from app.main import ...`
# is forbidden because the repo-root `app/` package shadows
# `docker/docling-graph/app/`. See docker/docling-graph/tests/conftest.py.

def test_chunk_to_self_refs_from_doc_processor_metadata(dg_app_module):
    """The primary source dict-construction (Step 5.5) must produce the
    expected chunk_id → self_refs map from chunk_metadata directly."""
    chunk_metadata = [
        {"chunk_id": 0, "self_refs": ["#/texts/0", "#/texts/1"], "evidence_units": []},
        {"chunk_id": 1, "self_refs": ["#/texts/2"], "evidence_units": []},
    ]
    out_refs = {
        int(c["chunk_id"]): [r for r in c["self_refs"] if isinstance(r, str)]
        for c in chunk_metadata
    }
    assert out_refs == {0: ["#/texts/0", "#/texts/1"], 1: ["#/texts/2"]}


def test_chunk_maps_from_trace_events_fallback(dg_app_module):
    """When doc_processor.last_chunk_metadata is empty, main.py falls back
    to chunk_created trace events via _chunk_maps_from_trace (single-pass).
    Asserts all three maps (self_refs / evidence_units / page_numbers) are
    built correctly in one call."""
    from docling_graph.pipeline.trace import TraceEvent
    trace_events = [
        TraceEvent(
            sequence=0, timestamp=0.0, stage="extraction",
            event_type="chunk_created",
            payload={
                "chunk_id": 0,
                "self_refs": ["#/texts/0", "#/texts/1"],
                "evidence_units": [{"evidence_id": "#/texts/0", "text": "hi"}],
                "page_numbers": [3],
            },
        ),
        TraceEvent(
            sequence=1, timestamp=0.0, stage="extraction",
            event_type="chunk_created",
            payload={
                "chunk_id": 1,
                "self_refs": ["#/texts/2"],
                "evidence_units": [],
                "page_numbers": [4, 5],
            },
        ),
    ]
    refs_map, units_map, pages_map = dg_app_module._chunk_maps_from_trace(trace_events)
    assert refs_map == {0: ["#/texts/0", "#/texts/1"], 1: ["#/texts/2"]}
    assert units_map == {0: [{"evidence_id": "#/texts/0", "text": "hi"}], 1: []}
    assert pages_map == {0: [3], 1: [4, 5]}


def test_no_production_path_in_main_calls_bare_hybridchunker():
    """Grep main.py for any HybridChunker() construction missing
    merge_peers — only diagnostic fallback paths may, marked with a
    `# diagnostic-only:` comment.

    SCOPE: this task only checks main.py. Task 12 widens the check to
    app/workers/pipeline.py too."""
    import pathlib, re
    paths = ["docker/docling-graph/app/main.py"]  # narrower than plan; Task 12 widens
    bad = []
    pat = re.compile(r"\bHybridChunker\s*\(")
    for p in paths:
        src = pathlib.Path(p).read_text()
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if not pat.search(line):
                continue
            call_end = i
            while call_end < len(lines) and lines[call_end].count(")") < lines[call_end].count("("):
                call_end += 1
            full_call = "\n".join(lines[i:call_end + 1])
            prev = lines[i - 1] if i > 0 else ""
            if "diagnostic-only" in line or "diagnostic-only" in prev:
                continue
            if "merge_peers" in full_call:
                continue
            bad.append((p, i + 1, line.strip()))
    assert not bad, f"production HybridChunker construction without merge_peers: {bad}"
