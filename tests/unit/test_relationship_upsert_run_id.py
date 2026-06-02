"""Lineage fix: domain relationship edges must persist pipeline_run_id.

Confirmed bug: committed domain relationship edges (ASSOCIATED_WITH, CUES, …)
had pipeline_run_id = NULL because _build_upsert_relationship_script never
emitted a `pipeline_run_id = :...` SET clause — even though the provenance
object passed to the upsert carries it and the schema defines it as a valid
edge column.

These tests pin the SQL builder directly (pure string builder, no live DB):
the generated script must SET pipeline_run_id in BOTH the CREATE EDGE branch
and the UPDATE branch, with the value bound in the params dict. When
provenance is None — or carries no pipeline_run_id — no spurious binding is
emitted.
"""


def _make_record(**kwargs):
    from app.services.graph_store import RelationshipRecord

    defaults = dict(
        from_type="RADAR",
        from_identity={"id": "R1"},
        to_type="MISSILE",
        to_identity={"id": "M1"},
        rel_type="ASSOCIATED_WITH",
        extraction_confidence=0.9,
    )
    defaults.update(kwargs)
    return RelationshipRecord(**defaults)


def _make_provenance(pipeline_run_id="run-123"):
    from app.services.graph_store import ProvenanceMetadata

    return ProvenanceMetadata(
        document_id="doc-abc",
        page_numbers=[1],
        pipeline_run_id=pipeline_run_id,
    )


def test_pipeline_run_id_bound_in_params():
    """provenance.pipeline_run_id must be bound in the params dict."""
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    rec = _make_record()
    prov = _make_provenance(pipeline_run_id="run-123")
    script, params = _build_upsert_relationship_script([rec], provenance=prov)

    run_id_params = {k: v for k, v in params.items() if v == "run-123"}
    assert run_id_params, (
        "pipeline_run_id 'run-123' not bound in params; "
        f"params keys: {list(params.keys())}"
    )


def test_pipeline_run_id_set_in_create_branch():
    """The CREATE EDGE branch must SET pipeline_run_id from the bound param."""
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    rec = _make_record()
    prov = _make_provenance(pipeline_run_id="run-123")
    script, params = _build_upsert_relationship_script([rec], provenance=prov)

    # Find the param key that carries the run id.
    run_id_keys = [k for k, v in params.items() if v == "run-123"]
    assert run_id_keys, "no param bound to 'run-123'"
    run_id_key = run_id_keys[0]

    create_branch = script.split("} ELSE {")[0]
    assert "pipeline_run_id" in create_branch, (
        f"pipeline_run_id not SET in CREATE branch:\n{create_branch}"
    )
    assert f":{run_id_key}" in create_branch, (
        f"pipeline_run_id param :{run_id_key} not referenced in CREATE branch:\n"
        f"{create_branch}"
    )


def test_pipeline_run_id_set_in_update_branch():
    """The UPDATE branch must also SET pipeline_run_id from the bound param."""
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    rec = _make_record()
    prov = _make_provenance(pipeline_run_id="run-123")
    script, params = _build_upsert_relationship_script([rec], provenance=prov)

    run_id_keys = [k for k, v in params.items() if v == "run-123"]
    assert run_id_keys, "no param bound to 'run-123'"
    run_id_key = run_id_keys[0]

    update_branch = script.split("} ELSE {", 1)[1]
    assert "pipeline_run_id" in update_branch, (
        f"pipeline_run_id not SET in UPDATE branch:\n{update_branch}"
    )
    assert f":{run_id_key}" in update_branch, (
        f"pipeline_run_id param :{run_id_key} not referenced in UPDATE branch:\n"
        f"{update_branch}"
    )


def test_pipeline_run_id_referenced_in_both_branches():
    """The same pipeline_run_id binding must appear in CREATE and UPDATE."""
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    rec = _make_record()
    prov = _make_provenance(pipeline_run_id="run-123")
    script, params = _build_upsert_relationship_script([rec], provenance=prov)

    run_id_keys = [k for k, v in params.items() if v == "run-123"]
    assert run_id_keys, "no param bound to 'run-123'"
    run_id_key = run_id_keys[0]

    occurrences = script.count(f":{run_id_key}")
    assert occurrences >= 2, (
        f"Expected pipeline_run_id binding :{run_id_key} in both CREATE and "
        f"UPDATE branches, found {occurrences} occurrence(s)"
    )


def test_no_pipeline_run_id_binding_when_provenance_none():
    """provenance=None must not emit any pipeline_run_id SET or binding."""
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    rec = _make_record()
    script, params = _build_upsert_relationship_script([rec], provenance=None)

    assert "pipeline_run_id" not in script, (
        f"unexpected pipeline_run_id in script when provenance=None:\n{script}"
    )
    assert not any("pipeline_run_id" in k for k in params), (
        f"unexpected pipeline_run_id param when provenance=None: {list(params.keys())}"
    )


def test_no_pipeline_run_id_binding_when_run_id_absent():
    """provenance present but pipeline_run_id=None must not emit the clause."""
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    rec = _make_record()
    prov = _make_provenance(pipeline_run_id=None)
    script, params = _build_upsert_relationship_script([rec], provenance=prov)

    assert "pipeline_run_id" not in script, (
        "unexpected pipeline_run_id in script when provenance.pipeline_run_id is "
        f"None:\n{script}"
    )
    assert not any("pipeline_run_id" in k for k in params), (
        "unexpected pipeline_run_id param when pipeline_run_id is None: "
        f"{list(params.keys())}"
    )
