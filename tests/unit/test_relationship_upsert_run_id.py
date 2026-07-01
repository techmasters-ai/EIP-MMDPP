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


# ---------------------------------------------------------------------------
# Re-upsert lineage regression (the real bug)
#
# The above tests only assert the lineage SET *string* is present. They pass
# even when the UPDATE that carries those SETs is gated by
# ``AND NOT (document_ids CONTAINS :doc_id)`` — which means on a re-ingest of
# the same (edge, doc) the UPDATE matches zero rows and pipeline_run_id /
# source_chunk_ids are NEVER written onto the pre-existing edge. These tests
# pin that the lineage/property SETs apply REGARDLESS of document_ids
# membership, while the document_ids append stays idempotent.
# ---------------------------------------------------------------------------


def _update_branch(script: str) -> str:
    """Return the UPDATE branch (the ELSE block) of the find-or-create script."""
    return script.split("} ELSE {", 1)[1]


def test_lineage_update_not_gated_by_document_ids_membership():
    """The UPDATE carrying lineage SETs must NOT be gated by a
    ``NOT (document_ids CONTAINS ...)`` predicate.

    Option A (chosen): a single UPDATE whose WHERE only matches the edge
    endpoints. Idempotent doc-id append lives in the SET via a CASE
    expression, so the lineage SETs (pipeline_run_id, source_chunk_ids,
    updated_at, props) always apply on re-upsert.
    """
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    rec = _make_record(properties={"source_chunk_ids": ["c1", "c2"]})
    prov = _make_provenance(pipeline_run_id="run-123")
    script, _ = _build_upsert_relationship_script([rec], provenance=prov)

    update_branch = _update_branch(script)
    assert "pipeline_run_id" in update_branch, (
        f"pipeline_run_id SET missing from UPDATE branch:\n{update_branch}"
    )
    # The lineage SETs must not be gated by document_ids membership: the
    # gating predicate that buried them must be gone from the UPDATE.
    assert "NOT (document_ids CONTAINS" not in update_branch, (
        "UPDATE branch still gates the lineage SETs with "
        "'NOT (document_ids CONTAINS ...)'; on re-upsert of the same doc the "
        "UPDATE matches zero rows and pipeline_run_id/source_chunk_ids are "
        f"never written:\n{update_branch}"
    )


def test_document_ids_append_remains_idempotent_via_case():
    """The doc-id append must stay idempotent (no duplicate on re-ingest).

    With the membership gate removed from the WHERE, idempotency moves into
    the SET via a CASE expression so we never append a doc id already present.
    """
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    rec = _make_record()
    prov = _make_provenance(pipeline_run_id="run-123")
    script, _ = _build_upsert_relationship_script([rec], provenance=prov)

    update_branch = _update_branch(script)
    assert "CASE WHEN document_ids CONTAINS" in update_branch, (
        "idempotent doc-id append (CASE WHEN document_ids CONTAINS ... "
        f"ELSE document_ids || [...] END) missing from UPDATE branch:\n{update_branch}"
    )


def test_source_chunk_ids_set_in_update_branch():
    """source_chunk_ids (a record property) must appear in the UPDATE SETs so
    it is overwritten on re-upsert, not only on the initial CREATE."""
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    rec = _make_record(properties={"source_chunk_ids": ["c1", "c2"]})
    prov = _make_provenance(pipeline_run_id="run-123")
    script, _ = _build_upsert_relationship_script([rec], provenance=prov)

    update_branch = _update_branch(script)
    assert "source_chunk_ids" in update_branch, (
        "source_chunk_ids not SET in UPDATE branch; per-edge chunk lineage is "
        f"dropped on re-upsert:\n{update_branch}"
    )


# ---------------------------------------------------------------------------
# All-int list edge properties (source_pages) must serialize as INLINE SQL
# list literals, NOT bound params.
#
# Confirmed ArcadeDB bug: deserializing an all-numeric JSON array (e.g. [6,7])
# yields a Java primitive array (long[]); binding that to a schema-declared
# LIST column via ``SET field = :param`` coerces it into a NESTED one-element
# list → [[6,7]]. ``WHERE source_pages CONTAINS 6`` then matches 0 rows.
# String lists deserialize to List<String> and stay flat, so source_chunk_ids
# / source_self_refs / document_ids MUST remain bound params. The fix emits
# all-int lists as inline SQL int-list literals (values are pipeline ints —
# str(int(x)) is injection-safe).
# ---------------------------------------------------------------------------


def _create_branch(script: str) -> str:
    return script.split("} ELSE {")[0]


def test_source_pages_inline_literal_in_both_branches():
    """source_pages (all-int list) must appear as an inline SQL literal
    ``source_pages = [1, 2, 3]`` in BOTH the CREATE and UPDATE branches."""
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    rec = _make_record(properties={
        "source_pages": [1, 2, 3],
        "source_chunk_ids": ["c1", "c2"],
    })
    prov = _make_provenance(pipeline_run_id="run-123")
    script, params = _build_upsert_relationship_script([rec], provenance=prov)

    create_branch = _create_branch(script)
    update_branch = _update_branch(script)

    assert "source_pages = [1, 2, 3]" in create_branch, (
        f"inline source_pages literal missing from CREATE branch:\n{create_branch}"
    )
    assert "source_pages = [1, 2, 3]" in update_branch, (
        f"inline source_pages literal missing from UPDATE branch:\n{update_branch}"
    )


def test_source_pages_not_bound_param():
    """source_pages must NOT be bound as a p_source_pages_* param (that is the
    code path that triggers the [[...]] nesting bug)."""
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    rec = _make_record(properties={
        "source_pages": [1, 2, 3],
        "source_chunk_ids": ["c1", "c2"],
    })
    prov = _make_provenance(pipeline_run_id="run-123")
    script, params = _build_upsert_relationship_script([rec], provenance=prov)

    offenders = [k for k in params if k.startswith("p_source_pages_")]
    assert not offenders, (
        f"source_pages must be inlined, not bound; found param key(s): {offenders}"
    )
    # And the inline literal must not reference any :param placeholder.
    assert ":p_source_pages" not in script, (
        f"source_pages still referenced via bound param in script:\n{script}"
    )


def test_source_chunk_ids_stays_bound_param():
    """String lists (source_chunk_ids) MUST stay bound params — they
    deserialize flat and must NOT be inlined."""
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    rec = _make_record(properties={
        "source_pages": [1, 2, 3],
        "source_chunk_ids": ["c1", "c2"],
    })
    prov = _make_provenance(pipeline_run_id="run-123")
    script, params = _build_upsert_relationship_script([rec], provenance=prov)

    chunk_keys = [k for k in params if k.startswith("p_source_chunk_ids_")]
    assert chunk_keys, (
        f"source_chunk_ids must remain a bound param; params keys: {list(params.keys())}"
    )
    chunk_key = chunk_keys[0]
    assert params[chunk_key] == ["c1", "c2"]
    assert f":{chunk_key}" in script, (
        f"source_chunk_ids bound param :{chunk_key} not referenced in script:\n{script}"
    )
    # The string values must never appear inlined as a SQL literal.
    assert "['c1', 'c2']" not in script and '["c1", "c2"]' not in script, (
        f"source_chunk_ids string list was inlined — must stay bound:\n{script}"
    )


def test_single_int_list_inlined():
    """A single-int list [7] must inline as ``= [7]``."""
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    rec = _make_record(properties={"source_pages": [7]})
    script, params = _build_upsert_relationship_script([rec], provenance=None)

    assert "source_pages = [7]" in script, (
        f"single-int list not inlined as [7]:\n{script}"
    )
    assert not any(k.startswith("p_source_pages_") for k in params)


def test_empty_int_list_does_not_trigger_inline_bug():
    """An empty list must not crash; it doesn't trigger the nesting bug.
    Whichever path is chosen, the emitted value must be a flat ``[]`` and
    page-membership semantics are unaffected."""
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    rec = _make_record(properties={"source_pages": []})
    script, params = _build_upsert_relationship_script([rec], provenance=None)

    # Either inlined as [] or bound as a param holding []; never nested.
    bound = [k for k in params if k.startswith("p_source_pages_")]
    if bound:
        assert params[bound[0]] == []
    else:
        assert "source_pages = []" in script, (
            f"empty source_pages not represented as flat []:\n{script}"
        )


def test_bool_list_not_treated_as_int_list():
    """[True] must NOT be inlined as an int list — bool is a subclass of int
    but represents a different SQL type; it must stay a bound param."""
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    rec = _make_record(properties={"flag_list": [True, False]})
    script, params = _build_upsert_relationship_script([rec], provenance=None)

    bound = [k for k in params if k.startswith("p_flag_list_")]
    assert bound, (
        f"bool list must stay a bound param, not inlined; params: {list(params.keys())}"
    )
    assert params[bound[0]] == [True, False]
    # Must not have been inlined as an int literal.
    assert "flag_list = [1, 0]" not in script and "flag_list = [True" not in script, (
        f"bool list was incorrectly inlined:\n{script}"
    )


def test_sql_list_set_fragment_helper_int_list_inlines():
    """The shared helper inlines an all-int list and does not bind it."""
    from app.services.arcadedb_graph import _sql_list_set_fragment

    params: dict = {}
    frag = _sql_list_set_fragment("source_pages", [4, 5, 6], params, "p_source_pages_0")
    assert frag == "source_pages = [4, 5, 6]"
    assert params == {}


def test_sql_list_set_fragment_helper_string_list_binds():
    """The shared helper binds a string list as a param."""
    from app.services.arcadedb_graph import _sql_list_set_fragment

    params: dict = {}
    frag = _sql_list_set_fragment(
        "source_chunk_ids", ["c1", "c2"], params, "p_source_chunk_ids_0"
    )
    assert frag == "source_chunk_ids = :p_source_chunk_ids_0"
    assert params == {"p_source_chunk_ids_0": ["c1", "c2"]}


def test_sql_list_set_fragment_helper_bool_list_binds():
    """The shared helper does NOT treat a bool list as an int list."""
    from app.services.arcadedb_graph import _sql_list_set_fragment

    params: dict = {}
    frag = _sql_list_set_fragment("flag_list", [True], params, "p_flag_list_0")
    assert frag == "flag_list = :p_flag_list_0"
    assert params == {"p_flag_list_0": [True]}


# ---------------------------------------------------------------------------
# Live ArcadeDB integration (opt-in; skips cleanly when DB unreachable).
# Proves the chosen SQL actually overwrites lineage on a second upsert of the
# same (edge, doc) without duplicating the doc id.
# ---------------------------------------------------------------------------


def _arcadedb_reachable() -> bool:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        "http://localhost:2480/api/v1/query/eip_knowledge_graph",
        data=b'{"language":"sql","command":"SELECT 1 as one"}',
        headers={
            "Content-Type": "application/json",
            # root:eip_arcadedb_secret
            "Authorization": "Basic cm9vdDplaXBfYXJjYWRlZGJfc2VjcmV0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5):
            return True
    except (urllib.error.URLError, OSError):
        return False


def test_live_reupsert_overwrites_lineage_no_doc_dup():
    """Live: re-upsert SAME (edge, doc) with a NEW run id + chunk ids must
    overwrite pipeline_run_id/source_chunk_ids and keep document_ids unique."""
    import json
    import urllib.request

    import pytest

    if not _arcadedb_reachable():
        pytest.skip("ArcadeDB not reachable at localhost:2480")

    def cmd(language: str, command: str, params: dict | None = None):
        body = {"language": language, "command": command}
        if params is not None:
            body["params"] = params
        req = urllib.request.Request(
            "http://localhost:2480/api/v1/command/eip_knowledge_graph",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Basic cm9vdDplaXBfYXJjYWRlZGJfc2VjcmV0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    # Throwaway types so we never touch domain data.
    cmd("sqlscript",
        "CREATE VERTEX TYPE ReupV IF NOT EXISTS; "
        "CREATE EDGE TYPE ReupE IF NOT EXISTS")
    try:
        cmd("sqlscript",
            "DELETE FROM ReupE; DELETE FROM ReupV; "
            "CREATE VERTEX ReupV SET k=1; CREATE VERTEX ReupV SET k=2")
        # Initial create (run-OLD, doc D, chunks c1)
        cmd("sqlscript",
            "LET s = SELECT FROM ReupV WHERE k=1; "
            "LET d = SELECT FROM ReupV WHERE k=2; "
            "CREATE EDGE ReupE FROM $s[0] TO $d[0] "
            "SET document_ids=[:doc], pipeline_run_id=:rid, "
            "source_chunk_ids=:scids",
            {"doc": "D", "rid": "run-OLD", "scids": ["c1"]})

        # Re-upsert SAME edge, SAME doc D, NEW run id + chunks — exactly the
        # UPDATE shape the builder now emits (CASE idempotent append, ungated).
        cmd("sql",
            "UPDATE ReupE SET "
            "document_ids = (CASE WHEN document_ids CONTAINS :doc "
            "THEN document_ids ELSE document_ids || [:doc] END), "
            "updated_at = sysdate(), pipeline_run_id = :rid, "
            "source_chunk_ids = :scids "
            "WHERE @out = (SELECT FROM ReupV WHERE k=1)[0].@rid "
            "AND @in = (SELECT FROM ReupV WHERE k=2)[0].@rid",
            {"doc": "D", "rid": "run-NEW", "scids": ["c1", "c2", "c3"]})

        rows = cmd("sql",
            "SELECT document_ids, pipeline_run_id, source_chunk_ids "
            "FROM ReupE")["result"]
        assert len(rows) == 1, f"expected exactly one edge, got: {rows}"
        edge = rows[0]
        assert edge["pipeline_run_id"] == "run-NEW", (
            f"pipeline_run_id not overwritten on re-upsert: {edge}"
        )
        assert edge["source_chunk_ids"] == ["c1", "c2", "c3"], (
            f"source_chunk_ids not overwritten on re-upsert: {edge}"
        )
        assert edge["document_ids"] == ["D"], (
            f"document_ids gained a duplicate on re-upsert of same doc: {edge}"
        )
    finally:
        cmd("sqlscript",
            "DELETE FROM ReupE; DELETE FROM ReupV; "
            "DROP TYPE ReupE IF EXISTS UNSAFE; "
            "DROP TYPE ReupV IF EXISTS UNSAFE")


def test_live_source_pages_stays_flat_and_contains_matches():
    """Live: a real upsert through _build_upsert_relationship_script must store
    source_pages FLAT ([1,2,3], not [[1,2,3]]) on a schema-declared LIST column,
    and ``WHERE source_pages CONTAINS 1`` must match. This is the bug under
    test: bound numeric arrays nest; inline literals stay flat."""
    import json
    import urllib.request

    import pytest

    if not _arcadedb_reachable():
        pytest.skip("ArcadeDB not reachable at localhost:2480")

    from app.services.arcadedb_graph import _build_upsert_relationship_script

    def cmd(language: str, command: str, params: dict | None = None):
        body = {"language": language, "command": command}
        if params is not None:
            body["params"] = params
        req = urllib.request.Request(
            "http://localhost:2480/api/v1/command/eip_knowledge_graph",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Basic cm9vdDplaXBfYXJjYWRlZGJfc2VjcmV0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    # Throwaway vertex types + an edge type with a LIST-typed source_pages
    # column to reproduce the production schema shape exactly.
    cmd("sqlscript",
        "CREATE VERTEX TYPE SpV IF NOT EXISTS; "
        "CREATE EDGE TYPE SpE IF NOT EXISTS; "
        "CREATE PROPERTY SpE.source_pages IF NOT EXISTS LIST")
    try:
        # Endpoints now resolve on the normalized <field>_key (case-insensitive
        # identity), so the throwaway vertices must carry id_key exactly as the
        # node-upsert path would persist it (norm('R1')='r1', norm('M1')='m1').
        cmd("sqlscript",
            "DELETE FROM SpE; DELETE FROM SpV; "
            "CREATE VERTEX SpV SET id='R1', id_key='r1'; "
            "CREATE VERTEX SpV SET id='M1', id_key='m1'")

        rec = _make_record(
            from_type="SpV",
            to_type="SpV",
            rel_type="SpE",
            properties={"source_pages": [1, 2, 3]},
        )
        script, params = _build_upsert_relationship_script([rec], provenance=None)
        cmd("sqlscript", script, params)

        rows = cmd("sql", "SELECT source_pages FROM SpE")["result"]
        assert len(rows) == 1, f"expected exactly one edge, got: {rows}"
        assert rows[0]["source_pages"] == [1, 2, 3], (
            f"source_pages stored NESTED (bug) instead of flat: {rows[0]}"
        )

        hit = cmd("sql", "SELECT count(*) AS n FROM SpE "
                         "WHERE source_pages CONTAINS 1")["result"]
        assert hit[0]["n"] == 1, (
            f"WHERE source_pages CONTAINS 1 matched 0 rows — page-membership "
            f"filter broken by nested list: {hit}"
        )
    finally:
        cmd("sqlscript",
            "DELETE FROM SpE; DELETE FROM SpV; "
            "DROP TYPE SpE IF EXISTS UNSAFE; "
            "DROP TYPE SpV IF EXISTS UNSAFE")
