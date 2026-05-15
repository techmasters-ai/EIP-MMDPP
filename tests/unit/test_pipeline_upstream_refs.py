"""Tests for upstream-ref machinery in app/workers/pipeline.py.

The docling-graph service sees these refs on document_plus_entity_refs passes
(system_links), and the merge resolver matches from_ref_id / to_ref_id against
them. Bugs here silently produce UNKNOWN_REF_ID rejections, which is why
these tests exist before the implementation.
"""
from types import SimpleNamespace

from app.workers.pipeline import _extend_upstream_refs


class _FakePassResult:
    def __init__(self, entities_by_type: dict):
        self._by_type = entities_by_type

    def iter_entities_of_type(self, entity_type: str):
        return iter(self._by_type.get(entity_type, []))


ONTOLOGY = {
    "entity_types": [
        {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
        {"name": "MISSILE_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
    ],
}


class TestExtendUpstreamRefs:

    def _pass_def(self, primary_types):
        return SimpleNamespace(
            name="radar_domain",
            primary_entity_types=primary_types,
        )

    def test_two_entity_types_produce_unique_ids(self):
        """Bug fix: previous impl reset the counter per entity type, so
        E001/E002 from type A were overwritten by E001/E002 from type B."""
        refs: dict = {}
        pass_result = _FakePassResult({
            "RADAR_SYSTEM": [
                SimpleNamespace(system_name="Fan Song", confidence=0.9),
                SimpleNamespace(system_name="Big Bird", confidence=0.8),
            ],
            "MISSILE_SYSTEM": [
                SimpleNamespace(system_name="SA-2", confidence=0.9),
                SimpleNamespace(system_name="SA-3", confidence=0.8),
            ],
        })
        _extend_upstream_refs(
            refs, pass_result,
            self._pass_def(["RADAR_SYSTEM", "MISSILE_SYSTEM"]),
            ONTOLOGY,
        )
        assert sorted(refs.keys()) == ["E001", "E002", "E003", "E004"]
        # Ids are unique: each ref points to a distinct entity
        assert len({r.identity_values["system_name"] for r in refs.values()}) == 4

    def test_appending_to_existing_refs_continues_counter(self):
        """A second pass should not clobber refs accumulated from a prior pass."""
        refs: dict = {
            "E001": SimpleNamespace(
                pass_origin="reference",
                entity_type="SECTION",
                identity_values={"heading": "Intro"},
                display_label="Intro",
            ),
        }
        pass_result = _FakePassResult({
            "RADAR_SYSTEM": [SimpleNamespace(system_name="Fan Song")],
        })
        _extend_upstream_refs(
            refs, pass_result,
            self._pass_def(["RADAR_SYSTEM"]),
            ONTOLOGY,
        )
        assert "E001" in refs and refs["E001"].entity_type == "SECTION"
        assert "E002" in refs and refs["E002"].entity_type == "RADAR_SYSTEM"

    def test_identity_values_filter_to_ontology_identity_fields(self):
        """instance.__dict__ may have confidence, nomenclature, etc.; only
        ontology identity_fields belong in identity_values (merge compares
        by identity tuple, so extra keys fragment identity)."""
        refs: dict = {}
        pass_result = _FakePassResult({
            "RADAR_SYSTEM": [
                SimpleNamespace(
                    system_name="Fan Song",
                    nomenclature="SA-2-RADAR",
                    confidence=0.9,
                ),
            ],
        })
        _extend_upstream_refs(refs, pass_result, self._pass_def(["RADAR_SYSTEM"]), ONTOLOGY)
        assert list(refs["E001"].identity_values.keys()) == ["system_name"]
        assert refs["E001"].identity_values["system_name"] == "Fan Song"

    def test_display_label_is_populated(self):
        refs: dict = {}
        pass_result = _FakePassResult({
            "RADAR_SYSTEM": [SimpleNamespace(system_name="Fan Song")],
        })
        _extend_upstream_refs(refs, pass_result, self._pass_def(["RADAR_SYSTEM"]), ONTOLOGY)
        assert refs["E001"].display_label == "Fan Song"

    def test_unknown_ontology_type_is_skipped(self):
        """If a primary_entity_type isn't in the ontology, we shouldn't emit a
        ref with a malformed identity (previous impl would have crashed)."""
        refs: dict = {}
        pass_result = _FakePassResult({
            "UNREGISTERED_TYPE": [SimpleNamespace(name="X")],
        })
        _extend_upstream_refs(refs, pass_result, self._pass_def(["UNREGISTERED_TYPE"]), ONTOLOGY)
        assert refs == {}

    def test_empty_primary_types_is_noop(self):
        refs: dict = {}
        pass_result = _FakePassResult({})
        _extend_upstream_refs(refs, pass_result, self._pass_def([]), ONTOLOGY)
        assert refs == {}


from app.workers.pipeline import _is_valid_upstream_ref


class TestIsValidUpstreamRef:
    ONTOLOGY = {
        "entity_types": [
            {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
            {"name": "SPECIFICATION", "identity_fields": ["parameter", "value"], "identity_scope": "document"},
            # Some ontology entity types have no identity fields (e.g.
            # PROPULSION_STACK). They should not produce upstream refs.
            {"name": "PROPULSION_STACK", "identity_fields": [], "identity_scope": "global"},
        ],
    }

    def _ref(self, entity_type, identity_values):
        return SimpleNamespace(
            pass_origin="radar_domain",
            entity_type=entity_type,
            identity_values=identity_values,
            display_label="x",
        )

    def test_all_identity_fields_present_and_truthy_is_valid(self):
        assert _is_valid_upstream_ref(
            self._ref("RADAR_SYSTEM", {"system_name": "Fan Song"}),
            self.ONTOLOGY,
        ) is True

    def test_unknown_entity_type_invalid(self):
        assert _is_valid_upstream_ref(
            self._ref("BOGUS", {"system_name": "X"}),
            self.ONTOLOGY,
        ) is False

    def test_missing_identity_key_invalid(self):
        assert _is_valid_upstream_ref(
            self._ref("SPECIFICATION", {"parameter": "range"}),  # missing value
            self.ONTOLOGY,
        ) is False

    def test_none_identity_value_invalid(self):
        assert _is_valid_upstream_ref(
            self._ref("RADAR_SYSTEM", {"system_name": None}),
            self.ONTOLOGY,
        ) is False

    def test_empty_string_identity_value_invalid(self):
        assert _is_valid_upstream_ref(
            self._ref("RADAR_SYSTEM", {"system_name": ""}),
            self.ONTOLOGY,
        ) is False

    def test_whitespace_identity_value_invalid(self):
        assert _is_valid_upstream_ref(
            self._ref("RADAR_SYSTEM", {"system_name": "   "}),
            self.ONTOLOGY,
        ) is False

    def test_all_fields_truthy_multifield_valid(self):
        assert _is_valid_upstream_ref(
            self._ref("SPECIFICATION", {"parameter": "range", "value": "150"}),
            self.ONTOLOGY,
        ) is True

    def test_zero_identity_fields_invalid(self):
        """Rule (b): an entity type with no identity anchors can't be a
        useful upstream ref — nothing to hand the LLM and nothing merge
        can resolve."""
        assert _is_valid_upstream_ref(
            self._ref("PROPULSION_STACK", {}),
            self.ONTOLOGY,
        ) is False


from app.workers.pipeline import _select_upstream_refs_for_pass


class TestSelectUpstreamRefsForPass:

    def _ref(self, pass_origin, entity_type, identity_value):
        return SimpleNamespace(
            pass_origin=pass_origin,
            entity_type=entity_type,
            identity_values={"system_name": identity_value},
            display_label=identity_value,
        )

    ONTOLOGY = {
        "entity_types": [
            {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
            {"name": "MISSILE_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
            {"name": "IADS", "identity_fields": ["system_name"], "identity_scope": "global"},
        ],
    }

    def test_filters_by_depends_on(self):
        all_refs = {
            "E001": self._ref("radar_domain", "RADAR_SYSTEM", "Fan Song"),
            "E002": self._ref("missile_domain", "MISSILE_SYSTEM", "SA-2"),
            "E003": self._ref("other_systems", "IADS", "System-X"),
        }
        pass_def = SimpleNamespace(
            name="system_links",
            depends_on=["radar_domain", "missile_domain"],
            extracted_relationship_types=[],  # no rel-type narrowing
        )
        selected = _select_upstream_refs_for_pass(pass_def, all_refs, self.ONTOLOGY)
        assert set(selected.keys()) == {"E001", "E002"}

    def test_empty_depends_on_selects_nothing(self):
        all_refs = {"E001": self._ref("radar_domain", "RADAR_SYSTEM", "X")}
        pass_def = SimpleNamespace(
            name="reference", depends_on=[], extracted_relationship_types=[],
        )
        assert _select_upstream_refs_for_pass(pass_def, all_refs, self.ONTOLOGY) == {}

    def test_invalid_refs_are_dropped(self):
        """Shared validity rule: a ref whose identity values are None/empty is
        filtered out before the service ever sees it."""
        all_refs = {
            "E001": self._ref("radar_domain", "RADAR_SYSTEM", "Fan Song"),
            "E002": self._ref("radar_domain", "RADAR_SYSTEM", None),   # invalid
            "E003": self._ref("radar_domain", "RADAR_SYSTEM", ""),     # invalid
        }
        pass_def = SimpleNamespace(
            name="system_links",
            depends_on=["radar_domain"],
            extracted_relationship_types=[],  # no narrowing → all valid refs pass
        )
        selected = _select_upstream_refs_for_pass(pass_def, all_refs, self.ONTOLOGY)
        assert set(selected.keys()) == {"E001"}

    def test_sort_uses_ontology_identity_field_order_not_alphabetical(self):
        """Multi-field identities must sort by ontology-declared order to
        match LogicalIdentity's canonical identity_tuple. Alphabetical
        dict-key order would put 'parameter' after 'value' in a field list
        declared as [parameter, value], which would make the prompt
        preamble and the merge identity disagree on the first value."""
        ontology = {
            "entity_types": [
                # Declared order matters: parameter first, value second.
                {"name": "SPECIFICATION",
                 "identity_fields": ["parameter", "value"],
                 "identity_scope": "document"},
            ],
            "validation_matrix": [],
        }
        refs = {
            "E002": SimpleNamespace(
                pass_origin="radar_domain", entity_type="SPECIFICATION",
                identity_values={"parameter": "B", "value": "1"},
                display_label="B=1",
            ),
            "E001": SimpleNamespace(
                pass_origin="radar_domain", entity_type="SPECIFICATION",
                identity_values={"parameter": "A", "value": "2"},
                display_label="A=2",
            ),
        }
        pass_def = SimpleNamespace(
            name="system_links",
            depends_on=["radar_domain"],
            extracted_relationship_types=[],
        )
        selected = _select_upstream_refs_for_pass(pass_def, refs, ontology)
        # Sorted by (pass_origin, entity_type, (parameter, value))
        # in ontology-declared order:
        #   ("radar_domain", "SPECIFICATION", ("A", "2")) → A=2
        #   ("radar_domain", "SPECIFICATION", ("B", "1")) → B=1
        assert [r.display_label for r in selected.values()] == ["A=2", "B=1"]

    def test_narrows_to_validation_matrix_endpoint_types(self):
        """system_links extracts ASSOCIATED_WITH / CUES, which only connect
        system-level entities (validation_matrix rows 1118-1217). Upstream
        refs of types that can't source OR target any of those relationships
        (e.g. GUIDANCE_METHOD from missile_domain) must be dropped here,
        before the LLM ever sees them."""
        ontology = {
            "entity_types": [
                {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
                {"name": "MISSILE_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
                {"name": "GUIDANCE_METHOD", "identity_fields": ["name"], "identity_scope": "global"},
            ],
            "validation_matrix": [
                {"source": "RADAR_SYSTEM", "relationship": "ASSOCIATED_WITH",
                 "target": "MISSILE_SYSTEM"},
                {"source": "RADAR_SYSTEM", "relationship": "CUES",
                 "target": "MISSILE_SYSTEM"},
                # GUIDANCE_METHOD is not on either side of any ASSOCIATED_WITH/CUES row.
            ],
        }
        all_refs = {
            "E001": SimpleNamespace(
                pass_origin="radar_domain",
                entity_type="RADAR_SYSTEM",
                identity_values={"system_name": "Fan Song"},
                display_label="Fan Song",
            ),
            "E002": SimpleNamespace(
                pass_origin="missile_domain",
                entity_type="MISSILE_SYSTEM",
                identity_values={"system_name": "SA-2"},
                display_label="SA-2",
            ),
            "E003": SimpleNamespace(  # legally valid ref, but wrong type for system_links
                pass_origin="missile_domain",
                entity_type="GUIDANCE_METHOD",
                identity_values={"name": "Command"},
                display_label="Command",
            ),
        }
        pass_def = SimpleNamespace(
            name="system_links",
            depends_on=["radar_domain", "missile_domain"],
            extracted_relationship_types=["ASSOCIATED_WITH", "CUES"],
        )
        selected = _select_upstream_refs_for_pass(pass_def, all_refs, ontology)
        assert set(selected.keys()) == {"E001", "E002"}  # GUIDANCE_METHOD dropped

    def test_deterministic_order_under_shuffled_input(self):
        """Ordering is (pass_origin, entity_type, identity tuple) so
        repeat runs of the same extraction produce the same preamble."""
        all_refs = {
            "E005": self._ref("radar_domain", "RADAR_SYSTEM", "Zebra"),
            "E001": self._ref("radar_domain", "RADAR_SYSTEM", "Alpha"),
            "E003": self._ref("missile_domain", "MISSILE_SYSTEM", "Bravo"),
        }
        pass_def = SimpleNamespace(
            name="system_links",
            depends_on=["radar_domain", "missile_domain"],
            extracted_relationship_types=[],
        )
        selected = _select_upstream_refs_for_pass(pass_def, all_refs, self.ONTOLOGY)
        ordered = list(selected.values())
        # Sorted by (pass_origin, entity_type, identity tuple)
        # → missile_domain/MISSILE_SYSTEM/Bravo, radar_domain/RADAR_SYSTEM/Alpha,
        #   radar_domain/RADAR_SYSTEM/Zebra
        assert [r.display_label for r in ordered] == ["Bravo", "Alpha", "Zebra"]


from app.workers.pipeline import _build_extract_pass_request


class TestBuildExtractPassRequest:

    def test_document_id_included_in_body(self):
        pass_def = SimpleNamespace(name="reference", primary_entity_types=[])
        body = _build_extract_pass_request(
            bundle_key="air_defense_v3",
            pass_def=pass_def,
            doc_json={"stub": True},
            upstream_refs=None,
            document_id="doc-42",
        )
        assert body["document_id"] == "doc-42"

    def test_upstream_entities_carry_all_fields(self):
        ref = SimpleNamespace(
            pass_origin="radar_domain",
            entity_type="RADAR_SYSTEM",
            identity_values={"system_name": "Fan Song"},
            display_label="Fan Song",
        )
        pass_def = SimpleNamespace(name="system_links", primary_entity_types=[])
        body = _build_extract_pass_request(
            bundle_key="air_defense_v3",
            pass_def=pass_def,
            doc_json={"stub": True},
            upstream_refs={"E001": ref},
            document_id="doc-42",
        )
        assert body["upstream_entities"] == [
            {
                "ref_id": "E001",
                "entity_type": "RADAR_SYSTEM",
                "identity_values": {"system_name": "Fan Song"},
                "display_label": "Fan Song",
            },
        ]


def test_build_rejections_by_reason_uses_lowercase_enum_values():
    """RelationshipRejectionReason values are lowercase in the enum
    (extraction_merge.py:35-42). The persisted key MUST match so
    downstream queries don't need to case-normalise."""
    from app.workers.pipeline import _build_rejections_by_reason
    from app.services.extraction_merge import RelationshipRejectionReason

    result = _build_rejections_by_reason([
        (object(), RelationshipRejectionReason.UNKNOWN_REF_ID),
        (object(), RelationshipRejectionReason.UNKNOWN_REF_ID),
        (object(), RelationshipRejectionReason.INVALID_TRIPLE),
    ])
    assert result == {
        "unknown_ref_id": 2,
        "invalid_triple": 1,
    }


def test_build_rejections_by_reason_accepts_pre_merge_tuples():
    """pass_result.pre_merge_rejections shape: (rel, reason)."""
    from app.workers.pipeline import _build_rejections_by_reason
    from app.services.extraction_merge import RelationshipRejectionReason
    result = _build_rejections_by_reason([
        (object(), RelationshipRejectionReason.MISSING_REL_TYPE),
    ])
    assert result == {"missing_rel_type": 1}


def test_build_rejections_by_reason_accepts_merged_rejected_edges_tuples():
    """MergedExtraction.rejected_edges shape: (source_pass, raw_rel, reason)
    (extraction_merge.py:159). The helper must handle this 3-tuple shape
    alongside the 2-tuple pre_merge_rejections shape without a separate
    caller-side conditional."""
    from app.workers.pipeline import _build_rejections_by_reason
    from app.services.extraction_merge import RelationshipRejectionReason
    result = _build_rejections_by_reason([
        ("system_links", object(), RelationshipRejectionReason.UNKNOWN_REF_ID),
        ("system_links", object(), RelationshipRejectionReason.UNKNOWN_REF_ID),
        ("radar_domain", object(), RelationshipRejectionReason.FROM_ENDPOINT_NOT_FOUND),
    ])
    assert result == {
        "unknown_ref_id": 2,
        "from_endpoint_not_found": 1,
    }


def test_build_rejections_by_reason_empty_list_returns_empty_dict():
    from app.workers.pipeline import _build_rejections_by_reason
    assert _build_rejections_by_reason([]) == {}
    assert _build_rejections_by_reason(None) == {}


def test_write_stage_run_persists_metrics_dict_into_jsonb(monkeypatch):
    """When counts includes a 'metrics' key with rejections_by_reason,
    _write_stage_run MUST include that dict in the values dict it passes
    to pg_insert(...).values(...), so it lands in the StageRun.metrics
    JSONB column.

    Asserting against the compiled SQL is brittle — JSONB/UUID rendering
    depends on the dialect and SA version. Instead, intercept the
    Insert statement before execute() and inspect the .values() dict
    directly.
    """
    from app.workers import pipeline as _pipeline
    from sqlalchemy.dialects.postgresql import Insert as _PgInsert

    captured = {}

    # The real _write_stage_run chains:
    #   pg_insert(StageRun).values(**values).on_conflict_do_update(...)
    # We wrap .values() so we can snapshot the dict that was passed in.
    orig_values = _PgInsert.values

    def _spy_values(self, **kwargs):
        captured.setdefault("values_calls", []).append(kwargs)
        return orig_values(self, **kwargs)

    monkeypatch.setattr(_PgInsert, "values", _spy_values)

    class _FakeDB:
        def execute(self, stmt): pass
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass

    monkeypatch.setattr(_pipeline, "_get_db", lambda: _FakeDB())
    _pipeline._write_stage_run(
        pipeline_run_id="00000000-0000-0000-0000-000000000000",
        pass_def=SimpleNamespace(name="system_links"),
        attempt=1,
        execution_status="COMPLETE",
        yield_status="HIT",
        skip_reason=None,
        counts={
            "primary_entities_extracted": 0,
            "relationships_extracted": 1,
            "relationships_rejected": 2,
            "metrics": {"rejections_by_reason": {"unknown_ref_id": 2}},
        },
        error=None,
    )

    # At least one .values() call must carry the metrics dict.
    assert captured["values_calls"], "pg_insert(...).values(...) was never called"
    metrics_values = [
        v["metrics"] for v in captured["values_calls"] if "metrics" in v
    ]
    assert metrics_values, "metrics key was not forwarded into the Insert values dict"
    assert metrics_values[-1] == {"rejections_by_reason": {"unknown_ref_id": 2}}


def test_apply_post_merge_yield_updates_writes_rejections_by_reason_per_pass(monkeypatch):
    """_apply_post_merge_yield_updates is the single post-merge hook that
    touches per-pass StageRun rows (pipeline.py:477+). It must also write
    rejections_by_reason into row.metrics so the per-pass success query
    in Task 6 finds unknown_ref_id counts.

    The helper also reads row.yield_status, row.primary_entities_extracted,
    and row.bridge_entities_extracted (pipeline.py:521-527) when
    recomputing the HIT → DEGRADED transition. Fake rows must supply those
    attributes, otherwise the test fails on AttributeError before it
    checks the new metrics write."""
    from app.workers import pipeline as _pipeline
    from app.services.extraction_merge import (
        MergedExtraction, RelationshipRejectionReason,
    )

    def _fake_row(pass_name, *, metrics=None, yield_status="HIT",
                  primary=0, bridge=0):
        return SimpleNamespace(
            pass_name=pass_name,
            metrics=metrics,
            yield_status=yield_status,
            primary_entities_extracted=primary,
            bridge_entities_extracted=bridge,
            relationships_extracted=0,
            relationships_rejected=0,
        )

    # Fake StageRun rows the post-merge hook should update.
    row_links = _fake_row("system_links", metrics=None, yield_status="HIT",
                          primary=0, bridge=0)
    row_radar = _fake_row("radar_domain", metrics={"preexisting": 1},
                          yield_status="HIT", primary=2, bridge=0)

    # Fake session returning those rows. Matches the `with get_sync_session()
    # as session:` context-manager shape used at pipeline.py:499.
    class _FakeQuery:
        def filter(self, *a, **k): return self
        def all(self): return [row_links, row_radar]
    class _FakeSession:
        def query(self, _cls): return _FakeQuery()
        def commit(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(_pipeline, "get_sync_session", lambda: _FakeSession())

    merged = MergedExtraction(
        entities=[], edges=[],
        rejected_edges=[
            ("system_links", object(), RelationshipRejectionReason.UNKNOWN_REF_ID),
            ("system_links", object(), RelationshipRejectionReason.UNKNOWN_REF_ID),
            ("radar_domain", object(), RelationshipRejectionReason.INVALID_TRIPLE),
        ],
        rejections_by_pass={"system_links": 2, "radar_domain": 1},
        pipeline_run_id="run-1", document_id="doc-1",
    )
    manifest = SimpleNamespace(
        passes=[],
        find_pass=lambda p: SimpleNamespace(
            name=p,
            kind=("relationships_only" if p == "system_links" else "entities_and_relationships"),
        ),
    )
    _pipeline._apply_post_merge_yield_updates("run-1", merged, manifest)

    assert row_links.metrics["rejections_by_reason"] == {"unknown_ref_id": 2}
    # Pre-existing metrics keys survive; rejections_by_reason is added.
    assert row_radar.metrics["preexisting"] == 1
    assert row_radar.metrics["rejections_by_reason"] == {"invalid_triple": 1}
    # Sanity: the existing relationship-count update still fires.
    assert row_links.relationships_rejected == 2
    assert row_radar.relationships_rejected == 1


class TestExtendUpstreamRefsDedupe:
    """After the radar field-group cutover, 5 sub-passes each emit a
    partial RADAR_SYSTEM with system_name='Fan Song'. They must collapse
    to a single upstream ref before the relationship pass sees them."""

    def _pass_def(self, name: str, primary_types):
        return SimpleNamespace(name=name, primary_entity_types=primary_types)

    def test_five_partial_radars_collapse_to_one_upstream_ref(self):
        refs: dict = {}
        for pass_name in (
            "radar_identity", "radar_power_rf", "radar_antenna",
            "radar_timing", "radar_modulation",
        ):
            pass_result = _FakePassResult({
                "RADAR_SYSTEM": [SimpleNamespace(system_name="Fan Song")],
            })
            _extend_upstream_refs(
                refs, pass_result,
                self._pass_def(pass_name, ["RADAR_SYSTEM"]),
                ONTOLOGY,
            )
        # Exactly one ref for Fan Song, regardless of how many sub-passes
        # emitted it.
        fan_song_refs = [
            r for r in refs.values()
            if r.identity_values.get("system_name") == "Fan Song"
        ]
        assert len(fan_song_refs) == 1, (
            f"expected 1 dedup'd ref for Fan Song; got {len(fan_song_refs)}: "
            f"{fan_song_refs!r}"
        )

    def test_dedupe_is_per_identity_not_per_pass(self):
        """Different system_names from different passes must NOT collapse."""
        refs: dict = {}
        _extend_upstream_refs(
            refs,
            _FakePassResult({"RADAR_SYSTEM": [SimpleNamespace(system_name="Fan Song")]}),
            self._pass_def("radar_identity", ["RADAR_SYSTEM"]),
            ONTOLOGY,
        )
        _extend_upstream_refs(
            refs,
            _FakePassResult({"RADAR_SYSTEM": [SimpleNamespace(system_name="Spoon Rest")]}),
            self._pass_def("radar_power_rf", ["RADAR_SYSTEM"]),
            ONTOLOGY,
        )
        names = {r.identity_values.get("system_name") for r in refs.values()}
        assert names == {"Fan Song", "Spoon Rest"}

    def test_dedupe_normalizes_whitespace_and_case(self):
        """Spec §4.5: dedupe by `(entity_type, normalized identity_values)`.

        The same entity emitted with whitespace/case variation across
        sub-passes ("Fan Song" / "  Fan  Song  " / "fan song") must
        collapse to a single ref. Without normalization, the relationship
        pass would receive 3 distinct E### ref-ids for the same entity.
        """
        refs: dict = {}
        for variant in ("Fan Song", "  Fan  Song  ", "fan song"):
            _extend_upstream_refs(
                refs,
                _FakePassResult({"RADAR_SYSTEM": [SimpleNamespace(system_name=variant)]}),
                self._pass_def("radar_identity", ["RADAR_SYSTEM"]),
                ONTOLOGY,
            )
        # Exactly one ref. The retained identity_values may carry either
        # the canonical form or the first-seen form — either is fine, as
        # long as count == 1.
        radar_refs = [
            r for r in refs.values()
            if getattr(r, "entity_type", None) == "RADAR_SYSTEM"
        ]
        assert len(radar_refs) == 1, (
            f"expected 1 dedup'd ref across whitespace/case variants; "
            f"got {len(radar_refs)}: {radar_refs!r}"
        )


# --- Item 2: alias plumbing tests ---------------------------------------

class TestUpstreamAliasCollection:
    """Verify _extend_upstream_refs harvests `nomenclature` and `name` from
    extracted entities into ref.aliases, so the relationship pass can
    match table-cell names back to ref_ids."""

    def _pass_def(self, primary_types):
        return SimpleNamespace(
            name="missile_identity",
            primary_entity_types=primary_types,
        )

    def test_aliases_populated_from_nomenclature_and_name(self):
        refs: dict = {}
        pass_result = _FakePassResult({
            "MISSILE_SYSTEM": [
                SimpleNamespace(
                    system_name="1D",
                    nomenclature="SA-75",
                    name="SA-2A",
                ),
            ],
        })
        _extend_upstream_refs(
            refs, pass_result,
            self._pass_def(["MISSILE_SYSTEM"]),
            ONTOLOGY,
        )
        assert len(refs) == 1
        ref = next(iter(refs.values()))
        assert ref.identity_values == {"system_name": "1D"}
        assert "SA-75" in ref.aliases
        assert "SA-2A" in ref.aliases

    def test_aliases_excludes_identity_and_display_label(self):
        """system_name is the identity — must not also appear in aliases.
        display_label is what _NAME_LIKE_KEYS resolves to; aliases must
        exclude it to avoid duplicates in the prompt preamble."""
        refs: dict = {}
        pass_result = _FakePassResult({
            "MISSILE_SYSTEM": [
                SimpleNamespace(
                    system_name="SA-2",          # also the display_label by default
                    nomenclature="SA-2",         # duplicate of identity — must be filtered
                    name="Guideline",
                ),
            ],
        })
        _extend_upstream_refs(
            refs, pass_result,
            self._pass_def(["MISSILE_SYSTEM"]),
            ONTOLOGY,
        )
        ref = next(iter(refs.values()))
        assert "SA-2" not in ref.aliases  # filtered (matches identity)
        assert ref.aliases == ["Guideline"]

    def test_aliases_empty_when_no_alias_fields_present(self):
        refs: dict = {}
        pass_result = _FakePassResult({
            "MISSILE_SYSTEM": [
                SimpleNamespace(system_name="SA-2"),
            ],
        })
        _extend_upstream_refs(
            refs, pass_result,
            self._pass_def(["MISSILE_SYSTEM"]),
            ONTOLOGY,
        )
        ref = next(iter(refs.values()))
        assert ref.aliases == []

    def test_aliases_dedup_within_one_ref(self):
        """If nomenclature and name happen to be the same string, only one
        alias entry should be emitted."""
        refs: dict = {}
        pass_result = _FakePassResult({
            "MISSILE_SYSTEM": [
                SimpleNamespace(
                    system_name="1D",
                    nomenclature="SA-75",
                    name="SA-75",           # same as nomenclature
                ),
            ],
        })
        _extend_upstream_refs(
            refs, pass_result,
            self._pass_def(["MISSILE_SYSTEM"]),
            ONTOLOGY,
        )
        ref = next(iter(refs.values()))
        assert ref.aliases == ["SA-75"]


def test_build_extract_pass_request_includes_aliases_when_populated():
    """Aliases harvested on the ref must be forwarded in the HTTP payload.
    The relationship pass's name-map relies on them."""
    from app.workers.pipeline import _build_extract_pass_request
    from types import SimpleNamespace
    ref = SimpleNamespace(
        pass_origin="missile_identity",
        entity_type="MISSILE_SYSTEM",
        identity_values={"system_name": "1D"},
        display_label="1D",
        aliases=["SA-75", "SA-2A", "Guideline"],
    )
    pass_def = SimpleNamespace(name="system_links", primary_entity_types=[])
    body = _build_extract_pass_request(
        bundle_key="air_defense_v3",
        pass_def=pass_def,
        doc_json={"stub": True},
        upstream_refs={"E020": ref},
        document_id="doc-42",
    )
    assert body["upstream_entities"] == [
        {
            "ref_id": "E020",
            "entity_type": "MISSILE_SYSTEM",
            "identity_values": {"system_name": "1D"},
            "display_label": "1D",
            "aliases": ["SA-75", "SA-2A", "Guideline"],
        },
    ]


def test_build_extract_pass_request_omits_aliases_when_empty_or_missing():
    """Backward compat: refs with no aliases should not get an `aliases`
    key in the payload at all (older consumers don't need to handle it)."""
    from app.workers.pipeline import _build_extract_pass_request
    from types import SimpleNamespace
    ref_no_attr = SimpleNamespace(
        pass_origin="radar_identity",
        entity_type="RADAR_SYSTEM",
        identity_values={"system_name": "Fan Song"},
        display_label="Fan Song",
    )
    ref_empty = SimpleNamespace(
        pass_origin="radar_identity",
        entity_type="RADAR_SYSTEM",
        identity_values={"system_name": "Spoon Rest"},
        display_label="Spoon Rest",
        aliases=[],
    )
    pass_def = SimpleNamespace(name="system_links", primary_entity_types=[])
    body = _build_extract_pass_request(
        bundle_key="air_defense_v3",
        pass_def=pass_def,
        doc_json={"stub": True},
        upstream_refs={"E001": ref_no_attr, "E002": ref_empty},
        document_id="doc-42",
    )
    for entry in body["upstream_entities"]:
        assert "aliases" not in entry
