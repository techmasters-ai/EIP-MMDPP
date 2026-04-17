"""Phase 8 Task 52a — provenance aggregation into MergedEntityRecord.

merge_and_resolve, after its entity-merge loop, walks each pass's
``PassResult.provenance`` rows and attaches them to the matching
``MergedEntityRecord.provenance``. Dedup on ``(instance_id, element_uid)``
pairs, not on identity. Malformed rows + post-merge-absent rows are
dropped with distinct WARNING messages.
"""
from __future__ import annotations

import logging

import pytest

from app.services.extraction_merge import (
    ExtractionMetadata,
    ExtractionProvenance,
    PassResult,
    merge_and_resolve,
    _build_logical_identity,
    logical_identity_from_dict,
)
from ontology_bundles.air_defense_v3.entities import AntennaEntity


DOC_ID = "doc-test-1"
RUN_ID = "run-1"


def _radar_ontology():
    """Minimal ontology dict with ANTENNA (for the identity-helper tests)."""
    return {
        "entity_types": [
            {
                "name": "ANTENNA",
                "identity_fields": ["name"],
                "identity_scope": "document",
                "properties": ["name", "gain_dbi"],
            },
        ],
        "relationship_types": [],
    }


def _pass_result_with_entities_and_provenance(pass_name, entity_names, provenance_rows):
    """Build a PassResult whose template_instance emits AntennaEntity
    instances named from entity_names, attached to a SimpleNamespace
    pass-root that exposes them under the ``antennas`` attribute."""
    from types import SimpleNamespace
    antennas = [AntennaEntity(name=n) for n in entity_names]
    template = SimpleNamespace(antennas=antennas, relationships=[])
    return PassResult(
        pass_name=pass_name,
        template_instance=template,
        metadata=ExtractionMetadata(
            schema_size_chars=0, structured_output_mode="strict",
        ),
        pre_merge_rejections=[],
        provenance=provenance_rows,
    )


def _ontology_with_antenna_fixture():
    ont = _radar_ontology()
    # The merge loop iterates entity_types from ontology and calls
    # pass_result.iter_entities_of_type("ANTENNA"); the fallback path
    # looks for "antennas" list on the template. Our SimpleNamespace
    # exposes antennas so iter_entities_of_type finds them.
    return ont


def _prov(**kwargs):
    base = {
        "instance_id": "uuid-x",
        "ontology_name": "ANTENNA",
        "identity_values": {"name": "FF-1"},
        "element_uid": "#/texts/0",
    }
    base.update(kwargs)
    return ExtractionProvenance(**base)


# ---------------------------------------------------------------------------
# (a) Two passes, two distinct instance_ids, same logical identity → 2 rows
# ---------------------------------------------------------------------------


def test_two_passes_with_distinct_instance_ids_aggregate_to_two_provenance_rows():
    pass_a = _pass_result_with_entities_and_provenance(
        "radar_domain", ["FF-1"],
        [_prov(instance_id="uuid-a", element_uid="#/texts/0")],
    )
    pass_b = _pass_result_with_entities_and_provenance(
        "system_links", ["FF-1"],
        [_prov(instance_id="uuid-b", element_uid="#/texts/5")],
    )

    merged = merge_and_resolve(
        {"radar_domain": pass_a, "system_links": pass_b},
        manifest=None,
        ontology=_ontology_with_antenna_fixture(),
        document_id=DOC_ID,
        pipeline_run_id=RUN_ID,
    )

    assert len(merged.entities) == 1
    record = merged.entities[0]
    assert len(record.provenance) == 2
    instance_ids = {p.instance_id for p in record.provenance}
    assert instance_ids == {"uuid-a", "uuid-b"}


# ---------------------------------------------------------------------------
# (b) Same (instance_id, element_uid) duplicate → collapses to 1
# ---------------------------------------------------------------------------


def test_duplicate_instance_id_and_element_uid_collapses_to_one_row():
    pass_a = _pass_result_with_entities_and_provenance(
        "radar_domain", ["FF-1"],
        [
            _prov(instance_id="uuid-a", element_uid="#/texts/0"),
            _prov(instance_id="uuid-a", element_uid="#/texts/0"),  # echo
        ],
    )

    merged = merge_and_resolve(
        {"radar_domain": pass_a},
        manifest=None,
        ontology=_ontology_with_antenna_fixture(),
        document_id=DOC_ID,
        pipeline_run_id=RUN_ID,
    )
    record = merged.entities[0]
    assert len(record.provenance) == 1


# ---------------------------------------------------------------------------
# (b2) Same instance_id + DIFFERENT element_uids → both retained
# ---------------------------------------------------------------------------


def test_same_instance_id_with_different_element_uids_retains_both_rows():
    pass_a = _pass_result_with_entities_and_provenance(
        "radar_domain", ["FF-1"],
        [
            _prov(instance_id="uuid-a", element_uid="#/texts/0"),
            _prov(instance_id="uuid-a", element_uid="#/texts/7"),  # different elem
        ],
    )

    merged = merge_and_resolve(
        {"radar_domain": pass_a},
        manifest=None,
        ontology=_ontology_with_antenna_fixture(),
        document_id=DOC_ID,
        pipeline_run_id=RUN_ID,
    )
    record = merged.entities[0]
    assert len(record.provenance) == 2
    elem_uids = {p.element_uid for p in record.provenance}
    assert elem_uids == {"#/texts/0", "#/texts/7"}


# ---------------------------------------------------------------------------
# (d) Identity-helper contract — _build_logical_identity ==
#     logical_identity_from_dict for equivalent inputs
# ---------------------------------------------------------------------------


def test_logical_identity_cross_caller_parity():
    ontology = _ontology_with_antenna_fixture()
    antenna = AntennaEntity(name="FF-1")
    from_instance = _build_logical_identity("ANTENNA", antenna, ontology, DOC_ID)
    from_dict = logical_identity_from_dict("ANTENNA", {"name": "FF-1"}, ontology, DOC_ID)
    # Equal via dataclass __eq__
    assert from_instance == from_dict


# ---------------------------------------------------------------------------
# (e) Document-scoped identity_scope — document_id participates in equality
# ---------------------------------------------------------------------------


def test_document_scoped_identity_carries_document_id_and_aggregates():
    ontology = _ontology_with_antenna_fixture()
    # ANTENNA already identity_scope="document" in _radar_ontology
    pass_a = _pass_result_with_entities_and_provenance(
        "radar_domain", ["FF-1"],
        [_prov(instance_id="uuid-a", element_uid="#/texts/0")],
    )
    merged = merge_and_resolve(
        {"radar_domain": pass_a},
        manifest=None,
        ontology=ontology,
        document_id=DOC_ID,
        pipeline_run_id=RUN_ID,
    )
    record = merged.entities[0]
    assert record.identity.document_id == DOC_ID
    assert record.identity.scope == "document"
    assert len(record.provenance) == 1


# ---------------------------------------------------------------------------
# (f) Malformed payload — identity_from_dict returns None → drop with WARNING
# ---------------------------------------------------------------------------


def test_malformed_provenance_payload_is_dropped_with_warning(caplog):
    pass_a = _pass_result_with_entities_and_provenance(
        "radar_domain", ["FF-1"],
        [
            _prov(instance_id="uuid-a", element_uid="#/texts/0"),  # good
            _prov(
                instance_id="uuid-malformed",
                ontology_name="ANTENNA",
                identity_values={"wrong_key": "FF-1"},  # missing "name"
                element_uid="#/texts/3",
            ),
        ],
    )

    with caplog.at_level(logging.WARNING, logger="app.services.extraction_merge"):
        merged = merge_and_resolve(
            {"radar_domain": pass_a},
            manifest=None,
            ontology=_ontology_with_antenna_fixture(),
            document_id=DOC_ID,
            pipeline_run_id=RUN_ID,
        )

    record = merged.entities[0]
    assert len(record.provenance) == 1
    assert record.provenance[0].instance_id == "uuid-a"
    assert any(
        "malformed provenance" in r.message and "uuid-malformed" in r.message
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# (g) Unknown entity_type — identity_from_dict returns None → drop with WARNING
# ---------------------------------------------------------------------------


def test_unknown_entity_type_in_provenance_is_dropped_with_warning(caplog):
    pass_a = _pass_result_with_entities_and_provenance(
        "radar_domain", ["FF-1"],
        [
            _prov(instance_id="uuid-a", element_uid="#/texts/0"),  # good
            _prov(
                instance_id="uuid-unknown",
                ontology_name="FABRICATED_TYPE",
                identity_values={"name": "FF-1"},
                element_uid="#/texts/4",
            ),
        ],
    )

    with caplog.at_level(logging.WARNING, logger="app.services.extraction_merge"):
        merged = merge_and_resolve(
            {"radar_domain": pass_a},
            manifest=None,
            ontology=_ontology_with_antenna_fixture(),
            document_id=DOC_ID,
            pipeline_run_id=RUN_ID,
        )

    record = merged.entities[0]
    assert len(record.provenance) == 1
    # Warning tied to the unknown type payload
    assert any(
        "malformed provenance" in r.message and "uuid-unknown" in r.message
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Post-merge-absent: identity valid but no merged record — drop with distinct msg
# ---------------------------------------------------------------------------


def test_provenance_for_post_merge_absent_identity_is_dropped_with_warning(caplog):
    """Provenance references a legal ANTENNA identity, but no pass emitted
    an entity with that name — so no MergedEntityRecord exists to attach
    to. Dropped with a distinct WARNING ("post-merge-absent" vs
    "malformed")."""
    pass_a = _pass_result_with_entities_and_provenance(
        "radar_domain", ["FF-1"],
        [
            _prov(instance_id="uuid-good", element_uid="#/texts/0"),
            _prov(
                instance_id="uuid-absent",
                identity_values={"name": "NOT-EMITTED"},
                element_uid="#/texts/9",
            ),
        ],
    )

    with caplog.at_level(logging.WARNING, logger="app.services.extraction_merge"):
        merged = merge_and_resolve(
            {"radar_domain": pass_a},
            manifest=None,
            ontology=_ontology_with_antenna_fixture(),
            document_id=DOC_ID,
            pipeline_run_id=RUN_ID,
        )

    record = merged.entities[0]
    assert len(record.provenance) == 1
    assert record.provenance[0].instance_id == "uuid-good"
    assert any(
        "post-merge-absent" in r.message and "uuid-absent" in r.message
        for r in caplog.records
    )
