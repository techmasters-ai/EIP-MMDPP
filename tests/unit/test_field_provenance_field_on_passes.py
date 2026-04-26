"""Phase 3 task 28 — RadarDomainPass / MissileDomainPass each carry a
top-level field_provenance: list[FieldProvenanceRow]."""


def test_radar_domain_pass_has_field_provenance():
    from ontology_bundles.air_defense_v3.extraction_schemas.radar_domain import (
        RadarDomainPass,
    )
    assert "field_provenance" in RadarDomainPass.model_fields
    inst = RadarDomainPass()
    assert inst.field_provenance == []


def test_missile_domain_pass_has_field_provenance():
    from ontology_bundles.air_defense_v3.extraction_schemas.missile_domain import (
        MissileDomainPass,
    )
    inst = MissileDomainPass()
    assert inst.field_provenance == []
