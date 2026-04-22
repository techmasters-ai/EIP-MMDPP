from ontology_bundles.air_defense_v3.extraction_schemas import missile_domain, radar_domain


def test_radar_domain_drops_cross_domain_entities_and_normalizes_null_text():
    model = radar_domain.RadarDomainPass.model_validate({
        "radar_systems": [
            {"system_name": " Fan Song ", "nomenclature": "None", "scan_type": " N/A "},
            {"system_name": "SA-2", "nomenclature": "V-750VK Dvina"},
            {"system_name": "U-2"},
        ]
    })

    assert [entity.system_name for entity in model.radar_systems] == ["Fan Song"]
    assert model.radar_systems[0].nomenclature is None
    assert model.radar_systems[0].scan_type is None


def test_missile_domain_drops_radars_and_targets_and_normalizes_null_text():
    model = missile_domain.MissileDomainPass.model_validate({
        "missile_systems": [
            {"system_name": " SA-2 ", "nomenclature": "V-750VK Dvina", "guidance_type": "None"},
            {"system_name": "Fan Song"},
            {"system_name": "U-2"},
        ]
    })

    assert [entity.system_name for entity in model.missile_systems] == ["SA-2"]
    assert model.missile_systems[0].nomenclature == "V-750VK Dvina"
    assert model.missile_systems[0].guidance_type is None


def test_radar_domain_prompt_examples_do_not_include_missile_names():
    field = radar_domain.RadarDomainPass.model_fields["radar_systems"]
    description = field.description or ""
    examples = str(field.examples or "")

    assert "e.g. 'SA-2'" not in description
    assert "SA-2 radar" not in examples
