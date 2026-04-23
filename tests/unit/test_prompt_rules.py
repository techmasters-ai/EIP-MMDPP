from ontology_bundles._shared.prompt_rules import (
    DELTA_SYSTEM_PROMPT,
    RELATIONSHIPS_ONLY_DELTA_SYSTEM_PROMPT,
    select_delta_system_prompt,
)
from ontology_bundles.air_defense_v3.extraction_schemas.missile_domain import MissileDomainPass
from ontology_bundles.air_defense_v3.extraction_schemas.system_links import SystemLinksPass


def test_delta_system_prompt_scopes_context_and_metadata_as_non_evidence():
    assert 'exactly two top-level keys: "nodes" and "relationships"' in DELTA_SYSTEM_PROMPT
    assert "Extract **only entities and relationships directly evidenced in the current batch document content**." in DELTA_SYSTEM_PROMPT
    assert "current batch document content" in DELTA_SYSTEM_PROMPT
    assert "Document Context" in DELTA_SYSTEM_PROMPT
    assert "classification blocks" in DELTA_SYSTEM_PROMPT
    assert "analyst notes" in DELTA_SYSTEM_PROMPT
    assert "next analytical step" in DELTA_SYSTEM_PROMPT
    assert "When in doubt, omit." in DELTA_SYSTEM_PROMPT
    assert "schema guidance" in DELTA_SYSTEM_PROMPT
    assert "upstream entities" in DELTA_SYSTEM_PROMPT
    assert "do not copy values across semantically different fields" in DELTA_SYSTEM_PROMPT
    assert "do not return an empty relationship list" in DELTA_SYSTEM_PROMPT


def test_delta_system_prompt_is_strict_about_property_evidence():
    assert "Dual requirement: high recall for entities, high precision for properties" in DELTA_SYSTEM_PROMPT
    assert "Apply these two standards simultaneously for **both radar extraction and missile extraction**." in DELTA_SYSTEM_PROMPT
    assert "missile extraction must have the same named-mention recall that radar extraction has" in DELTA_SYSTEM_PROMPT
    assert "do not be looser on missile property hallucination than radar property hallucination" in DELTA_SYSTEM_PROMPT
    assert "named mention can create the node" in DELTA_SYSTEM_PROMPT
    assert "named mention alone cannot populate unsupported technical/admin/spec fields" in DELTA_SYSTEM_PROMPT
    assert "Does the current batch directly state this exact field value" in DELTA_SYSTEM_PROMPT
    assert "Unsupported missile fields must be `null`." in DELTA_SYSTEM_PROMPT
    assert "filling spec-sheet values from domain knowledge" in DELTA_SYSTEM_PROMPT
    assert "inferring confidence or quality scores" in DELTA_SYSTEM_PROMPT
    assert "guidance, seeker, propulsion, or lifecycle fields" in DELTA_SYSTEM_PROMPT
    assert "guidance type, seeker details, DIEQP, nomenclature, range, altitude, speed" in DELTA_SYSTEM_PROMPT
    assert "Sparse and correct is better than rich and hallucinated." in DELTA_SYSTEM_PROMPT


def test_delta_system_prompt_is_symmetric_for_missile_and_radar_recall():
    assert "This rule applies equally to:" in DELTA_SYSTEM_PROMPT
    assert "* radar systems" in DELTA_SYSTEM_PROMPT
    assert "* missile systems" in DELTA_SYSTEM_PROMPT
    assert "proper-noun missile names or missile designations in prose" in DELTA_SYSTEM_PROMPT
    assert "explicit missile name does not support guidance type unless guidance is separately stated" in DELTA_SYSTEM_PROMPT
    assert "place command guidance language into seeker fields" in DELTA_SYSTEM_PROMPT
    assert "missile extraction recall matches radar extraction recall" in DELTA_SYSTEM_PROMPT


def test_relationships_only_prompt_adds_root_and_relationship_node_rules():
    assert 'emit the required root pass node at path `""`' in RELATIONSHIPS_ONLY_DELTA_SYSTEM_PROMPT
    assert 'emit each `relationships[]` record as a node in top-level `"nodes"`' in RELATIONSHIPS_ONLY_DELTA_SYSTEM_PROMPT
    assert 'leave the top-level `"relationships"` array empty' in RELATIONSHIPS_ONLY_DELTA_SYSTEM_PROMPT


def test_select_delta_system_prompt_uses_relationships_only_variant_for_system_links():
    assert select_delta_system_prompt(template_class=SystemLinksPass) == RELATIONSHIPS_ONLY_DELTA_SYSTEM_PROMPT
    assert select_delta_system_prompt(pass_name="system_links") == RELATIONSHIPS_ONLY_DELTA_SYSTEM_PROMPT
    assert select_delta_system_prompt(user_prompt="Schema paths: relationships[] with from_ref_id and to_ref_id") == RELATIONSHIPS_ONLY_DELTA_SYSTEM_PROMPT
    assert select_delta_system_prompt(template_class=MissileDomainPass) == DELTA_SYSTEM_PROMPT
