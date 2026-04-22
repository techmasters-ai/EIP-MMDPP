from ontology_bundles._shared.prompt_rules import DELTA_SYSTEM_PROMPT


def test_delta_system_prompt_scopes_context_and_metadata_as_non_evidence():
    assert 'exactly two top-level keys: "nodes" and "relationships"' in DELTA_SYSTEM_PROMPT
    assert "Extract **only radar-domain entities and relationships" in DELTA_SYSTEM_PROMPT
    assert "current batch document content" in DELTA_SYSTEM_PROMPT
    assert "Document Context" in DELTA_SYSTEM_PROMPT
    assert "classification blocks" in DELTA_SYSTEM_PROMPT
    assert "analyst notes" in DELTA_SYSTEM_PROMPT
    assert "next analytical step" in DELTA_SYSTEM_PROMPT
    assert "When in doubt, omit." in DELTA_SYSTEM_PROMPT
    assert "schema guidance" in DELTA_SYSTEM_PROMPT
    assert "are **never evidence** by themselves." in DELTA_SYSTEM_PROMPT
    assert "do not move values across semantically different fields" in DELTA_SYSTEM_PROMPT
    assert "do **not** return an empty relationship list" in DELTA_SYSTEM_PROMPT


def test_delta_system_prompt_is_strict_about_property_evidence():
    assert "Non-Inference Rule (Highest Priority)" in DELTA_SYSTEM_PROMPT
    assert "Every emitted property value must be directly supported by the current batch document." in DELTA_SYSTEM_PROMPT
    assert "fill missing specs from general domain knowledge" in DELTA_SYSTEM_PROMPT
    assert "infer confidence scores" in DELTA_SYSTEM_PROMPT
    assert "A sparse output is preferable to an enriched but unsupported output." in DELTA_SYSTEM_PROMPT
    assert "every emitted property value is directly evidenced by the current batch document" in DELTA_SYSTEM_PROMPT
