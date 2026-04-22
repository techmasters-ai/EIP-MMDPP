from ontology_bundles._shared.prompt_rules import DELTA_SYSTEM_PROMPT


def test_delta_system_prompt_scopes_context_and_metadata_as_non_evidence():
    assert 'exactly two top-level keys: "nodes" and "relationships"' in DELTA_SYSTEM_PROMPT
    assert "Extract **only** from the **current batch document content**." in DELTA_SYSTEM_PROMPT
    assert "Document Context" in DELTA_SYSTEM_PROMPT
    assert "classification blocks" in DELTA_SYSTEM_PROMPT
    assert "analyst notes" in DELTA_SYSTEM_PROMPT
    assert "next analytical step" in DELTA_SYSTEM_PROMPT
    assert "When in doubt, omit." in DELTA_SYSTEM_PROMPT
    assert "schema guidance" in DELTA_SYSTEM_PROMPT
    assert "never evidence by themselves" in DELTA_SYSTEM_PROMPT
    assert "do not place slant range into effective intercept range" in DELTA_SYSTEM_PROMPT
    assert "missile-command radar is `FIRE_CONTROL`" in DELTA_SYSTEM_PROMPT
    assert "do **not** return an empty relationship list" in DELTA_SYSTEM_PROMPT
