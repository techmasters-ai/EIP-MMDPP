from ontology_bundles._shared.prompt_rules import DELTA_SYSTEM_PROMPT


def test_delta_system_prompt_scopes_context_and_metadata_as_non_evidence():
    assert "DOCUMENT CONTEXT block" in DELTA_SYSTEM_PROMPT
    assert "Extract ONLY from the current batch's direct document content" in DELTA_SYSTEM_PROMPT
    assert "Classification" in DELTA_SYSTEM_PROMPT
    assert "Extracted Technical Takeaways" in DELTA_SYSTEM_PROMPT
    assert "Analyst Notes" in DELTA_SYSTEM_PROMPT
    assert "next analytical step" in DELTA_SYSTEM_PROMPT
    assert "Empty output for a domain pass is correct" in DELTA_SYSTEM_PROMPT
