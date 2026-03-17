"""Tests for group-specific LLM prompts."""

from __future__ import annotations

import pytest

from app.prompts import GROUP_PROMPTS, get_entity_prompt, get_relationship_prompt
from app.templates import GROUP_MAP


class TestGroupPromptCoverage:
    """Every ontology group must have a corresponding prompt."""

    def test_every_group_has_prompt(self) -> None:
        for group_name in GROUP_MAP:
            assert group_name in GROUP_PROMPTS, (
                f"GROUP_MAP key {group_name!r} has no entry in GROUP_PROMPTS"
            )


class TestGetEntityPrompt:
    def test_entity_prompt_returns_string(self) -> None:
        result = get_entity_prompt("equipment")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_entity_prompt_unknown_group_raises(self) -> None:
        with pytest.raises(KeyError):
            get_entity_prompt("nonexistent_group")


class TestGetRelationshipPrompt:
    def test_relationship_prompt_includes_entity_context(self) -> None:
        entities = [
            {"name": "SA-20", "entity_type": "MISSILE_SYSTEM"},
            {"name": "S-band", "entity_type": "FREQUENCY_BAND"},
        ]
        result = get_relationship_prompt(entities)
        assert "SA-20" in result
        assert "MISSILE_SYSTEM" in result
        assert "S-band" in result
        assert "FREQUENCY_BAND" in result

    def test_relationship_prompt_empty_entities(self) -> None:
        result = get_relationship_prompt([])
        assert isinstance(result, str)
        assert "no entities extracted" in result
