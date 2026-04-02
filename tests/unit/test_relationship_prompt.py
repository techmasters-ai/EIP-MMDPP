"""Unit tests for relationship extraction prompt content."""
import sys
from pathlib import Path
import pytest

# docling-graph lives in a separate container; add its source to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docker" / "docling-graph"))
from app.prompts import get_relationship_prompt

pytestmark = pytest.mark.unit


class TestRelationshipPrompt:
    def test_prompt_contains_specification_linking_instruction(self):
        """Prompt must explicitly instruct linking SPECIFICATION to systems."""
        entities = [
            {"name": "SA-2 Guideline", "entity_type": "MISSILE_SYSTEM"},
            {"name": "Maximum missile range", "entity_type": "SPECIFICATION"},
        ]
        prompt = get_relationship_prompt(entities, "")
        lower = prompt.lower()
        assert "specification" in lower
        assert "specified_by" in lower
        # Must instruct to connect specs to their parent system
        assert "parent" in lower or "belongs to" in lower or "connect each" in lower

    def test_user_prompt_contains_specification_instruction(self):
        """The fallback prompt must mention SPECIFIED_BY."""
        prompt = get_relationship_prompt([], "")
        assert "SPECIFIED_BY" in prompt
