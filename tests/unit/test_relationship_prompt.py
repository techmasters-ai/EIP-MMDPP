"""Unit tests for relationship extraction prompt content."""
import importlib.util
from pathlib import Path
import pytest

# docling-graph lives in a separate container; use importlib to avoid
# collision with the main app package on sys.path.
_prompts_path = Path(__file__).resolve().parents[2] / "docker" / "docling-graph" / "app" / "prompts.py"
_spec = importlib.util.spec_from_file_location("docling_graph_prompts", _prompts_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
get_relationship_prompt = _mod.get_relationship_prompt

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
