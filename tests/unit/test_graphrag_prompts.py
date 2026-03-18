"""Unit tests for GraphRAG military ontology prompts."""

import pytest

pytestmark = pytest.mark.unit


class TestCommunityReportPrompt:
    def test_prompt_contains_ontology_layers(self):
        from app.services.graphrag_prompts import get_community_report_prompt

        prompt = get_community_report_prompt()
        assert "LAYER 1" in prompt
        assert "LAYER 2" in prompt
        assert "LAYER 3" in prompt
        assert "LAYER 4" in prompt
        assert "LAYER 5" in prompt

    def test_prompt_contains_key_entity_types(self):
        from app.services.graphrag_prompts import get_community_report_prompt

        prompt = get_community_report_prompt()
        for entity_type in [
            "PLATFORM",
            "RADAR_SYSTEM",
            "MISSILE_SYSTEM",
            "FREQUENCY_BAND",
            "SEEKER",
            "CAPABILITY",
        ]:
            assert entity_type in prompt

    def test_prompt_contains_key_relationships(self):
        from app.services.graphrag_prompts import get_community_report_prompt

        prompt = get_community_report_prompt()
        for rel in [
            "PART_OF",
            "CONTAINS",
            "OPERATES_IN_BAND",
            "CUES",
            "TRACKS",
            "ENGAGES",
            "LAUNCHES",
        ]:
            assert rel in prompt

    def test_prompt_contains_scoring_weights(self):
        from app.services.graphrag_prompts import get_community_report_prompt

        prompt = get_community_report_prompt()
        assert "0.95" in prompt
        assert "0.90" in prompt
        assert "0.85" in prompt


class TestSearchPrompts:
    def test_local_search_prompt_exists(self):
        from app.services.graphrag_prompts import get_local_search_prompt

        prompt = get_local_search_prompt()
        assert len(prompt) > 100
        assert "military" in prompt.lower()

    def test_global_search_map_prompt_exists(self):
        from app.services.graphrag_prompts import get_global_search_map_prompt

        prompt = get_global_search_map_prompt()
        assert len(prompt) > 100

    def test_global_search_reduce_prompt_exists(self):
        from app.services.graphrag_prompts import get_global_search_reduce_prompt

        prompt = get_global_search_reduce_prompt()
        assert len(prompt) > 100

    def test_drift_search_prompt_exists(self):
        from app.services.graphrag_prompts import get_drift_search_prompt

        prompt = get_drift_search_prompt()
        assert len(prompt) > 100

    def test_basic_search_prompt_exists(self):
        from app.services.graphrag_prompts import get_basic_search_prompt

        prompt = get_basic_search_prompt()
        assert len(prompt) > 100


class TestWritePromptFiles:
    def test_writes_all_prompt_files(self, tmp_path):
        from app.services.graphrag_prompts import write_prompt_files

        write_prompt_files(tmp_path)
        expected_files = [
            "community_report.txt",
            "local_search_system_prompt.txt",
            "global_search_map_system_prompt.txt",
            "global_search_reduce_system_prompt.txt",
            "drift_search_system_prompt.txt",
            "basic_search_system_prompt.txt",
        ]
        for fname in expected_files:
            assert (tmp_path / fname).exists(), f"Missing prompt file: {fname}"
            content = (tmp_path / fname).read_text()
            assert len(content) > 50, f"Prompt file too short: {fname}"
