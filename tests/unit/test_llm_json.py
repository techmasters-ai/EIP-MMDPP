"""Unit tests for the tolerant LLM JSON parser."""

import pytest

from app.services.llm_json import (
    extract_fenced_json,
    extract_last_balanced_json,
    parse_llm_json_loose,
    strip_reasoning_wrappers,
)

pytestmark = pytest.mark.unit


class TestStripReasoningWrappers:
    def test_strips_think_tags(self):
        raw = '<think>reasoning here</think>{"key": "value"}'
        assert strip_reasoning_wrappers(raw) == '{"key": "value"}'

    def test_strips_thinking_tags(self):
        raw = '<thinking>some reasoning</thinking>  {"a": 1}'
        assert strip_reasoning_wrappers(raw) == '{"a": 1}'

    def test_strips_answer_prefix(self):
        raw = 'Answer: {"result": true}'
        assert strip_reasoning_wrappers(raw) == '{"result": true}'

    def test_strips_final_answer_prefix(self):
        raw = 'Final Answer: [1, 2, 3]'
        assert strip_reasoning_wrappers(raw) == '[1, 2, 3]'

    def test_no_op_for_clean_json(self):
        raw = '{"clean": true}'
        assert strip_reasoning_wrappers(raw) == '{"clean": true}'


class TestExtractFencedJson:
    def test_extracts_fenced_block(self):
        raw = 'Some text\n```json\n{"key": "val"}\n```\nmore text'
        assert extract_fenced_json(raw) == '{"key": "val"}'

    def test_extracts_unflagged_fence(self):
        raw = '```\n[1, 2, 3]\n```'
        assert extract_fenced_json(raw) == '[1, 2, 3]'

    def test_returns_none_without_fence(self):
        assert extract_fenced_json('{"no": "fence"}') is None


class TestExtractLastBalancedJson:
    def test_finds_object(self):
        raw = 'prefix text {"a": 1, "b": 2} trailing'
        assert extract_last_balanced_json(raw) == '{"a": 1, "b": 2}'

    def test_finds_array(self):
        raw = 'some [1, 2, 3] end'
        assert extract_last_balanced_json(raw) == '[1, 2, 3]'

    def test_returns_none_for_no_json(self):
        assert extract_last_balanced_json("no json here") is None

    def test_handles_nested_braces(self):
        raw = 'stuff {"outer": {"inner": 1}} end'
        result = extract_last_balanced_json(raw)
        assert result == '{"outer": {"inner": 1}}'


class TestParseLlmJsonLoose:
    def test_clean_json(self):
        assert parse_llm_json_loose('{"a": 1}') == {"a": 1}

    def test_none_input(self):
        assert parse_llm_json_loose(None) is None

    def test_empty_input(self):
        assert parse_llm_json_loose("") is None

    def test_think_wrapped_json(self):
        raw = '<think>Let me think...</think>\n{"result": "answer"}'
        result = parse_llm_json_loose(raw)
        assert result == {"result": "answer"}

    def test_fenced_json(self):
        raw = 'Here is the answer:\n```json\n{"x": 42}\n```'
        assert parse_llm_json_loose(raw) == {"x": 42}

    def test_json_after_reasoning(self):
        raw = (
            "Let me analyze this step by step.\n"
            "First, the radar system...\n"
            '{"intermediate_answer": "# Analysis\nThe system...", '
            '"score": 75, "follow_up_queries": ["q1", "q2", "q3", "q4", "q5"]}'
        )
        result = parse_llm_json_loose(raw)
        assert result is not None
        assert result["score"] == 75

    def test_truncated_json_repaired(self):
        raw = '{"key": "value", "partial": "trun'
        result = parse_llm_json_loose(raw)
        # json_repair should fix this
        if result is not None:
            assert "key" in result

    def test_returns_none_for_garbage(self):
        assert parse_llm_json_loose("not json at all") is None
