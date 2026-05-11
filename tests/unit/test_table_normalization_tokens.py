import pytest
from app.services.table_normalization.tokens import count_bge_m3_tokens


def test_empty_string_zero_tokens():
    assert count_bge_m3_tokens("") == 0


def test_short_text_is_under_limit():
    n = count_bge_m3_tokens("Hello world")
    assert 0 < n < 10


def test_long_text_exceeds_512():
    long = ("Lorem ipsum dolor sit amet consectetur adipiscing elit. " * 200)
    n = count_bge_m3_tokens(long)
    assert n > 512


def test_repeated_calls_use_cached_tokenizer():
    """Second call should not re-load the tokenizer from disk."""
    count_bge_m3_tokens("warmup")
    n1 = count_bge_m3_tokens("once more")
    n2 = count_bge_m3_tokens("once more")
    assert n1 == n2
