"""bge-m3 tokenizer wrapper — lazy load, module-level cache.

The chunker in pipeline.py:5524-5525 uses bge-m3 with max_tokens=512.
We use the same tokenizer for size measurements so render-side budgeting
matches chunker-side budgeting."""
from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

_TOKENIZER_NAME = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def _tokenizer():
    """Lazy-load + cache the HF tokenizer."""
    from transformers import AutoTokenizer
    logger.info("table_normalization.tokens: loading tokenizer %s (first call only)", _TOKENIZER_NAME)
    return AutoTokenizer.from_pretrained(_TOKENIZER_NAME)


def count_bge_m3_tokens(text: str) -> int:
    """Return the bge-m3 token count of `text`. Empty string → 0."""
    if not text:
        return 0
    tok = _tokenizer()
    return len(tok.encode(text, add_special_tokens=False))
