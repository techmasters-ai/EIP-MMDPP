"""Post-processing cleanup for Docling-generated markdown.

Removes VLM hallucination artifacts: repeated lines, watermark spam,
bare image placeholders, and excessive whitespace.
"""

from __future__ import annotations

import re
from collections import Counter


def clean_markdown(text: str) -> str:
    """Apply all cleanup rules to a markdown string."""
    text = _collapse_consecutive_duplicates(text)
    text = _strip_spam_lines(text)
    text = _remove_bare_image_comments(text)
    text = _collapse_blank_lines(text)
    text = _strip_trailing_whitespace(text)
    return text


def _collapse_consecutive_duplicates(text: str) -> str:
    """Collapse 3+ consecutive identical lines into one."""
    lines = text.split("\n")
    result: list[str] = []
    prev = None
    count = 0
    for line in lines:
        stripped = line.strip()
        if stripped == prev:
            count += 1
            if count < 3:
                result.append(line)
        else:
            prev = stripped
            count = 1
            result.append(line)
    return "\n".join(result)


def _strip_spam_lines(text: str) -> str:
    """Remove lines that appear 5+ times in the document (keep first occurrence)."""
    lines = text.split("\n")
    stripped_counts = Counter(line.strip() for line in lines if line.strip())

    spam_lines = {line for line, count in stripped_counts.items() if count >= 5}
    # Don't strip common markdown patterns
    spam_lines -= {"", "---", "***", "___", "|", "<!-- image -->"}

    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in spam_lines:
            if stripped not in seen:
                seen.add(stripped)
                result.append(line)
            # else: skip duplicate spam line
        else:
            result.append(line)
    return "\n".join(result)


def _remove_bare_image_comments(text: str) -> str:
    """Remove standalone <!-- image --> comment lines.

    These are Docling placeholders that will be replaced by actual
    image references in the API endpoint.
    """
    return re.sub(r"^\s*<!-- image -->\s*$", "", text, flags=re.MULTILINE)


def _collapse_blank_lines(text: str) -> str:
    """Collapse 3+ consecutive blank lines into 2."""
    return re.sub(r"\n{4,}", "\n\n\n", text)


def _strip_trailing_whitespace(text: str) -> str:
    """Strip trailing whitespace from each line."""
    return "\n".join(line.rstrip() for line in text.split("\n"))
