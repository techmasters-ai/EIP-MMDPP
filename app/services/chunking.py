"""Structure-aware document chunking.

Produces text chunks that respect document structure:
  - Never split tables or equations mid-chunk
  - Keep headings with following content
  - Break at paragraph boundaries
  - Preserve section_path metadata
  - Fallback to word-based splitting for oversized elements
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHUNK_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 64


@dataclass
class StructuredChunk:
    """A structure-aware text chunk with document provenance."""
    text: str
    chunk_index: int
    modality: str  # text | table | heading | equation | ocr
    page_number: Optional[int] = None
    section_path: Optional[str] = None
    element_uids: list[str] = field(default_factory=list)
    heading_text: Optional[str] = None


def structure_aware_chunk(
    elements: list[dict],
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[StructuredChunk]:
    """Chunk document elements respecting structure.

    Args:
        elements: List of element dicts from Docling conversion.
            Expected keys: element_type, content_text, page_number,
            section_path, element_uid, element_order, heading_level.
        max_chunk_tokens: Approximate max tokens per chunk.
        overlap_tokens: Token overlap between consecutive text chunks.

    Returns:
        List of StructuredChunk instances.
    """
    if not elements:
        return []

    # Sort by element_order to preserve document flow
    sorted_elements = sorted(elements, key=lambda e: e.get("element_order", 0))

    chunks: list[StructuredChunk] = []
    chunk_index = 0

    # Buffer for accumulating text paragraphs
    buffer_texts: list[str] = []
    buffer_uids: list[str] = []
    buffer_page: Optional[int] = None
    buffer_section: Optional[str] = None
    buffer_heading: Optional[str] = None

    def _flush_buffer():
        nonlocal chunk_index, buffer_texts, buffer_uids, buffer_page, buffer_section, buffer_heading
        if not buffer_texts:
            return
        combined = "\n\n".join(buffer_texts)
        # Split into smaller chunks if oversized
        sub_chunks = _split_text(combined, max_chunk_tokens, overlap_tokens)
        for sc in sub_chunks:
            chunks.append(StructuredChunk(
                text=sc,
                chunk_index=chunk_index,
                modality="text",
                page_number=buffer_page,
                section_path=buffer_section,
                element_uids=list(buffer_uids),
                heading_text=buffer_heading,
            ))
            chunk_index += 1
        buffer_texts = []
        buffer_uids = []
        buffer_page = None
        buffer_section = None
        buffer_heading = None

    for elem in sorted_elements:
        etype = elem.get("element_type", "text")
        content = (elem.get("content_text") or "").strip()
        if not content:
            continue

        page = elem.get("page_number")
        section = elem.get("section_path")
        uid = elem.get("element_uid", "")

        if etype == "heading":
            # Flush current buffer before starting new section
            _flush_buffer()
            buffer_heading = content
            buffer_section = section
            buffer_page = page
            # Add heading text to buffer so it stays with following content
            buffer_texts.append(content)
            buffer_uids.append(uid)

        elif etype == "table":
            # Tables are always their own chunk — never split
            current_heading = buffer_heading  # save before flush clears it
            _flush_buffer()
            chunks.append(StructuredChunk(
                text=content,
                chunk_index=chunk_index,
                modality="table",
                page_number=page,
                section_path=section,
                element_uids=[uid],
                heading_text=current_heading,
            ))
            chunk_index += 1

        elif etype == "equation":
            # Equations are their own chunk
            current_heading = buffer_heading
            _flush_buffer()
            chunks.append(StructuredChunk(
                text=content,
                chunk_index=chunk_index,
                modality="equation",
                page_number=page,
                section_path=section,
                element_uids=[uid],
                heading_text=current_heading,
            ))
            chunk_index += 1

        elif etype == "image":
            # Image captions get their own chunk (image embeddings handled separately)
            if content:
                current_heading = buffer_heading
                _flush_buffer()
                chunks.append(StructuredChunk(
                    text=content,
                    chunk_index=chunk_index,
                    modality="image",
                    page_number=page,
                    section_path=section,
                    element_uids=[uid],
                    heading_text=current_heading,
                ))
                chunk_index += 1

        else:
            # Regular text — accumulate in buffer
            # Check if we'd exceed the max chunk size
            current_len = _approx_tokens("\n\n".join(buffer_texts)) if buffer_texts else 0
            new_len = _approx_tokens(content)

            if current_len + new_len > max_chunk_tokens and buffer_texts:
                _flush_buffer()
                # Carry forward section context
                buffer_section = section
                buffer_page = page

            buffer_texts.append(content)
            buffer_uids.append(uid)
            if buffer_page is None:
                buffer_page = page
            if buffer_section is None:
                buffer_section = section

    # Flush remaining buffer
    _flush_buffer()

    return chunks


def _approx_tokens(text: str) -> int:
    """Approximate token count using word splitting (~1.3 tokens per word)."""
    return int(len(text.split()) * 1.3)


def _split_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text into chunks by paragraph boundaries, then by words if needed."""
    if _approx_tokens(text) <= max_tokens:
        return [text]

    # Try splitting by paragraphs first
    paragraphs = text.split("\n\n")
    if len(paragraphs) > 1:
        chunks = []
        current = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = _approx_tokens(para)
            if current_tokens + para_tokens > max_tokens and current:
                chunks.append("\n\n".join(current))
                # Overlap: keep last paragraph
                if overlap_tokens > 0 and current:
                    overlap_text = current[-1]
                    if _approx_tokens(overlap_text) <= overlap_tokens:
                        current = [overlap_text]
                        current_tokens = _approx_tokens(overlap_text)
                    else:
                        current = []
                        current_tokens = 0
                else:
                    current = []
                    current_tokens = 0

            current.append(para)
            current_tokens += para_tokens

        if current:
            chunks.append("\n\n".join(current))
        return chunks

    # Fallback: word-based splitting
    words = text.split()
    chunks = []
    max_words = max(int(max_tokens / 1.3), 1)
    overlap_words = max(int(overlap_tokens / 1.3), 0)

    # Ensure forward progress even when overlap is very large.
    if overlap_words >= max_words:
        overlap_words = max_words - 1

    i = 0
    while i < len(words):
        end = min(i + max_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end >= len(words):
            break

        next_i = end - overlap_words if overlap_words > 0 else end
        if next_i <= i:
            next_i = i + 1
        i = next_i

    return chunks


# ---------------------------------------------------------------------------
# Image description section splitting
# ---------------------------------------------------------------------------

# Section header patterns for image description splitting
_SECTION_HEADER_PATTERNS = [
    re.compile(r'^#{1,4}\s+.+', re.MULTILINE),           # Markdown: # / ## / ### / ####
    re.compile(r'^\d{1,2}\)\s+.+', re.MULTILINE),        # Numbered: 1) / 2)
    re.compile(r'^\d{1,2}\.\s+.+', re.MULTILINE),        # Numbered: 1. / 2.
    re.compile(r'^\*\*[^*]+[:\.]\*\*', re.MULTILINE),    # Bold: **Title:** / **Title.**
]

_MIN_SECTION_LENGTH = 10


def split_description_sections(description: str) -> list[str]:
    """Split an image description into sections by headers.

    Handles markdown headers (# / ## / ###), numbered headers (1) / 1.),
    and bold headers (**Title:**). Falls back to paragraph splitting.
    Returns list of section strings with headers prepended.
    Skips sections shorter than 10 characters.
    """
    if not description or not description.strip():
        return []

    description = description.strip()

    # Try each header pattern to find split points
    for pattern in _SECTION_HEADER_PATTERNS:
        matches = list(pattern.finditer(description))
        if len(matches) >= 2:
            return _split_at_matches(description, matches)

    # Fallback: paragraph splitting
    paragraphs = [p.strip() for p in description.split("\n\n") if p.strip()]
    paragraphs = [p for p in paragraphs if len(p) >= _MIN_SECTION_LENGTH]
    return paragraphs if paragraphs else ([description] if len(description) >= _MIN_SECTION_LENGTH else [])


def _split_at_matches(description: str, matches: list) -> list[str]:
    """Split description text at header match positions."""
    sections: list[str] = []

    # Preamble before first header
    if matches[0].start() > 0:
        preamble = description[:matches[0].start()].strip()
        if len(preamble) >= _MIN_SECTION_LENGTH:
            sections.append(preamble)

    # Each header + its body until the next header
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(description)
        section = description[start:end].strip()
        if len(section) >= _MIN_SECTION_LENGTH:
            sections.append(section)

    return sections
