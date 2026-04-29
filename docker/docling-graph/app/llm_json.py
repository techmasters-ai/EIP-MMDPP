"""Tolerant JSON parser (docling-graph mirror).

MIRRORED FROM app/services/llm_json.py — keep byte-for-byte identical
below the SHARED CODE marker.
"""
# === SHARED CODE BELOW THIS LINE ===
import json
import re

# Pre-compiled patterns
_RE_THINK_TAGS = re.compile(
    r"<(?:think|thinking)>.*?</(?:think|thinking)>", re.DOTALL
)
_RE_ANSWER_MARKERS = [
    re.compile(r"(?:^|\n)\s*(?:Final\s+)?Answer\s*:\s*", re.IGNORECASE),
]
_RE_FENCED_JSON = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def strip_reasoning_wrappers(raw: str) -> str:
    """Remove <think> blocks and answer prefixes from LLM output."""
    cleaned = _RE_THINK_TAGS.sub("", raw).strip()
    for pattern in _RE_ANSWER_MARKERS:
        m = pattern.search(cleaned)
        if m:
            after = cleaned[m.end():].strip()
            if after and after[0] in ("{", "["):
                cleaned = after
                break
    return cleaned


def extract_fenced_json(raw: str) -> str | None:
    """Extract the first fenced ```json code block from raw text."""
    m = _RE_FENCED_JSON.search(raw)
    return m.group(1).strip() if m else None


def extract_last_balanced_json(raw: str) -> str | None:
    """Find the last complete JSON object or array via bracket-matching."""
    candidates: list[tuple[int, int]] = []
    for close_ch, open_ch in [("}", "{"), ("]", "[")]:
        end_pos = raw.rfind(close_ch)
        if end_pos < 0:
            continue
        depth = 0
        for i in range(end_pos, -1, -1):
            if raw[i] == close_ch:
                depth += 1
            elif raw[i] == open_ch:
                depth -= 1
                if depth == 0:
                    candidates.append((i, end_pos))
                    break

    if not candidates:
        return None

    # Pick the outermost (earliest start) candidate
    json_start, json_end = min(candidates, key=lambda c: c[0])
    return raw[json_start:json_end + 1]


def parse_llm_json_loose(raw: str) -> dict | list | None:
    """Parse JSON from LLM output using multiple recovery strategies.

    Returns the parsed object/array, or None if all strategies fail.
    """
    if not raw:
        return None
    raw = raw.strip()

    # Strategy 1: whole string as JSON
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: strip reasoning markers and retry
    cleaned = strip_reasoning_wrappers(raw)
    if cleaned != raw:
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            pass
        raw = cleaned

    # Strategy 3: fenced ```json code block
    fenced = extract_fenced_json(raw)
    if fenced:
        try:
            return json.loads(fenced)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 4: last balanced JSON object/array
    snippet = extract_last_balanced_json(raw)
    if snippet:
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

        # Strategy 5: json_repair for truncated/malformed output
        try:
            import json_repair
            return json_repair.loads(snippet)
        except Exception:
            pass

    return None
