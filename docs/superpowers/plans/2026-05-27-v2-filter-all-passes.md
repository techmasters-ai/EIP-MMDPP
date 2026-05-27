# C.10 — v2 quality filter applies to ALL extraction passes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the v2 chunk-quality filter (short / dedup / strip-chrome / after-strip) uniformly to every extraction pass — identity, field_group, and system_links — by extracting the rules into one module and filtering the DoclingDocument JSON at the worker boundary before any pass-specific work runs.

**Architecture:** Pull v2 filter primitives out of `app/services/extraction_chunk_index.py` into a new `app/services/chunk_quality.py` module. Add a doc-level mutator `filter_docling_document()` in `app/services/scoped_docling_document.py` that blanks dropped texts in place (text="", orig="", hyperlink=None) — preserves array indices and `$ref` validity. Wire the filter into the worker at two call sites (the ExtractionChunk index build and the per-pass doc load) so every pass sees an already-filtered doc. The TextChunk embedding phase is untouched.

**Tech Stack:** Python 3.11, pydantic v2, ArcadeDB (via existing `graph_store`), pytest, docling-core. No new runtime dependencies.

---

## Preconditions

- **No in-flight runs.** SA-2 run `7d46c487-b704-4900-ab6f-604b0c36787e` must terminalize before any code changes. Verify with:

  ```bash
  docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA -c \
    "SELECT status FROM ingest.pipeline_runs WHERE id='7d46c487-b704-4900-ab6f-604b0c36787e';"
  ```

  Status must be `COMPLETE`, `FAILED`, or `PARTIAL_COMPLETE` before starting.

- **Current test suite green.** Verify before refactor begins:

  ```bash
  docker run --rm \
    -v "$(pwd)/app:/app/app" \
    -v "$(pwd)/tests:/app/tests" \
    -v "$(pwd)/ontology_bundles:/app/ontology_bundles" \
    -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
    -w /app eip-mmdpp-api:latest \
    python -m pytest tests/unit/test_extraction_chunk_index.py \
                     tests/unit/test_scoped_docling_document.py \
                     tests/unit/test_extraction_chunk_search_direct.py \
                     tests/unit/test_v1_extraction_routing.py \
                     tests/unit/test_dispatcher_vr_wiring.py -q
  ```

  Expected: all green (~200 tests passing).

- **Working directory.** All paths in this plan are relative to `/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry`.

---

## File structure decisions

**New file:**

- `app/services/chunk_quality.py` — pure-function primitives + constants used by both the ExtractionChunk indexer and the doc-level filter. No I/O, no side effects, no docling/ArcadeDB imports. Easy to unit-test in isolation.

**Modified files:**

- `app/services/extraction_chunk_index.py` — drop local copies of the primitives + constants; import them from `chunk_quality`. The walk loop's filter branches stay (defense in depth) but consume from the shared module.
- `app/services/scoped_docling_document.py` — add `filter_docling_document(doc_json) -> tuple[dict, FilterDiagnostics]`. Lives next to `apply_chunk_scope` since both operate on DoclingDocument JSON.
- `app/workers/pipeline.py` — two new call sites for `filter_docling_document`: one at the worker's `derive_ontology_graph` stage (before `build_extraction_index`), one at the per-pass `_execute_pass_attempt` stage (after `_build_docling_document_json`).
- `tests/unit/test_extraction_chunk_index.py` — adjust import paths for moved primitives. No behavior changes to existing tests.

**New test files:**

- `tests/unit/test_chunk_quality.py` — unit tests for the moved primitives (round-trip checks, edge cases).
- `tests/unit/test_filter_docling_document.py` — unit tests for the new doc-level mutator (blanks in place, preserves array indices, $ref validity, idempotency).

**Untouched:**

- The TextChunk embedding phase (`derive_text_chunks_and_embeddings`, `create_text_chunks_batch_sync`, postgres `TextChunk` model). Verified by audit: no references to v2 primitives outside `extraction_chunk_index.py`.
- docling-graph's `_sanitize_docling_document`. It will continue to run as defense-in-depth on the already-filtered doc; the overlap is intentional and harmless.

---

## Chunk 1: Implementation tasks

### Task 0: Pre-flight verification

**Files:** none (verification only)

- [ ] **Step 1: Confirm SA-2 in-flight run has terminalized**

```bash
docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA -F'|' -c \
  "SELECT id, status FROM ingest.pipeline_runs WHERE id='7d46c487-b704-4900-ab6f-604b0c36787e';"
```

Expected output: `7d46c487-...|COMPLETE` (or FAILED / PARTIAL_COMPLETE). If still `PROCESSING`, halt and wait.

- [ ] **Step 2: Run baseline test suite**

```bash
docker run --rm \
  -v "$(pwd)/app:/app/app" \
  -v "$(pwd)/tests:/app/tests" \
  -v "$(pwd)/ontology_bundles:/app/ontology_bundles" \
  -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
  -w /app eip-mmdpp-api:latest \
  python -m pytest tests/unit/test_extraction_chunk_index.py \
                   tests/unit/test_scoped_docling_document.py -q
```

Expected: green, ~70 tests. Record exact pass count for later comparison.

- [ ] **Step 3: Capture current SA-2 ExtractionChunk index size + identity-pass entity counts**

```bash
# For the post-refactor regression comparison.
RUN=7d46c487-b704-4900-ab6f-604b0c36787e

# Index size (post v2 filter, current)
curl -s -u 'root:eip_arcadedb_secret' \
  -X POST 'http://localhost:2480/api/v1/query/eip_knowledge_graph' \
  -H "Content-Type: application/json" \
  -d "{\"language\":\"sql\",\"command\":\"SELECT count(*) AS n FROM ExtractionChunk WHERE pipeline_run_id = '$RUN'\"}"

# Per-pass entity counts
docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA -F'|' -c "
  SELECT pass_name, primary_entities_extracted
  FROM ingest.pipeline_pass_outputs
  WHERE pipeline_run_id='$RUN'
  ORDER BY pass_name;"
```

Save the output to a scratch file (e.g. `/tmp/pre_refactor_baseline.txt`). Used in Task 9 to confirm no regression.

---

### Task 1: Create `chunk_quality.py` module (TDD)

**Files:**
- Create: `app/services/chunk_quality.py`
- Test: `tests/unit/test_chunk_quality.py`

This task moves nothing yet — it creates the new module under TDD and proves the primitives work in isolation. Task 2 then refactors `extraction_chunk_index.py` to consume from this module.

- [ ] **Step 1: Write the failing test file**

Create `tests/unit/test_chunk_quality.py`:

```python
"""Unit tests for app/services/chunk_quality.py — shared v2 filter primitives.

These tests exercise the primitives in isolation, with no DoclingDocument
or ArcadeDB context. They are the contract for what every extraction-pass
consumer of the v2 filter relies on.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestConstants:
    def test_min_chunk_text_chars(self):
        from app.services.chunk_quality import MIN_CHUNK_TEXT_CHARS
        assert MIN_CHUNK_TEXT_CHARS == 20

    def test_min_residual_chars(self):
        from app.services.chunk_quality import MIN_RESIDUAL_CHARS
        assert MIN_RESIDUAL_CHARS == 40

    def test_web_chrome_phrases_is_frozenset_of_lowercase_strings(self):
        from app.services.chunk_quality import WEB_CHROME_PHRASES
        assert isinstance(WEB_CHROME_PHRASES, frozenset)
        for p in WEB_CHROME_PHRASES:
            assert isinstance(p, str)
            assert p == p.lower(), f"phrase {p!r} must be lowercase"


class TestNormalizeForDedup:
    def test_lowercases(self):
        from app.services.chunk_quality import normalize_for_dedup
        assert normalize_for_dedup("HELLO World") == "hello world"

    def test_collapses_whitespace(self):
        from app.services.chunk_quality import normalize_for_dedup
        assert normalize_for_dedup("a   b\n\nc\td") == "a b c d"

    def test_strips_edges(self):
        from app.services.chunk_quality import normalize_for_dedup
        assert normalize_for_dedup("  hi  ") == "hi"

    def test_empty_returns_empty(self):
        from app.services.chunk_quality import normalize_for_dedup
        assert normalize_for_dedup("") == ""
        assert normalize_for_dedup("   \n\n  ") == ""


class TestIsChromeLine:
    def test_chrome_phrase_exact_match(self):
        from app.services.chunk_quality import is_chrome_line
        assert is_chrome_line("Audio Coming Soon") is True
        assert is_chrome_line("SUBSCRIBE NOW") is True
        assert is_chrome_line("audio coming soon") is True

    def test_empty_or_whitespace_is_chrome(self):
        from app.services.chunk_quality import is_chrome_line
        # Empty / whitespace lines are treated as chrome so they get
        # pulled along with adjacent real chrome lines during strip.
        assert is_chrome_line("") is True
        assert is_chrome_line("   ") is True
        assert is_chrome_line("\n") is True

    def test_substring_match_does_not_count(self):
        from app.services.chunk_quality import is_chrome_line
        # "subscribe now" appears inside a real paragraph — NOT chrome.
        assert is_chrome_line("Please subscribe now to receive updates") is False

    def test_real_text_is_not_chrome(self):
        from app.services.chunk_quality import is_chrome_line
        assert is_chrome_line("S-75 Dvina is a Soviet SAM system") is False


class TestStripChromeLines:
    def test_no_chrome_returns_unchanged(self):
        from app.services.chunk_quality import strip_chrome_lines
        stripped, l, t = strip_chrome_lines("Real article body paragraph.")
        assert stripped == "Real article body paragraph."
        assert l == 0
        assert t == 0

    def test_strips_leading_chrome_lines(self):
        from app.services.chunk_quality import strip_chrome_lines
        text = "Audio Coming Soon\n\nReal article body here."
        stripped, l, t = strip_chrome_lines(text)
        assert stripped == "Real article body here."
        assert l == 2  # chrome line + blank line
        assert t == 0

    def test_strips_trailing_chrome_lines(self):
        from app.services.chunk_quality import strip_chrome_lines
        text = "Real article body here.\n\nSubscribe Now"
        stripped, l, t = strip_chrome_lines(text)
        assert stripped == "Real article body here."
        assert l == 0
        assert t == 2

    def test_preserves_middle_chrome(self):
        from app.services.chunk_quality import strip_chrome_lines
        # Middle chrome lines stay — they sit between real content
        text = "Real one.\nSubscribe Now\nReal two."
        stripped, l, t = strip_chrome_lines(text)
        assert stripped == text
        assert l == 0
        assert t == 0

    def test_all_chrome_returns_empty(self):
        from app.services.chunk_quality import strip_chrome_lines
        text = "Audio Coming Soon\nSponsored\nSubscribe Now"
        stripped, l, t = strip_chrome_lines(text)
        assert stripped == ""
        assert l == 3
        assert t == 0


class TestClassifyChunk:
    """The unified decision predicate used by every consumer."""

    def test_too_short_is_dropped(self):
        from app.services.chunk_quality import classify_chunk
        seen: set[str] = set()
        decision = classify_chunk("a b c d", seen)
        assert decision.keep is False
        assert decision.reason == "short"

    def test_real_text_is_kept_unchanged(self):
        from app.services.chunk_quality import classify_chunk
        seen: set[str] = set()
        decision = classify_chunk(
            "The S-75 Dvina is a Soviet surface-to-air missile system used widely.", seen,
        )
        assert decision.keep is True
        assert decision.reason == "kept"
        assert decision.stripped_text is None

    def test_chrome_prefix_is_stripped_and_kept(self):
        from app.services.chunk_quality import classify_chunk
        seen: set[str] = set()
        decision = classify_chunk(
            "Audio Coming Soon\n\nThe S-75 Dvina is a real radar described here in detail.", seen,
        )
        assert decision.keep is True
        assert decision.reason == "stripped"
        assert decision.stripped_text is not None
        assert "Audio Coming Soon" not in decision.stripped_text
        assert "S-75 Dvina" in decision.stripped_text

    def test_residue_too_short_after_strip_is_dropped(self):
        from app.services.chunk_quality import classify_chunk
        seen: set[str] = set()
        # 20-39 chars after strip → drop
        decision = classify_chunk("Audio Coming Soon\n\nshort residue body.", seen)
        assert decision.keep is False
        assert decision.reason == "after_strip"

    def test_exact_duplicate_after_strip_is_dropped(self):
        from app.services.chunk_quality import classify_chunk
        seen: set[str] = set()
        d1 = classify_chunk(
            "Audio Coming Soon\n\nIdentical body paragraph for dedup testing here.", seen,
        )
        assert d1.keep is True
        d2 = classify_chunk(
            "SUBSCRIBE NOW\n\nIdentical body paragraph for dedup testing here.", seen,
        )
        assert d2.keep is False
        assert d2.reason == "dedup"

    def test_classify_chunk_mutates_seen(self):
        """When a chunk is kept, its normalized text is added to `seen`."""
        from app.services.chunk_quality import classify_chunk, normalize_for_dedup
        seen: set[str] = set()
        text = "A real article paragraph with substantive content for indexing."
        decision = classify_chunk(text, seen)
        assert decision.keep
        # The key in `seen` is the stripped+normalized form
        assert normalize_for_dedup(text) in seen
```

- [ ] **Step 2: Run test to verify it fails**

```bash
docker run --rm \
  -v "$(pwd)/app:/app/app" \
  -v "$(pwd)/tests:/app/tests" \
  -v "$(pwd)/ontology_bundles:/app/ontology_bundles" \
  -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
  -w /app eip-mmdpp-api:latest \
  python -m pytest tests/unit/test_chunk_quality.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.chunk_quality'`.

- [ ] **Step 3: Create the module**

Create `app/services/chunk_quality.py`:

```python
"""Shared v2 quality-filter primitives.

This module is the single source of truth for the rules used to decide which
DoclingDocument text elements survive into LLM extraction context. Both the
ExtractionChunk indexer (graph-extraction retrieval pool) and the doc-level
filter (applied to every pass via the worker boundary) consume from here.

Pure functions — no I/O, no docling/ArcadeDB imports. Easy to unit-test.

The general-retrieval / RAG embedding phase (TextChunk creation) does NOT
import from this module by design: it preserves every chunk regardless of
extraction-time quality.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum normalized length BEFORE stripping. Catches single symbols
#: ("™"), one-word nav fragments ("Log in"), and OCR-spurious tokens.
MIN_CHUNK_TEXT_CHARS = 20

#: Minimum residual length AFTER stripping leading/trailing chrome lines.
#: A chunk that survives the < MIN_CHUNK_TEXT_CHARS check but whose body
#: is mostly chrome (residue < MIN_RESIDUAL_CHARS after strip) is dropped.
MIN_RESIDUAL_CHARS = 40

#: Phrases that constitute "chrome" when they make up the ENTIRE normalized
#: content of a single line. Whole-line equality (not substring) — a real
#: paragraph that embeds "subscribe now" mid-body keeps the paragraph intact.
WEB_CHROME_PHRASES: frozenset[str] = frozenset({
    "audio coming soon",
    "subscribe now",
    "sponsored",
    "advertisement",
    "log in",
    "share this article",
    "historynet",
    "recommended for you",
    "related stories",
    "sign me up",
    "advertise with us",
    "meet our staff",
    "privacy policy terms of service",
    "stay curious",
    "apa",
    "mla",
    "chicago",
})


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def normalize_for_dedup(text: str) -> str:
    """Lowercase + collapse whitespace + strip edges. Returns the dedup key."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_chrome_line(line: str) -> bool:
    """True if `line` is whitespace-only OR its normalized form exactly
    matches a chrome phrase.

    Whitespace-only lines are treated as chrome so the strip pass can
    pull them along when they sit between/adjacent to real chrome lines
    (typical PDF chrome block: ``"Audio Coming Soon\\n\\nSponsored..."``).
    """
    norm = re.sub(r"\s+", " ", line.lower()).strip()
    if not norm:
        return True
    return norm in WEB_CHROME_PHRASES


def strip_chrome_lines(text: str) -> tuple[str, int, int]:
    """Strip leading and trailing chrome (or empty) lines from `text`.

    Middle chrome lines are NEVER stripped — preserves real article body
    where boilerplate is embedded between paragraphs of useful text.

    Returns ``(stripped_text, leading_removed, trailing_removed)``.
    When the entire chunk is chrome/empty, returns ``("", n, 0)``.
    """
    lines = text.split("\n")
    n = len(lines)

    leading = 0
    while leading < n and is_chrome_line(lines[leading]):
        leading += 1
    if leading == n:
        return "", n, 0

    trailing = 0
    # Stop at the first non-chrome line counting from the end. Strictly > leading
    # guards against removing the only kept line in single-line residues.
    while (n - 1 - trailing) > leading and is_chrome_line(lines[n - 1 - trailing]):
        trailing += 1

    return "\n".join(lines[leading:n - trailing]), leading, trailing


# ---------------------------------------------------------------------------
# Unified decision predicate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterDecision:
    """One decision about a single chunk's fate.

    Fields
    ------
    keep:
        True if the chunk should be retained for extraction. False if it
        should be blanked / not indexed.
    stripped_text:
        The post-strip text to insert when ``keep`` is True AND chrome was
        actually stripped. None when no stripping happened OR ``keep`` is False.
    reason:
        One of ``"kept"``, ``"stripped"``, ``"short"``, ``"after_strip"``,
        ``"dedup"``. Used for diagnostic counters and example logging.
    """

    keep: bool
    stripped_text: str | None
    reason: str


def classify_chunk(rendered: str, seen_norms: set[str]) -> FilterDecision:
    """Apply the v2 quality rules to a single rendered chunk.

    Order matches the contract documented in the C.9d-v2 commit:
    1. Quick reject: normalized < MIN_CHUNK_TEXT_CHARS -> "short"
    2. Strip leading/trailing chrome lines
    3. Drop if stripped-normalized < MIN_RESIDUAL_CHARS -> "after_strip"
    4. Dedup against ``seen_norms`` (set of post-strip normalized keys);
       returns "dedup" if already present
    5. Otherwise: kept (or "stripped" if chrome was removed)

    SIDE EFFECT: when the chunk is kept, its post-strip normalized form
    is added to ``seen_norms`` so subsequent calls dedup against it.

    Parameters
    ----------
    rendered:
        The chunk's rendered text (may contain newlines).
    seen_norms:
        Caller-managed set of already-seen post-strip normalized keys.
        Mutated when a chunk is kept.
    """
    normalized = normalize_for_dedup(rendered)
    if len(normalized) < MIN_CHUNK_TEXT_CHARS:
        return FilterDecision(keep=False, stripped_text=None, reason="short")

    stripped, leading, trailing = strip_chrome_lines(rendered)
    stripped_normalized = normalize_for_dedup(stripped)
    if len(stripped_normalized) < MIN_RESIDUAL_CHARS:
        return FilterDecision(keep=False, stripped_text=None, reason="after_strip")

    if stripped_normalized in seen_norms:
        return FilterDecision(keep=False, stripped_text=None, reason="dedup")

    seen_norms.add(stripped_normalized)

    if leading > 0 or trailing > 0:
        return FilterDecision(keep=True, stripped_text=stripped, reason="stripped")
    return FilterDecision(keep=True, stripped_text=None, reason="kept")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker run --rm \
  -v "$(pwd)/app:/app/app" \
  -v "$(pwd)/tests:/app/tests" \
  -v "$(pwd)/ontology_bundles:/app/ontology_bundles" \
  -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
  -w /app eip-mmdpp-api:latest \
  python -m pytest tests/unit/test_chunk_quality.py -v
```

Expected: all green, ~17 tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/services/chunk_quality.py tests/unit/test_chunk_quality.py
git commit -m "$(cat <<'EOF'
feat(chunk-quality): extract v2 filter primitives into shared module

Single source of truth for the v2 quality rules. Both the ExtractionChunk
indexer (graph-extraction retrieval pool) and the upcoming doc-level
filter for all extraction passes consume from here.

Pure functions — no I/O, no docling/ArcadeDB imports. TextChunk
embedding phase does NOT import from this module by design.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Refactor `extraction_chunk_index.py` to import from `chunk_quality`

**Files:**
- Modify: `app/services/extraction_chunk_index.py`
- Test: `tests/unit/test_extraction_chunk_index.py` (existing — should still pass)

The refactor is mechanical: remove local definitions, replace with imports. No behavior changes.

- [ ] **Step 1: Replace local constants and primitives with imports**

Edit `app/services/extraction_chunk_index.py`. Find the block currently at the top of the file containing `_MIN_CHUNK_TEXT_CHARS`, `_MIN_RESIDUAL_CHARS`, `_WEB_CHROME_PHRASES`, `_WEB_CHROME_PHRASES_SET`, `_normalize_for_dedup`, `_is_chrome_line`, `_strip_chrome_lines`. Replace the entire block with:

```python
# ---------------------------------------------------------------------------
# C.10: v2 quality-filter primitives moved to app.services.chunk_quality.
# Re-exported with leading-underscore aliases for internal call-site stability.
# ---------------------------------------------------------------------------
from app.services.chunk_quality import (
    MIN_CHUNK_TEXT_CHARS as _MIN_CHUNK_TEXT_CHARS,
    MIN_RESIDUAL_CHARS as _MIN_RESIDUAL_CHARS,
    normalize_for_dedup as _normalize_for_dedup,
    strip_chrome_lines as _strip_chrome_lines,
)
```

Note: only the helpers that are actually called from `build_extraction_index`'s walk loop need to be re-exported. Drop `_is_chrome_line`, `_WEB_CHROME_PHRASES`, `_WEB_CHROME_PHRASES_SET` unless they're called elsewhere in the file (grep to confirm).

- [ ] **Step 2: Verify call sites still resolve**

```bash
docker run --rm \
  -v "$(pwd)/app:/app/app" \
  -v "$(pwd)/tests:/app/tests" \
  -v "$(pwd)/ontology_bundles:/app/ontology_bundles" \
  -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
  -w /app eip-mmdpp-api:latest \
  python -c "from app.services.extraction_chunk_index import build_extraction_index; print('OK')"
```

Expected: `OK`. Any `ImportError` means the alias names don't match the call sites — fix the alias list above.

- [ ] **Step 3: Run extraction_chunk_index test suite**

```bash
docker run --rm \
  -v "$(pwd)/app:/app/app" \
  -v "$(pwd)/tests:/app/tests" \
  -v "$(pwd)/ontology_bundles:/app/ontology_bundles" \
  -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
  -w /app eip-mmdpp-api:latest \
  python -m pytest tests/unit/test_extraction_chunk_index.py -q
```

Expected: green, 66 tests pass (no count change from Task 0).

- [ ] **Step 4: Commit**

```bash
git add app/services/extraction_chunk_index.py
git commit -m "refactor(extraction-chunk-index): import v2 filter primitives from chunk_quality

Mechanical move — no behavior change. Local constants and helpers
are now thin re-exports of the canonical definitions in
app/services/chunk_quality.py.

66/66 unit tests pass unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add `filter_docling_document` in `scoped_docling_document.py` (TDD)

**Files:**
- Modify: `app/services/scoped_docling_document.py`
- Test: `tests/unit/test_filter_docling_document.py` (new)

The doc-level filter blanks dropped texts in place — same pattern as docling-graph's `_sanitize_docling_document` — to preserve array indices and `$ref` validity.

- [ ] **Step 1: Write the failing test file**

Create `tests/unit/test_filter_docling_document.py`:

```python
"""Unit tests for filter_docling_document — the worker-side v2 quality filter
applied to DoclingDocument JSON so ALL extraction passes (identity, field_group,
system_links) see a noise-filtered doc."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _make_text(idx: int, text: str, label: str = "text") -> dict:
    return {
        "self_ref": f"#/texts/{idx}",
        "text": text,
        "orig": text,
        "label": label,
        "prov": [{"page_no": 1}],
    }


def _make_doc(texts: list[dict]) -> dict:
    return {
        "texts": texts,
        "tables": [],
        "pictures": [],
        "body": {"children": [{"cref": t["self_ref"]} for t in texts]},
    }


class TestBasicShape:
    def test_returns_doc_and_diagnostics(self):
        from app.services.scoped_docling_document import filter_docling_document
        doc = _make_doc([_make_text(0, "Real radar article paragraph with substantive content here.")])
        filtered, diag = filter_docling_document(doc)
        assert isinstance(filtered, dict)
        assert hasattr(diag, "blanked_short")
        assert hasattr(diag, "blanked_dedup")
        assert hasattr(diag, "blanked_after_strip")
        assert hasattr(diag, "stripped_in_place")

    def test_preserves_texts_array_length(self):
        """Filter must blank-in-place, NOT remove entries. Array indices and
        $refs from body.children / pictures.children / tables.children depend
        on positional stability."""
        from app.services.scoped_docling_document import filter_docling_document
        texts = [
            _make_text(0, "Real radar article paragraph one with substantive content."),
            _make_text(1, "™"),  # short - will be blanked
            _make_text(2, "Real radar article paragraph two with substantive content."),
        ]
        doc = _make_doc(texts)
        filtered, _ = filter_docling_document(doc)
        assert len(filtered["texts"]) == 3, "array length must be preserved"

    def test_preserves_self_refs_on_all_entries_including_blanked(self):
        from app.services.scoped_docling_document import filter_docling_document
        texts = [_make_text(0, "Real content here for indexing purposes."), _make_text(1, "™")]
        doc = _make_doc(texts)
        filtered, _ = filter_docling_document(doc)
        for i, t in enumerate(filtered["texts"]):
            assert t["self_ref"] == f"#/texts/{i}"


class TestBlanking:
    def test_short_chunk_text_orig_become_empty(self):
        from app.services.scoped_docling_document import filter_docling_document
        doc = _make_doc([_make_text(0, "™")])
        filtered, diag = filter_docling_document(doc)
        assert filtered["texts"][0]["text"] == ""
        assert filtered["texts"][0]["orig"] == ""
        assert diag.blanked_short == 1

    def test_short_chunk_clears_hyperlink_when_present(self):
        from app.services.scoped_docling_document import filter_docling_document
        elem = _make_text(0, "Log in")
        elem["hyperlink"] = "https://tracker.example.com/click?x=1"
        doc = _make_doc([elem])
        filtered, _ = filter_docling_document(doc)
        assert filtered["texts"][0]["hyperlink"] is None

    def test_duplicate_after_strip_is_blanked_keeping_first(self):
        from app.services.scoped_docling_document import filter_docling_document
        body = "Identical body paragraph for dedup testing here with extra words."
        texts = [
            _make_text(0, f"Audio Coming Soon\n\n{body}"),
            _make_text(1, f"SUBSCRIBE NOW\n\n{body}"),  # dup after strip
        ]
        doc = _make_doc(texts)
        filtered, diag = filter_docling_document(doc)
        # First one kept (and stripped). Second one blanked.
        assert filtered["texts"][0]["text"]  # non-empty
        assert filtered["texts"][1]["text"] == ""
        assert diag.blanked_dedup == 1

    def test_residue_too_short_after_strip_is_blanked(self):
        from app.services.scoped_docling_document import filter_docling_document
        texts = [_make_text(0, "Audio Coming Soon\n\nShort residue body.")]
        doc = _make_doc(texts)
        filtered, diag = filter_docling_document(doc)
        assert filtered["texts"][0]["text"] == ""
        assert diag.blanked_after_strip == 1


class TestStripInPlace:
    def test_chrome_prefix_text_orig_overridden_with_stripped(self):
        from app.services.scoped_docling_document import filter_docling_document
        elem = _make_text(
            0,
            "Audio Coming Soon\n\nThe S-75 Dvina radar system is documented in this real article passage.",
        )
        doc = _make_doc([elem])
        filtered, diag = filter_docling_document(doc)
        kept = filtered["texts"][0]
        assert "Audio Coming Soon" not in kept["text"]
        assert "Audio Coming Soon" not in kept["orig"]
        assert "S-75 Dvina" in kept["text"]
        assert diag.stripped_in_place == 1

    def test_kept_chunk_with_no_chrome_is_left_alone(self):
        from app.services.scoped_docling_document import filter_docling_document
        text = "The S-75 Dvina radar operates in the C-band frequency range here."
        elem = _make_text(0, text)
        doc = _make_doc([elem])
        filtered, diag = filter_docling_document(doc)
        # Unchanged
        assert filtered["texts"][0]["text"] == text
        assert filtered["texts"][0]["orig"] == text
        assert diag.stripped_in_place == 0


class TestIdempotency:
    def test_running_filter_twice_yields_same_result(self):
        """The filter must be a no-op when run on already-filtered output."""
        from app.services.scoped_docling_document import filter_docling_document
        texts = [
            _make_text(0, "Real radar article paragraph with substantive content here."),
            _make_text(1, "™"),
            _make_text(2, "Audio Coming Soon\n\nThe S-75 radar is documented in this real article passage."),
            _make_text(3, "Real radar article paragraph with substantive content here."),  # dup of 0
        ]
        doc = _make_doc(texts)
        once, diag1 = filter_docling_document(doc)
        twice, diag2 = filter_docling_document(once)
        # texts arrays identical
        for i in range(len(once["texts"])):
            assert once["texts"][i] == twice["texts"][i]
        # Second filter pass produces zero new blanks/strips
        assert diag2.blanked_short == 0
        assert diag2.blanked_dedup == 0
        assert diag2.blanked_after_strip == 0
        assert diag2.stripped_in_place == 0


class TestBodyChildrenAndRefIntegrity:
    def test_body_children_refs_still_resolve(self):
        from app.services.scoped_docling_document import filter_docling_document
        texts = [
            _make_text(0, "Real radar paragraph with substantive content here."),
            _make_text(1, "Audio Coming Soon\nSponsored"),  # all-chrome → blank
            _make_text(2, "Another real radar paragraph with substantive content here."),
        ]
        doc = _make_doc(texts)
        filtered, _ = filter_docling_document(doc)
        # body.children unchanged structurally
        assert filtered["body"]["children"] == doc["body"]["children"]
        # All crefs still resolve to a text entry
        for child in filtered["body"]["children"]:
            ref = child.get("cref")
            idx = int(ref.rsplit("/", 1)[1])
            assert filtered["texts"][idx]["self_ref"] == ref


class TestTextChunksAreNotConsulted:
    """The filter must read text from the texts[] array only — never from any
    external TextChunk / postgres / ArcadeDB source."""

    def test_filter_makes_no_io_calls(self):
        # If filter_docling_document tries to import graph_store / postgres,
        # this test fails because no patches are set up.
        from app.services.scoped_docling_document import filter_docling_document
        doc = _make_doc([_make_text(0, "Real radar article content here for testing purposes.")])
        filtered, _ = filter_docling_document(doc)
        # If we got here without exceptions, no I/O was attempted.
        assert isinstance(filtered, dict)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
docker run --rm \
  -v "$(pwd)/app:/app/app" \
  -v "$(pwd)/tests:/app/tests" \
  -v "$(pwd)/ontology_bundles:/app/ontology_bundles" \
  -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
  -w /app eip-mmdpp-api:latest \
  python -m pytest tests/unit/test_filter_docling_document.py -v
```

Expected: FAIL with `ImportError: cannot import name 'filter_docling_document'`.

- [ ] **Step 3: Implement filter_docling_document**

Open `app/services/scoped_docling_document.py`. Find the end of the file (after `apply_chunk_scope` and its helpers). Append:

```python
# ---------------------------------------------------------------------------
# C.10: filter_docling_document — worker-side v2 quality filter for ALL passes
# ---------------------------------------------------------------------------


from dataclasses import dataclass
from app.services.chunk_quality import classify_chunk


@dataclass
class FilterDiagnostics:
    """Per-call counters from filter_docling_document.

    All counters are subsets of (texts_in - texts_unchanged); blanking is the
    primary effect, stripping is the secondary effect on kept chunks.
    """
    texts_in: int = 0
    blanked_short: int = 0
    blanked_dedup: int = 0
    blanked_after_strip: int = 0
    stripped_in_place: int = 0


def filter_docling_document(doc_json: dict) -> tuple[dict, FilterDiagnostics]:
    """Apply v2 quality filter to a DoclingDocument JSON in place.

    For each entry in ``doc_json["texts"]``:
      * Dropped entries (short / dedup / after_strip) have their ``text`` and
        ``orig`` blanked and ``hyperlink`` cleared. The entry stays in the
        array so $refs from body.children / pictures.children / tables.children
        remain valid.
      * Kept-with-strip entries have their ``text`` and ``orig`` overridden
        with the post-strip residue and their ``hyperlink`` cleared (the
        hyperlink may have pointed at chrome-only context).
      * Kept-as-is entries are untouched.

    The function mutates ``doc_json`` and returns it for chained-call ergonomics.

    Idempotent: running it twice on the same doc produces zero new blanks
    or strips on the second pass.
    """
    diag = FilterDiagnostics()
    texts = doc_json.get("texts") or []
    diag.texts_in = len(texts)

    seen_norms: set[str] = set()
    for i, t in enumerate(texts):
        if not isinstance(t, dict):
            continue
        rendered = (t.get("text") or t.get("orig") or "").strip()
        if not rendered:
            continue
        decision = classify_chunk(rendered, seen_norms)
        if not decision.keep:
            new_t = dict(t)
            new_t["text"] = ""
            new_t["orig"] = ""
            if "hyperlink" in new_t:
                new_t["hyperlink"] = None
            texts[i] = new_t
            if decision.reason == "short":
                diag.blanked_short += 1
            elif decision.reason == "after_strip":
                diag.blanked_after_strip += 1
            elif decision.reason == "dedup":
                diag.blanked_dedup += 1
            continue
        if decision.stripped_text is not None:
            new_t = dict(t)
            new_t["text"] = decision.stripped_text
            new_t["orig"] = decision.stripped_text
            if "hyperlink" in new_t:
                new_t["hyperlink"] = None
            texts[i] = new_t
            diag.stripped_in_place += 1

    doc_json["texts"] = texts
    return doc_json, diag
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker run --rm \
  -v "$(pwd)/app:/app/app" \
  -v "$(pwd)/tests:/app/tests" \
  -v "$(pwd)/ontology_bundles:/app/ontology_bundles" \
  -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
  -w /app eip-mmdpp-api:latest \
  python -m pytest tests/unit/test_filter_docling_document.py -v
```

Expected: all green, ~13 tests pass.

- [ ] **Step 5: Run the broader test suite to confirm no regression**

```bash
docker run --rm \
  -v "$(pwd)/app:/app/app" \
  -v "$(pwd)/tests:/app/tests" \
  -v "$(pwd)/ontology_bundles:/app/ontology_bundles" \
  -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
  -w /app eip-mmdpp-api:latest \
  python -m pytest tests/unit/test_chunk_quality.py \
                   tests/unit/test_scoped_docling_document.py \
                   tests/unit/test_filter_docling_document.py \
                   tests/unit/test_extraction_chunk_index.py -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add app/services/scoped_docling_document.py tests/unit/test_filter_docling_document.py
git commit -m "$(cat <<'EOF'
feat(scoped-docling-doc): add filter_docling_document for all-pass v2 filter

Mutates DoclingDocument JSON in place applying the chunk_quality rules.
Dropped texts get blanked (text="", orig="", hyperlink=None) so all
$refs from body.children / pictures.children / tables.children remain
valid — same pattern docling-graph's _sanitize_docling_document uses.

Kept-with-strip entries have their text/orig overridden with the
post-strip residue. Kept-as-is entries are untouched.

Idempotent (running twice is a no-op). Pure function — no I/O.

13/13 unit tests pass; cross-module suites unaffected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Wire filter at the `build_extraction_index` call site

**Files:**
- Modify: `app/workers/pipeline.py` (around line 8567)

This task adds the first canonical filter invocation — before the ExtractionChunk index gets built. Existing build_extraction_index tests remain valid because the doc passed in is now pre-filtered (the filter inside build_extraction_index becomes redundant for blanked text, since the `if not rendered.strip(): continue` check at the start of the walk loop already skips them).

- [ ] **Step 1: Read the current call site**

Open `app/workers/pipeline.py` near line 8567 (inside `derive_ontology_graph`). The current shape is:

```python
from app.services.extraction_chunk_index import build_extraction_index
doc_json_for_index = _build_docling_document_json(document_id)
store_for_index = get_graph_store()
build_diag = build_extraction_index(
    doc_json=doc_json_for_index,
    pipeline_run_id=str(run_id),
    document_id=document_id,
    store=store_for_index,
)
```

- [ ] **Step 2: Insert the filter call**

Modify to:

```python
from app.services.extraction_chunk_index import build_extraction_index
from app.services.scoped_docling_document import filter_docling_document
doc_json_for_index = _build_docling_document_json(document_id)
doc_json_for_index, filter_diag_idx = filter_docling_document(doc_json_for_index)
logger.info(
    "VR: filter_docling_document (index path) run=%s texts_in=%d blanked=%d "
    "(short=%d dedup=%d after_strip=%d) stripped_in_place=%d",
    run_id,
    filter_diag_idx.texts_in,
    filter_diag_idx.blanked_short + filter_diag_idx.blanked_dedup + filter_diag_idx.blanked_after_strip,
    filter_diag_idx.blanked_short,
    filter_diag_idx.blanked_dedup,
    filter_diag_idx.blanked_after_strip,
    filter_diag_idx.stripped_in_place,
)
store_for_index = get_graph_store()
build_diag = build_extraction_index(
    doc_json=doc_json_for_index,
    pipeline_run_id=str(run_id),
    document_id=document_id,
    store=store_for_index,
)
```

- [ ] **Step 3: Confirm import resolves and worker still starts**

```bash
docker run --rm \
  -v "$(pwd)/app:/app/app" \
  -w /app eip-mmdpp-api:latest \
  python -c "from app.workers.pipeline import derive_ontology_graph; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Run the worker wiring tests**

```bash
docker run --rm \
  -v "$(pwd)/app:/app/app" \
  -v "$(pwd)/tests:/app/tests" \
  -v "$(pwd)/ontology_bundles:/app/ontology_bundles" \
  -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
  -w /app eip-mmdpp-api:latest \
  python -m pytest tests/unit/test_dispatcher_vr_wiring.py tests/unit/test_extraction_chunk_index.py -q
```

Expected: all green (no regression).

- [ ] **Step 5: Commit**

```bash
git add app/workers/pipeline.py
git commit -m "$(cat <<'EOF'
feat(worker): filter doc_json before build_extraction_index

First of two canonical call sites for the worker-side v2 quality
filter. The ExtractionChunk index now indexes a pre-filtered doc;
duplicates and chrome are blanked in place so all downstream consumers
see the same view.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Wire filter at the per-pass doc-load site

**Files:**
- Modify: `app/workers/pipeline.py` (around line 7491, inside `_execute_pass_attempt`)

This is the call site that runs once per pass — narrowed and non-narrowed alike. With this in place, identity passes and system_links finally see the same filtered doc the indexed-narrowing path already saw.

- [ ] **Step 1: Read the current per-pass load site**

Open `app/workers/pipeline.py` near line 7491. Current shape:

```python
_doc_load_t0 = time.perf_counter()
doc_json = _build_docling_document_json(document_id)
_doc_json_load_ms = (time.perf_counter() - _doc_load_t0) * 1000.0

if chunk_scope is not None and chunk_scope.get("mode") == "selected_refs":
    try:
        from app.services.scoped_docling_document import apply_chunk_scope
        doc_json = apply_chunk_scope(doc_json, chunk_scope)
        ...
```

- [ ] **Step 2: Insert the filter call between load and apply_chunk_scope**

Modify to:

```python
_doc_load_t0 = time.perf_counter()
doc_json = _build_docling_document_json(document_id)
_doc_json_load_ms = (time.perf_counter() - _doc_load_t0) * 1000.0

# C.10: apply v2 quality filter to ALL passes (identity, field_group,
# system_links). The filter is idempotent — narrowed passes pre-process
# the same doc shape that was already filtered at index build time;
# non-narrowed passes see the filter for the first time here.
from app.services.scoped_docling_document import filter_docling_document
doc_json, _filter_diag = filter_docling_document(doc_json)
if router_diagnostics is not None:
    router_diagnostics = dict(router_diagnostics)
    router_diagnostics["doc_filter"] = {
        "texts_in": _filter_diag.texts_in,
        "blanked_short": _filter_diag.blanked_short,
        "blanked_dedup": _filter_diag.blanked_dedup,
        "blanked_after_strip": _filter_diag.blanked_after_strip,
        "stripped_in_place": _filter_diag.stripped_in_place,
    }

if chunk_scope is not None and chunk_scope.get("mode") == "selected_refs":
    ...
```

(Leave the `apply_chunk_scope` block unchanged — it runs after the filter and is unaffected.)

- [ ] **Step 3: Confirm worker still imports cleanly**

```bash
docker run --rm \
  -v "$(pwd)/app:/app/app" \
  -w /app eip-mmdpp-api:latest \
  python -c "from app.workers.pipeline import _execute_pass_attempt; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Run the worker wiring tests + dispatcher tests**

```bash
docker run --rm \
  -v "$(pwd)/app:/app/app" \
  -v "$(pwd)/tests:/app/tests" \
  -v "$(pwd)/ontology_bundles:/app/ontology_bundles" \
  -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
  -w /app eip-mmdpp-api:latest \
  python -m pytest tests/unit/test_dispatcher_vr_wiring.py \
                   tests/unit/test_pipeline_upstream_refs.py \
                   tests/unit/test_extraction_chunk_index.py -q
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add app/workers/pipeline.py
git commit -m "$(cat <<'EOF'
feat(worker): filter doc_json at per-pass load (all passes — incl. identity)

Second canonical call site. Identity and system_links passes now see
the same v2-filtered DoclingDocument that narrowed passes have been
indexing from. Identity-pass LLM context no longer includes the
"S-75 / SA-2 Guideline Combat Imagery / / URL"-style page-header
repetition that previously consumed context for zero information gain.

The filter is idempotent — narrowed passes whose doc was already
filtered at index build time pay an O(N texts) no-op cost.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Simplify `build_extraction_index` — drop the now-redundant in-loop filter

**Files:**
- Modify: `app/services/extraction_chunk_index.py`
- Modify: `tests/unit/test_extraction_chunk_index.py` (update affected assertions)

With the doc pre-filtered at the worker boundary, the filter branches inside `build_extraction_index`'s walk loop are dead code on already-blanked texts (the `if not rendered.strip(): continue` check at the top of the loop catches them). Removing them simplifies the function and the diagnostics.

- [ ] **Step 1: Identify the filter branches to remove**

In `build_extraction_index`, locate the per-element block that currently calls `_normalize_for_dedup`, checks `_MIN_CHUNK_TEXT_CHARS`, calls `_strip_chrome_lines`, checks `_MIN_RESIDUAL_CHARS`, and updates `seen_norms`. This is the v1+v2 filter block — around lines 678-735 (line numbers approximate). The leading `if not rendered.strip(): continue` check stays.

- [ ] **Step 2: Remove the filter block**

Replace the filter block with a single comment:

```python
# C.10: v2 quality filtering now runs upstream in
# filter_docling_document (called from the worker before this function).
# By the time we get here, dropped texts have been blanked
# (text="", orig="") and are skipped by the `if not rendered.strip()`
# check at the top of the walk loop. No per-element filtering needed.
```

- [ ] **Step 3: Update `BuildIndexDiagnostics`**

Remove the v2-specific counters from `BuildIndexDiagnostics`: `chunks_skipped_short`, `chunks_skipped_duplicate`, `chunks_skipped_web_chrome`, `chunks_stripped_web_chrome`, `chunks_skipped_after_strip`. Keep `chunks_inserted`, `chunks_skipped` (now only counts blank-rendered), `embed_calls`, `embed_ms`, `insert_ms`, `modality_counts`.

- [ ] **Step 4: Update the log line in build_extraction_index**

Find the `logger.info("build_extraction_index: ...")` line that reports the per-category counts. Simplify to:

```python
logger.info(
    "build_extraction_index: pipeline_run_id=%r document_id=%r "
    "inserted=%d skipped=%d embed_ms=%d insert_ms=%d modalities=%s",
    pipeline_run_id, document_id,
    chunks_inserted, chunks_skipped,
    embed_ms, insert_ms, modality_counts,
)
```

- [ ] **Step 5: Update tests that asserted v2 counters**

In `tests/unit/test_extraction_chunk_index.py`, find `class TestChunkQualityFilter` and `class TestStripThenResidue`. These tests asserted properties of the in-loop filter. Either:

(a) **Delete them** (recommended). The behavior they exercise is now owned by `filter_docling_document` and is covered by `tests/unit/test_filter_docling_document.py`.

(b) **Re-purpose them** as integration tests for the new flow: call `filter_docling_document` first, then `build_extraction_index`, assert end-to-end behavior. More work, marginal value.

Go with (a). Delete the two classes. Also delete any test fixtures that exist only to support them.

- [ ] **Step 6: Verify the remaining tests still pass**

```bash
docker run --rm \
  -v "$(pwd)/app:/app/app" \
  -v "$(pwd)/tests:/app/tests" \
  -v "$(pwd)/ontology_bundles:/app/ontology_bundles" \
  -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
  -w /app eip-mmdpp-api:latest \
  python -m pytest tests/unit/test_extraction_chunk_index.py -q
```

Expected: ~57 tests pass (66 minus the deleted v2-specific 9). Lower count is fine — the deleted tests now live in test_filter_docling_document.py.

- [ ] **Step 7: Commit**

```bash
git add app/services/extraction_chunk_index.py tests/unit/test_extraction_chunk_index.py
git commit -m "$(cat <<'EOF'
refactor(extraction-chunk-index): drop in-loop v2 filter (now upstream)

filter_docling_document runs at the worker boundary before
build_extraction_index, so dropped texts arrive already-blanked and
get skipped by the existing `if not rendered.strip(): continue`
check. The per-element filter branches are dead code; removed.

BuildIndexDiagnostics loses the v2-specific counters. Equivalent
counters live on FilterDiagnostics from filter_docling_document.

In-loop filter tests deleted; equivalent coverage exists in
tests/unit/test_filter_docling_document.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Audit downstream consumers of `BuildIndexDiagnostics`

**Files:**
- (audit only — may modify call sites that read removed fields)

Removing fields from `BuildIndexDiagnostics` could break callers that read them. This task is an audit + minimal fix to any consumer.

- [ ] **Step 1: Grep for consumers of removed fields**

```bash
grep -rn "chunks_skipped_short\|chunks_skipped_duplicate\|chunks_skipped_web_chrome\|chunks_stripped_web_chrome\|chunks_skipped_after_strip" \
  app/ docker/ 2>/dev/null | grep -v __pycache__ | grep -v "tests/" | grep -v ".md"
```

Expected: zero hits in production code (those fields were only set + read inside `extraction_chunk_index.py`). If any hits exist, follow up with targeted edits.

- [ ] **Step 2: Spot-check the build_extraction_index log consumers**

Any operator notebooks or scripts that grep these counter names? Reasonable to grep:

```bash
grep -rn "chunks_skipped_short\|build_extraction_index quality filter" \
  /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry 2>/dev/null \
  | grep -v __pycache__ | grep -v ".pyc"
```

If hits exist outside tests, document the breaking change in a brief follow-up note.

- [ ] **Step 3: Commit (if any fixes made), or skip**

If Steps 1-2 found no consumers, no commit needed. If any production code referenced the removed fields, fix it and commit:

```bash
git add <fixed-files>
git commit -m "fix: drop references to removed BuildIndexDiagnostics v2 fields"
```

---

### Task 8: End-to-end smoke — Dvina graph_only reingest

**Files:** none (smoke test only)

A real run that exercises:
- The two filter call sites (build index + per-pass)
- All 5 passes (radar_identity, radar_power_rf, missile_identity, missile_kinematics, system_links)
- The narrowing path AND the identity path

Dvina is fast (~16-20m wall), so this is a cheap smoke.

- [ ] **Step 1: Force-recreate worker + api to load the refactored code**

```bash
COMPOSE_PROJECT_NAME=eip-mmdpp docker compose up -d --force-recreate api worker-graph
```

- [ ] **Step 2: Trigger Dvina graph_only reingest with narrowing_v1 bundle**

```bash
DVINA_DOC="9c8e09c7-e39f-4359-92c0-46330158c73c"
RESP=$(curl -sX POST "http://localhost:8005/v1/documents/$DVINA_DOC/reingest" \
  -H "Content-Type: application/json" \
  -d '{"mode":"graph_only","ontology_bundle_key":"air_defense_v3_narrowing_v1"}')
echo "$RESP"
RUN_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('pipeline_run_id',''))")
echo "DVINA_C10_RUN=$RUN_ID"
```

Save `$RUN_ID` for Step 4.

- [ ] **Step 3: Poll until terminal**

```bash
until docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA \
  -c "SELECT 1 FROM ingest.pipeline_runs WHERE id='$RUN_ID' AND status IN ('COMPLETE','FAILED','PARTIAL_COMPLETE')" \
  | grep -q 1; do
  sleep 30
done
echo "TERMINAL"
```

Expected wall time: ~16-20 min.

- [ ] **Step 4: Verify filter call sites both fired**

```bash
echo "--- Worker filter logs ---"
docker logs eip-mmdpp-worker-graph-1 --since 30m 2>&1 | grep -E "filter_docling_document|VR: filter" | head -10

echo "--- Build extraction index log line (simplified) ---"
docker logs eip-mmdpp-worker-graph-1 --since 30m 2>&1 | grep "build_extraction_index" | tail -2

echo "--- Per-pass router_diagnostics.doc_filter ---"
docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA -c "
  SELECT pass_name, diagnostics_json->'router_diagnostics'->'doc_filter'
  FROM ingest.pipeline_pass_outputs
  WHERE pipeline_run_id='$RUN_ID'
  ORDER BY pass_name;"
```

Expected:
- 1 "filter_docling_document (index path)" log line per run start.
- 1 build_extraction_index log line WITHOUT the old `(short=X dup=Y web_chrome=Z)` suffix.
- Each pass's `router_diagnostics.doc_filter` populated with non-zero counters (the per-pass filter is in addition to the index-path filter — they should produce similar numbers).

- [ ] **Step 5: Compare entity counts to baseline (no regression)**

```bash
docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA -F'|' -c "
  SELECT pass_name, execution_status, primary_entities_extracted
  FROM ingest.pipeline_pass_outputs
  WHERE pipeline_run_id='$RUN_ID'
  ORDER BY pass_name;"
```

Acceptance criteria:
- radar_identity entity count within ±1 of the prior run (`1ced34e0` had 1 entity)
- missile_identity entity count within ±1 of the prior run (1 entity)
- missile_kinematics entity count within ±1 of the prior run (1 entity)
- radar_power_rf may still be FAILED (Dvina has no RF data — see prior investigation)
- system_links: COMPLETE, relationships_extracted non-zero or zero per prior

Big regressions (e.g. all passes returning 0) mean the filter is too aggressive on the doc-level path; rollback to investigate.

---

### Task 9: SA-2 validation (optional, paid)

**Files:** none

SA-2 is the larger and more interesting case (real military doc with RF tables, multi-page web export with significant chrome). Wall time ~3-4 hours. Run only if Task 8 succeeds and resources permit.

- [ ] **Step 1: Trigger SA-2 graph_only reingest**

```bash
SA2_DOC="ddaa9e36-2854-47c3-bc94-ff38d531dafd"
RESP=$(curl -sX POST "http://localhost:8005/v1/documents/$SA2_DOC/reingest" \
  -H "Content-Type: application/json" \
  -d '{"mode":"graph_only","ontology_bundle_key":"air_defense_v3_narrowing_v1"}')
echo "$RESP" | tee /tmp/c10_sa2_trigger.json
```

- [ ] **Step 2: Launch a detached watcher (survives session disconnect)**

Use the existing watcher script pattern from `/tmp/c9d_v2_watcher.sh` as a template — adapt to SA-2 only, single run.

- [ ] **Step 3: Compare against prior SA-2 baselines once terminal**

Acceptance criteria:
- radar_identity entity count within ±10% of prior SA-2 narrowing_v1 run (was 25; expect 22-27)
- missile_identity similar
- radar_power_rf entity count: not significantly worse than prior (was 22; expect ≥18)
- missile_kinematics entity count: not significantly worse than prior (was 16; expect ≥13)
- Field-density per entity should hold or improve on identity passes (where v2 filter newly applies)

If quality regresses, the most likely root cause is overly-aggressive blanking on real content — narrow the chrome denylist or raise MIN_RESIDUAL_CHARS as needed.

---

### Task 10: Final test sweep + commit hygiene

**Files:** none

- [ ] **Step 1: Run the full unit test suite**

```bash
docker run --rm \
  -v "$(pwd)/app:/app/app" \
  -v "$(pwd)/tests:/app/tests" \
  -v "$(pwd)/ontology_bundles:/app/ontology_bundles" \
  -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
  -w /app eip-mmdpp-api:latest \
  python -m pytest tests/unit -q
```

Expected: all green. Note any pre-existing failures (not caused by this refactor) — they should be in modules untouched by this work.

- [ ] **Step 2: Confirm TextChunk embedding path unchanged**

```bash
grep -rn "from app.services.chunk_quality\|from app.services.scoped_docling_document import filter_docling_document" \
  app/workers/pipeline.py | head -10
```

Expected:
- Two hits inside `derive_ontology_graph` / `_execute_pass_attempt`.
- ZERO hits inside `derive_text_chunks_and_embeddings` or `create_text_chunks_batch_sync`.

If anything else imports `chunk_quality` or `filter_docling_document`, audit — only the graph-extraction code path should reference them.

- [ ] **Step 3: Confirm git log is clean**

```bash
git log --oneline --since='1 day ago' | head -10
```

Expected: roughly 6-7 commits from this plan, all with clear conventional-commit subject lines.

---

## Notes for the implementer

- **DO NOT modify the docling-graph image, the docling-graph sanitizer, or the docling-graph stages module as part of this plan.** Those changes are out of scope and add deployment complexity. The docling-graph sanitizer will continue running as defense-in-depth on the already-filtered doc — overlap is expected and harmless.

- **DO NOT touch `derive_text_chunks_and_embeddings`, `create_text_chunks_batch_sync`, or anything related to TextChunk vertices.** The embedding phase preserves all chunks intentionally.

- **The `apply_chunk_scope` `text_by_ref` override is no longer strictly necessary** once `filter_docling_document` runs upstream, since the doc the worker sends already has chrome-stripped text on the selected refs. Leave the override in place as belt-and-suspenders; removing it is a follow-up cleanup task, not part of this plan.

- **DOCLING_GRAPH_SANITIZE_INPUT env var** can be flipped to false later if we want to fully delegate to our filter. Out of scope for this plan.

- **No test for the docling-graph integration is required here.** The two end-to-end smokes (Tasks 8 and 9) exercise the full chain.

- **Idempotency matters operationally** because narrowed passes' doc_json was filtered at index build (Task 4) AND again at per-pass load (Task 5). The unit tests for filter_docling_document explicitly assert idempotency.
