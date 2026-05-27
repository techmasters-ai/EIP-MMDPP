# C.10 — v2 quality filter applies to ALL extraction passes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the v2 chunk-quality filter (short / dedup / strip-chrome / after-strip) uniformly to every extraction pass — identity, field_group, and system_links — by extracting the rules into one module and filtering the DoclingDocument JSON at the worker boundary before any pass-specific work runs.

**Architecture:** Pull v2 filter primitives out of `app/services/extraction_chunk_index.py` into a new `app/services/chunk_quality.py` module. Add a doc-level mutator `filter_docling_document()` in `app/services/scoped_docling_document.py` that blanks dropped texts in place (text="", orig="", hyperlink=None) — preserves array indices and `$ref` validity. Wire the filter into the worker at two call sites (the ExtractionChunk index build and the per-pass doc load) so every pass sees an already-filtered doc. The TextChunk embedding phase is untouched.

**Layering with the existing in-loop filter:** `filter_docling_document` operates on `texts[]` only (and protects `label=="caption"` from dedup/blank). The in-loop filter inside `build_extraction_index` is **NOT** removed — it stays as a second layer that catches rendered table-cell markdown and any rendering-time content that doesn't have a `texts[]` source. Identity passes get the texts[]-level filter (via the worker boundary); narrowed passes get both layers; the layers are idempotent so the double-pass is a no-op on already-clean text.

**Tech Stack:** Python 3.11, pydantic v2, ArcadeDB (via existing `graph_store`), pytest, docling-core. No new runtime dependencies.

---

## Preconditions

- **No in-flight runs.** SA-2 run `7d46c487-b704-4900-ab6f-604b0c36787e` must terminalize before any code changes. Verify with:

  ```bash
  docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA -c \
    "SELECT status FROM ingest.pipeline_runs WHERE id='7d46c487-b704-4900-ab6f-604b0c36787e';"
  ```

  Status must be `COMPLETE`, `FAILED`, or `PARTIAL_COMPLETE` before starting.

- **No stale background processes.** The `/tmp/c9d_v2_watcher.sh` script from earlier session work may still be running. Verify it has exited:

  ```bash
  ps -ef | grep -v grep | grep -E "c9d_v2_watcher|c9c_dvina|c9d.*watcher" || echo "no watchers running"
  ```

  Expected: `no watchers running`. If any process exists, kill it (`pkill -f c9d_v2_watcher`) before proceeding.

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

Expected: all green, 22 tests pass.

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
git commit -m "$(cat <<'EOF'
refactor(extraction-chunk-index): import v2 filter primitives from chunk_quality

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


class TestCaptionProtection:
    """label="caption" entries are protected from blanking + dedup. docling-graph's
    own sanitizer (main.py:482) uses the same protection — match its semantics."""

    def test_short_caption_is_not_blanked(self):
        from app.services.scoped_docling_document import filter_docling_document
        # 8 chars — would normally be blanked as "short"
        elem = _make_text(0, "Fig. 1.", label="caption")
        doc = _make_doc([elem])
        filtered, diag = filter_docling_document(doc)
        assert filtered["texts"][0]["text"] == "Fig. 1."  # untouched
        assert diag.blanked_short == 0

    def test_duplicate_captions_both_kept(self):
        """Two real captions with identical text — both must be preserved."""
        from app.services.scoped_docling_document import filter_docling_document
        c1 = _make_text(0, "Figure: SA-2 launcher schematic.", label="caption")
        c2 = _make_text(1, "Figure: SA-2 launcher schematic.", label="caption")
        doc = _make_doc([c1, c2])
        filtered, diag = filter_docling_document(doc)
        assert filtered["texts"][0]["text"] == "Figure: SA-2 launcher schematic."
        assert filtered["texts"][1]["text"] == "Figure: SA-2 launcher schematic."
        assert diag.blanked_dedup == 0


class TestDefensiveEdgeCases:
    """Malformed docs must not crash the filter. The worker wraps the call in
    try/except but the function itself should also fail-safe for the most common
    shapes."""

    def test_missing_texts_key(self):
        from app.services.scoped_docling_document import filter_docling_document
        filtered, diag = filter_docling_document({"body": {"children": []}})
        assert diag.texts_in == 0

    def test_texts_is_empty_list(self):
        from app.services.scoped_docling_document import filter_docling_document
        filtered, diag = filter_docling_document({"texts": []})
        assert diag.texts_in == 0

    def test_text_element_is_not_a_dict(self):
        from app.services.scoped_docling_document import filter_docling_document
        doc = {"texts": ["not-a-dict", _make_text(0, "Real article content for indexing.")]}
        filtered, diag = filter_docling_document(doc)
        # Non-dict skipped; real dict processed
        assert filtered["texts"][0] == "not-a-dict"  # untouched
        assert filtered["texts"][1]["text"] == "Real article content for indexing."

    def test_text_field_is_none(self):
        from app.services.scoped_docling_document import filter_docling_document
        elem = _make_text(0, "")
        elem["text"] = None  # docling has been observed to emit None
        elem["orig"] = None
        doc = _make_doc([elem])
        # Must not raise on None
        filtered, diag = filter_docling_document(doc)
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

Open `app/services/scoped_docling_document.py`. Add the `dataclass` and `classify_chunk` imports at the **top of the file** alongside existing imports (PEP 8 placement). At the end of the file, append the new function:

```python
# At top of file, add to the existing import block:
from dataclasses import dataclass
from app.services.chunk_quality import classify_chunk

# At end of file, append:

# ---------------------------------------------------------------------------
# C.10: filter_docling_document — worker-side v2 quality filter for ALL passes
# ---------------------------------------------------------------------------


#: Label values whose text content is protected from blanking and dedup.
#: Matches docling-graph's own sanitizer (docker/docling-graph/app/main.py:482):
#: image captions carry intentional repetition (figure numbers, "see also" notes)
#: that an aggressive dedup would silently destroy.
_PROTECTED_LABELS: frozenset[str] = frozenset({"caption"})


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
    protected_captions: int = 0


def filter_docling_document(doc_json: dict) -> tuple[dict, FilterDiagnostics]:
    """Apply v2 quality filter to a DoclingDocument JSON in place.

    For each entry in ``doc_json["texts"]``:
      * Entries with ``label`` in ``_PROTECTED_LABELS`` (e.g. ``"caption"``)
        are NEVER blanked or deduped. The filter records them in
        ``diag.protected_captions`` and moves on.
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

    Defensive: missing/None/non-list texts and non-dict elements are skipped
    without raising. The worker still wraps the call in try/except for
    catastrophic shapes (see Task 4 / Task 5), but the most common malformed
    cases must not crash this function.
    """
    diag = FilterDiagnostics()
    texts = doc_json.get("texts") or []
    if not isinstance(texts, list):
        return doc_json, diag
    diag.texts_in = len(texts)

    seen_norms: set[str] = set()
    for i, t in enumerate(texts):
        if not isinstance(t, dict):
            continue

        # Caption protection: docling-graph sanitizer parallel (main.py:482).
        # Image-description captions must survive even when their text is
        # short or duplicate across figures.
        label = (t.get("label") or "").lower()
        if label in _PROTECTED_LABELS:
            diag.protected_captions += 1
            continue

        # Defensive: text/orig can be None per observed docling output.
        rendered_raw = t.get("text") if t.get("text") is not None else t.get("orig")
        if rendered_raw is None:
            continue
        rendered = str(rendered_raw).strip()
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

Expected: all green, 17 tests pass.

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

- [ ] **Step 2: Insert the filter call (with defensive try/except)**

Modify to:

```python
from app.services.extraction_chunk_index import build_extraction_index
from app.services.scoped_docling_document import filter_docling_document
doc_json_for_index = _build_docling_document_json(document_id)
try:
    doc_json_for_index, filter_diag_idx = filter_docling_document(doc_json_for_index)
    logger.info(
        "VR: filter_docling_document (index path) run=%s texts_in=%d blanked=%d "
        "(short=%d dedup=%d after_strip=%d) stripped_in_place=%d protected_captions=%d",
        run_id,
        filter_diag_idx.texts_in,
        filter_diag_idx.blanked_short + filter_diag_idx.blanked_dedup + filter_diag_idx.blanked_after_strip,
        filter_diag_idx.blanked_short,
        filter_diag_idx.blanked_dedup,
        filter_diag_idx.blanked_after_strip,
        filter_diag_idx.stripped_in_place,
        filter_diag_idx.protected_captions,
    )
except Exception as exc:
    # Fail-open: a malformed doc must not terminalize the pipeline_run.
    # Proceed to build_extraction_index with the unfiltered doc; the
    # existing in-loop filter inside build_extraction_index will still
    # apply its texts[]-level filter as a second layer.
    logger.warning(
        "VR: filter_docling_document (index path) FAILED run=%s: %r "
        "— proceeding with unfiltered doc",
        run_id, exc,
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
- Modify: `app/workers/pipeline.py` (around line 7491, inside the Celery wrapper task `derive_ontology_graph_pass`)

This is the call site that runs once per pass — narrowed and non-narrowed alike. With this in place, identity passes and system_links finally see the same filtered doc the indexed-narrowing path already saw.

**Important — function identity:** The call site is the Celery wrapper task `derive_ontology_graph_pass` (which calls `_execute_pass_attempt` later in its body around line 7526). Some earlier scoping language in this plan referred to `_execute_pass_attempt`; the actual edit happens BEFORE that call, in the wrapper task body.

**Important — execution order:** The full sequence after this edit is `_build_docling_document_json` → `filter_docling_document` → (optional) `apply_chunk_scope` (with `text_by_ref` override from chunk-scope endpoint). The `text_by_ref` override sources its content from `ExtractionChunk.chunk_text`, which was built from the **already-filtered** doc (Task 4). So the override re-injects the same stripped text the per-pass filter would have produced — the override and filter are coherent, not in conflict.

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

- [ ] **Step 2: Insert the filter call between load and apply_chunk_scope (with defensive try/except + log)**

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
try:
    doc_json, _filter_diag = filter_docling_document(doc_json)
    logger.info(
        "VR: filter_docling_document (per-pass) run=%s pass=%s texts_in=%d "
        "blanked=%d (short=%d dedup=%d after_strip=%d) stripped_in_place=%d "
        "protected_captions=%d",
        run_id, pass_name,
        _filter_diag.texts_in,
        _filter_diag.blanked_short + _filter_diag.blanked_dedup + _filter_diag.blanked_after_strip,
        _filter_diag.blanked_short,
        _filter_diag.blanked_dedup,
        _filter_diag.blanked_after_strip,
        _filter_diag.stripped_in_place,
        _filter_diag.protected_captions,
    )
    # Initialize router_diagnostics to {} if currently None so doc_filter
    # always lands in DB diagnostics — identity and other non-narrowed
    # passes have router_diagnostics=None by default (line 7354) and
    # would otherwise lose this signal silently.
    if router_diagnostics is None:
        router_diagnostics = {}
    else:
        router_diagnostics = dict(router_diagnostics)
    router_diagnostics["doc_filter"] = {
        "texts_in": _filter_diag.texts_in,
        "blanked_short": _filter_diag.blanked_short,
        "blanked_dedup": _filter_diag.blanked_dedup,
        "blanked_after_strip": _filter_diag.blanked_after_strip,
        "stripped_in_place": _filter_diag.stripped_in_place,
        "protected_captions": _filter_diag.protected_captions,
    }
except Exception as exc:
    # Fail-open: a malformed doc must not terminalize the pass.
    # Proceed with the unfiltered doc.
    logger.warning(
        "VR: filter_docling_document (per-pass) FAILED run=%s pass=%s: %r "
        "— proceeding with unfiltered doc",
        run_id, pass_name, exc,
    )
    if router_diagnostics is None:
        router_diagnostics = {}
    else:
        router_diagnostics = dict(router_diagnostics)
    router_diagnostics["doc_filter"] = {"error": str(exc)}

if chunk_scope is not None and chunk_scope.get("mode") == "selected_refs":
    ...
```

(Leave the `apply_chunk_scope` block unchanged — it runs after the filter. The `text_by_ref` override inside `apply_chunk_scope` re-injects the stripped chunk text from ExtractionChunk — which was indexed from the already-filtered doc in Task 4 — so the override is consistent with what the per-pass filter produced.)

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

### Task 6: Keep the in-loop filter as a defense-in-depth second layer (no removal)

**Files:** (none — no code edits)

The original plan called for deleting the in-loop filter in `build_extraction_index`. The gap-analysis review caught a real issue: `filter_docling_document` operates on `texts[]` only. The walker inside `build_extraction_index` also yields **table** and **picture_caption** chunks whose rendered text comes from sources outside `texts[]` (table cells rendered via `_render_table_chunk`, picture caption resolution through `_resolve_caption_ref`).

If we delete the in-loop filter:
- Tables: cell-markdown rendered chunks bypass the v2 filter entirely (tables[].data.grid is not in texts[]).
- Picture captions: text resolved from texts[] WOULD be filtered (because filter_docling_document hits texts[]), but the rendered picture chunk text concatenates the resolved caption with other metadata — there's no guarantee the in-loop filter's text-level rules would still apply.

**Decision: keep the in-loop filter unchanged.** The two layers are intentionally redundant:

| Layer | Scope | Applies to |
|---|---|---|
| `filter_docling_document` (worker) | `texts[]` entries | All passes (identity + narrowed + system_links) |
| In-loop filter (build_extraction_index) | Rendered chunk text (any modality) | Narrowed passes only (via ExtractionChunk pool) |

The double-pass is operationally cheap (the in-loop filter sees already-blanked text and short-circuits at the existing `if not rendered.strip(): continue` check for blanked text-modality chunks; it does real work for table-modality and picture-caption chunks).

- [ ] **Step 1: Verify both layers coexist cleanly**

Read `app/services/extraction_chunk_index.py` and confirm:
- The `_normalize_for_dedup`, `_strip_chrome_lines` aliases imported from `chunk_quality` in Task 2 are still wired correctly.
- The walk loop's existing `if not rendered.strip(): continue` (skips already-blanked text chunks) is still in place.
- The v2 filter branches (short / after-strip / dedup) inside the walk loop are unchanged from the current code.

- [ ] **Step 2: Run the test suite as a smoke check**

```bash
docker run --rm \
  -v "$(pwd)/app:/app/app" \
  -v "$(pwd)/tests:/app/tests" \
  -v "$(pwd)/ontology_bundles:/app/ontology_bundles" \
  -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
  -w /app eip-mmdpp-api:latest \
  python -m pytest tests/unit/test_extraction_chunk_index.py \
                   tests/unit/test_filter_docling_document.py \
                   tests/unit/test_chunk_quality.py -q
```

Expected: all green. No test files modified in this task.

- [ ] **Step 3: No commit (no code changes)**

This task is documentation-only — it codifies the layering decision. Move on to Task 7.

---

### Task 7: Audit + cross-reference test fixtures (no schema changes)

**Files:** (audit only)

Since Task 6 keeps the in-loop filter, `BuildIndexDiagnostics`'s v2 fields stay. This task becomes a lightweight verification that the two filter layers don't double-count in ways that confuse operators reading logs.

- [ ] **Step 1: Spot-check no double-counted blanks**

Manually trace one chunk through both layers for confidence:
- Input: a duplicate page-header chunk like `"S-75 / Combat Imagery / / 10/6/25"`
- `filter_docling_document` (worker, Task 4) sees it via texts[], blanks it (text=""), increments `diag.blanked_dedup`
- `build_extraction_index` (Task 6 — unchanged) walks the chunk, calls `_render_text_chunk` which returns "", hits `if not rendered.strip(): continue`, increments `chunks_skipped` BUT not `chunks_skipped_duplicate` (because the explicit dedup branch never executes — empty rendered text bails first)

So `diag.blanked_dedup` (from filter_docling_document) and `chunks_skipped_duplicate` (from build_extraction_index) count DIFFERENT things: the former is the worker-boundary dedup; the latter is the index-time dedup that only fires for non-text-modality chunks (table/picture-caption rendered text that wasn't pre-filtered). No double-counting.

- [ ] **Step 2: Verify the documented behavior in the build_extraction_index log**

Read the existing `logger.info("build_extraction_index: ...")` line. Confirm the counters it reports describe the in-loop layer only (it doesn't reach into FilterDiagnostics from filter_docling_document). Both log lines emit independently — operators see two filtering signals per pass.

- [ ] **Step 3: No commit (no code changes)**

Document-only.

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

echo "--- Per-pass router.doc_filter (worker writes router_diagnostics under the 'router' key) ---"
docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA -c "
  SELECT pass_name, jsonb_pretty(diagnostics_json->'router'->'doc_filter')
  FROM ingest.pipeline_pass_outputs
  WHERE pipeline_run_id='$RUN_ID'
  ORDER BY pass_name;"
```

Expected:
- 1 "filter_docling_document (index path)" log line per run start.
- 1 build_extraction_index log line WITHOUT the old `(short=X dup=Y web_chrome=Z)` suffix.
- Each pass's `router_diagnostics.doc_filter` populated with non-zero counters (the per-pass filter is in addition to the index-path filter — they should produce similar numbers).

- [ ] **Step 5: Compare entity counts AND field density to baseline (no regression)**

```bash
docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA -F'|' -c "
  SELECT pass_name, execution_status, primary_entities_extracted
  FROM ingest.pipeline_pass_outputs
  WHERE pipeline_run_id='$RUN_ID'
  ORDER BY pass_name;"

echo --- Field density per entity for narrowed passes ---
# Dump pass_output JSON and tally non-null spec fields per entity
for PASS in radar_identity radar_power_rf missile_identity missile_kinematics; do
  docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA -c "
    SELECT jsonb_pretty(extract_pass_response_json->'pass_output')
    FROM ingest.pipeline_pass_outputs
    WHERE pipeline_run_id='$RUN_ID' AND pass_name='$PASS'
    ORDER BY attempt DESC LIMIT 1;" > "/tmp/c10_${PASS}.json"
  echo "  $PASS: $(wc -l < /tmp/c10_${PASS}.json) lines"
done
```

Compare against the **Task 0 baseline file `/tmp/pre_refactor_baseline.txt`**, NOT against any externally-named run UUID. The baseline captured in Task 0 is the canonical reference for this validation; that file records the state immediately before this refactor began.

Acceptance criteria (against the Task 0 baseline):
- Per-pass entity counts within ±10% of baseline OR within ±1 entity if baseline count ≤ 10
- Identity-pass (radar_identity, missile_identity) field density per entity should be ≥ baseline. If it drops, captions or page-header context may have been over-filtered — investigate.
- radar_power_rf may still be FAILED (Dvina has no RF data — known invariant from prior investigation, NOT caused by this refactor)
- system_links: COMPLETE; relationships_extracted within ±1 of baseline

Big regressions (e.g. all passes returning 0) mean the filter is too aggressive on the doc-level path; rollback per the "Rollback runbook" section below.

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

- **Interaction with the chunked-batches patch.** docling-graph's `docker/docling-graph/patches/0002-stages-chunked-batches-for-docling-document-input.patch` routes delta-contract docs through a chunker that consumes the rendered markdown. After C.10, that chunker sees a DoclingDocument with some `texts[]` entries blanked (text="", orig="", hyperlink=None). The existing `_sanitize_docling_document` in docling-graph already produces this exact shape (it has been blanking nav/tracking texts the same way for months), so the chunked-batches path is already known to tolerate it. No code change required, but the implementer should be aware that the patched path runs over an already-blanked doc when DOCLING_GRAPH_SANITIZE_INPUT=true (default).

- **The index-path filter (Task 4) writes to logs only.** Unlike the per-pass site (Task 5), there's no `router_diagnostics` channel into `pipeline_pass_outputs` at the index-build call site — the index is built once per run, before any pass executes. A future plan could persist the index-path FilterDiagnostics under `pipeline_runs.metrics.doc_filter_index` for postmortem visibility; not in scope here.

---

## Task dependencies

For subagent-driven execution, set `blockedBy` relationships:

| Task | Depends on (blockedBy) |
|---|---|
| Task 0: Pre-flight | — |
| Task 1: Create chunk_quality.py | Task 0 |
| Task 2: Refactor extraction_chunk_index to import | Task 1 |
| Task 3: Add filter_docling_document | Task 1 |
| Task 4: Wire filter at build_extraction_index | Task 3 |
| Task 5: Wire filter at per-pass doc load | Task 3 |
| Task 6: Verify in-loop filter layering (no-op task) | Tasks 4, 5 |
| Task 7: Counter audit (no-op task) | Task 6 |
| Task 8: Dvina smoke | Tasks 4, 5, 6 |
| Task 9: SA-2 validation (optional) | Task 8 |
| Task 10: Final test sweep | Task 9 (or Task 8 if Task 9 skipped) |

---

## Rollback runbook

If Task 8 or Task 9 shows quality regression beyond the acceptance criteria — for example, identity-pass entity counts drop by >10%, field density per entity drops noticeably, or downstream stages fail unexpectedly — follow these exact steps:

1. **Stop any in-flight reingest.** Check for active runs first; do NOT use the `/cancel` endpoint (it hard-deletes; see `[[cancel-endpoint-hard-deletes]]`).
   ```bash
   docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tA -F'|' -c \
     "SELECT id, status FROM ingest.pipeline_runs WHERE status='PROCESSING' AND started_at >= NOW() - INTERVAL '1 hour';"
   ```
   Wait for any active run to terminalize naturally.

2. **Reset the worktree to the pre-C.10 commit (with safety branch).**
   ```bash
   git status                              # confirm no uncommitted work to save
   git branch c10-rollback-snapshot        # snapshot current HEAD before destroying it
   git reset --hard 0290748
   ```
   The snapshot branch `c10-rollback-snapshot` preserves the failed-C.10 commits in case any artifact from them (e.g. an investigation script, a partial test) is worth recovering later. Commit `0290748` was created as the explicit rollback target before any C.10 work began.

3. **Rebuild + force-recreate the services that load the rolled-back code.**
   The worker-graph + api services bind-mount `app/`; the docling-graph service builds via COPY. Filter changes here do not touch docling-graph, so only the bind-mount services need recreation:
   ```bash
   COMPOSE_PROJECT_NAME=eip-mmdpp docker compose up -d --force-recreate api worker-graph
   ```

4. **Purge stale ExtractionChunks from failed runs.**
   `cleanup_extraction_index(run_id, store)` is defined at `app/services/extraction_chunk_index.py:943` and is synchronous (no asyncio).
   ```bash
   FAILED_RUN_ID=<the regression run UUID from Task 8 or 9>
   docker exec eip-mmdpp-api-1 python -c "
   from app.db.session import get_graph_store
   from app.services.extraction_chunk_index import cleanup_extraction_index
   store = get_graph_store()
   deleted = cleanup_extraction_index('$FAILED_RUN_ID', store=store)
   print(f'Deleted {deleted} ExtractionChunk rows for run $FAILED_RUN_ID')
   "
   ```

5. **Confirm rollback worked.**
   ```bash
   git log --oneline -3            # latest commit should be 0290748
   docker run --rm \
     -v "$(pwd)/app:/app/app" \
     -v "$(pwd)/tests:/app/tests" \
     -v "$(pwd)/ontology_bundles:/app/ontology_bundles" \
     -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
     -w /app eip-mmdpp-api:latest \
     python -m pytest tests/unit/test_extraction_chunk_index.py -q
   ```
   Expected: 66/66 (pre-refactor count). If tests fail, the rollback didn't take — investigate before re-attempting.

6. **Trigger a baseline reingest to verify rolled-back behavior.**
   ```bash
   DVINA_DOC="9c8e09c7-e39f-4359-92c0-46330158c73c"
   curl -sX POST "http://localhost:8005/v1/documents/$DVINA_DOC/reingest" \
     -H "Content-Type: application/json" \
     -d '{"mode":"graph_only","ontology_bundle_key":"air_defense_v3_narrowing_v1"}'
   ```
   Wait for it to terminalize. Confirm entity counts match the Task 0 baseline.

If the rollback step fails or leaves the system in an inconsistent state, the operator should escalate rather than attempting destructive operations (`/cancel`, `git reset --hard` to deeper commits, manual ArcadeDB DELETE, etc.).
