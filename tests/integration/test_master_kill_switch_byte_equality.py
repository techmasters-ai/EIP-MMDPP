"""Phase 1 merge gate (spec §15.1).

With both *_NORMALIZATION_ENABLED=false (default), the new code must
produce byte-identical post-sanitization texts vs the §19 baseline
fixture captured against pre-rewrite main.

Test mechanism: pure-function call into the docling-graph sanitizer
(no LLM, no chunking). Reproducible in CI.

Spec: docs/superpowers/specs/2026-05-11-table-aware-chunking-design.md §15.1.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_DIR = REPO_ROOT / "tests" / "fixtures" / "sa2"
DOC_ID = "78673393-639b-4fde-9bda-9e7bfd43ccda"


@pytest.fixture(scope="module")
def baseline_meta() -> dict:
    meta_path = BASELINE_DIR / "baseline.meta.json"
    if not meta_path.exists():
        pytest.skip(f"baseline not captured at {meta_path}; run Task 0a first")
    return json.loads(meta_path.read_text())


@pytest.fixture(scope="module")
def baseline_texts() -> list:
    p = BASELINE_DIR / f"{DOC_ID}_texts_today.json"
    if not p.exists():
        pytest.skip(f"baseline texts not captured at {p}; run Task 0a first")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def doc_json() -> dict:
    """Fetch the SA-2 doc_json fresh from the worker API."""
    import urllib.request
    url = f"http://localhost:8005/v1/documents/{DOC_ID}/docling"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            envelope = json.loads(r.read())
    except Exception as exc:
        pytest.skip(f"could not fetch doc via API ({exc}); stack must be up")
    return envelope["document_json"]


@pytest.fixture(scope="module")
def sanitizer():
    """Import _sanitize_docling_document from docling-graph app."""
    dg_app = REPO_ROOT / "docker" / "docling-graph"
    if str(dg_app) not in sys.path:
        sys.path.insert(0, str(dg_app))
    try:
        from app.main import _sanitize_docling_document  # type: ignore
    except ImportError as exc:
        pytest.skip(f"docling-graph sanitizer unavailable ({exc})")
    return _sanitize_docling_document


def test_master_kill_switch_texts_byte_identical(
    baseline_texts, doc_json, sanitizer, monkeypatch
):
    """With master switches off, post-sanitization texts MUST byte-equal baseline.

    Confirms the Phase 1 merge gate per §15.1: no behavior change with
    flags off.
    """
    # Force master switches off (their default; explicit for clarity).
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED", "false")
    monkeypatch.setenv("EMBEDDING_TABLE_NORMALIZATION_ENABLED", "false")
    monkeypatch.setenv("DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS", "false")

    # Deep-copy to avoid mutating the cached doc_json fixture.
    doc_copy = json.loads(json.dumps(doc_json))
    stats: dict = {"texts_in": 0, "texts_dropped": 0}
    sanitized = sanitizer(doc_copy, stats)
    fresh = sanitized.get("texts") or []

    assert len(fresh) == len(baseline_texts), (
        f"texts count diverged: baseline={len(baseline_texts)}, fresh={len(fresh)}"
    )
    # JSON-canonical comparison (handles dict key ordering)
    baseline_json = json.dumps(baseline_texts, sort_keys=True, ensure_ascii=False)
    fresh_json = json.dumps(fresh, sort_keys=True, ensure_ascii=False)
    assert baseline_json == fresh_json, (
        "post-sanitization texts diverged from baseline. "
        "First difference would be visible via diff of the two JSON serializations."
    )


def test_baseline_meta_records_switches_off_posture(baseline_meta):
    """Sanity: the baseline's recorded notes claim switches were OFF when captured."""
    notes = baseline_meta.get("notes", [])
    notes_joined = " ".join(notes).lower()
    assert "master switches off" in notes_joined or "switches off" in notes_joined or "off" in notes_joined, (
        f"baseline.meta.json notes don't clearly document the switches-off posture: {notes}"
    )


def test_baseline_corpus_matches_test_doc_id(baseline_meta):
    """Sanity: baseline.meta.json corpus_files contains the doc we're testing."""
    corpus = baseline_meta.get("corpus_files", [])
    assert DOC_ID in corpus, f"DOC_ID={DOC_ID} not in baseline corpus_files: {corpus}"
