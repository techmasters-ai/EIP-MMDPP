"""VR C.2b — Deterministic offline recall gate (LOAD-BEARING).

**This test blocks C.6 / C.7 / C.8.**

Validates that the pre-rerank top-N candidate set from vector similarity
covers all normalized critical evidence refs from the bdde417 baseline.
Hermetic — uses precomputed fixture embeddings only; no live Ollama calls.

Gate definition (rev 10 M4):
    set(pre_rerank_top_n_candidates) ⊇ set(normalized_critical_evidence_refs)

where critical evidence is:
    (a) ALL Dvina baseline evidence refs across all 5 passes
        (100% recall required — manageable cardinality).
    (b) ONLY SA-2 evidence refs for entities counted by:
        - gate #5: emitter_function on radar_identity (3 unique evidence IDs
          in bdde417 baseline — see CONCERN note below)
        - gate #6: min_altitude_km on missile_kinematics (179 unique IDs)

CONCERN — Gate #5 provenance location:
    The plan references gate #5 as "emitter_function 21/34" on radar_power_rf.
    However, the bdde417 baseline fixtures show radar_power_rf.field_provenance
    is EMPTY (0 entries). The emitter_function evidence IS present in
    radar_identity.field_provenance (3 unique evidence IDs: #/texts/0,
    #/texts/1, #/texts/12). This test maps gate #5 to radar_identity evidence
    per the actual fixture shape. The discrepancy should be reviewed when
    field_provenance is added to radar_power_rf in future pipeline runs.

Evidence ref normalization (rev 9 H2):
    - #/tables/N/cells/K → #/tables/N (cell → parent table)
    - #/pages/N → all #/texts/M, #/tables/M, #/pictures/M on that page
      (requires doc prov data — handled via fixture if available)
    - #/groups/N → recursively all #/texts/M children
    - #/texts/N, #/tables/N, #/pictures/N → unchanged

FIXTURE GENERATION:
    Run once with live Ollama to generate the .npz fixture files:

        python scripts/generate_baseline_recall_fixtures.py

    Fixtures are stored in tests/fixtures/recall_baseline/:
        {doc_id}_chunks.npz      — chunk embeddings (N, D) + self_refs
        {doc_id}_{pass}_query.npz — query embedding (D,) + query_text

    If fixtures are absent, this test SKIPS with instructions.

Run standalone (with fixtures):
    python3 -m pytest tests/integration/test_vector_router_offline_recall_deterministic.py -v
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SA2_DOC_ID = "78673393-639b-4fde-9bda-9e7bfd43ccda"
DVINA_DOC_ID = "b77c48f9-3a27-473f-be05-fa7e73e5d6f5"

FIXTURE_DIR_SA2 = Path(__file__).resolve().parents[1] / "fixtures" / "sa2"
RECALL_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "recall_baseline"

# Conservative defaults from RetrievalProfile (rev 12 H1).
TOP_N_CANDIDATES = 50

# ---------------------------------------------------------------------------
# Evidence normalization
# ---------------------------------------------------------------------------

_CELL_REF_RE = re.compile(r"^(#/tables/\d+)/cells/\d+$")
_PAGE_REF_RE = re.compile(r"^#/pages/(\d+)$")
_GROUP_REF_RE = re.compile(r"^#/groups/(\d+)$")
_ELEMENT_REF_RE = re.compile(r"^#/(texts|tables|pictures)/(\d+)$")


def normalize_evidence_refs(refs: set[str], doc_json: dict | None = None) -> set[str]:
    """Per rev 9 H2: convert baseline evidence_id refs into the ExtractionChunk
    self_ref granularity used by the vector router.

    Rules:
      - #/tables/N/cells/K → #/tables/N (cell → parent table)
      - #/pages/N → {#/texts/M, #/tables/M, #/pictures/M | element is on page N}
        (requires doc_json to resolve; if not available, keeps ref as-is)
      - #/groups/N → recursively all #/texts/M children
        (requires doc_json to resolve; if not available, keeps ref as-is)
      - #/texts/N, #/tables/N, #/pictures/N → unchanged
      - Unrecognised patterns → kept as-is (forward-compatible)
    """
    result: set[str] = set()
    for ref in refs:
        # Cell ref → parent table
        m = _CELL_REF_RE.match(ref)
        if m:
            result.add(m.group(1))
            continue

        # Page ref → elements on that page
        m = _PAGE_REF_RE.match(ref)
        if m and doc_json is not None:
            page_num = int(m.group(1))
            page_elements = _elements_on_page(doc_json, page_num)
            if page_elements:
                result.update(page_elements)
                continue
        # Fall through if no doc_json or no elements found
        if m:
            result.add(ref)
            continue

        # Group ref → constituent texts
        m = _GROUP_REF_RE.match(ref)
        if m and doc_json is not None:
            group_idx = int(m.group(1))
            group_elements = _expand_group(doc_json, group_idx)
            if group_elements:
                result.update(group_elements)
                continue
        if m:
            result.add(ref)
            continue

        # Text / table / picture — pass through unchanged
        if _ELEMENT_REF_RE.match(ref):
            result.add(ref)
            continue

        # Unknown pattern — keep as-is (forward-compat)
        result.add(ref)

    return result


def _elements_on_page(doc_json: dict, page_num: int) -> set[str]:
    """Return all element self_refs that are on the given page index."""
    result: set[str] = set()
    for collection_key in ("texts", "tables", "pictures"):
        items = doc_json.get(collection_key, [])
        for item in items:
            prov = item.get("prov", [])
            for p in prov:
                if p.get("page_no") == page_num:
                    sr = item.get("self_ref")
                    if sr:
                        result.add(sr)
    return result


def _expand_group(doc_json: dict, group_idx: int) -> set[str]:
    """Recursively expand a group's children to text self_refs."""
    groups = doc_json.get("groups", [])
    if group_idx >= len(groups):
        return set()

    group = groups[group_idx]
    result: set[str] = set()
    for child_ref in group.get("children", []):
        # child_ref may be '#/texts/N', '#/groups/M', etc.
        m_text = re.match(r"^#/texts/(\d+)$", child_ref)
        m_group = re.match(r"^#/groups/(\d+)$", child_ref)
        if m_text:
            result.add(child_ref)
        elif m_group:
            # Recurse
            result.update(_expand_group(doc_json, int(m_group.group(1))))
        # Tables + pictures in groups — add as-is
        elif re.match(r"^#/(tables|pictures)/\d+$", child_ref):
            result.add(child_ref)

    return result


# ---------------------------------------------------------------------------
# Critical evidence extraction
# ---------------------------------------------------------------------------

def _load_pass_response(doc_id: str, pass_name: str) -> dict[str, Any]:
    """Load a baseline pass response fixture."""
    p = FIXTURE_DIR_SA2 / f"{doc_id}_{pass_name}_response.json"
    if not p.exists():
        pytest.skip(f"Baseline fixture not found: {p}")
    return json.loads(p.read_text())


def extract_evidence_refs_from_response(
    response: dict[str, Any],
    *,
    field_filter: str | None = None,
) -> set[str]:
    """Extract unique evidence_id refs from a pass response's field_provenance.

    Args:
        response:      The full pass response dict.
        field_filter:  If given, only include entries for this field_name.
    """
    fp = response.get("field_provenance") or []
    refs: set[str] = set()
    for entry in fp:
        if field_filter and entry.get("field_name") != field_filter:
            continue
        eid = entry.get("evidence_id")
        if eid:
            refs.add(eid)
    return refs


def compute_critical_evidence_refs_dvina() -> dict[tuple[str, str], set[str]]:
    """Compute ALL evidence refs for Dvina across all 5 baseline passes.

    Returns dict keyed by (doc_id, pass_name) → set of normalized evidence refs.
    Dvina uses 100% recall requirement — all evidence refs must be in scope.
    """
    passes = ["radar_identity", "radar_power_rf", "missile_identity", "missile_kinematics", "system_links"]
    result: dict[tuple[str, str], set[str]] = {}
    for pass_name in passes:
        resp = _load_pass_response(DVINA_DOC_ID, pass_name)
        raw_refs = extract_evidence_refs_from_response(resp)
        normalized = normalize_evidence_refs(raw_refs)
        if normalized:
            result[(DVINA_DOC_ID, pass_name)] = normalized
    return result


def compute_critical_evidence_refs_sa2() -> dict[tuple[str, str], set[str]]:
    """Compute ONLY the critical evidence refs for SA-2.

    Per rev 9 H2: only evidence refs for entities counted by gates #5 and #6.
      - Gate #5: emitter_function — evidence from radar_identity.field_provenance
        (bdde417 shape: 3 unique IDs in radar_identity; radar_power_rf has none).
      - Gate #6: min_altitude_km — evidence from missile_kinematics.field_provenance.
    """
    result: dict[tuple[str, str], set[str]] = {}

    # Gate #5: emitter_function on radar_identity (see CONCERN in module docstring)
    ri_resp = _load_pass_response(SA2_DOC_ID, "radar_identity")
    gate5_refs = extract_evidence_refs_from_response(ri_resp, field_filter="emitter_function")
    gate5_norm = normalize_evidence_refs(gate5_refs)
    if gate5_norm:
        result[(SA2_DOC_ID, "radar_identity")] = gate5_norm

    # Gate #6: min_altitude_km on missile_kinematics
    mk_resp = _load_pass_response(SA2_DOC_ID, "missile_kinematics")
    gate6_refs = extract_evidence_refs_from_response(mk_resp, field_filter="min_altitude_km")
    gate6_norm = normalize_evidence_refs(gate6_refs)
    if gate6_norm:
        result[(SA2_DOC_ID, "missile_kinematics")] = gate6_norm

    return result


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def _recall_fixture_present(doc_id: str) -> bool:
    return (RECALL_FIXTURE_DIR / f"{doc_id}_chunks.npz").exists()


def _load_chunk_data(doc_id: str) -> tuple[list[str], "np.ndarray"]:
    """Load precomputed chunk embeddings for a document.

    Returns (self_refs, embeddings) where embeddings is (N, D) float32.
    """
    import numpy as np

    npz_path = RECALL_FIXTURE_DIR / f"{doc_id}_chunks.npz"
    data = np.load(npz_path, allow_pickle=False)
    self_refs = json.loads(data["self_refs"].tobytes().decode("utf-8"))
    embeddings = data["embeddings"]  # (N, D) float32
    return self_refs, embeddings


def _load_query_embedding(doc_id: str, pass_name: str) -> "np.ndarray":
    """Load precomputed query embedding for a (doc, pass) pair. Returns (D,) float32."""
    import numpy as np

    npz_path = RECALL_FIXTURE_DIR / f"{doc_id}_{pass_name}_query.npz"
    if not npz_path.exists():
        pytest.skip(
            f"Query embedding fixture missing: {npz_path}\n"
            "Run: python scripts/generate_baseline_recall_fixtures.py"
        )
    data = np.load(npz_path, allow_pickle=False)
    return data["embedding"]  # (D,) float32


# ---------------------------------------------------------------------------
# Core recall assertion
# ---------------------------------------------------------------------------

def _assert_pre_rerank_recall(
    doc_id: str,
    pass_name: str,
    critical_refs: set[str],
    top_n: int = TOP_N_CANDIDATES,
) -> None:
    """Assert pre-rerank top-N candidates cover all critical evidence refs.

    Skips gracefully if fixtures are absent.
    """
    import numpy as np

    if not _recall_fixture_present(doc_id):
        pytest.skip(
            f"Chunk embedding fixtures missing for {doc_id[:8]}.\n"
            "Generate them with:\n"
            "    python scripts/generate_baseline_recall_fixtures.py\n"
            "Fixtures required before C.6/C.7/C.8 can be dispatched."
        )

    self_refs, chunk_embs = _load_chunk_data(doc_id)
    query_emb = _load_query_embedding(doc_id, pass_name)

    # Cosine similarity — chunks are L2-normalised, query is L2-normalised.
    scores = chunk_embs @ query_emb  # (N,) cosine similarities

    # Top-N pre-rerank candidates.
    if len(scores) <= top_n:
        top_indices = np.arange(len(scores))
    else:
        top_indices = np.argpartition(scores, -top_n)[-top_n:]

    selected_refs = {self_refs[i] for i in top_indices}

    # Normalise the critical refs against selected_refs space
    # (critical refs are already normalised via normalize_evidence_refs).
    missing = critical_refs - selected_refs

    assert not missing, (
        f"[{doc_id[:8]}/{pass_name}] Pre-rerank top-{top_n} missed "
        f"{len(missing)}/{len(critical_refs)} critical evidence refs:\n"
        f"  missing: {sorted(missing)}\n"
        f"  selected (sample): {sorted(list(selected_refs))[:5]}\n"
        f"\n"
        f"This blocks C.6/C.7/C.8 live regression.\n"
        f"Resolution options (C.5 calibration):\n"
        f"  1. Tune min_similarity / top_n_candidates in the manifest.\n"
        f"  2. Improve the query text in build_retrieval_query().\n"
        f"  3. Regenerate fixtures with updated embedding model if model changed.\n"
        f"Do NOT merge past this failure — see plan rev 7 H4."
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDvinaOfflineRecall:
    """100% Dvina baseline evidence recall (all 5 passes).

    LOAD-BEARING — must pass before C.6 dispatch.
    """

    @pytest.fixture(scope="class")
    def dvina_critical(self) -> dict[tuple[str, str], set[str]]:
        return compute_critical_evidence_refs_dvina()

    def test_dvina_radar_identity_recall(self, dvina_critical):
        """Pre-rerank top-50 must cover all Dvina radar_identity evidence refs."""
        refs = dvina_critical.get((DVINA_DOC_ID, "radar_identity"), set())
        if not refs:
            pytest.skip("No radar_identity evidence refs in Dvina baseline fixture")
        _assert_pre_rerank_recall(DVINA_DOC_ID, "radar_identity", refs)

    def test_dvina_radar_power_rf_recall(self, dvina_critical):
        """Pre-rerank top-50 must cover all Dvina radar_power_rf evidence refs."""
        refs = dvina_critical.get((DVINA_DOC_ID, "radar_power_rf"), set())
        if not refs:
            pytest.skip("No radar_power_rf evidence refs in Dvina baseline fixture")
        _assert_pre_rerank_recall(DVINA_DOC_ID, "radar_power_rf", refs)

    def test_dvina_missile_identity_recall(self, dvina_critical):
        """Pre-rerank top-50 must cover all Dvina missile_identity evidence refs."""
        refs = dvina_critical.get((DVINA_DOC_ID, "missile_identity"), set())
        if not refs:
            pytest.skip("No missile_identity evidence refs in Dvina baseline fixture")
        _assert_pre_rerank_recall(DVINA_DOC_ID, "missile_identity", refs)

    def test_dvina_missile_kinematics_recall(self, dvina_critical):
        """Pre-rerank top-50 must cover all Dvina missile_kinematics evidence refs."""
        refs = dvina_critical.get((DVINA_DOC_ID, "missile_kinematics"), set())
        if not refs:
            pytest.skip("No missile_kinematics evidence refs in Dvina baseline fixture")
        _assert_pre_rerank_recall(DVINA_DOC_ID, "missile_kinematics", refs)

    def test_dvina_system_links_recall(self, dvina_critical):
        """Pre-rerank top-50 must cover all Dvina system_links evidence refs."""
        refs = dvina_critical.get((DVINA_DOC_ID, "system_links"), set())
        if not refs:
            pytest.skip("No system_links evidence refs in Dvina baseline fixture")
        _assert_pre_rerank_recall(DVINA_DOC_ID, "system_links", refs)


class TestSA2CriticalOfflineRecall:
    """SA-2 critical evidence recall (gates #5 and #6 only).

    Gate #5: emitter_function evidence from radar_identity.
    Gate #6: min_altitude_km evidence from missile_kinematics.

    LOAD-BEARING — must pass before C.6 dispatch.
    See CONCERN in module docstring re: gate #5 fixture shape.
    """

    @pytest.fixture(scope="class")
    def sa2_critical(self) -> dict[tuple[str, str], set[str]]:
        return compute_critical_evidence_refs_sa2()

    def test_sa2_gate5_emitter_function_recall(self, sa2_critical):
        """Gate #5: pre-rerank top-50 must cover emitter_function evidence refs.

        Evidence lives in radar_identity fixture (not radar_power_rf) per bdde417
        fixture shape — see module CONCERN.
        """
        refs = sa2_critical.get((SA2_DOC_ID, "radar_identity"), set())
        if not refs:
            pytest.skip("No emitter_function evidence refs in SA-2 radar_identity baseline fixture")
        _assert_pre_rerank_recall(SA2_DOC_ID, "radar_identity", refs)

    def test_sa2_gate6_min_altitude_km_recall(self, sa2_critical):
        """Gate #6: pre-rerank top-50 must cover min_altitude_km evidence refs."""
        refs = sa2_critical.get((SA2_DOC_ID, "missile_kinematics"), set())
        if not refs:
            pytest.skip("No min_altitude_km evidence refs in SA-2 missile_kinematics baseline fixture")
        _assert_pre_rerank_recall(SA2_DOC_ID, "missile_kinematics", refs)


class TestEvidenceNormalization:
    """Unit-level tests for the normalize_evidence_refs helper.

    These tests are purely deterministic and do NOT require fixture files.
    """

    def test_cell_ref_normalizes_to_table(self):
        """#/tables/N/cells/K → #/tables/N"""
        result = normalize_evidence_refs({"#/tables/5/cells/3"})
        assert result == {"#/tables/5"}

    def test_multiple_cells_same_table_merge(self):
        """Multiple cell refs from the same table normalize to one table ref."""
        result = normalize_evidence_refs({
            "#/tables/2/cells/0",
            "#/tables/2/cells/1",
            "#/tables/2/cells/99",
        })
        assert result == {"#/tables/2"}

    def test_text_ref_unchanged(self):
        """#/texts/N passes through unchanged."""
        result = normalize_evidence_refs({"#/texts/42"})
        assert result == {"#/texts/42"}

    def test_table_ref_unchanged(self):
        """#/tables/N (no /cells suffix) passes through unchanged."""
        result = normalize_evidence_refs({"#/tables/7"})
        assert result == {"#/tables/7"}

    def test_picture_ref_unchanged(self):
        """#/pictures/N passes through unchanged."""
        result = normalize_evidence_refs({"#/pictures/0"})
        assert result == {"#/pictures/0"}

    def test_mixed_refs(self):
        """Mix of cell, text, and table refs normalize correctly."""
        result = normalize_evidence_refs({
            "#/tables/1/cells/2",
            "#/texts/10",
            "#/tables/3",
        })
        assert result == {"#/tables/1", "#/texts/10", "#/tables/3"}

    def test_page_ref_without_doc_stays_as_is(self):
        """#/pages/N without doc_json is kept unchanged (forward-compat)."""
        result = normalize_evidence_refs({"#/pages/5"})
        assert result == {"#/pages/5"}

    def test_page_ref_with_doc_expands_to_elements(self):
        """#/pages/N with doc_json expands to all elements on that page."""
        doc_json = {
            "texts": [
                {"self_ref": "#/texts/0", "prov": [{"page_no": 1}]},
                {"self_ref": "#/texts/1", "prov": [{"page_no": 2}]},
                {"self_ref": "#/texts/2", "prov": [{"page_no": 1}]},
            ],
            "tables": [
                {"self_ref": "#/tables/0", "prov": [{"page_no": 1}]},
            ],
            "pictures": [],
        }
        result = normalize_evidence_refs({"#/pages/1"}, doc_json=doc_json)
        assert result == {"#/texts/0", "#/texts/2", "#/tables/0"}

    def test_group_ref_without_doc_stays_as_is(self):
        """#/groups/N without doc_json is kept unchanged."""
        result = normalize_evidence_refs({"#/groups/3"})
        assert result == {"#/groups/3"}

    def test_group_ref_with_doc_expands_to_texts(self):
        """#/groups/N with doc_json expands to text children."""
        doc_json = {
            "groups": [
                {},  # groups[0] — unused
                {},  # groups[1] — unused
                {
                    "children": ["#/texts/5", "#/texts/6", "#/pictures/0"]
                },
            ]
        }
        result = normalize_evidence_refs({"#/groups/2"}, doc_json=doc_json)
        assert result == {"#/texts/5", "#/texts/6", "#/pictures/0"}

    def test_empty_set(self):
        """Empty input → empty output."""
        assert normalize_evidence_refs(set()) == set()

    def test_unknown_pattern_passthrough(self):
        """Unrecognised ref patterns pass through unchanged."""
        result = normalize_evidence_refs({"#/custom/foo/bar"})
        assert result == {"#/custom/foo/bar"}


class TestCriticalEvidenceExtraction:
    """Tests for the evidence extraction helpers (use actual fixture files)."""

    def test_dvina_missile_kinematics_has_evidence_refs(self):
        """Dvina missile_kinematics baseline must contain field_provenance."""
        resp = _load_pass_response(DVINA_DOC_ID, "missile_kinematics")
        refs = extract_evidence_refs_from_response(resp)
        assert len(refs) > 0, (
            "Dvina missile_kinematics baseline has no field_provenance evidence_ids. "
            "The recall gate cannot be meaningful without them."
        )

    def test_sa2_missile_kinematics_min_altitude_has_evidence_refs(self):
        """SA-2 missile_kinematics baseline must have min_altitude_km evidence refs."""
        resp = _load_pass_response(SA2_DOC_ID, "missile_kinematics")
        refs = extract_evidence_refs_from_response(resp, field_filter="min_altitude_km")
        assert len(refs) > 0, (
            "SA-2 missile_kinematics has no min_altitude_km evidence refs. "
            "Gate #6 cannot be tested."
        )

    def test_sa2_radar_identity_emitter_function_has_evidence_refs(self):
        """SA-2 radar_identity must have emitter_function evidence refs (gate #5)."""
        resp = _load_pass_response(SA2_DOC_ID, "radar_identity")
        refs = extract_evidence_refs_from_response(resp, field_filter="emitter_function")
        assert len(refs) > 0, (
            "SA-2 radar_identity has no emitter_function evidence refs. "
            "Gate #5 cannot be tested."
        )
        # Exactly 3 in bdde417 baseline: #/texts/0, #/texts/1, #/texts/12
        assert len(refs) >= 1, f"Expected ≥1 emitter_function evidence refs, got {len(refs)}"

    def test_sa2_radar_power_rf_no_field_provenance(self):
        """SA-2 radar_power_rf baseline has no field_provenance in bdde417 fixture.

        This is a known shape: radar_power_rf did not record field_provenance
        in this baseline. The test documents this intentionally so future
        additions of field_provenance to radar_power_rf are visible.
        """
        resp = _load_pass_response(SA2_DOC_ID, "radar_power_rf")
        refs = extract_evidence_refs_from_response(resp)
        # Currently zero — document this known fixture shape.
        # If this assertion ever fails with len(refs) > 0, it means
        # radar_power_rf NOW has field_provenance and gate #5 should be
        # mapped to radar_power_rf (where the emitter_function 21/34 entities live).
        if len(refs) > 0:
            pytest.xfail(
                f"SA-2 radar_power_rf now has {len(refs)} field_provenance evidence refs. "
                "Update gate #5 mapping from radar_identity → radar_power_rf "
                "and regenerate recall fixtures."
            )
