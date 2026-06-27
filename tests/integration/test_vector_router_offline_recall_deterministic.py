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
        - gate #5: emitter_function on radar_power_rf (rev 15 Option B —
          identity-names anchor; see critical_evidence_refs_via_identity_names)
        - gate #6: min_altitude_km on missile_kinematics (rev 15 Option B —
          identity-names anchor)

Gate #5/#6 evidence anchoring (rev 15 Option B):
    emitter_function is a PROPAGATED field (Step 2 propagation copies from
    radar_identity onto radar_power_rf entities via system_name match).
    The bdde417 baseline records no field_provenance for radar_power_rf
    because the field isn't directly extracted there — it's propagated.

    The semantically correct VR recall gate must verify that radar_power_rf's
    narrowed input still contains chunks mentioning the radar system_names
    that radar_identity extracted. Without those chunks, radar_power_rf
    produces no entity to propagate onto.

    critical_evidence_refs_via_identity_names() implements this:
      - loads the upstream identity pass's baseline entities
      - extracts system_names + nomenclature strings
      - searches the doc's text fixtures for case-insensitive matches
      - returns the set of self_refs containing any identity name

    This test catches "field_group narrowing loses identity-name chunks" —
    exactly the failure mode that breaks gate #5 / #6 silently.

    Mapping (rev 15):
      - radar_power_rf   ← radar_identity   (entity system_names)
      - missile_kinematics ← missile_identity (entity system_names)
    Same logic for Dvina: both field_group passes use identity-name anchors.
    Identity passes themselves (radar_identity, missile_identity) keep their
    field_provenance-based recall — they extract their own fields directly.

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
# FIXTURE_ROOT is an alias for FIXTURE_DIR_SA2; both SA-2 and Dvina fixtures
# reside here.  Used by critical_evidence_refs_via_identity_names().
FIXTURE_ROOT = FIXTURE_DIR_SA2
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

def _load_pass_response(doc_id: str, pass_name: str, fixture_dir: Path | None = None) -> dict[str, Any]:
    """Load a baseline pass response fixture."""
    d = fixture_dir if fixture_dir is not None else FIXTURE_DIR_SA2
    p = d / f"{doc_id}_{pass_name}_response.json"
    if not p.exists():
        pytest.skip(f"Baseline fixture not found: {p}")
    return json.loads(p.read_text())


def _load_texts_list(doc_id: str, fixture_dir: Path | None = None) -> list[dict] | None:
    """Load the flat texts list fixture for a document.

    Returns None if the fixture is absent (caller should handle gracefully).
    The texts_today.json files are a flat list of text element dicts, each
    with at minimum 'self_ref' and 'text' keys.
    """
    d = fixture_dir if fixture_dir is not None else FIXTURE_DIR_SA2
    p = d / f"{doc_id}_texts_today.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Identity-names based critical evidence helper (rev 15 Option B)
# ---------------------------------------------------------------------------

# Maps each field_group pass to its upstream identity pass.  Only text-type
# identity passes are relevant for system_name extraction.  The mapping is
# driven by entity type (radar vs missile), not by literal pass names.
_FIELD_GROUP_TO_IDENTITY_PASS: dict[str, str] = {
    "radar_power_rf": "radar_identity",
    "radar_antenna": "radar_identity",
    "radar_timing": "radar_identity",
    "radar_modulation": "radar_identity",
    "missile_kinematics": "missile_identity",
    "missile_guidance": "missile_identity",
    "missile_airframe": "missile_identity",
    "missile_speed_timing": "missile_identity",
    "missile_propulsion": "missile_identity",
}

# Maps each identity pass name to the key inside pass_output that holds the
# list of entity dicts.
_IDENTITY_PASS_ENTITY_KEY: dict[str, str] = {
    "radar_identity": "radar_systems",
    "missile_identity": "missile_systems",
}


def _extract_system_names_from_identity_response(response: dict[str, Any]) -> set[str]:
    """Extract all system_names and nomenclature strings from an identity pass response.

    Fixture shape (bdde417 baseline): entities are flat dicts inside
    pass_output["radar_systems"] or pass_output["missile_systems"].
    The relevant string fields are:
      - system_name  (str | None)
      - nomenclature (str | None)  — may contain slash-separated alternatives
        like "S-75 Dvina/Desna/Volkhov"; we include the whole string as well
        as each slash-split token so substring matches work on partial names.

    Returns a set of non-empty stripped strings.
    """
    pass_name = response.get("pass_name", "")
    entity_list_key = _IDENTITY_PASS_ENTITY_KEY.get(pass_name)
    if entity_list_key is None:
        # Attempt generic lookup by inspecting pass_output keys
        pass_output = response.get("pass_output") or {}
        for candidate_key in ("radar_systems", "missile_systems"):
            if candidate_key in pass_output:
                entity_list_key = candidate_key
                break

    if entity_list_key is None:
        return set()

    entities = (response.get("pass_output") or {}).get(entity_list_key) or []
    names: set[str] = set()

    for entity in entities:
        for field in ("system_name", "nomenclature"):
            val = entity.get(field)
            if not val or not isinstance(val, str):
                continue
            val = val.strip()
            if not val:
                continue
            names.add(val)
            # Also add slash-split tokens (e.g. "S-75 Dvina/Desna/Volkhov"
            # → "S-75 Dvina", "Desna", "Volkhov").
            for token in val.split("/"):
                token = token.strip()
                if token:
                    names.add(token)

    return names


def critical_evidence_refs_via_identity_names(
    *,
    field_group_pass: str,
    doc_id: str,
    fixture_dir: Path | None = None,
) -> set[str]:
    """Return self_refs whose text contains any identity-pass system_name.

    Rev 15 Option B decision: for field_group passes whose key metric is a
    PROPAGATED field (e.g. radar_power_rf.emitter_function is propagated from
    radar_identity via Step 2 entity-name match), the semantically correct VR
    recall gate must verify that the field_group pass's narrowed input still
    includes chunks mentioning the radar/missile system_names that the upstream
    identity pass extracted.  If those chunks are lost by the vector-router's
    narrowing step, no radar_power_rf entity is created and Step 2 propagation
    has nothing to write emitter_function onto.

    Algorithm:
      1. Determine the upstream identity pass for this field_group_pass using
         _FIELD_GROUP_TO_IDENTITY_PASS.
      2. Load the identity pass's baseline response fixture.
      3. Extract all entity system_names + nomenclature strings.
      4. Load the doc's text fixture (flat list of text element dicts).
      5. For each text element, check whether any identity name appears
         (case-insensitive substring) in the element's text.
      6. Return the set of matching self_refs.

    Returns an empty set if:
      - field_group_pass is not in _FIELD_GROUP_TO_IDENTITY_PASS.
      - The identity pass response fixture is absent (caller pytest.skip).
      - The identity pass yielded no system_names (caller pytest.skip).
      - The texts fixture is absent (returns empty set; best-effort).

    Fixture location: {fixture_dir}/{doc_id}_{pass_name}_response.json and
    {fixture_dir}/{doc_id}_texts_today.json.
    """
    identity_pass = _FIELD_GROUP_TO_IDENTITY_PASS.get(field_group_pass)
    if identity_pass is None:
        return set()

    # Load identity pass response; pytest.skip if absent.
    identity_response = _load_pass_response(doc_id, identity_pass, fixture_dir)
    system_names = _extract_system_names_from_identity_response(identity_response)

    if not system_names:
        return set()

    # Load text elements list for this doc.
    texts_list = _load_texts_list(doc_id, fixture_dir)
    if not texts_list:
        return set()

    names_lower = [n.lower() for n in system_names]
    matching_refs: set[str] = set()

    for elem in texts_list:
        self_ref = elem.get("self_ref")
        if not self_ref:
            continue
        text = (elem.get("text") or "").lower()
        if not text:
            continue
        if any(name in text for name in names_lower):
            matching_refs.add(self_ref)

    return matching_refs


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

    Returns dict keyed by (doc_id, pass_name) → set of critical evidence refs.
    Dvina uses 100% recall requirement — all evidence refs must be in scope.

    Per rev 15 Option B:
      - radar_identity, missile_identity, system_links: use field_provenance
        (these passes extract their own fields directly).
      - radar_power_rf, missile_kinematics: use identity-names anchor (these
        passes' key metrics are propagated; field_provenance is empty in
        bdde417 baseline).  The critical evidence is self_refs of text chunks
        containing any upstream identity system_name.
    """
    result: dict[tuple[str, str], set[str]] = {}

    # Identity passes and system_links: field_provenance-based recall.
    for pass_name in ("radar_identity", "missile_identity", "system_links"):
        resp = _load_pass_response(DVINA_DOC_ID, pass_name)
        raw_refs = extract_evidence_refs_from_response(resp)
        normalized = normalize_evidence_refs(raw_refs)
        if normalized:
            result[(DVINA_DOC_ID, pass_name)] = normalized

    # Field-group passes: identity-names-based recall (rev 15 Option B).
    for pass_name in ("radar_power_rf", "missile_kinematics"):
        refs = critical_evidence_refs_via_identity_names(
            field_group_pass=pass_name,
            doc_id=DVINA_DOC_ID,
            fixture_dir=FIXTURE_ROOT,
        )
        if refs:
            result[(DVINA_DOC_ID, pass_name)] = refs

    return result


def compute_critical_evidence_refs_sa2() -> dict[tuple[str, str], set[str]]:
    """Compute ONLY the critical evidence refs for SA-2 gates #5 and #6.

    Per rev 15 Option B: both gates now use identity-names anchoring.

    Gate #5 (radar_power_rf.emitter_function ≥ 21/34):
      emitter_function is propagated from radar_identity onto radar_power_rf
      entities via Step 2 system_name match.  The bdde417 baseline records
      no field_provenance for radar_power_rf.  Critical evidence = self_refs
      of text chunks containing any radar_identity system_name.
      Keyed as (SA2_DOC_ID, "radar_power_rf") in the result.

    Gate #6 (missile_kinematics.min_altitude_km ≥ 9/44):
      min_altitude_km is extracted per-variant, so the variant names from
      missile_identity are the critical anchors.  Critical evidence = self_refs
      of text chunks containing any missile_identity system_name.
      Keyed as (SA2_DOC_ID, "missile_kinematics") in the result.
    """
    result: dict[tuple[str, str], set[str]] = {}

    # Gate #5: radar_power_rf — identity-names anchor (rev 15 Option B).
    gate5_refs = critical_evidence_refs_via_identity_names(
        field_group_pass="radar_power_rf",
        doc_id=SA2_DOC_ID,
        fixture_dir=FIXTURE_ROOT,
    )
    if gate5_refs:
        result[(SA2_DOC_ID, "radar_power_rf")] = gate5_refs

    # Gate #6: missile_kinematics — identity-names anchor (rev 15 Option B).
    gate6_refs = critical_evidence_refs_via_identity_names(
        field_group_pass="missile_kinematics",
        doc_id=SA2_DOC_ID,
        fixture_dir=FIXTURE_ROOT,
    )
    if gate6_refs:
        result[(SA2_DOC_ID, "missile_kinematics")] = gate6_refs

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

    Identity passes (radar_identity, missile_identity, system_links) use
    field_provenance-based recall.  Field-group passes (radar_power_rf,
    missile_kinematics) use identity-names anchoring per rev 15 Option B.

    LOAD-BEARING — must pass before C.6 dispatch.
    """

    @pytest.fixture(scope="class")
    def dvina_critical(self) -> dict[tuple[str, str], set[str]]:
        return compute_critical_evidence_refs_dvina()

    def test_dvina_radar_identity_recall(self, dvina_critical):
        """Pre-rerank top-50 must cover all Dvina radar_identity field_provenance refs."""
        refs = dvina_critical.get((DVINA_DOC_ID, "radar_identity"), set())
        if not refs:
            pytest.skip("No radar_identity evidence refs in Dvina baseline fixture")
        _assert_pre_rerank_recall(DVINA_DOC_ID, "radar_identity", refs)

    def test_dvina_radar_power_rf_recall(self, dvina_critical):
        """Pre-rerank top-50 for radar_power_rf must cover chunks containing
        radar_identity system_names (rev 15 Option B — identity-names anchor).

        Dvina radar_power_rf.field_provenance is empty in bdde417 baseline;
        identity-names anchor provides the meaningful gate.
        """
        refs = dvina_critical.get((DVINA_DOC_ID, "radar_power_rf"), set())
        if not refs:
            pytest.skip(
                "No identity-name text chunks found for Dvina radar_power_rf.\n"
                "Check that Dvina radar_identity fixture + texts_today.json exist."
            )
        _assert_pre_rerank_recall(DVINA_DOC_ID, "radar_power_rf", refs)

    def test_dvina_missile_identity_recall(self, dvina_critical):
        """Pre-rerank top-50 must cover all Dvina missile_identity field_provenance refs."""
        refs = dvina_critical.get((DVINA_DOC_ID, "missile_identity"), set())
        if not refs:
            pytest.skip("No missile_identity evidence refs in Dvina baseline fixture")
        _assert_pre_rerank_recall(DVINA_DOC_ID, "missile_identity", refs)

    def test_dvina_missile_kinematics_recall(self, dvina_critical):
        """Pre-rerank top-50 for missile_kinematics must cover chunks containing
        missile_identity system_names (rev 15 Option B — identity-names anchor).
        """
        refs = dvina_critical.get((DVINA_DOC_ID, "missile_kinematics"), set())
        if not refs:
            pytest.skip(
                "No identity-name text chunks found for Dvina missile_kinematics.\n"
                "Check that Dvina missile_identity fixture + texts_today.json exist."
            )
        _assert_pre_rerank_recall(DVINA_DOC_ID, "missile_kinematics", refs)

    def test_dvina_system_links_recall(self, dvina_critical):
        """Pre-rerank top-50 must cover all Dvina system_links field_provenance refs."""
        refs = dvina_critical.get((DVINA_DOC_ID, "system_links"), set())
        if not refs:
            pytest.skip("No system_links evidence refs in Dvina baseline fixture")
        _assert_pre_rerank_recall(DVINA_DOC_ID, "system_links", refs)


class TestSA2CriticalOfflineRecall:
    """SA-2 critical evidence recall (gates #5 and #6 only).

    Gate #5: radar_power_rf.emitter_function — identity-names anchor.
      VR must preserve chunks containing radar_identity system_names in the
      radar_power_rf narrowed input; without them, no radar_power_rf entity
      is created, and emitter_function propagation (Step 2) cannot run.

    Gate #6: missile_kinematics.min_altitude_km — identity-names anchor.
      VR must preserve chunks containing missile_identity system_names (variant
      names like "1D", "13D", "5Ya23", "SA-2", "V-750") in the
      missile_kinematics narrowed input.

    Per rev 15 Option B: both gates use critical_evidence_refs_via_identity_names().
    LOAD-BEARING — must pass before C.6 dispatch.
    """

    @pytest.fixture(scope="class")
    def sa2_critical(self) -> dict[tuple[str, str], set[str]]:
        return compute_critical_evidence_refs_sa2()

    def test_radar_power_rf_emitter_function_recall(self, sa2_critical):
        """Gate #5: pre-rerank top-50 for radar_power_rf must cover chunks
        containing radar_identity system_names (rev 15 Option B).

        This is a HARD gate — xfail removed per rev 15 decision.
        """
        refs = sa2_critical.get((SA2_DOC_ID, "radar_power_rf"), set())
        if not refs:
            pytest.skip(
                "No identity-name text chunks found for SA-2 radar_power_rf.\n"
                "Check that tests/fixtures/sa2/{SA2_DOC_ID}_radar_identity_response.json\n"
                "and tests/fixtures/sa2/{SA2_DOC_ID}_texts_today.json both exist."
            )
        _assert_pre_rerank_recall(SA2_DOC_ID, "radar_power_rf", refs)

    def test_missile_kinematics_min_altitude_recall(self, sa2_critical):
        """Gate #6: pre-rerank top-50 for missile_kinematics must cover chunks
        containing missile_identity system_names (rev 15 Option B).

        This is a HARD gate.
        """
        refs = sa2_critical.get((SA2_DOC_ID, "missile_kinematics"), set())
        if not refs:
            pytest.skip(
                "No identity-name text chunks found for SA-2 missile_kinematics.\n"
                "Check that tests/fixtures/sa2/{SA2_DOC_ID}_missile_identity_response.json\n"
                "and tests/fixtures/sa2/{SA2_DOC_ID}_texts_today.json both exist."
            )
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
        """SA-2 radar_identity must have emitter_function evidence refs."""
        resp = _load_pass_response(SA2_DOC_ID, "radar_identity")
        refs = extract_evidence_refs_from_response(resp, field_filter="emitter_function")
        assert len(refs) > 0, (
            "SA-2 radar_identity has no emitter_function evidence refs in field_provenance."
        )

    def test_sa2_radar_power_rf_no_field_provenance(self):
        """SA-2 radar_power_rf baseline has no field_provenance in bdde417 fixture.

        This is the known shape that motivated rev 15 Option B: emitter_function
        on radar_power_rf is a PROPAGATED field (Step 2), not directly extracted,
        so field_provenance is empty.  Gate #5 uses identity-names anchoring
        instead.  This test documents the known fixture shape so that future
        additions of field_provenance to radar_power_rf are visible as a change.
        """
        resp = _load_pass_response(SA2_DOC_ID, "radar_power_rf")
        refs = extract_evidence_refs_from_response(resp)
        # Expected: zero in bdde417 baseline.
        # NOTE: if this ever fails (refs > 0), it means radar_power_rf now
        # has field_provenance.  That is a welcome change — update this comment
        # and consider whether identity-names anchoring should be supplemented
        # with field_provenance refs for gate #5.
        assert len(refs) == 0, (
            f"SA-2 radar_power_rf now has {len(refs)} field_provenance evidence refs "
            "(expected 0 in bdde417 baseline).  Review gate #5 anchoring strategy."
        )


class TestIdentityNamesHelper:
    """Snapshot tests for critical_evidence_refs_via_identity_names().

    These tests verify that the helper returns a non-empty set for the two
    gate-relevant passes (radar_power_rf and missile_kinematics) given the
    SA-2 and Dvina baseline fixtures.  They also snapshot a few specific
    self_refs known to contain identity system_names in the bdde417 baseline.

    These tests require the fixture files to exist (skip if absent).
    """

    # ---------------------------------------------------------------------------
    # SA-2 snapshot tests
    # ---------------------------------------------------------------------------

    def test_sa2_radar_power_rf_identity_names_nonempty(self):
        """Helper returns non-empty set for SA-2 radar_power_rf."""
        refs = critical_evidence_refs_via_identity_names(
            field_group_pass="radar_power_rf",
            doc_id=SA2_DOC_ID,
            fixture_dir=FIXTURE_ROOT,
        )
        assert len(refs) > 0, (
            "critical_evidence_refs_via_identity_names returned empty set for "
            "SA-2 radar_power_rf.  Verify that the radar_identity fixture and "
            "texts_today.json both exist and contain matching text."
        )

    def test_sa2_radar_power_rf_contains_fan_song_ref(self):
        """SA-2 radar_power_rf refs must contain a chunk with 'Fan Song'.

        In bdde417 baseline, 'Fan Song' appears in several text elements
        (e.g. #/texts/34, #/texts/58, #/texts/59).  At least one must appear
        in the returned set.
        """
        refs = critical_evidence_refs_via_identity_names(
            field_group_pass="radar_power_rf",
            doc_id=SA2_DOC_ID,
            fixture_dir=FIXTURE_ROOT,
        )
        # 'Fan Song' is a system_name from SA-2 radar_identity baseline.
        # Verify the helper found chunks containing it.
        known_fan_song_refs = {"#/texts/34", "#/texts/58", "#/texts/59"}
        assert refs & known_fan_song_refs, (
            f"Expected at least one of {known_fan_song_refs} in helper output "
            f"(Fan Song text chunks), but got none.  refs={sorted(refs)[:10]}"
        )

    def test_sa2_missile_kinematics_identity_names_nonempty(self):
        """Helper returns non-empty set for SA-2 missile_kinematics."""
        refs = critical_evidence_refs_via_identity_names(
            field_group_pass="missile_kinematics",
            doc_id=SA2_DOC_ID,
            fixture_dir=FIXTURE_ROOT,
        )
        assert len(refs) > 0, (
            "critical_evidence_refs_via_identity_names returned empty set for "
            "SA-2 missile_kinematics.  Verify that the missile_identity fixture "
            "and texts_today.json both exist and contain matching text."
        )

    def test_sa2_missile_kinematics_returns_only_text_self_refs(self):
        """All returned self_refs must be #/texts/N (texts_today fixture only)."""
        refs = critical_evidence_refs_via_identity_names(
            field_group_pass="missile_kinematics",
            doc_id=SA2_DOC_ID,
            fixture_dir=FIXTURE_ROOT,
        )
        bad = {r for r in refs if not r.startswith("#/texts/")}
        assert not bad, (
            f"Non-text self_refs in helper output (unexpected): {bad}"
        )

    # ---------------------------------------------------------------------------
    # Dvina snapshot tests
    # ---------------------------------------------------------------------------

    def test_dvina_radar_power_rf_identity_names_nonempty(self):
        """Helper returns non-empty set for Dvina radar_power_rf."""
        refs = critical_evidence_refs_via_identity_names(
            field_group_pass="radar_power_rf",
            doc_id=DVINA_DOC_ID,
            fixture_dir=FIXTURE_ROOT,
        )
        assert len(refs) > 0, (
            "critical_evidence_refs_via_identity_names returned empty set for "
            "Dvina radar_power_rf.  Dvina has 1 radar entity (RSNA-75M); verify "
            "Dvina radar_identity fixture and texts_today.json exist."
        )

    def test_dvina_missile_kinematics_identity_names_nonempty(self):
        """Helper returns non-empty set for Dvina missile_kinematics."""
        refs = critical_evidence_refs_via_identity_names(
            field_group_pass="missile_kinematics",
            doc_id=DVINA_DOC_ID,
            fixture_dir=FIXTURE_ROOT,
        )
        assert len(refs) > 0, (
            "critical_evidence_refs_via_identity_names returned empty set for "
            "Dvina missile_kinematics.  Dvina has 1 missile entity (S-75 Dvina); "
            "verify Dvina missile_identity fixture and texts_today.json exist."
        )

    # ---------------------------------------------------------------------------
    # Unit tests — helper logic without fixture I/O
    # ---------------------------------------------------------------------------

    def test_extract_system_names_flat_shape(self):
        """_extract_system_names_from_identity_response handles flat entity dicts."""
        response = {
            "pass_name": "radar_identity",
            "pass_output": {
                "radar_systems": [
                    {"system_name": "Fan Song", "nomenclature": "RSNA-75/SNR-75"},
                    {"system_name": "Spoon Rest", "nomenclature": None},
                ]
            },
        }
        names = _extract_system_names_from_identity_response(response)
        assert "Fan Song" in names
        assert "Spoon Rest" in names
        assert "RSNA-75/SNR-75" in names
        # Slash split
        assert "RSNA-75" in names
        assert "SNR-75" in names

    def test_extract_system_names_slash_split_nomenclature(self):
        """Slash-split nomenclature tokens are added individually."""
        response = {
            "pass_name": "missile_identity",
            "pass_output": {
                "missile_systems": [
                    {"system_name": "S-75", "nomenclature": "S-75 Dvina/Desna/Volkhov"},
                ]
            },
        }
        names = _extract_system_names_from_identity_response(response)
        assert "S-75 Dvina/Desna/Volkhov" in names
        assert "S-75 Dvina" in names
        assert "Desna" in names
        assert "Volkhov" in names

    def test_extract_system_names_skips_none_values(self):
        """None system_name and nomenclature are skipped gracefully."""
        response = {
            "pass_name": "radar_identity",
            "pass_output": {
                "radar_systems": [
                    {"system_name": None, "nomenclature": None},
                    {"system_name": "Side Net", "nomenclature": None},
                ]
            },
        }
        names = _extract_system_names_from_identity_response(response)
        assert "Side Net" in names
        # None values must not appear
        assert None not in names

    def test_extract_system_names_unknown_pass_empty(self):
        """Unknown pass_name with no matching entity key returns empty set."""
        response = {
            "pass_name": "system_links",
            "pass_output": {},
        }
        names = _extract_system_names_from_identity_response(response)
        assert names == set()

    def test_helper_returns_empty_for_unmapped_pass(self):
        """critical_evidence_refs_via_identity_names returns empty set for
        passes not in _FIELD_GROUP_TO_IDENTITY_PASS (e.g. 'system_links')."""
        refs = critical_evidence_refs_via_identity_names(
            field_group_pass="system_links",
            doc_id=SA2_DOC_ID,
            fixture_dir=FIXTURE_ROOT,
        )
        assert refs == set()
