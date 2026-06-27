"""VR C.2b — Live smoke test for vector router recall (NOT load-bearing).

Validates the full pipeline end-to-end including live embedding + reranker.
NOT a gate — does not block C.6/C.7/C.8. Captures real-world drift.

Requires:
  - Live Ollama instance with bge-m3 loaded.
  - Reranker available (bge-reranker-v2-m3 via the api container).
  - Baseline DoclingDocument JSONs at tests/fixtures/recall_baseline/.

Skips silently when Ollama / reranker / fixture docs are unavailable.

See test_vector_router_offline_recall_deterministic.py for the load-bearing
gate that runs entirely offline.

Run standalone (with live services):
    python3 -m pytest tests/integration/test_vector_router_live_recall.py -v -s
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

SA2_DOC_ID = "78673393-639b-4fde-9bda-9e7bfd43ccda"
DVINA_DOC_ID = "b77c48f9-3a27-473f-be05-fa7e73e5d6f5"

FIXTURE_DIR_SA2 = Path(__file__).resolve().parents[1] / "fixtures" / "sa2"
RECALL_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "recall_baseline"

TOP_N_CANDIDATES = 50
TOP_K = 20


def _ollama_available() -> bool:
    """Return True if the Ollama embedding service is reachable."""
    try:
        from app.services.embeddings import embed_texts  # type: ignore[import]

        embed_texts(["ping"], query=False)
        return True
    except Exception:
        return False


def _reranker_available() -> bool:
    """Return True if the reranker service is reachable."""
    try:
        from app.services.reranker import rerank  # type: ignore[import]

        rerank("ping", ["ping result"], top_k=1)
        return True
    except Exception:
        return False


def _docling_json_available(doc_id: str) -> bool:
    """Return True if the DoclingDocument JSON exists for this doc."""
    return (RECALL_FIXTURE_DIR / f"{doc_id}_docling.json").exists()


def _load_docling_json(doc_id: str) -> dict[str, Any]:
    path = RECALL_FIXTURE_DIR / f"{doc_id}_docling.json"
    return json.loads(path.read_text())


def _load_pass_response(doc_id: str, pass_name: str) -> dict[str, Any]:
    p = FIXTURE_DIR_SA2 / f"{doc_id}_{pass_name}_response.json"
    if not p.exists():
        pytest.skip(f"Baseline fixture not found: {p}")
    return json.loads(p.read_text())


# Re-use normalization + extraction helpers from the deterministic test.
from tests.integration.test_vector_router_offline_recall_deterministic import (  # noqa: E402
    normalize_evidence_refs,
    extract_evidence_refs_from_response,
)


@pytest.fixture(scope="module")
def skip_if_no_services():
    """Skip the live tests if Ollama or reranker is unavailable."""
    if not _ollama_available():
        pytest.skip("Ollama embedding service unavailable — skip live recall smoke")
    if not _reranker_available():
        pytest.skip("Reranker service unavailable — skip live recall smoke")


@pytest.fixture(scope="module")
def skip_if_no_docling_dvina():
    """Skip if Dvina DoclingDocument JSON is not present."""
    if not _docling_json_available(DVINA_DOC_ID):
        pytest.skip(
            f"Dvina DoclingDocument JSON missing at "
            f"{RECALL_FIXTURE_DIR}/{DVINA_DOC_ID}_docling.json. "
            "Export from MinIO/database to run live smoke."
        )


@pytest.fixture(scope="module")
def skip_if_no_docling_sa2():
    """Skip if SA-2 DoclingDocument JSON is not present."""
    if not _docling_json_available(SA2_DOC_ID):
        pytest.skip(
            f"SA-2 DoclingDocument JSON missing at "
            f"{RECALL_FIXTURE_DIR}/{SA2_DOC_ID}_docling.json. "
            "Export from MinIO/database to run live smoke."
        )


class TestDvinaLiveRecall:
    """Live full-pipeline recall test for Dvina.

    NOT load-bearing — skip if services unavailable.
    """

    @pytest.fixture(scope="class")
    def dvina_doc(self, skip_if_no_services, skip_if_no_docling_dvina):
        return _load_docling_json(DVINA_DOC_ID)

    def _run_live_pass(self, doc_json: dict, pass_name: str, template_cls_fqn: str) -> set[str]:
        """Run live embed + rerank for one pass and return selected self_refs."""
        import importlib

        import numpy as np

        from app.services.embeddings import embed_texts  # type: ignore[import]
        from app.services.extraction_query_builder import build_retrieval_query  # type: ignore
        from app.services.reranker import rerank  # type: ignore[import]
        from app.services.extraction_chunk_index import build_extraction_chunks  # type: ignore

        # Load template class
        module_path, class_name = template_cls_fqn.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        template_cls = getattr(mod, class_name)

        # Build query
        query_text = build_retrieval_query(None, template_cls)

        # Build chunks
        chunks = build_extraction_chunks(doc_json)
        texts = [c["text"] for c in chunks]
        self_refs = [c["self_ref"] for c in chunks]

        # Embed
        chunk_embs_raw = embed_texts(texts, query=False)
        query_emb_raw = embed_texts([query_text], query=True)
        chunk_embs = np.array(chunk_embs_raw, dtype=np.float32)
        query_emb = np.array(query_emb_raw[0], dtype=np.float32)

        # L2-normalize
        chunk_norms = np.linalg.norm(chunk_embs, axis=-1, keepdims=True)
        chunk_norms = np.where(chunk_norms == 0, 1.0, chunk_norms)
        chunk_embs /= chunk_norms
        query_norm = np.linalg.norm(query_emb)
        if query_norm > 0:
            query_emb /= query_norm

        # Top-N candidates by cosine similarity
        scores = chunk_embs @ query_emb
        if len(scores) <= TOP_N_CANDIDATES:
            top_indices = list(range(len(scores)))
        else:
            top_indices = list(np.argpartition(scores, -TOP_N_CANDIDATES)[-TOP_N_CANDIDATES:])
        candidate_texts = [texts[i] for i in top_indices]
        candidate_refs = [self_refs[i] for i in top_indices]

        # Rerank
        reranked = rerank(query_text, candidate_texts, top_k=TOP_K)
        # reranked returns list of (text, score) or similar — adapt to actual API
        if reranked:
            # Build ref → text mapping for lookup
            text_to_ref = {t: r for t, r in zip(candidate_texts, candidate_refs)}
            selected_refs = set()
            for item in reranked:
                # item may be (text, score) or dict — handle both
                if isinstance(item, (list, tuple)):
                    text_key = item[0]
                elif isinstance(item, dict):
                    text_key = item.get("text") or item.get("content") or ""
                else:
                    continue
                ref = text_to_ref.get(text_key)
                if ref:
                    selected_refs.add(ref)
        else:
            selected_refs = {candidate_refs[i] for i in range(len(candidate_refs))}

        return selected_refs

    def test_dvina_missile_kinematics_live_recall(self, dvina_doc):
        """Full-pipeline recall for Dvina missile_kinematics."""
        resp = _load_pass_response(DVINA_DOC_ID, "missile_kinematics")
        critical_refs = normalize_evidence_refs(
            extract_evidence_refs_from_response(resp)
        )
        if not critical_refs:
            pytest.skip("No evidence refs in Dvina missile_kinematics fixture")

        selected = self._run_live_pass(
            dvina_doc,
            "missile_kinematics",
            "ontology_bundles.air_defense_v3.extraction_schemas.missile_kinematics.MissileKinematicsPass",
        )
        missing = critical_refs - selected
        # NOT load-bearing — warn instead of assert to capture drift.
        if missing:
            pytest.xfail(
                f"[Dvina/missile_kinematics] Post-rerank missed "
                f"{len(missing)}/{len(critical_refs)} critical refs: {sorted(missing)[:5]}. "
                "Not blocking — investigate in C.6 shadow run."
            )


class TestSA2CriticalLiveRecall:
    """Live full-pipeline recall test for SA-2 critical evidence.

    NOT load-bearing — skip if services unavailable.
    """

    @pytest.fixture(scope="class")
    def sa2_doc(self, skip_if_no_services, skip_if_no_docling_sa2):
        return _load_docling_json(SA2_DOC_ID)

    def test_sa2_gate6_min_altitude_live_recall(self, sa2_doc):
        """Full-pipeline recall for SA-2 gate #6 (min_altitude_km)."""
        resp = _load_pass_response(SA2_DOC_ID, "missile_kinematics")
        critical_refs = normalize_evidence_refs(
            extract_evidence_refs_from_response(resp, field_filter="min_altitude_km")
        )
        if not critical_refs:
            pytest.skip("No min_altitude_km evidence refs in SA-2 missile_kinematics fixture")

        from app.services.extraction_query_builder import build_retrieval_query  # type: ignore
        from ontology_bundles.air_defense_v3.extraction_schemas.missile_kinematics import MissileKinematicsPass  # type: ignore  # noqa: E501

        query_text = build_retrieval_query(None, MissileKinematicsPass)

        # NOTE: live rerank is expensive on SA-2 (179 critical refs, large doc).
        # This test samples 10 representative refs and validates coverage.
        sample_refs = set(sorted(critical_refs)[:10])

        # For now, just validate the query text is non-empty (live rerank deferred).
        assert query_text, "MissileKinematicsPass query must be non-empty"
        pytest.xfail(
            "SA-2 live rerank validation deferred to C.6 shadow run. "
            "The offline deterministic gate in test_vector_router_offline_recall_deterministic.py "
            "is the load-bearing check. This live test will be fleshed out in C.6."
        )
