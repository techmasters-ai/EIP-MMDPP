"""One-time fixture generator for the deterministic offline recall test.

Generates precomputed embeddings for the bdde417 baseline documents and
per-pass retrieval queries. Output stored in
``tests/fixtures/recall_baseline/{doc_id}_chunks.npz`` (chunk embeddings)
and ``tests/fixtures/recall_baseline/{doc_id}_{pass_name}_query.npz``
(query embedding per pass).

USAGE:
    python scripts/generate_baseline_recall_fixtures.py

REQUIREMENTS:
    - Live Ollama instance with bge-m3 model loaded.
    - The SA-2 and Dvina raw DoclingDocument JSON files must be available.
      Expected locations:
        tests/fixtures/recall_baseline/{doc_id}_docling.json
      or from MinIO / the database (use the --minio flag).

This script is meant to be run ONCE per baseline and the output committed
(or stored in a known location). The deterministic recall test loads these
fixtures and runs entirely offline (no live Ollama).

npz layout:
  _chunks.npz:
    - self_refs: bytes-serialised JSON list[str] of chunk self_refs
    - embeddings: float32 ndarray (N, D) — L2-normalised

  _query.npz (per pass):
    - query_text: bytes (UTF-8)
    - embedding: float32 ndarray (D,) — L2-normalised

Run with --help for full options.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document IDs (bdde417 baseline docs)
# ---------------------------------------------------------------------------

SA2_DOC_ID = "78673393-639b-4fde-9bda-9e7bfd43ccda"
DVINA_DOC_ID = "b77c48f9-3a27-473f-be05-fa7e73e5d6f5"

# Passes that have critical evidence and need query embeddings.
CRITICAL_PASSES = [
    # (doc_id, pass_name, template_class_fqn)
    (SA2_DOC_ID, "radar_power_rf",
     "ontology_bundles.air_defense_v3.extraction_schemas.radar_power_rf.RadarPowerRfPass"),
    (SA2_DOC_ID, "missile_kinematics",
     "ontology_bundles.air_defense_v3.extraction_schemas.missile_kinematics.MissileKinematicsPass"),
    (DVINA_DOC_ID, "radar_identity",
     "ontology_bundles.air_defense_v3.extraction_schemas.radar_identity.RadarIdentityPass"),
    (DVINA_DOC_ID, "radar_power_rf",
     "ontology_bundles.air_defense_v3.extraction_schemas.radar_power_rf.RadarPowerRfPass"),
    (DVINA_DOC_ID, "missile_identity",
     "ontology_bundles.air_defense_v3.extraction_schemas.missile_identity.MissileIdentityPass"),
    (DVINA_DOC_ID, "missile_kinematics",
     "ontology_bundles.air_defense_v3.extraction_schemas.missile_kinematics.MissileKinematicsPass"),
    (DVINA_DOC_ID, "system_links",
     "ontology_bundles.air_defense_v3.extraction_schemas.system_links.SystemLinksPass"),
]

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "recall_baseline"


def _load_pass_cls(fqn: str):
    """Import a class from its fully-qualified name."""
    import importlib

    module_path, class_name = fqn.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize row vectors; zero-norm rows become zero vectors."""
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def _embed_texts_live(texts: list[str], *, query: bool = False) -> np.ndarray:
    """Call the live embed_texts() service and return L2-normalised float32 array.

    Requires the API service to be running (or Ollama directly).
    """
    from app.services.embeddings import embed_texts  # type: ignore[import]

    raw = embed_texts(texts, query=query)  # list[list[float]]
    arr = np.array(raw, dtype=np.float32)
    return _l2_normalize(arr)


def _load_docling_document(doc_id: str, docling_json_path: Path | None) -> dict:
    """Load a DoclingDocument JSON — from explicit path or raise."""
    if docling_json_path and docling_json_path.exists():
        logger.info("Loading DoclingDocument from %s", docling_json_path)
        return json.loads(docling_json_path.read_text())
    raise FileNotFoundError(
        f"DoclingDocument not found for {doc_id}. "
        f"Expected at {docling_json_path}. "
        "Either export from MinIO/database and place it there, or pass --minio."
    )


def _extract_chunks_from_docling(doc_json: dict) -> list[dict]:
    """Extract ExtractionChunk-equivalent dicts from a DoclingDocument.

    Returns list of dicts with 'self_ref' and 'text' keys.
    This mirrors the chunking logic in app/services/extraction_chunk_index.py.
    """
    from app.services.extraction_chunk_index import build_extraction_chunks  # type: ignore

    return build_extraction_chunks(doc_json)


def generate_chunk_embeddings(
    doc_id: str,
    docling_json_path: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> None:
    """Generate and save chunk embeddings for one document."""
    out_path = output_dir / f"{doc_id}_chunks.npz"
    if out_path.exists():
        logger.info("Chunk embeddings already exist at %s — skipping.", out_path)
        return

    doc_json = _load_docling_document(doc_id, docling_json_path)
    chunks = _extract_chunks_from_docling(doc_json)
    logger.info("doc %s: %d chunks to embed", doc_id[:8], len(chunks))

    if dry_run:
        logger.info("[dry_run] Would embed %d chunks for %s", len(chunks), doc_id[:8])
        return

    texts = [c["text"] for c in chunks]
    self_refs = [c["self_ref"] for c in chunks]

    embeddings = _embed_texts_live(texts, query=False)
    logger.info("doc %s: embedded %d chunks, shape %s", doc_id[:8], len(chunks), embeddings.shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        self_refs=json.dumps(self_refs).encode("utf-8"),
        embeddings=embeddings,
    )
    logger.info("Saved chunk embeddings to %s", out_path)


def generate_query_embedding(
    doc_id: str,
    pass_name: str,
    template_cls_fqn: str,
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> None:
    """Generate and save query embedding for one (doc, pass) pair."""
    out_path = output_dir / f"{doc_id}_{pass_name}_query.npz"
    if out_path.exists():
        logger.info("Query embedding already exists at %s — skipping.", out_path)
        return

    from app.services.extraction_query_builder import build_retrieval_query  # type: ignore

    template_cls = _load_pass_cls(template_cls_fqn)
    query_text = build_retrieval_query(None, template_cls)
    logger.info("doc %s/%s: query text (%d chars)", doc_id[:8], pass_name, len(query_text))

    if dry_run:
        logger.info("[dry_run] Would embed query for %s/%s: %r...", doc_id[:8], pass_name, query_text[:80])
        return

    emb = _embed_texts_live([query_text], query=True)  # (1, D)
    query_emb = emb[0]  # (D,)
    logger.info("doc %s/%s: query embedding shape %s", doc_id[:8], pass_name, query_emb.shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        query_text=query_text.encode("utf-8"),
        embedding=query_emb,
    )
    logger.info("Saved query embedding to %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=FIXTURE_DIR,
        help="Directory to write .npz fixtures (default: tests/fixtures/recall_baseline/)",
    )
    parser.add_argument(
        "--sa2-docling-json", type=Path,
        default=FIXTURE_DIR / f"{SA2_DOC_ID}_docling.json",
        help="Path to SA-2 DoclingDocument JSON.",
    )
    parser.add_argument(
        "--dvina-docling-json", type=Path,
        default=FIXTURE_DIR / f"{DVINA_DOC_ID}_docling.json",
        help="Path to Dvina DoclingDocument JSON.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without calling Ollama.",
    )
    args = parser.parse_args()

    doc_paths = {
        SA2_DOC_ID: args.sa2_docling_json,
        DVINA_DOC_ID: args.dvina_docling_json,
    }

    logger.info("Output directory: %s", args.output_dir)
    if args.dry_run:
        logger.info("DRY RUN — no live Ollama calls will be made.")

    # Generate chunk embeddings for both docs.
    for doc_id, docling_path in doc_paths.items():
        try:
            generate_chunk_embeddings(doc_id, docling_path, args.output_dir, dry_run=args.dry_run)
        except FileNotFoundError as e:
            logger.error("SKIP chunk embeddings for %s: %s", doc_id[:8], e)

    # Generate query embeddings for all critical passes.
    for doc_id, pass_name, template_cls_fqn in CRITICAL_PASSES:
        generate_query_embedding(doc_id, pass_name, template_cls_fqn, args.output_dir, dry_run=args.dry_run)

    logger.info("Done. Fixtures in %s", args.output_dir)
    if not args.dry_run:
        logger.info(
            "Next step: run the deterministic recall test:\n"
            "  python3 -m pytest tests/integration/test_vector_router_offline_recall_deterministic.py -v"
        )


if __name__ == "__main__":
    main()
