"""Phase 0 (Task 0a) baseline capture for spec 2026-05-11-table-aware-chunking.

Mirrors the notebook sweep pattern from notebooks/extraction_walkthrough.ipynb:
- Fetches the SA-2 doc via /v1/documents/{id}/docling
- Calls /extract-pass for each pass at temp=0.1
- Captures per-pass: entity count, avg_fill, json_fail, relationships count
- Captures input doc + post-sanitization texts as byte-equality target

Output: tests/fixtures/sa2/<docid>_*.json + baseline.meta.json.

Run from repo root with master switches OFF (default). The captured baseline
is the §15.1 byte-equality target and §15.2 flip-gate target.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "sa2"

# Notebook constants
DOCLING_GRAPH_BASE = "http://localhost:8002"
DOCLING_GRAPH_URL = f"{DOCLING_GRAPH_BASE}/extract-pass"
API_BASE = "http://localhost:8005"
BUNDLE_KEY = "air_defense_v3"

# SA-2 doc per user-supplied sweep run
DEFAULT_DOC_ID = "78673393-639b-4fde-9bda-9e7bfd43ccda"
DEFAULT_TEMP = 0.1
DEFAULT_RUNS = 3

# Pass subset matched to the user's reported sweep output (5 passes shown)
DEFAULT_PASSES = [
    "radar_identity",
    "radar_power_rf",
    "missile_identity",
    "missile_kinematics",
    "system_links",
]


def fetch_docling_document(document_id: str) -> dict:
    url = f"{API_BASE}/v1/documents/{document_id}/docling"
    with urllib.request.urlopen(url, timeout=60) as r:
        envelope = json.loads(r.read())
    doc_json = envelope.get("document_json")
    if not doc_json:
        raise RuntimeError(f"no document_json field in /v1/documents/{document_id}/docling response")
    return doc_json


def call_extract_pass(
    pass_name: str,
    document: dict,
    *,
    temperature: float,
    upstream_entities: list[dict] | None = None,
    timeout: int = 14400,
) -> tuple[dict, float]:
    """POST one /extract-pass call. Returns (response_dict, elapsed_seconds)."""
    request_body: dict[str, Any] = {
        "bundle_key": BUNDLE_KEY,
        "pass_name": pass_name,
        "document_id": f"baseline-{pass_name}",
        "docling_document_json": document,
        "temperature": float(temperature),
    }
    if upstream_entities:
        request_body["upstream_entities"] = upstream_entities
    payload = json.dumps(request_body).encode()
    req = urllib.request.Request(
        DOCLING_GRAPH_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        response = json.loads(r.read())
    elapsed = time.monotonic() - t0
    return response, elapsed


def _primary_list_field_for_pass(pass_name: str, pass_output: dict) -> str | None:
    """Heuristic: the primary entity list is usually a list field in pass_output.
    Returns the first list field with non-empty entries, or None."""
    for k, v in pass_output.items():
        if isinstance(v, list) and v:
            return k
    return None


def compute_metrics(pass_name: str, response: dict) -> dict:
    """Compute the sweep metric shape: entity_count, avg_fill, json_fail, relationships."""
    pass_output = response.get("pass_output") or {}
    metadata = response.get("metadata") or {}
    diag = response.get("diagnostics") or {}
    log = diag.get("library_log") or ""

    # json_fail signal (mirrors notebook's heuristic)
    json_fail = (
        bool(diag.get("pipeline_error"))
        or "No valid JSON returned" in log
    )

    # Entity count + avg_fill: primary list field for pass
    entity_count = 0
    fields_filled = 0
    fields_total = 0
    schema_size = 0
    list_field = _primary_list_field_for_pass(pass_name, pass_output)
    if list_field:
        entities = pass_output[list_field]
        entity_count = len(entities)
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            for k, v in ent.items():
                if k in ("confidence",):
                    continue
                fields_total += 1
                if v is not None and v != "" and v != []:
                    fields_filled += 1
        if entity_count > 0 and entities and isinstance(entities[0], dict):
            schema_size = len([k for k in entities[0].keys() if k not in ("confidence",)])

    avg_fill_num = fields_filled / max(entity_count, 1)

    # Relationship count for system_links
    relationships = 0
    field_provenance = response.get("field_provenance") or []
    relationship_provenance = response.get("relationship_provenance") or []
    if pass_name == "system_links":
        for k, v in pass_output.items():
            if isinstance(v, list):
                relationships += len(v)

    return {
        "pass_name": pass_name,
        "entity_count": entity_count,
        "avg_fill_num": round(avg_fill_num, 2),
        "schema_size": schema_size,
        "fields_filled_total": fields_filled,
        "fields_total": fields_total,
        "json_fail": json_fail,
        "relationships": relationships,
        "field_provenance_count": len(field_provenance),
        "relationship_provenance_count": len(relationship_provenance),
        "node_count": (metadata or {}).get("node_count", 0),
        "edge_count": (metadata or {}).get("edge_count", 0),
    }


def sanitize_offline(doc_json: dict) -> dict:
    """Run the docling-graph sanitizer offline by importing it from the
    docling-graph app path. Mirrors what main.py:556 does at extract-pass
    time, so we can capture the post-sanitization texts[] without needing
    the CAPTURE_BASELINE_TEXTS env-hook."""
    # Make the docling-graph app importable from the host.
    dg_app = REPO_ROOT / "docker" / "docling-graph"
    if str(dg_app) not in sys.path:
        sys.path.insert(0, str(dg_app))
    try:
        from app.main import _sanitize_docling_document  # type: ignore
    except ImportError as exc:
        print(f"[warn] could not import docling-graph sanitizer offline: {exc}", file=sys.stderr)
        print("[warn] capturing un-sanitized texts as fallback.", file=sys.stderr)
        return doc_json
    stats = {"texts_in": 0, "texts_dropped": 0}
    return _sanitize_docling_document(doc_json, stats)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SA-2 baseline for spec 2026-05-11.")
    parser.add_argument("--doc-id", default=DEFAULT_DOC_ID, help="Document UUID to ingest")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP, help="Temperature")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Runs per pass for variance")
    parser.add_argument("--passes", nargs="+", default=DEFAULT_PASSES, help="Pass names to run")
    args = parser.parse_args()

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[info] capturing baseline for doc {args.doc_id}, passes={args.passes}, runs={args.runs}, temp={args.temp}")

    # 1. Fetch the doc
    print("[info] fetching docling_document.json …")
    doc_json = fetch_docling_document(args.doc_id)
    print(f"[info] fetched {len(doc_json.get('texts') or [])} texts, "
          f"{len(doc_json.get('tables') or [])} tables, "
          f"{len(doc_json.get('pictures') or [])} pictures")

    # 2. Capture post-sanitization texts
    sanitized = sanitize_offline(json.loads(json.dumps(doc_json)))  # deep-copy to avoid mutation
    texts_path = FIXTURE_DIR / f"{args.doc_id}_texts_today.json"
    texts_path.write_text(json.dumps(sanitized.get("texts") or [], indent=2, ensure_ascii=False))
    print(f"[info] wrote post-sanitization texts to {texts_path} "
          f"({len(sanitized.get('texts') or [])} entries)")

    # 3. Run extraction passes
    counts_path = FIXTURE_DIR / f"{args.doc_id}_extraction_counts_today.json"
    aggregate: dict[str, dict] = {}
    upstream_entities: list[dict] = []  # for system_links — accumulate from earlier passes

    for pass_name in args.passes:
        runs: list[dict] = []
        for r_idx in range(args.runs):
            print(f"[info] {pass_name} run {r_idx + 1}/{args.runs} …", flush=True)
            try:
                response, elapsed = call_extract_pass(
                    pass_name, doc_json,
                    temperature=args.temp,
                    upstream_entities=upstream_entities if pass_name == "system_links" else None,
                )
                metrics = compute_metrics(pass_name, response)
                metrics["elapsed_s"] = round(elapsed, 1)
                runs.append(metrics)
                print(f"  → entities={metrics['entity_count']} avg_fill={metrics['avg_fill_num']:.2f}/{metrics['schema_size']} "
                      f"json_fail={'.' if not metrics['json_fail'] else 'F'} elapsed={metrics['elapsed_s']}s")
                # Capture upstream entities from radar_identity + missile_identity for system_links
                if pass_name in ("radar_identity", "missile_identity") and r_idx == 0:
                    pass_output = response.get("pass_output") or {}
                    for k, v in pass_output.items():
                        if isinstance(v, list):
                            for ent in v:
                                if isinstance(ent, dict) and ent.get("name"):
                                    upstream_entities.append({
                                        "entity_type": ent.get("entity_type") or k.rstrip("s"),
                                        "name": ent.get("name"),
                                    })
            except Exception as exc:
                print(f"[error] {pass_name} run {r_idx + 1} failed: {exc}", flush=True)
                runs.append({
                    "pass_name": pass_name,
                    "error": repr(exc),
                    "entity_count": 0,
                    "avg_fill_num": 0.0,
                    "json_fail": True,
                    "relationships": 0,
                })
        aggregate[pass_name] = {"runs": runs}

    counts_path.write_text(json.dumps(aggregate, indent=2))
    print(f"[info] wrote per-pass counts to {counts_path}")

    # 4. Determine comparison_mode per §19 decision rule
    comparison_mode = "strict"
    for pass_name, payload in aggregate.items():
        exacts = [r.get("entity_count", 0) for r in payload["runs"]]
        if exacts and (max(exacts) - min(exacts)) >= 2:
            comparison_mode = "median"
            break

    # 5. Write baseline.meta.json
    main_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
    ).decode().strip()
    meta_path = FIXTURE_DIR / "baseline.meta.json"
    meta = {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "current_sha": main_sha,
        "corpus_files": [args.doc_id],
        "temperature": args.temp,
        "runs_per_doc": args.runs,
        "comparison_mode": comparison_mode,
        "passes": args.passes,
        "ingest_command": "doc was previously ingested via the standard pipeline (status COMPLETE in ingest.documents)",
        "scoring_command": f"python scripts/capture_sa2_baseline.py --doc-id {args.doc_id} --temp {args.temp} --runs {args.runs}",
        "notes": [
            "Baseline captured with all master switches OFF (today's production behavior).",
            "Metric shape mirrors notebook sweep: entity_count + avg_fill + json_fail + relationships.",
            "Phase 2 flip-gate compares against this baseline at the same flag posture.",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[info] wrote baseline meta to {meta_path}")
    print(f"[info] comparison_mode={comparison_mode}")
    print(f"[ok] baseline capture complete for doc {args.doc_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
