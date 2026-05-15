"""Capture just the system_links pass for the SA-2 baseline.

Uses existing RADAR_SYSTEM + MISSILE_SYSTEM names from ArcadeDB as
upstream_entities (the prior baseline run failed system_links with 422
due to malformed upstream shape). Merges the result into the existing
extraction_counts_today.json.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "sa2"

DOCLING_GRAPH_URL = "http://localhost:8002/extract-pass"
BUNDLE_KEY = "air_defense_v3"
DOC_ID = "78673393-639b-4fde-9bda-9e7bfd43ccda"
TEMP = 0.1
ARCADEDB_URL = "http://localhost:2480/api/v1/query/eip_knowledge_graph"
ARCADEDB_USER = "root"
ARCADEDB_PW = "eip_arcadedb_secret"


def fetch_entity_names(entity_type: str) -> list[str]:
    """Pull entity names of a given type from ArcadeDB."""
    import base64
    auth = base64.b64encode(f"{ARCADEDB_USER}:{ARCADEDB_PW}".encode()).decode()
    body = json.dumps({
        "language": "sql",
        "command": f"SELECT system_name FROM {entity_type}",
    }).encode()
    req = urllib.request.Request(
        ARCADEDB_URL, data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.loads(r.read())
    return [
        row["system_name"]
        for row in (result.get("result") or [])
        if row.get("system_name")
    ]


def fetch_docling_document(document_id: str) -> dict:
    url = f"http://localhost:8005/v1/documents/{document_id}/docling"
    with urllib.request.urlopen(url, timeout=60) as r:
        envelope = json.loads(r.read())
    return envelope["document_json"]


def main() -> int:
    print(f"[info] capturing system_links for {DOC_ID} (temp={TEMP})")

    # 1. Pull upstream entities
    radars = fetch_entity_names("RADAR_SYSTEM")
    missiles = fetch_entity_names("MISSILE_SYSTEM")
    print(f"[info] upstream: {len(radars)} RADAR_SYSTEM + {len(missiles)} MISSILE_SYSTEM")

    upstream_entities: list[dict] = []
    for name in radars:
        upstream_entities.append({
            "ref_id":          f"RADAR_SYSTEM:{name}",
            "entity_type":     "RADAR_SYSTEM",
            "identity_values": {"system_name": name},
            "display_label":   name,
        })
    for name in missiles:
        upstream_entities.append({
            "ref_id":          f"MISSILE_SYSTEM:{name}",
            "entity_type":     "MISSILE_SYSTEM",
            "identity_values": {"system_name": name},
            "display_label":   name,
        })

    # 2. Fetch the doc
    print("[info] fetching docling_document.json …")
    doc_json = fetch_docling_document(DOC_ID)

    # 3. Call /extract-pass for system_links
    print("[info] system_links run 1/1 …", flush=True)
    request_body = {
        "bundle_key": BUNDLE_KEY,
        "pass_name": "system_links",
        "document_id": f"baseline-system_links",
        "docling_document_json": doc_json,
        "temperature": TEMP,
        "upstream_entities": upstream_entities,
    }
    payload = json.dumps(request_body).encode()
    req = urllib.request.Request(
        DOCLING_GRAPH_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=14400) as r:
            response = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:1000]
        print(f"[error] HTTP {e.code}: {body}")
        return 1
    elapsed = time.monotonic() - t0

    # 4. Compute metrics
    pass_output = response.get("pass_output") or {}
    diag = response.get("diagnostics") or {}
    log = diag.get("library_log") or ""
    json_fail = bool(diag.get("pipeline_error")) or "No valid JSON returned" in log
    relationships = 0
    for k, v in pass_output.items():
        if isinstance(v, list):
            relationships += len(v)

    metrics = {
        "pass_name": "system_links",
        "entity_count": 0,
        "avg_fill_num": 0.0,
        "schema_size": 0,
        "json_fail": json_fail,
        "relationships": relationships,
        "elapsed_s": round(elapsed, 1),
    }
    print(f"  → relationships={relationships} json_fail={'.' if not json_fail else 'F'} elapsed={metrics['elapsed_s']}s")

    # 5. Merge into existing counts JSON
    counts_path = FIXTURE_DIR / f"{DOC_ID}_extraction_counts_today.json"
    existing: dict = {}
    if counts_path.exists():
        existing = json.loads(counts_path.read_text())
    existing["system_links"] = {"runs": [metrics]}
    counts_path.write_text(json.dumps(existing, indent=2))
    print(f"[ok] merged system_links into {counts_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
