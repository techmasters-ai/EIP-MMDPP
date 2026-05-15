"""System-links spike with FULL RESPONSE CAPTURE.

Hits /extract-pass system_links once and writes the entire response JSON to a
fixture so we can inspect the actual relationships extracted. Uses
ArcadeDB's baseline-era RADAR_SYSTEM/MISSILE_SYSTEM names as
upstream_entities — same set the baseline run saw.

Goal: see (a) how many relationships the current chunking config produces
against baseline names, (b) which specific (source, target, kind) triples
the LLM extracted, (c) compare against baseline's 31.
"""
from __future__ import annotations

import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "spike"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

DOCLING_GRAPH_URL = "http://localhost:8002/extract-pass"
BUNDLE_KEY = "air_defense_v3"
DOC_ID = "78673393-639b-4fde-9bda-9e7bfd43ccda"
TEMP = 0.1
ARCADEDB_URL = "http://localhost:2480/api/v1/query/eip_knowledge_graph"
ARCADEDB_USER = "root"
ARCADEDB_PW = "eip_arcadedb_secret"


def fetch_entity_names(entity_type: str) -> list[str]:
    auth = base64.b64encode(f"{ARCADEDB_USER}:{ARCADEDB_PW}".encode()).decode()
    body = json.dumps({
        "language": "sql",
        "command": f"SELECT name FROM {entity_type}",
    }).encode()
    req = urllib.request.Request(
        ARCADEDB_URL, data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.loads(r.read())
    names = [row["name"] for row in (result.get("result") or []) if row.get("name")]
    return names


def fetch_docling_document(document_id: str) -> dict:
    url = f"http://localhost:8005/v1/documents/{document_id}/docling"
    with urllib.request.urlopen(url, timeout=60) as r:
        envelope = json.loads(r.read())
    return envelope["document_json"]


def main() -> int:
    radars = fetch_entity_names("RADAR_SYSTEM")
    missiles = fetch_entity_names("MISSILE_SYSTEM")
    print(f"[info] upstream: {len(radars)} RADAR + {len(missiles)} MISSILE")
    print(f"  radars:   {radars}")
    print(f"  missiles: {missiles}")

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

    print("[info] fetching docling_document.json …")
    doc_json = fetch_docling_document(DOC_ID)
    print(f"[info] fetched {len(doc_json.get('texts') or [])} texts, "
          f"{len(doc_json.get('tables') or [])} tables")

    print(f"[info] calling /extract-pass system_links (temp={TEMP}) …", flush=True)
    request_body = {
        "bundle_key": BUNDLE_KEY,
        "pass_name": "system_links",
        "document_id": "spike-system_links",
        "docling_document_json": doc_json,
        "temperature": TEMP,
        "upstream_entities": upstream_entities,
    }
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(
            DOCLING_GRAPH_URL,
            data=json.dumps(request_body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=14400) as r:
            response = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:1000]
        print(f"[error] HTTP {e.code}: {body}")
        return 1
    elapsed = time.monotonic() - t0

    pass_output = response.get("pass_output") or {}
    relationships: list = []
    for k, v in pass_output.items():
        if isinstance(v, list):
            relationships.extend(v)

    print(f"\n[ok] elapsed={elapsed:.1f}s relationships={len(relationships)}")
    print(f"[ok] pass_output keys: {list(pass_output.keys())}")
    if relationships:
        sample = relationships[0]
        if isinstance(sample, dict):
            print(f"[ok] first relationship keys: {list(sample.keys())}")
        print("\n=== ALL RELATIONSHIPS ===")
        for i, rel in enumerate(relationships):
            print(f"  [{i}] {json.dumps(rel, ensure_ascii=False)}")

    out_path = FIXTURE_DIR / "system_links_response_with_normalization.json"
    out_path.write_text(json.dumps(response, indent=2, ensure_ascii=False))
    print(f"\n[ok] full response saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
