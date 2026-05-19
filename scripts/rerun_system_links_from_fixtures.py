"""One-off helper: rerun ONLY system_links, building upstream_entities from
existing radar_identity + missile_identity fixtures on disk rather than
re-running those passes.

Mirrors the upstream-entities harvesting in scripts/capture_sa2_baseline.py
(EntityRef shape: ref_id / entity_type / identity_values / display_label /
optional aliases).

Usage:
  python3 scripts/rerun_system_links_from_fixtures.py \
      --doc-id 78673393-639b-4fde-9bda-9e7bfd43ccda \
      --temp 0.1
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

API_BASE = "http://localhost:8005"
DOCLING_GRAPH_URL = "http://localhost:8002/extract-pass"
BUNDLE_KEY = "air_defense_v3"
FIXTURE_DIR = Path("tests/fixtures/sa2")


def fetch_docling_document(document_id: str) -> dict:
    url = f"{API_BASE}/v1/documents/{document_id}/docling"
    with urllib.request.urlopen(url, timeout=60) as r:
        envelope = json.loads(r.read())
    doc_json = envelope.get("document_json")
    if not doc_json:
        raise RuntimeError(f"no document_json field in response")
    return doc_json


def harvest_upstream_entities_from_fixture(
    doc_id: str, pass_name: str, entity_type: str, list_key: str,
) -> list[dict]:
    """Mirror capture_sa2_baseline.py's harvest logic, but read from a saved fixture."""
    path = FIXTURE_DIR / f"{doc_id}_{pass_name}_response.json"
    response = json.loads(path.read_text())
    pass_output = response.get("pass_output") or {}
    out: list[dict] = []
    for ent in (pass_output.get(list_key) or []):
        if not isinstance(ent, dict):
            continue
        name = ent.get("system_name") or ent.get("name")
        if not name:
            continue
        alias_candidates = [
            ent.get("nomenclature"),
            ent.get("name"),
            ent.get("dieqp"),
        ]
        aliases: list[str] = []
        seen: set[str] = set()
        for a in alias_candidates:
            if not isinstance(a, str):
                continue
            a = a.strip()
            if not a or a == name or a in seen:
                continue
            seen.add(a)
            aliases.append(a)
        entry: dict = {
            "ref_id":          f"{entity_type}:{name}",
            "entity_type":     entity_type,
            "identity_values": {"system_name": name},
            "display_label":   name,
        }
        if aliases:
            entry["aliases"] = aliases
        # Item 3 (role-aware CUES validation): expose emitter_function on
        # properties so the relationship postprocess can validate CUES
        # direction structurally. Generic — any non-identity field a
        # postprocess might need can ride on properties.
        properties: dict = {}
        ef = ent.get("emitter_function")
        if isinstance(ef, str) and ef:
            properties["emitter_function"] = ef
        if properties:
            entry["properties"] = properties
        out.append(entry)
    return out


def call_extract_pass(
    pass_name: str,
    document: dict,
    *,
    temperature: float,
    upstream_entities: list[dict],
    timeout: int = 14400,
) -> tuple[dict, float]:
    request_body: dict[str, Any] = {
        "bundle_key": BUNDLE_KEY,
        "pass_name": pass_name,
        "document_id": f"baseline-{pass_name}",
        "docling_document_json": document,
        "temperature": float(temperature),
        "upstream_entities": upstream_entities,
    }
    payload = json.dumps(request_body).encode()
    req = urllib.request.Request(
        DOCLING_GRAPH_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        response = json.loads(r.read())
    return response, time.monotonic() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--doc-id", required=True)
    p.add_argument("--temp", type=float, default=0.1)
    args = p.parse_args()

    print(f"[info] harvesting upstream entities from existing fixtures …")
    radar = harvest_upstream_entities_from_fixture(
        args.doc_id, "radar_identity", "RADAR_SYSTEM", "radar_systems",
    )
    missile = harvest_upstream_entities_from_fixture(
        args.doc_id, "missile_identity", "MISSILE_SYSTEM", "missile_systems",
    )
    upstream = radar + missile
    print(f"[info] upstream catalog: {len(radar)} radar + {len(missile)} missile = {len(upstream)} total")
    print(f"[info] radar sample refs: {[e['ref_id'] for e in radar[:3]]}")
    print(f"[info] missile sample refs: {[e['ref_id'] for e in missile[:3]]}")

    print(f"[info] fetching docling_document.json …")
    doc_json = fetch_docling_document(args.doc_id)
    print(f"[info] fetched {len(doc_json.get('texts') or [])} texts, "
          f"{len(doc_json.get('tables') or [])} tables")

    print(f"[info] calling /extract-pass for system_links …")
    response, elapsed = call_extract_pass(
        "system_links", doc_json,
        temperature=args.temp,
        upstream_entities=upstream,
    )

    out_path = FIXTURE_DIR / f"{args.doc_id}_system_links_response.json"
    out_path.write_text(json.dumps(response, indent=2, ensure_ascii=False))
    rels = (response.get("pass_output") or {}).get("relationships") or []
    print(f"[ok] system_links complete in {elapsed:.0f}s. {len(rels)} relationships. wrote {out_path}")


if __name__ == "__main__":
    main()
