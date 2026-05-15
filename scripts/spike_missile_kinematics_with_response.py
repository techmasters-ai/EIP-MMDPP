"""Missile-kinematics spike — single /extract-pass call with FULL RESPONSE
CAPTURE for the SA-2 document, used to verify the post-edit Unit Policy +
UNIT_HINT changes let Gemma4 fill kinematic numeric fields that were 0/N in v9.

Comparable baseline: tests/fixtures/sa2/<doc_id>_missile_kinematics_response_v9.json
(40 entities, 0/6 avg_fill in v9).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "tmp" / "claude_vs_gemma4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOCLING_GRAPH_URL = "http://localhost:8002/extract-pass"
BUNDLE_KEY = "air_defense_v3"
DOC_ID = "78673393-639b-4fde-9bda-9e7bfd43ccda"
TEMP = 0.1


def fetch_docling_document(document_id: str) -> dict:
    url = f"http://localhost:8005/v1/documents/{document_id}/docling"
    with urllib.request.urlopen(url, timeout=60) as r:
        envelope = json.loads(r.read())
    return envelope["document_json"]


def main() -> int:
    print("[info] fetching docling_document.json …", flush=True)
    doc_json = fetch_docling_document(DOC_ID)
    print(f"[info] fetched {len(doc_json.get('texts') or [])} texts, "
          f"{len(doc_json.get('tables') or [])} tables")

    request_body = {
        "bundle_key": BUNDLE_KEY,
        "pass_name": "missile_kinematics",
        "document_id": "spike-missile_kinematics",
        "docling_document_json": doc_json,
        "temperature": TEMP,
        "upstream_entities": [],
    }
    print(f"[info] calling /extract-pass missile_kinematics (temp={TEMP}) …",
          flush=True)
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
        body = e.read().decode()[:1500]
        print(f"[error] HTTP {e.code}: {body}")
        return 1
    elapsed = time.monotonic() - t0

    pass_output = response.get("pass_output") or {}
    missiles = pass_output.get("missile_systems") or []

    n_entities = len(missiles)
    field_keys = [
        "min_intercept_km", "max_intercept_km",
        "min_altitude_km", "max_altitude_km",
        "max_launch_angle_deg",
    ]
    filled_count = 0
    total_possible = 0
    for m in missiles:
        for k in field_keys:
            total_possible += 1
            if m.get(k) is not None:
                filled_count += 1
    avg_fill = filled_count / n_entities if n_entities else 0.0

    print(f"\n[ok] elapsed={elapsed:.1f}s")
    print(f"[ok] entities={n_entities}, filled={filled_count}/{total_possible} "
          f"({100 * filled_count / max(total_possible, 1):.1f}%)")
    print(f"[ok] avg_fill = {avg_fill:.2f}/{len(field_keys)}")

    if missiles:
        print("\n=== ENTITIES + KINEMATIC FIELDS ===")
        for m in missiles:
            row = {k: m.get(k) for k in field_keys}
            print(f"  {m.get('system_name'):<10} {row}")

    out_path = OUT_DIR / "missile_kinematics_response_post_edit.json"
    out_path.write_text(json.dumps(response, indent=2, ensure_ascii=False))
    print(f"\n[ok] full response saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
