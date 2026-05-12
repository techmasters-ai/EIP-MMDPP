"""Task 0b: verify channel-A cell_refs flow end-to-end.

Calls /extract-pass with DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED=true
on the SA-2 doc for missile_kinematics (a pass that extracts numeric
values which are typical table-cell contents). Inspects the response's
field_provenance for entries with non-empty cell_refs.

Pass criterion: at least one ExtractionFieldProvenance row carries a
cell_refs list with entries matching '#/tables/N/data/table_cells/M'.

If it fails: channel-A wiring needs adjustment (likely the bridge map
isn't being read at the field-provenance enrichment site — see spec
§11.6 and main.py:~1482 wrapper).
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

DOCLING_GRAPH_URL = "http://localhost:8002/extract-pass"
API_BASE = "http://localhost:8005"
BUNDLE_KEY = "air_defense_v3"
DOC_ID = "78673393-639b-4fde-9bda-9e7bfd43ccda"
TEMP = 0.1
PASS_NAME = "missile_kinematics"

CELL_REF_RE = re.compile(r"^#/tables/\d+/data/table_cells/\d+$")


def main() -> int:
    # Fetch the doc
    url = f"{API_BASE}/v1/documents/{DOC_ID}/docling"
    with urllib.request.urlopen(url, timeout=60) as r:
        doc_json = json.loads(r.read())["document_json"]
    print(f"[info] doc fetched: {len(doc_json.get('texts') or [])} texts, {len(doc_json.get('tables') or [])} tables")

    # Call /extract-pass
    body = json.dumps({
        "bundle_key": BUNDLE_KEY,
        "pass_name": PASS_NAME,
        "document_id": f"spike-{PASS_NAME}",
        "docling_document_json": doc_json,
        "temperature": TEMP,
    }).encode()
    req = urllib.request.Request(
        DOCLING_GRAPH_URL, data=body,
        headers={"Content-Type": "application/json"},
    )
    print(f"[info] calling /extract-pass for {PASS_NAME} (table-norm flag ON)…", flush=True)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=14400) as r:
            response = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"[error] HTTP {e.code}: {e.read().decode()[:1000]}")
        return 1
    elapsed = time.monotonic() - t0

    pass_output = response.get("pass_output") or {}
    field_prov = response.get("field_provenance") or []
    diag = response.get("diagnostics") or {}

    # Count primary entities
    entity_count = 0
    for v in pass_output.values():
        if isinstance(v, list):
            entity_count += len(v)
    print(f"[info] elapsed={elapsed:.1f}s entity_count={entity_count} field_provenance={len(field_prov)}")

    # Inspect field_provenance for cell_refs
    rows_with_cells: list[dict] = []
    for r in field_prov:
        cell_refs = r.get("cell_refs") or []
        if cell_refs:
            rows_with_cells.append(r)

    print(f"[info] field_provenance rows with non-empty cell_refs: {len(rows_with_cells)}")

    # Always save the response for the regression gate (Task 22).
    out_path = Path("tests/fixtures/spike/missile_kinematics_response_with_normalization.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(response, indent=2, ensure_ascii=False, default=str))
    print(f"[info] full response saved to {out_path}")

    if rows_with_cells:
        print("[info] sample (up to 3):")
        for r in rows_with_cells[:3]:
            print(f"  - field_name={r.get('field_name')}  cell_refs={r.get('cell_refs')[:3]}")
            print(f"    chunk_index={r.get('chunk_index')}  evidence_id={r.get('evidence_id')}")

        # Validate ref shape
        all_valid = all(
            CELL_REF_RE.match(ref)
            for r in rows_with_cells
            for ref in r["cell_refs"]
        )
        print(f"[info] all cell_refs match #/tables/N/data/table_cells/M shape: {all_valid}")
        if all_valid:
            print("[ok] channel-A cell_refs flow verified end-to-end")
            return 0
        else:
            print("[error] some cell_refs have wrong shape")
            return 2
    else:
        # Possible reasons:
        # (a) Normalization didn't fire (flag off?)
        # (b) Normalization fired but no entities were extracted from synthesized chunks
        # (c) Bridge wasn't populated
        # (d) Field-prov wrapper didn't run
        print("[fail] NO field_provenance rows have cell_refs — channel-A flow needs debugging")
        print("[debug] checking diagnostics for normalization signals…")
        norm_diag = diag.get("service_table_normalization")
        print(f"  service_table_normalization: {norm_diag}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
