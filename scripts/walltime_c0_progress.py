"""Live per-pass progress reporter for the C0+C1+P6 SA-2 + Dvina regression run.

Polls pipeline_pass_outputs for new terminal rows against two pipeline_run_ids
and emits one comparison line per newly-completed pass: NEW wall + counts
side-by-side with the bdde417 baseline + Δ.

Usage:
    python scripts/walltime_c0_progress.py <sa2_run_id> <dvina_run_id>

Stops when both runs have 5 terminal passes (radar_identity, radar_power_rf,
missile_identity, missile_kinematics, system_links).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

# Baseline data from memory/project_extraction_baseline_bdde417.md
# (commit bdde417, captured 2026-05-20T21:29:46Z)
SA2_BASELINE = {
    "radar_identity":     {"wall_s": 3283.3, "entities": 23, "fills": 48,  "fills_total": 253},
    "radar_power_rf":     {"wall_s": 1632.2, "entities": 34, "fills": 55,  "fills_total": 157},
    "missile_identity":   {"wall_s": 6989.9, "entities": 44, "fills": 109, "fills_total": 440},
    "missile_kinematics": {"wall_s": 4327.3, "entities": 44, "fills": 87,  "fills_total": 264},
    "system_links":       {"wall_s": 780.0,  "rels": 29,     "fills": None,"fills_total": None},
}

DVINA_BASELINE = {
    "radar_identity":     {"wall_s": 166.4, "entities": 1, "fills": 1,  "fills_total": 11},
    "radar_power_rf":     {"wall_s": 111.3, "entities": 1, "fills": 1,  "fills_total": 4},
    "missile_identity":   {"wall_s": 417.7, "entities": 1, "fills": 10, "fills_total": 16},
    "missile_kinematics": {"wall_s": 295.8, "entities": 2, "fills": 14, "fills_total": 14},
    "system_links":       {"wall_s": 60.0,  "rels": 1,     "fills": None,"fills_total": None},
}

PASS_ORDER = [
    "radar_identity", "radar_power_rf", "missile_identity",
    "missile_kinematics", "system_links",
]


def query_passes(sa2_run_id: str, dvina_run_id: str) -> list[dict]:
    """Return all terminal pipeline_pass_outputs rows for the two runs,
    ordered by created_at."""
    sql = f"""
SELECT
    ppo.pipeline_run_id::text AS run_id,
    ppo.pass_name,
    ppo.execution_status,
    ppo.yield_status,
    ppo.primary_entities_extracted,
    ppo.bridge_entities_extracted,
    ppo.relationships_extracted,
    ppo.relationships_rejected,
    COALESCE(ppo.diagnostics_json->>'pass_wall_ms', '0')              AS pass_wall_ms,
    COALESCE(ppo.diagnostics_json->>'run_pipeline_ms', '0')           AS run_pipeline_ms,
    COALESCE(ppo.diagnostics_json->>'chunk_count', '0')               AS chunk_count,
    COALESCE(ppo.diagnostics_json->>'batch_count', '0')               AS batch_count,
    COALESCE(ppo.diagnostics_json->>'doc_json_load_ms', '0')          AS doc_json_load_ms,
    COALESCE(ppo.diagnostics_json->>'request_bytes', '0')             AS request_bytes,
    COALESCE(ppo.diagnostics_json->>'response_bytes', '0')            AS response_bytes,
    COALESCE(ppo.diagnostics_json->>'service_queue_wait_ms', '0')     AS service_queue_wait_ms,
    COALESCE(ppo.diagnostics_json->>'sanitize_ms', '0')               AS sanitize_ms,
    COALESCE(ppo.diagnostics_json->>'table_normalization_ms', '0')    AS table_normalization_ms,
    COALESCE(ppo.diagnostics_json->>'table_overlay_ms', '0')          AS table_overlay_ms,
    COALESCE(ppo.diagnostics_json->>'postprocess_ms', '0')            AS postprocess_ms,
    COALESCE(ppo.diagnostics_json->>'field_provenance_ms', '0')       AS field_provenance_ms,
    EXTRACT(EPOCH FROM ppo.created_at) AS created_epoch
FROM ingest.pipeline_pass_outputs ppo
WHERE ppo.pipeline_run_id IN ('{sa2_run_id}', '{dvina_run_id}')
ORDER BY ppo.created_at ASC;
"""
    cmd = [
        "docker", "exec", "eip-mmdpp-postgres-1",
        "psql", "-U", "eip", "-d", "eip", "-tA", "-F", "|",
        "-c", sql,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        return []
    out = []
    for line in res.stdout.strip().splitlines():
        if not line.strip():
            continue
        cols = line.split("|")
        if len(cols) < 21:
            continue
        try:
            out.append({
                "run_id": cols[0],
                "pass_name": cols[1],
                "execution_status": cols[2],
                "yield_status": cols[3],
                "primary_entities": int(cols[4] or 0),
                "bridge_entities": int(cols[5] or 0),
                "relationships": int(cols[6] or 0),
                "relationships_rejected": int(cols[7] or 0),
                "pass_wall_ms": float(cols[8] or 0),
                "run_pipeline_ms": float(cols[9] or 0),
                "chunk_count": int(cols[10] or 0),
                "batch_count": int(cols[11] or 0),
                "doc_json_load_ms": float(cols[12] or 0),
                "request_bytes": int(cols[13] or 0),
                "response_bytes": int(cols[14] or 0),
                "service_queue_wait_ms": float(cols[15] or 0),
                "sanitize_ms": float(cols[16] or 0),
                "table_normalization_ms": float(cols[17] or 0),
                "table_overlay_ms": float(cols[18] or 0),
                "postprocess_ms": float(cols[19] or 0),
                "field_provenance_ms": float(cols[20] or 0),
            })
        except (ValueError, IndexError):
            continue
    return out


def _gate_marker(delta: float, threshold_pct: float = 5.0) -> str:
    """OK if within ±threshold_pct, REGRESSION if below baseline by more than
    threshold_pct, IMPROVEMENT if above by more than threshold_pct."""
    if delta < -threshold_pct:
        return "⚠ REGRESSION"
    if delta > threshold_pct:
        return "★ IMPROVEMENT"
    return "✓ OK"


def format_row(row: dict, sa2_run_id: str, dvina_run_id: str) -> str:
    pass_name = row["pass_name"]
    doc_label = "SA-2"
    baseline = SA2_BASELINE.get(pass_name)
    if row["run_id"] == dvina_run_id:
        doc_label = "Dvina"
        baseline = DVINA_BASELINE.get(pass_name)

    new_wall_s = row["pass_wall_ms"] / 1000.0
    new_entities = row["primary_entities"]
    new_rels = row["relationships"]

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"[{doc_label} / {pass_name}] {row['execution_status']}/{row['yield_status'] or '?'}")

    if baseline:
        # Wall comparison
        base_wall = baseline["wall_s"]
        wall_delta_pct = ((base_wall - new_wall_s) / base_wall) * 100.0 if base_wall else 0.0
        lines.append(
            f"  Wall:    NEW {new_wall_s:>7.1f}s ({new_wall_s/60.0:>5.1f}m)  "
            f"BASE {base_wall:>7.1f}s ({base_wall/60.0:>5.1f}m)  "
            f"Δ {('-' if new_wall_s < base_wall else '+')}{abs(new_wall_s-base_wall):>6.1f}s  "
            f"({wall_delta_pct:+5.1f}%)  {_gate_marker(wall_delta_pct)}"
        )
    else:
        lines.append(f"  Wall:    NEW {new_wall_s:>7.1f}s  (no baseline pinned for this pass)")

    # C0 internals breakdown
    lines.append(
        f"  Phases:  run_pipeline={row['run_pipeline_ms']/1000.0:.1f}s  "
        f"sanitize={row['sanitize_ms']:.0f}ms  "
        f"table_norm={row['table_normalization_ms']:.0f}ms  "
        f"overlay={row['table_overlay_ms']:.0f}ms  "
        f"postprocess={row['postprocess_ms']:.0f}ms  "
        f"field_prov={row['field_provenance_ms']:.0f}ms"
    )
    lines.append(
        f"  Network: req={row['request_bytes']/1024.0:>7.1f}KB  "
        f"resp={row['response_bytes']/1024.0:>7.1f}KB  "
        f"queue_wait={row['service_queue_wait_ms']:.0f}ms  "
        f"chunks={row['chunk_count']}  batches={row['batch_count']}"
    )

    if baseline and "entities" in baseline:
        base_ent = baseline["entities"]
        ent_delta = new_entities - base_ent
        lines.append(
            f"  Entities: NEW {new_entities:>3}  BASE {base_ent:>3}  Δ {ent_delta:+d}  "
            f"{_gate_marker(((new_entities - base_ent) / base_ent) * 100.0 if base_ent else 0.0)}"
        )
    if baseline and "rels" in baseline:
        base_rels = baseline["rels"]
        rels_delta = new_rels - base_rels
        lines.append(
            f"  Relationships: NEW {new_rels:>3}  BASE {base_rels:>3}  Δ {rels_delta:+d}  "
            f"{_gate_marker(((new_rels - base_rels) / base_rels) * 100.0 if base_rels else 0.0)}"
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    if len(sys.argv) != 3:
        print("usage: walltime_c0_progress.py <sa2_run_id> <dvina_run_id>", file=sys.stderr)
        sys.exit(2)
    sa2_run_id = sys.argv[1]
    dvina_run_id = sys.argv[2]

    seen: set[tuple[str, str]] = set()
    expected_total = len(PASS_ORDER) * 2
    poll_interval_s = 30

    print(f"[walltime-progress] watching SA-2={sa2_run_id} Dvina={dvina_run_id} "
          f"(target {expected_total} terminal passes)", flush=True)

    while True:
        rows = query_passes(sa2_run_id, dvina_run_id)
        for row in rows:
            if row["execution_status"] not in ("COMPLETE", "FAILED", "SKIPPED"):
                continue
            key = (row["run_id"], row["pass_name"])
            if key in seen:
                continue
            seen.add(key)
            print(format_row(row, sa2_run_id, dvina_run_id), flush=True)

        if len(seen) >= expected_total:
            print(f"[walltime-progress] all {expected_total} passes terminal — exiting", flush=True)
            return 0
        time.sleep(poll_interval_s)


if __name__ == "__main__":
    sys.exit(main())
