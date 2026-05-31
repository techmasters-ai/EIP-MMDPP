#!/usr/bin/env python3
"""Diagnose the entity-commit + lineage break for a graph_only run.

Read-only. Run via host python; uses ``docker exec`` for in-container merge
replay so the REAL code paths execute inside the worker-graph container.

Two-config diagnostic (spec 2026-05-30-element-uid-lineage-fix-design.md,
Component 1 / plan Task 0):

  (A) Where do merged entities fail to reach ArcadeDB?
  (B) Why are element_uid/page empty, and on which delta route?
  (C) Is production config (air_defense_v3, non-narrowed) affected?

Usage:
  # Offline merge replay for an existing run (no fresh extraction needed):
  python3 scripts/diagnose_lineage_commit.py \
      --run 9d48fc1e-62fd-4b03-a98c-98a35bda3b8e \
      --doc ddaa9e36-2854-47c3-bc94-ff38d531dafd

  # Pre/post snapshot for run-attributable commit delta around a FRESH run
  # (entity vertices carry NO run_id, so absolute counts are not run-scoped):
  python3 scripts/diagnose_lineage_commit.py --snapshot pre   # -> record P
  #   ... trigger a fresh graph_only run on an idle pool, wait terminal ...
  python3 scripts/diagnose_lineage_commit.py --snapshot post  # -> record Q
  #   committed delta for the run = Q - P
"""
import argparse
import json
import subprocess

WORKER_GRAPH_CONTAINER = "eip-mmdpp-worker-graph-1"
ARCADEDB_URL = "http://localhost:2480/api/v1/command/eip_knowledge_graph"
ARCADEDB_AUTH = "root:eip_arcadedb_secret"

# Entity vertex types to sum for the GLOBAL committed-entity count. Entity
# vertices carry NO run_id/document_id property (verified 2026-05-30:
# RADAR_SYSTEM/MISSILE_SYSTEM rows have document_id=null, pipeline_run_id=null),
# so this sum is NOT run-scoped — see committed_entity_count() docstring.
ENTITY_TYPES = [
    "RADAR_SYSTEM",
    "MISSILE_SYSTEM",
    "FIRE_CONTROL_SYSTEM",
    "WEAPON_SYSTEM",
    "EQUIPMENT_SYSTEM",
    "LAUNCHER_SYSTEM",
    "ELECTRONIC_WARFARE_SYSTEM",
    "AIR_DEFENSE_ARTILLERY_SYSTEM",
    "INTEGRATED_AIR_DEFENSE_SYSTEM",
]


def adb(sql):
    """Run a single SQL command against ArcadeDB; return the result list."""
    out = subprocess.run(
        [
            "curl", "-s", "--max-time", "10", "-u", ARCADEDB_AUTH,
            "-X", "POST", ARCADEDB_URL,
            "-H", "Content-Type: application/json",
            "-d", json.dumps({"language": "sql", "command": sql}),
        ],
        capture_output=True, text=True, timeout=20,
    ).stdout
    try:
        return json.loads(out).get("result", [])
    except Exception:
        return []


def committed_entity_count():
    """GLOBAL entity-vertex count across all entity types.

    Entity vertices carry NO run_id/document_id property (verified 2026-05-30:
    RADAR_SYSTEM/MISSILE_SYSTEM schemas have only domain + housekeeping fields;
    sampled rows show document_id=null, pipeline_run_id=null), so this count is
    NOT run-scoped.

    Review finding High-1: a global count can false-OK a broken run when other
    entities already exist. Therefore the diagnostic uses a PRE/POST SNAPSHOT
    delta around a fresh extraction (``--snapshot pre|post``), NOT an absolute
    count, to attribute commits to the run under test.
    """
    total = 0
    per_type = {}
    for t in ENTITY_TYPES:
        r = adb(f"SELECT count(*) AS n FROM `{t}`")
        n = (r[0]["n"] if r else 0)
        per_type[t] = n
        total += n
    return total, per_type


def merge_replay(run_id, bundle, doc_id):
    """Replay load -> rehydrate -> merge_and_resolve IN the worker-graph
    container so the REAL code paths run. Prints per-pass cached-entity counts,
    merged entity count, and merged edge count.

    NOTE: MergedExtraction exposes ``.edges`` (NOT ``.relationships``).
    """
    py = (
        "from app.workers.pipeline import _rehydrate_pass_result,_get_db,load_completed_pass_outputs;"
        "from app.services.ontology_bundles import load_bundle_manifest;"
        "from app.services.ontology_templates import load_ontology;"
        "from app.services.extraction_merge import merge_and_resolve;"
        f"db=_get_db();m=load_bundle_manifest('{bundle}');o=load_ontology(bundle_key='{bundle}');"
        f"outs=load_completed_pass_outputs(db,'{run_id}');"
        f"reh={{n:_rehydrate_pass_result(r,m,o,'{doc_id}',db=db,run_id='{run_id}') for n,r in outs.items()}};"
        "per={n:len(p._cached_entities()) for n,p in reh.items()};"
        f"mg=merge_and_resolve(pass_results=reh,manifest=m,ontology=o,document_id='{doc_id}',pipeline_run_id='{run_id}');"
        "import json;print('DIAG',json.dumps({'per_pass':per,'merged_entities':len(mg.entities),"
        "'merged_edges':len(getattr(mg,'edges',[]) or [])}));db.close()"
    )
    proc = subprocess.run(
        ["docker", "exec", WORKER_GRAPH_CONTAINER, "python", "-c", py],
        capture_output=True, text=True, timeout=300,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("DIAG "):
            return json.loads(line[5:])
    # Surface the in-container error so the caller doesn't guess.
    return {
        "_error": "no DIAG line",
        "_stdout_tail": proc.stdout[-2000:],
        "_stderr_tail": proc.stderr[-2000:],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        help="existing run to replay merge for (offline). Not needed in --snapshot mode.",
    )
    ap.add_argument("--bundle", default="air_defense_v3_merged_v1")
    ap.add_argument(
        "--doc",
        help="document_id for the run. Not needed in --snapshot mode.",
    )
    ap.add_argument(
        "--snapshot", choices=["pre", "post"], default=None,
        help="for a FRESH run: capture global entity count before/after to get a "
             "run-attributable delta (review finding High-1)",
    )
    a = ap.parse_args()

    if a.snapshot:
        # Pre/post snapshot mode: print the GLOBAL count; caller diffs pre vs
        # post around a fresh extraction to attribute commits to THIS run
        # (entity vertices carry no run_id, so absolute counts are not
        # run-scoped).
        total, per_type = committed_entity_count()
        print(f"SNAPSHOT_{a.snapshot.upper()} global_entity_count={total}")
        print(f"  per-type: {per_type}")
        return

    if not a.run or not a.doc:
        ap.error("--run and --doc are required unless --snapshot is given")

    rep = merge_replay(a.run, a.bundle, a.doc)
    if rep.get("_error"):
        print(f"MERGE_REPLAY_ERROR: {rep['_error']}")
        print("--- in-container stdout tail ---")
        print(rep.get("_stdout_tail", ""))
        print("--- in-container stderr tail ---")
        print(rep.get("_stderr_tail", ""))
        return

    total, per_type = committed_entity_count()
    print(f"run={a.run} bundle={a.bundle} doc={a.doc}")
    print(f"  per-pass rehydrated entities: {rep.get('per_pass')}")
    print(
        f"  merge_and_resolve entities:   {rep.get('merged_entities')}  "
        f"edges: {rep.get('merged_edges')}"
    )
    print(
        f"  ArcadeDB GLOBAL entities:     {total}  per-type={per_type}\n"
        f"      (NOT run-scoped — entity vertices carry no run_id; use "
        f"--snapshot pre/post around a fresh run for a run-attributable delta)"
    )
    print(
        f"  >>> merge replay yields {rep.get('merged_entities')} entities; "
        f"if a fresh run's post-pre snapshot delta is < that, the COMMIT BREAK "
        f"is downstream of merge."
    )


if __name__ == "__main__":
    main()
