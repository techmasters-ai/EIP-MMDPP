#!/usr/bin/env bash
# Detached driver for the Task-5 lineage verification gate.
# Survives session close (launched via setsid/nohup, reparented to init).
# Waits for the SA-2 gate run to reach terminal, captures snapshot-post,
# runs verify_lineage_e2e.py, and writes the verdict to a durable file.
#
# The extraction run itself already runs in the docker containers (Celery) and
# completes regardless of any session; this driver owns the VERIFY follow-up.
set -u
WT=/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
RUN="${1:-8e3c2aa2-9d11-4d7e-8336-48c75939d78f}"
PRE="${2:-0}"                       # SNAPSHOT_PRE captured before the run
OUT="$WT/reports/collection/lineage_gate_result.txt"
LOG="$WT/reports/collection/lineage_gate_driver.log"
mkdir -p "$(dirname "$OUT")"

log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "=== lineage gate driver started (pid $$) run=$RUN pre=$PRE ==="

# Poll until the run leaves PROCESSING (cap ~8h at 2-min cadence).
ST=""
for i in $(seq 1 240); do
  sleep 120
  ST=$(docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c \
        "SELECT status FROM ingest.pipeline_runs WHERE id='$RUN';" 2>/dev/null)
  if [ -n "$ST" ] && [ "$ST" != "PROCESSING" ]; then
    log "run reached status=$ST after ~$((i*2))m"
    break
  fi
done

{
  echo "=== Lineage gate result — run $RUN ==="
  echo "run status: ${ST:-UNKNOWN}"
  echo "snapshot_pre: $PRE"
  echo ""
  echo "--- snapshot_post ---"
  python3 "$WT/scripts/diagnose_lineage_commit.py" --snapshot post 2>&1 | grep -i SNAPSHOT_POST
  echo ""
  echo "--- per-pass committed (postgres) ---"
  docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -c \
    "SELECT DISTINCT ON (pass_name) pass_name, execution_status, primary_entities_extracted AS ents, relationships_extracted AS rels FROM ingest.pipeline_pass_outputs WHERE pipeline_run_id='$RUN' ORDER BY pass_name, attempt DESC;" 2>/dev/null
  echo ""
  echo "--- verifier (verify_lineage_e2e.py) ---"
  python3 "$WT/scripts/verify_lineage_e2e.py" --run "$RUN" --pre "$PRE" 2>&1
  echo ""
  echo "--- truncation + provenance-drop counts this run window ---"
  echo "TRUNCATION: $(docker logs eip-mmdpp-docling-graph-1 --since 8h 2>&1 | grep -c TRUNCATION_AT_NUM_PREDICT)"
  echo "provenance-drop warnings: $(docker logs eip-mmdpp-docling-graph-1 --since 8h 2>&1 | grep -c 'dropping provenance row')"
  echo "LINEAGE_GATE rejections: $(docker logs eip-mmdpp-worker-graph-1 --since 8h 2>&1 | grep -c 'LINEAGE_GATE: rejected')"
  echo "UpsertNotDurableError: $(docker logs eip-mmdpp-worker-graph-1 --since 8h 2>&1 | grep -c 'UpsertNotDurableError')"
} > "$OUT" 2>&1

log "wrote verdict to $OUT (run status=$ST)"
log "=== lineage gate driver done ==="
