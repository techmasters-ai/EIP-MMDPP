#!/usr/bin/env bash
# Detached, session-proof driver: poll the A0 shadow run to completion, then
# AUTO-RUN the captured-component separation analysis. Survives session close
# (launch via `setsid`). Mirrors scripts/run_precise_lineage_gate.sh.
#
#   setsid bash scripts/run_a0_separation_on_complete.sh <RUN_ID> </dev/null >/dev/null 2>&1 &
#
# Primary analysis runs host-side (proven: reads postgres :5437 + arcadedb :2480
# directly, no app import). Fallback runs inside the api container (worktree app/
# bind-mount + sklearn) with internal DSNs if the host path ever fails. Either
# way the raw component + USED data is durable in postgres/arcadedb, so nothing
# is lost even if both analysis paths fail.
set -u
WT=/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
RUN="${1:?run-id required}"
R8="${RUN:0:8}"
LOG="$WT/reports/collection/A0_shadow_driver.log"
SNAP="$WT/reports/collection/A0_shadow_${RUN}_result.txt"
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "driver(v2:auto-analysis) start pid=$$ run=$RUN"

# 1) poll to terminal (up to ~16h; cache-cold sleeps are fine for a background driver)
ST=""
for i in $(seq 1 320); do
  sleep 180
  ST=$(docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c \
        "SELECT status FROM ingest.pipeline_runs WHERE id='$RUN';" 2>/dev/null | tr -d '[:space:]')
  if [ -n "$ST" ] && [ "$ST" != "PROCESSING" ]; then
    log "run=$ST after ~$((i * 3))min"
    break
  fi
done

# 2) durable status snapshot (shadow-full-doc confirmation + per-pass capture)
{
  echo "=== A0 shadow run $RUN — status: ${ST:-UNKNOWN} ($(date -u +%FT%TZ)) ==="
  echo "--- per-pass: chunk_scope_applied (shadow⇒false/RUN_FULL) | n_score_components | n_provenance ---"
  docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -F'|' -c \
    "SELECT pass_name, diagnostics_json->'router'->>'chunk_scope_applied', \
            COALESCE(jsonb_array_length(diagnostics_json->'router'->'score_components_all'),0), \
            COALESCE(jsonb_array_length(extract_pass_response_json->'provenance'),0) \
     FROM ingest.pipeline_pass_outputs WHERE pipeline_run_id='$RUN' ORDER BY pass_name;" 2>/dev/null
} > "$SNAP" 2>&1
log "wrote snapshot $SNAP"

# 3) auto-run the separation analysis on completion
ANALYSIS_OK=0
if [ "${ST:-}" = "PROCESSING" ] || [ -z "${ST:-}" ]; then
  log "run did not reach terminal within poll window (ST=${ST:-empty}); skipping analysis, data is durable"
else
  log "running separation analysis (host python3) ..."
  if python3 -m scripts.a0_captured_separation --run-id "$RUN" --out-dir "$WT/reports/collection" >> "$LOG" 2>&1; then
    ANALYSIS_OK=1
    log "analysis OK (host) → reports/collection/a0_separation_${R8}.{md,json,csv}"
  else
    log "host analysis FAILED; trying api container fallback"
    docker exec eip-mmdpp-api-1 mkdir -p /app/scripts /app/reports/collection 2>>"$LOG"
    docker cp "$WT/scripts/phase1_score_used_separation.py" eip-mmdpp-api-1:/app/scripts/ 2>>"$LOG"
    docker cp "$WT/scripts/a0_captured_separation.py" eip-mmdpp-api-1:/app/scripts/ 2>>"$LOG"
    if docker exec -w /app \
        -e A0_DATABASE_URL='postgresql+psycopg2://eip:eip_secret@postgres:5432/eip' \
        -e A0_ARCADEDB_URL='http://arcadedb:2480' \
        eip-mmdpp-api-1 python -m scripts.a0_captured_separation \
        --run-id "$RUN" --out-dir /app/reports/collection >> "$LOG" 2>&1; then
      ANALYSIS_OK=1
      log "analysis OK (api container); copying results to host"
      for ext in csv json md; do
        docker cp "eip-mmdpp-api-1:/app/reports/collection/a0_components_${R8}.${ext}" "$WT/reports/collection/" 2>/dev/null
        docker cp "eip-mmdpp-api-1:/app/reports/collection/a0_separation_${R8}.${ext}" "$WT/reports/collection/" 2>/dev/null
      done
    else
      log "api container analysis ALSO failed — raw data intact in postgres/arcadedb for manual analysis"
    fi
  fi
fi
log "driver(v2) done analysis_ok=$ANALYSIS_OK"
