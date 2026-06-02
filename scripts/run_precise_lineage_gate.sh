#!/usr/bin/env bash
# Detached driver for the Task-6 PRECISE-lineage gate. Survives session close
# (setsid-reparented to init). The SA-2 extraction itself runs in Celery and
# completes regardless of any session; this driver owns the VERIFY follow-up so
# the gate result is captured even if the controlling session drops.
set -u
WT=/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
RUN="${1:-cd2e081f-deee-4fc7-9d0e-6dff7dc6fc5f}"
PRE="${2:-0}"
DOC=ddaa9e36-2854-47c3-bc94-ff38d531dafd
OUT="$WT/reports/collection/precise_lineage_gate_result.txt"
LOG="$WT/reports/collection/precise_lineage_gate_driver.log"
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "=== precise-lineage gate driver started (pid $$) run=$RUN pre=$PRE ==="

ST=""
for i in $(seq 1 320); do            # ~8h cap at 90s cadence
  sleep 90
  ST=$(docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c \
        "SELECT status FROM ingest.pipeline_runs WHERE id='$RUN';" 2>/dev/null | tr -d '[:space:]')
  if [ -n "$ST" ] && [ "$ST" != "PROCESSING" ]; then
    log "run reached status=$ST after ~$((i*90/60))m"; break
  fi
done

{
  echo "=== Precise-lineage gate — run $RUN ==="
  echo "run status: ${ST:-UNKNOWN}   (driver pid $$, $(date -u +%FT%TZ))"
  echo "pre-extracted-from: $PRE"
  echo ""
  echo "--- per-pass element_uid populated? ---"
  docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -F'|' -c \
    "SELECT pass_name, COALESCE(jsonb_array_length(extract_pass_response_json->'provenance'),0) AS n_prov, (SELECT count(*) FROM jsonb_array_elements(extract_pass_response_json->'provenance') p WHERE COALESCE(p->>'element_uid','')<>'') AS euid_set FROM ingest.pipeline_pass_outputs WHERE pipeline_run_id='$RUN' ORDER BY pass_name;" 2>/dev/null
  echo ""
  echo "--- no-chunk-metadata warnings (run window) + truncations ---"
  echo "no_chunk_metadata: $(docker logs eip-mmdpp-docling-graph-1 --since 9h 2>&1 | grep -c 'no chunk metadata available')"
  echo "truncations: $(docker logs eip-mmdpp-docling-graph-1 --since 9h 2>&1 | grep -c TRUNCATION_AT_NUM_PREDICT)"
  echo ""
  if [ "$ST" = "COMPLETE" ]; then
    echo "--- HARDENED GATE (verify_lineage_e2e.py) ---"
    python3 "$WT/scripts/verify_lineage_e2e.py" --run "$RUN" --pre-extracted-from "$PRE" 2>&1
    echo ""
    echo "--- per-entity EXTRACTED_FROM (precise≈1 vs doc chunk count) ---"
    ADB(){ curl -s --max-time 15 -u root:eip_arcadedb_secret -X POST http://localhost:2480/api/v1/command/eip_knowledge_graph -H "Content-Type: application/json" -d "{\"language\":\"sql\",\"command\":\"$1\"}"; }
    echo "doc TextChunk count: $(ADB "SELECT count(*) AS c FROM TextChunk WHERE document_id='$DOC'" | python3 -c "import sys,json;print(json.load(sys.stdin)['result'][0]['c'])" 2>/dev/null)"
    ADB "SELECT system_name, out('EXTRACTED_FROM').size() AS ef FROM RADAR_SYSTEM WHERE out('EXTRACTED_FROM').size()>0 LIMIT 8" 2>/dev/null
  else
    echo "run did NOT reach COMPLETE (status=${ST:-UNKNOWN}) — gate NOT run; investigate."
  fi
} > "$OUT" 2>&1
log "wrote verdict to $OUT (status=$ST)"
log "=== driver done ==="
