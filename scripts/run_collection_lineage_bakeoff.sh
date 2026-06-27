#!/usr/bin/env bash
# Session-proof ~10-doc CLEAN-LINEAGE collection + fit + 5-model bake-off.
#
#   setsid bash scripts/run_collection_lineage_bakeoff.sh </dev/null >/dev/null 2>&1 &
#
# Re-runs each doc graph_only under air_defense_v3 (EXPLICIT override) on the
# DEPLOYED per-field __property_provenance lineage fix + shadow mode (captured
# score_components_all). Sequential / idle-pool. Per-doc: analyze with the
# FIXED-lineage target (--target lineage) + a value-grounding health line.
# Final: pooled + leave-one-document-out 5-model bake-off on the clean target,
# INCLUDING the already-finished SNR-75 run. Survives session close (setsid);
# all data is durable in postgres/arcadedb regardless.
set -u
WT=/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
API=http://localhost:8005
BUNDLE=air_defense_v3
A0_DATABASE_URL="postgresql+psycopg2://eip:eip_secret@localhost:5437/eip"
export A0_DATABASE_URL

# SNR-75 already done on the fixed code — seed it into the final bake-off.
PREDONE_RUNS=(e79a4866-750b-41e3-a4b4-521f3e31ff26)

# 9 docs to run, ordered fast/diverse first, SA-2 (value-dense, ~5h) LAST so a
# usable collection lands early and SA-2 enriches it.
DOCS=(
  6369f186-f738-43dc-90e7-6a718aee4586  # radar2_waveform1.pdf       (radar modulation)
  33574bd4-6ec9-4d3d-b4d8-77b7858f4338  # lf99 (combined doc)        (radar, prose)
  f365f1cc-f717-4d29-906c-c7fc00165d83  # V-75 SA-2 GUIDELINE.pdf    (missile)
  bd9c416d-cff2-475c-9b44-0d488928e8e0  # Fan_Song_Radar.jpeg        (image-heavy radar)
  1915cd62-dc8e-45b9-9a11-2a3b8ff898af  # SA-2 museum display        (missile prose)
  128e48f9-06f9-459a-b2f0-6d42bf62c42d  # SA-2_and_SR-71             (multi-system)
  24576b93-4719-46c4-b0a7-195d87233e04  # chinese_handwritten_notes  (non-English)
  9c8e09c7-e39f-4359-92c0-46330158c73c  # S-75 Dvina.pdf             (missile, sparse)
  ddaa9e36-2854-47c3-bc94-ff38d531dafd  # SA-2 Guideline RU          (value-dense, ~5h)
)

LOG="$WT/reports/collection/COLLECTION_lineage_driver.log"
STATUS="$WT/reports/collection/COLLECTION_LINEAGE_STATUS.txt"
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

run_status() {
  docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c \
    "SELECT status FROM ingest.pipeline_runs WHERE id='$1';" 2>/dev/null | tr -d '[:space:]'
}
n_processing() {
  docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c \
    "SELECT count(*) FROM ingest.pipeline_runs WHERE status='PROCESSING';" 2>/dev/null | tr -d '[:space:]'
}
poll_terminal() {
  local run="$1" st="" i
  for i in $(seq 1 480); do      # up to ~24h (180s * 480)
    sleep 180
    st=$(run_status "$run")
    case "$st" in COMPLETE | FAILED) echo "$st"; return 0 ;; esac
  done
  echo "${st:-TIMEOUT}"; return 1
}
analyze() {  # per-run separation on the FIXED-lineage target
  if python3 -m scripts.a0_captured_separation --run-id "$1" --target lineage \
       --out-dir "$WT/reports/collection" >> "$LOG" 2>&1; then
    log "analyze(lineage) OK run=$1 → a0_separation_${1:0:8}.md"
  else
    log "analyze FAILED run=$1 (data still durable in db)"
  fi
  # value-grounding health line (numeric+string fields physically in chunk)
  python3 -m scripts.verify_field_chunk_lineage --run "$1" >> "$LOG" 2>&1 \
    && log "verify(lineage) OK run=$1" || log "verify skipped run=$1"
}
wait_idle() {
  local i np
  for i in $(seq 1 480); do
    np=$(n_processing)
    if [ "$np" = "0" ]; then log "pool idle (settling 120s before launch)"; sleep 120; return 0; fi
    sleep 180
  done
  return 1
}
launch() {  # $1=doc → echoes run id
  local doc="$1" resp run
  resp=$(curl -s --max-time 25 -X POST "$API/v1/documents/$doc/reingest" \
    -H 'Content-Type: application/json' \
    -d "{\"mode\":\"graph_only\",\"ontology_bundle_key\":\"$BUNDLE\"}")
  run=$(printf '%s' "$resp" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(d.get('pipeline_run_id') or '')" 2>/dev/null || true)
  log "launch doc=$doc bundle=$BUNDLE → run=${run:-NONE} resp=${resp:0:200}"
  printf '%s' "$run"
}

log "=== COLLECTION lineage start pid=$$ bundle=$BUNDLE docs=${#DOCS[@]} predone=${PREDONE_RUNS[*]} ==="
ALL_RUNS=("${PREDONE_RUNS[@]}")
i=0
for doc in "${DOCS[@]}"; do
  i=$((i+1))
  log "--- [$i/${#DOCS[@]}] doc $doc ---"
  echo "running [$i/${#DOCS[@]}] doc=$doc ($(date -u +%FT%TZ))" > "$STATUS"
  if ! wait_idle; then log "pool never idle; aborting"; break; fi
  run=$(launch "$doc")
  if [ -z "$run" ]; then log "launch FAILED for $doc; skipping"; continue; fi
  ALL_RUNS+=("$run")
  st=$(poll_terminal "$run"); log "doc $doc run $run reached $st"
  analyze "$run"
done

JOINED=$(IFS=,; echo "${ALL_RUNS[*]}")
log "=== FIT + 5-MODEL BAKE-OFF (--target lineage) on runs: $JOINED ==="
if python3 -m scripts.a0_captured_separation --fit-runs "$JOINED" --target lineage \
     --out-dir "$WT/reports/collection" >> "$LOG" 2>&1; then
  log "FIT+BAKEOFF OK"
else
  log "FIT+BAKEOFF FAILED"
fi

TAG=$(echo "$JOINED" | cut -c1-8)
{
  echo "COLLECTION lineage bake-off finished $(date -u +%FT%TZ)"
  echo "bundle: $BUNDLE   target: lineage (per-field __property_provenance extracted-from)"
  echo "runs (${#ALL_RUNS[@]}): $JOINED"
  echo "per-run reports : reports/collection/a0_separation_<run8>.md"
  echo "fitted model    : reports/collection/a0_fitvalidate_${TAG}.json"
  echo "5-model bakeoff : reports/collection/a0_bakeoff_${TAG}.md (+ .json)"
  echo "NOTE: VECTOR_ROUTER_MODE still 'shadow' — revert to narrow_only after calibration."
} > "$STATUS"
log "=== collection done ==="
