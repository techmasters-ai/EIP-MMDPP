#!/usr/bin/env bash
# Unattended, session-proof CLEAN re-run + fit + 5-model bake-off.
#
#   setsid bash scripts/run_a0_clean_rerun.sh <DOC_1> <DOC_2> <DOC_3> </dev/null >/dev/null 2>&1 &
#
# Re-runs each doc graph_only under the SAME bundle (air_defense_v3, EXPLICIT
# override so no doc inherits a calibration bundle) with the fixed code + shadow
# mode (full-doc + captured score_components_all). Sequential / idle-pool only.
# Per-run analysis, then fits the model + runs the 5-classifier bake-off
# (LogisticRegression / RandomForest / GradientBoosting / HistGradientBoosting /
# MLP), each scored by pooled CV AND leave-one-document-out. Survives session
# close (setsid). All data is durable in postgres/arcadedb regardless.
set -u
WT=/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
API=http://localhost:8005
BUNDLE=air_defense_v3
DOCS=("$@")
LOG="$WT/reports/collection/A0_cleanrerun_driver.log"
STATUS="$WT/reports/collection/A0_CLEANRERUN_STATUS.txt"
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
  for i in $(seq 1 360); do      # up to ~18h
    sleep 180
    st=$(run_status "$run")
    case "$st" in COMPLETE | FAILED) echo "$st"; return 0 ;; esac
  done
  echo "${st:-TIMEOUT}"; return 1
}
analyze() {
  if python3 -m scripts.a0_captured_separation --run-id "$1" --out-dir "$WT/reports/collection" >> "$LOG" 2>&1; then
    log "analyze OK run=$1 → a0_separation_${1:0:8}.md"
  else
    log "analyze FAILED run=$1 (data still durable in db)"
  fi
}
wait_idle() {
  local i np
  for i in $(seq 1 360); do
    np=$(n_processing)
    if [ "$np" = "0" ]; then log "pool idle (settling 120s before launch)"; sleep 120; return 0; fi
    sleep 180
  done
  return 1
}
launch() {  # $1=doc → echoes run id; passes EXPLICIT bundle override
  local doc="$1" resp run
  resp=$(curl -s --max-time 25 -X POST "$API/v1/documents/$doc/reingest" \
    -H 'Content-Type: application/json' \
    -d "{\"mode\":\"graph_only\",\"ontology_bundle_key\":\"$BUNDLE\"}")
  run=$(printf '%s' "$resp" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(d.get('pipeline_run_id') or '')" 2>/dev/null || true)
  log "launch doc=$doc bundle=$BUNDLE → run=${run:-NONE} resp=${resp:0:200}"
  printf '%s' "$run"
}

log "=== A0 clean-rerun start pid=$$ bundle=$BUNDLE docs=${DOCS[*]} ==="
ALL_RUNS=()
for doc in "${DOCS[@]}"; do
  log "--- doc $doc ---"
  if ! wait_idle; then log "pool never idle; aborting"; break; fi
  run=$(launch "$doc")
  if [ -z "$run" ]; then log "launch FAILED for $doc; skipping"; continue; fi
  ALL_RUNS+=("$run")
  st=$(poll_terminal "$run"); log "doc $doc run $run reached $st"
  analyze "$run"
done

JOINED=$(IFS=,; echo "${ALL_RUNS[*]}")
log "=== FIT + 5-MODEL BAKE-OFF on runs: $JOINED ==="
if python3 -m scripts.a0_captured_separation --fit-runs "$JOINED" --out-dir "$WT/reports/collection" >> "$LOG" 2>&1; then
  log "FIT+BAKEOFF OK"
else
  log "FIT+BAKEOFF FAILED"
fi

TAG=$(echo "$JOINED" | cut -c1-8)
{
  echo "A0 clean re-run finished $(date -u +%FT%TZ)"
  echo "bundle: $BUNDLE (all 3 docs — consistent)"
  echo "runs: $JOINED"
  echo "per-run reports : reports/collection/a0_separation_<run8>.md"
  echo "fitted model    : reports/collection/a0_fitvalidate_${TAG}.json"
  echo "5-model bakeoff : reports/collection/a0_bakeoff_${TAG}.md (+ .json)"
  echo "NOTE: VECTOR_ROUTER_MODE still 'shadow' — revert to narrow_only after calibration."
} > "$STATUS"
log "=== clean-rerun done ==="
