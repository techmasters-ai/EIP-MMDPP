#!/usr/bin/env bash
# Session-proof FINISH driver after the loop-fix deploy.
#   setsid bash scripts/run_a0_finish.sh <SA2_RUN> <SNR_RUN> <LF99_DOC> </dev/null >/dev/null 2>&1 &
#
# State at launch: SNR-75 run COMPLETE (good data); SA-2 run already PROCESSING
# under the fixed loop-fill code; lf99 prior run FAILED (mid-run worker restart)
# → re-run fresh. Then fit + 5-model bake-off on all three.
set -u
WT=/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
API=http://localhost:8005
BUNDLE=air_defense_v3
SA2_RUN="$1"
SNR_RUN="$2"
LF99_DOC="$3"
LOG="$WT/reports/collection/A0_finish_driver.log"
STATUS="$WT/reports/collection/A0_FINISH_STATUS.txt"
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

run_status() { docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c \
  "SELECT status FROM ingest.pipeline_runs WHERE id='$1';" 2>/dev/null | tr -d '[:space:]'; }
n_processing() { docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c \
  "SELECT count(*) FROM ingest.pipeline_runs WHERE status='PROCESSING';" 2>/dev/null | tr -d '[:space:]'; }
poll_terminal() {
  local run="$1" st="" i
  for i in $(seq 1 400); do
    sleep 180
    st=$(run_status "$run")
    case "$st" in COMPLETE | FAILED) echo "$st"; return 0 ;; esac
  done
  echo "${st:-TIMEOUT}"; return 1
}
analyze() {
  if python3 -m scripts.a0_captured_separation --run-id "$1" --out-dir "$WT/reports/collection" >> "$LOG" 2>&1; then
    log "analyze OK run=$1 → a0_separation_${1:0:8}.md"
  else log "analyze FAILED run=$1 (data still durable)"; fi
}
wait_idle() {
  local i np
  for i in $(seq 1 400); do
    np=$(n_processing)
    if [ "$np" = "0" ]; then log "pool idle (settling 120s)"; sleep 120; return 0; fi
    sleep 180
  done
  return 1
}
launch() {
  local doc="$1" resp run
  resp=$(curl -s --max-time 25 -X POST "$API/v1/documents/$doc/reingest" \
    -H 'Content-Type: application/json' -d "{\"mode\":\"graph_only\",\"ontology_bundle_key\":\"$BUNDLE\"}")
  run=$(printf '%s' "$resp" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(d.get('pipeline_run_id') or '')" 2>/dev/null || true)
  log "launch doc=$doc bundle=$BUNDLE → run=${run:-NONE} resp=${resp:0:200}"
  printf '%s' "$run"
}

log "=== A0 finish start pid=$$ sa2=$SA2_RUN snr=$SNR_RUN lf99_doc=$LF99_DOC ==="
ALL_RUNS=("$SNR_RUN")

# 1. SA-2 (already running under the loop-fix) — wait + analyze
st=$(run_status "$SA2_RUN")
case "$st" in
  COMPLETE | FAILED) log "SA-2 already $st" ;;
  *) log "waiting on SA-2 ($SA2_RUN) ..."; st=$(poll_terminal "$SA2_RUN"); log "SA-2 reached $st" ;;
esac
analyze "$SA2_RUN"; ALL_RUNS+=("$SA2_RUN")

# 2. lf99 — re-run fresh under the fix (prior run FAILED)
log "--- re-run lf99 $LF99_DOC ---"
if wait_idle; then
  run=$(launch "$LF99_DOC")
  if [ -n "$run" ]; then
    ALL_RUNS+=("$run")
    st=$(poll_terminal "$run"); log "lf99 run $run reached $st"
    analyze "$run"
  else log "lf99 launch FAILED"; fi
else log "pool never idle; skipping lf99 re-run"; fi

# 3. fit + 5-model bake-off on all three
JOINED=$(IFS=,; echo "${ALL_RUNS[*]}")
log "=== FIT + 5-MODEL BAKE-OFF on runs: $JOINED ==="
if python3 -m scripts.a0_captured_separation --fit-runs "$JOINED" --out-dir "$WT/reports/collection" >> "$LOG" 2>&1; then
  log "FIT+BAKEOFF OK"
else log "FIT+BAKEOFF FAILED"; fi

TAG=$(echo "$JOINED" | cut -c1-8)
{
  echo "A0 finish (post loop-fix) done $(date -u +%FT%TZ)"
  echo "bundle: $BUNDLE (all 3 consistent)"
  echo "runs: $JOINED  [SNR-75, SA-2(loop-fix), lf99(re-run)]"
  echo "per-run reports : reports/collection/a0_separation_<run8>.md"
  echo "5-model bakeoff : reports/collection/a0_bakeoff_${TAG}.md (+ .json)"
  echo "NOTE: VECTOR_ROUTER_MODE still 'shadow' — revert to narrow_only after."
} > "$STATUS"
log "=== finish done ==="
